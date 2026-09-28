from __future__ import absolute_import, division, print_function

import sys
sys.path.append('/users/PCON0020/liuxf2021/a-bert-models')
sys.path.append('/users/PCON0020/liuxf2021/a-tools')

import argparse
import glob
import logging
import os
import numpy as np
import json
import torch
from torch.utils.data import DataLoader, SequentialSampler,TensorDataset#RandomSampler,
from torch.utils.data.distributed import DistributedSampler
from torch.nn import DataParallel
import traceback
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import f1_score, precision_score, recall_score
from processors import collate_fn
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer#, get_linear_schedule_with_warmup
from transformers import (BertConfig, BertModel, BertForSequenceClassification, BertTokenizer)

from metrics.glue_compute_metrics import compute_metrics
from processors import glue_output_modes as output_modes
from processors import glue_processors as processors
from processors import glue_convert_examples_to_features as convert_examples_to_features

from tools.common import seed_everything
from tools.common import init_logger,logger
from tools.progressbar import ProgressBar

def load_and_cache_examples(args, task, tokenizer, pop, data_type):
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()

    processor = processors[task]()
    output_mode = output_modes[task]

    logger.info("Creating features from dataset file at %s", args.data_dir)
    label_list = processor.get_labels()

    org = None
    if data_type == 'predict':
        examples, org = processor.get_predict_examples(args.data_dir)
        
    features, cv_labels = convert_examples_to_features(examples, tokenizer, label_list=label_list,
                                                       max_length=args.max_seq_length, 
                                                       output_mode=output_mode,
                                                       pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0])

    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    all_lens = torch.tensor([f.input_len for f in features],dtype=torch.long)
    all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
        
    all_dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_lens, all_labels)

    return (all_dataset, org)


def final_extract(args, predict_dataset, predict_org, model, tokenizer, predict, prefix=""):
    eval_task_names = (args.task_name,)
    eval_outputs_dirs = (args.output_dir,)

    results = {}
    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)

        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        eval_sampler = SequentialSampler(predict_dataset) if args.local_rank == -1 else DistributedSampler(predict_dataset)
        eval_dataloader = DataLoader(predict_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size,collate_fn= collate_fn)

        
        logger.info("***** Running prediction {} *****".format(prefix))
        logger.info("  Num examples = %d", len(predict_dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)

        preds = None
        pbar = ProgressBar(n_total=len(eval_dataloader),desc = "Predicting")
        for step, batch in enumerate(eval_dataloader):
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)
            with torch.no_grad():
                inputs = {'input_ids':      batch[0],
                          'attention_mask': batch[1],
                          'labels':         batch[3]}
                
                inputs['token_type_ids'] = batch[2]
                outputs = model(**inputs)
                logits = outputs.logits
                softmax_out = torch.nn.functional.softmax(logits, dim=1)

            if preds is None:
                preds = softmax_out.detach().cpu().numpy()
            else:
                preds = np.append(preds, softmax_out.detach().cpu().numpy(), axis=0)
            pbar(step)   

        print(' ')
        if 'cuda' in str(args.device):
            torch.cuda.empty_cache()
        
        manual_list = []

        pbar2 = ProgressBar(n_total=len(preds),desc = "saving")
        for step, i in enumerate(range(len(preds))):
            label_pred = np.where(preds[i, 1]>= predict, 1, 0)
            manual_list.append(str(predict_org[i][0]) + '\t' + str(predict_org[i][1]) + '\t' + str(predict_org[i][2]) + '\t' + str(preds[i,0]) + '\t' + str(preds[i,1]) + '\t' + str(label_pred)) 
            
            pbar2(step)
       
    return manual_list


def main(args, predict_threshold, pop):
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type selected in the list")
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model")
    parser.add_argument("--task_name", default=None, type=str, required=True,
                        help="The name of the task to train selected in the list")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_predict", action='store_true',
                        help="Whether to run the model in inference mode on the test set.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.01, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for,E.g., 0.1 = 10% of training.")

    parser.add_argument('--logging_steps', type=int, default=10, 
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=1000,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=23,
                        help="random seed for initialization")

    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--server_ip', type=str, default='', help="For distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="For distant debugging.")
    args = parser.parse_args(args=args)

    # args.output_dir = args.output_dir +'{}'.format(args.model_type)
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    init_logger(log_file=args.output_dir+'/{}-{}-{}.log'.format(pop,args.model_type,args.task_name))
    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError("Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(args.output_dir))

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                    args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # Set seed
    seed_everything(args.seed)

    # Prepare GLUE task
    args.task_name = args.task_name.lower()
    if args.task_name not in processors:
        raise ValueError("Task not found: %s" % (args.task_name))
    processor = processors[args.task_name]()
    args.output_mode = output_modes[args.task_name]
    label_list = processor.get_labels()
    num_labels = len(label_list)

    # Load pretrained model and tokenizer
    args.model_type = args.model_type.lower()
    
    config = BertConfig.from_pretrained(args.config_name if args.config_name else args.model_name_or_path, num_labels=num_labels, finetuning_task=args.task_name)
    tokenizer = BertTokenizer.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path, do_lower_case=args.do_lower_case)
    model = BertForSequenceClassification.from_pretrained(args.model_name_or_path, from_tf=bool('.ckpt' in args.model_name_or_path), config=config)
    print('Model loaded.')

    model.to(args.device)
    logger.info("Training/evaluation parameters %s", args)

    try: 
        predict_dataset, predict_org = load_and_cache_examples(args, args.task_name, tokenizer, pop, data_type='predict')
        manual_list = final_extract(args, predict_dataset, predict_org, model, tokenizer, predict = predict_threshold)
    except Exception as e:
        logger.error(str(e))
        logger.error(traceback.format_exc())
    
    return manual_list

if __name__ == "__main__":
    
    resource_folder = f"/users/PCON0020/liuxf2021/2025_MPRINT_6cls/6_large_scale_screening/2_chunk/"
    current_folder = f"/users/PCON0020/liuxf2021/2025_MPRINT_6cls/6_large_scale_screening/3_prediction/"
    model_folder = f"/users/PCON0020/liuxf2021/2025_MPRINT_6cls/6_large_scale_screening/"
    
    studies_to_chosen_fold_ckpt_thresh = {
        "Biomarker": (0.95), #threshold
        "CT": (0.99),
        "FBNSTP": (0.01),
        "PE": (0.99),
        "PK": (0.99),
        "VC": (0.01)}
    
    populations = ["maternal", "pediatric"]
    
    for study in studies_to_chosen_fold_ckpt_thresh:
        if (study != "FBNSTP"):
            continue

        model = model_folder + f"{study}_checkpoint/"# getting the saved model folder
        predict_threshold = studies_to_chosen_fold_ckpt_thresh[study]#[0]#getting the threshold of the model
        #print(predict_threshold)
        
        for pop in populations:
            if (pop != "maternal"):
                continue
            
            output_dir = current_folder + f"output_{pop}/{study}/"
            if not os.path.exists(output_dir):# creating directory if not already created
                os.makedirs(output_dir)
            #print(output_dir)
            
            for filename in os.listdir(resource_folder + f"{pop}_chunks/"):
                #print(filename)
                if (filename.startswith(f"{pop}_w_abstract")) & (filename.endswith('.txt')):
                    data_dir = resource_folder + f"{pop}_chunks/{filename}"
                    print(data_dir)

                    # creating and clearing predicted file
                    file_name = output_dir + f'predicted_{filename}'
                    open(file_name, 'w').close()

                    # initializing all the arguments to be used at each iteration
                    args = ['--model_type', 'bert', '--model_name_or_path', model, '--task_name', 'pediatric_drug', 
                            '--do_train', '--do_eval', '--do_lower_case', '--data_dir', data_dir, '--max_seq_length', '512', 
                            '--per_gpu_train_batch_size', '16', '--per_gpu_eval_batch_size', '16', '--learning_rate', '2e-5', 
                            '--num_train_epochs', '15', '--logging_steps', '30', '--save_steps', '30', 
                            '--output_dir', output_dir, '--overwrite_output_dir']
                    
                    # obtaining the predicted results
                    manual_list = main(args, predict_threshold, pop)

                    with open(file_name, "a") as around_half_log:
                        for i in range(len(manual_list)):
                            around_half_log.write(manual_list[i])    
                            around_half_log.write('\n')
                    around_half_log.close()

                    # clearing cache
                    torch.cuda.empty_cache()