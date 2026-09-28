# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import csv
import sys
import logging
import numpy as np

logger = logging.getLogger(__name__)

try:
    from scipy.stats import pearsonr, spearmanr
    from sklearn.metrics import matthews_corrcoef, f1_score, precision_score, recall_score
    _has_sklearn = True
except (AttributeError, ImportError) as e:
    logger.warning("To use data.metrics please install scikit-learn. See https://scikit-learn.org/stable/index.html")
    _has_sklearn = False
    
def simple_accuracy(preds, labels):
    return (preds == labels).mean()

def acc_and_f1(preds, pred_prob, labels, predict = -1):
    
    if predict == -1:
        acc = simple_accuracy(preds, labels)
        f1 = f1_score(y_true=labels, y_pred=preds)
        p = precision_score(y_true=labels, y_pred=preds)
        r = recall_score(y_true=labels, y_pred=preds)
        
        
        fnl_threshold = 10.0
        final_r = 10.0

        for threshold in range(100,-1,-1):
            t = threshold/100
            tp = 0
            for i in range(len(labels)):
                if (pred_prob[i][1] >= t) and (labels[i] ==1):
                    tp = tp+1                
            rt = tp/np.sum(labels == 1)
            if rt >= 0.95:
                break        
        fnl_threshold = t
        final_r = rt
        print (fnl_threshold, final_r)

        predt = np.copy(labels)
        for j in range(len(labels)):
            if pred_prob[j][1] >= fnl_threshold:
                predt[j] = 1
            else:
                predt[j] = 0
        cmb = predt + labels
        acc_95 = (np.sum(cmb == 2)+np.sum(cmb == 0))/len(labels)
        p_95 = np.sum(cmb == 2)/np.sum(predt == 1)
        r_95 = final_r
        f1_95 = (2*p_95*r_95)/(p_95+r_95)
        
        return {"acc": acc, "p": p, "r": r,
            "f1": f1, "threshold": fnl_threshold, "@acc95r": acc_95,
            "@p95r": p_95, "@r95": r_95, "@f195r": f1_95}

    else:
        predt_prd = np.copy(labels)
        
        for k in range(len(labels)):
            if pred_prob[k][1] >= predict:
                predt_prd[k] = 1
            else:
                predt_prd[k] = 0
                
        cmb = predt_prd + labels
        acc_95prd = (np.sum(cmb == 2)+np.sum(cmb == 0))/len(labels)
        p_95prd = np.sum(cmb == 2)/np.sum(predt_prd == 1)
        r_95prd = np.sum(cmb == 2)/np.sum(labels == 1)
        f1_95prd = (2*p_95prd*r_95prd)/(p_95prd+r_95prd)

        return {"predict_threshold": predict, "acc_95prd": acc_95prd,
            "p_95prd": p_95prd, "r_95prd": r_95prd, "f1_95prd": f1_95prd}

def compute_metrics(task_name, pred_prob, labels, predict = -1):
    assert len(pred_prob) == len(labels)
    #pred_prob = pred_prob
    preds = np.argmax(pred_prob, axis=1)
    
    if task_name == "pediatric_drug":
        return acc_and_f1(preds, pred_prob, labels, predict)
    else:
        raise KeyError(task_name)
