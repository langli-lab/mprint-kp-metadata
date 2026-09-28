# MPRINT BioBERT checkpoints

This directory contains configuration and tokenizer files for six supplied BioBERT
checkpoints. The large weights are distributed separately as GitHub Release assets.
The checkpoint names are preserved from the OSC export.

| Checkpoint directory | Release archive |
|---|---|
| `Biomarker_checkpoint/` | `biobert-Biomarker.zip` |
| `CT_checkpoint/` | `biobert-CT.zip` |
| `FBNSTP_checkpoint/` | `biobert-FBNSTP.zip` |
| `PE_checkpoint/` | `biobert-PE.zip` |
| `PK_checkpoint/` | `biobert-PK.zip` |
| `VC_checkpoint/` | `biobert-VC.zip` |

## Download and extract

The proposed release tag is `biobert-v1`. Until that release is published, the
archives are available only in the maintainer's local `release-assets/biobert-v1/`
directory. After publication, download the desired archive from the repository's
[Releases page](https://github.com/langli-lab/mprint-kp-metadata/releases).

Each archive contains its named checkpoint directory, including
`model.safetensors`, `config.json`, and the saved tokenizer files. Extract into
`6_models_MPRINT/` to put the weights alongside the configuration files. Keep the
files from a checkpoint together. Downloading the repository's source-code ZIP
alone does not include weights.

`SHA256SUMS.txt` records the archive checksums. On macOS, from the directory holding
all six downloaded archives and that checksum file, verify them with:

```bash
shasum -a 256 -c SHA256SUMS.txt
```

## Checkpoint details

The saved configurations declare `BertForSequenceClassification`; each supplied
weight file has a two-class classification head. The models have 12 encoder
layers, a hidden size of 768, a vocabulary of 28,996 tokens, and capacity for 512
positions. The saved tokenizers lowercase input. Use the tokenizer supplied with
each checkpoint and explicitly truncate input to at most 512 tokens.

The configuration files do not provide descriptive class labels. The PK
configuration records an OSC base-model path ending in
`biobert-base-cased-torch/`, but an exact upstream model version, training dataset,
and evaluation results were not included in the export. Do not infer class
meanings from the generic saved configuration fields.

The original example in `code/running_code_using_maternal_FBNSTP_as_example.py`
is preserved as a reference. It depends on OSC-specific paths and helper modules
that are not included, so it is not a standalone inference program.

## Prepare and publish a release

From the repository root, after downloading all six checkpoint directories:

```bash
python3 scripts/package_biobert_models.py
```

The script creates six ZIPs and `SHA256SUMS.txt` under
`release-assets/biobert-v1/`. It packages an explicit set of model and tokenizer
files, checks ZIP integrity, and computes SHA-256 checksums. It leaves the
downloaded files untouched and refuses to overwrite existing output. The older
`pytorch_model.old` files and notebook/cache files are not included. ZIPs use
uncompressed storage for fast packaging, approximately 434 MB per model.

Weight files, old checkpoints, and generated archives are ignored by Git. Commit
the documentation, packaging script, configurations, tokenizers, and original
source code normally. Archive verification does not establish inference accuracy;
inference has not been tested as part of this packaging step.

