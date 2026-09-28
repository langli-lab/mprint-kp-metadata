Six MPRINT BioBERT sequence-classification checkpoints exported from OSC:
Biomarker, CT, FBNSTP, PE, PK, and VC.

Download one ZIP per model needed. Each ZIP contains the named checkpoint folder
with its `model.safetensors` weights, configuration, and tokenizer files. Extract
it into `6_models_MPRINT/` in the repository, or another local model directory.
Keep the configuration, tokenizer, and weights together.

The GitHub-generated source-code archives do not contain model weights. Use the
six separately attached `biobert-*.zip` files. `SHA256SUMS.txt` contains checksums
for those files. The older `pytorch_model.old` files are omitted from the release;
the original downloads remain unchanged.

All six archives have been checked for ZIP integrity. This packaging check does
not validate inference behavior or model accuracy. The saved configurations omit
descriptive class labels; training and evaluation documentation were not included
in the supplied export. The original example code requires OSC-specific helpers
and paths.
