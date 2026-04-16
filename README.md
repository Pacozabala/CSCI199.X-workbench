# CSCI199.X-workbench
Repository for all relevant modules to the pipeline.

## How to Use
Run these commands in the base directory.

The following installs the needed python modules, as well as the specific model needed for the SpaCy.
```[bash]
pip install -r requirements.txt
pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_md-3.7.1/en_core_web_md-3.7.1-py3-none-any.whl
```

`bow_polarity_label.py` assigns polarity labels to the MFRC, using the rules declared in the args above. I find that these settings balance the dataset most effectively.
```[bash]
python scripts/bow_polarity_label.py \
    --output_path data/MFRC_polarity.csv \
    --use_lemma \
    --use_frequency \
    --tie_break drop \
    --drop_zero_signal
```

`prep_binary_data.py` creates directories for each of the binary classifiers (10 total) in the `binary_data/` each with three CSVs (train, val, test) for the training of the independent binary classifier.
```[bash]
python scripts/prep_binary_data.py --input "data/MFRC_polarity.csv"
```

`train_roberta.py` trains one classifier. To run this, choose a foundation and pole (virtue/vice). This will create:
- `output/` folder, if one does not exist yet
- model and tokenizer checkpoints
```[bash]
python scripts/train_roberta.py --foundation authority --pole vice
```


## Multi-method methodology (bow, llama_zeroshot, llama_fewshot)

Use the same directory naming convention for each method to keep experiments comparable.

### 1) Prepare method-specific source files

- `bow` method input: `data/MFRC_polarity.csv` (created by `bow_polarity_label.py`).
- LLaMA method inputs must be converted first:

```bash
python scripts/convert_llama_to_polarity.py \
  --input_path data/llama_zeroshot_data.csv \
  --output_path data/MFRC_multi_llama_zeroshot.csv

python scripts/convert_llama_to_polarity.py \
  --input_path data/llama_fewshot_data.csv \
  --output_path data/MFRC_multi_llama_fewshot.csv
```

### 2) Run all methods with explicit output folders

```bash
for method in bow llama_zeroshot llama_fewshot; do
  if [ "$method" = "bow" ]; then
    input_file="data/MFRC_polarity.csv"
  elif [ "$method" = "llama_zeroshot" ]; then
    input_file="data/MFRC_multi_llama_zeroshot.csv"
  else
    input_file="data/MFRC_multi_llama_fewshot.csv"
  fi

  python scripts/prep_multi_data.py \
    --input "$input_file" \
    --output "data/hierarchical_dataset_${method}"

  python scripts/train_multi_roberta.py \
    --data_dir "data/hierarchical_dataset_${method}" \
    --model_dir "models/${method}" \
    --tokenizer_dir "tokenizer/${method}" \
    --results_dir "results/${method}"
done
```

This produces separate data folders for each method:
- Dataset: `data/hierarchical_dataset_<method>/...`
- Model: `models/<method>/...`
- Tokenizer: `tokenizer/<method>/...`
- Metrics/logs: `results/<method>/...`