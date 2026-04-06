# CSCI199.X-workbench
Repository for all relevant modules to the pipeline.

## How to Use
Run these commands in the base directory.

The following installs the needed python modules, as well as the specific model needed for the SpaCy.
```[bash]
pip install -r requirements.txt
pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_md-3.7.1/en_core_web_md-3.7.1-py3-none-any.whl
```
### Independent Classifier (obsolete)
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

### Hierarchical Classifier Process
