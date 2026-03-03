python scripts/bow_polarity_label.py --output_path data/MFRC_multi.csv --use_lemma --use_frequency --tie_break neutral --drop_not_confident

python scripts/prep_multi_data.py --input data/MFRC_multi.csv

python scripts/train_multi_roberta.py --epochs 5 --batch_size 128 --lr 5e-5 --max_len 150