#! /bin/bash
source activate sp3
SEED=100
NBITER=100
LOAD=False
DATA=test-clean-100
NAME=pgd
SNR=35
#python evaluate.py pgd.yaml --root=$RSROOT --data_csv_name=$DATA --model_label=tiny --nb_iter=$NBITER --load_audio=$LOAD --seed=$SEED --attack_name=$NAME
#python evaluate.py pgd.yaml --root=$RSROOT --data_csv_name=$DATA --model_label=tiny.en --tokenizer_name=gpt2 --nb_iter=$NBITER --load_audio=$LOAD --seed=$SEED  --attack_name=$NAME
#python evaluate.py pgd.yaml --root=$RSROOT --data_csv_name=$DATA --model_label=base --nb_iter=$NBITER --load_audio=$LOAD --seed=$SEED --attack_name=$NAME
#python evaluate.py pgd.yaml --root=$RSROOT --data_csv_name=$DATA --model_label=base.en --tokenizer_name=gpt2 --nb_iter=$NBITER --load_audio=$LOAD --seed=$SEED  --attack_name=$NAME
#python evaluate.py pgd.yaml --root=$RSROOT --data_csv_name=$DATA --model_label=small --nb_iter=$NBITER --load_audio=$LOAD --seed=$SEED --attack_name=$NAME
#python evaluate.py pgd.yaml --root=$RSROOT --data_csv_name=$DATA --model_label=small.en --tokenizer_name=gpt2 --nb_iter=$NBITER --load_audio=$LOAD --seed=$SEED  --attack_name=$NAME
#python evaluate.py pgd.yaml --root=$RSROOT --data_csv_name=$DATA --model_label=medium --nb_iter=$NBITER --load_audio=$LOAD --seed=$SEED --attack_name=$NAME 
#python evaluate.py pgd.yaml --root=$RSROOT --data_csv_name=$DATA --model_label=medium.en --tokenizer_name=gpt2 --nb_iter=$NBITER --load_audio=$LOAD --seed=$SEED --attack_name=$NAME
SEED=235
NBITER=200
python evaluate.py pgd.yaml --root=$RSROOT --data_csv_name=$DATA --model_label=small --nb_iter=$NBITER --load_audio=$LOAD --seed=$SEED --attack_name=$NAME --snr=$SNR
python evaluate.py pgd.yaml --root=$RSROOT --data_csv_name=$DATA --model_label=small.en --tokenizer_name=gpt2 --nb_iter=$NBITER --load_audio=$LOAD --seed=$SEED  --attack_name=$NAME --snr=$SNR
python evaluate.py pgd.yaml --root=$RSROOT --data_csv_name=$DATA --model_label=medium --nb_iter=$NBITER --load_audio=$LOAD --seed=$SEED --attack_name=$NAME  --snr=$SNR
python evaluate.py pgd.yaml --root=$RSROOT --data_csv_name=$DATA --model_label=medium.en --tokenizer_name=gpt2 --nb_iter=$NBITER --load_audio=$LOAD --seed=$SEED --attack_name=$NAME --snr=$SNR
python evaluate.py pgd.yaml --root=$RSROOT --data_csv_name=$DATA --model_label=tiny --nb_iter=$NBITER --load_audio=$LOAD --seed=$SEED --attack_name=$NAME --snr=$SNR
python evaluate.py pgd.yaml --root=$RSROOT --data_csv_name=$DATA --model_label=tiny.en --tokenizer_name=gpt2 --nb_iter=$NBITER --load_audio=$LOAD --seed=$SEED  --attack_name=$NAME --snr=$SNR
python evaluate.py pgd.yaml --root=$RSROOT --data_csv_name=$DATA --model_label=base --nb_iter=$NBITER --load_audio=$LOAD --seed=$SEED --attack_name=$NAME --snr=$SNR
python evaluate.py pgd.yaml --root=$RSROOT --data_csv_name=$DATA --model_label=base.en --tokenizer_name=gpt2 --nb_iter=$NBITER --load_audio=$LOAD --seed=$SEED  --attack_name=$NAME --snr=$SNR
