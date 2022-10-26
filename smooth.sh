#! /bin/bash
source activate sp3
LOAD=True
DATA=test-clean-100
NAME=pgd
NBITER=200

SNR=35
SEED=335
python evaluate.py pgd.yaml --root=$RSROOT --data_csv_name=$DATA --model_label=base --nb_iter=$NBITER --load_audio=$LOAD --seed=$SEED --attack_name=$NAME  --snr=$SNR
python evaluate.py pgd.yaml --root=$RSROOT --data_csv_name=$DATA --model_label=base.smooth.02 --nb_iter=$NBITER --load_audio=$LOAD --seed=$SEED --attack_name=$NAME  --snr=$SNR
python evaluate.py pgd.yaml --root=$RSROOT --data_csv_name=$DATA --model_label=base.smooth.03 --nb_iter=$NBITER --load_audio=$LOAD --seed=$SEED --attack_name=$NAME  --snr=$SNR

SNR=40
SEED=340
python evaluate.py pgd.yaml --root=$RSROOT --data_csv_name=$DATA --model_label=base --nb_iter=$NBITER --load_audio=$LOAD --seed=$SEED --attack_name=$NAME  --snr=$SNR
python evaluate.py pgd.yaml --root=$RSROOT --data_csv_name=$DATA --model_label=base.smooth.02 --nb_iter=$NBITER --load_audio=$LOAD --seed=$SEED --attack_name=$NAME  --snr=$SNR
python evaluate.py pgd.yaml --root=$RSROOT --data_csv_name=$DATA --model_label=base.smooth.03 --nb_iter=$NBITER --load_audio=$LOAD --seed=$SEED --attack_name=$NAME  --snr=$SNR
