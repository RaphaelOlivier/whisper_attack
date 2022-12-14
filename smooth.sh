#! /bin/bash
LOAD=True
DATA=test-clean-100
NAME=pgd
NBITER=200

SNR=35
SEED=335
python run_attack.py attack_configs/pgd.yaml --root=$RSROOT --data_csv_name=$DATA --model_label=base --nb_iter=$NBITER --load_audio=$LOAD --seed=$SEED --attack_name=$NAME  --snr=$SNR
python run_attack.py attack_configs/pgd.yaml --root=$RSROOT --data_csv_name=$DATA --model_label=base.smooth.02 --nb_iter=$NBITER --load_audio=$LOAD --seed=$SEED --attack_name=$NAME  --snr=$SNR
python run_attack.py attack_configs/pgd.yaml --root=$RSROOT --data_csv_name=$DATA --model_label=base.smooth.03 --nb_iter=$NBITER --load_audio=$LOAD --seed=$SEED --attack_name=$NAME  --snr=$SNR

SNR=40
SEED=340
python run_attack.py attack_configs/pgd.yaml --root=$RSROOT --data_csv_name=$DATA --model_label=base --nb_iter=$NBITER --load_audio=$LOAD --seed=$SEED --attack_name=$NAME  --snr=$SNR
python run_attack.py attack_configs/pgd.yaml --root=$RSROOT --data_csv_name=$DATA --model_label=base.smooth.02 --nb_iter=$NBITER --load_audio=$LOAD --seed=$SEED --attack_name=$NAME  --snr=$SNR
python run_attack.py attack_configs/pgd.yaml --root=$RSROOT --data_csv_name=$DATA --model_label=base.smooth.03 --nb_iter=$NBITER --load_audio=$LOAD --seed=$SEED --attack_name=$NAME  --snr=$SNR
