#! /bin/bash
source activate sp3
NBITER=2000
SEED=2000
LOAD=False
NAME=cw
EPS=0.1
MAXDECR=8
DATA=test-clean-20
CONF=0.0
DECRFACTOR=0.7
CST=4
LR=0.01
python run_attack.py attack_configs/cw.yaml --root=$RSROOT --data_csv_name=$DATA --model_label=tiny --nb_iter=$NBITER --load_audio=$LOAD --seed=$SEED --attack_name=$NAME --eps=$EPS --const=$CST --max_decr=$MAXDECR --lr=$LR --confidence=$CONF --decrease_factor_eps=$DECRFACTOR
python run_attack.py attack_configs/cw.yaml --root=$RSROOT --data_csv_name=$DATA --model_label=tiny.en --tokenizer_name=gpt2 --nb_iter=$NBITER --load_audio=$LOAD --seed=$SEED --attack_name=$NAME --eps=$EPS --const=$CST --max_decr=$MAXDECR --lr=$LR --confidence=$CONF --decrease_factor_eps=$DECRFACTOR
python run_attack.py attack_configs/cw.yaml --root=$RSROOT --data_csv_name=$DATA --model_label=base --nb_iter=$NBITER --load_audio=$LOAD --seed=$SEED --attack_name=$NAME --eps=$EPS --const=$CST --max_decr=$MAXDECR --lr=$LR --confidence=$CONF --decrease_factor_eps=$DECRFACTOR
python run_attack.py attack_configs/cw.yaml --root=$RSROOT --data_csv_name=$DATA --model_label=base.en --tokenizer_name=gpt2 --nb_iter=$NBITER --load_audio=$LOAD --seed=$SEED --attack_name=$NAME --eps=$EPS --const=$CST --max_decr=$MAXDECR --lr=$LR --confidence=$CONF --decrease_factor_eps=$DECRFACTOR
python run_attack.py attack_configs/cw.yaml --root=$RSROOT --data_csv_name=$DATA --model_label=small --nb_iter=$NBITER --load_audio=$LOAD --seed=$SEED --attack_name=$NAME --eps=$EPS --const=$CST --max_decr=$MAXDECR --lr=$LR --confidence=$CONF --decrease_factor_eps=$DECRFACTOR
python run_attack.py attack_configs/cw.yaml --root=$RSROOT --data_csv_name=$DATA --model_label=small.en --tokenizer_name=gpt2 --nb_iter=$NBITER --load_audio=$LOAD --seed=$SEED --attack_name=$NAME --eps=$EPS --const=$CST --max_decr=$MAXDECR --lr=$LR --confidence=$CONF --decrease_factor_eps=$DECRFACTOR
python run_attack.py attack_configs/cw.yaml --root=$RSROOT --data_csv_name=$DATA --model_label=medium --nb_iter=$NBITER --load_audio=$LOAD --seed=$SEED --attack_name=$NAME --eps=$EPS --const=$CST --max_decr=$MAXDECR --lr=$LR --confidence=$CONF --decrease_factor_eps=$DECRFACTOR
python run_attack.py attack_configs/cw.yaml --root=$RSROOT --data_csv_name=$DATA --model_label=medium.en --tokenizer_name=gpt2 --nb_iter=$NBITER --load_audio=$LOAD --seed=$SEED --attack_name=$NAME --eps=$EPS --const=$CST --max_decr=$MAXDECR --lr=$LR --confidence=$CONF --decrease_factor_eps=$DECRFACTOR
python run_attack.py attack_configs/cw.yaml --root=$RSROOT --data_csv_name=$DATA --model_label=large --nb_iter=$NBITER --load_audio=$LOAD --seed=$SEED --attack_name=$NAME --eps=$EPS --const=$CST --max_decr=$MAXDECR --lr=$LR --confidence=$CONF --decrease_factor_eps=$DECRFACTOR
