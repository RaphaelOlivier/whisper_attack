#! /bin/bash
source activate sp3
NBITER=4000
SEED=1000
LOAD=True
NAME=cw
EPS=0.1
CST=100
MAXDECR=3
DATA=sanity
LANGATTACK=en
python evaluate.py cw.yaml --lang_CV=as --lang_attack=$LANGATTACK --data_csv_name=$DATA --model_label=tiny --root=$RSROOT --load_audio=$LOAD --seed=$SEED
python evaluate.py cw.yaml --lang_CV=bg --lang_attack=$LANGATTACK --data_csv_name=$DATA --model_label=tiny --root=$RSROOT --load_audio=$LOAD --seed=$SEED
python evaluate.py cw.yaml --lang_CV=da --lang_attack=$LANGATTACK --data_csv_name=$DATA --model_label=tiny --root=$RSROOT --load_audio=$LOAD --seed=$SEED
python evaluate.py cw.yaml --lang_CV=ga-IE --lang_attack=$LANGATTACK --data_csv_name=$DATA --model_label=tiny --root=$RSROOT --load_audio=$LOAD --seed=$SEED
python evaluate.py cw.yaml --lang_CV=vi --lang_attack=$LANGATTACK --data_csv_name=$DATA --model_label=tiny --root=$RSROOT --load_audio=$LOAD --seed=$SEED
python evaluate.py cw.yaml --lang_CV=id --lang_attack=$LANGATTACK --data_csv_name=$DATA --model_label=tiny --root=$RSROOT --load_audio=$LOAD --seed=$SEED
