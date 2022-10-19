#! /bin/bash
source activate sp3
NBITER=4000
SEED=1002
LOAD=True
NAME=cw
EPS=0.1
CST=100
MAXDECR=3
DATA=test-clean-20
#python evaluate.py cw.yaml --root=$RSROOT --data_csv_name=$DATA --model_label=tiny --nb_iter=$NBITER --load_audio=$LOAD --seed=$SEED --attack_name=$NAME --eps=$EPS --const=$CST --max_decr=$MAXDECR
#python evaluate.py cw.yaml --root=$RSROOT --data_csv_name=$DATA --model_label=tiny.en --tokenizer_name=gpt2 --nb_iter=$NBITER --load_audio=$LOAD --seed=$SEED --attack_name=$NAME --eps=$EPS --const=$CST --max_decr=$MAXDECR
#python evaluate.py cw.yaml --root=$RSROOT --data_csv_name=$DATA --model_label=base --nb_iter=$NBITER --load_audio=$LOAD --seed=$SEED --attack_name=$NAME --eps=$EPS --const=$CST --max_decr=$MAXDECR
#python evaluate.py cw.yaml --root=$RSROOT --data_csv_name=$DATA --model_label=base.en --tokenizer_name=gpt2 --nb_iter=$NBITER --load_audio=$LOAD --seed=$SEED --attack_name=$NAME --eps=$EPS --const=$CST --max_decr=$MAXDECR
#python evaluate.py cw.yaml --root=$RSROOT --data_csv_name=$DATA --model_label=small --nb_iter=$NBITER --load_audio=$LOAD --seed=$SEED --attack_name=$NAME --eps=$EPS --const=$CST --max_decr=$MAXDECR
#python evaluate.py cw.yaml --root=$RSROOT --data_csv_name=$DATA --model_label=small.en --tokenizer_name=gpt2 --nb_iter=$NBITER --load_audio=$LOAD --seed=$SEED --attack_name=$NAME --eps=$EPS --const=$CST --max_decr=$MAXDECR
python evaluate.py cw.yaml --root=$RSROOT --data_csv_name=$DATA --model_label=medium --nb_iter=$NBITER --load_audio=$LOAD --seed=$SEED --attack_name=$NAME --eps=$EPS --const=$CST --max_decr=$MAXDECR --lr=0.001
python evaluate.py cw.yaml --root=$RSROOT --data_csv_name=$DATA --model_label=medium.en --tokenizer_name=gpt2 --nb_iter=$NBITER --load_audio=$LOAD --seed=$SEED --attack_name=$NAME --eps=$EPS --const=$CST --max_decr=$MAXDECR --lr=0.001
#python evaluate.py cw.yaml --root=$RSROOT --data_csv_name=$DATA --model_label=large --nb_iter=$NBITER --load_audio=$LOAD --seed=$SEED --attack_name=$NAME --eps=$EPS --const=$CST --max_decr=$MAXDECR
