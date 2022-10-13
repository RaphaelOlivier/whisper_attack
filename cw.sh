#! /bin/bash
source activate sp3
NBITER=1000
SEED=1004
LOAD=False
NAME=cw
EPS=0.015
CST=10

python evaluate.py cw.yaml --root=$RSROOT --data_csv_name=test-clean-20 --model_label=tiny --nb_iter=$NBITER --load_audio=$LOAD --seed=$SEED --attack_name=$NAME --eps=$EPS --const=$CST
python evaluate.py cw.yaml --root=$RSROOT --data_csv_name=test-clean-20 --model_label=tiny.en --tokenizer_name=gpt2 --nb_iter=$NBITER --load_audio=$LOAD --seed=$SEED --attack_name=$NAME --eps=$EPS --const=$CST
python evaluate.py cw.yaml --root=$RSROOT --data_csv_name=test-clean-20 --model_label=base --nb_iter=$NBITER --load_audio=$LOAD --seed=$SEED --attack_name=$NAME --eps=$EPS --const=$CST
python evaluate.py cw.yaml --root=$RSROOT --data_csv_name=test-clean-20 --model_label=base.en --tokenizer_name=gpt2 --nb_iter=$NBITER --load_audio=$LOAD --seed=$SEED --attack_name=$NAME --eps=$EPS --const=$CST
python evaluate.py cw.yaml --root=$RSROOT --data_csv_name=test-clean-20 --model_label=small --nb_iter=$NBITER --load_audio=$LOAD --seed=$SEED --attack_name=$NAME --eps=$EPS --const=$CST
python evaluate.py cw.yaml --root=$RSROOT --data_csv_name=test-clean-20 --model_label=small.en --tokenizer_name=gpt2 --nb_iter=$NBITER --load_audio=$LOAD --seed=$SEED --attack_name=$NAME --eps=$EPS --const=$CST
python evaluate.py cw.yaml --root=$RSROOT --data_csv_name=test-clean-20 --model_label=medium --nb_iter=$NBITER --load_audio=$LOAD --seed=$SEED --attack_name=$NAME --eps=$EPS --const=$CST
python evaluate.py cw.yaml --root=$RSROOT --data_csv_name=test-clean-20 --model_label=medium.en --tokenizer_name=gpt2 --nb_iter=$NBITER --load_audio=$LOAD --seed=$SEED --attack_name=$NAME --eps=$EPS --const=$CST
python evaluate.py cw.yaml --root=$RSROOT --data_csv_name=test-clean-20 --model_label=large --nb_iter=$NBITER --load_audio=$LOAD --seed=$SEED --attack_name=$NAME --eps=$EPS --const=$CST
