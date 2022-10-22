#! /bin/bash
source activate sp3
NBITER=1
SEED=1000
LOAD=False
NAME=lang
DATA=sanity
LANGATTACK=en
MODEL=medium
python evaluate.py lang.yaml --lang_CV=hy-AM --lang_attack=$LANGATTACK --data_csv_name=$DATA --model_label=$MODEL --root=$RSROOT --load_audio=$LOAD --seed=$SEED --nb_iter=$NBITER
python evaluate.py lang.yaml --lang_CV=lt --lang_attack=$LANGATTACK --data_csv_name=$DATA --model_label=$MODEL --root=$RSROOT --load_audio=$LOAD --seed=$SEED --nb_iter=$NBITER
python evaluate.py lang.yaml --lang_CV=cs --lang_attack=$LANGATTACK --data_csv_name=$DATA --model_label=$MODEL --root=$RSROOT --load_audio=$LOAD --seed=$SEED --nb_iter=$NBITER
python evaluate.py lang.yaml --lang_CV=da --lang_attack=$LANGATTACK --data_csv_name=$DATA --model_label=$MODEL --root=$RSROOT --load_audio=$LOAD --seed=$SEED --nb_iter=$NBITER
python evaluate.py lang.yaml --lang_CV=id --lang_attack=$LANGATTACK --data_csv_name=$DATA --model_label=$MODEL --root=$RSROOT --load_audio=$LOAD --seed=$SEED --nb_iter=$NBITER
python evaluate.py lang.yaml --lang_CV=it --lang_attack=$LANGATTACK --data_csv_name=$DATA --model_label=$MODEL --root=$RSROOT --load_audio=$LOAD --seed=$SEED --nb_iter=$NBITER
LANGATTACK=tl
python evaluate.py lang.yaml --lang_CV=hy-AM --lang_attack=$LANGATTACK --data_csv_name=$DATA --model_label=$MODEL --root=$RSROOT --load_audio=$LOAD --seed=$SEED --nb_iter=$NBITER
python evaluate.py lang.yaml --lang_CV=lt --lang_attack=$LANGATTACK --data_csv_name=$DATA --model_label=$MODEL --root=$RSROOT --load_audio=$LOAD --seed=$SEED --nb_iter=$NBITER
python evaluate.py lang.yaml --lang_CV=cs --lang_attack=$LANGATTACK --data_csv_name=$DATA --model_label=$MODEL --root=$RSROOT --load_audio=$LOAD --seed=$SEED --nb_iter=$NBITER
python evaluate.py lang.yaml --lang_CV=da --lang_attack=$LANGATTACK --data_csv_name=$DATA --model_label=$MODEL --root=$RSROOT --load_audio=$LOAD --seed=$SEED --nb_iter=$NBITER
python evaluate.py lang.yaml --lang_CV=id --lang_attack=$LANGATTACK --data_csv_name=$DATA --model_label=$MODEL --root=$RSROOT --load_audio=$LOAD --seed=$SEED --nb_iter=$NBITER
python evaluate.py lang.yaml --lang_CV=it --lang_attack=$LANGATTACK --data_csv_name=$DATA --model_label=$MODEL --root=$RSROOT --load_audio=$LOAD --seed=$SEED --nb_iter=$NBITER
LANGATTACK=sr
python evaluate.py lang.yaml --lang_CV=hy-AM --lang_attack=$LANGATTACK --data_csv_name=$DATA --model_label=$MODEL --root=$RSROOT --load_audio=$LOAD --seed=$SEED --nb_iter=$NBITER
python evaluate.py lang.yaml --lang_CV=lt --lang_attack=$LANGATTACK --data_csv_name=$DATA --model_label=$MODEL --root=$RSROOT --load_audio=$LOAD --seed=$SEED --nb_iter=$NBITER
python evaluate.py lang.yaml --lang_CV=cs --lang_attack=$LANGATTACK --data_csv_name=$DATA --model_label=$MODEL --root=$RSROOT --load_audio=$LOAD --seed=$SEED --nb_iter=$NBITER
python evaluate.py lang.yaml --lang_CV=da --lang_attack=$LANGATTACK --data_csv_name=$DATA --model_label=$MODEL --root=$RSROOT --load_audio=$LOAD --seed=$SEED --nb_iter=$NBITER
python evaluate.py lang.yaml --lang_CV=id --lang_attack=$LANGATTACK --data_csv_name=$DATA --model_label=$MODEL --root=$RSROOT --load_audio=$LOAD --seed=$SEED --nb_iter=$NBITER
python evaluate.py lang.yaml --lang_CV=it --lang_attack=$LANGATTACK --data_csv_name=$DATA --model_label=$MODEL --root=$RSROOT --load_audio=$LOAD --seed=$SEED --nb_iter=$NBITER


#python evaluate.py lang.yaml --lang_CV=mt --lang_attack=$LANGATTACK --data_csv_name=$DATA --model_label=$MODEL --root=$RSROOT --load_audio=$LOAD --seed=$SEED --nb_iter=$NBITER
#python evaluate.py lang.yaml --lang_CV=bg --lang_attack=$LANGATTACK --data_csv_name=$DATA --model_label=$MODEL --root=$RSROOT --load_audio=$LOAD --seed=$SEED --nb_iter=$NBITER
#python evaluate.py lang.yaml --lang_CV=vi --lang_attack=$LANGATTACK --data_csv_name=$DATA --model_label=$MODEL --root=$RSROOT --load_audio=$LOAD --seed=$SEED --nb_iter=$NBITER
#python evaluate.py lang.yaml --lang_CV=fi --lang_attack=$LANGATTACK --data_csv_name=$DATA --model_label=$MODEL --root=$RSROOT --load_audio=$LOAD --seed=$SEED --nb_iter=$NBITER
