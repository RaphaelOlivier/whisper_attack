#! /bin/bash
source activate sp3
NBITER=50
SEED=1000
LOAD=True
NAME=lang
DATA=test-100
MODEL=medium
SNR=40
LANGATTACK=en
python evaluate.py lang.yaml --lang_CV=hy-AM --lang_attack=$LANGATTACK --data_csv_name=$DATA --model_label=$MODEL --root=$RSROOT --load_audio=$LOAD --seed=$SEED --nb_iter=$NBITER --snr=$SNR
python evaluate.py lang.yaml --lang_CV=lt --lang_attack=$LANGATTACK --data_csv_name=$DATA --model_label=$MODEL --root=$RSROOT --load_audio=$LOAD --seed=$SEED --nb_iter=$NBITER --snr=$SNR
python evaluate.py lang.yaml --lang_CV=cs --lang_attack=$LANGATTACK --data_csv_name=$DATA --model_label=$MODEL --root=$RSROOT --load_audio=$LOAD --seed=$SEED --nb_iter=$NBITER --snr=$SNR
python evaluate.py lang.yaml --lang_CV=da --lang_attack=$LANGATTACK --data_csv_name=$DATA --model_label=$MODEL --root=$RSROOT --load_audio=$LOAD --seed=$SEED --nb_iter=$NBITER --snr=$SNR
python evaluate.py lang.yaml --lang_CV=id --lang_attack=$LANGATTACK --data_csv_name=$DATA --model_label=$MODEL --root=$RSROOT --load_audio=$LOAD --seed=$SEED --nb_iter=$NBITER --snr=$SNR
python evaluate.py lang.yaml --lang_CV=it --lang_attack=$LANGATTACK --data_csv_name=$DATA --model_label=$MODEL --root=$RSROOT --load_audio=$LOAD --seed=$SEED --nb_iter=$NBITER --snr=$SNR
python evaluate.py lang.yaml --lang_CV=en --lang_attack=$LANGATTACK --data_csv_name=$DATA --model_label=$MODEL --root=$RSROOT --load_audio=$LOAD --seed=$SEED --nb_iter=$NBITER --snr=$SNR

LANGATTACK=tl
python evaluate.py lang.yaml --lang_CV=hy-AM --lang_attack=$LANGATTACK --data_csv_name=$DATA --model_label=$MODEL --root=$RSROOT --load_audio=$LOAD --seed=$SEED --nb_iter=$NBITER --snr=$SNR
python evaluate.py lang.yaml --lang_CV=lt --lang_attack=$LANGATTACK --data_csv_name=$DATA --model_label=$MODEL --root=$RSROOT --load_audio=$LOAD --seed=$SEED --nb_iter=$NBITER --snr=$SNR
python evaluate.py lang.yaml --lang_CV=cs --lang_attack=$LANGATTACK --data_csv_name=$DATA --model_label=$MODEL --root=$RSROOT --load_audio=$LOAD --seed=$SEED --nb_iter=$NBITER --snr=$SNR
python evaluate.py lang.yaml --lang_CV=da --lang_attack=$LANGATTACK --data_csv_name=$DATA --model_label=$MODEL --root=$RSROOT --load_audio=$LOAD --seed=$SEED --nb_iter=$NBITER --snr=$SNR
python evaluate.py lang.yaml --lang_CV=id --lang_attack=$LANGATTACK --data_csv_name=$DATA --model_label=$MODEL --root=$RSROOT --load_audio=$LOAD --seed=$SEED --nb_iter=$NBITER --snr=$SNR
python evaluate.py lang.yaml --lang_CV=it --lang_attack=$LANGATTACK --data_csv_name=$DATA --model_label=$MODEL --root=$RSROOT --load_audio=$LOAD --seed=$SEED --nb_iter=$NBITER --snr=$SNR
python evaluate.py lang.yaml --lang_CV=en --lang_attack=$LANGATTACK --data_csv_name=$DATA --model_label=$MODEL --root=$RSROOT --load_audio=$LOAD --seed=$SEED --nb_iter=$NBITER --snr=$SNR

LANGATTACK=sr
python evaluate.py lang.yaml --lang_CV=hy-AM --lang_attack=$LANGATTACK --data_csv_name=$DATA --model_label=$MODEL --root=$RSROOT --load_audio=$LOAD --seed=$SEED --nb_iter=$NBITER --snr=$SNR
python evaluate.py lang.yaml --lang_CV=lt --lang_attack=$LANGATTACK --data_csv_name=$DATA --model_label=$MODEL --root=$RSROOT --load_audio=$LOAD --seed=$SEED --nb_iter=$NBITER --snr=$SNR
python evaluate.py lang.yaml --lang_CV=cs --lang_attack=$LANGATTACK --data_csv_name=$DATA --model_label=$MODEL --root=$RSROOT --load_audio=$LOAD --seed=$SEED --nb_iter=$NBITER --snr=$SNR
python evaluate.py lang.yaml --lang_CV=da --lang_attack=$LANGATTACK --data_csv_name=$DATA --model_label=$MODEL --root=$RSROOT --load_audio=$LOAD --seed=$SEED --nb_iter=$NBITER --snr=$SNR
python evaluate.py lang.yaml --lang_CV=id --lang_attack=$LANGATTACK --data_csv_name=$DATA --model_label=$MODEL --root=$RSROOT --load_audio=$LOAD --seed=$SEED --nb_iter=$NBITER --snr=$SNR
python evaluate.py lang.yaml --lang_CV=it --lang_attack=$LANGATTACK --data_csv_name=$DATA --model_label=$MODEL --root=$RSROOT --load_audio=$LOAD --seed=$SEED --nb_iter=$NBITER --snr=$SNR
python evaluate.py lang.yaml --lang_CV=en --lang_attack=$LANGATTACK --data_csv_name=$DATA --model_label=$MODEL --root=$RSROOT --load_audio=$LOAD --seed=$SEED --nb_iter=$NBITER --snr=$SNR

