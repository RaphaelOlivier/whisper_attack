#! /bin/bash
source activate sp3
NBITER=30
SEED=1030
LOAD=True
NAME=lang
DATA=test-100
MODEL=medium
SNR=45
# LANGATTACK=en
# python run_attack.py attack_configs/whisper/lang.yaml --lang_CV=hy-AM --lang_attack=$LANGATTACK --data_csv_name=$DATA --model_label=$MODEL --root=$RSROOT --load_audio=$LOAD --seed=$SEED --nb_iter=$NBITER --snr=$SNR
# python run_attack.py attack_configs/whisper/lang.yaml --lang_CV=lt --lang_attack=$LANGATTACK --data_csv_name=$DATA --model_label=$MODEL --root=$RSROOT --load_audio=$LOAD --seed=$SEED --nb_iter=$NBITER --snr=$SNR
# python run_attack.py attack_configs/whisper/lang.yaml --lang_CV=cs --lang_attack=$LANGATTACK --data_csv_name=$DATA --model_label=$MODEL --root=$RSROOT --load_audio=$LOAD --seed=$SEED --nb_iter=$NBITER --snr=$SNR
# python run_attack.py attack_configs/whisper/lang.yaml --lang_CV=da --lang_attack=$LANGATTACK --data_csv_name=$DATA --model_label=$MODEL --root=$RSROOT --load_audio=$LOAD --seed=$SEED --nb_iter=$NBITER --snr=$SNR
# python run_attack.py attack_configs/whisper/lang.yaml --lang_CV=id --lang_attack=$LANGATTACK --data_csv_name=$DATA --model_label=$MODEL --root=$RSROOT --load_audio=$LOAD --seed=$SEED --nb_iter=$NBITER --snr=$SNR
# python run_attack.py attack_configs/whisper/lang.yaml --lang_CV=it --lang_attack=$LANGATTACK --data_csv_name=$DATA --model_label=$MODEL --root=$RSROOT --load_audio=$LOAD --seed=$SEED --nb_iter=$NBITER --snr=$SNR
# python run_attack.py attack_configs/whisper/lang.yaml --lang_CV=en --lang_attack=$LANGATTACK --data_csv_name=$DATA --model_label=$MODEL --root=$RSROOT --load_audio=$LOAD --seed=$SEED --nb_iter=$NBITER --snr=$SNR

# LANGATTACK=tl
# python run_attack.py attack_configs/whisper/lang.yaml --lang_CV=hy-AM --lang_attack=$LANGATTACK --data_csv_name=$DATA --model_label=$MODEL --root=$RSROOT --load_audio=$LOAD --seed=$SEED --nb_iter=$NBITER --snr=$SNR
# python run_attack.py attack_configs/whisper/lang.yaml --lang_CV=lt --lang_attack=$LANGATTACK --data_csv_name=$DATA --model_label=$MODEL --root=$RSROOT --load_audio=$LOAD --seed=$SEED --nb_iter=$NBITER --snr=$SNR
# python run_attack.py attack_configs/whisper/lang.yaml --lang_CV=cs --lang_attack=$LANGATTACK --data_csv_name=$DATA --model_label=$MODEL --root=$RSROOT --load_audio=$LOAD --seed=$SEED --nb_iter=$NBITER --snr=$SNR
# python run_attack.py attack_configs/whisper/lang.yaml --lang_CV=da --lang_attack=$LANGATTACK --data_csv_name=$DATA --model_label=$MODEL --root=$RSROOT --load_audio=$LOAD --seed=$SEED --nb_iter=$NBITER --snr=$SNR
# python run_attack.py attack_configs/whisper/lang.yaml --lang_CV=id --lang_attack=$LANGATTACK --data_csv_name=$DATA --model_label=$MODEL --root=$RSROOT --load_audio=$LOAD --seed=$SEED --nb_iter=$NBITER --snr=$SNR
# python run_attack.py attack_configs/whisper/lang.yaml --lang_CV=it --lang_attack=$LANGATTACK --data_csv_name=$DATA --model_label=$MODEL --root=$RSROOT --load_audio=$LOAD --seed=$SEED --nb_iter=$NBITER --snr=$SNR
# python run_attack.py attack_configs/whisper/lang.yaml --lang_CV=en --lang_attack=$LANGATTACK --data_csv_name=$DATA --model_label=$MODEL --root=$RSROOT --load_audio=$LOAD --seed=$SEED --nb_iter=$NBITER --snr=$SNR

# LANGATTACK=sr
# python run_attack.py attack_configs/whisper/lang.yaml --lang_CV=hy-AM --lang_attack=$LANGATTACK --data_csv_name=$DATA --model_label=$MODEL --root=$RSROOT --load_audio=$LOAD --seed=$SEED --nb_iter=$NBITER --snr=$SNR
# python run_attack.py attack_configs/whisper/lang.yaml --lang_CV=lt --lang_attack=$LANGATTACK --data_csv_name=$DATA --model_label=$MODEL --root=$RSROOT --load_audio=$LOAD --seed=$SEED --nb_iter=$NBITER --snr=$SNR
# python run_attack.py attack_configs/whisper/lang.yaml --lang_CV=cs --lang_attack=$LANGATTACK --data_csv_name=$DATA --model_label=$MODEL --root=$RSROOT --load_audio=$LOAD --seed=$SEED --nb_iter=$NBITER --snr=$SNR
# python run_attack.py attack_configs/whisper/lang.yaml --lang_CV=da --lang_attack=$LANGATTACK --data_csv_name=$DATA --model_label=$MODEL --root=$RSROOT --load_audio=$LOAD --seed=$SEED --nb_iter=$NBITER --snr=$SNR
# python run_attack.py attack_configs/whisper/lang.yaml --lang_CV=id --lang_attack=$LANGATTACK --data_csv_name=$DATA --model_label=$MODEL --root=$RSROOT --load_audio=$LOAD --seed=$SEED --nb_iter=$NBITER --snr=$SNR
# python run_attack.py attack_configs/whisper/lang.yaml --lang_CV=it --lang_attack=$LANGATTACK --data_csv_name=$DATA --model_label=$MODEL --root=$RSROOT --load_audio=$LOAD --seed=$SEED --nb_iter=$NBITER --snr=$SNR
# python run_attack.py attack_configs/whisper/lang.yaml --lang_CV=en --lang_attack=$LANGATTACK --data_csv_name=$DATA --model_label=$MODEL --root=$RSROOT --load_audio=$LOAD --seed=$SEED --nb_iter=$NBITER --snr=$SNR

MODEL=base
LANGATTACK=sr
EPOCHS=100
NBITER=100
#python fit_attacker.py attack_configs/whisper/univ_lang_fit.yaml --lang_CV=it --lang_attack=$LANGATTACK --epochs=$EPOCHS --data_csv_name=$DATA --model_label=$MODEL --root=$RSROOT --load_audio=$LOAD --seed=$SEED --nb_iter=$NBITER --eps=0.005
python run_attack.py attack_configs/whisper/univ_lang.yaml --lang_CV=it --lang_attack=$LANGATTACK --model_label=$MODEL --data_csv_name=$DATA --root=$RSROOT --load_audio=$LOAD --seed=$SEED --nb_iter=$NBITER --eps=0.005