echo 'log_patch_worker_T_8_gamma_1'
CUDA_VISIBLE_DEVICES=0 python3 train.py --gpu 0 --dataset HistoSR > ./outputs/log_patch_worker_T_8_gamma_1.txt