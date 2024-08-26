# use pretrained agent2 to generate SR image for agent1
echo 'log_tpM_action_input_vector_T1_8_smAgent1_bicubic_last_layer_ch_512'
CUDA_VISIBLE_DEVICES=0 python3 train_tpM.py --gpu 0 --dataset HistoSR  --output_name outputs/log_tpM_action_input_vector_T1_8_smAgent1_bicubic_last_layer_ch_512.txt

