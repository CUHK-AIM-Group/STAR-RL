echo "test_results_T1_8_T2_3_bicubic_cut_edge_episode_14000"
CUDA_VISIBLE_DEVICES=1 python3 test.py --dataset HistoSR --gpu 0 \
--model_name 2_14_18_8 --episodes 14000 --episode_len_patch_test 8  --episode_len_test 3 \
--data_degradation bicubic --cut_edge \
> ./logs/2_14_18_8/test_results_T1_8_T2_3_bicubic_cut_edge_episode_14000.txt
