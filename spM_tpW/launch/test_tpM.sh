
echo "test_results_T1_25_T2_1_th_0.01_discriminator_13000_all_cut_edge"
CUDA_VISIBLE_DEVICES=0 python3 test_tpM.py --dataset HistoSR --gpu 0 \
--model_name 2_15_18_8 --episodes 13000 --episode_len_patch_test 25  --episode_len_test 1 \
--use_discriminator --disc_th 0.01 --cut_edge --save_images \
> ./logs/2_15_18_8/test_results_T1_25_T2_1_th_0.01_discriminator_13000_all_cut_edge_save_results.txt
