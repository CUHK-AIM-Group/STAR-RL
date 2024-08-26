#CUDA_VISIBLE_DEVICE=1
python main.py --batch-size 10 --resume --checkpoint checkpoints/model.pt --challenge singlecoil --data-path None --data-parallel --test
