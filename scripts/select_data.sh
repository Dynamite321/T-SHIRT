export CUDA_VISIBLE_DEVICES=0

python token_loss.py
python compute_stat.py --ratio 0.5
python tshirt_select.py --key avg_noisy_sifd_50 --output_path datasets/alpaca_gpt4/tshirt_k_50.json