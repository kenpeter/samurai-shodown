# Phase 1: Quick improvement with low exploration
python train.py --enable-jepa --resume trained_models_jepa_prime/ppo_jepa_prime_1000000_steps.zip --ent-coef 0.05 --total-timesteps 2000000 --batch-size 1024 --n-steps 2048

# Quick test after 2M more steps
python eval.py --episodes 5 --show-jepa-predictions




# Phase 2: Balanced exploration for strategy development  
python train.py --enable-jepa --resume trained_models_jepa_prime/ppo_jepa_prime_3000000_steps.zip --ent-coef 0.15 --total-timesteps 5000000 --batch-size 1536 --n-steps 4096

# Test progress
python eval.py --episodes 10 --deterministic