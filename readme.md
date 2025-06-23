# 1.1 Initial Exploration - Learn basic patterns
python train.py --ent-coef 0.5 --n-steps 512 --batch-size 512 --total-timesteps 1000000 --learning-rate 3e-4 --render

# 1.2 Pattern Recognition - Start seeing opponent habits  
python train.py --resume trained_models_jepa_prime/ppo_jepa_prime_1000000_steps.zip --ent-coef 0.4 --n-steps 1024 --batch-size 512 --total-timesteps 1500000 --learning-rate 2.5e-4


fix:
python train.py --resume trained_models_jepa_prime/ppo_jepa_prime_2050000_steps.zip --ent-coef 0.8 --n-steps 512 --batch-size 3072 --total-timesteps 500000 --learning-rate 4e-4








# 1.3 Basic Strategic Learning
python train.py --resume trained_models_jepa_prime/ppo_jepa_prime_2500000_steps.zip --ent-coef 0.3 --n-steps 1536 --batch-size 768 --total-timesteps 2000000 --learning-rate 2e-4





# 2.1 Counter-Attack Training - Learn to respond
python train.py --resume trained_models_jepa_prime/ppo_jepa_prime_4500000_steps.zip --ent-coef 0.25 --n-steps 2048 --batch-size 1024 --total-timesteps 2500000 --learning-rate 2e-4

# 2.2 Timing Mastery - Perfect counter-attack timing
python train.py --resume trained_models_jepa_prime/ppo_jepa_prime_7000000_steps.zip --ent-coef 0.2 --n-steps 2048 --batch-size 1024 --total-timesteps 2000000 --learning-rate 1.8e-4

# 2.3 Advanced Planning - Multi-step strategies
python train.py --resume trained_models_jepa_prime/ppo_jepa_prime_9000000_steps.zip --ent-coef 0.15 --n-steps 3072 --batch-size 1536 --total-timesteps 2000000 --learning-rate 1.5e-4





# 3.1 Strategic Mastery - Consistent performance
python train.py --resume trained_models_jepa_prime/ppo_jepa_prime_11000000_steps.zip --ent-coef 0.1 --n-steps 3072 --batch-size 1536 --total-timesteps 2000000 --learning-rate 1.2e-4

# 3.2 Precision Training - Fine-tune responses
python train.py --resume trained_models_jepa_prime/ppo_jepa_prime_13000000_steps.zip --ent-coef 0.08 --n-steps 4096 --batch-size 2048 --total-timesteps 1500000 --learning-rate 1e-4

# 3.3 Final Polish - Tournament ready
python train.py --resume trained_models_jepa_prime/ppo_jepa_prime_14500000_steps.zip --ent-coef 0.05 --n-steps 4096 --batch-size 2048 --total-timesteps 500000 --learning-rate 8e-5