# Stage 1: Initial learning
python train.py --enable-jepa --total-timesteps 5000000 --ent-coef 0.15

# Stage 2: Refinement
python train.py --enable-jepa --resume trained_models_jepa_prime/ppo_jepa_prime_final.zip \
  --total-timesteps 10000000 --ent-coef 0.08 --learning-rate 1.0e-4