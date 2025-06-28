# Stage 1: Initial learning
python train.py --enable-jepa --total-timesteps 5000000 --ent-coef 0.15

# Stage 2: Refinement
python train.py --enable-jepa --resume trained_models_jepa_prime/ppo_jepa_prime_final.zip \
  --total-timesteps 10000000 --ent-coef 0.08 --learning-rate 1.0e-4





# Resume from your 11.8M model, but with a lower LR to adapt to the new reward function
python train.py --resume trained_models_jepa_robust/ppo_jepa_robust_11900000_steps.zip --lr 1e-4