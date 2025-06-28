# Stage 1: Initial learning
python train.py --enable-jepa --total-timesteps 5000000 --ent-coef 0.15

# Stage 2: Refinement
python train.py --enable-jepa --resume trained_models_jepa_prime/ppo_jepa_prime_final.zip \
  --total-timesteps 10000000 --ent-coef 0.08 --learning-rate 1.0e-4





python train.py \
    --total-timesteps 20000000 \
    --n-steps 4096 \
    --batch-size 512 \
    --lr 0.0003 \
    --ent-coef 0.015 \
    --frame-stack 8 \
    --render