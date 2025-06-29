# Stage 1: Initial learning
python train.py --enable-jepa --total-timesteps 5000000 --ent-coef 0.15

# Stage 2: Refinement
python train.py --enable-jepa --resume trained_models_jepa_prime/ppo_jepa_prime_final.zip \
  --total-timesteps 10000000 --ent-coef 0.08 --learning-rate 1.0e-4





python train.py \
    --total-timesteps 1000000 \
    --n-steps 3096 \
    --batch-size 1024 \
    --lr 0.0003 \
    --ent-coef 0.015 \
    --frame-stack 8 \
    --render \
    --resume trained_models_jepa_reward_shaped/ppo_jepa_shaped_750000_steps.zip






Input Observation: (batch, channels, height, width)
↓
CNN Features: (batch, feature_dim)
↓  
Visual History: (seq_len, feature_dim) → Stack → (1, seq_len, feature_dim)
Game History: (seq_len, 8) → Stack → (1, seq_len, 8)
↓
Transformer Input: (1, seq_len, d_model) where d_model = feature_dim + 8
↓
Positional Encoding: Adds (1, seq_len, d_model) 
↓
Transformer Output: (1, seq_len, d_model)
↓
Final Representation: (1, d_model) [last timestep]
↓
Predictions: (1, prediction_horizon) for each outcome