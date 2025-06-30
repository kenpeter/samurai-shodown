python train.py \
    --total-timesteps 5000000 \
    --n-steps 4096 \
    --batch-size 512 \
    --lr 0.00025 \
    --ent-coef 0.01 \
    --frame-stack 8 \
    --resume trained_models_jepa_anti_spam/ppo_jepa_anti_spam_750000_steps.zip --render




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






# Evaluate with JEPA features and rendering
python eval.py trained_models_jepa_reward_shaped/ppo_jepa_shaped_7400000_steps.zip --episodes 5 --render

# Quick evaluation without rendering
python eval.py trained_models_jepa_reward_shaped/ppo_jepa_shaped_7400000_steps.zip --episodes 10

# Evaluate without JEPA (standard CNN mode)
python eval.py trained_models_jepa_reward_shaped/ppo_jepa_shaped_7400000_steps.zip --no-jepa --episodes 10

# Save results to JSON
python eval.py trained_models_jepa_reward_shaped/ppo_jepa_shaped_7400000_steps.zip --episodes 20 --output results.json