




python train.py --resume trained_models/ppo_sf2_7999760_steps.zip --total-timesteps 10000000 --learning-rate 1e-3 --render




python train.py \
  --resume trained_models/ppo_sf2_7999760_steps.zip \
  --total-timesteps 50000000 \
  --learning-rate 0.0004 \
  --ent-coef 0.03 \
  --render