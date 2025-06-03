





python train.py --resume trained_models_samurai/ppo_samurai_4env_10000000_steps.zip --render --total-timesteps 10000000  --learning-rate 1*e-3



python train.py --render --total-timesteps 10000000




python train.py --resume trained_models_samurai/ppo_samurai_4env_20000000_steps.zip --num-envs 84 --learning-rate 1e-4 --render --total-timesteps 10000000 