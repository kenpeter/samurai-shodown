





python train.py --resume trained_models_simple/ppo_simple_10150000_steps.zip --render --total-timesteps 10000000  --learning-rate 3e-4



python train.py --render --total-timesteps 100000000




python train.py --resume trained_models_samurai/ppo_samurai_4env_30000000_steps.zip



python train.py --total-timesteps 10000000 --learning-rate 4e-3 --render



python improved_training_script.py --batch-size 1024 --mixed-precision --target-vram 10.0