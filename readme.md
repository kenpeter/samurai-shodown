





python train.py --resume trained_models_simple/ppo_simple_10150000_steps.zip --render --total-timesteps 10000000  --learning-rate 3e-4






python train.py --batch-size 128 --n-steps 128 --target-vram 8.0 --render --total-timesteps 100000000 --learning-rate 0.0004 --mixed-precision --model-size full --no-accumulation

