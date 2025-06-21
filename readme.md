





python train.py --resume ttrained_models_simple_prime/ppo_simple_prime_4575000_steps.zip --render --total-timesteps 4000000 --batch-size 3072 --n-steps 3072  --learning-rate 4e-4




python train.py --batch-size 3072 --ent-coef 0.9 --n-steps 512 --total-timesteps 1000000 --render


python train.py --resume trained_models_simple_prime/ppo_simple_prime_final.zip --batch-size 3072 --ent-coef 0.7 --n-steps 1024 --total-timesteps 1000000 --render


python train.py --resume trained_models_simple_prime/ppo_simple_prime_final.zip --batch-size 3072 --ent-coef 0.5 --n-steps 1024 --total-timesteps 1000000 --render


python train.py --resume trained_models_simple_prime/ppo_simple_prime_final.zip --batch-size 3072 --ent-coef 0.3 --n-steps 1024 --total-timesteps 1000000 --render


python train.py --resume trained_models_simple_prime/ppo_simple_prime_final.zip --batch-size 3072 --ent-coef 0.2 --n-steps 2048 --total-timesteps 1000000

python train.py --resume trained_models_simple_prime/ppo_simple_prime_final.zip --batch-size 3072 --ent-coef 0.1 --n-steps 3072 --total-timesteps 1000000



~/anaconda3/envs/samurai-showdown/lib/python3.10/site-packages/retro/data/stable/SamuraiShodown-Genesis

{
  "info": {
    "enemy_health": {
      "address": 16730511,
      "type": "|u1"
    },
    "health": {
      "address": 16730175,
      "type": "|u1"
    },
    "round": {
      "address": 16755265,
      "type": "|u1"
    },
    "score": {
      "address": 16730166,
      "type": ">d4"
    }
  }
}



python -c "import torch; torch.cuda.empty_cache(); torch.cuda.ipc_collect()"
