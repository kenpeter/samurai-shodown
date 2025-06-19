





python train.py --resume trained_models_fighting_optimized/ppo_fighting_optimized_16950000_steps.zip --render --total-timesteps 10000000  --learning-rate 1e-4 --enable-prime



python train.py --render --total-timesteps 100000000  --batch-size 256 --enable-prime --n-steps 256


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
