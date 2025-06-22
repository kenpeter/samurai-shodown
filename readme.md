# Day 1-2: Quick Start
python train.py --styles aggressive --total-timesteps 150000 --ent-coef 0.10 --learning-rate 5e-4 --n-steps 512 --batch-size 256 --render

# Day 3-4: Balanced Learning  
python train.py --styles balanced --total-timesteps 150000 --ent-coef 0.08 --learning-rate 4e-4 --n-steps 768 --batch-size 384

# Day 5-7: Foundation Complete
python train.py --styles aggressive balanced --total-timesteps 200000 --ent-coef 0.07 --learning-rate 3e-4 --n-steps 1024 --batch-size 512 --sequential





week2

# Day 8-9: Multi-Agent Introduction
python train.py --styles aggressive balanced defensive --total-timesteps 150000 --ent-coef 0.06 --learning-rate 3e-4 --n-steps 1024 --batch-size 512 --sequential

# Day 10-12: Breakthrough Push (Resume from Week 1)
python train.py --resume ncsoft_breakthrough_models/aggressive_final.zip --styles aggressive --total-timesteps 200000 --ent-coef 0.08 --learning-rate 3.5e-4 --n-steps 1536 --batch-size 768

python train.py --resume ncsoft_breakthrough_models/balanced_final.zip --styles balanced --total-timesteps 200000 --ent-coef 0.06 --learning-rate 3e-4 --n-steps 1536 --batch-size 768

# Day 13-14: Cross-Style Mastery
python train.py --styles aggressive balanced defensive --total-timesteps 150000 --ent-coef 0.05 --learning-rate 2.5e-4 --n-steps 1536 --batch-size 1024 --sequential



week3


# Day 15-16: Large Batch Stability
python train.py --resume ncsoft_breakthrough_models/aggressive_final.zip --styles aggressive --total-timesteps 150000 --ent-coef 0.05 --learning-rate 2.5e-4 --n-steps 2048 --batch-size 1024

python train.py --resume ncsoft_breakthrough_models/balanced_final.zip --styles balanced --total-timesteps 150000 --ent-coef 0.04 --learning-rate 2e-4 --n-steps 2048 --batch-size 1024

# Day 17-19: Advanced Multi-Agent
python train.py --styles aggressive balanced defensive --total-timesteps 200000 --ent-coef 0.04 --learning-rate 2e-4 --n-steps 2048 --batch-size 1536 --sequential

# Day 20-21: Tactical Refinement
python train.py --resume ncsoft_breakthrough_models/balanced_final.zip --styles balanced --total-timesteps 150000 --ent-coef 0.03 --learning-rate 1.5e-4 --n-steps 2048 --batch-size 1536



week4

# Day 22-24: Fine-Tuning
python train.py --resume ncsoft_breakthrough_models/aggressive_final.zip --styles aggressive --total-timesteps 150000 --ent-coef 0.03 --learning-rate 1.5e-4 --n-steps 2048 --batch-size 2048

python train.py --resume ncsoft_breakthrough_models/balanced_final.zip --styles balanced --total-timesteps 150000 --ent-coef 0.025 --learning-rate 1e-4 --n-steps 2048 --batch-size 2048

# Day 25-26: Professional Level
python train.py --styles aggressive balanced defensive --total-timesteps 150000 --ent-coef 0.02 --learning-rate 1e-4 --n-steps 2048 --batch-size 2048 --sequential

# Day 27-28: Final Polish
python train.py --resume ncsoft_breakthrough_models/balanced_final.zip --styles balanced --total-timesteps 200000 --ent-coef 0.02 --learning-rate 8e-5 --n-steps 2048 --batch-size 2048






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
