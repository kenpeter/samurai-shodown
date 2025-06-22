day 1,2
python train.py --styles aggressive --total-timesteps 150000 --ent-coef 0.10 --learning-rate 5e-4 --n-steps 512 --batch-size 256 --render

day 3,4
python train.py --styles balanced --total-timesteps 150000 --ent-coef 0.08 --learning-rate 4e-4 --n-steps 768 --batch-size 384 --render

day 5, 7
# Continue aggressive agent with better hyperparameters
python train.py --resume ncsoft_breakthrough_models/aggressive_final.zip --styles aggressive --total-timesteps 100000 --ent-coef 0.07 --learning-rate 3e-4 --n-steps 1024 --batch-size 512 --render

# Continue balanced agent (can now fight aggressive agents!)
python train.py --resume ncsoft_breakthrough_models/balanced_final.zip --styles balanced --total-timesteps 100000 --ent-coef 0.07 --learning-rate 3e-4 --n-steps 1024 --batch-size 512 --render



day8,9
# Create defensive agent (can fight aggressive + balanced from Week 1)
python train.py --styles defensive --total-timesteps 150000 --ent-coef 0.06 --learning-rate 3e-4 --n-steps 1024 --batch-size 512 --render


day10-12
# Resume aggressive with breakthrough parameters*
python train.py --resume ncsoft_breakthrough_models/aggressive_final.zip --styles aggressive --total-timesteps 200000 --ent-coef 0.08 --learning-rate 3.5e-4 --n-steps 1536 --batch-size 768 --render

# Resume balanced with breakthrough parameters  
python train.py --resume ncsoft_breakthrough_models/balanced_final.zip --styles balanced --total-timesteps 200000 --ent-coef 0.06 --learning-rate 3e-4 --n-steps 1536 --batch-size 768 --render

# Resume defensive with breakthrough parameters
python train.py --resume ncsoft_breakthrough_models/defensive_final.zip --styles defensive --total-timesteps 200000 --ent-coef 0.05 --learning-rate 2.8e-4 --n-steps 1536 --batch-size 768 --render





day13,14
# Resume aggressive for cross-style mastery
python train.py --resume ncsoft_breakthrough_models/aggressive_final.zip --styles aggressive --total-timesteps 75000 --ent-coef 0.05 --learning-rate 2.5e-4 --n-steps 1536 --batch-size 1024 --render

# Resume balanced for cross-style mastery
python train.py --resume ncsoft_breakthrough_models/balanced_final.zip --styles balanced --total-timesteps 75000 --ent-coef 0.05 --learning-rate 2.5e-4 --n-steps 1536 --batch-size 1024 --render

# Resume defensive for cross-style mastery
python train.py --resume ncsoft_breakthrough_models/defensive_final.zip --styles defensive --total-timesteps 75000 --ent-coef 0.04 --learning-rate 2.3e-4 --n-steps 1536 --batch-size 1024 --render




day15,16
# Resume aggressive with large batch training
python train.py --resume ncsoft_breakthrough_models/aggressive_final.zip --styles aggressive --total-timesteps 150000 --ent-coef 0.05 --learning-rate 2.5e-4 --n-steps 2048 --batch-size 1024 --render

# Resume balanced with large batch training
python train.py --resume ncsoft_breakthrough_models/balanced_final.zip --styles balanced --total-timesteps 150000 --ent-coef 0.04 --learning-rate 2e-4 --n-steps 2048 --batch-size 1024 --render

# Resume defensive with large batch training
python train.py --resume ncsoft_breakthrough_models/defensive_final.zip --styles defensive --total-timesteps 150000 --ent-coef 0.035 --learning-rate 1.8e-4 --n-steps 2048 --batch-size 1024 --render



day17,19
# Resume aggressive for advanced multi-agent
python train.py --resume ncsoft_breakthrough_models/aggressive_final.zip --styles aggressive --total-timesteps 67000 --ent-coef 0.04 --learning-rate 2e-4 --n-steps 2048 --batch-size 1536 --render

# Resume balanced for advanced multi-agent
python train.py --resume ncsoft_breakthrough_models/balanced_final.zip --styles balanced --total-timesteps 67000 --ent-coef 0.04 --learning-rate 2e-4 --n-steps 2048 --batch-size 1536 --render

# Resume defensive for advanced multi-agent
python train.py --resume ncsoft_breakthrough_models/defensive_final.zip --styles defensive --total-timesteps 66000 --ent-coef 0.035 --learning-rate 1.8e-4 --n-steps 2048 --batch-size 1536 --render



day20,21
# Resume balanced for tactical refinement (usually best performer)
python train.py --resume ncsoft_breakthrough_models/balanced_final.zip --styles balanced --total-timesteps 150000 --ent-coef 0.03 --learning-rate 1.5e-4 --n-steps 2048 --batch-size 1536 --render




day 22,24
# Resume aggressive for fine-tuning
python train.py --resume ncsoft_breakthrough_models/aggressive_final.zip --styles aggressive --total-timesteps 150000 --ent-coef 0.03 --learning-rate 1.5e-4 --n-steps 2048 --batch-size 2048

# Resume balanced for fine-tuning
python train.py --resume ncsoft_breakthrough_models/balanced_final.zip --styles balanced --total-timesteps 150000 --ent-coef 0.025 --learning-rate 1e-4 --n-steps 2048 --batch-size 2048

# Resume defensive for fine-tuning
python train.py --resume ncsoft_breakthrough_models/defensive_final.zip --styles defensive --total-timesteps 150000 --ent-coef 0.02 --learning-rate 1.2e-4 --n-steps 2048 --batch-size 2048



25,26
# Resume aggressive for professional level
python train.py --resume ncsoft_breakthrough_models/aggressive_final.zip --styles aggressive --total-timesteps 50000 --ent-coef 0.02 --learning-rate 1e-4 --n-steps 2048 --batch-size 2048

# Resume balanced for professional level
python train.py --resume ncsoft_breakthrough_models/balanced_final.zip --styles balanced --total-timesteps 50000 --ent-coef 0.02 --learning-rate 1e-4 --n-steps 2048 --batch-size 2048

# Resume defensive for professional level  
python train.py --resume ncsoft_breakthrough_models/defensive_final.zip --styles defensive --total-timesteps 50000 --ent-coef 0.018 --learning-rate 9e-5 --n-steps 2048 --batch-size 2048



27,28
# Resume best performing agent for final polish (check win rates to determine)
python train.py --resume ncsoft_breakthrough_models/balanced_final.zip --styles balanced --total-timesteps 200000 --ent-coef 0.02 --learning-rate 8e-5 --n-steps 2048 --batch-size 2048

# Alternatively, if aggressive is performing better:
# python train.py --resume ncsoft_breakthrough_models/aggressive_final.zip --styles aggressive --total-timesteps 200000 --ent-coef 0.02 --learning-rate 8e-5 --n-steps 2048 --batch-size 2048




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
