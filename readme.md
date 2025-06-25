# Start VERY HIGH for JEPA discovery
python train.py --enable-jepa --ent-coef 0.6 --total-timesteps 1500000 --batch-size 1024 --n-steps 2048  --render

# Then moderate for pattern learning  
python train.py --enable-jepa --resume model.zip --ent-coef 0.3 --total-timesteps 2500000

# Finally low for mastery
python train.py --enable-jepa --resume model.zip --ent-coef 0.1 --total-timesteps 4000000