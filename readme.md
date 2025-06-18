




python train.py --model-size efficientnet-b3 --target-vram 9.0 --batch-size 256 --total-timesteps 100000000 --learning-rate 3e-4 --resume trained_models_fighting_optimized/ppo_fighting_optimized_2400000_steps.zip --render



python train.py --model-size efficientnet-b3 --target-vram 9 --batch-size 256 --render



===


# COMPLETE MEMORY OPTIMIZATION FOR 11.6GB GPU

# Step 1: Clear GPU memory first
python -c "import torch; torch.cuda.empty_cache(); print('GPU cache cleared')"

# Step 2: Check GPU memory
nvidia-smi

# Step 3: Try the SAFEST configuration first
python train.py --model-size ultra-light --target-vram 8.0 --batch-size 256 --n-steps 256

# If that works, gradually increase:
# Step 4a: Increase batch size
python train.py --model-size ultra-light --target-vram 8.0 --batch-size 512 --n-steps 256

# Step 4b: Increase n-steps  
python train.py --model-size ultra-light --target-vram 8.0 --batch-size 512 --n-steps 512

# Step 4c: Increase target VRAM
python train.py --model-size ultra-light --target-vram 9.0 --batch-size 512 --n-steps 512

# If ultra-light is too slow, try basic with conservative settings:
python train.py --model-size basic --target-vram 7.0 --batch-size 128 --n-steps 256

# DEBUGGING COMMANDS:
# Monitor memory during training:
watch -n 1 nvidia-smi

# Check PyTorch memory usage:
python -c "
import torch
print(f'GPU memory allocated: {torch.cuda.memory_allocated()/1024**3:.2f} GB')
print(f'GPU memory reserved: {torch.cuda.memory_reserved()/1024**3:.2f} GB')
"

# If still getting OOM, use MINIMAL settings:
python train.py --model-size ultra-light --target-vram 6.0 --batch-size 64 --n-steps 128 --total-timesteps 1000000