import os
import sys
import argparse
import time
import torch
import psutil

import retro
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv

# Import the wrapper
from wrapper import SamuraiShowdownCustomWrapper


def make_env_for_subprocess(game, state, rendering=False, seed=0, env_id=0):
    """Create environment function for SubprocVecEnv - all imports MUST be inside"""

    def _init():
        # Import everything inside the function for subprocess compatibility
        import retro
        import gymnasium as gym
        from stable_baselines3.common.monitor import Monitor
        from wrapper import SamuraiShowdownCustomWrapper

        # Create environment (all environments now have the same render_mode)
        env = retro.make(
            game=game,
            state=state,
            use_restricted_actions=retro.Actions.FILTERED,
            obs_type=retro.Observations.IMAGE,
            render_mode="human" if rendering else None,
        )

        env = SamuraiShowdownCustomWrapper(
            env,
            reset_round=True,
            rendering=rendering,
            max_episode_steps=5000,
        )

        env = Monitor(env)
        env.reset(seed=seed)
        return env

    return _init


def linear_schedule(initial_value, final_value=0.0):
    """Linear scheduler"""

    def scheduler(progress):
        return final_value + progress * (initial_value - final_value)

    return scheduler


def check_system_resources(num_envs):
    """Check if system can handle the requested number of environments"""

    # Check CPU cores
    cpu_cores = psutil.cpu_count(logical=True)
    recommended_envs = min(cpu_cores, 84)  # Cap at 84 but consider CPU limits

    print(f"🖥️  System Resource Check:")
    print(f"   CPU Cores: {cpu_cores}")
    print(f"   Requested Envs: {num_envs}")

    if num_envs > cpu_cores:
        print(f"⚠️  Warning: {num_envs} environments > {cpu_cores} CPU cores")
        print(f"💡 Consider reducing to {recommended_envs} for optimal performance")

    # Check RAM
    ram_gb = psutil.virtual_memory().total / (1024**3)
    estimated_ram_usage = num_envs * 0.1  # ~100MB per environment

    print(f"   RAM: {ram_gb:.1f} GB")
    print(f"   Estimated RAM usage: {estimated_ram_usage:.1f} GB")

    if estimated_ram_usage > ram_gb * 0.8:
        print(f"❌ ERROR: Insufficient RAM!")
        print(f"💡 Reduce environments or add more RAM")
        return False

    return True


def get_actual_observation_dims(game, state):
    """Get actual observation dimensions from the wrapper"""
    try:
        # Create a temporary environment to get actual dimensions
        temp_env = retro.make(
            game=game,
            state=state,
            use_restricted_actions=retro.Actions.FILTERED,
            obs_type=retro.Observations.IMAGE,
            render_mode=None,
        )
        wrapped_env = SamuraiShowdownCustomWrapper(temp_env)
        obs_shape = wrapped_env.observation_space.shape
        temp_env.close()
        wrapped_env.env.close()
        return obs_shape
    except Exception as e:
        print(f"⚠️ Could not determine observation dimensions: {e}")
        # Fallback to estimated dimensions
        return (9, 168, 240)  # 9 frames, 75% of 224x320


def main():
    parser = argparse.ArgumentParser(description="Train Samurai Showdown Agent")
    parser.add_argument(
        "--total-timesteps", type=int, default=10000000, help="Total timesteps to train"
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=84,
        help="Number of parallel environments (default: 84)",
    )
    parser.add_argument(
        "--learning-rate", type=float, default=4e-3, help="Learning rate"
    )
    parser.add_argument(
        "--resume", type=str, default=None, help="Resume from saved model path"
    )
    parser.add_argument("--render", action="store_true", help="Enable rendering")
    parser.add_argument(
        "--use-default-state", action="store_true", help="Use default game state"
    )
    # New argument for batch size scaling
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Custom batch size (auto-calculated if not provided)",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=None,
        help="Custom n_steps (auto-calculated if not provided)",
    )

    args = parser.parse_args()

    # System resource check
    if not check_system_resources(args.num_envs):
        sys.exit(1)

    # GPU Check and Setup - MANDATORY
    print("🔍 Checking GPU availability...")
    if not torch.cuda.is_available():
        print("❌ ERROR: No GPU available!")
        print("💡 This training script requires CUDA GPU")
        print("💡 Check: nvidia-smi")
        print(
            "💡 Install: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
        )
        sys.exit(1)

    gpu_count = torch.cuda.device_count()
    current_gpu = torch.cuda.current_device()
    gpu_name = torch.cuda.get_device_name(current_gpu)
    gpu_memory = torch.cuda.get_device_properties(current_gpu).total_memory / 1e9

    print(f"✅ GPU Detected: {gpu_name}")
    print(f"   GPU Memory: {gpu_memory:.1f} GB")
    print(f"   Available GPUs: {gpu_count}")
    print(f"   Current GPU: {current_gpu}")

    device = "cuda"

    # Set CUDA optimizations
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False  # For maximum performance
    print("✅ CUDA optimizations enabled")

    game = "SamuraiShodown-Genesis"

    # Test if the game works
    print(f"🎮 Testing {game}...")
    try:
        test_env = retro.make(
            game=game,
            state=None,
            use_restricted_actions=retro.Actions.FILTERED,
            obs_type=retro.Observations.IMAGE,
            render_mode=None,
        )
        test_env.close()
        print(f"✅ Basic environment test passed")
    except Exception as e:
        print(f"❌ Basic environment test failed: {e}")
        return

    # Handle state
    if args.use_default_state:
        state = None
        print(f"🎮 Using default game state")
    else:
        if os.path.exists("samurai.state"):
            state = os.path.abspath("samurai.state")
            print(f"🎮 Using samurai.state file")
        else:
            print(f"❌ samurai.state not found, using default state")
            state = None

    # FIXED: Get actual observation dimensions
    print("🔍 Determining actual observation dimensions...")
    obs_shape = get_actual_observation_dims(game, state)
    num_frames, obs_height, obs_width = obs_shape
    print(f"✅ Observation shape: {obs_shape}")

    # FIXED: Correct memory calculation (1 byte per pixel for uint8)
    obs_size_mb = (num_frames * obs_height * obs_width) / (1024 * 1024)
    print(f"📊 Observation size: {obs_size_mb:.2f} MB per observation")

    # Determine actual number of environments for rendering
    if args.render:
        if args.num_envs > 8:
            print(f"🎮 Rendering mode: Using 4 environments with rendering")
            print(
                f"💡 For optimal viewing experience with {args.num_envs} requested environments"
            )
            actual_envs = 4
        else:
            print(f"🎮 Rendering mode: All {args.num_envs} environments will render")
            actual_envs = args.num_envs
    else:
        actual_envs = args.num_envs

    # FIXED: Calculate hyperparameters once with actual environment count
    print(f"🧠 Calculating hyperparameters for {actual_envs} environments...")

    # Calculate batch size
    if args.batch_size is None:
        batch_size = min(1344, actual_envs * 16)  # 16 per env, max 1344
    else:
        batch_size = args.batch_size

    # Calculate n_steps based on memory constraints
    if args.n_steps is None:
        # Target: Keep buffer under 4GB
        target_buffer_gb = 3.0  # Conservative target
        max_buffer_size_mb = target_buffer_gb * 1024

        # Calculate max steps based on memory constraint
        # Buffer size = n_steps * n_envs * obs_size_mb
        max_n_steps = int(max_buffer_size_mb / (actual_envs * obs_size_mb))

        # Choose n_steps based on environment count and memory constraints
        if actual_envs >= 64:
            n_steps = min(64, max_n_steps)  # Very small for many envs
        elif actual_envs >= 32:
            n_steps = min(128, max_n_steps)  # Small for moderate envs
        elif actual_envs >= 8:
            n_steps = min(256, max_n_steps)  # Medium for fewer envs
        else:
            n_steps = min(512, max_n_steps)  # Larger for very few envs

        # Ensure minimum viable n_steps
        n_steps = max(32, n_steps)
    else:
        n_steps = args.n_steps

    # Calculate actual buffer memory usage
    buffer_memory_gb = (n_steps * actual_envs * obs_size_mb) / 1024

    print(f"📈 Optimized hyperparameters for {actual_envs} environments:")
    print(f"   Batch size: {batch_size}")
    print(f"   N-steps: {n_steps}")
    print(f"   Buffer memory: {buffer_memory_gb:.2f} GB")

    # Memory safety check
    if buffer_memory_gb > 6:
        print(f"❌ ERROR: Buffer memory too large ({buffer_memory_gb:.2f} GB)!")
        max_safe_steps = int(5 * 1024 / (actual_envs * obs_size_mb))
        print(f"💡 Reduce n_steps to {max_safe_steps} or fewer")
        print(f"💡 Or reduce num_envs")
        sys.exit(1)

    # Enhanced VRAM estimation with correct values
    base_model_vram = 2.0  # Base model memory ~2GB
    env_vram_per_env = buffer_memory_gb / actual_envs  # Actual per-env memory
    batch_processing_vram = 1.5  # Additional memory for batch processing

    estimated_vram = base_model_vram + buffer_memory_gb + batch_processing_vram

    print(f"📊 VRAM estimation for {actual_envs} environments:")
    print(f"   Base model: {base_model_vram:.1f} GB")
    print(f"   Buffer memory: {buffer_memory_gb:.2f} GB")
    print(f"   Batch processing: {batch_processing_vram:.1f} GB")
    print(f"   Total estimated: {estimated_vram:.2f} GB")

    if estimated_vram > gpu_memory * 0.9:
        print("❌ ERROR: Insufficient GPU memory!")
        print(f"   Required: {estimated_vram:.2f} GB")
        print(f"   Available: {gpu_memory:.1f} GB")
        print("💡 Reduce --num-envs or --n-steps")
        sys.exit(1)
    elif estimated_vram > gpu_memory * 0.7:
        print("⚠️ Warning: High VRAM usage expected")
        print("💡 Monitor with: watch -n 1 nvidia-smi")

    save_dir = "trained_models_samurai"
    os.makedirs(save_dir, exist_ok=True)

    print(f"🚀 Samurai Showdown Training - Optimized Setup")
    print(f"   Game: {game}")
    print(f"   State: {state}")
    print(f"   Device: {device} (GPU MANDATORY)")
    print(f"   Total timesteps: {args.total_timesteps:,}")
    print(f"   Environments: {actual_envs}")
    print(f"   Learning rate: {args.learning_rate}")

    # Create environments with SubprocVecEnv
    print(f"🔧 Creating {actual_envs} environments with SubprocVecEnv...")

    try:
        env_fns = [
            make_env_for_subprocess(
                game,
                state=state,
                rendering=args.render,
                seed=i,
                env_id=i,
            )
            for i in range(actual_envs)
        ]

        env = SubprocVecEnv(env_fns)

        if args.render:
            print(f"✅ {actual_envs} environments created with rendering enabled")
        else:
            print(
                f"✅ {actual_envs} environments created with SubprocVecEnv (no rendering)"
            )

    except Exception as e:
        print(f"❌ Failed to create SubprocVecEnv environments: {e}")
        import traceback

        traceback.print_exc()
        return

    # Create or load model - GPU ONLY with optimized settings
    if args.resume and os.path.exists(args.resume):
        print(f"📂 Loading model from: {args.resume}")
        print(
            f"⚠️  Warning: Overriding saved model hyperparameters for memory optimization"
        )

        # Load model but override hyperparameters for new environment setup
        model = PPO.load(args.resume, env=env, device="cuda")

        # Force update hyperparameters to match current memory-optimized setup
        model.n_steps = n_steps
        model.batch_size = batch_size
        model.n_epochs = 3

        # Recreate rollout buffer with new parameters
        model._setup_model()

        print(f"✅ Model loaded on GPU with overridden hyperparameters:")
        print(f"   n_steps: {model.n_steps}")
        print(f"   batch_size: {model.batch_size}")
        print(f"   n_epochs: {model.n_epochs}")
    else:
        print(f"🧠 Creating new PPO model optimized for {actual_envs} environments")
        lr_schedule = linear_schedule(args.learning_rate, args.learning_rate * 0.3)

        model = PPO(
            "CnnPolicy",
            env,
            device="cuda",  # GPU ONLY
            verbose=1,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=3,
            gamma=0.99,
            learning_rate=lr_schedule,
            clip_range=linear_schedule(0.2, 0.1),
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            gae_lambda=0.95,
            tensorboard_log="logs_samurai",
        )
        print(f"✅ PPO model created on GPU with optimized settings")

        # Verify model is on GPU
        print(f"🔍 Model device verification:")
        for name, param in model.policy.named_parameters():
            print(f"   {name}: {param.device}")
            break  # Just show first parameter as example

    # Checkpoint callback - adjusted frequency
    checkpoint_freq = max(500000 // actual_envs, 10000)  # At least every 10k steps
    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_freq,
        save_path=save_dir,
        name_prefix=f"ppo_samurai_{actual_envs}env",
    )

    print(f"💾 Checkpoint frequency: every {checkpoint_freq} steps")

    # Training - GPU ONLY
    start_time = time.time()
    print(
        f"🏋️ Starting GPU training with {actual_envs} environments for {args.total_timesteps:,} timesteps"
    )
    print("💡 Monitor GPU usage with: watch -n 1 nvidia-smi")
    print("💡 Monitor system resources with: htop")

    try:
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=[checkpoint_callback],
            reset_num_timesteps=not bool(args.resume),
        )

        training_time = time.time() - start_time
        print(f"🎉 GPU training completed in {training_time/3600:.1f} hours!")

    except KeyboardInterrupt:
        print(f"⏹️ Training interrupted")
        training_time = time.time() - start_time
        print(f"Training time: {training_time/3600:.1f} hours")

    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback

        traceback.print_exc()

    finally:
        env.close()
        torch.cuda.empty_cache()  # Clear GPU memory
        print("🧹 GPU memory cleared")

    # Save final model
    final_model_path = os.path.join(save_dir, f"ppo_samurai_final_{actual_envs}env.zip")
    model.save(final_model_path)
    print(f"💾 Final model saved to: {final_model_path}")

    print("✅ Training complete!")


if __name__ == "__main__":
    main()
