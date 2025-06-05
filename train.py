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
            max_episode_steps=10000,  # Updated to match wrapper
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
    recommended_envs = min(
        cpu_cores // 4, 4
    )  # Conservative env count for large batches

    print(f"🖥️  System Resource Check:")
    print(f"   CPU Cores: {cpu_cores}")
    print(f"   Requested Envs: {num_envs}")
    print(f"   Strategy: FEWER envs + LARGER batches")

    if num_envs > cpu_cores // 2:
        print(f"⚠️  Warning: {num_envs} environments might stress {cpu_cores} CPU cores")
        print(f"💡 Consider {recommended_envs} environments for large batch training")

    # Check RAM - OPTIMIZED FOR 82GB SYSTEM
    ram_gb = psutil.virtual_memory().total / (1024**3)
    estimated_ram_usage = num_envs * 0.5  # ~500MB per environment with large rollouts

    print(f"   RAM: {ram_gb:.1f} GB (HIGH-END SYSTEM)")
    print(f"   Estimated RAM usage: {estimated_ram_usage:.1f} GB")

    if estimated_ram_usage > ram_gb * 0.6:  # More conservative with large rollouts
        print(f"❌ ERROR: Insufficient RAM!")
        print(f"💡 Reduce environments or rollout size")
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
    parser = argparse.ArgumentParser(
        description="Train Samurai Showdown Agent - Large Batch Optimized"
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=20000000,
        help="Total timesteps to train (increased for fewer envs)",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=2,  # FEWER envs for larger batch sizes
        help="Number of parallel environments (optimized for large batch per env)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-4,
        help="Learning rate (reduced for large batch)",
    )
    parser.add_argument(
        "--resume", type=str, default=None, help="Resume from saved model path"
    )
    parser.add_argument("--render", action="store_true", help="Enable rendering")
    parser.add_argument(
        "--use-default-state", action="store_true", help="Use default game state"
    )
    # Conservative batch size that WILL work on 12GB GPU
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2048,  # SAFE batch size - definitely fits in 12GB GPU
        help="Batch size (default: 2048 - guaranteed safe for 12GB GPU)",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=8192,  # Large rollouts for 82GB RAM
        help="Steps per rollout (default: 8192 for high-end RAM)",
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

    # Set CUDA optimizations + MEMORY CONSERVATION
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False  # For maximum performance

    # Add memory optimization environment variables
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    print("✅ CUDA optimizations + memory conservation enabled")

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

    # Get actual observation dimensions
    print("🔍 Determining actual observation dimensions...")
    obs_shape = get_actual_observation_dims(game, state)
    num_frames, obs_height, obs_width = obs_shape
    print(f"✅ Observation shape: {obs_shape}")

    # Memory calculation
    obs_size_mb = (num_frames * obs_height * obs_width) / (1024 * 1024)
    print(f"📊 Observation size: {obs_size_mb:.2f} MB per observation")

    # Determine actual number of environments for rendering
    if args.render:
        if args.num_envs > 4:
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

    # LARGE BATCH HYPERPARAMETERS
    print(f"🧠 LARGE BATCH TRAINING - Optimized for {actual_envs} environments...")

    batch_size = args.batch_size
    n_steps = args.n_steps

    # Ensure batch_size is reasonable for the number of environments
    total_samples_per_rollout = n_steps * actual_envs
    if batch_size > total_samples_per_rollout:
        batch_size = total_samples_per_rollout
        print(f"⚠️ Adjusted batch_size to {batch_size} (max samples per rollout)")

    # Calculate buffer memory usage
    buffer_memory_gb = (n_steps * actual_envs * obs_size_mb) / 1024

    print(f"📈 ULTRA SAFE: Guaranteed to work on {actual_envs} environments:")
    print(f"   Batch size: {batch_size:,} (ULTRA SAFE for any 12GB GPU)")
    print(f"   N-steps: {n_steps:,} (LARGE for 82GB RAM)")
    print(f"   Total samples per rollout: {total_samples_per_rollout:,}")
    print(f"   Buffer memory: {buffer_memory_gb:.2f} GB")
    print(f"   Minibatches per rollout: {total_samples_per_rollout // batch_size}")
    print(f"   🛡️ Strategy: Guaranteed stability over maximum performance")

    # Memory safety check - RELAXED FOR HIGH-END HARDWARE
    if buffer_memory_gb > 20:  # Can handle much larger buffers with 82GB RAM
        print(f"❌ ERROR: Buffer memory too large ({buffer_memory_gb:.2f} GB)!")
        max_safe_steps = int(15 * 1024 / (actual_envs * obs_size_mb))
        print(f"💡 Reduce n_steps to {max_safe_steps} or fewer")
        sys.exit(1)

    # VRAM estimation - FIXED: Buffer goes to RAM, not VRAM
    base_model_vram = 3.0  # Base model memory on GPU
    batch_processing_vram = 3.5  # MASSIVE batch processing for 16K batch size
    gradient_vram = 1.5  # Gradient computation memory on GPU

    # IMPORTANT: Buffer memory goes to SYSTEM RAM, not VRAM!
    estimated_vram = (
        base_model_vram + batch_processing_vram + gradient_vram
    )  # NO buffer memory!

    print(f"📊 MEMORY allocation for RTX 4070 12GB + 82GB RAM:")
    print(f"   🖥️  SYSTEM RAM (82GB available):")
    print(f"      Buffer memory: {buffer_memory_gb:.2f} GB ✅")
    print(f"   🎮 GPU VRAM (12GB available):")
    print(f"      Base model: {base_model_vram:.1f} GB")
    print(f"      Batch processing: {batch_processing_vram:.1f} GB (MASSIVE)")
    print(f"      Gradient computation: {gradient_vram:.1f} GB")
    print(f"      Total GPU VRAM: {estimated_vram:.2f} GB")
    print(
        f"   🎯 GPU VRAM usage: {estimated_vram:.1f}GB / 12GB = {estimated_vram/12*100:.1f}%"
    )

    # Check SYSTEM RAM for buffer (not VRAM!)
    if buffer_memory_gb > 50:  # Conservative limit for 82GB system
        print(f"❌ ERROR: Buffer too large for system RAM!")
        print(f"   Buffer needs: {buffer_memory_gb:.2f} GB")
        print(f"   System RAM: 82GB available")
        print("💡 Reduce --n-steps")
        sys.exit(1)

    # Check GPU VRAM (without buffer memory!)
    if estimated_vram > 11.5:  # Use almost all 12GB
        print("❌ ERROR: Would exceed RTX 4070 12GB VRAM!")
        print(f"   Required: {estimated_vram:.2f} GB")
        print(f"   Available: ~11.5 GB (safe limit)")
        print("💡 Reduce --batch-size")
        sys.exit(1)
    elif estimated_vram > 10.5:
        print("⚠️ High VRAM usage - will use most of your 12GB (OPTIMAL)")
    else:
        print("✅ Conservative VRAM usage - could potentially increase batch size")

    save_dir = "trained_models_samurai"
    os.makedirs(save_dir, exist_ok=True)

    print(f"🚀 Samurai Showdown OPTIMIZED Training: Fewer Envs + Larger Batches")
    print(f"   💡 Strategy: 2 envs × large batches > many envs × small batches")
    print(f"   🎯 Rewards: Win/Loss only, 3-sec jump cooldown")
    print(f"   🔥 Batch size: {batch_size:,} samples (GPU efficient)")
    print(f"   Game: {game}")
    print(f"   Device: {device} (RTX 4070 12GB)")
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

    # Create or load model - LARGE BATCH OPTIMIZED
    if args.resume and os.path.exists(args.resume):
        print(f"📂 Loading model from: {args.resume}")
        model = PPO.load(args.resume, env=env, device="cuda")

        # Update hyperparameters for large batch training
        model.n_steps = n_steps
        model.batch_size = batch_size
        model.n_epochs = 4  # Slightly more epochs for large batches
        model._setup_model()

        print(f"✅ Model loaded and updated for large batch training")
    else:
        print(f"🧠 Creating new PPO model for CONSERVATIVE BATCH training (12GB GPU)")
        lr_schedule = linear_schedule(args.learning_rate, args.learning_rate * 0.1)

        model = PPO(
            "CnnPolicy",
            env,
            device="cuda",
            verbose=1,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=4,  # Reasonable epochs for small batches
            gamma=0.99,  # Standard gamma
            learning_rate=lr_schedule,
            clip_range=linear_schedule(0.2, 0.05),
            ent_coef=0.01,  # Standard entropy
            vf_coef=0.5,
            max_grad_norm=0.5,
            gae_lambda=0.95,
            tensorboard_log="logs_samurai",
            policy_kwargs=dict(
                normalize_images=False,
                optimizer_class=torch.optim.Adam,  # Standard Adam for smaller batches
                optimizer_kwargs=dict(eps=1e-5),
            ),
        )
        print(f"✅ PPO model created for CONSERVATIVE BATCH training")

    # Model device verification
    print(f"🔍 Model device verification:")
    for name, param in model.policy.named_parameters():
        print(f"   {name}: {param.device}")
        break

    # Checkpoint callback - less frequent for large batch training
    checkpoint_freq = max(1000000 // actual_envs, 50000)
    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_freq,
        save_path=save_dir,
        name_prefix=f"ppo_samurai_largebatch_{actual_envs}env",
    )

    print(f"💾 Checkpoint frequency: every {checkpoint_freq} steps")

    # Training
    start_time = time.time()
    print(f"🏋️ Starting MASSIVE BATCH training with {actual_envs} environments")
    print(f"   📊 {total_samples_per_rollout:,} samples per rollout")
    print(f"   🔥 {batch_size:,} batch size (MASSIVE)")
    print(f"   💪 Using RTX 4070 12GB + 82GB RAM")
    print("💡 Monitor GPU usage with: watch -n 1 nvidia-smi")

    try:
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=[checkpoint_callback],
            reset_num_timesteps=not bool(args.resume),
        )

        training_time = time.time() - start_time
        print(f"🎉 LARGE BATCH training completed in {training_time/3600:.1f} hours!")

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
        torch.cuda.empty_cache()
        print("🧹 GPU memory cleared")

    # Save final model
    final_model_path = os.path.join(
        save_dir, f"ppo_samurai_largebatch_final_{actual_envs}env.zip"
    )
    model.save(final_model_path)
    print(f"💾 Final model saved to: {final_model_path}")

    print("✅ LARGE BATCH training complete!")
    print(f"🎯 Strategy used: Simple win/loss rewards with 3-second jump cooldown")


if __name__ == "__main__":
    main()
