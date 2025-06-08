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

        # Create environment
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
            max_episode_steps=15000,  # Slightly reduced for CUDA efficiency
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





def calculate_optimal_batch_size(obs_shape, num_envs, target_vram_gb=10.0):
    """Calculate optimal batch size based on VRAM constraints"""
    
    # Observation memory calculation
    num_frames, height, width = obs_shape
    obs_size_bytes = num_frames * height * width * 4  # float32
    obs_size_mb = obs_size_bytes / (1024 * 1024)
    
    print(f"📊 VRAM optimization for {target_vram_gb}GB target:")
    print(f"   Observation size: {obs_size_mb:.2f} MB (float32)")
    
    # Estimate model size (CNN + policy networks)
    estimated_model_vram = 1.5  # GB for model weights
    
    # Available VRAM for batches
    available_vram = target_vram_gb - estimated_model_vram
    available_vram_bytes = available_vram * 1024 * 1024 * 1024
    
    # Calculate maximum batch size
    # Each sample in batch needs: obs + gradients + activations
    memory_per_sample = obs_size_bytes * 3  # Conservative estimate
    max_batch_size = int(available_vram_bytes / memory_per_sample)
    
    # Round down to nearest multiple of 64 for efficiency
    optimal_batch_size = (max_batch_size // 64) * 64
    
    # Ensure minimum batch size
    optimal_batch_size = max(optimal_batch_size, 512)
    
    # Cap at reasonable maximum
    optimal_batch_size = min(optimal_batch_size, 4096)
    
    estimated_batch_vram = (optimal_batch_size * memory_per_sample) / (1024**3)
    
    print(f"   Model VRAM: {estimated_model_vram:.1f} GB")
    print(f"   Available for batches: {available_vram:.1f} GB")
    print(f"   Optimal batch size: {optimal_batch_size}")
    print(f"   Estimated batch VRAM: {estimated_batch_vram:.1f} GB")
    
    return optimal_batch_size


def check_system_resources(num_envs, n_steps, obs_shape):
    """Check if system can handle the requested configuration"""
    
    # Check CPU cores
    cpu_cores = psutil.cpu_count(logical=True)
    
    print(f"🖥️  CUDA System Resource Check:")
    print(f"   CPU Cores: {cpu_cores}")
    print(f"   Requested Envs: {num_envs}")
    print(f"   Strategy: CUDA + 82GB RAM hybrid training")
    
    # Check RAM - Buffer data stays in RAM, model on GPU
    ram_gb = psutil.virtual_memory().total / (1024**3)
    
    # Calculate buffer memory (stays in RAM)
    num_frames, height, width = obs_shape
    obs_size_mb = (num_frames * height * width) / (1024 * 1024)
    buffer_memory_gb = (n_steps * num_envs * obs_size_mb * 4) / 1024  # float32
    
    print(f"   RAM: {ram_gb:.1f} GB")
    print(f"   Buffer memory (RAM): {buffer_memory_gb:.1f} GB")
    
    if buffer_memory_gb > ram_gb * 0.6:
        print(f"❌ ERROR: Insufficient RAM for buffer!")
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
        description="Train Samurai Showdown Agent - CUDA OPTIMIZED with MASSIVE BATCHES"
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=50000000,
        help="Total timesteps to train",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=8,
        help="Number of parallel environments (8 for CUDA efficiency)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-4,
        help="Learning rate (optimized for CUDA)",
    )
    parser.add_argument(
        "--resume", type=str, default=None, help="Resume from saved model path"
    )
    parser.add_argument("--render", action="store_true", help="Enable rendering")
    parser.add_argument(
        "--use-default-state", action="store_true", help="Use default game state"
    )
    parser.add_argument(
        "--target-vram",
        type=float,
        default=10.0,
        help="Target VRAM usage in GB (default: 10.0 out of 12GB)",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=8192,
        help="Steps per rollout (optimized for CUDA)",
    )

    args = parser.parse_args()

    # Set device to CUDA directly
    device = "cuda"
    print(f"🚀 CUDA TRAINING MODE - Using {device} + 82GB RAM")
    print(f"   🎯 Target: LARGEST POSSIBLE BATCHES (2000+)")
    print(f"   💾 VRAM Target: {args.target_vram}GB / 12GB")
    print(f"   🧠 RAM: 82GB for massive buffers")

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
    print(f"✅ Observation shape: {obs_shape}")

    # System resource check
    if not check_system_resources(args.num_envs, args.n_steps, obs_shape):
        sys.exit(1)

    # Calculate optimal batch size for CUDA
    optimal_batch_size = calculate_optimal_batch_size(
        obs_shape, args.num_envs, args.target_vram
    )

    # Determine actual number of environments for rendering
    if args.render:
        if args.num_envs > 2:
            print(f"🎮 Rendering mode: Using 2 environments")
            actual_envs = 2
        else:
            actual_envs = args.num_envs
    else:
        actual_envs = args.num_envs

    n_steps = args.n_steps
    batch_size = optimal_batch_size

    # Ensure batch_size doesn't exceed total samples
    total_samples_per_rollout = n_steps * actual_envs
    if batch_size > total_samples_per_rollout:
        batch_size = total_samples_per_rollout
        print(f"⚠️ Adjusted batch_size to {batch_size} (max samples per rollout)")

    print(f"🔥 MASSIVE CUDA BATCH hyperparameters:")
    print(f"   💪 Batch size: {batch_size:,} (LARGEST POSSIBLE)")
    print(f"   📏 N-steps: {n_steps:,}")
    print(f"   🎮 Environments: {actual_envs}")
    print(f"   📊 Total samples per rollout: {total_samples_per_rollout:,}")
    print(f"   🔄 Minibatches per rollout: {total_samples_per_rollout // batch_size}")
    print(f"   🚀 Strategy: CUDA model + RAM buffer for MAXIMUM speed")

    save_dir = "trained_models_samurai_cuda"
    os.makedirs(save_dir, exist_ok=True)

    print(f"🚀 Samurai Showdown CUDA MASSIVE BATCH Training")
    print(f"   💻 Device: {device} (12GB VRAM) + 82GB RAM")
    print(f"   🎯 Strategy: Model on GPU, buffer in RAM")
    print(f"   🔥 Batch size: {batch_size:,} samples (LARGEST EVER)")
    print(f"   🎮 No jump prevention - full action space")
    print(f"   Game: {game}")

    # Create environments with SubprocVecEnv
    print(f"🔧 Creating {actual_envs} environments...")

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
        print(f"✅ {actual_envs} environments created")

    except Exception as e:
        print(f"❌ Failed to create environments: {e}")
        import traceback
        traceback.print_exc()
        return

    # Monitor VRAM before model creation
    torch.cuda.empty_cache()
    vram_before = torch.cuda.memory_allocated() / (1024**3)
    print(f"   VRAM before model: {vram_before:.2f} GB")

    # Create or load model - CUDA OPTIMIZED
    if args.resume and os.path.exists(args.resume):
        print(f"📂 Loading model from: {args.resume}")
        model = PPO.load(args.resume, env=env, device=device)

        # Update hyperparameters
        model.n_steps = n_steps
        model.batch_size = batch_size
        model.n_epochs = 4  # Fewer epochs for massive batches
        model._setup_model()

        print(f"✅ Model loaded and updated for CUDA training")
    else:
        print(f"🚀 Creating PPO for CUDA MASSIVE BATCH training")

        lr_schedule = linear_schedule(args.learning_rate, args.learning_rate * 0.1)

        model = PPO(
            "CnnPolicy",
            env,
            device=device,
            verbose=1,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=4,  # Optimized for massive batches
            gamma=0.99,
            learning_rate=lr_schedule,
            clip_range=linear_schedule(0.2, 0.02),
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            gae_lambda=0.95,
            tensorboard_log="logs_samurai_cuda",
            policy_kwargs=dict(
                normalize_images=False,
                optimizer_class=torch.optim.Adam,
                optimizer_kwargs=dict(eps=1e-5),
            ),
        )

        print(f"✅ PPO model created for CUDA training")

    # Monitor VRAM after model creation
    vram_after = torch.cuda.memory_allocated() / (1024**3)
    model_vram = vram_after - vram_before
    print(f"   VRAM after model: {vram_after:.2f} GB")
    print(f"   Model VRAM usage: {model_vram:.2f} GB")
    print(f"   Available VRAM: {12 - vram_after:.2f} GB")

    # Verify model is on correct device
    print(f"🔍 Model device verification:")
    for name, param in model.policy.named_parameters():
        print(f"   {name}: {param.device}")
        break

    # Checkpoint callback
    checkpoint_freq = 500000
    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_freq,
        save_path=save_dir,
        name_prefix=f"ppo_samurai_cuda_massive_{actual_envs}env",
    )

    print(f"💾 Checkpoint frequency: every {checkpoint_freq} steps")

    # Training
    start_time = time.time()
    print(f"🏋️ Starting CUDA MASSIVE BATCH training")
    print(f"   📊 {total_samples_per_rollout:,} samples per rollout")
    print(f"   💪 {batch_size:,} batch size (LARGEST EVER)")
    print(f"   🚀 Using CUDA for maximum speed")

    try:
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=[checkpoint_callback],
            reset_num_timesteps=not bool(args.resume),
        )

        training_time = time.time() - start_time
        print(f"🎉 CUDA training completed in {training_time/3600:.1f} hours!")

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
        print("🧹 Memory cleared")

    # Save final model
    final_model_path = os.path.join(
        save_dir, f"ppo_samurai_cuda_massive_final_{actual_envs}env.zip"
    )
    model.save(final_model_path)
    print(f"💾 Final model saved to: {final_model_path}")

    # Final VRAM report
    final_vram = torch.cuda.memory_allocated() / (1024**3)
    max_vram = torch.cuda.max_memory_allocated() / (1024**3)
    print(f"📊 Final VRAM usage: {final_vram:.2f} GB")
    print(f"📊 Peak VRAM usage: {max_vram:.2f} GB")

    print("✅ CUDA MASSIVE BATCH training complete!")
    print(f"🎯 Strategy: Used CUDA + 82GB RAM for largest possible batches")


if __name__ == "__main__":
    main()