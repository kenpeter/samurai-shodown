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
            max_episode_steps=20000,  # MASSIVE episodes for huge batches
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

    print(f"🖥️  CPU TRAINING System Resource Check:")
    print(f"   CPU Cores: {cpu_cores}")
    print(f"   Requested Envs: {num_envs}")
    print(f"   Strategy: MASSIVE RAM + CPU training")

    # Check RAM - OPTIMIZED FOR 82GB SYSTEM
    ram_gb = psutil.virtual_memory().total / (1024**3)
    estimated_ram_usage = num_envs * 2.0  # ~2GB per environment with massive rollouts

    print(f"   RAM: {ram_gb:.1f} GB (MASSIVE RAM SYSTEM)")
    print(f"   Estimated RAM usage: {estimated_ram_usage:.1f} GB")
    print(f"   Available for batches: {ram_gb - estimated_ram_usage - 10:.1f} GB")

    if estimated_ram_usage > ram_gb * 0.4:
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
        description="Train Samurai Showdown Agent - MASSIVE CPU BATCH Training"
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=50000000,
        help="Total timesteps to train (massive for CPU training)",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=2,
        help="Number of parallel environments (2 for massive batches)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate (lower for massive batches)",
    )
    parser.add_argument(
        "--resume", type=str, default=None, help="Resume from saved model path"
    )
    parser.add_argument("--render", action="store_true", help="Enable rendering")
    parser.add_argument(
        "--use-default-state", action="store_true", help="Use default game state"
    )
    # SAFE large batch sizes using 82GB RAM (accounting for SB3 float32 conversion)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8192,  # LARGE but safe batch size
        help="Batch size (default: 8192 - large and safe)",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=12288,  # LARGE but safe rollouts (within 177GB limit)
        help="Steps per rollout (default: 12288 - large and safe)",
    )

    args = parser.parse_args()

    # System resource check
    if not check_system_resources(args.num_envs):
        sys.exit(1)

    # CPU TRAINING SETUP
    print("💻 CPU TRAINING MODE - Using 82GB RAM for MASSIVE batches")

    # Force CPU usage
    device = "cpu"
    print(f"✅ Device: {device} (82GB RAM)")

    # Optimize for CPU training
    torch.set_num_threads(psutil.cpu_count(logical=True))
    print(f"✅ CPU threads: {torch.get_num_threads()}")

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
        if args.num_envs > 2:
            print(f"🎮 Rendering mode: Using 2 environments with rendering")
            actual_envs = 2
        else:
            print(f"🎮 Rendering mode: All {args.num_envs} environments will render")
            actual_envs = args.num_envs
    else:
        actual_envs = args.num_envs

    # MASSIVE BATCH HYPERPARAMETERS for 82GB RAM
    print(f"🧠 MASSIVE CPU BATCH TRAINING - 82GB RAM Power!")

    batch_size = args.batch_size
    n_steps = args.n_steps

    # Ensure batch_size is reasonable for the number of environments
    total_samples_per_rollout = n_steps * actual_envs
    if batch_size > total_samples_per_rollout:
        batch_size = total_samples_per_rollout
        print(f"⚠️ Adjusted batch_size to {batch_size} (max samples per rollout)")

    # Calculate buffer memory usage - ALL IN RAM!
    buffer_memory_gb = (n_steps * actual_envs * obs_size_mb) / 1024

    print(f"📈 LARGE CPU BATCH hyperparameters for {actual_envs} environments:")
    print(f"   💪 Batch size: {batch_size:,} (LARGE - using 82GB RAM efficiently)")
    print(f"   📏 N-steps: {n_steps:,} (LARGE rollouts)")
    print(f"   📊 Total samples per rollout: {total_samples_per_rollout:,}")
    print(f"   💾 Buffer memory: {buffer_memory_gb:.2f} GB (all in RAM)")
    print(f"   🔄 Minibatches per rollout: {total_samples_per_rollout // batch_size}")
    print(f"   🎯 Strategy: Efficient use of 82GB RAM without overflow")

    # RAM safety check - Account for float32 conversion
    # SB3 converts uint8 observations to float32, multiplying memory by 4x
    actual_buffer_memory = buffer_memory_gb * 4  # float32 vs uint8

    if actual_buffer_memory > 50:
        print(f"❌ ERROR: SB3 buffer memory too large ({actual_buffer_memory:.2f} GB)!")
        print(f"   SB3 converts uint8 to float32 (4x memory usage)")
        max_safe_steps = int(40 * 1024 / (actual_envs * obs_size_mb * 4))
        print(f"💡 Reduce n_steps to {max_safe_steps} or fewer")
        sys.exit(1)

    # RAM allocation breakdown - REALISTIC for SB3
    model_ram = 2.0  # Model in RAM
    actual_buffer_ram = buffer_memory_gb * 4  # SB3 uses float32 (4x memory)
    system_overhead = 10.0  # OS and other processes

    total_estimated_ram = model_ram + actual_buffer_ram + system_overhead

    print(f"📊 REALISTIC RAM allocation for CPU training:")
    print(f"   🧠 Model: {model_ram:.1f} GB")
    print(f"   📊 Buffer data (uint8): {buffer_memory_gb:.2f} GB")
    print(f"   🔄 SB3 buffer (float32): {actual_buffer_ram:.2f} GB (4x conversion)")
    print(f"   🖥️  System overhead: {system_overhead:.1f} GB")
    print(f"   📈 Total estimated: {total_estimated_ram:.2f} GB")
    print(
        f"   💪 RAM usage: {total_estimated_ram:.1f}GB / 82GB = {total_estimated_ram/82*100:.1f}%"
    )

    if total_estimated_ram > 75:
        print("⚠️ Warning: Very high RAM usage - monitor system performance")
    else:
        print("✅ Excellent RAM usage - plenty of headroom")

    save_dir = "trained_models_samurai_cpu"
    os.makedirs(save_dir, exist_ok=True)

    print(f"🚀 Samurai Showdown LARGE CPU BATCH Training")
    print(f"   💻 Device: CPU with 82GB RAM")
    print(f"   🎯 Strategy: Large batches without GPU memory limits")
    print(f"   🔥 Batch size: {batch_size:,} samples (LARGE)")
    print(f"   🎮 Rewards: Win/Loss only, 3-sec jump cooldown")
    print(f"   Game: {game}")
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

    # Create or load model - MASSIVE CPU BATCH OPTIMIZED
    if args.resume and os.path.exists(args.resume):
        print(f"📂 Loading model from: {args.resume}")
        model = PPO.load(args.resume, env=env, device="cpu")

        # Update hyperparameters for massive batch training
        model.n_steps = n_steps
        model.batch_size = batch_size
        model.n_epochs = 8  # More epochs for massive batches on CPU
        model._setup_model()

        print(f"✅ Model loaded and updated for MASSIVE CPU batch training")
    else:
        print(f"💻 Creating PPO for MASSIVE CPU BATCH training (82GB RAM)")

        # Check RAM before creating model
        ram_before = psutil.virtual_memory().used / 1e9
        print(f"   RAM before model creation: {ram_before:.2f} GB")

        lr_schedule = linear_schedule(args.learning_rate, args.learning_rate * 0.1)

        model = PPO(
            "CnnPolicy",
            env,
            device="cpu",  # CPU training
            verbose=1,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=8,  # More epochs for CPU training
            gamma=0.995,  # Higher gamma for longer episodes
            learning_rate=lr_schedule,
            clip_range=linear_schedule(
                0.2, 0.02
            ),  # Tighter clipping for massive batches
            ent_coef=0.005,  # Lower entropy for focused learning
            vf_coef=0.5,
            max_grad_norm=0.5,
            gae_lambda=0.95,
            tensorboard_log="logs_samurai_cpu",
            policy_kwargs=dict(
                normalize_images=False,
                optimizer_class=torch.optim.AdamW,  # AdamW for CPU training
                optimizer_kwargs=dict(
                    eps=1e-7,
                    weight_decay=1e-5,  # Weight decay for stability
                ),
            ),
        )

        # Check RAM after creating model
        ram_after = psutil.virtual_memory().used / 1e9
        model_memory = ram_after - ram_before
        print(f"   RAM after model creation: {ram_after:.2f} GB")
        print(f"   Model uses: {model_memory:.2f} GB")

        available_ram = 82 - ram_after - 5  # 5GB safety buffer
        print(f"   Available for training: {available_ram:.2f} GB")

        print(f"✅ PPO model created for MASSIVE CPU batch training")

    # Model device verification
    print(f"🔍 Model device verification:")
    for name, param in model.policy.named_parameters():
        print(f"   {name}: {param.device}")
        break

    # Checkpoint callback - less frequent for massive batch training
    checkpoint_freq = 300000
    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_freq,
        save_path=save_dir,
        name_prefix=f"ppo_samurai_cpu_massive_{actual_envs}env",
    )

    print(f"💾 Checkpoint frequency: every {checkpoint_freq} steps")

    # Training
    start_time = time.time()
    print(f"🏋️ Starting MASSIVE CPU BATCH training with {actual_envs} environments")
    print(f"   📊 {total_samples_per_rollout:,} samples per rollout")
    print(f"   💪 {batch_size:,} batch size (MASSIVE)")
    print(f"   💻 Using CPU with 82GB RAM")
    print("💡 Monitor RAM usage with: htop")

    try:
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=[checkpoint_callback],
            reset_num_timesteps=not bool(args.resume),
        )

        training_time = time.time() - start_time
        print(
            f"🎉 MASSIVE CPU BATCH training completed in {training_time/3600:.1f} hours!"
        )

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
        print("🧹 Memory cleared")

    # Save final model
    final_model_path = os.path.join(
        save_dir, f"ppo_samurai_cpu_massive_final_{actual_envs}env.zip"
    )
    model.save(final_model_path)
    print(f"💾 Final model saved to: {final_model_path}")

    print("✅ MASSIVE CPU BATCH training complete!")
    print(f"🎯 Strategy: Used 82GB RAM for massive batch sizes impossible on GPU")


if __name__ == "__main__":
    main()
