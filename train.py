import os
import sys
import argparse
import time
import torch
import psutil

import retro
import gymnasium as gym
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv

# Import the wrapper (now includes Decision Transformer)
from wrapper import (
    SamuraiShowdownCustomWrapper,
    DecisionTransformer,
    collect_trajectories,
    train_decision_transformer,
)


def make_env_for_subprocess(game, state, rendering=False, seed=0, env_id=0):
    """Create environment function for SubprocVecEnv - all imports MUST be inside"""

    def _init():
        # Import everything inside the function for subprocess compatibility
        import retro
        import gymnasium as gym
        from stable_baselines3.common.monitor import Monitor
        from wrapper import SamuraiShowdownCustomWrapper

        # Create environment - FIXED FOR STABLE-RETRO
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
    """Check if system can handle the requested number of environments - OPTIMIZED FOR GPU"""

    # Check GPU VRAM
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"🚀 GPU TRAINING System Resource Check:")
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {gpu_memory:.1f} GB")
        print(f"   Strategy: Maximum batch size Decision Transformer")
    else:
        print(f"❌ CUDA not available! Falling back to CPU")
        return False

    # Check CPU cores
    cpu_cores = psutil.cpu_count(logical=True)
    print(f"   CPU Cores: {cpu_cores}")
    print(f"   Requested Envs: {num_envs}")

    # Check RAM - OPTIMIZED FOR 82GB SYSTEM
    ram_gb = psutil.virtual_memory().total / (1024**3)
    estimated_ram_usage = num_envs * 3.0  # ~3GB per environment with massive rollouts

    print(f"   RAM: {ram_gb:.1f} GB (MASSIVE RAM SYSTEM)")
    print(f"   Estimated RAM usage: {estimated_ram_usage:.1f} GB")
    print(f"   Available for batches: {ram_gb - estimated_ram_usage - 15:.1f} GB")

    if estimated_ram_usage > ram_gb * 0.5:
        print(f"❌ ERROR: Insufficient RAM!")
        print(f"💡 Reduce environments or rollout size")
        return False

    return True


def calculate_optimal_batch_size(obs_shape, context_length=30, vram_gb=12):
    """Calculate maximum batch size that fits in VRAM"""
    print(f"🧮 Calculating optimal batch size for {vram_gb}GB VRAM...")

    # Memory estimation per sample (in bytes)
    # States: batch_size * context_length * obs_shape * 4 bytes (float32)
    # Actions: batch_size * context_length * 4 bytes (int32)
    # Returns: batch_size * context_length * 4 bytes (float32)
    # Timesteps: batch_size * context_length * 4 bytes (int32)

    obs_memory_per_sample = (
        context_length * obs_shape[0] * obs_shape[1] * obs_shape[2] * 4
    )
    other_memory_per_sample = context_length * 4 * 3  # actions, returns, timesteps
    total_memory_per_sample = obs_memory_per_sample + other_memory_per_sample

    # Model memory (rough estimate for transformer)
    model_memory = 256 * 1024 * 1024 * 4  # ~1GB for model parameters

    # Available memory (leave 2GB buffer for CUDA operations)
    available_memory = (vram_gb - 2) * 1024 * 1024 * 1024
    memory_for_batch = available_memory - model_memory

    # Calculate max batch size
    max_batch_size = int(memory_for_batch / total_memory_per_sample)

    # Round down to nearest power of 2 for efficiency
    optimal_batch_size = 1
    while optimal_batch_size * 2 <= max_batch_size:
        optimal_batch_size *= 2

    print(f"   Memory per sample: {total_memory_per_sample / (1024**2):.1f} MB")
    print(f"   Available for batches: {memory_for_batch / (1024**3):.1f} GB")
    print(f"   Theoretical max batch: {max_batch_size}")
    print(f"   Optimal batch size: {optimal_batch_size}")

    return optimal_batch_size


def get_actual_observation_dims(game, state):
    """Get actual observation dimensions from the wrapper"""
    try:
        # Create a temporary environment to get actual dimensions - FIXED
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
        return obs_shape
    except Exception as e:
        print(f"⚠️ Could not determine observation dimensions: {e}")
        # Fallback to estimated dimensions
        return (9, 168, 240)  # 9 frames, 75% of 224x320


def main():
    parser = argparse.ArgumentParser(
        description="Train Samurai Showdown Agent - Decision Transformer (CUDA Optimized)"
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=50000000,  # Reduced for faster testing
        help="Episodes to collect for training data",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=4,  # Start with single env for testing
        help="Number of parallel environments",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=4e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--resume", type=str, default=None, help="Resume from saved model path"
    )
    parser.add_argument("--render", action="store_true", help="Enable rendering")
    parser.add_argument(
        "--use-default-state", action="store_true", help="Use default game state"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=0,  # 0 = auto-calculate optimal
        help="Batch size (0 for auto-optimal)",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=50,  # Reduced epochs for testing
        help="Training epochs",
    )
    parser.add_argument(
        "--context-length",
        type=int,
        default=30,  # Shorter context for testing
        help="Context length for Decision Transformer",
    )

    args = parser.parse_args()

    # System resource check
    if not check_system_resources(args.num_envs):
        sys.exit(1)

    # CUDA SETUP - FORCE GPU USAGE
    if not torch.cuda.is_available():
        print("❌ CUDA not available! This script requires GPU.")
        sys.exit(1)

    device = "cuda"
    print(f"🚀 CUDA TRAINING MODE - Using {torch.cuda.get_device_name(0)}")

    # Optimize CUDA settings
    torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes
    torch.backends.cudnn.deterministic = False  # Allow non-deterministic for speed

    # Clear GPU cache
    torch.cuda.empty_cache()

    print(f"✅ Device: {device}")
    print(f"✅ CUDA optimizations enabled")

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
        test_obs, _ = test_env.reset()
        print(f"✅ Basic environment test passed - obs shape: {test_obs.shape}")
        test_env.close()
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

    # Calculate optimal batch size if not specified
    if args.batch_size == 0:
        optimal_batch_size = calculate_optimal_batch_size(
            obs_shape, args.context_length, 12
        )
        args.batch_size = max(8, optimal_batch_size)  # Minimum batch size of 8

    print(f"🎯 Using batch size: {args.batch_size}")

    save_dir = "trained_models"
    os.makedirs(save_dir, exist_ok=True)

    print(f"💾 Save directory: {save_dir}")

    # Create training environment - SIMPLIFIED FOR TESTING
    print("🏗️ Creating training environment...")

    # Always use single environment for initial testing
    env = retro.make(
        game=game,
        state=state,
        use_restricted_actions=retro.Actions.FILTERED,
        obs_type=retro.Observations.IMAGE,
        render_mode="human" if args.render else None,
    )
    env = SamuraiShowdownCustomWrapper(
        env,
        reset_round=True,
        rendering=args.render,
        max_episode_steps=15000,  # Reasonable episode length
    )
    env = Monitor(env)

    print(f"✅ Environment created successfully")

    # Test environment
    print("🧪 Testing environment...")
    try:
        obs, info = env.reset()
        print(f"✅ Reset successful - obs shape: {obs.shape}")

        # Test a few steps
        for i in range(5):
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)
            print(f"   Step {i+1}: action={action}, reward={reward}, done={done}")
            if done or truncated:
                obs, info = env.reset()
                break

    except Exception as e:
        print(f"❌ Environment test failed: {e}")
        return

    # Create Decision Transformer model
    print("🧠 Creating Decision Transformer model...")
    action_dim = env.action_space.n

    model = DecisionTransformer(
        observation_shape=obs_shape,
        action_dim=action_dim,
        hidden_size=256,  # Smaller for testing
        n_layer=4,  # Fewer layers for testing
        n_head=4,
        max_ep_len=2000,
    )

    print(
        f"✅ Model created with {sum(p.numel() for p in model.parameters())} parameters"
    )

    # Collect trajectories
    print(f"📊 Collecting {args.total_timesteps} episodes of training data...")

    trajectories = collect_trajectories(env, args.total_timesteps)

    print(f"✅ Collected {len(trajectories)} trajectories")

    # Filter trajectories with some reward signal
    good_trajectories = [
        t for t in trajectories if len(t["rewards"]) > 10 and sum(t["rewards"]) != 0
    ]
    print(f"✅ Found {len(good_trajectories)} trajectories with reward signal")

    if len(good_trajectories) < 5:
        print("⚠️ Using all trajectories due to limited good data")
        good_trajectories = trajectories

    # Only proceed if we have enough data
    if len(good_trajectories) < 2:
        print("❌ Not enough trajectories for training")
        return

    # Train Decision Transformer
    print(f"🏋️ Training Decision Transformer...")
    print(f"   Epochs: {args.n_steps}")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Learning rate: {args.learning_rate}")
    print(f"   Context length: {args.context_length}")

    trained_model = train_decision_transformer(
        model=model,
        trajectories=good_trajectories,
        epochs=args.n_steps,
        batch_size=args.batch_size,
        lr=args.learning_rate,
        device=device,
        context_length=args.context_length,
    )

    # Save model
    model_path = os.path.join(save_dir, "decision_transformer_samurai.pth")
    trained_model.save(model_path)

    print(f"🎉 Training completed!")
    print(f"💾 Model saved to: {model_path}")

    # Print GPU memory usage
    if torch.cuda.is_available():
        print(f"🔧 GPU Memory Usage:")
        print(f"   Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"   Cached: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")

    # Close environments
    env.close()


if __name__ == "__main__":
    main()
