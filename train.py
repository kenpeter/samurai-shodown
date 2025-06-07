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
from wrapper import SamuraiShowdownCustomWrapper, DecisionTransformer, collect_trajectories, train_decision_transformer


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
    print(f"   Strategy: Decision Transformer training")

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
        description="Train Samurai Showdown Agent - Decision Transformer"
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=200,  # Now episodes to collect
        help="Episodes to collect for training data",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=1,  # DT uses single env
        help="Number of parallel environments",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate",
    )
    parser.add_argument("--resume", type=str, default=None, help="Resume from saved model path")
    parser.add_argument("--render", action="store_true", help="Enable rendering")
    parser.add_argument(
        "--use-default-state", action="store_true", help="Use default game state"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=100,  # Now training epochs
        help="Training epochs",
    )

    args = parser.parse_args()

    # System resource check
    if not check_system_resources(args.num_envs):
        sys.exit(1)

    # CPU TRAINING SETUP
    print("💻 CPU TRAINING MODE - Using 82GB RAM for Decision Transformer")

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

    save_dir = "traine