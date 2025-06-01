import os
import sys
import argparse
import time

import retro
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback

# Import the wrapper
from wrapper import SamuraiShowdownCustomWrapper


def create_single_env(game, state, rendering=False):
    """Create a single environment (no multiprocessing)"""

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
    return env


def main():
    parser = argparse.ArgumentParser(
        description="Simple Samurai Showdown Training Test"
    )
    parser.add_argument(
        "--total-timesteps", type=int, default=10000, help="Total timesteps to train"
    )
    parser.add_argument("--render", action="store_true", help="Enable rendering")
    parser.add_argument(
        "--use-default-state", action="store_true", help="Use default game state"
    )

    args = parser.parse_args()

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
            state = "samurai.state"
            print(f"🎮 Using samurai.state file")
        else:
            print(f"❌ samurai.state not found, using default state")
            state = None

    # Create single environment (no multiprocessing)
    print(f"🔧 Creating single environment...")
    try:
        env = create_single_env(game, state=state, rendering=args.render)
        print("✅ Environment created successfully")

        # Test reset
        obs = env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
        print(f"✅ Reset successful, obs shape: {obs.shape}")

    except Exception as e:
        print(f"❌ Failed to create environment: {e}")
        import traceback

        traceback.print_exc()
        return

    # Create simple PPO model
    print("🧠 Creating PPO model...")
    try:
        model = PPO(
            "CnnPolicy",
            env,
            device="cpu",  # Use CPU for simplicity
            verbose=1,
            n_steps=64,  # Small for testing
            batch_size=64,
            n_epochs=4,
            gamma=0.99,
            learning_rate=3e-4,
            tensorboard_log="logs_samurai_test",
        )
        print("✅ Model created successfully")
    except Exception as e:
        print(f"❌ Failed to create model: {e}")
        import traceback

        traceback.print_exc()
        return

    # Short training test
    print(f"🏋️ Starting training for {args.total_timesteps} timesteps...")
    try:
        model.learn(total_timesteps=args.total_timesteps)
        print(f"🎉 Training completed successfully!")

        # Save model
        model.save("test_samurai_model.zip")
        print(f"💾 Model saved as test_samurai_model.zip")

    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback

        traceback.print_exc()
    finally:
        env.close()

    print("✅ Test complete!")


if __name__ == "__main__":
    main()
