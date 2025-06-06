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

    print(
        f"🔧 create_single_env called with state: type={type(state)}, value='{state}'"
    )

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


def navigate_start_menu(env, max_steps=100):
    """Navigate past the start menu by pressing START button"""
    print("🎮 Navigating past start menu...")

    # Find START button index
    start_button_idx = None
    if "START" in env.buttons:
        start_button_idx = env.buttons.index("START")

    for step in range(max_steps):
        # Create action array
        action = [False] * len(env.buttons)

        # Press START button every few frames
        if start_button_idx is not None and step % 10 == 0:
            action[start_button_idx] = True

        obs, reward, done, truncated, info = env.step(action)

        # Check if we've moved past the menu (look for some game state change)
        if step % 20 == 0:
            print(f"   Step {step}: Still navigating...")

        # You can add conditions here to detect when you're in an actual match
        # For now, just run for a bit to get past menus

        if step > 50:  # Assume we're past menus after 50 steps
            print("✅ Likely past start menu")
            break

    return obs


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
            state = os.path.abspath("samurai.state")  # Use absolute path
            print(f"🎮 Using samurai.state file")
            print(f"🔍 Absolute path: {state}")
            print(f"🔍 File exists: {os.path.exists(state)}")
            print(f"🔍 File size: {os.path.getsize(state)} bytes")
        else:
            print(f"❌ samurai.state not found in current directory")
            print("🔍 Current directory .state files:")
            for f in os.listdir("."):
                if f.endswith(".state"):
                    print(f"   - {f}")
            print("💡 Using default state instead")
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

        # Navigate past start menu if using default state
        if args.use_default_state:
            obs = navigate_start_menu(env)

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
