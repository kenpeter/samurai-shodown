import os
import argparse
import time
import numpy as np

import retro
import gymnasium as gym
from stable_baselines3 import PPO

# Import the wrapper
from wrapper import StreetFighterCustomWrapper


class FilteredActionWrapper(gym.Wrapper):
    """Wrapper to filter out START and MODE buttons to prevent cheating"""

    def __init__(self, env):
        super(FilteredActionWrapper, self).__init__(env)
        # Street Fighter II MultiBinary actions:
        # [B, Y, SELECT, START, UP, DOWN, LEFT, RIGHT, A, X, L, R]
        # We want to disable START (index 3) and SELECT (index 2)
        self.disabled_buttons = [2, 3]  # SELECT and START

    def step(self, action):
        # Filter out disabled buttons by setting them to 0
        if hasattr(action, "__len__") and len(action) >= 4:
            filtered_action = action.copy() if hasattr(action, "copy") else list(action)
            for button_idx in self.disabled_buttons:
                if button_idx < len(filtered_action):
                    filtered_action[button_idx] = 0
            return self.env.step(filtered_action)
        return self.env.step(action)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


def create_eval_env(game, state):
    """Create evaluation environment aligned with training setup"""
    # Handle state file path consistently with train.py
    if os.path.isfile(state):
        state_file = os.path.abspath(state)
        print(f"Using custom state file: {state_file}")
    else:
        if state.endswith(".state"):
            state_file = state[:-6]  # Remove .state extension for built-in states
            print(f"Using built-in state: {state_file}")
        else:
            state_file = state
            print(f"Using state: {state_file}")

    # Create retro environment with rendering enabled
    env = retro.make(
        game=game,
        state=state_file,
        use_restricted_actions=retro.Actions.FILTERED,
        obs_type=retro.Observations.IMAGE,
        render_mode="human",  # Enable rendering for human observation
    )

    # Apply custom wrapper with same settings as training
    # IMPORTANT: Must match training wrapper configuration exactly!
    env = StreetFighterCustomWrapper(
        env,
        reset_round=True,
        rendering=True,
        max_episode_steps=5000,  # ADDED: Match training configuration
    )

    # Print observation space for debugging
    print(f"🔍 Evaluation environment observation space: {env.observation_space.shape}")

    return env


def convert_observation_format(obs, target_shape):
    """Convert observation between channels-first and channels-last formats"""
    current_shape = obs.shape

    if current_shape == target_shape:
        return obs

    # Check if we need to transpose from (H, W, C) to (C, H, W)
    if len(current_shape) == 3 and len(target_shape) == 3:
        if (current_shape[0], current_shape[1], current_shape[2]) == (
            target_shape[1],
            target_shape[2],
            target_shape[0],
        ):
            # Transpose from (H, W, C) to (C, H, W)
            # print(f"🔄 Converting observation from {current_shape} to {target_shape}")
            return np.transpose(obs, (2, 0, 1))
        elif (current_shape[0], current_shape[1], current_shape[2]) == (
            target_shape[2],
            target_shape[0],
            target_shape[1],
        ):
            # Transpose from (C, H, W) to (H, W, C)
            print(f"🔄 Converting observation from {current_shape} to {target_shape}")
            return np.transpose(obs, (1, 2, 0))

    print(
        f"⚠️  Warning: Cannot convert observation shape {current_shape} to {target_shape}"
    )
    return obs


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate trained Street Fighter II Agent"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="trained_models/ppo_sf2_25999220_steps.zip",
        help="Path to the trained model",
    )
    parser.add_argument(
        "--state-file",
        type=str,
        default="ken_bison_12.state",
        help="State file to use",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=3,
        help="Number of episodes to run",
    )
    parser.add_argument(
        "--use-built-in-state",
        action="store_true",
        help="Use built-in state (removes .state extension)",
    )

    args = parser.parse_args()

    # Check if model exists
    if not os.path.exists(args.model_path):
        print(f"❌ Error: Model file not found at {args.model_path}")
        print("Available models in trained_models/:")
        if os.path.exists("trained_models"):
            for f in os.listdir("trained_models"):
                if f.endswith(".zip"):
                    print(f"   - {f}")
        return

    game = "StreetFighterIISpecialChampionEdition-Genesis"

    # Handle state file properly
    if args.use_built_in_state:
        # Use built-in state (remove .state extension if present)
        state_file = (
            args.state_file[:-6]
            if args.state_file.endswith(".state")
            else args.state_file
        )
    else:
        # Use custom state file
        state_file = args.state_file

    print(f"🤖 Loading model from: {args.model_path}")
    print(f"🎮 Using state file: {state_file}")
    print(f"🔄 Will run {args.episodes} episodes")
    print("⚡ Running at 60 FPS for smooth gameplay")
    print("\n🔧 FIXED: Automatic observation format conversion enabled!")
    print("\nPress Ctrl+C to quit at any time")
    print("=" * 60)

    # Create evaluation environment
    try:
        env = create_eval_env(game, state_file)
        print("✅ Environment created successfully!")
    except Exception as e:
        print(f"❌ Error creating environment: {e}")
        print(
            "\n💡 Try using --use-built-in-state flag if you're using a built-in state"
        )
        return

    # Load the trained model
    try:
        print("🧠 Loading model...")
        model = PPO.load(args.model_path, device="cuda")
        print("✅ Model loaded successfully!")

        # Check observation space compatibility
        model_shape = model.observation_space.shape
        env_shape = env.observation_space.shape

        print(f"🔍 Model expects observation shape: {model_shape}")
        print(f"🔍 Environment provides shape: {env_shape}")

        if model_shape != env_shape:
            print("🔧 Observation shapes differ - will auto-convert during evaluation")
        else:
            print("✅ Observation shapes match perfectly!")

    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    try:
        total_wins = 0
        total_episodes = 0

        for episode in range(args.episodes):
            print(f"\n🥊 --- Episode {episode + 1}/{args.episodes} ---")

            obs, info = env.reset()
            episode_reward = 0
            step_count = 0

            print("🎬 Starting new match... Watch the game window!")

            while True:
                # FIXED: Convert observation format if needed
                obs_for_model = convert_observation_format(
                    obs, model.observation_space.shape
                )

                # Get action from the trained model
                action, _states = model.predict(obs_for_model, deterministic=False)

                # Take step in environment
                obs, reward, terminated, truncated, info = env.step(action)

                episode_reward += reward
                step_count += 1

                # Maintain 60 FPS for smooth viewing
                time.sleep(0.0167)  # 60 FPS timing

                # Check if episode is done
                if terminated or truncated:
                    break

                # Optional: Add some info display every 5 seconds at 60fps
                if step_count % 300 == 0:  # Every 5 seconds at 60fps (300 steps)
                    player_hp = info.get("agent_hp", "?")
                    enemy_hp = info.get("enemy_hp", "?")
                    print(
                        f"   Step {step_count}: Player HP: {player_hp}, Enemy HP: {enemy_hp}"
                    )

            total_episodes += 1
            print(f"🏁 Episode {episode + 1} finished!")
            print(f"   Total reward: {episode_reward:.1f}")
            print(f"   Steps taken: {step_count}")

            # Get final health values
            player_hp = info.get("agent_hp", "?")
            enemy_hp = info.get("enemy_hp", "?")
            print(f"   Final - Player HP: {player_hp}, Enemy HP: {enemy_hp}")

            # Determine winner
            if player_hp <= 0:
                print("   🔴 AI Lost this round")
            elif enemy_hp <= 0:
                print("   🟢 AI Won this round")
                total_wins += 1
            else:
                print("   ⚪ Round ended without knockout")

            # Pause between episodes
            if episode < args.episodes - 1:
                print("\n⏳ Waiting 3 seconds before next episode...")
                time.sleep(3)

        # Final statistics
        print(f"\n📊 Final Results:")
        print(f"   Wins: {total_wins}/{total_episodes}")
        print(f"   Win Rate: {(total_wins/total_episodes)*100:.1f}%")

    except KeyboardInterrupt:
        print("\n\n⏹️  Evaluation interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during evaluation: {e}")
        print("💡 Check that your model and wrapper configurations are compatible")
        import traceback

        traceback.print_exc()
    finally:
        env.close()
        print("\n✅ Evaluation complete!")


if __name__ == "__main__":
    main()
