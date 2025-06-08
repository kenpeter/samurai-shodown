import os
import argparse
import time
import numpy as np

import retro
import gymnasium as gym
from stable_baselines3 import PPO

# Import the wrapper
from wrapper import SamuraiShowdownCustomWrapper


def create_eval_env(game, state):
    """Create evaluation environment aligned with CUDA training setup"""
    # Handle state file path
    if state and os.path.isfile(state):
        state_file = os.path.abspath(state)
        print(f"Using custom state file: {state_file}")
    else:
        state_file = state
        print(f"Using state: {state_file if state_file else 'default'}")

    # Create retro environment with rendering enabled
    env = retro.make(
        game=game,
        state=state_file,
        use_restricted_actions=retro.Actions.FILTERED,
        obs_type=retro.Observations.IMAGE,
        render_mode="human",  # Enable rendering for human observation
    )

    # Apply custom wrapper with same settings as CUDA training
    # IMPORTANT: Must match training wrapper configuration exactly!
    env = SamuraiShowdownCustomWrapper(
        env,
        reset_round=True,
        rendering=True,
        max_episode_steps=15000,  # Match CUDA training configuration
    )

    # Print observation space for debugging
    print(f"🔍 Evaluation environment observation space: {env.observation_space.shape}")

    return env


def convert_observation_format(obs, target_shape):
    """Convert observation between different formats if needed"""
    current_shape = obs.shape

    if current_shape == target_shape:
        return obs

    # Handle shape mismatches
    if len(current_shape) == 3 and len(target_shape) == 3:
        # Check if it's just a dimension ordering issue
        if (current_shape[0], current_shape[1], current_shape[2]) == (
            target_shape[1],
            target_shape[2],
            target_shape[0],
        ):
            # Transpose from (H, W, C) to (C, H, W)
            print(f"🔄 Converting observation from {current_shape} to {target_shape}")
            return np.transpose(obs, (2, 0, 1))
        elif (current_shape[0], current_shape[1], current_shape[2]) == (
            target_shape[2],
            target_shape[0],
            target_shape[1],
        ):
            # Transpose from (C, H, W) to (H, W, C)
            print(f"🔄 Converting observation from {current_shape} to {target_shape}")
            return np.transpose(obs, (1, 2, 0))

    # If shapes are completely different, try to resize
    if len(current_shape) == 3 and len(target_shape) == 3:
        if current_shape[0] == target_shape[0]:  # Same number of channels/frames
            print(f"🔄 Resizing observation from {current_shape} to {target_shape}")
            # Simple nearest neighbor resize for each frame
            resized_frames = []
            for i in range(current_shape[0]):
                frame = obs[i]
                # Simple resize using array indexing
                h_ratio = frame.shape[0] / target_shape[1]
                w_ratio = frame.shape[1] / target_shape[2]
                h_indices = (np.arange(target_shape[1]) * h_ratio).astype(int)
                w_indices = (np.arange(target_shape[2]) * w_ratio).astype(int)
                resized_frame = frame[np.ix_(h_indices, w_indices)]
                resized_frames.append(resized_frame)
            return np.stack(resized_frames, axis=0)

    print(
        f"⚠️  Warning: Cannot convert observation shape {current_shape} to {target_shape}"
    )
    return obs


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate trained Samurai Showdown Agent - CUDA Version"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="trained_models_samurai_cuda/ppo_samurai_cuda_massive_4env_final.zip",
        help="Path to the trained CUDA model",
    )
    parser.add_argument(
        "--state-file",
        type=str,
        default="samurai.state",
        help="State file to use (default: samurai.state)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=5,
        help="Number of episodes to run",
    )
    parser.add_argument(
        "--use-default-state",
        action="store_true",
        help="Use default game state (ignore state file)",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use deterministic actions (less random)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=60,
        help="Target FPS for rendering (default: 60)",
    )
    parser.add_argument(
        "--force-cpu",
        action="store_true",
        help="Force CPU evaluation even if CUDA is available",
    )

    args = parser.parse_args()

    # Check if model exists
    if not os.path.exists(args.model_path):
        print(f"❌ Error: Model file not found at {args.model_path}")
        print("Available models:")

        # Check both CUDA and CPU model directories
        for dir_name in [
            "trained_models_samurai_cuda",
            "trained_models_samurai_cpu",
            "trained_models_samurai",
        ]:
            if os.path.exists(dir_name):
                print(f"   {dir_name}/:")
                for f in os.listdir(dir_name):
                    if f.endswith(".zip"):
                        print(f"     - {f}")
        return

    game = "SamuraiShodown-Genesis"

    # Handle state file properly
    if args.use_default_state:
        state_file = None
        print("Using default game state")
    else:
        state_file = args.state_file

    print(f"🚀 CUDA Model Evaluation")
    print(f"🤖 Loading model from: {args.model_path}")
    print(f"🎮 Using state file: {state_file if state_file else 'default'}")
    print(f"🔄 Will run {args.episodes} episodes")
    print(f"⚡ Running at {args.fps} FPS for smooth gameplay")
    print(f"🎯 Deterministic actions: {'Yes' if args.deterministic else 'No'}")
    print(f"✅ Full action space enabled (including jumps)")
    print("\n🔧 Automatic observation format conversion enabled!")
    print("\nPress Ctrl+C to quit at any time")
    print("=" * 60)

    # Create evaluation environment
    try:
        env = create_eval_env(game, state_file)
        print("✅ Environment created successfully!")
    except Exception as e:
        print(f"❌ Error creating environment: {e}")
        print("\n💡 Troubleshooting:")
        print("   - Check if samurai.state file exists")
        print("   - Try using --use-default-state flag")
        print("   - Ensure SamuraiShodown-Genesis ROM is installed")
        return

    # Load the trained model with device selection
    try:
        print("🧠 Loading model...")

        # Determine device
        if args.force_cpu:
            device = "cpu"
            print("💻 Forced CPU evaluation")
        else:
            try:
                import torch

                if torch.cuda.is_available():
                    device = "cuda"
                    print("🚀 Using CUDA for evaluation")
                else:
                    device = "cpu"
                    print("💻 CUDA not available, using CPU")
            except ImportError:
                device = "cpu"
                print("💻 PyTorch not available, using CPU")

        # Load model on selected device
        try:
            model = PPO.load(args.model_path, device=device)
            print(f"✅ Model loaded on {device.upper()}!")
        except Exception as e:
            print(f"⚠️ Failed to load on {device}, trying CPU...")
            model = PPO.load(args.model_path, device="cpu")
            device = "cpu"
            print("✅ Model loaded on CPU!")

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
        print("\n💡 Common issues:")
        print("   - Model was trained with different wrapper settings")
        print("   - GPU/CPU compatibility issues")
        print("   - Model file is corrupted")
        print("   - Try using --force-cpu flag")
        return

    # Calculate frame timing
    frame_time = 1.0 / args.fps

    try:
        total_wins = 0
        total_losses = 0
        total_episodes = 0
        total_reward = 0
        total_steps = 0

        # Track action usage
        action_counts = {}

        for episode in range(args.episodes):
            print(f"\n⚔️  --- Episode {episode + 1}/{args.episodes} ---")

            obs, info = env.reset()
            episode_reward = 0
            step_count = 0
            episode_start_time = time.time()
            episode_actions = []

            print("🎬 Starting new match... Watch the game window!")
            print("🎮 AI can now use full action space (including jumps)!")

            while True:
                step_start_time = time.time()

                # Convert observation format if needed
                obs_for_model = convert_observation_format(
                    obs, model.observation_space.shape
                )

                # Get action from the trained model
                action, _states = model.predict(
                    obs_for_model, deterministic=args.deterministic
                )

                # Track action usage
                action_key = str(action) if hasattr(action, "__iter__") else int(action)
                action_counts[action_key] = action_counts.get(action_key, 0) + 1
                episode_actions.append(action_key)

                # Take step in environment
                obs, reward, terminated, truncated, info = env.step(action)

                episode_reward += reward
                step_count += 1

                # Frame rate limiting
                elapsed = time.time() - step_start_time
                if elapsed < frame_time:
                    time.sleep(frame_time - elapsed)

                # Check if episode is done
                if terminated or truncated:
                    break

                # Optional: Add some info display every 5 seconds
                if step_count % (args.fps * 5) == 0:  # Every 5 seconds
                    player_hp = info.get("health", "?")
                    enemy_hp = info.get("enemy_health", "?")
                    print(
                        f"   Step {step_count}: Player HP: {player_hp}, Enemy HP: {enemy_hp}"
                    )

            # Episode finished
            episode_time = time.time() - episode_start_time
            total_episodes += 1
            total_reward += episode_reward
            total_steps += step_count

            print(f"🏁 Episode {episode + 1} finished!")
            print(f"   Total reward: {episode_reward:.1f}")
            print(f"   Steps taken: {step_count}")
            print(f"   Episode duration: {episode_time:.1f}s")

            # Get final health values
            player_hp = info.get("health", 0)
            enemy_hp = info.get("enemy_health", 0)
            print(f"   Final - Player HP: {player_hp}, Enemy HP: {enemy_hp}")

            # Determine winner
            if player_hp <= 0 and enemy_hp > 0:
                print("   🔴 AI Lost this round")
                total_losses += 1
            elif enemy_hp <= 0 and player_hp > 0:
                print("   🟢 AI Won this round")
                total_wins += 1
            else:
                print("   ⚪ Round ended without clear winner")

            # Show action diversity for this episode
            unique_actions = len(set(episode_actions))
            print(f"   🎮 Action diversity: {unique_actions} unique actions used")

            # Pause between episodes
            if episode < args.episodes - 1:
                print("\n⏳ Waiting 3 seconds before next episode...")
                time.sleep(3)

        # Final statistics
        print(f"\n📊 Final Results:")
        print(f"   Episodes: {total_episodes}")
        print(f"   Wins: {total_wins}")
        print(f"   Losses: {total_losses}")
        print(f"   Draws/Timeouts: {total_episodes - total_wins - total_losses}")
        if total_episodes > 0:
            win_rate = (total_wins / total_episodes) * 100
            print(f"   Win Rate: {win_rate:.1f}%")
            avg_reward = total_reward / total_episodes
            avg_steps = total_steps / total_episodes
            print(f"   Average Reward: {avg_reward:.1f}")
            print(f"   Average Steps: {avg_steps:.0f}")

        # Action analysis
        print(f"\n🎮 Action Usage Analysis:")
        print(f"   Total unique actions used: {len(action_counts)}")
        print(f"   Total actions taken: {sum(action_counts.values())}")

        # Show top 5 most used actions
        sorted_actions = sorted(action_counts.items(), key=lambda x: x[1], reverse=True)
        print(f"   Top 5 most used actions:")
        for i, (action, count) in enumerate(sorted_actions[:5]):
            percentage = (count / sum(action_counts.values())) * 100
            print(f"     {i+1}. Action {action}: {count} times ({percentage:.1f}%)")

        # Performance assessment
        if total_episodes > 0:
            if win_rate >= 70:
                print("🏆 Excellent performance!")
            elif win_rate >= 50:
                print("👍 Good performance!")
            elif win_rate >= 30:
                print("📈 Improving performance!")
            else:
                print("🔧 Needs more training!")

        # Training assessment
        print(f"\n🧠 Training Assessment:")
        if len(action_counts) >= 8:
            print("✅ Good action diversity - AI learned to use multiple moves")
        elif len(action_counts) >= 5:
            print("👍 Moderate action diversity - AI uses several moves")
        else:
            print("⚠️ Limited action diversity - AI might need more training")

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
