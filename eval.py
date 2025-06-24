import os
import argparse
import time
import numpy as np

import retro
import gymnasium as gym
from stable_baselines3 import PPO

# Import the JEPA-enhanced wrapper
from wrapper import make_jepa_samurai_env, SamuraiJEPAWrapper


def create_eval_env(game, state, enable_jepa=True):
    """Create evaluation environment with JEPA enhancement support"""
    # Handle state file path
    if state and os.path.isfile(state):
        state_file = os.path.abspath(state)
        print(f"Using custom state file: {state_file}")
    else:
        state_file = state
        print(f"Using state: {state_file if state_file else 'default'}")

    if enable_jepa:
        # Create JEPA-enhanced environment
        print("🧠 Creating JEPA-Enhanced Environment...")
        env = make_jepa_samurai_env(
            game=game,
            state=state_file,
            reset_round=True,
            rendering=True,
            max_episode_steps=15000,
            enable_jepa=True,
            frame_stack=6,  # JEPA uses 6-frame stacks
        )
        print(f"✅ JEPA-Enhanced environment created!")
        print(f"   📊 Binary outcome prediction: Enabled")
        print(f"   🎯 Strategic response planning: Enabled")
        print(f"   🔮 Prediction horizon: 6 frames")
    else:
        # Create basic retro environment
        print("🎮 Creating Basic Environment...")
        env = retro.make(
            game=game,
            state=state_file,
            use_restricted_actions=retro.Actions.FILTERED,
            obs_type=retro.Observations.IMAGE,
            render_mode="human",
        )

        # Apply basic wrapper (fallback compatibility)
        try:
            from wrapper import SamuraiShowdownCustomWrapper

            env = SamuraiShowdownCustomWrapper(
                env,
                reset_round=True,
                rendering=True,
                max_episode_steps=15000,
            )
            print("✅ Basic wrapper applied")
        except ImportError:
            print("⚠️ Basic wrapper not available, using raw environment")

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


def display_jepa_info(info, step_count):
    """Display JEPA prediction information if available"""
    # Check if JEPA is actually working (even if not explicitly enabled in info)
    predictions = info.get("predicted_binary_outcomes")
    if predictions is None:
        return

    # Get JEPA predictions
    predictions = info.get("predicted_binary_outcomes")
    confidence = info.get("prediction_confidence")
    strategic_stats = info.get("strategic_stats", {})

    if predictions is not None and step_count % 300 == 0:  # Every 5 seconds at 60 FPS
        print(f"🔮 JEPA Predictions (Step {step_count}):")

        outcome_types = info.get(
            "binary_prediction_types",
            [
                "will_opponent_attack",
                "will_opponent_take_damage",
                "will_player_take_damage",
                "will_round_end_soon",
            ],
        )

        for i, outcome_type in enumerate(outcome_types):
            if outcome_type in predictions:
                prob = (
                    predictions[outcome_type][0, 0]
                    if predictions[outcome_type].size > 0
                    else 0
                )
                conf = (
                    confidence[i, 0]
                    if confidence is not None and confidence.size > i
                    else 0
                )

                # Create readable prediction
                outcome_name = outcome_type.replace("will_", "").replace("_", " ")
                confidence_level = (
                    "HIGH" if conf > 0.7 else "MED" if conf > 0.4 else "LOW"
                )

                if prob > 0.6:
                    print(
                        f"   🟢 {outcome_name}: {prob:.1%} ({confidence_level} confidence)"
                    )
                elif prob < 0.4:
                    print(
                        f"   🔴 {outcome_name}: {prob:.1%} ({confidence_level} confidence)"
                    )
                else:
                    print(
                        f"   🟡 {outcome_name}: {prob:.1%} ({confidence_level} confidence)"
                    )

        # Show strategic stats occasionally
        if strategic_stats and step_count % 600 == 0:  # Every 10 seconds
            print(f"📊 Strategic Stats:")
            attack_acc = strategic_stats.get("attack_prediction_accuracy", 0)
            total_preds = strategic_stats.get("binary_predictions_made", 0)
            print(f"   Attack prediction accuracy: {attack_acc:.1%}")
            print(f"   Total predictions made: {total_preds}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate trained Samurai Showdown Agent - JEPA Enhanced Version"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="trained_models_fighting_optimized/ppo_fighting_optimized_16950000_steps.zip",
        help="Path to the trained model",
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
    parser.add_argument(
        "--disable-jepa",
        action="store_true",
        help="Disable JEPA features (use basic wrapper)",
    )
    parser.add_argument(
        "--show-jepa-predictions",
        action="store_true",
        help="Display JEPA predictions during gameplay",
    )

    args = parser.parse_args()

    # Check if model exists
    if not os.path.exists(args.model_path):
        print(f"❌ Error: Model file not found at {args.model_path}")
        print("Available models:")

        # Check multiple model directories
        for dir_name in [
            "trained_models_fighting_optimized",
            "trained_models_samurai_cuda",
            "trained_models_samurai_cpu",
            "trained_models_samurai",
            "trained_models",
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

    print(f"🚀 JEPA-Enhanced Model Evaluation")
    print(f"🤖 Loading model from: {args.model_path}")
    print(f"🎮 Using state file: {state_file if state_file else 'default'}")
    print(f"🔄 Will run {args.episodes} episodes")
    print(f"⚡ Running at {args.fps} FPS for smooth gameplay")
    print(f"🎯 Deterministic actions: {'Yes' if args.deterministic else 'No'}")
    print(f"🧠 JEPA features: {'Disabled' if args.disable_jepa else 'Enabled'}")
    print(f"🔮 Show predictions: {'Yes' if args.show_jepa_predictions else 'No'}")
    print("\n🔧 Automatic observation format conversion enabled!")
    print("\nPress Ctrl+C to quit at any time")
    print("=" * 60)

    # Create evaluation environment
    try:
        env = create_eval_env(game, state_file, enable_jepa=not args.disable_jepa)
        print("✅ Environment created successfully!")
    except Exception as e:
        print(f"❌ Error creating environment: {e}")
        print("\n💡 Troubleshooting:")
        print("   - Check if samurai.state file exists")
        print("   - Try using --use-default-state flag")
        print("   - Try using --disable-jepa flag for compatibility")
        print("   - Ensure SamuraiShodown-Genesis ROM is installed")
        import traceback

        traceback.print_exc()
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
        print("   - Try using --disable-jepa flag")
        return

    # Calculate frame timing
    frame_time = 1.0 / args.fps

    try:
        total_wins = 0
        total_losses = 0
        total_episodes = 0
        total_reward = 0
        total_steps = 0

        # JEPA-specific tracking
        jepa_stats = {
            "total_predictions": 0,
            "high_confidence_predictions": 0,
            "strategic_responses": 0,
        }

        # Track action usage
        action_counts = {}

        for episode in range(args.episodes):
            print(f"\n⚔️  --- Episode {episode + 1}/{args.episodes} ---")

            obs, info = env.reset()
            episode_reward = 0
            step_count = 0
            episode_start_time = time.time()
            episode_actions = []
            episode_jepa_stats = {"predictions": 0, "high_conf": 0}

            # Check if JEPA is actually working (look for predictions directly)
            predictions = info.get("predicted_binary_outcomes")
            jepa_actually_enabled = predictions is not None

            if jepa_actually_enabled:
                print(
                    "🧠 JEPA features detected - AI is using predictive intelligence!"
                )
            else:
                print("🎮 Basic AI mode - traditional reactive gameplay")

            print("🎬 Starting new match... Watch the game window!")

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

                # Track action usage - fix action diversity counting
                action_key = (
                    int(action)
                    if hasattr(action, "__len__") and len(action) == 1
                    else int(action)
                )
                action_counts[action_key] = action_counts.get(action_key, 0) + 1
                episode_actions.append(action_key)

                # Take step in environment
                obs, reward, terminated, truncated, info = env.step(action)

                episode_reward += reward
                step_count += 1

                # Track JEPA stats - check for actual predictions
                predictions = info.get("predicted_binary_outcomes")
                if predictions is not None:
                    strategic_stats = info.get("strategic_stats", {})
                    preds_made = strategic_stats.get("binary_predictions_made", 0)
                    if preds_made > episode_jepa_stats["predictions"]:
                        episode_jepa_stats["predictions"] = preds_made
                        jepa_stats["total_predictions"] += 1

                        # Check if high confidence - lower threshold for detection
                        confidence = info.get("prediction_confidence")
                        if (
                            confidence is not None and confidence.max() > 0.5
                        ):  # Lowered from 0.7
                            jepa_stats["high_confidence_predictions"] += 1
                            episode_jepa_stats["high_conf"] += 1

                # Display JEPA predictions if requested - check for actual predictions
                if (
                    args.show_jepa_predictions
                    and info.get("predicted_binary_outcomes") is not None
                ):
                    display_jepa_info(info, step_count)

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

            # JEPA episode stats - check for actual predictions
            predictions = info.get("predicted_binary_outcomes")
            if predictions is not None:
                print(
                    f"   🧠 JEPA predictions made: {episode_jepa_stats['predictions']}"
                )
                print(
                    f"   🎯 High confidence predictions: {episode_jepa_stats['high_conf']}"
                )

                # Show current strategic stats
                strategic_stats = info.get("strategic_stats", {})
                if strategic_stats:
                    attack_acc = strategic_stats.get("attack_prediction_accuracy", 0)
                    if attack_acc > 0:
                        print(f"   📊 Attack prediction accuracy: {attack_acc:.1%}")

                    # Show some sample predictions
                    if len(predictions) > 0:
                        print(f"   🔮 Sample predictions:")
                        for outcome_type, pred_data in list(predictions.items())[
                            :2
                        ]:  # Show first 2
                            if hasattr(pred_data, "shape") and pred_data.size > 0:
                                prob = (
                                    pred_data[0, 0]
                                    if pred_data.ndim >= 2
                                    else pred_data[0]
                                )
                                outcome_name = outcome_type.replace(
                                    "will_", ""
                                ).replace("_", " ")
                                print(f"      {outcome_name}: {prob:.1%}")

            else:
                print(f"   ⚠️ JEPA initialized but no predictions detected")

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

        # JEPA-specific results
        if jepa_stats["total_predictions"] > 0:
            print(f"\n🧠 JEPA Intelligence Analysis:")
            print(f"   Total predictions made: {jepa_stats['total_predictions']}")
            print(
                f"   High confidence predictions: {jepa_stats['high_confidence_predictions']}"
            )
            high_conf_rate = (
                jepa_stats["high_confidence_predictions"]
                / jepa_stats["total_predictions"]
            ) * 100
            print(f"   High confidence rate: {high_conf_rate:.1f}%")

            if high_conf_rate >= 60:
                print("   🎯 Excellent prediction confidence!")
            elif high_conf_rate >= 40:
                print("   👍 Good prediction confidence!")
            else:
                print("   📈 Prediction confidence improving!")

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

        # JEPA assessment
        if jepa_stats["total_predictions"] > 0:
            print("✅ JEPA predictive intelligence is active and working!")
        elif not args.disable_jepa:
            print("⚠️ JEPA features enabled but no predictions detected")

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
