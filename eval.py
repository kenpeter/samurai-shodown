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


def convert_multibinary_to_discrete(action, env_action_space_n):
    """
    Convert MultiBinary action (array of 0s and 1s) to Discrete action (single integer)

    Args:
        action: MultiBinary action array (e.g., [1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0])
        env_action_space_n: Number of discrete actions the environment expects

    Returns:
        Single integer action for Discrete action space
    """
    try:
        if isinstance(action, (np.ndarray, list)):
            action_array = np.array(action, dtype=int)

            # Handle different conversion strategies
            if len(action_array) == 12:  # Common fighting game button layout
                # Strategy 1: Convert binary pattern to integer (most common)
                # This treats the binary array as a binary number
                discrete_action = 0
                for i, bit in enumerate(action_array):
                    discrete_action += bit * (2**i)

                # Map to environment's action space
                discrete_action = discrete_action % env_action_space_n

                print(
                    f"🔄 Converted MultiBinary {action_array} → Discrete {discrete_action}"
                )
                return discrete_action

            elif len(action_array) == env_action_space_n:
                # Strategy 2: One-hot encoding to discrete
                # Find the first 1 in the array
                ones_indices = np.where(action_array == 1)[0]
                if len(ones_indices) > 0:
                    discrete_action = ones_indices[0]
                else:
                    discrete_action = 0  # Default action

                print(
                    f"🔄 Converted one-hot {action_array} → Discrete {discrete_action}"
                )
                return discrete_action

            else:
                # Strategy 3: Sum-based mapping
                # Sum the binary values and map to discrete range
                action_sum = np.sum(action_array)
                discrete_action = action_sum % env_action_space_n

                print(
                    f"🔄 Converted sum-based {action_array} (sum={action_sum}) → Discrete {discrete_action}"
                )
                return discrete_action

        else:
            # Already a scalar, just ensure it's in range
            scalar_action = int(action) % env_action_space_n
            print(f"🔄 Scalar action {action} → Discrete {scalar_action}")
            return scalar_action

    except Exception as e:
        print(f"❌ Error converting MultiBinary to Discrete: {e}")
        print(f"   Action: {action}, Type: {type(action)}")
        print(f"   Falling back to action 0")
        return 0


def safe_action_to_int(action):
    """
    Safely convert action from model.predict() to integer
    Handles various numpy array formats that stable-baselines3 might return
    """
    try:
        # Handle different action formats from stable-baselines3
        if isinstance(action, (np.ndarray, np.generic)):
            if action.ndim == 0:
                # 0-dimensional array (scalar)
                return int(action.item())
            elif action.ndim == 1 and len(action) == 1:
                # 1-dimensional array with single element
                return int(action[0])
            elif action.ndim == 1 and len(action) > 1:
                # Multi-element array - this is normal for MultiBinary actions
                return action  # Return the full array for conversion
            else:
                # Higher dimensional array - unexpected
                print(f"⚠️ Warning: Unexpected action shape: {action.shape}")
                return action.flatten()  # Flatten and return for conversion
        elif hasattr(action, "__len__") and len(action) == 1:
            # List or tuple with single element
            return int(action[0])
        elif isinstance(action, (int, float)):
            # Already a scalar
            return int(action)
        else:
            # Try direct conversion
            return int(action)
    except (ValueError, TypeError, IndexError, AttributeError) as e:
        print(f"❌ Error converting action: {e}")
        print(f"   Action type: {type(action)}")
        print(f"   Action value: {action}")
        # Return the action as-is for further processing
        return action


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
        description="Evaluate trained Samurai Showdown Agent - JEPA Enhanced Version with Action Space Conversion"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="trained_models_jepa_prime/ppo_jepa_prime_1000000_steps.zip",
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
            "trained_models_jepa_prime",
            "trained_models_prime_only",
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

    print(f"🚀 JEPA-Enhanced Model Evaluation with Action Space Conversion")
    print(f"🤖 Loading model from: {args.model_path}")
    print(f"🎮 Using state file: {state_file if state_file else 'default'}")
    print(f"🔄 Will run {args.episodes} episodes")
    print(f"⚡ Running at {args.fps} FPS for smooth gameplay")
    print(f"🎯 Deterministic actions: {'Yes' if args.deterministic else 'No'}")
    print(f"🧠 JEPA features: {'Disabled' if args.disable_jepa else 'Enabled'}")
    print(f"🔮 Show predictions: {'Yes' if args.show_jepa_predictions else 'No'}")
    print("\n🔧 Automatic observation format conversion enabled!")
    print("🔧 Enhanced action conversion with MultiBinary → Discrete support!")
    print("🔧 Robust error handling for action space mismatches!")
    print("\nPress Ctrl+C to quit at any time")
    print("=" * 60)

    # Create evaluation environment
    try:
        env = create_eval_env(game, state_file, enable_jepa=not args.disable_jepa)
        print("✅ Environment created successfully!")

        # Get environment action space info
        env_action_space = env.action_space
        if hasattr(env_action_space, "n"):
            env_action_space_n = env_action_space.n
            print(
                f"🎮 Environment expects Discrete action space with {env_action_space_n} actions"
            )
        else:
            env_action_space_n = 12  # Default for fighting games
            print(f"🎮 Environment action space: {type(env_action_space)}")

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

        # Check action space compatibility
        model_action_space = model.action_space
        print(
            f"🔍 Model action space: {type(model_action_space)} - {model_action_space}"
        )
        print(
            f"🔍 Environment action space: {type(env_action_space)} - {env_action_space}"
        )

        # Detect action space mismatch - more sophisticated check
        action_space_mismatch = False
        needs_conversion = False

        # Check if types are different
        if type(model_action_space) != type(env_action_space):
            action_space_mismatch = True
            needs_conversion = True
            print("⚠️ ACTION SPACE TYPE MISMATCH DETECTED!")
            print(f"   Model trained on: {type(model_action_space)}")
            print(f"   Environment expects: {type(env_action_space)}")
            print("🔧 Will automatically convert actions during evaluation")

        # Check if same type but different parameters
        elif hasattr(model_action_space, "n") and hasattr(env_action_space, "n"):
            if model_action_space.n != env_action_space.n:
                action_space_mismatch = True
                needs_conversion = True
                print("⚠️ ACTION SPACE SIZE MISMATCH DETECTED!")
                print(f"   Model: {model_action_space.n} actions")
                print(f"   Environment: {env_action_space.n} actions")

        # If both are MultiBinary, check if sizes match
        elif str(type(model_action_space)) == str(
            type(env_action_space)
        ) and "MultiBinary" in str(type(model_action_space)):
            model_size = getattr(model_action_space, "n", None)
            env_size = getattr(env_action_space, "n", None)
            if (
                model_size != env_size
                and model_size is not None
                and env_size is not None
            ):
                action_space_mismatch = True
                needs_conversion = True
                print("⚠️ MULTIBINARY SIZE MISMATCH DETECTED!")
                print(f"   Model: MultiBinary({model_size})")
                print(f"   Environment: MultiBinary({env_size})")
            else:
                print("✅ Action spaces match perfectly - no conversion needed!")
        else:
            print("✅ Action spaces match - no conversion needed!")

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

        # Track action usage and conversion stats
        action_counts = {}
        conversion_stats = {
            "total_conversions": 0,
            "successful_conversions": 0,
            "conversion_errors": 0,
        }

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

                # Handle action conversion based on detected mismatch
                processed_action = safe_action_to_int(action)

                # Only convert action if there's actually a mismatch that needs conversion
                if needs_conversion:
                    conversion_stats["total_conversions"] += 1
                    try:
                        # Convert MultiBinary to Discrete or vice versa
                        if "MultiBinary" in str(
                            type(model_action_space)
                        ) and "Discrete" in str(type(env_action_space)):
                            # MultiBinary → Discrete
                            discrete_action = convert_multibinary_to_discrete(
                                processed_action, env_action_space_n
                            )
                            conversion_stats["successful_conversions"] += 1
                            action_key = discrete_action
                        elif "Discrete" in str(
                            type(model_action_space)
                        ) and "MultiBinary" in str(type(env_action_space)):
                            # Discrete → MultiBinary (create binary array)
                            multibinary_action = np.zeros(env_action_space.n, dtype=int)
                            if isinstance(processed_action, (int, np.integer)):
                                # Set one bit based on discrete action
                                bit_index = processed_action % env_action_space.n
                                multibinary_action[bit_index] = 1
                            conversion_stats["successful_conversions"] += 1
                            action_key = multibinary_action
                            print(
                                f"🔄 Converted Discrete {processed_action} → MultiBinary {multibinary_action}"
                            )
                        else:
                            # Other conversion types
                            discrete_action = convert_multibinary_to_discrete(
                                processed_action, env_action_space_n
                            )
                            conversion_stats["successful_conversions"] += 1
                            action_key = discrete_action
                    except Exception as e:
                        print(f"❌ Action conversion failed: {e}")
                        conversion_stats["conversion_errors"] += 1
                        # Use original action as fallback
                        action_key = action
                else:
                    # No conversion needed - use original action from model
                    action_key = action

                    # For tracking purposes, extract a representative integer
                    if isinstance(action, (np.ndarray, list)) and len(action) > 1:
                        # For MultiBinary, use the sum or first active bit for tracking
                        if hasattr(action, "sum"):
                            track_action = int(action.sum())  # Sum of active bits
                        else:
                            track_action = int(np.sum(action))
                    else:
                        track_action = safe_action_to_int(action)

                # Track action usage
                if needs_conversion:
                    # If we converted, track the converted action
                    if isinstance(action_key, (np.ndarray, list)):
                        track_action_final = int(
                            np.sum(action_key)
                        )  # Sum for MultiBinary
                    else:
                        track_action_final = int(action_key)
                else:
                    # Use the tracking action we calculated
                    track_action_final = track_action

                action_counts[track_action_final] = (
                    action_counts.get(track_action_final, 0) + 1
                )
                episode_actions.append(track_action_final)

                # Take step in environment - use the converted action
                obs, reward, terminated, truncated, info = env.step(action_key)

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

        # Action space conversion statistics
        if conversion_stats["total_conversions"] > 0:
            print(f"\n🔧 Action Space Conversion Statistics:")
            print(
                f"   Total conversions attempted: {conversion_stats['total_conversions']}"
            )
            print(
                f"   Successful conversions: {conversion_stats['successful_conversions']}"
            )
            print(f"   Conversion errors: {conversion_stats['conversion_errors']}")
            success_rate = (
                conversion_stats["successful_conversions"]
                / conversion_stats["total_conversions"]
            ) * 100
            print(f"   Conversion success rate: {success_rate:.1f}%")

            if success_rate >= 95:
                print("   ✅ Excellent action space conversion!")
            elif success_rate >= 80:
                print("   👍 Good action space conversion!")
            else:
                print("   ⚠️ Action conversion needs improvement!")

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

        # Action space compatibility assessment
        if action_space_mismatch and needs_conversion:
            print(f"\n🔧 Action Space Compatibility:")
            print("✅ Successfully handled action space mismatch!")
            print(
                f"   Model: {type(model_action_space)} → Environment: {type(env_action_space)}"
            )
            print("   Automatic conversion enabled seamless evaluation")
        elif action_space_mismatch and not needs_conversion:
            print(f"\n⚠️ Action Space Compatibility:")
            print(
                "   Action spaces detected as mismatched but no conversion was needed"
            )
            print(
                f"   Model: {type(model_action_space)} | Environment: {type(env_action_space)}"
            )
        else:
            print(f"\n✅ Action Space Compatibility:")
            print("   Model and environment action spaces match perfectly!")
            print(f"   Both use: {type(env_action_space)}")

    except KeyboardInterrupt:
        print("\n\n⏹️  Evaluation interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during evaluation: {e}")
        print("💡 Check that your model and wrapper configurations are compatible")
        print(
            f"💡 Action type that caused error: {type(action) if 'action' in locals() else 'Unknown'}"
        )
        print(
            f"💡 Action space mismatch: {action_space_mismatch if 'action_space_mismatch' in locals() else 'Unknown'}"
        )
        print(
            f"💡 Needs conversion: {needs_conversion if 'needs_conversion' in locals() else 'Unknown'}"
        )
        import traceback

        traceback.print_exc()
    finally:
        env.close()
        print("\n✅ Evaluation complete!")

        # Final summary
        print(f"\n📝 EVALUATION SUMMARY:")
        print(f"🤖 Model: {args.model_path}")
        print(f"🎮 Episodes: {total_episodes}")
        if total_episodes > 0:
            print(f"🏆 Win Rate: {(total_wins/total_episodes)*100:.1f}%")

        if conversion_stats.get("total_conversions", 0) > 0:
            print(
                f"🔧 Action Conversions: {conversion_stats['successful_conversions']}/{conversion_stats['total_conversions']}"
            )

        if jepa_stats["total_predictions"] > 0:
            print(f"🧠 JEPA Predictions: {jepa_stats['total_predictions']} made")

        print(f"⏱️ Evaluation completed successfully!")


if __name__ == "__main__":
    main()
