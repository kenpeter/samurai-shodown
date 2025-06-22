import os
import argparse
import time
import numpy as np
import cv2

# Use stable-retro for gymnasium compatibility
try:
    import stable_retro as retro

    print("🎮 Using stable-retro (gymnasium compatible)")
except ImportError:
    try:
        import retro

        print("🎮 Using retro (legacy)")
    except ImportError:
        raise ImportError(
            "Neither stable-retro nor retro found. Install with: pip install stable-retro"
        )

import gymnasium as gym
from stable_baselines3 import PPO

# Import the PRIME-optimized wrapper
from wrapper import SamuraiShowdownCustomWrapper


def create_eval_env(game, state):
    """
    Create evaluation environment EXACTLY matching PRIME training setup
    CRITICAL: Must match wrapper.py configuration precisely
    """
    # Handle state file path
    if state and os.path.isfile(state):
        state_file = os.path.abspath(state)
        print(f"🎮 Using custom state file: {state_file}")
    else:
        state_file = state
        print(f"🎮 Using state: {state_file if state_file else 'default'}")

    # Create retro environment with rendering enabled
    env = retro.make(
        game=game,
        state=state_file,
        use_restricted_actions=retro.Actions.FILTERED,
        obs_type=retro.Observations.IMAGE,
        render_mode="human",  # Enable rendering for human observation
    )

    # Apply PRIME-optimized wrapper with EXACT training settings
    # CRITICAL: These parameters must match training exactly!
    env = SamuraiShowdownCustomWrapper(
        env,
        reset_round=True,
        rendering=True,
        max_episode_steps=15000,  # Match training configuration
        frame_stack=4,  # 4 frames for memory efficiency
        frame_skip=4,  # Match training frame skip
        target_size=(180, 126),  # Match training target size (width, height)
    )

    # Verify observation space matches training expectations
    expected_shape = (12, 180, 126)  # 4 frames × 3 RGB = 12 channels
    actual_shape = env.observation_space.shape

    print(f"🔍 Expected observation shape: {expected_shape}")
    print(f"🔍 Actual observation shape: {actual_shape}")

    if actual_shape == expected_shape:
        print("✅ Observation shapes match perfectly!")
    else:
        print("⚠️ WARNING: Observation shape mismatch detected!")
        print("   This may cause model prediction errors.")

    return env


def validate_model_compatibility(model, env):
    """
    Validate that the loaded model is compatible with the evaluation environment
    """
    model_obs_shape = model.observation_space.shape
    env_obs_shape = env.observation_space.shape

    print(f"\n🔍 Model Compatibility Check:")
    print(f"   Model expects: {model_obs_shape}")
    print(f"   Environment provides: {env_obs_shape}")

    if model_obs_shape == env_obs_shape:
        print("✅ Perfect compatibility!")
        return True
    else:
        print("❌ Shape mismatch detected!")
        print("   This will likely cause prediction errors.")
        return False


def test_model_prediction(model, env):
    """
    Test a single model prediction to ensure everything works
    """
    print("\n🧪 Testing model prediction...")
    try:
        obs, _ = env.reset()
        print(f"   Reset obs shape: {obs.shape}")
        print(f"   Reset obs type: {type(obs)}")
        print(f"   Reset obs dtype: {obs.dtype}")

        # Test prediction
        action, _states = model.predict(obs, deterministic=True)
        print(f"   Prediction successful: action = {action}")
        print(f"   Action type: {type(action)}")
        print("✅ Model prediction test passed!")
        return True

    except Exception as e:
        print(f"❌ Model prediction test failed: {e}")
        return False


def analyze_episode_performance(info, episode_reward, step_count, outcome_reward):
    """
    Analyze episode performance using PRIME metrics
    Fixed outcome detection using multiple signals
    """
    game_info = info.get("game_info", {})
    player_health = game_info.get("player_health", 0)
    opponent_health = game_info.get("opponent_health", 0)

    # PRIME-specific metrics
    process_reward = info.get("process_reward", 0)
    combo_length = info.get("combo_length", 0)
    win_rate = info.get("win_rate", 0) * 100 if info.get("win_rate") else 0

    # IMPROVED outcome detection using multiple signals
    outcome = "UNKNOWN"
    outcome_emoji = "❓"

    # Method 1: Use outcome_reward from PRIME wrapper (most reliable)
    if outcome_reward > 0:
        outcome = "WIN"
        outcome_emoji = "🟢"
    elif outcome_reward < 0:
        outcome = "LOSS"
        outcome_emoji = "🔴"
    else:
        # Method 2: Fallback to health comparison
        if player_health > opponent_health:
            outcome = "WIN"
            outcome_emoji = "🟢"
        elif opponent_health > player_health:
            outcome = "LOSS"
            outcome_emoji = "🔴"
        else:
            # Method 3: Check if both are dead (true draw or timeout)
            if player_health <= 0 and opponent_health <= 0:
                # Could be a draw or we need more info
                if abs(episode_reward) < 1000:  # Small total reward suggests timeout
                    outcome = "TIMEOUT"
                    outcome_emoji = "⏰"
                else:
                    outcome = "DRAW"
                    outcome_emoji = "⚪"
            else:
                outcome = "TIMEOUT"
                outcome_emoji = "⏰"

    return {
        "outcome": outcome,
        "outcome_emoji": outcome_emoji,
        "player_health": player_health,
        "opponent_health": opponent_health,
        "process_reward": process_reward,
        "outcome_reward": outcome_reward,
        "combo_length": combo_length,
        "win_rate": win_rate,
        "episode_reward": episode_reward,
        "step_count": step_count,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate PRIME-trained Samurai Showdown Agent"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="trained_models_simple_prime/ppo_simple_prime_975000_steps.zip",
        help="Path to the trained PRIME model",
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
        "--debug",
        action="store_true",
        help="Enable debug mode with extra information",
    )

    args = parser.parse_args()

    # Check if model exists
    if not os.path.exists(args.model_path):
        print(f"❌ Error: Model file not found at {args.model_path}")
        print("\n📁 Available models:")

        # Check for model directories
        model_dirs = [
            "trained_models_simple_prime",
            "trained_models_samurai_cuda",
            "trained_models_samurai_cpu",
            "trained_models_samurai",
            "trained_models",
        ]

        for dir_name in model_dirs:
            if os.path.exists(dir_name):
                print(f"   📂 {dir_name}/:")
                for f in os.listdir(dir_name):
                    if f.endswith(".zip"):
                        file_size = os.path.getsize(os.path.join(dir_name, f)) / (
                            1024 * 1024
                        )
                        print(f"     🤖 {f} ({file_size:.1f} MB)")
        return

    game = "SamuraiShodown-Genesis"

    # Handle state file properly
    if args.use_default_state:
        state_file = None
        print("🎮 Using default game state")
    else:
        state_file = args.state_file

    print(f"🚀 PRIME MODEL EVALUATION")
    print(f"🤖 Model: {args.model_path}")
    print(f"🎮 State: {state_file if state_file else 'default'}")
    print(f"🔄 Episodes: {args.episodes}")
    print(f"⚡ FPS: {args.fps}")
    print(f"🎯 Deterministic: {'Yes' if args.deterministic else 'No'}")
    print(f"🔧 Debug mode: {'Yes' if args.debug else 'No'}")
    print(f"🧠 PRIME methodology: Process + Outcome rewards")
    print(f"🎨 Simple CNN architecture")
    print("\n" + "=" * 60)

    # Create evaluation environment
    print("\n🔧 Creating PRIME evaluation environment...")
    try:
        env = create_eval_env(game, state_file)
        print("✅ Environment created successfully!")
    except Exception as e:
        print(f"❌ Error creating environment: {e}")
        print("\n💡 Troubleshooting:")
        print("   • Check if samurai.state file exists")
        print("   • Try using --use-default-state flag")
        print("   • Ensure SamuraiShodown-Genesis ROM is installed")
        print("   • Verify stable-retro is properly installed")
        return

    # Load the trained model with device selection
    print("\n🧠 Loading PRIME model...")
    try:
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
                    # Print CUDA info
                    print(f"   GPU: {torch.cuda.get_device_name()}")
                    print(
                        f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
                    )
                else:
                    device = "cpu"
                    print("💻 CUDA not available, using CPU")
            except ImportError:
                device = "cpu"
                print("💻 PyTorch not available, using CPU")

        # Load model on selected device
        try:
            model = PPO.load(args.model_path, device=device)
            print(f"✅ PRIME model loaded on {device.upper()}!")
        except Exception as e:
            print(f"⚠️ Failed to load on {device}, trying CPU...")
            model = PPO.load(args.model_path, device="cpu")
            device = "cpu"
            print("✅ Model loaded on CPU!")

        # Validate model compatibility
        is_compatible = validate_model_compatibility(model, env)

        if not is_compatible:
            print("⚠️ Model and environment shapes don't match!")
            print("   This may indicate a configuration mismatch.")
            print("   Proceeding anyway, but expect potential issues...")

        # Test model prediction
        if not test_model_prediction(model, env):
            print("❌ Model prediction test failed!")
            print("   Cannot proceed with evaluation.")
            return

    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("\n💡 Common issues:")
        print("   • Model trained with different wrapper settings")
        print("   • GPU/CPU compatibility issues")
        print("   • Model file is corrupted")
        print("   • Try using --force-cpu flag")
        import traceback

        traceback.print_exc()
        return

    # Calculate frame timing
    frame_time = 1.0 / args.fps

    print(f"\n🎬 Starting PRIME evaluation...")
    print("   Press Ctrl+C to quit at any time")
    print("   Watch the game window for AI gameplay!")

    # Get initial wrapper stats
    try:
        initial_stats = env.current_stats.copy()
        initial_wins = initial_stats.get("wins", 0)
        initial_losses = initial_stats.get("losses", 0)
        print(f"   📊 Initial wrapper stats: {initial_wins}W/{initial_losses}L")
    except:
        initial_wins = 0
        initial_losses = 0
        print("   📊 Could not read initial wrapper stats")

    try:
        # Statistics tracking
        total_wins = 0
        total_losses = 0
        total_draws = 0
        total_timeouts = 0
        total_episodes = 0
        total_reward = 0
        total_steps = 0
        total_process_reward = 0
        total_outcome_reward = 0

        # Action tracking
        action_counts = {}
        all_episode_stats = []

        for episode in range(args.episodes):
            print(f"\n⚔️ === Episode {episode + 1}/{args.episodes} ===")

            # Reset environment
            obs, info = env.reset()
            episode_reward = 0
            episode_process_reward = 0
            episode_outcome_reward = 0
            step_count = 0
            episode_start_time = time.time()
            episode_actions = []

            if args.debug:
                print(f"🔍 Reset obs shape: {obs.shape}")
                print(f"🔍 Reset obs type: {obs.dtype}")

            print("🎬 Match starting... Watch the AI play!")

            while True:
                step_start_time = time.time()

                # Get action from the trained model
                try:
                    action, _states = model.predict(
                        obs, deterministic=args.deterministic
                    )
                except Exception as e:
                    print(f"❌ Prediction error: {e}")
                    break

                # Track action usage
                action_key = str(action) if hasattr(action, "__iter__") else int(action)
                action_counts[action_key] = action_counts.get(action_key, 0) + 1
                episode_actions.append(action_key)

                # Take step in environment
                try:
                    obs, reward, terminated, truncated, info = env.step(action)
                except Exception as e:
                    print(f"❌ Environment step error: {e}")
                    break

                # Track rewards
                episode_reward += reward
                step_count += 1

                # Track PRIME-specific rewards
                process_reward = info.get("process_reward", 0)
                outcome_reward = info.get("outcome_reward", 0)
                episode_process_reward += process_reward
                episode_outcome_reward += outcome_reward

                # Frame rate limiting
                elapsed = time.time() - step_start_time
                if elapsed < frame_time:
                    time.sleep(frame_time - elapsed)

                # Debug output
                if args.debug and step_count % (args.fps * 10) == 0:  # Every 10 seconds
                    game_info = info.get("game_info", {})
                    player_hp = game_info.get("player_health", "?")
                    enemy_hp = game_info.get("opponent_health", "?")
                    combo = info.get("combo_length", 0)
                    print(
                        f"   🔍 Step {step_count}: HP {player_hp}vs{enemy_hp}, Combo: {combo}"
                    )

                # Check if episode is done
                if terminated or truncated:
                    break

            # Episode finished - analyze performance
            episode_time = time.time() - episode_start_time

            # Get the final outcome reward for proper win/loss detection
            final_outcome_reward = info.get("outcome_reward", 0)

            performance = analyze_episode_performance(
                info, episode_reward, step_count, final_outcome_reward
            )

            # Update statistics
            total_episodes += 1
            total_reward += episode_reward
            total_steps += step_count
            total_process_reward += episode_process_reward
            total_outcome_reward += episode_outcome_reward

            # Count outcomes
            if performance["outcome"] == "WIN":
                total_wins += 1
            elif performance["outcome"] == "LOSS":
                total_losses += 1
            elif performance["outcome"] == "DRAW":
                total_draws += 1
            else:
                total_timeouts += 1

            # Store episode stats
            all_episode_stats.append(performance)

            # Print episode summary
            print(f"\n🏁 Episode {episode + 1} Complete!")
            print(
                f"   {performance['outcome_emoji']} Outcome: {performance['outcome']}"
            )
            print(
                f"   🏥 Final HP: Player {performance['player_health']}, Enemy {performance['opponent_health']}"
            )
            print(f"   🎯 Total Reward: {episode_reward:.1f}")
            print(f"   🔄 Process Reward: {episode_process_reward:.1f}")
            print(f"   🎖️ Outcome Reward: {final_outcome_reward:.1f}")
            print(f"   🥊 Max Combo: {performance['combo_length']}")
            print(f"   ⏱️ Duration: {episode_time:.1f}s ({step_count} steps)")

            # Debug outcome detection
            if args.debug:
                print(f"   🔍 Debug - Outcome detection:")
                print(f"       Outcome reward: {final_outcome_reward}")
                print(
                    f"       Health comparison: {performance['player_health']} vs {performance['opponent_health']}"
                )
                print(f"       Final classification: {performance['outcome']}")

            # Action diversity
            unique_actions = len(set(episode_actions))
            print(f"   🎮 Action Diversity: {unique_actions} unique actions")

            # Pause between episodes
            if episode < args.episodes - 1:
                print("\n⏳ Waiting 3 seconds before next episode...")
                time.sleep(3)

        # Calculate final statistics
        print(f"\n" + "=" * 60)
        print(f"📊 PRIME EVALUATION RESULTS")
        print(f"=" * 60)

        # Get final wrapper stats for comparison
        try:
            final_stats = env.current_stats.copy()
            final_wins = final_stats.get("wins", 0)
            final_losses = final_stats.get("losses", 0)
            wrapper_wins = final_wins - initial_wins
            wrapper_losses = final_losses - initial_losses

            print(f"🎯 WRAPPER STATS (Most Accurate):")
            print(f"   Wins during evaluation: {wrapper_wins}")
            print(f"   Losses during evaluation: {wrapper_losses}")
            if wrapper_wins + wrapper_losses > 0:
                wrapper_win_rate = (
                    wrapper_wins / (wrapper_wins + wrapper_losses)
                ) * 100
                print(f"   Wrapper Win Rate: {wrapper_win_rate:.1f}%")
            print(f"   Total wrapper stats: {final_wins}W/{final_losses}L")
        except:
            print(f"🎯 Could not read wrapper stats")
            wrapper_wins = None
            wrapper_losses = None

        # Outcome statistics from our detection
        print(f"\n🏆 Our Detection Results:")
        print(f"   Wins: {total_wins} ({total_wins/total_episodes*100:.1f}%)")
        print(f"   Losses: {total_losses} ({total_losses/total_episodes*100:.1f}%)")
        print(f"   Draws: {total_draws} ({total_draws/total_episodes*100:.1f}%)")
        print(
            f"   Timeouts: {total_timeouts} ({total_timeouts/total_episodes*100:.1f}%)"
        )

        # Additional debugging for outcome detection
        if args.debug:
            print(f"\n🔍 Debug - Outcome Details:")
            for i, stats in enumerate(all_episode_stats):
                print(
                    f"   Episode {i+1}: {stats['outcome']} (OR: {stats['outcome_reward']:.1f})"
                )

        # Performance metrics
        win_rate = (total_wins / total_episodes) * 100
        avg_reward = total_reward / total_episodes
        avg_steps = total_steps / total_episodes
        avg_process_reward = total_process_reward / total_episodes
        avg_outcome_reward = total_outcome_reward / total_episodes

        print(f"\n📈 Performance Metrics:")
        print(f"   Win Rate: {win_rate:.1f}%")
        print(f"   Average Reward: {avg_reward:.1f}")
        print(f"   Average Process Reward: {avg_process_reward:.1f}")
        print(f"   Average Outcome Reward: {avg_outcome_reward:.1f}")
        print(f"   Average Episode Length: {avg_steps:.0f} steps")

        # Action analysis
        print(f"\n🎮 Action Analysis:")
        print(f"   Total Actions: {sum(action_counts.values()):,}")
        print(f"   Unique Actions Used: {len(action_counts)}")

        # Top actions
        if action_counts:
            sorted_actions = sorted(
                action_counts.items(), key=lambda x: x[1], reverse=True
            )
            print(f"   Top 5 Actions:")
            for i, (action, count) in enumerate(sorted_actions[:5]):
                percentage = (count / sum(action_counts.values())) * 100
                print(
                    f"     {i+1}. Action {action}: {count:,} times ({percentage:.1f}%)"
                )

        # Performance assessment using the most accurate data
        actual_wins = wrapper_wins if wrapper_wins is not None else total_wins
        actual_total = (
            (wrapper_wins + wrapper_losses)
            if (wrapper_wins is not None and wrapper_losses is not None)
            else total_episodes
        )
        actual_win_rate = (actual_wins / actual_total * 100) if actual_total > 0 else 0

        print(
            f"\n🎯 PRIME Assessment (Using {'Wrapper' if wrapper_wins is not None else 'Detection'} Stats):"
        )
        print(f"   Actual Win Rate: {actual_win_rate:.1f}%")

        if actual_win_rate >= 80:
            print("🏆 EXCELLENT! Outstanding PRIME performance!")
        elif actual_win_rate >= 65:
            print("🥇 GREAT! Strong PRIME training results!")
        elif actual_win_rate >= 50:
            print("👍 GOOD! Solid PRIME performance!")
        elif actual_win_rate >= 35:
            print("📈 IMPROVING! PRIME showing progress!")
        else:
            print("🔧 TRAINING NEEDED! Requires more PRIME optimization!")

        # Action diversity assessment
        action_diversity = len(action_counts)
        if action_diversity >= 12:
            print("✅ Excellent action diversity - AI mastered complex moves!")
        elif action_diversity >= 8:
            print("👍 Good action diversity - AI uses varied strategies!")
        elif action_diversity >= 5:
            print("📊 Moderate action diversity - AI learned basic combinations!")
        else:
            print("⚠️ Limited action diversity - Consider longer training!")

        # PRIME-specific insights
        process_vs_outcome = avg_process_reward / (avg_outcome_reward + 1e-6)
        print(f"\n🧠 PRIME Insights:")
        print(f"   Process/Outcome Ratio: {process_vs_outcome:.2f}")
        if process_vs_outcome > 2.0:
            print(
                "   ✅ Strong process learning - AI mastered step-by-step improvements!"
            )
        elif process_vs_outcome > 1.0:
            print("   👍 Balanced learning - Good mix of process and outcome rewards!")
        else:
            print("   📊 Outcome-focused - AI learned to win but may lack finesse!")

    except KeyboardInterrupt:
        print("\n\n⏹️ Evaluation interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during evaluation: {e}")
        print("💡 Check that your model and wrapper configurations match training!")
        if args.debug:
            import traceback

            traceback.print_exc()
    finally:
        env.close()
        print("\n✅ PRIME evaluation complete!")
        print("🎮 Thanks for using the PRIME evaluation system!")


if __name__ == "__main__":
    main()
