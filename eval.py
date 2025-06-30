#!/usr/bin/env python3
"""
JEPA Model Performance Evaluation Script - UPDATED FOR CURRENT WRAPPER
"""
import os
import argparse
import time
import torch
import numpy as np
import json

# Import required modules
try:
    import stable_retro as retro
except ImportError:
    # Fallback to legacy retro if stable-retro is not available
    try:
        import retro
    except ImportError:
        raise ImportError("Neither stable-retro nor retro found.")

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from wrapper import SamuraiJEPAWrapperImproved  # Updated wrapper name


def evaluate_model(
    model_path, num_episodes=10, enable_jepa=True, render=False, verbose=True
):
    """
    Evaluate a trained model's performance.
    """
    if verbose:
        print(f"🚀 MODEL EVALUATION")
        print(f"   📁 Model: {model_path}")
        print(f"   🎮 Episodes: {num_episodes}")
        print(f"   🧠 JEPA: {enable_jepa}")
        print(f"   👁️ Render: {render}")

    # --- Create Environment ---
    try:
        game = "SamuraiShodown-Genesis"

        # Use absolute path for the state file
        state_filename = "samurai.state"
        if os.path.exists(state_filename):
            state_path = os.path.abspath(state_filename)
            if verbose:
                print(f"   🕹️ Using state file: {state_path}")
        else:
            state_path = retro.State.DEFAULT
            if verbose:
                print("   🕹️ State file not found. Using default initial state.")

        env = retro.make(
            game=game,
            state=state_path,
            use_restricted_actions=retro.Actions.FILTERED,
            obs_type=retro.Observations.IMAGE,
            render_mode="human" if render else None,
        )

        # Updated wrapper with correct parameters
        env = SamuraiJEPAWrapperImproved(
            env,
            frame_stack=8,  # Match training parameters
            enable_jepa=enable_jepa,
        )

        # Add Monitor for episode tracking
        env = Monitor(env)

        if verbose:
            print(f"✅ Environment created successfully")

    except Exception as e:
        print(f"❌ Failed to create environment: {e}")
        return None

    # --- Load Model ---
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = PPO.load(model_path, env=env, device=device)
        if verbose:
            print(f"✅ Model loaded on {device}")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        env.close()
        return None

    # Updated JEPA injection for current wrapper
    if enable_jepa:
        try:
            # Access the wrapper directly (before Monitor wrapper)
            wrapper_env = env.env  # Get past Monitor wrapper
            if hasattr(wrapper_env, "inject_feature_extractor"):
                wrapper_env.inject_feature_extractor(model.policy.features_extractor)
                if verbose:
                    print("   💉 Injected PPO feature extractor into JEPA wrapper.")
            else:
                if verbose:
                    print(
                        "   ⚠️ JEPA injection not available - running without JEPA features"
                    )
        except Exception as e:
            if verbose:
                print(f"   ⚠️ JEPA injection failed: {e}")

    # --- Evaluation Loop ---
    stats = {
        "wins": 0,
        "losses": 0,
        "episode_rewards": [],
        "episode_lengths": [],
        "strategic_stats": {
            "attack_accuracy": [],
            "defense_accuracy": [],
        },
    }

    start_time = time.time()
    for episode in range(num_episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0
        episode_length = 0
        episode_start_time = time.time()

        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)

            episode_reward += reward
            episode_length += 1

            if done or truncated:
                break

        # Extract final stats from wrapper
        final_stats = info.get("current_stats", {})
        strategic_stats = info.get("strategic_stats", {})

        # Update episode stats
        stats["episode_rewards"].append(episode_reward)
        stats["episode_lengths"].append(episode_length)

        # Determine episode outcome from wrapper stats
        episode_wins = final_stats.get("wins", 0)
        episode_losses = final_stats.get("losses", 0)

        # Check if this episode resulted in a win or loss
        # (Compare with previous totals if we had them, or use heuristics)
        outcome = "DRAW"
        if episode_reward > 0:  # Positive reward generally indicates winning
            outcome = "WIN"
            stats["wins"] += 1
        else:
            outcome = "LOSS"
            stats["losses"] += 1

        # Extract JEPA accuracy if available
        if enable_jepa and strategic_stats:
            accuracies = strategic_stats.get("strategic_accuracies", {})
            if accuracies:
                attack_acc = accuracies.get("is_best_time_to_attack", 0) * 100
                defense_acc = accuracies.get("is_best_time_to_defend", 0) * 100
                stats["strategic_stats"]["attack_accuracy"].append(attack_acc)
                stats["strategic_stats"]["defense_accuracy"].append(defense_acc)

        episode_time = time.time() - episode_start_time

        if verbose:
            if enable_jepa and strategic_stats:
                accuracies = strategic_stats.get("strategic_accuracies", {})
                attack_acc = accuracies.get("is_best_time_to_attack", 0) * 100
                defense_acc = accuracies.get("is_best_time_to_defend", 0) * 100
                print(
                    f"   Episode {episode+1:2d}/{num_episodes}: {outcome:4s} | "
                    f"Reward: {episode_reward:6.1f} | Length: {episode_length:4d} | "
                    f"Attack: {attack_acc:.1f}% | Defense: {defense_acc:.1f}% | "
                    f"Time: {episode_time:.1f}s"
                )
            else:
                print(
                    f"   Episode {episode+1:2d}/{num_episodes}: {outcome:4s} | "
                    f"Reward: {episode_reward:6.1f} | Length: {episode_length:4d} | "
                    f"Time: {episode_time:.1f}s"
                )

    # --- Final Analysis ---
    env.close()
    total_time = time.time() - start_time

    total_episodes = len(stats["episode_rewards"])
    win_rate = stats["wins"] / total_episodes if total_episodes > 0 else 0
    avg_reward = np.mean(stats["episode_rewards"]) if total_episodes > 0 else 0
    avg_length = np.mean(stats["episode_lengths"]) if total_episodes > 0 else 0

    # Calculate JEPA accuracy averages
    avg_attack_acc = 0
    avg_defense_acc = 0
    if enable_jepa and stats["strategic_stats"]["attack_accuracy"]:
        avg_attack_acc = np.mean(stats["strategic_stats"]["attack_accuracy"])
        avg_defense_acc = np.mean(stats["strategic_stats"]["defense_accuracy"])

    # Print final results
    if verbose:
        print(f"\n📊 EVALUATION RESULTS ({total_time:.1f}s total)")
        print(
            f"   🎯 Win Rate: {win_rate*100:.1f}% ({stats['wins']}W/{stats['losses']}L)"
        )
        print(f"   🏆 Average Reward: {avg_reward:.2f}")
        print(f"   📏 Average Length: {avg_length:.0f} steps")

        if enable_jepa and avg_attack_acc > 0:
            print(f"\n🧠 JEPA STRATEGIC PERFORMANCE:")
            print(f"   ⚔️ Attack Timing Accuracy: {avg_attack_acc:.1f}%")
            print(f"   🛡️ Defense Timing Accuracy: {avg_defense_acc:.1f}%")

    results_dict = {
        "model_path": model_path,
        "num_episodes": num_episodes,
        "win_rate": win_rate,
        "avg_reward": avg_reward,
        "avg_length": avg_length,
        "wins": stats["wins"],
        "losses": stats["losses"],
        "avg_attack_accuracy": avg_attack_acc,
        "avg_defense_accuracy": avg_defense_acc,
        "evaluation_time": total_time,
    }

    return results_dict


def main():
    parser = argparse.ArgumentParser(description="Model Performance Evaluation")
    parser.add_argument("model_path", help="Path to trained model (.zip file)")
    parser.add_argument(
        "--episodes", type=int, default=20, help="Number of evaluation episodes"
    )
    parser.add_argument(
        "--no-jepa",
        action="store_true",
        help="Disable JEPA features (run as standard CNN)",
    )
    parser.add_argument("--render", action="store_true", help="Render the game")
    parser.add_argument("--output", type=str, help="Save results to JSON file")
    parser.add_argument("--quiet", action="store_true", help="Minimal output")

    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        print(f"❌ Model file not found: {args.model_path}")
        return

    # Run evaluation
    results = evaluate_model(
        model_path=args.model_path,
        num_episodes=args.episodes,
        enable_jepa=not args.no_jepa,
        render=args.render,
        verbose=not args.quiet,
    )

    if results is None:
        print("❌ Evaluation failed")
        return

    # Save results if requested
    if args.output:
        # Make results JSON serializable
        serializable_results = {
            k: (float(v) if isinstance(v, np.generic) else v)
            for k, v in results.items()
        }
        with open(args.output, "w") as f:
            json.dump(serializable_results, f, indent=2)
        print(f"\n💾 Results saved to {args.output}")

    # Performance assessment with JEPA considerations
    win_rate = results["win_rate"]
    attack_acc = results.get("avg_attack_accuracy", 0)
    defense_acc = results.get("avg_defense_accuracy", 0)

    if win_rate >= 0.7:
        performance = "🏆 EXCELLENT"
    elif win_rate >= 0.6:
        performance = "🥇 GOOD"
    elif win_rate >= 0.5:
        performance = "🥈 DECENT"
    elif win_rate >= 0.4:
        performance = "🥉 LEARNING"
    else:
        performance = "📚 NEEDS MORE TRAINING"

    print(f"\nPERFORMANCE: {performance} (Win Rate: {win_rate*100:.1f}%)")

    if attack_acc > 0:
        if attack_acc >= 80 and defense_acc >= 60:
            jepa_performance = "🧠 JEPA: EXCELLENT STRATEGIC AWARENESS"
        elif attack_acc >= 70 and defense_acc >= 40:
            jepa_performance = "🧠 JEPA: GOOD STRATEGIC LEARNING"
        elif attack_acc >= 60:
            jepa_performance = "🧠 JEPA: DEVELOPING STRATEGIC SKILLS"
        else:
            jepa_performance = "🧠 JEPA: BASIC STRATEGIC UNDERSTANDING"

        print(jepa_performance)


if __name__ == "__main__":
    main()
