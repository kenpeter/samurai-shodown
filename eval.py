#!/usr/bin/env python3
"""
JEPA Model Performance Evaluation Script - CORRECTED VERSION
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
from wrapper import SamuraiJEPAWrapper


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

    # --- Create Environment (with corrected state path) ---
    try:
        game = "SamuraiShodown-Genesis"

        # **FIX 1: Use absolute path for the state file**
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

        env = SamuraiJEPAWrapper(
            env,
            frame_stack=6,
            enable_jepa=enable_jepa,
        )
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

    # **FIX 2: Inject the feature extractor into the wrapper if JEPA is enabled**
    if enable_jepa:
        # The environment object is not a VecEnv, so we access the wrapper directly
        env.unwrapped.feature_extractor = model.policy.features_extractor
        if verbose:
            print("   💉 Injected PPO feature extractor into JEPA wrapper.")

    # --- Evaluation Loop ---
    stats = {
        "wins": 0,
        "losses": 0,
        "episode_rewards": [],
        "episode_lengths": [],
    }

    start_time = time.time()
    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        episode_length = 0

        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)

            episode_reward += reward
            episode_length += 1
            if done or truncated:
                break

        # Update stats from the final info dict
        stats["episode_rewards"].append(episode_reward)
        stats["episode_lengths"].append(episode_length)

        # The wrapper's `current_stats` holds the win/loss record
        final_stats = info.get("current_stats", {})
        # Note: We aggregate wins/losses after all episodes are done for simplicity

        if verbose:
            outcome = "Unknown"
            # Determine outcome based on health from the final step's info
            if info.get("game_info", {}).get("player_health", 0) > info.get(
                "game_info", {}
            ).get("opponent_health", 0):
                outcome = "WIN"
                stats["wins"] += 1
            else:
                outcome = "LOSS"
                stats["losses"] += 1
            print(
                f"   Episode {episode+1:2d}/{num_episodes}: {outcome:4s} | Reward: {episode_reward:6.1f} | Length: {episode_length:4d}"
            )

    # --- Final Analysis ---
    env.close()
    total_time = time.time() - start_time

    total_episodes = len(stats["episode_rewards"])
    win_rate = stats["wins"] / total_episodes if total_episodes > 0 else 0
    avg_reward = np.mean(stats["episode_rewards"]) if total_episodes > 0 else 0
    avg_length = np.mean(stats["episode_lengths"]) if total_episodes > 0 else 0

    # Print final results
    if verbose:
        print(f"\n📊 EVALUATION RESULTS ({total_time:.1f}s total)")
        print(
            f"   🎯 Win Rate: {win_rate*100:.1f}% ({stats['wins']}W/{stats['losses']}L)"
        )
        print(f"   🏆 Average Reward: {avg_reward:.2f}")
        print(f"   📏 Average Length: {avg_length:.0f} steps")

    results_dict = {
        "model_path": model_path,
        "num_episodes": num_episodes,
        "win_rate": win_rate,
        "avg_reward": avg_reward,
        "avg_length": avg_length,
        "wins": stats["wins"],
        "losses": stats["losses"],
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

    # Quick performance assessment
    win_rate = results["win_rate"]
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


if __name__ == "__main__":
    main()
