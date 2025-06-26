#!/usr/bin/env python3
"""
Quick JEPA Model Performance Evaluation Script
Rapidly assess trained model performance with detailed stats
"""

import os
import sys
import argparse
import time
import torch
import numpy as np
from collections import defaultdict
import json

# Import required modules
try:
    import stable_retro as retro
except ImportError:
    import retro

from stable_baselines3 import PPO
from wrapper import SamuraiJEPAWrapper


def evaluate_model(
    model_path, num_episodes=10, enable_jepa=True, render=False, verbose=True
):
    """
    Quickly evaluate a trained model's performance

    Args:
        model_path: Path to the trained model
        num_episodes: Number of evaluation episodes
        enable_jepa: Whether to enable JEPA features
        render: Whether to render the game
        verbose: Print detailed stats

    Returns:
        dict: Performance statistics
    """

    if verbose:
        print(f"🚀 QUICK MODEL EVALUATION")
        print(f"   📁 Model: {model_path}")
        print(f"   🎮 Episodes: {num_episodes}")
        print(f"   🧠 JEPA: {enable_jepa}")
        print(f"   👁️ Render: {render}")

    # Create environment
    try:
        game = "SamuraiShodown-Genesis"
        state = "samurai.state" if os.path.exists("samurai.state") else None

        env = retro.make(
            game=game,
            state=state,
            use_restricted_actions=retro.Actions.FILTERED,
            obs_type=retro.Observations.IMAGE,
            render_mode="human" if render else None,
        )

        env = SamuraiJEPAWrapper(
            env,
            reset_round=True,
            rendering=render,
            max_episode_steps=5000,  # Shorter for quick eval
            frame_stack=6,
            enable_jepa=enable_jepa,
        )

        if verbose:
            print(f"✅ Environment created successfully")

    except Exception as e:
        print(f"❌ Failed to create environment: {e}")
        return None

    # Load model
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = PPO.load(model_path, env=env, device=device)
        if verbose:
            print(f"✅ Model loaded on {device}")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        env.close()
        return None

    # Evaluation metrics
    stats = {
        "episodes": 0,
        "wins": 0,
        "losses": 0,
        "draws": 0,
        "total_rounds": 0,
        "total_reward": 0.0,
        "episode_rewards": [],
        "episode_lengths": [],
        "win_rate": 0.0,
        "avg_reward": 0.0,
        "avg_length": 0.0,
        "damage_dealt": 0,
        "damage_taken": 0,
        "damage_efficiency": 0.0,
    }

    # JEPA-specific stats
    if enable_jepa:
        stats.update(
            {
                "jepa_predictions": 0,
                "attack_accuracy": 0.0,
                "damage_accuracy": 0.0,
                "overall_accuracy": 0.0,
                "successful_responses": 0,
                "total_responses": 0,
            }
        )

    start_time = time.time()

    # Run evaluation episodes
    for episode in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0.0
        episode_length = 0
        round_count = 0

        episode_start = time.time()

        while True:
            # Get action from model (deterministic for evaluation)
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)

            episode_reward += reward
            episode_length += 1

            # Track round endings
            if info.get("round_ended", False):
                round_count += 1

            if done or truncated:
                break

        episode_time = time.time() - episode_start

        # Extract final game info
        game_info = info.get("game_info", {})
        current_stats = info.get("current_stats", {})
        strategic_stats = info.get("strategic_stats", {})

        # Determine episode outcome
        player_health = game_info.get("player_health", 0)
        opponent_health = game_info.get("opponent_health", 0)

        if player_health > opponent_health:
            stats["wins"] += 1
            outcome = "WIN"
        elif opponent_health > player_health:
            stats["losses"] += 1
            outcome = "LOSS"
        else:
            stats["draws"] += 1
            outcome = "DRAW"

        # Update stats
        stats["episodes"] += 1
        stats["total_rounds"] += round_count
        stats["total_reward"] += episode_reward
        stats["episode_rewards"].append(episode_reward)
        stats["episode_lengths"].append(episode_length)

        # Health tracking
        stats["damage_dealt"] += current_stats.get("total_damage_dealt", 0)
        stats["damage_taken"] += current_stats.get("total_damage_taken", 0)

        # JEPA stats
        if enable_jepa and strategic_stats:
            stats["jepa_predictions"] += strategic_stats.get(
                "binary_predictions_made", 0
            )
            stats["attack_accuracy"] = strategic_stats.get(
                "attack_prediction_accuracy", 0.0
            )
            stats["damage_accuracy"] = strategic_stats.get(
                "damage_prediction_accuracy", 0.0
            )
            stats["overall_accuracy"] = strategic_stats.get(
                "overall_prediction_accuracy", 0.0
            )
            stats["successful_responses"] += strategic_stats.get(
                "successful_responses", 0
            )
            stats["total_responses"] += strategic_stats.get("total_responses", 0)

        if verbose:
            print(
                f"   Episode {episode+1:2d}/{num_episodes}: {outcome:4s} | "
                f"Reward: {episode_reward:6.1f} | Length: {episode_length:4d} | "
                f"Time: {episode_time:.1f}s"
            )

    # Calculate final statistics
    total_time = time.time() - start_time

    if stats["episodes"] > 0:
        stats["win_rate"] = stats["wins"] / stats["episodes"]
        stats["avg_reward"] = stats["total_reward"] / stats["episodes"]
        stats["avg_length"] = np.mean(stats["episode_lengths"])

        if stats["damage_taken"] > 0:
            stats["damage_efficiency"] = stats["damage_dealt"] / stats["damage_taken"]

    # Close environment
    env.close()

    # Print final results
    if verbose:
        print(f"\n📊 EVALUATION RESULTS ({total_time:.1f}s total)")
        print(
            f"   🎯 Win Rate: {stats['win_rate']*100:.1f}% ({stats['wins']}W/{stats['losses']}L/{stats['draws']}D)"
        )
        print(f"   🏆 Average Reward: {stats['avg_reward']:.1f}")
        print(f"   📏 Average Length: {stats['avg_length']:.0f} steps")
        print(f"   🎮 Total Rounds: {stats['total_rounds']}")
        print(f"   ⚔️ Damage Efficiency: {stats['damage_efficiency']:.2f}")

        if enable_jepa and stats["jepa_predictions"] > 0:
            print(f"   🔮 JEPA Predictions: {stats['jepa_predictions']}")
            print(f"   🎯 Attack Accuracy: {stats['attack_accuracy']*100:.1f}%")
            print(f"   💥 Damage Accuracy: {stats['damage_accuracy']*100:.1f}%")
            print(f"   📊 Overall Accuracy: {stats['overall_accuracy']*100:.1f}%")
            print(
                f"   ⚡ Response Success: {stats['successful_responses']}/{stats['total_responses']}"
            )

        print(f"   ⏱️ Episodes/min: {stats['episodes']/(total_time/60):.1f}")

    return stats


def main():
    parser = argparse.ArgumentParser(description="Quick Model Performance Evaluation")
    parser.add_argument("model_path", help="Path to trained model (.zip file)")
    parser.add_argument(
        "--episodes", type=int, default=10, help="Number of evaluation episodes"
    )
    parser.add_argument(
        "--disable-jepa", action="store_true", help="Disable JEPA features"
    )
    parser.add_argument("--render", action="store_true", help="Render the game")
    parser.add_argument("--output", type=str, help="Save results to JSON file")
    parser.add_argument("--quiet", action="store_true", help="Minimal output")

    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        print(f"❌ Model file not found: {args.model_path}")
        return

    enable_jepa = not args.disable_jepa
    verbose = not args.quiet

    # Run evaluation
    results = evaluate_model(
        model_path=args.model_path,
        num_episodes=args.episodes,
        enable_jepa=enable_jepa,
        render=args.render,
        verbose=verbose,
    )

    if results is None:
        print("❌ Evaluation failed")
        return

    # Save results if requested
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"💾 Results saved to {args.output}")

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
        performance = "📚 NEEDS TRAINING"

    print(f"\n{performance} (Win Rate: {win_rate*100:.1f}%)")


if __name__ == "__main__":
    main()
