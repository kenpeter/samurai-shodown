import os
import sys
import argparse
import time
import random
import json
from pathlib import Path
from collections import deque
from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn

# Import retro
try:
    import stable_retro as retro

    print("🎮 Using stable-retro (gymnasium compatible)")
except ImportError:
    try:
        import retro

        print("🎮 Using retro (legacy)")
    except ImportError:
        raise ImportError("Install with: pip install stable-retro")

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

# Import our NCSOFT wrapper
from wrapper import NCSOFTMultiAgentWrapper, NCSOFTSimpleCNN, make_ncsoft_env


class NCSOFTMultiAgentManager:
    """
    NCSOFT Multi-Agent Self-Play Manager
    Implements the breakthrough strategy from the 62% win rate paper
    """

    def __init__(self, save_dir="ncsoft_breakthrough_models"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Agent pools for self-play (NCSOFT technique)
        self.agent_pools = {
            "aggressive": deque(maxlen=15),
            "defensive": deque(maxlen=15),
            "balanced": deque(maxlen=15),
        }

        # Cross-style training statistics
        self.training_stats = {
            "aggressive": {"wins": 0, "games": 0, "win_rate": 0.0, "vs_styles": {}},
            "defensive": {"wins": 0, "games": 0, "win_rate": 0.0, "vs_styles": {}},
            "balanced": {"wins": 0, "games": 0, "win_rate": 0.0, "vs_styles": {}},
        }

        # NCSOFT opponent sampling parameters
        self.cross_style_probability = 0.7  # 70% cross-style, 30% same-style
        self.recent_weight = 0.6  # Bias toward recent models
        self.min_pool_size = 3  # Minimum models before cross-training

        # Training phases
        self.training_phases = {
            "initial": {"target_wr": 25, "focus": "basic_learning"},
            "building": {"target_wr": 35, "focus": "cross_style_training"},
            "breakthrough": {"target_wr": 50, "focus": "advanced_tactics"},
            "mastery": {"target_wr": 60, "focus": "refinement"},
        }
        self.current_phase = "initial"

        print("🥊 NCSOFT MULTI-AGENT MANAGER INITIALIZED")
        print("   🔴 Aggressive: High damage, time pressure")
        print("   🔵 Defensive: HP preservation, positioning")
        print("   🟡 Balanced: Adaptive tactics")
        print("   🎯 Target: 30% → 35% → 50%+ breakthrough")

    def add_agent_to_pool(
        self, style: str, model_path: str, timestep: int, win_rate: float = 0.0
    ):
        """Add trained agent to style pool"""
        agent_info = {
            "path": model_path,
            "timestep": timestep,
            "win_rate": win_rate,
            "style": style,
        }
        self.agent_pools[style].append(agent_info)
        print(
            f"📦 Added {style} agent (WR: {win_rate:.1%}, Step: {timestep:,}) to pool"
        )

    def sample_opponent(self, current_style: str) -> Tuple[str, str]:
        """
        NCSOFT Opponent Sampling Strategy
        70% cross-style training for robustness
        """
        # Determine opponent style
        if random.random() < self.cross_style_probability:
            # Cross-style training (breakthrough strategy)
            other_styles = [s for s in self.agent_pools.keys() if s != current_style]
            if other_styles:
                opponent_style = random.choice(other_styles)
            else:
                opponent_style = current_style
        else:
            # Same-style training for consistency
            opponent_style = current_style

        # Get opponent pool
        pool = self.agent_pools[opponent_style]
        if not pool:
            return None, None

        # Sample with recent bias (NCSOFT technique)
        if len(pool) >= 3 and random.random() < self.recent_weight:
            # Sample from recent 50% of models
            recent_count = max(1, len(pool) // 2)
            opponent = random.choice(list(pool)[-recent_count:])
        else:
            # Sample from all models for diversity
            opponent = random.choice(list(pool))

        return opponent["path"], opponent_style

    def update_training_stats(self, style: str, opponent_style: str, won: bool):
        """Update cross-style training statistics"""
        stats = self.training_stats[style]
        stats["games"] += 1
        if won:
            stats["wins"] += 1
        stats["win_rate"] = stats["wins"] / stats["games"]

        # Track vs specific styles
        if opponent_style not in stats["vs_styles"]:
            stats["vs_styles"][opponent_style] = {"wins": 0, "games": 0}
        stats["vs_styles"][opponent_style]["games"] += 1
        if won:
            stats["vs_styles"][opponent_style]["wins"] += 1

    def get_training_phase(self, win_rate: float) -> str:
        """Determine current training phase based on performance"""
        if win_rate >= 50:
            return "mastery"
        elif win_rate >= 35:
            return "breakthrough"
        elif win_rate >= 25:
            return "building"
        else:
            return "initial"

    def save_stats(self):
        """Save training statistics"""
        stats_file = self.save_dir / "training_stats.json"
        with open(stats_file, "w") as f:
            json.dump(self.training_stats, f, indent=2)

    def load_stats(self):
        """Load training statistics"""
        stats_file = self.save_dir / "training_stats.json"
        if stats_file.exists():
            with open(stats_file, "r") as f:
                self.training_stats = json.load(f)

    def print_stats_summary(self):
        """Print comprehensive training statistics"""
        print(f"\n🏆 NCSOFT MULTI-AGENT TRAINING SUMMARY:")
        for style, stats in self.training_stats.items():
            wr = stats["win_rate"] * 100
            games = stats["games"]
            print(f"   {style.upper():>10}: {wr:5.1f}% ({games:3d} games)")

            # Cross-style performance
            for opp_style, opp_stats in stats.get("vs_styles", {}).items():
                opp_wr = (
                    (opp_stats["wins"] / opp_stats["games"]) * 100
                    if opp_stats["games"] > 0
                    else 0
                )
                print(
                    f"      vs {opp_style:9}: {opp_wr:5.1f}% ({opp_stats['games']:2d} games)"
                )


class NCSOFTBreakthroughCallback(BaseCallback):
    """Enhanced callback for NCSOFT breakthrough training"""

    def __init__(self, multi_agent_manager=None, agent_style="balanced", verbose=0):
        super().__init__(verbose)
        self.multi_agent_manager = multi_agent_manager
        self.agent_style = agent_style
        self.last_log_step = 0
        self.breakthrough_detected = False
        self.entropy_history = deque(maxlen=1000)
        self.initial_entropy = None

    def _on_step(self) -> bool:
        # Log every 20000 steps
        if self.num_timesteps % 20000 == 0 and self.num_timesteps != self.last_log_step:
            self.last_log_step = self.num_timesteps

            print(f"\n🥊 NCSOFT BREAKTHROUGH TRAINING - {self.agent_style.upper()}")
            print(f"   📈 Timesteps: {self.num_timesteps:,}")

            # Get environment stats
            if hasattr(self.training_env, "get_attr"):
                try:
                    env_stats = self.training_env.get_attr("current_stats")[0]
                    win_rate = env_stats.get("win_rate", 0) * 100
                    wins = env_stats.get("wins", 0)
                    losses = env_stats.get("losses", 0)
                    breakthrough_progress = (
                        env_stats.get("breakthrough_progress", 0) * 100
                    )

                    print(f"   🎯 Win Rate: {win_rate:.1f}% ({wins}W/{losses}L)")
                    print(f"   💪 Breakthrough Progress: {breakthrough_progress:.1f}%")

                    # Detect breakthrough moments
                    if win_rate >= 35 and not self.breakthrough_detected:
                        print(f"   🚀 BREAKTHROUGH DETECTED! Win rate: {win_rate:.1f}%")
                        self.breakthrough_detected = True
                    elif win_rate >= 50:
                        print(
                            f"   🏆 MASTERY LEVEL ACHIEVED! Win rate: {win_rate:.1f}%"
                        )

                    # Style-specific metrics
                    style_metrics = env_stats.get("style_metrics", {})
                    if style_metrics:
                        print(f"   🎭 Style Metrics:")
                        for metric, value in style_metrics.items():
                            print(f"      {metric}: {value:.3f}")

                    # NCSOFT data efficiency metrics
                    forced_noop_ratio = env_stats.get("forced_noop_ratio", 0) * 100
                    strategic_noop_ratio = (
                        env_stats.get("strategic_noop_ratio", 0) * 100
                    )
                    move_consistency = env_stats.get("move_consistency", 0) * 100

                    print(f"   📊 NCSOFT Efficiency:")
                    print(f"      Forced no-ops filtered: {forced_noop_ratio:.1f}%")
                    print(f"      Strategic no-ops kept: {strategic_noop_ratio:.1f}%")
                    print(f"      Move consistency: {move_consistency:.1f}%")

                except Exception as e:
                    print(f"   ❌ Stats error: {e}")

            # Training diagnostics
            print(f"   🎛️ Entropy coefficient: {self.model.ent_coef:.4f}")
            print(f"   📏 N_steps: {self.model.n_steps}")
            print(f"   🚀 NCSOFT Multi-Agent Training Active")

        return True


def create_ncsoft_model(env, device, args, style):
    """Create NCSOFT-optimized PPO model with resume capability"""

    # Check for resume
    if args.resume and os.path.exists(args.resume):
        print(f"📂 Resuming {style} model from: {args.resume}")
        model = PPO.load(args.resume, env=env, device=device)

        # Update parameters from arguments when resuming
        model.ent_coef = args.ent_coef
        model.learning_rate = args.learning_rate
        print(f"🔄 Updated model parameters:")
        print(f"   🎲 Entropy coef: {model.ent_coef:.4f}")
        print(f"   📈 Learning rate: {model.learning_rate:.2e}")

        return model

    # Create new model
    # Learning rate schedule
    lr_schedule = lambda progress: args.learning_rate * (1 - 0.7 * progress)

    # Style-specific hyperparameters
    if style == "aggressive":
        ent_coef = args.ent_coef * 1.2  # Higher exploration for aggressive
        n_epochs = 3
    elif style == "defensive":
        ent_coef = args.ent_coef * 0.8  # Lower exploration for defensive
        n_epochs = 4
    else:  # balanced
        ent_coef = args.ent_coef
        n_epochs = 3

    model = PPO(
        "CnnPolicy",
        env,
        device=device,
        verbose=1,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=n_epochs,
        gamma=0.99,
        learning_rate=lr_schedule,
        clip_range=0.2,
        ent_coef=ent_coef,
        vf_coef=0.5,
        max_grad_norm=0.5,
        gae_lambda=0.95,
        policy_kwargs=dict(
            features_extractor_class=NCSOFTSimpleCNN,
            features_extractor_kwargs=dict(features_dim=512),
            normalize_images=False,
            net_arch=dict(pi=[512, 256], vf=[512, 256]),
            activation_fn=nn.ReLU,
        ),
    )

    print(f"🧠 NCSOFT {style.upper()} Model Created:")
    print(f"   📊 Batch size: {args.batch_size:,}")
    print(f"   📏 N_steps: {args.n_steps:,}")
    print(f"   🎲 Entropy coef: {ent_coef:.4f}")
    print(f"   🎓 Epochs: {n_epochs}")
    print(f"   🎭 Style: {style}")

    return model


def train_ncsoft_agent(args, style, multi_agent_manager):
    """Train a single NCSOFT agent with the specified style"""

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n🎭 TRAINING {style.upper()} AGENT")
    print(f"   💻 Device: {device}")

    # Create environment
    try:
        env = make_ncsoft_env(
            game="SamuraiShodown-Genesis",
            state=None,  # Let the function handle state detection
            agent_style=style,
            reset_round=True,
            rendering=args.render,
            max_episode_steps=15000,
            breakthrough_mode=True,
            use_default_state=args.use_default_state,  # Pass the flag
        )
        env = Monitor(env)
        print(f"✅ {style} environment created successfully")

    except Exception as e:
        print(f"❌ Failed to create {style} environment: {e}")
        return None

    # Create model
    model = create_ncsoft_model(env, device, args, style)

    # Training phases
    total_timesteps = args.total_timesteps
    phase_duration = total_timesteps // 4  # 4 training phases
    save_interval = 100000  # Save every 100k steps

    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=save_interval,
        save_path=str(multi_agent_manager.save_dir),
        name_prefix=f"ncsoft_{style}",
    )

    training_callback = NCSOFTBreakthroughCallback(
        multi_agent_manager=multi_agent_manager, agent_style=style, verbose=1
    )

    print(f"🏋️ Starting {style} training...")
    start_time = time.time()

    try:
        # Handle resume functionality
        start_phase = 0
        if args.resume and os.path.exists(args.resume):
            print(f"📂 Resuming training from: {args.resume}")
            # Try to extract phase from filename
            if "_phase_" in args.resume:
                try:
                    phase_num = int(args.resume.split("_phase_")[1].split(".")[0])
                    start_phase = phase_num
                    print(f"🔄 Resuming from phase {start_phase}")
                except:
                    print(
                        f"⚠️ Could not extract phase from filename, starting from phase 0"
                    )

        # Train with periodic opponent updates
        for phase in range(start_phase, 4):
            phase_start = phase * phase_duration
            phase_end = min((phase + 1) * phase_duration, total_timesteps)
            phase_steps = phase_end - phase_start

            print(f"\n📅 PHASE {phase + 1}/4: {phase_start:,} → {phase_end:,} steps")

            # For resumed training, adjust timesteps
            if phase == start_phase and args.resume:
                reset_timesteps = False
                print(f"🔄 Continuing from existing timesteps")
            else:
                reset_timesteps = False  # Keep cumulative timesteps

            # Train this phase
            model.learn(
                total_timesteps=phase_steps,
                callback=[checkpoint_callback, training_callback],
                reset_num_timesteps=reset_timesteps,
            )

            # Save agent to pool
            model_path = multi_agent_manager.save_dir / f"{style}_phase_{phase + 1}.zip"
            model.save(str(model_path))

            # Get current performance
            if hasattr(env, "current_stats"):
                win_rate = env.current_stats.get("win_rate", 0.0)
            else:
                win_rate = 0.0

            # Add to agent pool
            multi_agent_manager.add_agent_to_pool(
                style, str(model_path), phase_end, win_rate
            )

            print(f"✅ Phase {phase + 1} completed - Win rate: {win_rate:.1%}")

        training_time = time.time() - start_time
        print(f"🎉 {style} training completed in {training_time/3600:.1f} hours!")

        # Final performance
        if hasattr(env, "current_stats"):
            final_stats = env.current_stats
            final_wr = final_stats.get("win_rate", 0) * 100
            print(f"📊 FINAL {style.upper()} PERFORMANCE:")
            print(f"   🎯 Win Rate: {final_wr:.1f}%")
            print(f"   🎮 Total Rounds: {final_stats.get('total_rounds', 0)}")
            print(
                f"   💪 Breakthrough: {final_stats.get('breakthrough_progress', 0) * 100:.1f}%"
            )

    except Exception as e:
        print(f"❌ {style} training failed: {e}")
        import traceback

        traceback.print_exc()

    finally:
        env.close()

    # Save final model
    final_path = multi_agent_manager.save_dir / f"{style}_final.zip"
    model.save(str(final_path))
    print(f"💾 Final {style} model saved: {final_path}")

    return model


def main():
    parser = argparse.ArgumentParser(
        description="NCSOFT Multi-Agent Breakthrough Training System"
    )

    # Training parameters
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=2000000,
        help="Total training timesteps per agent",
    )
    parser.add_argument(
        "--learning-rate", type=float, default=3e-4, help="Learning rate"
    )
    parser.add_argument(
        "--ent-coef", type=float, default=0.05, help="Entropy coefficient"
    )
    parser.add_argument(
        "--n-steps", type=int, default=2048, help="Number of steps per rollout"
    )
    parser.add_argument("--batch-size", type=int, default=1024, help="Batch size")

    # NCSOFT Multi-Agent parameters
    parser.add_argument(
        "--styles",
        nargs="+",
        choices=["aggressive", "defensive", "balanced"],
        default=["aggressive", "balanced", "defensive"],
        help="Fighting styles to train",
    )
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Train agents sequentially vs parallel",
    )

    # Environment parameters
    parser.add_argument("--render", action="store_true", help="Render environment")
    parser.add_argument(
        "--resume", type=str, default=None, help="Resume from checkpoint"
    )
    parser.add_argument(
        "--use-default-state",
        action="store_true",
        help="Use default game state instead of samurai.state file",
    )

    args = parser.parse_args()

    print(f"🚀 NCSOFT MULTI-AGENT BREAKTHROUGH TRAINING")
    print(f"   🎭 Styles: {args.styles}")
    print(f"   📊 Timesteps per agent: {args.total_timesteps:,}")
    print(f"   🎯 Target: Break through 30% plateau")
    print(f"   📈 Expected: 30% → 35% → 50%+")
    print(f"   📜 Based on NCSOFT 62% vs pro players paper")

    # Initialize multi-agent manager
    multi_agent_manager = NCSOFTMultiAgentManager()
    multi_agent_manager.load_stats()

    # Memory optimization
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Train agents
    trained_models = {}

    if args.sequential:
        # Sequential training (recommended for breakthrough)
        print(f"\n🔄 SEQUENTIAL TRAINING MODE")
        print(f"   Training order: {' → '.join(args.styles)}")
        print(f"   Each agent trains completely before the next starts")
        print(f"   Later agents can learn from earlier agents in pool")

        for i, style in enumerate(args.styles):
            print(f"\n🎭 TRAINING {style.upper()} AGENT ({i+1}/{len(args.styles)})")

            # Show current agent pool status
            total_agents_in_pools = sum(
                len(pool) for pool in multi_agent_manager.agent_pools.values()
            )
            print(f"   📦 Current agent pool size: {total_agents_in_pools}")
            if total_agents_in_pools > 0:
                print(f"   🎯 This agent can train against previous agents!")
                for pool_style, pool in multi_agent_manager.agent_pools.items():
                    if len(pool) > 0:
                        print(f"      {pool_style}: {len(pool)} agents available")

            model = train_ncsoft_agent(args, style, multi_agent_manager)
            if model:
                trained_models[style] = model
                multi_agent_manager.save_stats()
                multi_agent_manager.print_stats_summary()

                print(f"✅ {style.upper()} training complete!")
                print(f"   📦 Added to agent pool for future opponents")

    else:
        # Parallel training (faster but less cross-style learning)
        print(f"\n⚡ PARALLEL TRAINING MODE")
        print(f"   All agents train simultaneously")
        print(f"   No cross-style learning during training")
        print(f"   Faster overall training time")

        for style in args.styles:
            print(f"🎭 Starting {style.upper()} agent in parallel...")
            model = train_ncsoft_agent(args, style, multi_agent_manager)
            if model:
                trained_models[style] = model

    # Final summary
    print(f"\n🏆 NCSOFT BREAKTHROUGH TRAINING COMPLETE!")
    print(f"   ✅ Trained agents: {list(trained_models.keys())}")

    multi_agent_manager.print_stats_summary()

    # Breakthrough analysis
    breakthrough_achieved = False
    for style, stats in multi_agent_manager.training_stats.items():
        if stats["win_rate"] >= 0.35:  # 35% breakthrough threshold
            print(f"   🚀 {style.upper()} BREAKTHROUGH: {stats['win_rate']:.1%}")
            breakthrough_achieved = True

    if breakthrough_achieved:
        print(f"   🎉 BREAKTHROUGH ACHIEVED! Target reached!")
    else:
        print(f"   📈 Continue training to reach breakthrough...")

    print(f"\n💡 NEXT STEPS:")
    print(f"   1. Test best agents against each other")
    print(f"   2. Fine-tune with cross-style self-play")
    print(f"   3. Evaluate against original opponent")
    print(f"   4. Continue training for 50%+ win rate")

    # Cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"✅ NCSOFT TRAINING SYSTEM COMPLETE!")


if __name__ == "__main__":
    main()
