#!/usr/bin/env python3
"""
JEPA-Enhanced Samurai Training Script - FIXED & UPGRADED VERSION

Key Fixes & Upgrades:
- Simplified arguments: JEPA is on by default, use --no-jepa to disable.
- Integrates the new Transformer-based JEPA wrapper.
- Robust training loop with detailed callbacks and VRAM management.
- Eliminates harmful randomness sources for stable training.
"""
import os
import argparse
import time
import torch
import torch.nn as nn
import numpy as np
from collections import deque

# Use stable-retro for gymnasium compatibility
try:
    import stable_retro as retro

    print("🎮 Using stable-retro (gymnasium compatible)")
except ImportError:
    raise ImportError("stable-retro not found. Install with: pip install stable-retro")

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

# Import JEPA-enhanced components from the fixed wrapper
from wrapper import SamuraiJEPAWrapper, JEPAEnhancedCNN


class JEPATrainingCallback(BaseCallback):
    """Callback for monitoring JEPA training performance."""

    def __init__(self, enable_jepa=True, verbose=0):
        super(JEPATrainingCallback, self).__init__(verbose)
        self.enable_jepa = enable_jepa
        self.last_log_time = time.time()

    def _on_step(self) -> bool:
        if time.time() - self.last_log_time > 60:  # Log every 60 seconds
            self.last_log_time = time.time()

            # Aggregate stats from all environments in the VecEnv
            all_stats = self.training_env.get_attr("current_stats")
            wins = sum(s.get("wins", 0) for s in all_stats)
            losses = sum(s.get("losses", 0) for s in all_stats)
            total_rounds = sum(s.get("total_rounds", 0) for s in all_stats)
            win_rate = (wins / total_rounds * 100) if total_rounds > 0 else 0

            print(f"\n--- 📊 JEPA Training Status @ Step {self.num_timesteps:,} ---")
            print(
                f"   🎯 Win Rate: {win_rate:.1f}% ({wins}W / {losses}L in {total_rounds} rounds)"
            )

            if self.enable_jepa:
                print(f"   🧠 Architecture: JEPA-Enhanced CNN + Transformer Predictor")
            else:
                print(f"   🧠 Architecture: Standard Enhanced CNN")

            # Log VRAM usage if on CUDA
            if torch.cuda.is_available():
                vram_alloc = torch.cuda.memory_allocated() / (1024**3)
                vram_res = torch.cuda.memory_reserved() / (1024**3)
                print(
                    f"   💾 VRAM: {vram_alloc:.2f} GB allocated / {vram_res:.2f} GB reserved"
                )

            print(f"   📈 Learning Rate: {self.model.learning_rate:.2e}")
            print("--------------------------------------------------")

        return True


def make_env(game, state, rendering, frame_stack, enable_jepa, rank, seed=0):
    """Utility function for multiprocessed env creation."""

    def _init():
        # Use a different state file per process to avoid conflicts if needed, or None
        env_state = state if rank == 0 else None

        env = retro.make(
            game=game,
            state=env_state,
            use_restricted_actions=retro.Actions.FILTERED,
            obs_type=retro.Observations.IMAGE,
            render_mode="human" if rendering and rank == 0 else None,
        )
        env = SamuraiJEPAWrapper(env, frame_stack=frame_stack, enable_jepa=enable_jepa)
        env = Monitor(env)
        # env.seed(seed + rank) # Seeding is handled by SB3
        return env

    return _init


def main():
    parser = argparse.ArgumentParser(
        description="JEPA Enhanced Fighting Game AI Training"
    )
    parser.add_argument("--total-timesteps", type=int, default=10_000_000)
    parser.add_argument(
        "--n-envs", type=int, default=8, help="Number of parallel environments"
    )
    parser.add_argument(
        "--n-steps", type=int, default=2048, help="PPO rollout buffer size per env"
    )
    parser.add_argument("--batch-size", type=int, default=512, help="PPO batch size")
    parser.add_argument("--lr", type=float, default=2.5e-4, help="Learning rate")
    parser.add_argument(
        "--ent-coef", type=float, default=0.01, help="Entropy coefficient"
    )
    parser.add_argument(
        "--frame-stack", type=int, default=6, help="Number of frames to stack"
    )
    parser.add_argument(
        "--no-jepa",
        action="store_true",
        help="Disable JEPA components and run as a standard CNN agent",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to a model zip file to resume training",
    )
    parser.add_argument(
        "--render", action="store_true", help="Render the first environment"
    )
    args = parser.parse_args()

    args.enable_jepa = not args.no_jepa

    device = "cuda" if torch.cuda.is_available() else "cpu"
    mode_name = "JEPA Enhanced" if args.enable_jepa else "Standard CNN"

    print(f"🚀 Starting {mode_name} Training")
    print(f"   💻 Device: {device.upper()}")
    print(f"    parallèles: {args.n_envs}")
    print(f"   📊 Timesteps: {args.total_timesteps:,}")

    # Environment setup
    game = "SamuraiShodown-Genesis"
    state_file = "samurai.state" if os.path.exists("samurai.state") else None
    if state_file:
        print(f"   🎮 Using state file: {state_file}")
    else:
        print("   🎮 Using default initial state.")

    # Use SubprocVecEnv for true parallelism
    env = SubprocVecEnv(
        [
            make_env(
                game, state_file, args.render, args.frame_stack, args.enable_jepa, i
            )
            for i in range(args.n_envs)
        ]
    )

    save_dir = (
        "trained_models_jepa_transformer" if args.enable_jepa else "trained_models_cnn"
    )
    os.makedirs(save_dir, exist_ok=True)

    # Model configuration
    policy_kwargs = dict(
        features_extractor_class=JEPAEnhancedCNN,
        features_extractor_kwargs=dict(features_dim=512),
        net_arch=dict(pi=[256], vf=[256]),  # Simplified network arch
        activation_fn=nn.ReLU,
    )

    if args.resume and os.path.exists(args.resume):
        print(f"📂 Resuming training from {args.resume}")
        model = PPO.load(
            args.resume,
            env=env,
            device=device,
            custom_objects={"learning_rate": args.lr},
        )
    else:
        print("🚀 Creating new model...")
        model = PPO(
            "CnnPolicy",
            env,
            policy_kwargs=policy_kwargs,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=4,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=args.ent_coef,
            vf_coef=0.5,
            max_grad_norm=0.5,
            learning_rate=args.lr,
            verbose=1,
            tensorboard_log=f"./{save_dir}_logs/",
            device=device,
        )

    # **KEY FOR JEPA**: Give the wrapper access to the model's feature extractor
    if args.enable_jepa:
        for i in range(args.n_envs):
            # This must be done after model creation
            env.env_method(
                "__setattr__",
                "feature_extractor",
                model.policy.features_extractor,
                indices=i,
            )
        print("   ✅ Injected PPO feature extractor into JEPA wrappers.")

    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=max(50000 // args.n_envs, 1),
        save_path=save_dir,
        name_prefix=f"ppo_{'jepa' if args.enable_jepa else 'cnn'}",
    )
    training_callback = JEPATrainingCallback(enable_jepa=args.enable_jepa)

    # Training
    try:
        start_time = time.time()
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=[checkpoint_callback, training_callback],
            reset_num_timesteps=not bool(args.resume),
        )
        training_time = (time.time() - start_time) / 3600
        print(f"🎉 Training completed in {training_time:.2f} hours!")

    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback

        traceback.print_exc()
    finally:
        env.close()

    # Save final model
    final_path = os.path.join(save_dir, "final_model.zip")
    model.save(final_path)
    print(f"💾 Final model saved to {final_path}")


if __name__ == "__main__":
    main()
