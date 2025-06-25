#!/usr/bin/env python3
"""
JEPA-Enhanced Samurai Training Script - FIXED VERSION
Key fix: Removed torch.randn bug on line 215
"""

import os
import sys
import argparse
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import psutil
from typing import Dict, Any, Optional, Type, Union, List
from collections import deque

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
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.policies import ActorCriticCnnPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv

# Import JEPA-enhanced components
from wrapper import (
    SamuraiJEPAWrapper,
    JEPAEnhancedCNN,
    JEPASimpleBinaryPredictor,
    SimpleBinaryResponsePlanner,
)


class JEPATrainingCallback(BaseCallback):
    """
    Enhanced callback for JEPA training with comprehensive monitoring
    """

    def __init__(self, enable_jepa=True, verbose=0):
        super(JEPATrainingCallback, self).__init__(verbose)
        self.enable_jepa = enable_jepa
        self.last_stats_log = 0

        # Training metrics
        self.prediction_accuracy_history = deque(maxlen=1000)

        # JEPA-specific metrics
        self.jepa_metrics = {
            "total_binary_predictions": 0,
            "binary_prediction_accuracy": 0,
            "response_planning_attempts": 0,
            "successful_responses": 0,
            "avg_prediction_confidence": 0.0,
        }

    def _on_step(self) -> bool:
        # Enhanced logging every 10000 steps for JEPA analysis
        if (
            self.num_timesteps % 10000 == 0
            and self.num_timesteps != self.last_stats_log
        ):
            self.last_stats_log = self.num_timesteps

            print(f"\n📊 JEPA TRAINING - Step {self.num_timesteps:,}")

            # Memory monitoring
            if torch.cuda.is_available():
                current_vram = torch.cuda.memory_allocated() / (1024**3)
                max_vram = torch.cuda.max_memory_allocated() / (1024**3)
                free_vram = torch.cuda.mem_get_info()[0] / (1024**3)
                total_vram = torch.cuda.mem_get_info()[1] / (1024**3)

                print(f"   💾 VRAM: {current_vram:.1f}GB / {max_vram:.1f}GB peak")
                print(f"   💾 Free: {free_vram:.1f}GB / {total_vram:.1f}GB total")

            # Get enhanced training stats
            if hasattr(self.training_env, "get_attr"):
                try:
                    # Get environment wrapper attributes directly
                    envs = self.training_env.get_attr("current_stats")
                    env_stats = envs[0] if envs and len(envs) > 0 else {}

                    # Get strategic stats
                    strategic_envs = self.training_env.get_attr("strategic_stats")
                    strategic_stats = (
                        strategic_envs[0]
                        if strategic_envs and len(strategic_envs) > 0
                        else {}
                    )

                    # Basic performance
                    win_rate = env_stats.get("win_rate", 0) * 100 if env_stats else 0
                    wins = env_stats.get("wins", 0) if env_stats else 0
                    losses = env_stats.get("losses", 0) if env_stats else 0
                    total_rounds = env_stats.get("total_rounds", 0) if env_stats else 0

                    print(f"   🎯 Win Rate: {win_rate:.1f}% ({wins}W/{losses}L)")
                    print(f"   🎮 Total Rounds: {total_rounds}")

                    # JEPA-specific stats (updated for binary predictions)
                    if self.enable_jepa and strategic_stats:
                        binary_predictions = strategic_stats.get(
                            "binary_predictions_made", 0
                        )
                        attack_accuracy = (
                            strategic_stats.get("attack_prediction_accuracy", 0) * 100
                        )
                        damage_accuracy = (
                            strategic_stats.get("damage_prediction_accuracy", 0) * 100
                        )
                        overall_accuracy = (
                            strategic_stats.get("overall_prediction_accuracy", 0) * 100
                        )
                        successful_responses = strategic_stats.get(
                            "successful_responses", 0
                        )
                        total_responses = strategic_stats.get("total_responses", 0)

                        print(f"   🔮 Binary predictions: {binary_predictions}")
                        print(f"   🎯 Attack accuracy: {attack_accuracy:.1f}%")
                        print(f"   💥 Damage accuracy: {damage_accuracy:.1f}%")
                        print(f"   📊 Overall accuracy: {overall_accuracy:.1f}%")
                        print(
                            f"   ⚔️ Response success: {successful_responses}/{total_responses}"
                        )

                        # Update JEPA metrics
                        self.jepa_metrics["total_binary_predictions"] = (
                            binary_predictions
                        )
                        if overall_accuracy > 0:
                            self.jepa_metrics["binary_prediction_accuracy"] = (
                                overall_accuracy / 100
                            )
                        self.jepa_metrics["response_planning_attempts"] = (
                            total_responses
                        )
                        self.jepa_metrics["successful_responses"] = successful_responses

                    # Model parameters
                    print(f"   🎛️ Entropy coefficient: {self.model.ent_coef:.4f}")
                    print(f"   📏 N_steps: {self.model.n_steps}")

                    if self.enable_jepa:
                        print(f"   🧠 JEPA + Enhanced CNN")
                        print(f"   🎮 Strategic AI with 6-frame binary prediction")
                    else:
                        print(f"   🧠 Enhanced CNN")

                except Exception as e:
                    # Simplified fallback stats
                    print(f"   🎯 Win Rate: Learning... (early training)")
                    print(f"   🏆 Record: Rounds in progress")
                    print(f"   🎛️ Entropy coefficient: {self.model.ent_coef:.4f}")
                    print(f"   📏 N_steps: {self.model.n_steps}")
                    if self.enable_jepa:
                        print(f"   🧠 JEPA + Enhanced CNN")
                        print(f"   🔮 6-frame binary strategic prediction")
                    else:
                        print(f"   🧠 Enhanced CNN")

            # Learning rate adaptation suggestions
            if self.num_timesteps > 50000:
                if (
                    self.enable_jepa
                    and self.jepa_metrics["total_binary_predictions"] > 100
                ):
                    binary_accuracy = self.jepa_metrics["binary_prediction_accuracy"]
                    if binary_accuracy < 0.55:
                        print(
                            f"   💡 Low binary prediction accuracy ({binary_accuracy:.2f}) - consider increasing entropy"
                        )
                    elif binary_accuracy > 0.75:
                        print(
                            f"   🎯 High binary prediction accuracy ({binary_accuracy:.2f}) - excellent opponent modeling!"
                        )

        return True

    def _on_training_end(self) -> None:
        """Print final JEPA analysis"""
        print(f"\n🎉 JEPA TRAINING COMPLETED!")

        if self.enable_jepa:
            print(f"\n📊 FINAL JEPA BINARY PREDICTION METRICS:")
            for key, value in self.jepa_metrics.items():
                if isinstance(value, float):
                    print(f"   {key}: {value:.3f}")
                else:
                    print(f"   {key}: {value}")


def calculate_jepa_batch_size(obs_shape, target_vram_gb=11.6, enable_jepa=True):
    """Calculate optimal batch size considering JEPA overhead"""
    num_frames, height, width = obs_shape
    obs_size_bytes = num_frames * height * width * 4
    obs_size_mb = obs_size_bytes / (1024 * 1024)

    print(f"📊 JEPA BATCH CALCULATION:")
    print(f"   GPU: {target_vram_gb:.1f} GB")
    print(f"   Obs per sample: {obs_size_mb:.2f} MB")
    print(f"   JEPA enabled: {enable_jepa}")

    # Base CNN memory overhead
    base_model_overhead = 0.4  # Enhanced CNN

    # JEPA adds additional overhead
    jepa_overhead = 0.6 if enable_jepa else 0.0  # JEPA predictor + planner

    total_model_overhead = base_model_overhead + jepa_overhead
    activation_multiplier = 2.0 if enable_jepa else 1.5  # Higher for JEPA processing

    # Calculate memory usage
    memory_per_sample = obs_size_bytes / (1024**3)
    total_overhead = total_model_overhead + 1.2  # Safety buffer

    # Available memory for batch
    available_for_batch = target_vram_gb - total_overhead
    max_batch_size = int(
        available_for_batch / (memory_per_sample * activation_multiplier)
    )

    # JEPA works well with medium-large batches
    if enable_jepa:
        target_batches = [512, 768, 1024, 1536, 2048]
    else:
        target_batches = [1024, 1536, 2048, 3072, 4096]

    final_batch = max([b for b in target_batches if b <= max_batch_size], default=512)

    estimated_usage = (
        total_overhead + final_batch * memory_per_sample * activation_multiplier
    )

    print(f"   🎯 Optimal batch size: {final_batch:,}")
    print(f"   📊 Estimated VRAM: {estimated_usage:.1f} GB")
    if enable_jepa:
        print(f"   🧠 JEPA overhead: {jepa_overhead:.1f} GB")
        print(f"   ⚔️ Strategic AI ready!")

    return final_batch


def create_jepa_model(env, device, args, enable_jepa=True):
    """Create JEPA-enhanced model - SIMPLIFIED VERSION"""

    # Learning rate schedule adapted for JEPA
    if enable_jepa:
        # More conservative schedule for JEPA stability
        lr_schedule = lambda progress: args.learning_rate * (1 - 0.6 * progress)
    else:
        lr_schedule = lambda progress: args.learning_rate * (1 - 0.8 * progress)

    # Enhanced CNN configuration for JEPA
    features_dim = 512
    net_arch = dict(pi=[512, 256], vf=[512, 256])

    # Optimized for JEPA + Large batch
    model = PPO(
        "CnnPolicy",
        env,
        device=device,
        verbose=1,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=3 if enable_jepa else 2,  # More epochs for JEPA learning
        gamma=0.99,
        learning_rate=lr_schedule,
        clip_range=0.2,
        ent_coef=args.ent_coef,
        vf_coef=0.5,
        max_grad_norm=0.5,
        gae_lambda=0.95,
        tensorboard_log=None,
        policy_kwargs=dict(
            features_extractor_class=JEPAEnhancedCNN,
            features_extractor_kwargs=dict(features_dim=features_dim),
            normalize_images=False,
            optimizer_class=torch.optim.AdamW,
            optimizer_kwargs=dict(eps=1e-5, weight_decay=1e-4),
            net_arch=net_arch,
            activation_fn=nn.ReLU,
        ),
    )

    architecture_name = "JEPA Enhanced" if enable_jepa else "Enhanced CNN"
    print(f"🧠 {architecture_name} Model Created:")
    print(f"   📊 Batch size: {args.batch_size:,}")
    print(f"   📏 N_steps: {args.n_steps:,}")
    print(f"   🎲 Entropy coefficient: {args.ent_coef:.4f}")
    print(f"   🎓 Epochs: {3 if enable_jepa else 2}")
    if enable_jepa:
        print(f"   🔮 Binary outcome prediction: Enabled")
        print(f"   ⚔️ Strategic planning: Enabled")
        print(f"   🎯 Enhanced reward system: Active")
        print(f"   📈 Prediction horizon: {args.frame_stack} frames")
    print(f"   💾 Memory optimized for 11.6GB GPU")

    return model


def main():
    parser = argparse.ArgumentParser(
        description="JEPA Enhanced Fighting Game AI Training"
    )
    parser.add_argument("--total-timesteps", type=int, default=15000000)
    parser.add_argument("--learning-rate", type=float, default=2.0e-4)
    parser.add_argument(
        "--ent-coef", type=float, default=0.1, help="Entropy coefficient"
    )
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--use-default-state", action="store_true")
    parser.add_argument("--target-vram", type=float, default=11.6)
    parser.add_argument(
        "--n-steps", type=int, default=2048, help="Number of steps per rollout"
    )
    parser.add_argument("--batch-size", type=int, default=1024, help="Batch size")
    parser.add_argument(
        "--frame-stack", type=int, default=6, help="Number of frames to stack"
    )
    parser.add_argument(
        "--enable-jepa", action="store_true", help="Enable JEPA opponent prediction"
    )
    parser.add_argument("--disable-jepa", action="store_true", help="Disable JEPA")

    args = parser.parse_args()

    # JEPA is enabled by default, unless explicitly disabled
    if not args.disable_jepa:
        args.enable_jepa = True
    else:
        args.enable_jepa = False

    # Memory optimization
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    mode_name = "JEPA Enhanced" if args.enable_jepa else "Enhanced CNN"
    print(f"🚀 {mode_name} FIGHTING GAME AI")
    print(f"   💻 Device: {device}")
    print(f"   🧠 Architecture: {mode_name}")
    print(f"   📊 Batch size: {args.batch_size:,}")
    print(f"   📏 N_steps: {args.n_steps:,}")
    print(f"   🎲 Entropy coefficient: {args.ent_coef:.4f}")
    print(f"   🖼️ Frame stack: {args.frame_stack}")
    if args.enable_jepa:
        print(f"   🔮 Prediction horizon: {args.frame_stack} frames")
        print(f"   ⚔️ Strategic opponent modeling: Enabled")
        print(f"   🎯 Binary outcome prediction: 4 types")
    print(f"   💾 GPU memory optimized for {args.target_vram}GB")

    game = "SamuraiShodown-Genesis"

    # Handle state file
    if args.use_default_state:
        state = None
    else:
        if os.path.exists("samurai.state"):
            state = os.path.abspath("samurai.state")
            print(f"🎮 Using samurai.state file: {state}")
        else:
            print(f"❌ samurai.state not found, using default state")
            state = None

    # Observation shape for 6-frame setup
    obs_shape = (18, 180, 126)  # 6 frames * 3 channels = 18
    print(f"📊 Observation shape: {obs_shape} (6-frame stack)")

    # Calculate optimal batch size
    optimal_batch_size = calculate_jepa_batch_size(
        obs_shape, args.target_vram, args.enable_jepa
    )

    if args.batch_size > optimal_batch_size:
        print(f"💡 Recommended batch size: {optimal_batch_size:,}")
        print(f"   Current target: {args.batch_size:,}")
        response = input(f"   Use recommended {optimal_batch_size:,}? (y/n): ")
        if response.lower() == "y":
            args.batch_size = optimal_batch_size

    # Ensure buffer compatibility
    if args.n_steps % args.batch_size != 0:
        new_n_steps = ((args.n_steps // args.batch_size) + 1) * args.batch_size
        print(
            f"📊 Adjusting n_steps from {args.n_steps} to {new_n_steps} for compatibility"
        )
        args.n_steps = new_n_steps

    print(f"🔮 FINAL PARAMETERS:")
    print(f"   🎮 Environment: JEPA-Enhanced Samurai Showdown")
    print(f"   💪 Batch size: {args.batch_size:,}")
    print(f"   📏 N_steps: {args.n_steps:,}")
    print(f"   🎲 Entropy coefficient: {args.ent_coef:.4f}")
    print(f"   🌈 Channels: 18 (6-frame optimized)")
    print(f"   🧠 Architecture: {mode_name}")

    # Create JEPA-enhanced environment
    print(f"🔧 Creating {mode_name} environment...")
    try:
        env = retro.make(
            game=game,
            state=state,
            use_restricted_actions=retro.Actions.FILTERED,
            obs_type=retro.Observations.IMAGE,
            render_mode="human" if args.render else None,
        )

        env = SamuraiJEPAWrapper(
            env,
            reset_round=True,
            rendering=args.render,
            max_episode_steps=15000,
            frame_stack=args.frame_stack,
            enable_jepa=args.enable_jepa,
        )

        env = Monitor(env)
        print(f"✅ {mode_name} environment created successfully")

    except Exception as e:
        print(f"❌ Failed to create environment: {e}")
        return

    # Create save directory
    save_dir = (
        "trained_models_jepa_prime" if args.enable_jepa else "trained_models_enhanced"
    )
    os.makedirs(save_dir, exist_ok=True)

    # Monitor initial VRAM
    if device == "cuda":
        torch.cuda.empty_cache()
        vram_before = torch.cuda.memory_allocated() / (1024**3)
        print(f"   VRAM before model: {vram_before:.2f} GB")

    # Create model
    if args.resume and os.path.exists(args.resume):
        print(f"📂 Loading model: {args.resume}")
        model = PPO.load(args.resume, env=env, device=device)

        # Update parameters from arguments when resuming
        model.ent_coef = args.ent_coef
        print(f"🔄 Updated ent_coef to: {args.ent_coef:.4f}")

    else:
        print(f"🚀 Creating {mode_name} model")
        model = create_jepa_model(env, device, args, args.enable_jepa)

    # Monitor VRAM after model creation
    if device == "cuda":
        vram_after = torch.cuda.memory_allocated() / (1024**3)
        model_vram = vram_after - vram_before
        print(f"   VRAM after model: {vram_after:.2f} GB")
        print(f"   Model VRAM: {model_vram:.2f} GB")
        if args.enable_jepa:
            print(f"   🧠 JEPA overhead included: Strategic AI ready!")

    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path=save_dir,
        name_prefix=f"ppo_{'jepa_prime' if args.enable_jepa else 'enhanced'}",
    )

    training_callback = JEPATrainingCallback(enable_jepa=args.enable_jepa, verbose=1)

    # Training
    start_time = time.time()
    print(f"🏋️ Starting {mode_name} Training")
    if args.enable_jepa:
        print(f"   🔮 Binary outcome prediction: Learning patterns")
        print(f"   ⚔️ Strategic planning: AI developing")
        print(f"   🎯 Enhanced reward system: Active")

    try:
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=[checkpoint_callback, training_callback],
            reset_num_timesteps=not bool(args.resume),
        )

        training_time = time.time() - start_time
        print(f"🎉 {mode_name} training completed in {training_time/3600:.1f} hours!")

    except Exception as e:
        print(f"❌ {mode_name} training failed: {e}")
        import traceback

        traceback.print_exc()
    finally:
        env.close()
        if device == "cuda":
            torch.cuda.empty_cache()

    # Save final model
    final_path = os.path.join(
        save_dir, f"ppo_{'jepa_prime' if args.enable_jepa else 'enhanced'}_final.zip"
    )
    model.save(final_path)
    print(f"💾 Model saved: {final_path}")

    print(f"✅ {mode_name} TRAINING COMPLETE!")


if __name__ == "__main__":
    main()
