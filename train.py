import os
import sys
import argparse
import time
import math
import torch
import torch.nn as nn
import psutil
from typing import Dict, Any, Optional, Type, Union

import retro
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.policies import ActorCriticCnnPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# Import the wrapper with correct name
from wrapper import SamuraiShowdownCustomWrapper


class DeepCNNFeatureExtractor(BaseFeaturesExtractor):
    """Enhanced CNN feature extractor optimized for pattern recognition"""

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 512):
        super().__init__(observation_space, features_dim)

        n_input_channels = observation_space.shape[0]  # 9 frames for temporal patterns

        # Enhanced network for pattern recognition
        self.conv_layers = nn.Sequential(
            # Initial feature extraction with larger receptive field
            nn.Conv2d(n_input_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Temporal pattern detection blocks
            self._make_conv_block(64, 128, 3, 1, 1),
            self._make_conv_block(128, 128, 3, 2, 1),
            # Spatial-temporal fusion
            self._make_conv_block(128, 256, 3, 1, 1),
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # High-level pattern extraction
            self._make_conv_block(256, 512, 3, 1, 1),
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            # Final feature compression
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )

        # Enhanced FC layers for pattern understanding
        self.fc_layers = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, features_dim),
            nn.ReLU(inplace=True),
        )

        print(f"🧠 ENHANCED CNN Network:")
        print(f"   📊 Input: {observation_space.shape}")
        print(f"   🔍 Pattern-optimized architecture")
        print(f"   💪 Channels: 9 → 64 → 128 → 256 → 512")
        print(f"   🎯 FC layers: 3 layers → {features_dim}")
        print(f"   🔮 Optimized for temporal pattern recognition")

    def _make_conv_block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        observations = observations.float() / 255.0
        features = self.conv_layers(observations)
        features = self.fc_layers(features)
        return features


class TrainingCallback(BaseCallback):
    """Custom callback to monitor enhanced performance"""

    def __init__(self, verbose=0):
        super(TrainingCallback, self).__init__(verbose)
        self.prediction_log = []
        self.last_stats_log = 0

    def _on_step(self) -> bool:
        # Log enhanced statistics every 10000 steps
        if (
            self.num_timesteps % 10000 == 0
            and self.num_timesteps != self.last_stats_log
        ):
            self.last_stats_log = self.num_timesteps

            # Get stats from wrapper if available
            if hasattr(self.training_env, "get_attr"):
                try:
                    env_stats = self.training_env.get_attr("current_stats")[0]

                    prediction_accuracy = env_stats.get("prediction_accuracy", 0) * 100
                    anticipation_rewards = env_stats.get("anticipation_rewards", 0)
                    recognized_patterns = env_stats.get("recognized_patterns", 0)

                    print(
                        f"\n🔮 ENHANCED TRAINING UPDATE - Step {self.num_timesteps:,}"
                    )
                    print(f"   🎯 Prediction Accuracy: {prediction_accuracy:.1f}%")
                    print(f"   🧠 Anticipation Rewards: {anticipation_rewards:.2f}")
                    print(f"   📊 Recognized Patterns: {recognized_patterns}")

                    # Log to tensorboard if available
                    if hasattr(self.logger, "record"):
                        self.logger.record("enhanced/accuracy", prediction_accuracy)
                        self.logger.record(
                            "enhanced/anticipation_rewards", anticipation_rewards
                        )
                        self.logger.record(
                            "enhanced/patterns_recognized", recognized_patterns
                        )

                except Exception as e:
                    if self.verbose:
                        print(f"   Warning: Could not get enhanced stats: {e}")

        return True


def calculate_maximum_batch_size(obs_shape, target_vram_gb=9.0):
    """Memory-safe batch size calculation for enhanced training"""
    num_frames, height, width = obs_shape
    obs_size_bytes = num_frames * height * width * 4
    obs_size_mb = obs_size_bytes / (1024 * 1024)

    print(f"📊 ENHANCED BATCH SIZE CALCULATION:")
    print(f"   Observation size: {obs_size_mb:.2f} MB per sample")

    # Conservative estimate for model + pattern tracking
    estimated_model_vram = 5.0
    available_vram = target_vram_gb - estimated_model_vram
    available_vram_bytes = available_vram * 1024 * 1024 * 1024

    # More conservative multiplier for enhanced features
    memory_per_sample = obs_size_bytes * 10.0
    max_batch_size = int(available_vram_bytes / memory_per_sample)

    optimal_batch_size = (
        2 ** int(math.log2(max_batch_size)) if max_batch_size > 0 else 64
    )
    optimal_batch_size = max(optimal_batch_size, 64)  # Minimum
    optimal_batch_size = min(optimal_batch_size, 512)  # Conservative maximum

    print(f"   🛡️ SAFE batch size: {optimal_batch_size:,}")
    print(f"   🔮 Optimized for enhanced training")

    return optimal_batch_size


def check_system_resources(n_steps, obs_shape, batch_size):
    """Check system resources for enhanced training"""
    print(f"🖥️  ENHANCED SYSTEM CHECK:")
    cpu_cores = psutil.cpu_count(logical=True)
    print(f"   CPU Cores: {cpu_cores}")
    ram_gb = psutil.virtual_memory().total / (1024**3)

    num_frames, height, width = obs_shape
    obs_size_mb = (num_frames * height * width * 4) / (1024 * 1024)
    buffer_memory_gb = (n_steps * obs_size_mb) / 1024

    # Additional memory for pattern tracking
    pattern_memory_gb = 0.5

    print(f"   RAM: {ram_gb:.1f} GB")
    print(f"   Buffer memory: {buffer_memory_gb:.1f} GB")
    print(f"   Pattern tracking: {pattern_memory_gb:.1f} GB")
    print(f"   Total required: {buffer_memory_gb + pattern_memory_gb:.1f} GB")

    return (buffer_memory_gb + pattern_memory_gb) < ram_gb * 0.4


def get_observation_dims(game, state):
    """Get actual observation dimensions"""
    try:
        temp_env = retro.make(
            game=game,
            state=state,
            use_restricted_actions=retro.Actions.FILTERED,
            obs_type=retro.Observations.IMAGE,
            render_mode=None,
        )
        # FIXED: Use correct wrapper class name
        wrapped_env = SamuraiShowdownCustomWrapper(temp_env, rendering=False)
        obs_shape = wrapped_env.observation_space.shape
        temp_env.close()
        del wrapped_env
        return obs_shape
    except Exception as e:
        print(f"⚠️ Could not determine observation dimensions: {e}")
        return (9, 126, 180)


def linear_schedule(initial_value, final_value=0.0, decay_type="linear"):
    """Advanced learning rate schedule for enhanced training"""

    def scheduler(progress):
        if decay_type == "linear":
            return final_value + progress * (initial_value - final_value)
        elif decay_type == "cosine":
            return final_value + (initial_value - final_value) * 0.5 * (
                1 + math.cos(math.pi * (1 - progress))
            )
        elif decay_type == "exponential":
            return initial_value * (final_value / initial_value) ** (1 - progress)
        else:
            return initial_value

    return scheduler


def main():
    parser = argparse.ArgumentParser(description="Enhanced Samurai Showdown Training")
    parser.add_argument("--total-timesteps", type=int, default=10000000)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--use-default-state", action="store_true")
    parser.add_argument("--target-vram", type=float, default=9.0)
    parser.add_argument("--n-steps", type=int, default=4096)
    parser.add_argument("--prediction-horizon", type=int, default=30)
    parser.add_argument(
        "--lr-schedule",
        type=str,
        default="cosine",
        choices=["linear", "cosine", "exponential"],
    )

    args = parser.parse_args()

    # Memory optimization
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    device = "cuda"
    torch.cuda.empty_cache()

    print(f"🚀 ENHANCED SAMURAI TRAINING")
    print(f"   💻 Device: {device}")
    print(f"   🔮 Prediction horizon: {args.prediction_horizon} frames")
    print(f"   📚 Learning schedule: {args.lr_schedule}")
    print(f"   🛡️ Memory optimized for {args.target_vram}GB VRAM")

    game = "SamuraiShodown-Genesis"

    # Test environment
    print(f"🎮 Testing {game}...")
    try:
        test_env = retro.make(game=game, state=None)
        test_env.close()
        print(f"✅ Environment test passed")
    except Exception as e:
        print(f"❌ Environment test failed: {e}")
        return

    # Handle state file
    if args.use_default_state:
        state = None
        print(f"🎮 Using default game state")
    else:
        if os.path.exists("samurai.state"):
            state = os.path.abspath("samurai.state")
            print(f"🎮 Using samurai.state file: {state}")
        else:
            print(f"❌ samurai.state not found, using default state")
            state = None

    # Get observation dimensions
    obs_shape = get_observation_dims(game, state)
    print(f"📊 Observation shape: {obs_shape}")

    # Calculate optimal batch size for enhanced training
    max_batch_size = calculate_maximum_batch_size(obs_shape, args.target_vram)

    # System check
    if not check_system_resources(args.n_steps, obs_shape, max_batch_size):
        print("❌ Insufficient system resources for enhanced training")
        return

    n_steps = args.n_steps
    batch_size = min(max_batch_size, n_steps)

    print(f"🔮 ENHANCED TRAINING PARAMETERS:")
    print(f"   🎮 Environments: 1 (focused training)")
    print(f"   💪 Batch size: {batch_size:,}")
    print(f"   📏 N-steps: {n_steps:,}")
    print(f"   🧠 Pattern prediction enabled")
    print(f"   🎯 Enhanced reward system")

    # Create environment with enhanced wrapper
    print(f"🔧 Creating enhanced environment...")
    try:
        env = retro.make(
            game=game,
            state=state,
            use_restricted_actions=retro.Actions.FILTERED,
            obs_type=retro.Observations.IMAGE,
            render_mode="human" if args.render else None,
        )

        # FIXED: Use correct wrapper class name
        env = SamuraiShowdownCustomWrapper(
            env,
            reset_round=True,
            rendering=args.render,
            max_episode_steps=15000,  # Longer episodes for pattern learning
            prediction_horizon=args.prediction_horizon,
        )

        env = Monitor(env)
        print(f"✅ Enhanced environment created")

    except Exception as e:
        print(f"❌ Failed to create environment: {e}")
        return

    # Create save directory
    save_dir = "trained_models_ultra_deep"
    os.makedirs(save_dir, exist_ok=True)

    torch.cuda.empty_cache()
    vram_before = torch.cuda.memory_allocated() / (1024**3)
    print(f"   VRAM before model: {vram_before:.2f} GB")

    if args.resume and os.path.exists(args.resume):
        print(f"📂 Loading model: {args.resume}")
        model = PPO.load(args.resume, env=env, device=device)
        model.n_steps = n_steps
        model.batch_size = batch_size
        model._setup_model()
    else:
        print(f"🚀 Creating ENHANCED PPO model")

        lr_schedule = linear_schedule(
            args.learning_rate, args.learning_rate * 0.1, args.lr_schedule
        )

        model = PPO(
            "CnnPolicy",
            env,
            device=device,
            verbose=1,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=4,  # More epochs for better pattern learning
            gamma=0.995,  # Slightly higher for long-term planning
            learning_rate=lr_schedule,
            clip_range=linear_schedule(0.2, 0.05),
            ent_coef=0.015,  # Encourage exploration for pattern discovery
            vf_coef=0.5,
            max_grad_norm=0.5,
            gae_lambda=0.95,
            tensorboard_log="logs_ultra_deep",
            policy_kwargs=dict(
                features_extractor_class=DeepCNNFeatureExtractor,
                features_extractor_kwargs=dict(features_dim=512),
                normalize_images=False,
                optimizer_class=torch.optim.Adam,
                optimizer_kwargs=dict(eps=1e-5, weight_decay=1e-4),
                net_arch=dict(
                    pi=[512, 256, 128],  # Policy network for pattern-based decisions
                    vf=[512, 256, 128],  # Value network for long-term assessment
                ),
                activation_fn=nn.ReLU,
            ),
        )

    # Monitor VRAM usage
    vram_after = torch.cuda.memory_allocated() / (1024**3)
    model_vram = vram_after - vram_before
    print(f"   VRAM after model: {vram_after:.2f} GB")
    print(f"   Model VRAM: {model_vram:.2f} GB")

    # Enhanced callbacks for training
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,  # More frequent saves for pattern preservation
        save_path=save_dir,
        name_prefix="ppo_ultra_deep",
    )

    training_callback = TrainingCallback(verbose=1)

    # Training with enhanced monitoring
    start_time = time.time()
    print(f"🏋️ Starting ENHANCED TRAINING")
    print(f"   🔮 Focus: Pattern recognition and anticipation")
    print(f"   🧠 Enhanced reward system for prediction accuracy")

    try:
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=[checkpoint_callback, training_callback],
            reset_num_timesteps=not bool(args.resume),
        )

        training_time = time.time() - start_time
        print(f"🎉 Enhanced training completed in {training_time/3600:.1f} hours!")

        # Final enhanced performance assessment
        if hasattr(env, "current_stats"):
            final_stats = env.current_stats
            print(f"\n🔮 FINAL ENHANCED PERFORMANCE:")
            print(f"   🎯 Final Win Rate: {final_stats['win_rate']*100:.1f}%")
            print(
                f"   🧠 Prediction Accuracy: {final_stats['prediction_accuracy']*100:.1f}%"
            )
            print(f"   📊 Patterns Recognized: {final_stats['recognized_patterns']}")
            print(f"   🎭 Pattern Types: {final_stats['pattern_types']}")
            print(
                f"   🏆 Anticipation Rewards: {final_stats['anticipation_rewards']:.2f}"
            )

    except KeyboardInterrupt:
        print(f"⏹️ Enhanced training interrupted")
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback

        traceback.print_exc()
    finally:
        env.close()
        torch.cuda.empty_cache()

    # Save final model
    final_path = os.path.join(save_dir, "ppo_ultra_deep_final.zip")
    model.save(final_path)
    print(f"💾 Enhanced model saved: {final_path}")

    # Final VRAM report
    final_vram = torch.cuda.memory_allocated() / (1024**3)
    max_vram = torch.cuda.max_memory_allocated() / (1024**3)
    print(f"📊 Final VRAM: {final_vram:.2f} GB")
    print(f"📊 Peak VRAM: {max_vram:.2f} GB")

    print("✅ ENHANCED TRAINING COMPLETE!")
    print("🔮 Agent should now demonstrate:")
    print("   • Pattern recognition capabilities")
    print("   • Anticipatory decision making")
    print("   • Proactive counter-strategies")
    print("   • Enhanced reaction timing")


if __name__ == "__main__":
    main()
