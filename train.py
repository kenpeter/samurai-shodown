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

# Import the enhanced wrapper with PRIME
from wrapper import SamuraiShowdownCustomWrapper


class DeepCNNFeatureExtractor(BaseFeaturesExtractor):
    """Enhanced CNN feature extractor optimized for pattern recognition"""

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 512):
        super().__init__(observation_space, features_dim)

        n_input_channels = observation_space.shape[
            0
        ]  # 27 channels (9 frames × 3 RGB channels)

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
        print(f"   🎨 RGB Processing: 9 frames × 3 RGB channels = 27 input channels")
        print(f"   🔍 Pattern-optimized architecture")
        print(f"   💪 Channels: 27 → 64 → 128 → 256 → 512")
        print(f"   🎯 FC layers: 3 layers → {features_dim}")
        print(f"   🔮 Optimized for RGB temporal pattern recognition")

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
    """Custom callback to monitor training performance with PRIME support"""

    def __init__(self, verbose=0):
        super(TrainingCallback, self).__init__(verbose)
        self.last_stats_log = 0

    def _on_step(self) -> bool:
        # Log statistics every 10000 steps
        if (
            self.num_timesteps % 10000 == 0
            and self.num_timesteps != self.last_stats_log
        ):
            self.last_stats_log = self.num_timesteps

            # Get stats from wrapper if available
            if hasattr(self.training_env, "get_attr"):
                try:
                    env_stats = self.training_env.get_attr("current_stats")[0]

                    win_rate = env_stats.get("win_rate", 0) * 100
                    wins = env_stats.get("wins", 0)
                    losses = env_stats.get("losses", 0)
                    prime_enabled = env_stats.get("prime_enabled", False)

                    print(f"\n📊 TRAINING UPDATE - Step {self.num_timesteps:,}")
                    print(f"   🎯 Win Rate: {win_rate:.1f}%")
                    print(f"   🏆 Record: {wins}W/{losses}L")

                    if prime_enabled:
                        process_weight = env_stats.get("process_weight", 0.0)
                        outcome_weight = env_stats.get("outcome_weight", 0.0)
                        avg_process_reward = env_stats.get("avg_process_reward", 0.0)
                        print(
                            f"   🧠 PRIME: process={process_weight:.1f}, outcome={outcome_weight:.1f}"
                        )
                        print(f"   🎯 Avg Process Reward: {avg_process_reward:.4f}")
                    else:
                        print(f"   🎯 PRIME: disabled")

                except Exception as e:
                    if self.verbose:
                        print(f"   Warning: Could not get training stats: {e}")

        return True


def calculate_maximum_batch_size_prime(
    obs_shape, target_vram_gb=16.0, force_batch_size=None, enable_prime=True
):
    """Memory-safe batch size calculation with PRIME overhead consideration"""
    num_frames, height, width = obs_shape
    obs_size_bytes = num_frames * height * width * 4
    obs_size_mb = obs_size_bytes / (1024 * 1024)

    prime_overhead = 1.5 if enable_prime else 1.0  # PRIME adds ~50% memory overhead

    print(f"📊 {'PRIME ' if enable_prime else ''}BATCH SIZE CALCULATION:")
    print(f"   Observation size: {obs_size_mb:.2f} MB per sample")
    if enable_prime:
        print(f"   🧠 PRIME overhead multiplier: {prime_overhead:.1f}x")

    if force_batch_size:
        obs_vram_gb = (force_batch_size * obs_size_bytes * prime_overhead) / (1024**3)
        model_vram_gb = 0.08 if enable_prime else 0.06  # Slightly more for PRIME
        activation_vram_gb = obs_vram_gb * 3
        total_vram_gb = obs_vram_gb + model_vram_gb + activation_vram_gb + 1.5

        print(f"   🎯 FORCED batch size: {force_batch_size:,}")
        print(f"   📊 Estimated VRAM needed: {total_vram_gb:.1f} GB")

        if total_vram_gb > target_vram_gb:
            print(f"   ⚠️  WARNING: May exceed {target_vram_gb}GB VRAM limit!")
            if enable_prime:
                print(f"   💡 Consider disabling PRIME or reducing batch size")
        else:
            print(f"   ✅ Should fit within {target_vram_gb}GB VRAM")

        return force_batch_size

    # Conservative estimate including PRIME overhead
    estimated_model_vram = 6.0 if enable_prime else 5.0
    available_vram = target_vram_gb - estimated_model_vram
    available_vram_bytes = available_vram * 1024 * 1024 * 1024

    memory_per_sample = obs_size_bytes * 10.0 * prime_overhead
    max_batch_size = int(available_vram_bytes / memory_per_sample)

    optimal_batch_size = (
        2 ** int(math.log2(max_batch_size)) if max_batch_size > 0 else 512
    )
    optimal_batch_size = max(optimal_batch_size, 512)
    optimal_batch_size = min(optimal_batch_size, 2048)

    print(f"   🛡️ OPTIMAL batch size: {optimal_batch_size:,}")
    if enable_prime:
        print(f"   🧠 Includes PRIME implicit PRM overhead")

    return optimal_batch_size


def check_system_resources(n_steps, obs_shape, batch_size, enable_prime=True):
    """Check system resources for enhanced training with PRIME"""
    print(f"🖥️  SYSTEM CHECK {'(PRIME enabled)' if enable_prime else ''}:")
    cpu_cores = psutil.cpu_count(logical=True)
    print(f"   CPU Cores: {cpu_cores}")
    ram_gb = psutil.virtual_memory().total / (1024**3)

    num_frames, height, width = obs_shape
    obs_size_mb = (num_frames * height * width * 4) / (1024 * 1024)
    buffer_memory_gb = (n_steps * obs_size_mb) / 1024

    prime_memory_gb = 1.0 if enable_prime else 0.5

    print(f"   RAM: {ram_gb:.1f} GB")
    print(f"   Buffer memory: {buffer_memory_gb:.1f} GB")
    if enable_prime:
        print(f"   PRIME overhead: {prime_memory_gb:.1f} GB")
    print(f"   Total required: {buffer_memory_gb + prime_memory_gb:.1f} GB")

    return (buffer_memory_gb + prime_memory_gb) < ram_gb * 0.4


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
        wrapped_env = SamuraiShowdownCustomWrapper(
            temp_env, rendering=False, enable_prime=False
        )
        obs_shape = wrapped_env.observation_space.shape
        temp_env.close()
        del wrapped_env
        return obs_shape
    except Exception as e:
        print(f"⚠️ Could not determine observation dimensions: {e}")
        return (27, 126, 180)


def linear_schedule(initial_value, final_value=0.0, decay_type="linear"):
    """Advanced learning rate schedule"""

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
    parser = argparse.ArgumentParser(
        description="PRIME-Enhanced Samurai Showdown Training"
    )
    parser.add_argument("--total-timesteps", type=int, default=10000000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--use-default-state", action="store_true")
    parser.add_argument(
        "--target-vram", type=float, default=16.0
    )  # Increased for PRIME
    parser.add_argument("--n-steps", type=int, default=512)  # PRIME optimized
    parser.add_argument("--batch-size", type=int, default=None)  # Auto-calculate
    parser.add_argument("--mixed-precision", action="store_true")
    parser.add_argument(
        "--lr-schedule",
        type=str,
        default="cosine",
        choices=["linear", "cosine", "exponential"],
    )

    # PRIME-specific arguments
    parser.add_argument(
        "--enable-prime", action="store_true", help="Enable PRIME implicit PRM"
    )
    parser.add_argument(
        "--process-weight", type=float, default=0.3, help="PRIME process reward weight"
    )
    parser.add_argument(
        "--outcome-weight", type=float, default=0.7, help="PRIME outcome reward weight"
    )
    parser.add_argument(
        "--prm-lr", type=float, default=1e-6, help="PRIME PRM learning rate"
    )

    args = parser.parse_args()

    # Memory optimization
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    device = "cuda"
    torch.cuda.empty_cache()

    print(f"🚀 PRIME-ENHANCED SAMURAI TRAINING")
    print(f"   💻 Device: {device}")
    print(f"   🧠 PRIME enabled: {args.enable_prime}")
    if args.enable_prime:
        print(
            f"   🎯 Process/Outcome weights: {args.process_weight:.1f}/{args.outcome_weight:.1f}"
        )
        print(f"   📊 PRM learning rate: {args.prm_lr}")
    print(f"   🎨 RGB Processing: 9 frames × 3 channels = 27 input channels")
    print(f"   📊 Hyperparameters: n_steps={args.n_steps}")
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

    # Calculate optimal batch size
    if args.batch_size is None:
        optimal_batch_size = calculate_maximum_batch_size_prime(
            obs_shape, args.target_vram, enable_prime=args.enable_prime
        )
    else:
        optimal_batch_size = calculate_maximum_batch_size_prime(
            obs_shape,
            args.target_vram,
            force_batch_size=args.batch_size,
            enable_prime=args.enable_prime,
        )

    # System check
    if not check_system_resources(
        args.n_steps, obs_shape, optimal_batch_size, args.enable_prime
    ):
        print("❌ Insufficient system resources")
        if args.enable_prime:
            print("💡 Try disabling PRIME with --no-enable-prime")
        return

    print(f"🔮 OPTIMIZED PARAMETERS:")
    print(f"   🎮 Environments: 1")
    print(f"   💪 Batch size: {optimal_batch_size:,}")
    print(f"   📏 N-steps: {args.n_steps}")
    print(f"   🌈 RGB channels: 27")
    if args.enable_prime:
        print(f"   🧠 PRIME implicit PRM enabled")
    print(f"   🎪 PPO clip epsilon: 0.12")

    # Create environment with PRIME support
    print(f"🔧 Creating environment...")
    try:
        env = retro.make(
            game=game,
            state=state,
            use_restricted_actions=retro.Actions.FILTERED,
            obs_type=retro.Observations.IMAGE,
            render_mode="human" if args.render else None,
        )

        # Create wrapper with PRIME configuration
        env = SamuraiShowdownCustomWrapper(
            env,
            reset_round=True,
            rendering=args.render,
            max_episode_steps=15000,
            enable_prime=args.enable_prime,
            process_weight=args.process_weight,
            outcome_weight=args.outcome_weight,
            prm_lr=args.prm_lr,
            device=device,
        )

        env = Monitor(env)
        print(
            f"✅ Environment created {'with PRIME' if args.enable_prime else 'without PRIME'}"
        )

    except Exception as e:
        print(f"❌ Failed to create environment: {e}")
        return

    # Create save directory
    save_dir = (
        "trained_models_prime_optimized"
        if args.enable_prime
        else "trained_models_fighting_optimized"
    )
    os.makedirs(save_dir, exist_ok=True)

    torch.cuda.empty_cache()
    vram_before = torch.cuda.memory_allocated() / (1024**3)
    print(f"   VRAM before model: {vram_before:.2f} GB")

    if args.resume and os.path.exists(args.resume):
        print(f"📂 Loading model: {args.resume}")
        model = PPO.load(args.resume, env=env, device=device)
        model.n_steps = args.n_steps
        model.batch_size = optimal_batch_size
        model._setup_model()
    else:
        print(
            f"🚀 Creating {'PRIME-ENHANCED' if args.enable_prime else 'FIGHTING GAME OPTIMIZED'} PPO model"
        )

        lr_schedule = linear_schedule(
            args.learning_rate, args.learning_rate * 0.1, args.lr_schedule
        )

        model = PPO(
            "CnnPolicy",
            env,
            device=device,
            verbose=1,
            n_steps=args.n_steps,
            batch_size=optimal_batch_size,
            n_epochs=4,
            gamma=0.995,
            learning_rate=lr_schedule,
            clip_range=linear_schedule(0.12, 0.05),
            ent_coef=0.02,
            vf_coef=0.5,
            max_grad_norm=0.5,
            gae_lambda=0.95,
            tensorboard_log=None,
            policy_kwargs=dict(
                features_extractor_class=DeepCNNFeatureExtractor,
                features_extractor_kwargs=dict(features_dim=512),
                normalize_images=False,
                optimizer_class=torch.optim.Adam,
                optimizer_kwargs=dict(eps=1e-5, weight_decay=1e-4),
                net_arch=dict(
                    pi=[512, 256, 128],
                    vf=[512, 256, 128],
                ),
                activation_fn=nn.ReLU,
            ),
        )

    # Monitor VRAM usage
    vram_after = torch.cuda.memory_allocated() / (1024**3)
    model_vram = vram_after - vram_before
    print(f"   VRAM after model: {vram_after:.2f} GB")
    print(f"   Model VRAM: {model_vram:.2f} GB")

    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path=save_dir,
        name_prefix=(
            "ppo_prime_optimized" if args.enable_prime else "ppo_fighting_optimized"
        ),
    )

    training_callback = TrainingCallback(verbose=1)

    # Training
    start_time = time.time()
    print(
        f"🏋️ Starting {'PRIME-ENHANCED' if args.enable_prime else 'FIGHTING GAME OPTIMIZED'} TRAINING"
    )
    if args.enable_prime:
        print(f"   🧠 PRIME implicit PRM provides dense process rewards")
        print(
            f"   🎯 Process/outcome reward combination: {args.process_weight:.1f}/{args.outcome_weight:.1f}"
        )
    print(f"   📊 Batch size: {optimal_batch_size}")

    try:
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=[checkpoint_callback, training_callback],
            reset_num_timesteps=not bool(args.resume),
        )

        training_time = time.time() - start_time
        print(f"🎉 Training completed in {training_time/3600:.1f} hours!")

        # Final performance assessment
        if hasattr(env, "current_stats"):
            final_stats = env.current_stats
            print(f"\n🎯 FINAL PERFORMANCE:")
            print(f"   🏆 Final Win Rate: {final_stats['win_rate']*100:.1f}%")
            print(f"   🎮 Total Rounds: {final_stats['total_rounds']}")
            if args.enable_prime and final_stats.get("prime_enabled", False):
                print(f"   🧠 PRIME enabled throughout training")
                print(
                    f"   🎯 Final process weight: {final_stats.get('process_weight', 0.0):.1f}"
                )

    except KeyboardInterrupt:
        print(f"⏹️ Training interrupted")
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback

        traceback.print_exc()
    finally:
        env.close()
        torch.cuda.empty_cache()

    # Save final model
    final_path = os.path.join(
        save_dir,
        f"ppo_{'prime' if args.enable_prime else 'fighting'}_optimized_final.zip",
    )
    model.save(final_path)
    print(f"💾 Model saved: {final_path}")

    # Final VRAM report
    final_vram = torch.cuda.memory_allocated() / (1024**3)
    max_vram = torch.cuda.max_memory_allocated() / (1024**3)
    print(f"📊 Final VRAM: {final_vram:.2f} GB")
    print(f"📊 Peak VRAM: {max_vram:.2f} GB")

    print(
        f"✅ {'PRIME-ENHANCED' if args.enable_prime else 'FIGHTING GAME OPTIMIZED'} TRAINING COMPLETE!"
    )
    if args.enable_prime:
        print("🧠 Key PRIME benefits:")
        print("   • Dense process rewards for better credit assignment")
        print("   • Online PRM updates prevent reward hacking")
        print("   • Improved sample efficiency")
        print("   • No manual process annotation required")
    else:
        print("🎯 Standard training benefits:")
        print("   • Multi-component reward system")
        print("   • Fighting game optimized hyperparameters")
        print("   • Maximum batch size optimization")

    print(f"\n🎮 USAGE INSTRUCTIONS:")
    if args.enable_prime:
        print(
            f"   Enable PRIME: python train.py --enable-prime --batch-size {optimal_batch_size}"
        )
        print(f"   Adjust weights: --process-weight 0.4 --outcome-weight 0.6")
    else:
        print(
            f"   Standard training: python train.py --batch-size {optimal_batch_size}"
        )
    print(f"   Resume training: python train.py --resume {final_path}")
    print(f"   For lower VRAM: python train.py --mixed-precision --target-vram 12.0")


if __name__ == "__main__":
    main()
