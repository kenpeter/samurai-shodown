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
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.policies import ActorCriticCnnPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# Import the wrapper
from wrapper import SamuraiShowdownCustomWrapper


class DeepCNNFeatureExtractor(BaseFeaturesExtractor):
    """Ultra-deep CNN feature extractor for maximum network depth"""

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 1024):
        super().__init__(observation_space, features_dim)

        n_input_channels = observation_space.shape[0]  # 9 frames

        # Calculate adaptive filter sizes based on input dimensions
        height, width = observation_space.shape[1], observation_space.shape[2]

        # Ultra-deep network with residual connections
        self.conv_layers = nn.Sequential(
            # Initial feature extraction - larger filters
            nn.Conv2d(n_input_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            # Deep residual-style blocks
            self._make_conv_block(64, 64, 3, 1, 1),
            self._make_conv_block(64, 64, 3, 1, 1),
            self._make_conv_block(64, 128, 3, 2, 1),  # Downsample
            self._make_conv_block(128, 128, 3, 1, 1),
            self._make_conv_block(128, 128, 3, 1, 1),
            self._make_conv_block(128, 256, 3, 2, 1),  # Downsample
            self._make_conv_block(256, 256, 3, 1, 1),
            self._make_conv_block(256, 256, 3, 1, 1),
            self._make_conv_block(256, 256, 3, 1, 1),
            self._make_conv_block(256, 512, 3, 2, 1),  # Downsample
            self._make_conv_block(512, 512, 3, 1, 1),
            self._make_conv_block(512, 512, 3, 1, 1),
            self._make_conv_block(512, 512, 3, 1, 1),
            self._make_conv_block(512, 512, 3, 1, 1),
            # Final feature extraction
            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )

        # Deep fully connected layers that output exactly features_dim
        self.fc_layers = nn.Sequential(
            nn.Linear(1024, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, features_dim),  # Must output exactly features_dim (512)
            nn.ReLU(inplace=True),
        )

        print(f"🧠 ULTRA-DEEP CNN Network Architecture:")
        print(f"   📊 Input: {observation_space.shape}")
        print(f"   🔥 Conv layers: 18 layers (7x7 → 3x3 blocks)")
        print(f"   💪 Channels: 9 → 64 → 128 → 256 → 512 → 1024")
        print(f"   🎯 FC layers: 4 deep layers (2048 → 1024 → 512 → {features_dim})")
        print(f"   ⚡ Total depth: ~22 layers")
        print(f"   🔗 Output features: {features_dim} (feeds to policy networks)")

    def _make_conv_block(self, in_channels, out_channels, kernel_size, stride, padding):
        """Create a convolutional block with BatchNorm and ReLU"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # Normalize observations to [0, 1]
        observations = observations.float() / 255.0

        # Pass through conv layers
        features = self.conv_layers(observations)

        # Pass through FC layers
        features = self.fc_layers(features)

        return features


def calculate_maximum_batch_size(obs_shape, target_vram_gb=10.0):
    """Calculate the absolute maximum batch size for single environment"""

    # Observation memory calculation
    num_frames, height, width = obs_shape
    obs_size_bytes = num_frames * height * width * 4  # float32
    obs_size_mb = obs_size_bytes / (1024 * 1024)

    print(f"📊 MAXIMUM BATCH SIZE CALCULATION:")
    print(f"   Observation size: {obs_size_mb:.2f} MB per sample")

    # Estimate deep network VRAM usage
    # Ultra-deep CNN + deep policy networks
    estimated_model_vram = 3.5  # GB for ultra-deep network

    # Available VRAM for batches
    available_vram = target_vram_gb - estimated_model_vram
    available_vram_bytes = available_vram * 1024 * 1024 * 1024

    # Calculate maximum batch size
    # Each sample needs: obs + gradients + activations + intermediate features
    memory_per_sample = obs_size_bytes * 4.0  # Higher multiplier for deep network
    max_batch_size = int(available_vram_bytes / memory_per_sample)

    # Round to nearest power of 2 for optimal memory alignment
    optimal_batch_size = 2 ** int(math.log2(max_batch_size))

    # Ensure reasonable bounds
    optimal_batch_size = max(optimal_batch_size, 512)  # Minimum
    optimal_batch_size = min(optimal_batch_size, 16384)  # Maximum

    estimated_batch_vram = (optimal_batch_size * memory_per_sample) / (1024**3)

    print(f"   Deep network VRAM: {estimated_model_vram:.1f} GB")
    print(f"   Available for batches: {available_vram:.1f} GB")
    print(f"   MAXIMUM batch size: {optimal_batch_size:,}")
    print(f"   Estimated batch VRAM: {estimated_batch_vram:.1f} GB")
    print(f"   🎯 SINGLE ENV OPTIMIZATION: All VRAM for massive batches!")

    return optimal_batch_size


def check_system_resources(n_steps, obs_shape, batch_size):
    """Check system resources for single environment setup"""

    print(f"🖥️  SINGLE ENV SYSTEM CHECK:")

    # CPU - single env needs minimal cores
    cpu_cores = psutil.cpu_count(logical=True)
    print(f"   CPU Cores: {cpu_cores} (1 env needs minimal CPU)")

    # RAM calculation
    ram_gb = psutil.virtual_memory().total / (1024**3)

    # Buffer memory for single env
    num_frames, height, width = obs_shape
    obs_size_mb = (num_frames * height * width * 4) / (1024 * 1024)
    buffer_memory_gb = (n_steps * obs_size_mb) / 1024

    print(f"   RAM: {ram_gb:.1f} GB")
    print(f"   Buffer memory: {buffer_memory_gb:.1f} GB")
    print(f"   Strategy: 1 env = minimal RAM usage, maximum VRAM for batches")

    return buffer_memory_gb < ram_gb * 0.3  # Very conservative for single env


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
        wrapped_env = SamuraiShowdownCustomWrapper(temp_env, rendering=False)
        obs_shape = wrapped_env.observation_space.shape
        temp_env.close()
        del wrapped_env
        return obs_shape
    except Exception as e:
        print(f"⚠️ Could not determine observation dimensions: {e}")
        return (9, 126, 180)  # Fallback


def linear_schedule(initial_value, final_value=0.0):
    """Linear learning rate schedule"""

    def scheduler(progress):
        return final_value + progress * (initial_value - final_value)

    return scheduler


def main():
    parser = argparse.ArgumentParser(
        description="Ultra-Deep Single Environment Training with Maximum Batch Size"
    )
    parser.add_argument("--total-timesteps", type=int, default=50000000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--use-default-state", action="store_true")
    parser.add_argument("--target-vram", type=float, default=10.0)
    parser.add_argument("--n-steps", type=int, default=16384)

    args = parser.parse_args()

    # Force CUDA
    device = "cuda"
    torch.cuda.empty_cache()

    print(f"🚀 ULTRA-DEEP SINGLE ENV TRAINING")
    print(f"   💻 Device: {device}")
    print(f"   🎯 Strategy: 1 env + deepest network + maximum batch size")
    print(f"   🧠 Network: Ultra-deep CNN (~30 layers)")
    print(f"   💪 Batch: Maximum possible size")
    print(f"   🎮 UI: Single game window")

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

    # Handle state file properly
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

    # Calculate maximum batch size
    max_batch_size = calculate_maximum_batch_size(obs_shape, args.target_vram)

    # System check
    if not check_system_resources(args.n_steps, obs_shape, max_batch_size):
        print("❌ Insufficient system resources")
        return

    # Adjust n_steps if needed to accommodate batch size
    n_steps = args.n_steps
    if max_batch_size > n_steps:
        # Can use larger rollouts with single env
        n_steps = max_batch_size
        print(f"🔄 Adjusted n_steps to {n_steps} to match batch size")

    batch_size = min(max_batch_size, n_steps)

    print(f"🔥 FINAL HYPERPARAMETERS:")
    print(f"   🎮 Environments: 1 (single env + UI)")
    print(f"   💪 Batch size: {batch_size:,}")
    print(f"   📏 N-steps: {n_steps:,}")
    print(f"   🧠 Network: Ultra-deep (~30 layers)")
    print(f"   🎯 Strategy: Maximum depth + maximum batch size")

    # Create single environment with rendering
    print(f"🔧 Creating single environment with UI...")

    try:
        # Create environment with proper state handling
        print(f"🔧 Creating environment with state: {state}")
        env = retro.make(
            game=game,
            state=state,  # None for default state, or absolute path for custom state
            use_restricted_actions=retro.Actions.FILTERED,
            obs_type=retro.Observations.IMAGE,
            render_mode="human" if args.render else None,
        )

        env = SamuraiShowdownCustomWrapper(
            env,
            reset_round=True,
            rendering=args.render,
            max_episode_steps=12000,
        )

        env = Monitor(env)
        print(f"✅ Single environment created with UI")

    except Exception as e:
        print(f"❌ Failed to create environment: {e}")
        print(f"🔄 Trying with default state...")
        try:
            # Fallback to default state
            env = retro.make(
                game=game,
                state=None,  # Force default state
                use_restricted_actions=retro.Actions.FILTERED,
                obs_type=retro.Observations.IMAGE,
                render_mode="human" if args.render else None,
            )

            env = SamuraiShowdownCustomWrapper(
                env,
                reset_round=True,
                rendering=args.render,
                max_episode_steps=12000,
            )

            env = Monitor(env)
            print(f"✅ Environment created with default state")

        except Exception as e2:
            print(f"❌ Failed to create environment with default state: {e2}")
            import traceback

            traceback.print_exc()
            return

    # Create ultra-deep model
    save_dir = "trained_models_ultra_deep"
    os.makedirs(save_dir, exist_ok=True)

    # Monitor VRAM
    torch.cuda.empty_cache()
    vram_before = torch.cuda.memory_allocated() / (1024**3)
    print(f"   VRAM before model: {vram_before:.2f} GB")

    if args.resume and os.path.exists(args.resume):
        print(f"📂 Loading model: {args.resume}")
        model = PPO.load(args.resume, env=env, device=device)

        # Update hyperparameters
        model.n_steps = n_steps
        model.batch_size = batch_size
        model._setup_model()

    else:
        print(f"🚀 Creating ULTRA-DEEP PPO model")

        lr_schedule = linear_schedule(args.learning_rate, args.learning_rate * 0.1)

        # FIXED: Proper action space detection for retro environments
        print(f"🔍 Detecting action space type...")
        print(f"   Action space: {env.action_space}")
        print(f"   Action space type: {type(env.action_space)}")

        action_space_type = type(env.action_space).__name__
        print(f"   Action space class: {action_space_type}")

        # Most retro environments use MultiBinary action spaces
        if action_space_type == "MultiBinary" or hasattr(env.action_space, "shape"):
            policy_type = "CnnPolicy"  # CnnPolicy handles MultiBinary
            print(f"   🎮 Using CnnPolicy for MultiBinary action space")
            if hasattr(env.action_space, "shape"):
                print(f"   🎮 MultiBinary shape: {env.action_space.shape}")
        elif action_space_type == "Discrete":
            policy_type = "CnnPolicy"
            print(
                f"   🎮 Using CnnPolicy for Discrete action space ({env.action_space.n} actions)"
            )
        else:
            # For retro games, assume MultiBinary if uncertain
            policy_type = "CnnPolicy"
            print(f"   🎮 Assuming MultiBinary for retro game, using CnnPolicy")

        model = PPO(
            policy_type,
            env,
            device=device,
            verbose=1,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=3,  # Fewer epochs for massive batches
            gamma=0.99,
            learning_rate=lr_schedule,
            clip_range=linear_schedule(0.2, 0.02),
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            gae_lambda=0.95,
            tensorboard_log="logs_ultra_deep",
            policy_kwargs=dict(
                features_extractor_class=DeepCNNFeatureExtractor,
                features_extractor_kwargs=dict(
                    features_dim=512
                ),  # Use 512 for compatibility
                normalize_images=False,
                optimizer_class=torch.optim.Adam,
                optimizer_kwargs=dict(eps=1e-5, weight_decay=1e-4),
                net_arch=dict(
                    pi=[512, 256, 128],  # Deep actor network
                    vf=[512, 256, 128],  # Deep critic network
                ),
                activation_fn=nn.ReLU,
            ),
        )

    # Monitor VRAM after model creation
    vram_after = torch.cuda.memory_allocated() / (1024**3)
    model_vram = vram_after - vram_before
    print(f"   VRAM after model: {vram_after:.2f} GB")
    print(f"   Model VRAM: {model_vram:.2f} GB")
    print(f"   Available VRAM: {12 - vram_after:.2f} GB")

    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=100000,
        save_path=save_dir,
        name_prefix="ppo_ultra_deep_single_env",
    )

    # Training
    start_time = time.time()
    print(f"🏋️ Starting ULTRA-DEEP training")
    print(f"   🎮 Single environment with game UI")
    print(f"   🧠 Network depth: ~30 layers")
    print(f"   💪 Batch size: {batch_size:,}")

    try:
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=[checkpoint_callback],
            reset_num_timesteps=not bool(args.resume),
        )

        training_time = time.time() - start_time
        print(f"🎉 Training completed in {training_time/3600:.1f} hours!")

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
    final_path = os.path.join(save_dir, "ppo_ultra_deep_final.zip")
    model.save(final_path)
    print(f"💾 Model saved: {final_path}")

    # Final VRAM report
    final_vram = torch.cuda.memory_allocated() / (1024**3)
    max_vram = torch.cuda.max_memory_allocated() / (1024**3)
    print(f"📊 Final VRAM: {final_vram:.2f} GB")
    print(f"📊 Peak VRAM: {max_vram:.2f} GB")

    print("✅ ULTRA-DEEP SINGLE ENV TRAINING COMPLETE!")


if __name__ == "__main__":
    main()
