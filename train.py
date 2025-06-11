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
    """MEMORY-OPTIMIZED CNN feature extractor"""

    def __init__(
        self, observation_space: gym.spaces.Box, features_dim: int = 256
    ):  # REDUCED from 1024
        super().__init__(observation_space, features_dim)

        n_input_channels = observation_space.shape[0]  # 9 frames

        # SIMPLIFIED network architecture
        self.conv_layers = nn.Sequential(
            # Initial feature extraction
            nn.Conv2d(
                n_input_channels, 32, kernel_size=5, stride=2, padding=2
            ),  # REDUCED from 64
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Efficient conv blocks
            self._make_conv_block(32, 64, 3, 1, 1),
            self._make_conv_block(64, 64, 3, 2, 1),
            self._make_conv_block(64, 128, 3, 1, 1),
            self._make_conv_block(128, 128, 3, 2, 1),
            self._make_conv_block(128, 256, 3, 1, 1),
            self._make_conv_block(256, 256, 3, 2, 1),
            # Final features
            nn.Conv2d(
                256, 512, kernel_size=3, stride=1, padding=1
            ),  # REDUCED from 1024
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )

        # SIMPLIFIED FC layers
        self.fc_layers = nn.Sequential(
            nn.Linear(512, 512),  # SIMPLIFIED
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, features_dim),  # Output features_dim
            nn.ReLU(inplace=True),
        )

        print(f"🧠 MEMORY-OPTIMIZED CNN Network:")
        print(f"   📊 Input: {observation_space.shape}")
        print(f"   🔥 Conv layers: 9 layers (memory efficient)")
        print(f"   💪 Channels: 9 → 32 → 64 → 128 → 256 → 512")
        print(f"   🎯 FC layers: 2 layers → {features_dim}")
        print(f"   🛡️ MEMORY SAFE for 11GB VRAM")

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


def calculate_maximum_batch_size(obs_shape, target_vram_gb=8.0):  # REDUCED from 10.0
    """MEMORY-SAFE batch size calculation"""
    num_frames, height, width = obs_shape
    obs_size_bytes = num_frames * height * width * 4
    obs_size_mb = obs_size_bytes / (1024 * 1024)

    print(f"📊 MEMORY-SAFE BATCH SIZE:")
    print(f"   Observation size: {obs_size_mb:.2f} MB per sample")

    estimated_model_vram = 4.0  # CONSERVATIVE estimate
    available_vram = target_vram_gb - estimated_model_vram
    available_vram_bytes = available_vram * 1024 * 1024 * 1024

    memory_per_sample = obs_size_bytes * 8.0  # CONSERVATIVE multiplier
    max_batch_size = int(available_vram_bytes / memory_per_sample)

    optimal_batch_size = (
        2 ** int(math.log2(max_batch_size)) if max_batch_size > 0 else 64
    )
    optimal_batch_size = max(optimal_batch_size, 64)  # Minimum
    optimal_batch_size = min(optimal_batch_size, 1024)  # REDUCED maximum

    print(f"   SAFE batch size: {optimal_batch_size:,}")
    print(f"   🛡️ MEMORY-SAFE for 11GB VRAM")

    return optimal_batch_size


def check_system_resources(n_steps, obs_shape, batch_size):
    """Check system resources for single environment setup"""
    print(f"🖥️  SINGLE ENV SYSTEM CHECK:")
    cpu_cores = psutil.cpu_count(logical=True)
    print(f"   CPU Cores: {cpu_cores} (1 env needs minimal CPU)")
    ram_gb = psutil.virtual_memory().total / (1024**3)

    num_frames, height, width = obs_shape
    obs_size_mb = (num_frames * height * width * 4) / (1024 * 1024)
    buffer_memory_gb = (n_steps * obs_size_mb) / 1024

    print(f"   RAM: {ram_gb:.1f} GB")
    print(f"   Buffer memory: {buffer_memory_gb:.1f} GB")
    print(f"   Strategy: 1 env = minimal RAM usage, maximum VRAM for batches")

    return buffer_memory_gb < ram_gb * 0.3


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
        return (9, 126, 180)


def linear_schedule(initial_value, final_value=0.0):
    """Linear learning rate schedule"""

    def scheduler(progress):
        return final_value + progress * (initial_value - final_value)

    return scheduler


def main():
    parser = argparse.ArgumentParser(description="Memory-Safe Ultra-Deep Training")
    parser.add_argument("--total-timesteps", type=int, default=50000000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--use-default-state", action="store_true")
    parser.add_argument("--target-vram", type=float, default=8.0)  # REDUCED from 10.0
    parser.add_argument("--n-steps", type=int, default=2048)  # REDUCED from 16384

    args = parser.parse_args()

    # MEMORY OPTIMIZATION
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    device = "cuda"
    torch.cuda.empty_cache()

    print(f"🚀 MEMORY-SAFE TRAINING")
    print(f"   💻 Device: {device}")
    print(f"   🛡️ Memory optimized for 11GB VRAM")

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

    # MEMORY-SAFE batch size
    max_batch_size = calculate_maximum_batch_size(obs_shape, args.target_vram)

    # System check
    if not check_system_resources(args.n_steps, obs_shape, max_batch_size):
        print("❌ Insufficient system resources")
        return

    n_steps = args.n_steps
    batch_size = min(max_batch_size, n_steps)

    print(f"🔥 MEMORY-SAFE PARAMETERS:")
    print(f"   🎮 Environments: 1")
    print(f"   💪 Batch size: {batch_size:,}")
    print(f"   📏 N-steps: {n_steps:,}")
    print(f"   🛡️ Memory optimized")

    # Create environment
    print(f"🔧 Creating environment...")
    try:
        env = retro.make(
            game=game,
            state=state,
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
        print(f"✅ Environment created")

    except Exception as e:
        print(f"❌ Failed to create environment: {e}")
        return

    # Create model
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
        print(f"🚀 Creating MEMORY-SAFE PPO model")

        lr_schedule = linear_schedule(args.learning_rate, args.learning_rate * 0.1)

        model = PPO(
            "CnnPolicy",
            env,
            device=device,
            verbose=1,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=2,  # REDUCED from 3
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
                features_extractor_kwargs=dict(features_dim=256),  # REDUCED from 512
                normalize_images=False,
                optimizer_class=torch.optim.Adam,
                optimizer_kwargs=dict(eps=1e-5, weight_decay=1e-4),
                net_arch=dict(
                    pi=[256, 128],  # REDUCED from [512, 256, 128]
                    vf=[256, 128],  # REDUCED from [512, 256, 128]
                ),
                activation_fn=nn.ReLU,
            ),
        )

    # Monitor VRAM
    vram_after = torch.cuda.memory_allocated() / (1024**3)
    model_vram = vram_after - vram_before
    print(f"   VRAM after model: {vram_after:.2f} GB")
    print(f"   Model VRAM: {model_vram:.2f} GB")

    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=100000,
        save_path=save_dir,
        name_prefix="ppo_memory_safe",
    )

    # Training
    start_time = time.time()
    print(f"🏋️ Starting MEMORY-SAFE training")

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
    final_path = os.path.join(save_dir, "ppo_memory_safe_final.zip")
    model.save(final_path)
    print(f"💾 Model saved: {final_path}")

    # Final VRAM report
    final_vram = torch.cuda.memory_allocated() / (1024**3)
    max_vram = torch.cuda.max_memory_allocated() / (1024**3)
    print(f"📊 Final VRAM: {final_vram:.2f} GB")
    print(f"📊 Peak VRAM: {max_vram:.2f} GB")

    print("✅ MEMORY-SAFE TRAINING COMPLETE!")


if __name__ == "__main__":
    main()
