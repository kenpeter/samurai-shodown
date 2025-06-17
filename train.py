import os
import sys
import argparse
import time
import math
import logging
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import psutil
import numpy as np
from typing import Dict, Any, Optional, Union
import retro
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch.amp import autocast, GradScaler
import pickle

from wrapper import (
    SamuraiShowdownCustomWrapper,
    EfficientNetB3FeatureExtractor,
    LightweightEfficientNetFeatureExtractor,
    UltraLightCNNFeatureExtractor,
    HighPerformanceEfficientNetB3FeatureExtractor,
)

# Setup logging
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class DeepCNNFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 512):
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        self.conv_layers = nn.Sequential(
            nn.Conv2d(n_input_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
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
        logger.info(
            f"Enhanced CNN Network: Input {observation_space.shape}, Output {features_dim}"
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        observations = observations.float() / 255.0
        if torch.any(torch.isnan(observations)) or torch.any(torch.isinf(observations)):
            logger.warning("Invalid observations detected in DeepCNNFeatureExtractor")
        features = self.conv_layers(observations)
        features = self.fc_layers(features)
        return features


class TrainingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.last_stats_log = 0

    def _on_step(self) -> bool:
        if (
            self.num_timesteps % 10000 == 0
            and self.num_timesteps != self.last_stats_log
        ):
            self.last_stats_log = self.num_timesteps
            if hasattr(self.training_env, "get_attr"):
                try:
                    env_stats = self.training_env.get_attr("current_stats")[0]
                    win_rate = env_stats.get("win_rate", 0) * 100
                    wins = env_stats.get("wins", 0)
                    losses = env_stats.get("losses", 0)
                    logger.info(f"TRAINING UPDATE - Step {self.num_timesteps:,}")
                    logger.info(f"Win Rate: {win_rate:.1f}%")
                    logger.info(f"Record: {wins}W/{losses}L")
                except Exception as e:
                    if self.verbose:
                        logger.warning(f"Could not get training stats: {e}")
        return True


def calculate_maximum_batch_size(obs_shape, target_vram_gb=9.0, force_batch_size=None):
    num_frames, height, width = obs_shape
    obs_size_bytes = num_frames * height * width * 4
    obs_size_mb = obs_size_bytes / (1024 * 1024)
    logger.info(f"FIGHTING GAME BATCH SIZE CALCULATION:")
    logger.info(f"Observation size: {obs_size_mb:.2f} MB per sample")
    if force_batch_size:
        obs_vram_gb = (force_batch_size * obs_size_bytes) / (1024**3)
        model_vram_gb = 0.06
        activation_vram_gb = obs_vram_gb * 3
        total_vram_gb = obs_vram_gb + model_vram_gb + activation_vram_gb + 1.5
        logger.info(f"FORCED batch size: {force_batch_size:,}")
        logger.info(f"Estimated VRAM needed: {total_vram_gb:.1f} GB")
        if total_vram_gb > target_vram_gb:
            logger.warning(
                f"May exceed {target_vram_gb}GB VRAM limit! Consider mixed precision or gradient checkpointing"
            )
        return force_batch_size
    estimated_model_vram = 5.0
    available_vram = target_vram_gb - estimated_model_vram
    available_vram_bytes = available_vram * 1024 * 1024 * 1024
    memory_per_sample = obs_size_bytes * 10.0
    max_batch_size = int(available_vram_bytes / memory_per_sample)
    optimal_batch_size = (
        2 ** int(math.log2(max_batch_size)) if max_batch_size > 0 else 64
    )
    optimal_batch_size = max(optimal_batch_size, 128)
    optimal_batch_size = min(optimal_batch_size, 512)
    logger.info(f"SAFE batch size: {optimal_batch_size:,}")
    return optimal_batch_size


def check_system_resources(n_steps, obs_shape, batch_size):
    logger.info(f"ENHANCED SYSTEM CHECK:")
    cpu_cores = psutil.cpu_count(logical=True)
    logger.info(f"CPU Cores: {cpu_cores}")
    ram_gb = psutil.virtual_memory().total / (1024**3)
    num_frames, height, width = obs_shape
    obs_size_mb = (num_frames * height * width * 4) / (1024 * 1024)
    buffer_memory_gb = (n_steps * obs_size_mb) / 1024
    pattern_memory_gb = 0.5
    logger.info(f"RAM: {ram_gb:.1f} GB")
    logger.info(f"Buffer memory: {buffer_memory_gb:.1f} GB")
    logger.info(f"Total required: {buffer_memory_gb + pattern_memory_gb:.1f} GB")
    return (buffer_memory_gb + pattern_memory_gb) < ram_gb * 0.4


def get_observation_dims(game, state):
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
        logger.warning(f"Could not determine observation dimensions: {e}")
        return (27, 180, 126)


def linear_schedule(initial_value, final_value=0.0, decay_type="linear"):
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


def cleanup_log_folders():
    folders_to_remove = ["logs_simple", "logs", "tensorboard_logs", "tb_logs"]
    for folder in folders_to_remove:
        if os.path.exists(folder):
            try:
                import shutil

                shutil.rmtree(folder)
                logger.info(f"Removed log folder: {folder}")
            except Exception as e:
                logger.warning(f"Could not remove {folder}: {e}")


def save_model_state(model, path):
    """Save only the model state dictionary to avoid pickling issues."""
    try:
        state_dict = {
            "policy_state_dict": model.policy.state_dict(),
            "optimizer_state_dict": model.policy.optimizer.state_dict(),
            "n_steps": model.n_steps,
            "batch_size": model.batch_size,
            "n_epochs": model.n_epochs,
            "gamma": model.gamma,
            "gae_lambda": model.gae_lambda,
            "ent_coef": model.ent_coef,
            "vf_coef": model.vf_coef,
            "max_grad_norm": model.max_grad_norm,
        }
        torch.save(state_dict, path)
        logger.info(f"Model state saved: {path}")
    except Exception as e:
        logger.error(f"Failed to save model state: {e}")


def load_model_state(model, path, env, device):
    """Load model state dictionary."""
    try:
        checkpoint = torch.load(path, map_location=device)
        model.policy.load_state_dict(checkpoint["policy_state_dict"])
        model.policy.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        model.n_steps = checkpoint["n_steps"]
        model.batch_size = checkpoint["batch_size"]
        model.n_epochs = checkpoint["n_epochs"]
        model.gamma = checkpoint["gamma"]
        model.gae_lambda = checkpoint["gae_lambda"]
        model.ent_coef = checkpoint["ent_coef"]
        model.vf_coef = checkpoint["vf_coef"]
        model.max_grad_norm = checkpoint["max_grad_norm"]
        model._setup_model()
        logger.info(f"Model state loaded: {path}")
        return model
    except Exception as e:
        logger.error(f"Failed to load model state: {e}")
        return model


def validate_policy(model, env, device, n_steps=5):
    """Validate policy outputs for multiple environment steps."""
    logger.info("Validating policy outputs...")
    obs, _ = env.reset()
    for i in range(n_steps):
        obs_tensor = torch.tensor(obs, device=device).float().unsqueeze(0)
        try:
            with torch.no_grad():
                action, value, log_prob = model.policy(obs_tensor)
                logger.info(
                    f"Step {i+1} - Action: {action.shape}, Value: {value.shape}, Log Prob: {log_prob.shape}"
                )
                if torch.any(torch.isnan(action)) or torch.any(torch.isinf(action)):
                    logger.warning(f"Invalid actions at step {i+1}")
                if torch.any(torch.isnan(value)) or torch.any(torch.isinf(value)):
                    logger.warning(f"Invalid values at step {i+1}")
                if torch.any(torch.isnan(log_prob)) or torch.any(torch.isinf(log_prob)):
                    logger.warning(f"Invalid log probabilities at step {i+1}")
        except Exception as e:
            logger.error(f"Policy validation failed at step {i+1}: {e}")
        action = env.action_space.sample()  # Random action for testing
        obs, reward, done, _, _ = env.step(action)
        if done:
            obs, _ = env.reset()
    logger.info("Policy validation completed")


def debug_rollout_buffer(model):
    """Log rollout buffer data to check for invalid values."""
    buffer = model.rollout_buffer
    logger.debug(f"Rollout buffer size: {buffer.buffer_size}")
    for i in range(buffer.buffer_size):
        obs = buffer.observations[i]
        actions = buffer.actions[i]
        rewards = buffer.rewards[i]
        if np.any(np.isnan(obs)) or np.any(np.isinf(obs)):
            logger.warning(f"Invalid observation in rollout buffer at index {i}")
        if np.any(np.isnan(actions)) or np.any(np.isinf(actions)):
            logger.warning(f"Invalid action in rollout buffer at index {i}")
        if np.isnan(rewards) or np.isinf(rewards):
            logger.warning(f"Invalid reward in rollout buffer at index {i}")


def main():
    parser = argparse.ArgumentParser(description="Improved Samurai Showdown Training")
    parser.add_argument("--total-timesteps", type=int, default=10000000)
    parser.add_argument("--learning-rate", type=float, default=4e-4)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--use-default-state", action="store_true")
    parser.add_argument("--target-vram", type=float, default=8.0)
    parser.add_argument("--n-steps", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--mixed-precision", action="store_true", default=True)
    parser.add_argument("--no-accumulation", action="store_true", default=True)
    parser.add_argument(
        "--lr-schedule",
        type=str,
        default="cosine",
        choices=["linear", "cosine", "exponential"],
    )
    parser.add_argument(
        "--model-size",
        type=str,
        default="full",
        choices=["ultra-light", "lightweight", "full", "basic", "high-performance"],
    )
    args = parser.parse_args()

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    logger.info(f"🚀 IMPROVED SAMURAI TRAINING")
    logger.info(f"Device: {device}")
    logger.info(f"Model size: {args.model_size}")
    logger.info(
        f"Hyperparameters: n_steps={args.n_steps}, batch_size={args.batch_size}"
    )
    logger.info(f"Target VRAM: {args.target_vram}GB")
    if args.mixed_precision:
        logger.info(f"Mixed precision training enabled")
    if args.no_accumulation:
        logger.info(f"Gradient accumulation disabled")

    game = "SamuraiShodown-Genesis"
    logger.info(f"Testing {game}...")
    try:
        test_env = retro.make(game=game, state=None)
        test_env.close()
        logger.info(f"Environment test passed")
    except Exception as e:
        logger.error(f"Environment test failed: {e}")
        return

    if args.use_default_state:
        state = None
        logger.info(f"Using default game state")
    else:
        if os.path.exists("samurai.state"):
            state = os.path.abspath("samurai.state")
            logger.info(f"Using samurai.state file: {state}")
        else:
            logger.warning(f"samurai.state not found, using default state")
            state = None

    obs_shape = get_observation_dims(game, state)
    logger.info(f"Observation shape: {obs_shape}")

    max_batch_size = calculate_maximum_batch_size(
        obs_shape, args.target_vram, force_batch_size=args.batch_size
    )

    if not check_system_resources(args.n_steps, obs_shape, max_batch_size):
        logger.error("Insufficient system resources")
        return

    n_steps = args.n_steps
    batch_size = max_batch_size
    n_envs = 1
    total_buffer_size = n_steps * n_envs
    if batch_size > total_buffer_size:
        batch_size = total_buffer_size
        logger.info(f"Adjusted batch size to {batch_size} to avoid truncation")
    else:
        factors = [
            i for i in range(1, total_buffer_size + 1) if total_buffer_size % i == 0
        ]
        suitable_factors = [f for f in factors if f <= batch_size]
        if suitable_factors:
            batch_size = max(suitable_factors)
            logger.info(
                f"Adjusted batch size to {batch_size} (largest factor of {total_buffer_size})"
            )
        else:
            batch_size = total_buffer_size
            logger.info(f"Set batch size to {batch_size} as fallback")

    logger.info(f"OPTIMIZED PARAMETERS:")
    logger.info(f"Batch size: {batch_size:,}")
    logger.info(f"N-steps: {n_steps:,}")

    logger.info(f"Creating environment...")
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
            max_episode_steps=15000,
        )
        env = Monitor(env)
        logger.info(f"Environment created")
    except Exception as e:
        logger.error(f"Failed to create environment: {e}")
        return

    save_dir = "trained_models_fighting_optimized"
    os.makedirs(save_dir, exist_ok=True)

    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    vram_before = torch.cuda.memory_allocated() / (1024**3) if device == "cuda" else 0
    logger.info(f"VRAM before model: {vram_before:.2f} GB")

    if args.model_size == "high-performance":
        feature_extractor_class = HighPerformanceEfficientNetB3FeatureExtractor
        logger.info(f"Using HIGH-PERFORMANCE EfficientNet-B3 (~12GB VRAM)")
        features_dim = 1024
    elif args.model_size == "ultra-light":
        feature_extractor_class = UltraLightCNNFeatureExtractor
        logger.info(f"Using ULTRA-LIGHT CNN (~1GB VRAM)")
        features_dim = 256
    elif args.model_size == "lightweight":
        feature_extractor_class = LightweightEfficientNetFeatureExtractor
        logger.info(f"Using LIGHTWEIGHT EfficientNet-B0 (~3GB VRAM)")
        features_dim = 512
    elif args.model_size == "full":
        feature_extractor_class = EfficientNetB3FeatureExtractor
        logger.info(f"Using FULL EfficientNet-B3 (~8-10GB VRAM)")
        features_dim = 512
    else:
        feature_extractor_class = DeepCNNFeatureExtractor
        logger.info(f"Using BASIC CNN (~2GB VRAM)")
        features_dim = 512

    if args.model_size == "high-performance" and args.target_vram < 12.0:
        batch_size = 128
        logger.info(f"Set batch size to {batch_size} for 8GB VRAM compatibility")
    elif args.model_size == "ultra-light" and batch_size > 64:
        batch_size = 64
        logger.info(f"Reduced batch size to {batch_size} for ultra-light model")

    if args.resume and os.path.exists(args.resume):
        logger.info(f"Loading model: {args.resume}")
        model = PPO.load(args.resume, env=env, device=device)
        model.n_steps = n_steps
        model.batch_size = batch_size
        model._setup_model()
    else:
        logger.info(f"Creating PPO model with {args.model_size.upper()} architecture")
        lr_schedule = linear_schedule(
            args.learning_rate, args.learning_rate * 0.1, args.lr_schedule
        )
        if args.model_size == "high-performance":
            net_arch = dict(pi=[1024, 512, 256], vf=[1024, 512, 256])
        elif args.model_size == "ultra-light":
            net_arch = dict(pi=[256, 128], vf=[256, 128])
        else:
            net_arch = dict(pi=[512, 256, 128], vf=[512, 256, 128])
        model = PPO(
            "CnnPolicy",
            env,
            device=device,
            verbose=1,
            n_steps=n_steps,
            batch_size=batch_size,
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
                features_extractor_class=feature_extractor_class,
                features_extractor_kwargs=dict(features_dim=features_dim),
                normalize_images=False,
                optimizer_class=torch.optim.Adam,
                optimizer_kwargs=dict(eps=1e-5, weight_decay=1e-4),
                net_arch=net_arch,
                activation_fn=nn.ReLU,
            ),
        )

    if device == "cuda":
        vram_after = torch.cuda.memory_allocated() / (1024**3)
        model_vram = vram_after - vram_before
        logger.info(f"VRAM after model: {vram_after:.2f} GB")
        logger.info(f"Model VRAM: {model_vram:.2f} GB")

    # Validate policy before training
    validate_policy(model, env, device, n_steps=10)

    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path=save_dir,
        name_prefix="ppo_fighting_optimized",
    )
    training_callback = TrainingCallback(verbose=1)

    start_time = time.time()
    logger.info(f"Starting TRAINING")
    try:
        if args.mixed_precision and device == "cuda" and not args.no_accumulation:
            logger.info(
                f"Training with mixed precision (FP16) and gradient accumulation"
            )
            scaler = GradScaler("cuda")
            effective_batch_size = 512
            accumulation_steps = effective_batch_size // batch_size
            original_train = model.train

            def custom_train():
                model.policy.train()
                model.policy.zero_grad()
                loss_sum = 0.0
                valid_steps = 0
                debug_rollout_buffer(model)  # Debug rollout buffer
                for i in range(accumulation_steps):
                    logger.debug(f"Accumulation step {i+1}/{accumulation_steps}")
                    with autocast("cuda"):
                        loss = original_train()
                    logger.debug(
                        f"Loss type: {type(loss)}, value: {loss.item() if isinstance(loss, torch.Tensor) else loss}"
                    )
                    if isinstance(loss, torch.Tensor) and not (
                        torch.isnan(loss) or torch.isinf(loss)
                    ):
                        scaler.scale(loss).backward()
                        loss_sum += loss.item()
                        valid_steps += 1
                    else:
                        logger.warning(
                            f"Invalid loss in accumulation step {i+1}: {type(loss)} {loss}"
                        )
                        continue
                if valid_steps == 0:
                    logger.error("All accumulation steps produced invalid losses")
                    return 0.0
                scaler.unscale_(model.policy.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.policy.parameters(), model.max_grad_norm
                )
                scaler.step(model.policy.optimizer)
                scaler.update()
                return loss_sum / valid_steps if valid_steps > 0 else 0.0

            model.train = custom_train
        elif args.mixed_precision and device == "cuda":
            logger.info(
                f"Training with mixed precision (FP16) without gradient accumulation"
            )
            scaler = GradScaler("cuda")
            original_train = model.train

            def custom_train():
                model.policy.train()
                debug_rollout_buffer(model)  # Debug rollout buffer
                with autocast("cuda"):
                    loss = original_train()
                logger.debug(
                    f"Loss type: {type(loss)}, value: {loss.item() if isinstance(loss, torch.Tensor) else loss}"
                )
                if not isinstance(loss, torch.Tensor):
                    logger.error(f"Invalid loss type: {type(loss)}")
                    return 0.0
                if torch.isnan(loss) or torch.isinf(loss):
                    logger.error(f"Invalid loss value: {loss.item()}")
                    return 0.0
                model.policy.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(model.policy.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.policy.parameters(), model.max_grad_norm
                )
                scaler.step(model.policy.optimizer)
                scaler.update()
                return loss.item()

            model.train = custom_train

        model.learn(
            total_timesteps=args.total_timesteps,
            callback=[checkpoint_callback, training_callback],
            reset_num_timesteps=not bool(args.resume),
        )
        training_time = time.time() - start_time
        logger.info(f"Training completed in {training_time/3600:.1f} hours!")
        if hasattr(env, "current_stats"):
            final_stats = env.current_stats
            logger.info(f"FINAL PERFORMANCE:")
            logger.info(f"Final Win Rate: {final_stats['win_rate']*100:.1f}%")
            logger.info(f"Total Rounds: {final_stats['total_rounds']}")
    except KeyboardInterrupt:
        logger.info(f"Training interrupted")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback

        traceback.print_exc()
    finally:
        env.close()
        torch.cuda.empty_cache()

    final_path = os.path.join(save_dir, "ppo_fighting_optimized_final.pth")
    save_model_state(model, final_path)

    cleanup_log_folders()

    if device == "cuda":
        final_vram = torch.cuda.memory_allocated() / (1024**3)
        max_vram = torch.cuda.max_memory_allocated() / (1024**3)
        logger.info(f"Final VRAM: {final_vram:.2f} GB")
        logger.info(f"Peak VRAM: {max_vram:.2f} GB")

    logger.info("TRAINING COMPLETE!")
    logger.info(f"\nUSAGE INSTRUCTIONS:")
    logger.info(
        f"Run with: python train.py --batch-size 128 --n-steps 128 --target-vram 8.0 --mixed-precision --model-size full"
    )


if __name__ == "__main__":
    main()
