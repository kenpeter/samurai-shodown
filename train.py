#!/usr/bin/env python3
"""
train.py - Complete Training Script for Vision Pipeline Fighting Game AI
Optimized for 180×128 resolution with OpenCV detection + CNN + Vision Transformer
"""

import os
import argparse
import time
import torch
import torch.nn as nn
import retro

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from wrapper import VisionPipelineWrapper, JEPAEnhancedCNN


class VisionPipelineCallback(BaseCallback):
    """Callback to monitor vision pipeline training progress"""

    def __init__(self, enable_vision_transformer=True, verbose=0):
        super(VisionPipelineCallback, self).__init__(verbose)
        self.enable_vision_transformer = enable_vision_transformer
        self.last_log_time = time.time()
        self.log_interval = 60  # Log every 60 seconds

    def _on_step(self) -> bool:
        current_time = time.time()
        if current_time - self.last_log_time > self.log_interval:
            self.last_log_time = current_time
            self._log_training_stats()
        return True

    def _log_training_stats(self):
        """Log comprehensive training statistics"""
        try:
            print(
                f"\n--- 📊 Vision Pipeline Training @ Step {self.num_timesteps:,} ---"
            )

            # Get environment statistics
            all_stats = self.training_env.get_attr("stats")
            if all_stats and len(all_stats) > 0:
                stats = all_stats[0]

                # OpenCV detection statistics
                fire_knives = stats.get("fire_knives_detected", 0)
                bombs = stats.get("bombs_detected", 0)
                print(f"   🔍 OpenCV Detections (Total):")
                print(f"      🔥 Fire Knives: {fire_knives:,}")
                print(f"      💣 Bombs: {bombs:,}")

                # Vision transformer statistics
                if self.enable_vision_transformer:
                    vt_ready = stats.get("vision_transformer_ready", False)
                    predictions = stats.get("predictions_made", 0)

                    if vt_ready:
                        print(
                            f"   🧠 Vision Transformer: Ready ({predictions:,} predictions made)"
                        )
                    else:
                        print("   🧠 Vision Transformer: Initializing...")
                else:
                    print("   🧠 Vision Transformer: Disabled (OpenCV + CNN only)")

                # Detection rate analysis
                total_frames = self.num_timesteps
                if total_frames > 0:
                    fire_rate = fire_knives / total_frames * 1000
                    bomb_rate = bombs / total_frames * 1000
                    print(f"   📈 Detection Rates:")
                    print(f"      Fire knives: {fire_rate:.2f} per 1000 frames")
                    print(f"      Bombs: {bomb_rate:.2f} per 1000 frames")

            # System and training statistics
            self._log_system_stats()

            print("--------------------------------------------------")

        except Exception as e:
            print(f"   ⚠️ Logging error: {e}")

    def _log_system_stats(self):
        """Log system and training parameters"""
        try:
            # Memory usage
            if torch.cuda.is_available():
                vram_alloc = torch.cuda.memory_allocated() / (1024**3)
                vram_cached = torch.cuda.memory_reserved() / (1024**3)
                print(
                    f"   💾 VRAM: {vram_alloc:.2f}GB allocated / {vram_cached:.2f}GB cached"
                )

            # Learning rate
            if hasattr(self.model, "learning_rate"):
                lr = self.model.learning_rate
                if callable(lr):
                    lr = lr(self.model._current_progress_remaining)
                print(f"   📈 Learning Rate: {lr:.2e}")

            # Architecture info
            arch_info = "OpenCV + CNN"
            if self.enable_vision_transformer:
                arch_info += " + Vision Transformer"
            print(f"   🏗️ Architecture: {arch_info}")

        except Exception as e:
            print(f"   ⚠️ System stats error: {e}")


def make_env(game, state_path, render_mode, frame_stack, enable_vision_transformer):
    """Environment factory function"""

    def _init():
        try:
            # Create base retro environment
            env = retro.make(
                game=game,
                state=state_path,
                use_restricted_actions=retro.Actions.FILTERED,
                obs_type=retro.Observations.IMAGE,
                render_mode=render_mode,
            )

            # Wrap with vision pipeline (180×128 resolution)
            env = VisionPipelineWrapper(
                env,
                frame_stack=frame_stack,
                enable_vision_transformer=enable_vision_transformer,
            )

            # Monitor wrapper for episode statistics
            env = Monitor(env)

            return env

        except Exception as e:
            print(f"❌ Environment creation failed: {e}")
            raise

    return _init


def parse_arguments():
    """Parse and validate command line arguments"""
    parser = argparse.ArgumentParser(
        description="Vision Pipeline Fighting Game AI Training (180×128 optimized)"
    )

    # Training hyperparameters
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=10_000_000,
        help="Total training timesteps (default: 10M)",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=4096,
        help="Steps per environment per update (default: 4096)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=512, help="Minibatch size (default: 512)"
    )
    parser.add_argument(
        "--lr", type=float, default=2.5e-4, help="Learning rate (default: 2.5e-4)"
    )
    parser.add_argument(
        "--ent-coef",
        type=float,
        default=0.01,
        help="Entropy coefficient for exploration (default: 0.01)",
    )

    # Model configuration
    parser.add_argument(
        "--frame-stack",
        type=int,
        default=8,
        help="Number of frames to stack (default: 8)",
    )
    parser.add_argument(
        "--no-vision-transformer",
        action="store_true",
        help="Disable Vision Transformer (use OpenCV + CNN only)",
    )
    parser.add_argument(
        "--features-dim",
        type=int,
        default=512,
        help="CNN feature dimensions (default: 512)",
    )

    # Training utilities
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to model checkpoint to resume training",
    )
    parser.add_argument(
        "--render", action="store_true", help="Render the environment during training"
    )
    parser.add_argument(
        "--save-freq",
        type=int,
        default=100_000,
        help="Save model every N steps (default: 100k)",
    )

    # Game configuration
    parser.add_argument(
        "--game",
        type=str,
        default="SamuraiShodown-Genesis",
        help="Retro game to use (default: SamuraiShodown-Genesis)",
    )
    parser.add_argument(
        "--state",
        type=str,
        default="samurai.state",
        help="Game state file (default: samurai.state)",
    )

    args = parser.parse_args()

    # Validation
    if args.total_timesteps <= 0:
        raise ValueError("total-timesteps must be positive")
    if args.frame_stack < 1:
        raise ValueError("frame-stack must be at least 1")
    if args.lr <= 0:
        raise ValueError("learning rate must be positive")
    if args.features_dim < 64:
        raise ValueError("features-dim must be at least 64")

    # Set derived arguments
    args.enable_vision_transformer = not args.no_vision_transformer

    return args


def setup_model(env, args, device, save_dir):
    """Setup PPO model with optimized vision pipeline configuration"""

    # Policy configuration optimized for vision pipeline
    policy_kwargs = dict(
        features_extractor_class=JEPAEnhancedCNN,
        features_extractor_kwargs=dict(features_dim=args.features_dim),
        net_arch=dict(
            pi=[256, 128],  # Policy network architecture
            vf=[256, 128],  # Value network architecture
        ),
        activation_fn=nn.ReLU,
        optimizer_class=torch.optim.AdamW,
        optimizer_kwargs=dict(eps=1e-5, weight_decay=1e-4),  # L2 regularization
    )

    # Resume training if checkpoint exists
    if args.resume and os.path.exists(args.resume):
        print(f"📂 Resuming training from {args.resume}")
        try:
            model = PPO.load(
                args.resume,
                env=env,
                device=device,
                custom_objects={"learning_rate": args.lr},
            )
            print("   ✅ Model loaded successfully")

        except Exception as e:
            print(f"   ❌ Failed to load model: {e}")
            print("   🚀 Creating new model instead...")
            model = create_new_model(env, args, policy_kwargs, save_dir, device)
    else:
        print("🚀 Creating new model...")
        model = create_new_model(env, args, policy_kwargs, save_dir, device)

    return model


def create_new_model(env, args, policy_kwargs, save_dir, device):
    """Create new PPO model with optimized hyperparameters"""
    return PPO(
        "CnnPolicy",
        env,
        policy_kwargs=policy_kwargs,
        # Training hyperparameters
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=4,
        gamma=0.99,  # Discount factor
        gae_lambda=0.95,  # GAE lambda
        clip_range=0.2,  # PPO clip range
        clip_range_vf=None,  # Value function clip range
        ent_coef=args.ent_coef,  # Entropy coefficient
        vf_coef=0.5,  # Value function coefficient
        max_grad_norm=0.5,  # Gradient clipping
        learning_rate=args.lr,
        verbose=1,
        tensorboard_log=f"./{save_dir}_logs/",
        device=device,
    )


def inject_vision_components(env, model, enable_vision_transformer):
    """Inject CNN feature extractor into vision pipeline wrapper"""
    try:
        print("   💉 Injecting CNN feature extractor into Vision Pipeline...")

        # Navigate environment hierarchy
        # DummyVecEnv -> Monitor -> VisionPipelineWrapper
        monitor_env = env.envs[0]
        wrapper_env = monitor_env.env

        if hasattr(wrapper_env, "inject_feature_extractor"):
            wrapper_env.inject_feature_extractor(model.policy.features_extractor)

            if enable_vision_transformer:
                print(
                    "   ✅ Complete Vision Pipeline (OpenCV + CNN + Transformer) initialized!"
                )
            else:
                print("   ✅ Partial Vision Pipeline (OpenCV + CNN) initialized!")
        else:
            print(f"   ⚠️ Warning: Could not find inject_feature_extractor method")
            print(f"   Environment type: {type(wrapper_env)}")
            available_methods = [
                attr
                for attr in dir(wrapper_env)
                if not attr.startswith("_") and callable(getattr(wrapper_env, attr))
            ]
            print(f"   Available methods: {available_methods[:5]}...")

    except Exception as e:
        print(f"   ❌ Vision Pipeline injection failed: {e}")
        print("   Training will continue with base CNN only")


def validate_setup(env, args):
    """Validate that the vision pipeline is properly configured"""
    try:
        print("   🔍 Validating vision pipeline setup...")

        monitor_env = env.envs[0]
        wrapper_env = monitor_env.env

        # Check wrapper type
        if not hasattr(wrapper_env, "opencv_detector"):
            print("   ⚠️ Warning: OpenCV detector not found")
            return False

        # Check OpenCV detector configuration
        detector = wrapper_env.opencv_detector
        print(f"   ✅ OpenCV detector configured:")
        print(
            f"      🔥 Fire knife area: {detector.fire_knife_min_area}-{detector.fire_knife_max_area}px²"
        )
        print(
            f"      💣 Bomb area: {detector.bomb_min_area}-{detector.bomb_max_area}px²"
        )
        print(f"      📏 Floor threshold: y > {detector.floor_threshold}")

        # Check observation space
        obs_shape = wrapper_env.observation_space.shape
        expected_shape = (3 * args.frame_stack, 128, 180)  # CHW format
        if obs_shape != expected_shape:
            print(
                f"   ⚠️ Warning: Unexpected observation shape {obs_shape}, expected {expected_shape}"
            )
            return False

        print(f"   ✅ Observation space: {obs_shape} (CHW format)")
        print(f"   ✅ Frame resolution: 180×128 (optimized)")

        return True

    except Exception as e:
        print(f"   ⚠️ Setup validation failed: {e}")
        return False


def main():
    """Main training function"""
    try:
        # Parse arguments and setup
        args = parse_arguments()
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Print configuration
        mode_name = (
            "Complete Vision Pipeline"
            if args.enable_vision_transformer
            else "OpenCV + CNN"
        )
        print(f"🚀 Starting {mode_name} Training")
        print(f"   💻 Device: {device.upper()}")
        print(f"   🎯 Total Timesteps: {args.total_timesteps:,}")
        print(f"   📚 Frame Stack: {args.frame_stack}")
        print(f"   📏 Resolution: 180×128 (optimized)")
        print(f"   🔍 OpenCV Detection: ENABLED (fire knives + bombs)")
        print(
            f"   🧠 Vision Transformer: {'ENABLED' if args.enable_vision_transformer else 'DISABLED'}"
        )
        print(f"   🎮 Game: {args.game}")

        # Setup game state
        state_path = (
            os.path.abspath(args.state)
            if os.path.exists(args.state)
            else retro.State.DEFAULT
        )
        if state_path == retro.State.DEFAULT:
            print(f"   💾 State: DEFAULT (no custom state file found)")
        else:
            print(f"   💾 State: {state_path}")

        # Create environment
        render_mode = "human" if args.render else None
        env = DummyVecEnv(
            [
                make_env(
                    args.game,
                    state_path,
                    render_mode,
                    args.frame_stack,
                    args.enable_vision_transformer,
                )
            ]
        )

        # Validate setup
        setup_ok = validate_setup(env, args)
        if not setup_ok:
            print("   ⚠️ Warning: Setup validation failed, but continuing...")

        # Setup save directories
        save_dir = "trained_models_vision_pipeline"
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(f"{save_dir}_logs", exist_ok=True)
        print(f"   💾 Save directory: {save_dir}")

        # Setup model
        model = setup_model(env, args, device, save_dir)

        # Inject vision components
        inject_vision_components(env, model, args.enable_vision_transformer)

        # Setup training callbacks
        checkpoint_callback = CheckpointCallback(
            save_freq=max(args.save_freq, args.n_steps),
            save_path=save_dir,
            name_prefix="ppo_vision_pipeline",
        )

        training_callback = VisionPipelineCallback(
            enable_vision_transformer=args.enable_vision_transformer
        )

        # Start training
        print(f"\n🎯 Starting training with {mode_name}...")
        print("🔍 OpenCV Detection Targets:")
        print("   🔥 Big Fire Knives: 150-4000px² (12×12 to 63×63 pixels)")
        print("   💣 Small Bombs: 25-350px² (5×5 to 19×19 pixels)")

        if args.enable_vision_transformer:
            print("🧠 Vision Transformer Learning:")
            print("   📈 Visual patterns → Health/Score momentum associations")
            print("   🎯 Predictive action recommendations")

        # Execute training
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=[checkpoint_callback, training_callback],
            reset_num_timesteps=not bool(args.resume),
        )

        print("\n🎉 Training completed successfully!")

    except KeyboardInterrupt:
        print("\n⏸️ Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback

        print(f"Full traceback: {traceback.format_exc()}")
        raise
    finally:
        # Cleanup resources
        try:
            if "env" in locals():
                env.close()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("   🧹 Cleanup completed")
        except Exception as e:
            print(f"⚠️ Cleanup warning: {e}")

    # Save final model
    try:
        if "model" in locals():
            final_path = os.path.join(save_dir, "final_vision_pipeline_model.zip")
            model.save(final_path)
            print(f"💾 Final model saved to {final_path}")
    except Exception as e:
        print(f"⚠️ Failed to save final model: {e}")


if __name__ == "__main__":
    main()
