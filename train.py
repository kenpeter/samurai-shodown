#!/usr/bin/env python3
"""
JEPA-Enhanced Samurai Training Script - COMPREHENSIVE FIXES
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

from wrapper import SamuraiJEPAWrapper, JEPAEnhancedCNN


class JEPATrainingCallback(BaseCallback):
    def __init__(self, enable_jepa=True, verbose=0):
        super(JEPATrainingCallback, self).__init__(verbose)
        self.enable_jepa = enable_jepa
        self.last_log_time = time.time()
        self.log_interval = 60  # seconds

    def _on_step(self) -> bool:
        current_time = time.time()
        if current_time - self.last_log_time > self.log_interval:
            self.last_log_time = current_time
            self._log_training_stats()
        return True

    def _log_training_stats(self):
        """Comprehensive logging with proper error handling"""
        try:
            # --- Main Stats Logging ---
            all_current_stats = self.training_env.get_attr("current_stats")
            if all_current_stats and len(all_current_stats) > 0:
                stats = all_current_stats[0]
                wins = stats.get("wins", 0)
                losses = stats.get("losses", 0)
                total_rounds = wins + losses  # Fixed: Calculate total_rounds properly
                win_rate = (wins / total_rounds * 100) if total_rounds > 0 else 0

                print(f"\n--- 📊 Training Status @ Step {self.num_timesteps:,} ---")
                print(
                    f"   🎯 Win Rate: {win_rate:.1f}% ({wins}W / {losses}L / {total_rounds} total)"
                )

            # --- JEPA Accuracy Logging ---
            if self.enable_jepa:
                self._log_jepa_stats()

            # --- System and Training Param Logging ---
            self._log_system_stats()

        except Exception as e:
            print(f"   ⚠️ Logging error: {e}")

    def _log_jepa_stats(self):
        """Log JEPA individual accuracies only"""
        try:
            all_strategic_stats = self.training_env.get_attr("strategic_stats")
            all_jepa_ready = self.training_env.get_attr("jepa_ready")

            if all_strategic_stats and len(all_strategic_stats) > 0:
                strat_stats = all_strategic_stats[0]
                jepa_ready = all_jepa_ready[0] if all_jepa_ready else False

                if jepa_ready and strat_stats:
                    predictions_made = strat_stats.get("binary_predictions_made", 0)

                    if predictions_made > 0:
                        print(f"   🔮 JEPA Predictions Made: {predictions_made:,}")

                        # Show individual accuracies
                        individual_accuracies = strat_stats.get(
                            "individual_accuracies", {}
                        )
                        individual_predictions = strat_stats.get(
                            "individual_predictions", {}
                        )
                        individual_correct = strat_stats.get("individual_correct", {})

                        if individual_accuracies:
                            print("   📊 Individual Prediction Accuracies:")
                            for outcome, accuracy in individual_accuracies.items():
                                total_preds = individual_predictions.get(outcome, 0)
                                total_correct = individual_correct.get(outcome, 0)
                                # Create readable outcome names
                                readable_name = (
                                    outcome.replace("will_", "")
                                    .replace("_", " ")
                                    .title()
                                )
                                print(
                                    f"      • {readable_name}: {accuracy*100:.1f}% ({total_correct}/{total_preds})"
                                )
                    else:
                        print("   🔮 JEPA: Ready, awaiting predictions...")
                else:
                    print("   🔮 JEPA: Initializing predictor...")
        except Exception as e:
            print(f"   ⚠️ JEPA logging error: {e}")

    def _log_system_stats(self):
        """Log system and training parameters"""
        try:
            print(
                f"   🧠 Arch: {'JEPA+Transformer' if self.enable_jepa else 'Standard CNN'}"
            )

            if torch.cuda.is_available():
                vram_alloc = torch.cuda.memory_allocated() / (1024**3)
                vram_res = torch.cuda.memory_reserved() / (1024**3)
                print(
                    f"   💾 VRAM: {vram_alloc:.2f} GB alloc / {vram_res:.2f} GB reserved"
                )

            if hasattr(self.model, "learning_rate"):
                lr = self.model.learning_rate
                if callable(lr):
                    lr = lr(self.model._current_progress_remaining)
                print(f"   📈 LR: {lr:.2e}")

            print("--------------------------------------------------")
        except Exception as e:
            print(f"   ⚠️ System stats error: {e}")


def make_env(game, state_path, render_mode, frame_stack, enable_jepa):
    """Environment factory with proper error handling"""

    def _init():
        try:
            env = retro.make(
                game=game,
                state=state_path,
                use_restricted_actions=retro.Actions.FILTERED,
                obs_type=retro.Observations.IMAGE,
                render_mode=render_mode,
            )
            env = SamuraiJEPAWrapper(
                env, frame_stack=frame_stack, enable_jepa=enable_jepa
            )
            env = Monitor(env)
            return env
        except Exception as e:
            print(f"❌ Environment creation failed: {e}")
            raise

    return _init


def parse_arguments():
    """Parse command line arguments with validation"""
    parser = argparse.ArgumentParser(
        description="JEPA Enhanced Fighting Game AI Training"
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=20_000_000,
        help="Total training timesteps",
    )
    parser.add_argument(
        "--n-steps", type=int, default=4096, help="Steps per environment per update"
    )
    parser.add_argument("--batch-size", type=int, default=512, help="Minibatch size")
    parser.add_argument("--lr", type=float, default=2.5e-4, help="Learning rate")
    parser.add_argument(
        "--ent-coef", type=float, default=0.01, help="Entropy coefficient"
    )
    parser.add_argument(
        "--frame-stack", type=int, default=8, help="Number of frames to stack"
    )
    parser.add_argument(
        "--no-jepa", action="store_true", help="Disable JEPA components"
    )
    parser.add_argument(
        "--resume", type=str, default=None, help="Path to a model to resume"
    )
    parser.add_argument("--render", action="store_true", help="Render the environment")

    args = parser.parse_args()

    # Validation
    if args.total_timesteps <= 0:
        raise ValueError("total-timesteps must be positive")
    if args.frame_stack < 1:
        raise ValueError("frame-stack must be at least 1")
    if args.lr <= 0:
        raise ValueError("learning rate must be positive")

    args.enable_jepa = not args.no_jepa
    return args


def setup_model(env, args, device, save_dir):
    """Setup PPO model with proper configuration"""
    policy_kwargs = dict(
        features_extractor_class=JEPAEnhancedCNN,
        features_extractor_kwargs=dict(features_dim=512),
        net_arch=dict(pi=[256, 128], vf=[256, 128]),  # Slightly deeper network
        activation_fn=nn.ReLU,
        optimizer_class=torch.optim.AdamW,
        optimizer_kwargs=dict(eps=1e-5, weight_decay=1e-4),
    )

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
    """Create a new PPO model"""
    return PPO(
        "CnnPolicy",
        env,
        policy_kwargs=policy_kwargs,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        clip_range_vf=None,  # Let it adapt automatically
        ent_coef=args.ent_coef,
        vf_coef=0.5,
        max_grad_norm=0.5,
        learning_rate=args.lr,
        verbose=1,
        tensorboard_log=f"./{save_dir}_logs/",
        device=device,
    )


def inject_jepa_feature_extractor(env, model, enable_jepa):
    """Safely inject feature extractor into JEPA wrapper"""
    if not enable_jepa:
        return

    try:
        print("   💉 Injecting PPO feature extractor into JEPA wrapper...")
        # Navigate through the DummyVecEnv -> Monitor -> SamuraiJEPAWrapper hierarchy
        monitor_env = env.envs[0]
        wrapper_env = monitor_env.env  # This gets us to the SamuraiJEPAWrapper

        if hasattr(wrapper_env, "inject_feature_extractor"):
            wrapper_env.inject_feature_extractor(model.policy.features_extractor)
            print("   ✅ JEPA system fully initialized!")
        else:
            print(f"   ⚠️ Warning: Could not find inject_feature_extractor method")
            print(f"   Environment type: {type(wrapper_env)}")
            available_methods = [
                attr
                for attr in dir(wrapper_env)
                if not attr.startswith("_") and callable(getattr(wrapper_env, attr))
            ]
            print(f"   Available methods: {available_methods[:10]}...")  # Show first 10

    except Exception as e:
        print(f"   ❌ JEPA injection failed: {e}")
        print("   Training will continue without JEPA features")


def main():
    """Main training loop with comprehensive error handling"""
    try:
        args = parse_arguments()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        mode_name = "JEPA Enhanced" if args.enable_jepa else "Standard CNN"

        print(f"🚀 Starting {mode_name} Training (COMPREHENSIVE FIXES)")
        print(f"   💻 Device: {device.upper()}")
        print(f"   🎯 Timesteps: {args.total_timesteps:,}")
        print(f"   📚 Frame Stack: {args.frame_stack}")
        print(f"   🧠 JEPA Enabled: {args.enable_jepa}")

        # Setup game and state
        game, state_filename = "SamuraiShodown-Genesis", "samurai.state"
        state_path = (
            os.path.abspath(state_filename)
            if os.path.exists(state_filename)
            else retro.State.DEFAULT
        )
        print(f"   🎮 Using state: {state_path}")

        # Create environment
        render_mode = "human" if args.render else None
        env = DummyVecEnv(
            [
                make_env(
                    game, state_path, render_mode, args.frame_stack, args.enable_jepa
                )
            ]
        )

        # Setup directories
        save_dir = "trained_models_jepa_reward_shaped"
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(f"{save_dir}_logs", exist_ok=True)

        # Setup model
        model = setup_model(env, args, device, save_dir)

        # Inject JEPA feature extractor
        inject_jepa_feature_extractor(env, model, args.enable_jepa)

        # Setup callbacks
        checkpoint_callback = CheckpointCallback(
            save_freq=max(50000, args.n_steps),
            save_path=save_dir,
            name_prefix="ppo_jepa_shaped",
        )
        training_callback = JEPATrainingCallback(enable_jepa=args.enable_jepa)

        # Training loop
        print("\n🎯 Starting training...")
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=[checkpoint_callback, training_callback],
            reset_num_timesteps=not bool(args.resume),
        )

    except KeyboardInterrupt:
        print("\n⏸️ Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        raise
    finally:
        # Cleanup
        try:
            if "env" in locals():
                env.close()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            print(f"⚠️ Cleanup warning: {e}")

    # Save final model
    try:
        if "model" in locals():
            final_path = os.path.join(save_dir, "final_model.zip")
            model.save(final_path)
            print(f"💾 Final model saved to {final_path}")
    except Exception as e:
        print(f"⚠️ Failed to save final model: {e}")


if __name__ == "__main__":
    main()
