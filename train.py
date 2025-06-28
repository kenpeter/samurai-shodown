#!/usr/bin/env python3
"""
JEPA-Enhanced Samurai Training Script - FIXED PREDICTION TIMING
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

    def _on_step(self) -> bool:
        if time.time() - self.last_log_time > 60:
            self.last_log_time = time.time()

            # --- Main Stats Logging ---
            all_current_stats = self.training_env.get_attr("current_stats")
            if all_current_stats and len(all_current_stats) > 0:
                stats = all_current_stats[0]
                wins, losses, total_rounds = (
                    stats.get("wins", 0),
                    stats.get("losses", 0),
                    stats.get("total_rounds", 0),
                )
                win_rate = (wins / (wins + losses) * 100) if (wins + losses) > 0 else 0
                print(f"\n--- 📊 Training Status @ Step {self.num_timesteps:,} ---")
                print(f"   🎯 Win Rate: {win_rate:.1f}% ({wins}W / {losses}L)")

            # --- JEPA Accuracy Logging (FIXED) ---
            if self.enable_jepa:
                all_strategic_stats = self.training_env.get_attr("strategic_stats")
                all_jepa_ready = self.training_env.get_attr("jepa_ready")

                if all_strategic_stats and len(all_strategic_stats) > 0:
                    strat_stats = all_strategic_stats[0]
                    jepa_ready = all_jepa_ready[0] if all_jepa_ready else False

                    if jepa_ready:
                        accuracy = strat_stats.get("overall_accuracy", 0.0)
                        predictions_made = strat_stats.get("binary_predictions_made", 0)
                        print(
                            f"   🔮 JEPA Accuracy: {accuracy*100:.1f}% (after {predictions_made:,} predictions)"
                        )
                    else:
                        print("   🔮 JEPA: Initializing predictor...")

            # --- System and Training Param Logging ---
            print(
                f"   🧠 Arch: {'JEPA+Transformer' if self.enable_jepa else 'Standard CNN'}"
            )
            if torch.cuda.is_available():
                vram_alloc, vram_res = torch.cuda.memory_allocated() / (
                    1024**3
                ), torch.cuda.memory_reserved() / (1024**3)
                print(
                    f"   💾 VRAM: {vram_alloc:.2f} GB alloc / {vram_res:.2f} GB reserved"
                )
            print(f"   📈 LR: {self.model.learning_rate:.2e}")
            print("--------------------------------------------------")

        return True


def make_env(game, state_path, render_mode, frame_stack, enable_jepa):
    def _init():
        env = retro.make(
            game=game,
            state=state_path,
            use_restricted_actions=retro.Actions.FILTERED,
            obs_type=retro.Observations.IMAGE,
            render_mode=render_mode,
        )
        env = SamuraiJEPAWrapper(env, frame_stack=frame_stack, enable_jepa=enable_jepa)
        env = Monitor(env)
        return env

    return _init


def main():
    parser = argparse.ArgumentParser(
        description="JEPA Enhanced Fighting Game AI Training"
    )
    parser.add_argument("--total-timesteps", type=int, default=20_000_000)
    parser.add_argument("--n-steps", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=2.5e-4)
    parser.add_argument("--ent-coef", type=float, default=0.01)
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

    args.enable_jepa = not args.no_jepa
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mode_name = "JEPA Enhanced" if args.enable_jepa else "Standard CNN"
    print(f"🚀 Starting {mode_name} Training (FIXED PREDICTION TIMING)")
    print(
        f"   💻 Device: {device.upper()}, Timesteps: {args.total_timesteps:,}, Frame Stack: {args.frame_stack}"
    )

    game, state_filename = "SamuraiShodown-Genesis", "samurai.state"
    state_path = (
        os.path.abspath(state_filename)
        if os.path.exists(state_filename)
        else retro.State.DEFAULT
    )
    print(f"   🎮 Using state: {state_path}")

    render_mode = "human" if args.render else None
    env = DummyVecEnv(
        [make_env(game, state_path, render_mode, args.frame_stack, args.enable_jepa)]
    )

    save_dir = "trained_models_jepa_reward_shaped"
    os.makedirs(save_dir, exist_ok=True)

    policy_kwargs = dict(
        features_extractor_class=JEPAEnhancedCNN,
        features_extractor_kwargs=dict(features_dim=512),
        net_arch=dict(pi=[256], vf=[256]),
        activation_fn=nn.ReLU,
        optimizer_class=torch.optim.AdamW,
        optimizer_kwargs=dict(eps=1e-5, weight_decay=1e-4),
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

    # FIXED: Inject feature extractor IMMEDIATELY after model creation
    if args.enable_jepa:
        print("   💉 Injecting PPO feature extractor into JEPA wrapper...")
        # Navigate through the DummyVecEnv -> Monitor -> SamuraiJEPAWrapper hierarchy
        monitor_env = env.envs[0]
        wrapper_env = monitor_env.env  # This gets us to the SamuraiJEPAWrapper

        if hasattr(wrapper_env, "inject_feature_extractor"):
            wrapper_env.inject_feature_extractor(model.policy.features_extractor)
            print("   ✅ JEPA system fully initialized!")
        else:
            print(
                f"   ⚠️ Warning: Could not find inject_feature_extractor method on {type(wrapper_env)}"
            )
            print(
                f"   Available attributes: {[attr for attr in dir(wrapper_env) if not attr.startswith('_')]}"
            )

    checkpoint_callback = CheckpointCallback(
        save_freq=max(50000, args.n_steps),
        save_path=save_dir,
        name_prefix="ppo_jepa_shaped",
    )
    training_callback = JEPATrainingCallback(enable_jepa=args.enable_jepa)

    try:
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=[checkpoint_callback, training_callback],
            reset_num_timesteps=not bool(args.resume),
        )
    finally:
        env.close()

    final_path = os.path.join(save_dir, "final_model.zip")
    model.save(final_path)
    print(f"💾 Final model saved to {final_path}")


if __name__ == "__main__":
    main()
