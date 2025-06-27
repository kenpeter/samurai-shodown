#!/usr/bin/env python3
"""
JEPA-Enhanced Samurai Training Script - VERIFIED & ROBUST
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
    # (No changes needed here, it's already correct for VecEnv)
    def __init__(self, enable_jepa=True, verbose=0):
        super(JEPATrainingCallback, self).__init__(verbose)
        self.enable_jepa = enable_jepa
        self.last_log_time = time.time()

    def _on_step(self) -> bool:
        if time.time() - self.last_log_time > 60:
            self.last_log_time = time.time()
            all_stats = self.training_env.get_attr("current_stats")
            if all_stats and len(all_stats) > 0:
                stats = all_stats[0]
                wins, losses, total_rounds = (
                    stats.get("wins", 0),
                    stats.get("losses", 0),
                    stats.get("total_rounds", 0),
                )
                win_rate = (wins / total_rounds * 100) if total_rounds > 0 else 0
                print(
                    f"\n--- 📊 JEPA Training Status @ Step {self.num_timesteps:,} ---"
                )
                print(
                    f"   🎯 Win Rate: {win_rate:.1f}% ({wins}W / {losses}L in {total_rounds} rounds)"
                )
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


# *** FIX: Use a make_env helper for clean initialization ***
def make_env(game, state_path, render_mode, frame_stack, enable_jepa):
    """
    Utility function to create and wrap the environment.
    """

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
    print(f"🚀 Starting {mode_name} Training (Verified, Robustness Update)")
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

    # *** FIX: Create the VecEnv using the helper function ***
    render_mode = "human" if args.render else None
    env = DummyVecEnv(
        [make_env(game, state_path, render_mode, args.frame_stack, args.enable_jepa)]
    )

    save_dir = "trained_models_jepa_robust"
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

    if args.enable_jepa:
        # *** FIX: Use a clean and reliable way to inject the extractor ***
        # This accesses the actual env object inside the DummyVecEnv
        jepa_wrapper = env.envs[0].unwrapped
        jepa_wrapper.feature_extractor = model.policy.features_extractor
        print("   💉 Injected PPO feature extractor into JEPA wrapper.")

    checkpoint_callback = CheckpointCallback(
        save_freq=max(50000, args.n_steps),
        save_path=save_dir,
        name_prefix="ppo_jepa_robust",
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
