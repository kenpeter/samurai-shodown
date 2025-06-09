import os
import argparse
import torch
import numpy as np
import retro
import gymnasium as gym
from stable_baselines3.common.monitor import Monitor

from wrapper import (
    SamuraiShowdownCustomWrapper,
    DecisionTransformer,
    train_decision_transformer,
)


def simple_save_checkpoint(model, timesteps, save_dir):
    """Save checkpoint"""
    checkpoint_dir = os.path.join(save_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{timesteps}_timesteps.pth")
    torch.save(model.state_dict(), checkpoint_path)
    print(f"💾 SAVED: {timesteps:,} timesteps")


def collect_data(env, num_timesteps, max_trajectories=200):
    """Collect training data"""
    print(f"🎮 Collecting {num_timesteps:,} timesteps...")
    
    trajectories = []
    current_timesteps = 0
    
    while current_timesteps < num_timesteps and len(trajectories) < max_trajectories:
        trajectory = {"states": [], "actions": [], "rewards": []}
        obs, _ = env.reset()
        trajectory["states"].append(obs.copy())

        done = False
        truncated = False
        
        while not done and not truncated and current_timesteps < num_timesteps:
            action = env.action_space.sample()
            obs, reward, done, truncated, _ = env.step(action)
            
            trajectory["actions"].append(action)
            trajectory["rewards"].append(reward)
            current_timesteps += 1
            
            if not done and not truncated:
                trajectory["states"].append(obs.copy())

        if len(trajectory["actions"]) > 10:
            trajectories.append(trajectory)
            
        # Memory cleanup
        if len(trajectories) > max_trajectories:
            trajectories = trajectories[-max_trajectories//2:]

    print(f"✅ Collected {len(trajectories)} trajectories")
    return trajectories


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=100000)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    device = "cuda"
    save_dir = "trained_models"
    os.makedirs(save_dir, exist_ok=True)

    # Create environment
    env = retro.make(
        game="SamuraiShodown-Genesis",
        state=None,
        use_restricted_actions=retro.Actions.FILTERED,
        obs_type=retro.Observations.IMAGE,
        render_mode="human" if args.render else None,
    )
    env = SamuraiShowdownCustomWrapper(env, reset_round=True, rendering=args.render)
    env = Monitor(env)

    # Create model
    obs_shape = env.observation_space.shape
    action_dim = env.action_space.n
    model = DecisionTransformer(obs_shape, action_dim, hidden_size=256, n_layer=4, n_head=4)

    # Collect data
    trajectories = collect_data(env, args.timesteps)
    
    # Filter good trajectories
    good_trajectories = [t for t in trajectories if len(t["rewards"]) > 10]
    if len(good_trajectories) < 5:
        good_trajectories = trajectories

    # Train using the imported function
    print(f"🏋️ Training on {len(good_trajectories)} trajectories...")
    trained_model = train_decision_transformer(
        model=model,
        trajectories=good_trajectories,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=device,
        context_length=30,
    )

    # Save final model
    final_path = os.path.join(save_dir, "final_model.pth")
    torch.save(trained_model.state_dict(), final_path)
    print(f"💾 FINAL: {final_path}")

    env.close()


if __name__ == "__main__":
    main()