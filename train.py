import os
import sys
import argparse
import time
import torch
import psutil
import numpy as np
import gc
import collections

import retro
import gymnasium as gym
from stable_baselines3.common.monitor import Monitor

# Import the wrapper (now includes Decision Transformer)
from wrapper import (
    SamuraiShowdownCustomWrapper,
    DecisionTransformer,
    train_decision_transformer,
)


def check_system_resources():
    """Check system resources"""
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"🚀 MEMORY OPTIMIZED System Check:")
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {gpu_memory:.1f} GB")
    else:
        print(f"❌ CUDA not available! This script requires GPU.")
        return False

    ram_gb = psutil.virtual_memory().total / (1024**3)
    print(f"   RAM: {ram_gb:.1f} GB")
    return True


def get_actual_observation_dims(game, state):
    """Get actual observation dimensions from the wrapper"""
    try:
        temp_env = retro.make(
            game=game,
            state=state,
            use_restricted_actions=retro.Actions.FILTERED,
            obs_type=retro.Observations.IMAGE,
            render_mode=None,
        )
        wrapped_env = SamuraiShowdownCustomWrapper(temp_env)
        obs_shape = wrapped_env.observation_space.shape
        temp_env.close()
        return obs_shape
    except Exception as e:
        print(f"⚠️ Could not determine observation dimensions: {e}")
        return (9, 168, 240)  # Fallback


def collect_trajectories(
    env,
    total_timesteps,
    device="cuda",
    context_length=30,
):
    """Collect trajectories using random actions"""
    print(f"🎮 Collecting {total_timesteps:,} timesteps with RANDOM actions...")

    trajectories = []
    current_timesteps = 0
    episode_count = 0

    # MEMORY FIX: Reduced trajectory limits
    MAX_TRAJECTORIES = 50
    KEEP_COUNT = 10
    MAX_EPISODE_STEPS = 5000

    while current_timesteps < total_timesteps:
        trajectory = {"states": [], "actions": [], "rewards": []}
        obs, _ = env.reset()
        trajectory["states"].append(obs.copy())

        done = False
        truncated = False
        step_count = 0

        while (
            not done
            and not truncated
            and step_count < MAX_EPISODE_STEPS
            and current_timesteps < total_timesteps
        ):
            # Simple random action selection - wrapper will handle aggression
            action = env.action_space.sample()

            try:
                obs, reward, done, truncated, _ = env.step(action)

                # Store in trajectory
                trajectory["actions"].append(action)
                trajectory["rewards"].append(reward)
                current_timesteps += 1
                step_count += 1

                if not done and not truncated:
                    trajectory["states"].append(obs.copy())

                # FORCE episode termination if too long
                if step_count >= MAX_EPISODE_STEPS:
                    print(
                        f"⏰ Episode {episode_count} forced to end after {step_count} steps"
                    )
                    truncated = True
                    break

            except Exception as e:
                print(f"⚠️ Step error in episode {episode_count}: {e}")
                break

        # Save meaningful trajectories
        if len(trajectory["actions"]) > 10:
            trajectories.append(trajectory)

        episode_count += 1

        # Memory management - MUCH more aggressive
        if len(trajectories) > MAX_TRAJECTORIES:
            trajectories = trajectories[-KEEP_COUNT:]
            print(f"🧹 Memory cleanup: keeping only {len(trajectories)} trajectories")

        # EMERGENCY: Force cleanup every 20 episodes regardless
        if episode_count % 20 == 0 and len(trajectories) > 5:
            trajectories = trajectories[-3:]
            gc.collect()  # Force garbage collection
            print(
                f"🚨 EMERGENCY cleanup: keeping only {len(trajectories)} trajectories"
            )

        # Progress logging
        if current_timesteps > 0 and current_timesteps % 5000 == 0:
            recent_rewards = [sum(t["rewards"]) for t in trajectories[-20:]]
            avg_reward = np.mean(recent_rewards) if recent_rewards else 0
            win_episodes = (
                sum(1 for r in recent_rewards if r > 0) if recent_rewards else 0
            )
            win_rate = (
                (win_episodes / len(recent_rewards) * 100) if recent_rewards else 0
            )

            memory_usage = psutil.virtual_memory()
            memory_percent = memory_usage.percent
            memory_gb = memory_usage.used / (1024**3)

            print(
                f"   📊 {current_timesteps:,}/{total_timesteps:,} | "
                f"Eps: {episode_count:,} | Traj: {len(trajectories)} | "
                f"Win: {win_rate:.1f}% | "
                f"RAM: {memory_gb:.1f}GB ({memory_percent:.1f}%)"
            )

            # Emergency cleanup
            if memory_percent > 70:
                print(f"⚠️ MEMORY WARNING: {memory_percent:.1f}% - emergency cleanup")
                trajectories = trajectories[-5:]
                gc.collect()

    print(
        f"✅ Collected {len(trajectories)} trajectories over {current_timesteps:,} timesteps"
    )

    return trajectories


def main():
    parser = argparse.ArgumentParser(
        description="Train Samurai Showdown Agent - Single Run Collection and Training"
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=1000000,
        help="Total timesteps to collect",
    )
    parser.add_argument(
        "--learning-rate", type=float, default=4e-4, help="Learning rate"
    )
    parser.add_argument("--render", action="store_true", help="Enable rendering")
    parser.add_argument(
        "--resume", type=str, default=None, help="Resume from saved model path"
    )

    args = parser.parse_args()

    # Set good default values directly
    use_default_state = False  # Use samurai.state file if available
    batch_size = 32  # Good balance of memory usage and training stability
    n_steps = 3  # Train for 3 epochs over entire dataset
    context_length = 30  # Good context window for Decision Transformer

    # System check
    if not check_system_resources():
        sys.exit(1)

    # CUDA setup
    if not torch.cuda.is_available():
        print("❌ CUDA not available! This script requires GPU.")
        sys.exit(1)

    device = "cuda"
    print(f"🚀 Using {torch.cuda.get_device_name(0)}")

    # CUDA optimizations
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.cuda.empty_cache()

    game = "SamuraiShodown-Genesis"

    # Test environment
    print(f"🎮 Testing {game}...")
    try:
        test_env = retro.make(
            game=game,
            state=None,
            use_restricted_actions=retro.Actions.FILTERED,
            obs_type=retro.Observations.IMAGE,
            render_mode=None,
        )
        test_obs, _ = test_env.reset()
        print(f"✅ Environment test passed - obs shape: {test_obs.shape}")
        test_env.close()
    except Exception as e:
        print(f"❌ Environment test failed: {e}")
        return

    # Handle state
    if use_default_state:
        state = None
        print(f"🎮 Using default game state")
    else:
        if os.path.exists("samurai.state"):
            state = os.path.abspath("samurai.state")
            print(f"🎮 Using samurai.state file")
        else:
            print(f"❌ samurai.state not found, using default state")
            state = None

    # Get observation dimensions
    print("🔍 Determining observation dimensions...")
    obs_shape = get_actual_observation_dims(game, state)
    print(f"✅ Observation shape: {obs_shape}")

    # Use the set batch size
    print(f"🚀 Using batch size: {batch_size}")

    # Setup directories
    save_dir = "trained_models"
    os.makedirs(save_dir, exist_ok=True)

    # Create training environment
    print(f"🏗️ Creating training environment...")
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
    print(f"✅ Environment created successfully")

    # Test environment
    print("🧪 Testing environment...")
    try:
        obs, info = env.reset()
        print(f"✅ Reset successful - obs shape: {obs.shape}")

        # Quick test
        for i in range(3):
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)
            if done or truncated:
                obs, info = env.reset()
                break
    except Exception as e:
        print(f"❌ Environment test failed: {e}")
        return

    # Create Decision Transformer model
    print("🧠 Creating Decision Transformer model...")
    action_dim = env.action_space.n

    model = DecisionTransformer(
        observation_shape=obs_shape,
        action_dim=action_dim,
        hidden_size=256,
        n_layer=4,
        n_head=4,
        max_ep_len=2000,
    )

    # Load existing model if resuming
    if args.resume and os.path.exists(args.resume):
        print(f"📂 Loading model from: {args.resume}")

        if args.resume.endswith(".pkl"):
            import pickle

            try:
                with open(args.resume, "rb") as f:
                    checkpoint_data = pickle.load(f)

                model.load_state_dict(checkpoint_data["model_state_dict"])
                model._is_trained = True  # Mark as trained
                print(f"✅ Model loaded from .pkl")
            except Exception as e:
                print(f"❌ Error loading: {e}")
                return
        else:
            try:
                model.load_state_dict(torch.load(args.resume, map_location=device))
                model._is_trained = True  # Mark as trained
                print(f"✅ Model loaded")
            except Exception as e:
                print(f"❌ Error loading: {e}")
                return

    param_count = sum(p.numel() for p in model.parameters())
    print(f"✅ Model created with {param_count:,} parameters")

    # Move model to GPU
    model.to(device)

    # If resuming, use the model for inference instead of training
    if args.resume and os.path.exists(args.resume):
        temperature = (
            0.7  # Good value for inference: slightly focused but still exploratory
        )
        print(f"🎮 Running inference with loaded model (temperature={temperature})...")

        # Quick inference test
        obs, _ = env.reset()
        for i in range(100):  # Run a few steps to test
            # Prepare input for model
            states = torch.from_numpy(obs).float().unsqueeze(0).unsqueeze(0) / 255.0
            actions = torch.zeros(1, 1, dtype=torch.long)
            returns_to_go = torch.zeros(1, 1, dtype=torch.float32)
            timesteps = torch.zeros(1, 1, dtype=torch.long)

            # Move to device
            states = states.to(device)
            actions = actions.to(device)
            returns_to_go = returns_to_go.to(device)
            timesteps = timesteps.to(device)

            # Get action with temperature
            action = model.get_action(
                states, actions, returns_to_go, timesteps, temperature=temperature
            )
            obs, reward, done, truncated, _ = env.step(action)

            if done or truncated:
                obs, _ = env.reset()

        print(f"✅ Inference test completed!")
        env.close()
        return

    # Collect trajectories using random actions
    print(f"📊 Starting trajectory collection...")
    trajectories = collect_trajectories(
        env, args.total_timesteps, device=device, context_length=context_length
    )

    print(f"✅ Collected {len(trajectories)} total trajectories")

    # Filter good trajectories for training
    good_trajectories = [
        t for t in trajectories if len(t["rewards"]) > 10 and sum(t["rewards"]) != 0
    ]
    print(f"✅ Found {len(good_trajectories)} trajectories with reward signal")

    if len(good_trajectories) < 5:
        print("⚠️ Using all trajectories due to limited good data")
        good_trajectories = trajectories

    if len(good_trajectories) >= 2:
        print(f"🏋️ Training Decision Transformer...")
        trained_model = train_decision_transformer(
            model=model,
            trajectories=good_trajectories,
            epochs=n_steps,
            batch_size=batch_size,
            lr=args.learning_rate,
            device=device,
            context_length=context_length,
        )

        # Save final model
        import pickle

        final_checkpoint_data = {
            "model_state_dict": trained_model.state_dict(),
            "timesteps": args.total_timesteps,
            "model_config": {
                "observation_shape": trained_model.observation_shape,
                "action_dim": trained_model.action_dim,
                "hidden_size": trained_model.hidden_size,
                "max_ep_len": trained_model.max_ep_len,
            },
        }

        final_pkl_path = os.path.join(
            save_dir, "decision_transformer_samurai_final.pkl"
        )
        with open(final_pkl_path, "wb") as f:
            pickle.dump(final_checkpoint_data, f)

        print(f"🎉 Training completed!")
        print(f"💾 Final model saved to: {final_pkl_path}")
    else:
        print("❌ Not enough good trajectories for training")

    # Memory usage summary
    if torch.cuda.is_available():
        print(f"🔧 Final GPU Memory:")
        print(f"   Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"   Cached: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")

    final_memory = psutil.virtual_memory()
    print(
        f"🔧 Final RAM Usage: {final_memory.used / (1024**3):.1f} GB ({final_memory.percent:.1f}%)"
    )

    env.close()


if __name__ == "__main__":
    main()
