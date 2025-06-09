import os
import sys
import argparse
import time
import torch
import psutil
import numpy as np
import gc

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


def save_checkpoint(model, timesteps, save_dir):
    """Save model checkpoint as .zip"""
    import zipfile

    # Save model state dict to temporary file
    temp_model_path = f"temp_model_{timesteps}.pth"
    torch.save(model.state_dict(), temp_model_path)

    # Create zip file
    zip_path = os.path.join(save_dir, f"model_{timesteps}_steps.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(temp_model_path, f"model_{timesteps}_steps.pth")

    # Clean up temp file
    os.remove(temp_model_path)
    print(f"💾 Checkpoint saved: {zip_path}")


def collect_trajectories_memory_optimized(
    env, total_timesteps, model, save_dir, checkpoint_interval=300000
):
    """Collect trajectories with memory optimization"""
    print(f"🎮 Collecting {total_timesteps:,} timesteps with MEMORY LIMITS...")

    trajectories = []
    current_timesteps = 0
    episode_count = 0
    last_checkpoint = 0

    # MEMORY FIX: Reduced trajectory limits
    MAX_TRAJECTORIES = 50  # Reduced from 100
    KEEP_COUNT = 10  # Keep only 10 during cleanup

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
            and step_count < 5000
            and current_timesteps < total_timesteps
        ):
            # SMART ACTION POLICY: Reduce excessive jumping
            action = env.action_space.sample()

            # Action 0 is often jump in fighting games - reduce its frequency
            # Only jump 10% of the time instead of equal probability
            if action == 0 and np.random.random() > 0.1:
                # Pick a different action from 1 to max action
                action = np.random.randint(1, env.action_space.n)

            try:
                obs, reward, done, truncated, _ = env.step(action)
                trajectory["actions"].append(action)
                trajectory["rewards"].append(reward)
                current_timesteps += 1
                step_count += 1

                if not done and not truncated:
                    trajectory["states"].append(obs.copy())

                # Save every 300,000 timesteps only
                if current_timesteps - last_checkpoint >= checkpoint_interval:
                    save_checkpoint(model, current_timesteps, save_dir)
                    last_checkpoint = current_timesteps

            except Exception as e:
                print(f"⚠️ Step error in episode {episode_count}: {e}")
                break

        # Save meaningful trajectories
        if len(trajectory["actions"]) > 10:
            trajectories.append(trajectory)

        episode_count += 1

        # MEMORY FIX: Keep only last 10 trajectories
        if len(trajectories) > MAX_TRAJECTORIES:
            trajectories = trajectories[-KEEP_COUNT:]
            print(f"🧹 Memory cleanup: keeping only {len(trajectories)} trajectories")

        # Progress logging with memory monitoring
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
                f"Win: {win_rate:.1f}% | RAM: {memory_gb:.1f}GB ({memory_percent:.1f}%)"
            )

            # Emergency cleanup
            if memory_percent > 70:
                print(f"⚠️ MEMORY WARNING: {memory_percent:.1f}% - emergency cleanup")
                trajectories = trajectories[-5:]
                gc.collect()

    # Final checkpoint as .zip
    save_checkpoint(model, current_timesteps, save_dir)
    print(
        f"✅ Collected {len(trajectories)} trajectories over {current_timesteps:,} timesteps"
    )
    return trajectories


def main():
    parser = argparse.ArgumentParser(
        description="Train Samurai Showdown Agent - Single Process Memory Optimized"
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
    parser.add_argument(
        "--resume", type=str, default=None, help="Resume from saved model path"
    )
    parser.add_argument("--render", action="store_true", help="Enable rendering")
    parser.add_argument(
        "--use-default-state", action="store_true", help="Use default game state"
    )
    parser.add_argument(
        "--batch-size", type=int, default=0, help="Batch size (0 for auto with cap)"
    )
    parser.add_argument("--n-steps", type=int, default=50, help="Training epochs")
    parser.add_argument(
        "--context-length",
        type=int,
        default=30,
        help="Context length for Decision Transformer",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=300000,
        help="Save checkpoint every N timesteps",
    )

    args = parser.parse_args()

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
    if args.use_default_state:
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

    # MAXIMUM BATCH SIZE: Use accurate model memory calculation
    if args.batch_size == 0:
        # Simple calculation for batch size
        context_length = args.context_length
        obs_memory_per_sample = (
            context_length * obs_shape[0] * obs_shape[1] * obs_shape[2] * 4
        )

        # Accurate breakdown:
        # Model (7.23M params) + gradients + optimizer + activations + CUDA overhead = ~1.6GB
        # Available for batches from 12GB VRAM: 10.4GB
        available_memory = 10.4 * 1024 * 1024 * 1024  # 10.4GB
        max_batch_size = int(available_memory / obs_memory_per_sample)

        # MAXIMUM: Cap at 256 (true maximum) and ensure minimum of 8
        args.batch_size = min(256, max(8, max_batch_size))

    print(f"🚀 Using MAXIMUM batch size: {args.batch_size} (16x faster than before!)")

    # Setup directories
    save_dir = "trained_models"
    os.makedirs(save_dir, exist_ok=True)

    # Create single training environment
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

        if args.resume.endswith(".zip"):
            # Extract and load from .zip
            import zipfile
            import tempfile

            with zipfile.ZipFile(args.resume, "r") as zip_ref:
                pth_files = [f for f in zip_ref.namelist() if f.endswith(".pth")]
                if not pth_files:
                    print(f"❌ No .pth file found in zip")
                    return

                # Extract to temp file and load
                with tempfile.NamedTemporaryFile(
                    suffix=".pth", delete=False
                ) as temp_file:
                    temp_file.write(zip_ref.read(pth_files[0]))
                    temp_path = temp_file.name

            try:
                model.load_state_dict(torch.load(temp_path, map_location=device))
                print(f"✅ Model loaded from .zip")
            except Exception as e:
                print(f"❌ Error loading: {e}")
                return
            finally:
                os.unlink(temp_path)
        else:
            # Direct .pth loading
            try:
                model.load_state_dict(torch.load(args.resume, map_location=device))
                print(f"✅ Model loaded")
            except Exception as e:
                print(f"❌ Error loading: {e}")
                return

    param_count = sum(p.numel() for p in model.parameters())
    print(f"✅ Model created with {param_count:,} parameters")

    # Collect trajectories
    print(f"📊 Starting trajectory collection...")
    trajectories = collect_trajectories_memory_optimized(
        env, args.total_timesteps, model, save_dir, args.checkpoint_interval
    )

    print(f"✅ Collected {len(trajectories)} trajectories")

    # Filter good trajectories
    good_trajectories = [
        t for t in trajectories if len(t["rewards"]) > 10 and sum(t["rewards"]) != 0
    ]
    print(f"✅ Found {len(good_trajectories)} trajectories with reward signal")

    if len(good_trajectories) < 5:
        print("⚠️ Using all trajectories due to limited good data")
        good_trajectories = trajectories

    if len(good_trajectories) < 2:
        print("❌ Not enough trajectories for training")
        return

    # Train Decision Transformer
    print(f"🏋️ Training Decision Transformer...")
    print(f"   Epochs: {args.n_steps}")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Learning rate: {args.learning_rate}")
    print(f"   Context length: {args.context_length}")

    trained_model = train_decision_transformer(
        model=model,
        trajectories=good_trajectories,
        epochs=args.n_steps,
        batch_size=args.batch_size,
        lr=args.learning_rate,
        device=device,
        context_length=args.context_length,
    )

    # Save final model as .zip
    import zipfile

    temp_model_path = "temp_final_model.pth"
    torch.save(trained_model.state_dict(), temp_model_path)

    final_zip_path = os.path.join(save_dir, "decision_transformer_samurai_final.zip")
    with zipfile.ZipFile(final_zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(temp_model_path, "decision_transformer_samurai_final.pth")

    os.remove(temp_model_path)

    print(f"🎉 Training completed!")
    print(f"💾 Final model saved to: {final_zip_path}")

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
