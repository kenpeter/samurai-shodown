import os
import sys
import argparse
import time
import torch
import psutil
import numpy as np

import retro
import gymnasium as gym
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv

# Import the wrapper (now includes Decision Transformer)
from wrapper import (
    SamuraiShowdownCustomWrapper,
    DecisionTransformer,
    collect_trajectories,
    train_decision_transformer,
)


def make_env_for_subprocess(game, state, rendering=False, seed=0, env_id=0):
    """Create environment function for SubprocVecEnv - all imports MUST be inside"""

    def _init():
        # Import everything inside the function for subprocess compatibility
        import retro
        import gymnasium as gym
        from stable_baselines3.common.monitor import Monitor
        from wrapper import SamuraiShowdownCustomWrapper

        # Create environment - FIXED FOR STABLE-RETRO
        env = retro.make(
            game=game,
            state=state,
            use_restricted_actions=retro.Actions.FILTERED,
            obs_type=retro.Observations.IMAGE,
            render_mode="human" if rendering else None,
        )

        env = SamuraiShowdownCustomWrapper(
            env,
            reset_round=True,
            rendering=rendering,
            max_episode_steps=20000,  # MASSIVE episodes for huge batches
        )

        env = Monitor(env)
        env.reset(seed=seed)
        return env

    return _init


def linear_schedule(initial_value, final_value=0.0):
    """Linear scheduler"""

    def scheduler(progress):
        return final_value + progress * (initial_value - final_value)

    return scheduler


def check_system_resources(num_envs):
    """Check if system can handle the requested number of environments - OPTIMIZED FOR GPU"""

    # Check GPU VRAM
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"🚀 GPU TRAINING System Resource Check:")
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {gpu_memory:.1f} GB")
        print(f"   Strategy: Maximum batch size Decision Transformer")
    else:
        print(f"❌ CUDA not available! Falling back to CPU")
        return False

    # Check CPU cores
    cpu_cores = psutil.cpu_count(logical=True)
    print(f"   CPU Cores: {cpu_cores}")
    print(f"   Requested Envs: {num_envs}")

    # Check RAM - OPTIMIZED FOR 82GB SYSTEM
    ram_gb = psutil.virtual_memory().total / (1024**3)
    estimated_ram_usage = num_envs * 3.0  # ~3GB per environment with massive rollouts

    print(f"   RAM: {ram_gb:.1f} GB (MASSIVE RAM SYSTEM)")
    print(f"   Estimated RAM usage: {estimated_ram_usage:.1f} GB")
    print(f"   Available for batches: {ram_gb - estimated_ram_usage - 15:.1f} GB")

    if estimated_ram_usage > ram_gb * 0.5:
        print(f"❌ ERROR: Insufficient RAM!")
        print(f"💡 Reduce environments or rollout size")
        return False

    return True


def calculate_optimal_batch_size(obs_shape, context_length=30, vram_gb=12):
    """Calculate maximum batch size that fits in VRAM"""
    print(f"🧮 Calculating optimal batch size for {vram_gb}GB VRAM...")

    # Memory estimation per sample (in bytes)
    # States: batch_size * context_length * obs_shape * 4 bytes (float32)
    # Actions: batch_size * context_length * 4 bytes (int32)
    # Returns: batch_size * context_length * 4 bytes (float32)
    # Timesteps: batch_size * context_length * 4 bytes (int32)

    obs_memory_per_sample = (
        context_length * obs_shape[0] * obs_shape[1] * obs_shape[2] * 4
    )
    other_memory_per_sample = context_length * 4 * 3  # actions, returns, timesteps
    total_memory_per_sample = obs_memory_per_sample + other_memory_per_sample

    # Model memory (rough estimate for transformer)
    model_memory = 256 * 1024 * 1024 * 4  # ~1GB for model parameters

    # Available memory (leave 2GB buffer for CUDA operations)
    available_memory = (vram_gb - 2) * 1024 * 1024 * 1024
    memory_for_batch = available_memory - model_memory

    # Calculate max batch size
    max_batch_size = int(memory_for_batch / total_memory_per_sample)

    # Round down to nearest power of 2 for efficiency
    optimal_batch_size = 1
    while optimal_batch_size * 2 <= max_batch_size:
        optimal_batch_size *= 2

    print(f"   Memory per sample: {total_memory_per_sample / (1024**2):.1f} MB")
    print(f"   Available for batches: {memory_for_batch / (1024**3):.1f} GB")
    print(f"   Theoretical max batch: {max_batch_size}")
    print(f"   Optimal batch size: {optimal_batch_size}")

    return optimal_batch_size


def get_actual_observation_dims(game, state):
    """Get actual observation dimensions from the wrapper"""
    try:
        # Create a temporary environment to get actual dimensions - FIXED
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
        # Fallback to estimated dimensions
        return (9, 168, 240)  # 9 frames, 75% of 224x320


def collect_trajectories_simple(env, num_timesteps, max_trajectories=200):
    """Collect trajectories with memory management - NO SAVING HERE"""
    print(f"🎮 Collecting {num_timesteps:,} timesteps (max {max_trajectories} trajectories)...")
    
    trajectories = []
    current_timesteps = 0
    episode_count = 0

    while current_timesteps < num_timesteps and len(trajectories) < max_trajectories:
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
            and current_timesteps < num_timesteps
        ):
            action = env.action_space.sample()

            try:
                obs, reward, done, truncated, _ = env.step(action)

                trajectory["actions"].append(action)
                trajectory["rewards"].append(reward)
                current_timesteps += 1
                step_count += 1

                if not done and not truncated:
                    trajectory["states"].append(obs.copy())

            except Exception as e:
                print(f"⚠️ Step error in episode {episode_count}: {e}")
                break

        # Save trajectory if meaningful
        if len(trajectory["actions"]) > 10:
            trajectories.append(trajectory)

        episode_count += 1

        # Memory management
        if len(trajectories) > max_trajectories:
            trajectories = trajectories[-max_trajectories//2:]  # Keep newest half
            print(f"🧹 Memory cleanup: keeping {len(trajectories)} trajectories")

        # Logging every 10,000 timesteps
        if current_timesteps > 0 and current_timesteps % 10000 == 0:
            recent_rewards = [sum(t["rewards"]) for t in trajectories[-20:]]
            avg_reward = np.mean(recent_rewards) if recent_rewards else 0
            win_episodes = sum(1 for r in recent_rewards if r > 0) if recent_rewards else 0
            win_rate = (win_episodes / len(recent_rewards) * 100) if recent_rewards else 0

            print(f"   📊 {current_timesteps:,}/{num_timesteps:,} | Episodes: {episode_count:,} | Trajectories: {len(trajectories)} | Win Rate: {win_rate:.1f}%")

    print(f"✅ Collected {len(trajectories):,} trajectories over {current_timesteps:,} timesteps")
    return trajectories


def train_decision_transformer_with_checkpoints(
    model,
    trajectories,
    epochs=200,
    batch_size=64,
    lr=1e-4,
    device="cuda",
    context_length=50,
    save_dir=None,
    checkpoint_interval=20,  # Save every N epochs
):
    """FIXED: Train Decision Transformer with proper checkpoint saving during training"""
    
    # Import the training function from wrapper
    from wrapper import TrajectoryDataset
    from torch.utils.data import DataLoader
    from torch.optim import AdamW
    import torch.nn.functional as F

    # Create dataset and dataloader
    dataset = TrajectoryDataset(trajectories, context_length)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=4,
        persistent_workers=True,
    )

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4, betas=(0.9, 0.95))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    model.to(device)
    model.train()

    print(f"🚀 CUDA Training Decision Transformer with PROPER CHECKPOINTS:")
    print(f"   Device: {device}")
    print(f"   Epochs: {epochs}")
    print(f"   Batch size: {batch_size}")
    print(f"   Learning rate: {lr}")
    print(f"   Context length: {context_length}")
    print(f"   Dataset size: {len(dataset)}")
    print(f"   💾 Checkpoint every {checkpoint_interval} epochs")

    # Create checkpoint directory
    if save_dir:
        checkpoint_dir = os.path.join(save_dir, "training_checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)

    scaler = torch.cuda.amp.GradScaler()
    best_loss = float("inf")

    for epoch in range(epochs):
        total_loss = 0
        num_batches = 0
        epoch_start_time = time.time()

        for batch_idx, batch in enumerate(dataloader):
            states = batch["states"].to(device, non_blocking=True)
            actions = batch["actions"].to(device, non_blocking=True)
            returns_to_go = batch["returns_to_go"].to(device, non_blocking=True)
            timesteps = batch["timesteps"].to(device, non_blocking=True)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                action_logits = model(states, actions, returns_to_go, timesteps)

                if action_logits.shape[1] > 1:
                    targets = actions[:, 1:]
                    predictions = action_logits[:, :-1]
                    loss = F.cross_entropy(
                        predictions.reshape(-1, predictions.shape[-1]),
                        targets.reshape(-1),
                    )
                else:
                    continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            num_batches += 1

        scheduler.step()
        epoch_time = time.time() - epoch_start_time
        avg_loss = total_loss / max(num_batches, 1)

        if avg_loss < best_loss:
            best_loss = avg_loss

        # FIXED: Save checkpoints during actual training
        if save_dir and (epoch + 1) % checkpoint_interval == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"epoch_{epoch+1}_loss_{avg_loss:.4f}.pth")
            try:
                torch.save(model.state_dict(), checkpoint_path)
                file_size = os.path.getsize(checkpoint_path) / (1024 * 1024)
                print(f"💾 TRAINING CHECKPOINT SAVED: epoch_{epoch+1}_loss_{avg_loss:.4f}.pth ({file_size:.1f}MB)")
            except Exception as e:
                print(f"❌ Failed to save checkpoint: {e}")

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"   Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}, Time: {epoch_time:.1f}s")

            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**3
                cached = torch.cuda.memory_reserved() / 1024**3
                print(f"   GPU Memory: {allocated:.2f}GB allocated, {cached:.2f}GB cached")

    print(f"✅ TRAINING COMPLETE! Best loss: {best_loss:.4f}")
    
    # Save best model
    if save_dir:
        best_model_path = os.path.join(save_dir, f"best_model_loss_{best_loss:.4f}.pth")
        try:
            torch.save(model.state_dict(), best_model_path)
            print(f"💾 BEST MODEL SAVED: {best_model_path}")
        except Exception as e:
            print(f"❌ Failed to save best model: {e}")
    
    return model


def main():
    parser = argparse.ArgumentParser(
        description="Train Samurai Showdown Agent - Decision Transformer (FIXED SAVE VERSION)"
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=100000,  # Reduced for testing - increase later
        help="Total timesteps to collect for training data",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=1,  # Use single environment for stability
        help="Number of parallel environments (1 for stability)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=4e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--resume", type=str, default=None, help="Resume from saved model path"
    )
    parser.add_argument("--render", action="store_true", help="Enable rendering")
    parser.add_argument(
        "--use-default-state", action="store_true", help="Use default game state"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=0,  # 0 = auto-calculate optimal
        help="Batch size (0 for auto-optimal)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,  # Training epochs
        help="Training epochs",
    )
    parser.add_argument(
        "--context-length",
        type=int,
        default=30,
        help="Context length for Decision Transformer",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=60,  # Save every 60 epochs
        help="Save checkpoint every N training epochs",
    )
    parser.add_argument(
        "--max-trajectories",
        type=int,
        default=300,
        help="Maximum trajectories to keep in memory",
    )

    args = parser.parse_args()

    # System resource check
    if not check_system_resources(args.num_envs):
        sys.exit(1)

    # CUDA SETUP - FORCE GPU USAGE
    if not torch.cuda.is_available():
        print("❌ CUDA not available! This script requires GPU.")
        sys.exit(1)

    device = "cuda"
    print(f"🚀 CUDA TRAINING MODE - Using {torch.cuda.get_device_name(0)}")

    # Optimize CUDA settings
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    # Clear GPU cache
    torch.cuda.empty_cache()

    print(f"✅ Device: {device}")
    print(f"✅ CUDA optimizations enabled")
    print(f"💾 Training checkpoints will be saved every {args.checkpoint_interval} epochs")

    game = "SamuraiShodown-Genesis"

    # Test if the game works
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
        print(f"✅ Basic environment test passed - obs shape: {test_obs.shape}")
        test_env.close()
    except Exception as e:
        print(f"❌ Basic environment test failed: {e}")
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

    # Get actual observation dimensions
    print("🔍 Determining actual observation dimensions...")
    obs_shape = get_actual_observation_dims(game, state)
    num_frames, obs_height, obs_width = obs_shape
    print(f"✅ Observation shape: {obs_shape}")

    # Calculate optimal batch size if not specified
    if args.batch_size == 0:
        optimal_batch_size = calculate_optimal_batch_size(
            obs_shape, args.context_length, 12
        )
        args.batch_size = max(8, optimal_batch_size)

    print(f"🎯 Using batch size: {args.batch_size}")

    save_dir = "trained_models"
    os.makedirs(save_dir, exist_ok=True)

    print(f"💾 Save directory: {save_dir}")

    # Create training environment - SINGLE STABLE ENVIRONMENT
    print(f"🏗️ Creating stable single training environment...")

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
        max_episode_steps=5000,  # Reasonable episode length
    )
    env = Monitor(env)

    print(f"✅ Single environment created successfully")

    # Test environment
    print("🧪 Testing environment...")
    try:
        obs, info = env.reset()
        print(f"✅ Reset successful - obs shape: {obs.shape}")

        # Test a few steps
        for i in range(5):
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)
            print(f"   Step {i+1}: action={action}, reward={reward}, done={done}")
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
        model.load_state_dict(torch.load(args.resume, map_location=device))
        print(f"✅ Model loaded successfully")

    print(f"✅ Model created with {sum(p.numel() for p in model.parameters())} parameters")

    # FIXED: Collect trajectories WITHOUT saving untrained models
    print(f"📊 Collecting {args.total_timesteps:,} timesteps...")
    trajectories = collect_trajectories_simple(
        env, args.total_timesteps, args.max_trajectories
    )

    print(f"✅ Collected {len(trajectories)} trajectories")

    # Filter trajectories with some reward signal
    good_trajectories = [
        t for t in trajectories if len(t["rewards"]) > 10 and sum(t["rewards"]) != 0
    ]
    print(f"✅ Found {len(good_trajectories)} trajectories with reward signal")

    if len(good_trajectories) < 5:
        print("⚠️ Using all trajectories due to limited good data")
        good_trajectories = trajectories

    # Only proceed if we have enough data
    if len(good_trajectories) < 2:
        print("❌ Not enough trajectories for training")
        return

    # FIXED: Train Decision Transformer with proper checkpoint saving
    print(f"🏋️ Training Decision Transformer with PROPER CHECKPOINTS...")
    print(f"   Epochs: {args.epochs}")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Learning rate: {args.learning_rate}")
    print(f"   Context length: {args.context_length}")
    print(f"   💾 Checkpoints every {args.checkpoint_interval} epochs")

    trained_model = train_decision_transformer_with_checkpoints(
        model=model,
        trajectories=good_trajectories,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.learning_rate,
        device=device,
        context_length=args.context_length,
        save_dir=save_dir,
        checkpoint_interval=args.checkpoint_interval,
    )

    # Save final model
    final_model_path = os.path.join(save_dir, "decision_transformer_samurai_FINAL.pth")
    try:
        torch.save(trained_model.state_dict(), final_model_path)
        file_size = os.path.getsize(final_model_path) / (1024 * 1024)
        print(f"💾 FINAL MODEL SAVED: {final_model_path} ({file_size:.1f}MB)")
    except Exception as e:
        print(f"❌ Failed to save final model: {e}")

    print(f"🎉 Training completed!")
    print(f"💾 Check these directories for your saved models:")
    print(f"   📁 Training checkpoints: {save_dir}/training_checkpoints/")
    print(f"   📁 Final model: {final_model_path}")

    # List saved files
    try:
        checkpoint_dir = os.path.join(save_dir, "training_checkpoints")
        if os.path.exists(checkpoint_dir):
            checkpoints = os.listdir(checkpoint_dir)
            print(f"   📋 Saved {len(checkpoints)} training checkpoints")
            for cp in sorted(checkpoints)[:5]:  # Show first 5
                cp_path = os.path.join(checkpoint_dir, cp)
                size_mb = os.path.getsize(cp_path) / (1024 * 1024)
                print(f"      📄 {cp} ({size_mb:.1f}MB)")
            if len(checkpoints) > 5:
                print(f"      ... and {len(checkpoints) - 5} more")
    except Exception as e:
        print(f"⚠️ Could not list checkpoint files: {e}")

    # Print GPU memory usage
    if torch.cuda.is_available():
        print(f"🔧 GPU Memory Usage:")
        print(f"   Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"   Cached: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")

    # Close environment
    env.close()


if __name__ == "__main__":
    main()