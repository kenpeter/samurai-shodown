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


def model_based_action_selection(
    model,
    context_states,
    context_actions,
    context_rewards,
    timestep,
    device="cuda",
    temperature=1.0,
):
    """Use the trained model to select actions"""
    try:
        model.eval()
        with torch.no_grad():
            # Convert to tensors
            states_tensor = (
                torch.from_numpy(np.array(context_states)).float().unsqueeze(0) / 255.0
            )  # Normalize
            actions_tensor = (
                torch.from_numpy(np.array(context_actions)).long().unsqueeze(0)
            )

            # Calculate returns-to-go
            returns = np.array(context_rewards)
            returns_to_go = np.zeros_like(returns, dtype=np.float32)
            running_return = 0
            gamma = 0.99
            for i in reversed(range(len(returns))):
                running_return = returns[i] + gamma * running_return
                returns_to_go[i] = running_return

            rtg_tensor = torch.from_numpy(returns_to_go).float().unsqueeze(0)
            timesteps_tensor = (
                torch.from_numpy(np.arange(len(context_states))).long().unsqueeze(0)
            )

            # Move to device
            states_tensor = states_tensor.to(device)
            actions_tensor = actions_tensor.to(device)
            rtg_tensor = rtg_tensor.to(device)
            timesteps_tensor = timesteps_tensor.to(device)

            # Get action from model
            logits = model(states_tensor, actions_tensor, rtg_tensor, timesteps_tensor)
            last_logits = logits[0, -1] / temperature

            # Sample action
            probs = torch.softmax(last_logits, dim=-1)
            action = torch.multinomial(probs, 1).item()

            return action

    except Exception as e:
        print(f"⚠️ Model action selection failed: {e}")
        return None


def collect_trajectories_with_model(
    env,
    total_timesteps,
    model,
    save_dir,
    checkpoint_interval=300000,
    device="cuda",
    use_model_probability=0.8,
    context_length=30,
):
    """Collect trajectories using trained model for action selection"""
    print(f"🎮 Collecting {total_timesteps:,} timesteps with MODEL-BASED actions...")
    print(
        f"   Model usage: {use_model_probability*100:.0f}% model, {(1-use_model_probability)*100:.0f}% random"
    )

    trajectories = []
    current_timesteps = 0
    episode_count = 0
    last_checkpoint = 0
    model_actions = 0
    random_actions = 0

    # MEMORY FIX: Reduced trajectory limits
    MAX_TRAJECTORIES = 50
    KEEP_COUNT = 10

    # Check if model is trainable (has been trained at least once)
    model_is_trained = hasattr(model, "_is_trained") and model._is_trained

    while current_timesteps < total_timesteps:
        trajectory = {"states": [], "actions": [], "rewards": []}
        obs, _ = env.reset()
        trajectory["states"].append(obs.copy())

        done = False
        truncated = False
        step_count = 0

        # Context for model-based action selection
        context_states = collections.deque(maxlen=context_length)
        context_actions = collections.deque(maxlen=context_length)
        context_rewards = collections.deque(maxlen=context_length)

        while (
            not done
            and not truncated
            and step_count < 5000
            and current_timesteps < total_timesteps
        ):
            action = None

            # Decide whether to use model or random action
            use_model = (
                model_is_trained
                and len(context_states) >= min(10, context_length)  # Need some context
                and np.random.random() < use_model_probability
            )

            if use_model:
                # Use model for action selection
                action = model_based_action_selection(
                    model,
                    list(context_states),
                    list(context_actions),
                    list(context_rewards),
                    step_count,
                    device=device,
                    temperature=1.2,  # Slightly random for exploration
                )

                if action is not None:
                    model_actions += 1
                else:
                    # Fallback to random if model fails
                    action = env.action_space.sample()
                    random_actions += 1
            else:
                # Random action (for exploration or when model not ready)
                action = env.action_space.sample()
                random_actions += 1

            # Reduce excessive jumping (action 0 is often jump)
            if action == 0 and np.random.random() > 0.15:
                action = np.random.randint(1, env.action_space.n)

            try:
                obs, reward, done, truncated, _ = env.step(action)

                # Store in trajectory
                trajectory["actions"].append(action)
                trajectory["rewards"].append(reward)
                current_timesteps += 1
                step_count += 1

                # Update context for next action selection
                if len(context_states) > 0:  # Need at least one state
                    context_actions.append(action)
                    context_rewards.append(reward)

                if not done and not truncated:
                    trajectory["states"].append(obs.copy())
                    context_states.append(obs.copy())

                # Save checkpoint
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

        # Memory management
        if len(trajectories) > MAX_TRAJECTORIES:
            trajectories = trajectories[-KEEP_COUNT:]
            print(f"🧹 Memory cleanup: keeping only {len(trajectories)} trajectories")

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

            total_actions = model_actions + random_actions
            model_usage = (
                (model_actions / total_actions * 100) if total_actions > 0 else 0
            )

            print(
                f"   📊 {current_timesteps:,}/{total_timesteps:,} | "
                f"Eps: {episode_count:,} | Traj: {len(trajectories)} | "
                f"Win: {win_rate:.1f}% | Model: {model_usage:.1f}% | "
                f"RAM: {memory_gb:.1f}GB ({memory_percent:.1f}%)"
            )

            # Emergency cleanup
            if memory_percent > 70:
                print(f"⚠️ MEMORY WARNING: {memory_percent:.1f}% - emergency cleanup")
                trajectories = trajectories[-5:]
                gc.collect()

    # Final checkpoint
    save_checkpoint(model, current_timesteps, save_dir)

    print(
        f"✅ Collected {len(trajectories)} trajectories over {current_timesteps:,} timesteps"
    )
    if model_actions + random_actions > 0:
        final_model_usage = model_actions / (model_actions + random_actions) * 100
        print(
            f"📈 Final action distribution: {final_model_usage:.1f}% model, {100-final_model_usage:.1f}% random"
        )

    return trajectories


def collect_and_train_periodically(
    env, total_timesteps, model, save_dir, checkpoint_interval, args
):
    """Collect trajectories and train periodically with progressive model usage"""
    print(
        f"🎮 Starting periodic collection and training with PROGRESSIVE MODEL USAGE..."
    )

    all_trajectories = []
    current_timesteps = 0
    training_round = 0

    # Progressive model usage - start with more random, gradually use model more
    base_model_probability = 0.3  # Start with 30% model usage
    max_model_probability = 0.9  # End with 90% model usage

    while current_timesteps < total_timesteps:
        remaining_steps = min(checkpoint_interval, total_timesteps - current_timesteps)
        training_round += 1

        # Calculate progressive model usage probability
        progress = current_timesteps / total_timesteps
        model_probability = (
            base_model_probability
            + (max_model_probability - base_model_probability) * progress
        )

        print(f"\n🔄 Training Round {training_round}")
        print(f"   Collecting {remaining_steps:,} timesteps...")
        print(f"   Model usage: {model_probability*100:.1f}% (progressive)")

        # Collect trajectories with current model
        batch_trajectories = collect_trajectories_with_model(
            env,
            remaining_steps,
            model,
            save_dir,
            checkpoint_interval,
            device="cuda",
            use_model_probability=model_probability,
            context_length=args.context_length,
        )

        # Add to overall collection
        all_trajectories.extend(batch_trajectories)
        current_timesteps += remaining_steps

        # Memory management
        MAX_TOTAL_TRAJECTORIES = 100
        if len(all_trajectories) > MAX_TOTAL_TRAJECTORIES:
            all_trajectories = all_trajectories[-MAX_TOTAL_TRAJECTORIES:]
            print(
                f"🧹 Total trajectory cleanup: keeping {len(all_trajectories)} trajectories"
            )

        # Filter good trajectories for training
        good_trajectories = [
            t
            for t in all_trajectories
            if len(t["rewards"]) > 10 and sum(t["rewards"]) != 0
        ]

        if len(good_trajectories) < 5:
            print("⚠️ Using all trajectories due to limited good data")
            good_trajectories = all_trajectories

        if len(good_trajectories) >= 2:
            print(
                f"🏋️ Training Decision Transformer with {len(good_trajectories)} trajectories..."
            )

            # Train the model
            model = train_decision_transformer(
                model=model,
                trajectories=good_trajectories,
                epochs=min(args.n_steps, 20),  # Limit epochs per training round
                batch_size=args.batch_size,
                lr=args.learning_rate,
                device="cuda",
                context_length=args.context_length,
            )

            # Mark model as trained
            model._is_trained = True

            # Save intermediate model
            save_checkpoint(model, current_timesteps, save_dir)

            # Clear CUDA cache
            torch.cuda.empty_cache()
            gc.collect()

            print(f"✅ Round {training_round} complete - model improved!")
        else:
            print("⚠️ Not enough good trajectories for training yet")

    return all_trajectories, model


def main():
    parser = argparse.ArgumentParser(
        description="Train Samurai Showdown Agent - Progressive Model-Based Collection"
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=10000000,  # Reduced default for testing
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
        default=100000,  # Checkpoint every 100k steps
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

    # Calculate batch size
    if args.batch_size == 0:
        context_length = args.context_length
        obs_memory_per_sample = (
            context_length * obs_shape[0] * obs_shape[1] * obs_shape[2] * 4
        )
        available_memory = 10.4 * 1024 * 1024 * 1024  # 10.4GB
        max_batch_size = int(available_memory / obs_memory_per_sample)
        args.batch_size = min(256, max(8, max_batch_size))

    print(f"🚀 Using batch size: {args.batch_size}")

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

        if args.resume.endswith(".zip"):
            import zipfile
            import tempfile

            with zipfile.ZipFile(args.resume, "r") as zip_ref:
                pth_files = [f for f in zip_ref.namelist() if f.endswith(".pth")]
                if not pth_files:
                    print(f"❌ No .pth file found in zip")
                    return

                with tempfile.NamedTemporaryFile(
                    suffix=".pth", delete=False
                ) as temp_file:
                    temp_file.write(zip_ref.read(pth_files[0]))
                    temp_path = temp_file.name

            try:
                model.load_state_dict(torch.load(temp_path, map_location=device))
                model._is_trained = True  # Mark as trained
                print(f"✅ Model loaded from .zip")
            except Exception as e:
                print(f"❌ Error loading: {e}")
                return
            finally:
                os.unlink(temp_path)
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

    # Start periodic training process with progressive model usage
    print(
        f"📊 Starting PROGRESSIVE training every {args.checkpoint_interval:,} timesteps..."
    )
    print(f"🎯 Model usage will increase from 30% to 90% over training")

    trajectories, trained_model = collect_and_train_periodically(
        env, args.total_timesteps, model, save_dir, args.checkpoint_interval, args
    )

    print(f"✅ Collected {len(trajectories)} total trajectories")

    # Final comprehensive training
    good_trajectories = [
        t for t in trajectories if len(t["rewards"]) > 10 and sum(t["rewards"]) != 0
    ]
    print(f"✅ Found {len(good_trajectories)} trajectories with reward signal")

    if len(good_trajectories) < 5:
        print("⚠️ Using all trajectories due to limited good data")
        good_trajectories = trajectories

    if len(good_trajectories) >= 2:
        print(f"🏋️ Final comprehensive training...")
        trained_model = train_decision_transformer(
            model=trained_model,
            trajectories=good_trajectories,
            epochs=args.n_steps,
            batch_size=args.batch_size,
            lr=args.learning_rate,
            device=device,
            context_length=args.context_length,
        )

    # Save final model
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
