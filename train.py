import os
import sys
import argparse
import time
import torch

import retro
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv

# Import the wrapper
from wrapper import SamuraiShowdownCustomWrapper


def make_env_for_subprocess(game, state, rendering=False, seed=0):
    """Create environment function for SubprocVecEnv - all imports MUST be inside"""

    def _init():
        # Import everything inside the function for subprocess compatibility
        import retro
        import gymnasium as gym
        from stable_baselines3.common.monitor import Monitor
        from wrapper import SamuraiShowdownCustomWrapper

        # Create environment
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
            max_episode_steps=5000,
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


def main():
    parser = argparse.ArgumentParser(description="Train Samurai Showdown Agent")
    parser.add_argument(
        "--total-timesteps", type=int, default=10000000, help="Total timesteps to train"
    )
    parser.add_argument(
        "--num-envs", type=int, default=38, help="Number of parallel environments"
    )
    parser.add_argument(
        "--learning-rate", type=float, default=1e-3, help="Learning rate"
    )
    parser.add_argument(
        "--resume", type=str, default=None, help="Resume from saved model path"
    )
    parser.add_argument("--render", action="store_true", help="Enable rendering")
    parser.add_argument(
        "--use-default-state", action="store_true", help="Use default game state"
    )

    args = parser.parse_args()

    # GPU Check and Setup - MANDATORY
    print("🔍 Checking GPU availability...")
    if not torch.cuda.is_available():
        print("❌ ERROR: No GPU available!")
        print("💡 This training script requires CUDA GPU")
        print("💡 Check: nvidia-smi")
        print(
            "💡 Install: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
        )
        sys.exit(1)

    gpu_count = torch.cuda.device_count()
    current_gpu = torch.cuda.current_device()
    gpu_name = torch.cuda.get_device_name(current_gpu)
    gpu_memory = torch.cuda.get_device_properties(current_gpu).total_memory / 1e9

    print(f"✅ GPU Detected: {gpu_name}")
    print(f"   GPU Memory: {gpu_memory:.1f} GB")
    print(f"   Available GPUs: {gpu_count}")
    print(f"   Current GPU: {current_gpu}")

    device = "cuda"

    # Set CUDA optimizations
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False  # For maximum performance
    print("✅ CUDA optimizations enabled")

    # Memory requirement check
    estimated_vram = args.num_envs * 0.2  # ~200MB per environment
    print(f"📊 Estimated VRAM usage: {estimated_vram:.1f} GB")

    if estimated_vram > gpu_memory * 0.9:
        print("❌ ERROR: Insufficient GPU memory!")
        print(f"   Required: {estimated_vram:.1f} GB")
        print(f"   Available: {gpu_memory:.1f} GB")
        print("💡 Reduce --num-envs or use a GPU with more memory")
        sys.exit(1)
    elif estimated_vram > gpu_memory * 0.7:
        print("⚠️ Warning: High VRAM usage expected")
        print("💡 Monitor with: watch -n 1 nvidia-smi")

    game = "SamuraiShodown-Genesis"

    # Test if the game works (exactly like test.py)
    print(f"🎮 Testing {game}...")
    try:
        test_env = retro.make(
            game=game,
            state=None,
            use_restricted_actions=retro.Actions.FILTERED,
            obs_type=retro.Observations.IMAGE,
            render_mode=None,
        )
        test_env.close()
        print(f"✅ Basic environment test passed")
    except Exception as e:
        print(f"❌ Basic environment test failed: {e}")
        return

    # Handle state (exactly like test.py)
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

    save_dir = "trained_models_samurai"
    os.makedirs(save_dir, exist_ok=True)

    print(f"🚀 Samurai Showdown Training - GPU ONLY")
    print(f"   Game: {game}")
    print(f"   State: {state}")
    print(f"   Device: {device} (GPU MANDATORY)")
    print(f"   Total timesteps: {args.total_timesteps:,}")
    print(f"   Environments: {args.num_envs}")
    print(f"   Learning rate: {args.learning_rate}")

    # Create environments with SubprocVecEnv only
    print(f"🔧 Creating {args.num_envs} environments with SubprocVecEnv...")
    try:
        if args.render and args.num_envs > 1:
            print(
                "⚠️ Warning: Rendering with multiple environments will open multiple windows!"
            )
            print("💡 Consider using --num-envs 1 for rendering")

            # Ask for confirmation
            response = input("Continue with rendering all environments? (y/N): ")
            if response.lower() != "y":
                print("Disabling rendering for multiple environments...")
                args.render = False

        # All environments must have the same render_mode for SubprocVecEnv
        env_fns = [
            make_env_for_subprocess(
                game,
                state=state,
                rendering=args.render,  # Same rendering for ALL environments
                seed=i,
            )
            for i in range(args.num_envs)
        ]

        env = SubprocVecEnv(env_fns)
        if args.render:
            print(
                f"✅ {args.num_envs} environments created with SubprocVecEnv (all rendering)"
            )
        else:
            print(
                f"✅ {args.num_envs} environments created with SubprocVecEnv (no rendering)"
            )

    except Exception as e:
        print(f"❌ Failed to create SubprocVecEnv environments: {e}")
        import traceback

        traceback.print_exc()
        return

    # Create or load model - GPU ONLY
    if args.resume and os.path.exists(args.resume):
        print(f"📂 Loading model from: {args.resume}")
        model = PPO.load(args.resume, env=env, device="cuda")
        print(f"✅ Model loaded on GPU")
    else:
        print(f"🧠 Creating new PPO model on GPU")
        lr_schedule = linear_schedule(args.learning_rate, args.learning_rate * 0.3)

        model = PPO(
            "CnnPolicy",
            env,
            device="cuda",  # GPU ONLY
            verbose=1,
            n_steps=1024,
            batch_size=1024,
            n_epochs=8,
            gamma=0.995,
            learning_rate=lr_schedule,
            clip_range=linear_schedule(0.2, 0.1),
            ent_coef=0.03,
            vf_coef=0.8,
            max_grad_norm=0.5,
            gae_lambda=0.95,
            tensorboard_log="logs_samurai",
        )
        print(f"✅ PPO model created on GPU")

        # Verify model is on GPU
        print(f"🔍 Model device verification:")
        for name, param in model.policy.named_parameters():
            print(f"   {name}: {param.device}")
            break  # Just show first parameter as example

    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=1000000 // args.num_envs,  # Checkpoint every ~1M steps
        save_path=save_dir,
        name_prefix="ppo_samurai",
    )

    # Training - GPU ONLY
    start_time = time.time()
    print(f"🏋️ Starting GPU training for {args.total_timesteps:,} timesteps")
    print("💡 Monitor GPU usage with: watch -n 1 nvidia-smi")

    try:
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=[checkpoint_callback],
            reset_num_timesteps=not bool(args.resume),
        )

        training_time = time.time() - start_time
        print(f"🎉 GPU training completed in {training_time/3600:.1f} hours!")

    except KeyboardInterrupt:
        print(f"⏹️ Training interrupted")
        training_time = time.time() - start_time
        print(f"Training time: {training_time/3600:.1f} hours")

    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback

        traceback.print_exc()

    finally:
        env.close()
        torch.cuda.empty_cache()  # Clear GPU memory
        print("🧹 GPU memory cleared")

    # Save final model
    final_model_path = os.path.join(save_dir, "ppo_samurai_final.zip")
    model.save(final_model_path)
    print(f"💾 Final model saved to: {final_model_path}")

    print("✅ Training complete!")


if __name__ == "__main__":
    main()
