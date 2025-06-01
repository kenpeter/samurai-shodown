import os
import sys
import argparse
import time

import retro
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv

# Import the wrapper
from wrapper import SamuraiShowdownCustomWrapper


def create_env(game, state, rendering=False, seed=0):
    """Create a single environment (exactly like test.py)"""

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
        "--num-envs", type=int, default=1, help="Number of parallel environments"
    )
    parser.add_argument(
        "--learning-rate", type=float, default=3e-4, help="Learning rate"
    )
    parser.add_argument(
        "--resume", type=str, default=None, help="Resume from saved model path"
    )
    parser.add_argument("--render", action="store_true", help="Enable rendering")
    parser.add_argument(
        "--use-default-state", action="store_true", help="Use default game state"
    )

    args = parser.parse_args()

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

    print(f"🚀 Samurai Showdown Training")
    print(f"   Game: {game}")
    print(f"   State: {state}")
    print(f"   Total timesteps: {args.total_timesteps:,}")
    print(f"   Environments: {args.num_envs}")
    print(f"   Learning rate: {args.learning_rate}")

    # Create environments (no subprocess - exactly like test.py)
    print(f"🔧 Creating {args.num_envs} environment(s)...")
    try:
        if args.num_envs == 1:
            # Single environment - direct creation
            env = create_env(game, state=state, rendering=args.render, seed=0)
            # Wrap in DummyVecEnv for stable-baselines3 compatibility
            env = DummyVecEnv([lambda: env])
            print("✅ Single environment created")
        else:
            # Multiple environments - create each one directly
            envs = []
            for i in range(args.num_envs):
                env_instance = create_env(
                    game,
                    state=state,
                    rendering=(args.render and i == 0),  # Only render first env
                    seed=i,
                )
                envs.append(lambda env=env_instance: env)

            env = DummyVecEnv(envs)
            print(f"✅ {args.num_envs} environments created")

    except Exception as e:
        print(f"❌ Failed to create environments: {e}")
        import traceback

        traceback.print_exc()
        return

    # Create or load model
    if args.resume and os.path.exists(args.resume):
        print(f"📂 Loading model from: {args.resume}")
        model = PPO.load(args.resume, env=env, device="cuda")
        print("✅ Model loaded, resuming training")
    else:
        print("🧠 Creating new PPO model")
        lr_schedule = linear_schedule(args.learning_rate, args.learning_rate * 0.3)

        model = PPO(
            "CnnPolicy",
            env,
            device="cuda" if args.num_envs > 1 else "cpu",
            verbose=1,
            n_steps=1024 if args.num_envs > 1 else 512,
            batch_size=512 if args.num_envs > 1 else 256,
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

    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=max(100000 // args.num_envs, 10000),
        save_path=save_dir,
        name_prefix="ppo_samurai",
    )

    # Training
    start_time = time.time()
    print(f"🏋️ Starting training for {args.total_timesteps:,} timesteps")

    try:
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=[checkpoint_callback],
            reset_num_timesteps=not bool(args.resume),
        )

        training_time = time.time() - start_time
        print(f"🎉 Training completed in {training_time/3600:.1f} hours!")

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

    # Save final model
    final_model_path = os.path.join(save_dir, "ppo_samurai_final.zip")
    model.save(final_model_path)
    print(f"💾 Final model saved to: {final_model_path}")

    print("✅ Training complete!")


if __name__ == "__main__":
    main()
