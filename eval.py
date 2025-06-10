import os
import argparse
import time
import numpy as np
import torch
import zipfile
import tempfile

import retro
import gymnasium as gym

# Import the wrapper and Decision Transformer
from wrapper import SamuraiShowdownCustomWrapper, DecisionTransformer


def create_eval_env(game, state):
    """Create evaluation environment aligned with training setup"""
    if state and os.path.isfile(state):
        state_file = os.path.abspath(state)
        print(f"Using custom state file: {state_file}")
    else:
        state_file = state
        print(f"Using state: {state_file if state_file else 'default'}")

    # Create retro environment with rendering enabled
    env = retro.make(
        game=game,
        state=state_file,
        use_restricted_actions=retro.Actions.FILTERED,
        obs_type=retro.Observations.IMAGE,
        render_mode="human",  # Enable rendering
    )

    # Apply custom wrapper with same settings as training
    env = SamuraiShowdownCustomWrapper(
        env,
        reset_round=True,
        rendering=True,
        max_episode_steps=5000,
    )

    print(f"🔍 Environment observation space: {env.observation_space.shape}")
    return env


def load_decision_transformer_from_zip(zip_path, env, device="cuda"):
    """Load Decision Transformer model from .zip file"""
    print(f"📂 Loading model from: {zip_path}")

    # Extract .pth file from .zip
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        # Find .pth file in zip
        pth_files = [f for f in zip_ref.namelist() if f.endswith(".pth")]
        if not pth_files:
            raise ValueError("No .pth file found in zip archive")

        pth_file = pth_files[0]
        print(f"Found model file: {pth_file}")

        # Extract to temporary file
        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as temp_file:
            temp_file.write(zip_ref.read(pth_file))
            temp_path = temp_file.name

    try:
        # Create model with same architecture as training
        obs_shape = env.observation_space.shape
        action_dim = env.action_space.n

        model = DecisionTransformer(
            observation_shape=obs_shape,
            action_dim=action_dim,
            hidden_size=256,
            n_layer=4,
            n_head=4,
            max_ep_len=2000,
        )

        # Load state dict
        model.load_state_dict(torch.load(temp_path, map_location=device))
        model.to(device)
        model.eval()

        print(f"✅ Model loaded successfully on {device}")
        return model

    finally:
        # Clean up temp file
        os.unlink(temp_path)


class DecisionTransformerPlayer:
    """Wrapper for Decision Transformer evaluation"""

    def __init__(self, model, device, context_length=30):
        self.model = model
        self.device = device
        self.context_length = context_length

        # Initialize context buffers
        self.reset_context()

    def reset_context(self):
        """Reset context for new episode"""
        self.states = []
        self.actions = []
        self.returns_to_go = []
        self.timesteps = []

    def get_action(self, state, target_return=1.0):
        """Get action from Decision Transformer"""
        # Add current state to context
        self.states.append(state)

        # Calculate current timestep
        current_timestep = len(self.states) - 1
        self.timesteps.append(current_timestep)

        # Set return-to-go (target performance)
        self.returns_to_go.append(target_return)

        # Trim context to maximum length
        if len(self.states) > self.context_length:
            self.states = self.states[-self.context_length :]
            self.actions = self.actions[
                -(self.context_length - 1) :
            ]  # One less action than states
            self.returns_to_go = self.returns_to_go[-self.context_length :]
            self.timesteps = self.timesteps[-self.context_length :]

        # Prepare tensors for model
        states_tensor = (
            torch.from_numpy(np.array(self.states)).float().unsqueeze(0).to(self.device)
            / 255.0
        )

        # Pad actions if needed (actions are one less than states)
        if len(self.actions) == 0:
            actions_pad = [0]  # Dummy action for first step
        else:
            actions_pad = self.actions.copy()

        # Pad to match states length
        while len(actions_pad) < len(self.states):
            actions_pad.append(0)

        actions_tensor = (
            torch.from_numpy(np.array(actions_pad)).long().unsqueeze(0).to(self.device)
        )
        returns_tensor = (
            torch.from_numpy(np.array(self.returns_to_go))
            .float()
            .unsqueeze(0)
            .to(self.device)
        )
        timesteps_tensor = (
            torch.from_numpy(np.array(self.timesteps))
            .long()
            .unsqueeze(0)
            .to(self.device)
        )

        # Get action from model
        with torch.no_grad():
            action = self.model.get_action(
                states_tensor,
                actions_tensor,
                returns_tensor,
                timesteps_tensor,
                temperature=0.1,  # Low temperature for more deterministic actions
            )

        # Add action to context for next step
        self.actions.append(action)

        return action


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate trained Decision Transformer Samurai Showdown Agent"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="trained_models/model_4450000_steps.zip",
        help="Path to the trained Decision Transformer model (.zip file)",
    )
    parser.add_argument(
        "--state-file",
        type=str,
        default="samurai.state",
        help="State file to use (default: samurai.state)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=5,
        help="Number of episodes to run",
    )
    parser.add_argument(
        "--use-default-state",
        action="store_true",
        help="Use default game state (ignore state file)",
    )
    parser.add_argument(
        "--target-return",
        type=float,
        default=1.0,
        help="Target return for Decision Transformer (higher = more aggressive)",
    )
    parser.add_argument(
        "--context-length",
        type=int,
        default=30,
        help="Context length for Decision Transformer",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=60,
        help="Target FPS for rendering",
    )

    args = parser.parse_args()

    # Check if model exists
    if not os.path.exists(args.model_path):
        print(f"❌ Error: Model file not found at {args.model_path}")
        print("Available models in trained_models/:")
        if os.path.exists("trained_models"):
            for f in os.listdir("trained_models"):
                if f.endswith(".zip"):
                    print(f"   - {f}")
        return

    game = "SamuraiShodown-Genesis"

    # Handle state file
    if args.use_default_state:
        state_file = None
        print("Using default game state")
    else:
        state_file = args.state_file

    print(f"🤖 Decision Transformer Evaluation")
    print(f"📂 Model: {args.model_path}")
    print(f"🎮 State: {state_file if state_file else 'default'}")
    print(f"🔄 Episodes: {args.episodes}")
    print(f"🎯 Target Return: {args.target_return}")
    print(f"📏 Context Length: {args.context_length}")
    print(f"⚡ FPS: {args.fps}")
    print("=" * 60)

    # Create evaluation environment
    try:
        env = create_eval_env(game, state_file)
        print("✅ Environment created successfully!")
    except Exception as e:
        print(f"❌ Error creating environment: {e}")
        return

    # Load the Decision Transformer model
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🔧 Using device: {device}")

        model = load_decision_transformer_from_zip(args.model_path, env, device)

        # Create player
        player = DecisionTransformerPlayer(
            model, device, context_length=args.context_length
        )

    except Exception as e:
        print(f"❌ Error loading model: {e}")
        import traceback

        traceback.print_exc()
        return

    # Calculate frame timing
    frame_time = 1.0 / args.fps

    try:
        total_wins = 0
        total_losses = 0
        total_episodes = 0
        total_reward = 0
        total_steps = 0

        for episode in range(args.episodes):
            print(f"\n⚔️  --- Episode {episode + 1}/{args.episodes} ---")

            # Reset environment and player context
            obs, info = env.reset()
            player.reset_context()

            episode_reward = 0
            step_count = 0
            episode_start_time = time.time()

            print("🎬 Starting new match... Watch the game window!")

            while True:
                step_start_time = time.time()

                # Get action from Decision Transformer
                action = player.get_action(obs, target_return=args.target_return)

                # Take step in environment
                obs, reward, terminated, truncated, info = env.step(action)

                episode_reward += reward
                step_count += 1

                # Frame rate limiting
                elapsed = time.time() - step_start_time
                if elapsed < frame_time:
                    time.sleep(frame_time - elapsed)

                # Check if episode is done
                if terminated or truncated:
                    break

                # Info display every 5 seconds
                if step_count % (args.fps * 5) == 0:
                    player_hp = info.get("health", "?")
                    enemy_hp = info.get("enemy_health", "?")
                    print(
                        f"   Step {step_count}: Player HP: {player_hp}, Enemy HP: {enemy_hp}"
                    )

            # Episode finished
            episode_time = time.time() - episode_start_time
            total_episodes += 1
            total_reward += episode_reward
            total_steps += step_count

            print(f"🏁 Episode {episode + 1} finished!")
            print(f"   Total reward: {episode_reward:.1f}")
            print(f"   Steps taken: {step_count}")
            print(f"   Episode duration: {episode_time:.1f}s")

            # Get final health values
            player_hp = info.get("health", 0)
            enemy_hp = info.get("enemy_health", 0)
            print(f"   Final - Player HP: {player_hp}, Enemy HP: {enemy_hp}")

            # Determine winner
            if player_hp <= 0 and enemy_hp > 0:
                print("   🔴 AI Lost this round")
                total_losses += 1
            elif enemy_hp <= 0 and player_hp > 0:
                print("   🟢 AI Won this round")
                total_wins += 1
            else:
                print("   ⚪ Round ended without clear winner")

            # Pause between episodes
            if episode < args.episodes - 1:
                print("\n⏳ Waiting 3 seconds before next episode...")
                time.sleep(3)

        # Final statistics
        print(f"\n📊 Final Results:")
        print(f"   Episodes: {total_episodes}")
        print(f"   Wins: {total_wins}")
        print(f"   Losses: {total_losses}")
        print(f"   Draws/Timeouts: {total_episodes - total_wins - total_losses}")

        if total_episodes > 0:
            win_rate = (total_wins / total_episodes) * 100
            print(f"   Win Rate: {win_rate:.1f}%")
            avg_reward = total_reward / total_episodes
            avg_steps = total_steps / total_episodes
            print(f"   Average Reward: {avg_reward:.1f}")
            print(f"   Average Steps: {avg_steps:.0f}")

            # Performance assessment
            if win_rate >= 70:
                print("🏆 Excellent Decision Transformer performance!")
            elif win_rate >= 50:
                print("👍 Good Decision Transformer performance!")
            elif win_rate >= 30:
                print("📈 Improving Decision Transformer performance!")
            else:
                print("🔧 Decision Transformer needs more training!")

    except KeyboardInterrupt:
        print("\n\n⏹️  Evaluation interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during evaluation: {e}")
        import traceback

        traceback.print_exc()
    finally:
        env.close()
        print("\n✅ Decision Transformer evaluation complete!")


if __name__ == "__main__":
    main()
