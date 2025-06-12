from collections import deque
import gymnasium as gym
import numpy as np
import time
from datetime import datetime
import math


class SamuraiShowdownCustomWrapper(gym.Wrapper):
    """Enhanced wrapper with RGB processing and opponent pattern recognition"""

    def __init__(
        self,
        env,
        reset_round=True,
        rendering=False,
        max_episode_steps=12000,
        reward_coeff=1.0,
        prediction_horizon=30,  # How many frames to look ahead
    ):
        super(SamuraiShowdownCustomWrapper, self).__init__(env)
        self.env = env

        # Frame processing - RGB only for deep networks
        self.resize_scale = 0.75
        self.num_frames = 9
        self.frame_stack = deque(maxlen=self.num_frames)

        # Health tracking
        self.full_hp = 128
        self.prev_player_health = self.full_hp
        self.prev_opponent_health = self.full_hp

        # Reward system
        self.reward_coeff = reward_coeff

        # Episode management
        self.max_episode_steps = max_episode_steps
        self.episode_steps = 0
        self.reset_round = reset_round

        # Statistics tracking
        self.wins = 0
        self.losses = 0
        self.total_rounds = 0
        self.total_episodes = 0
        self.total_steps = 0

        # Performance tracking
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
        self.win_streak = 0
        self.best_win_streak = 0

        # Logging
        self.last_log_time = time.time()
        self.log_interval = 60
        self.session_start = time.time()

        # Get frame dimensions
        dummy_obs, _ = self.env.reset()
        original_height, original_width = dummy_obs.shape[:2]
        self.target_height = int(original_height * self.resize_scale)
        self.target_width = int(original_width * self.resize_scale)

        # Observation space for RGB ultra-deep network
        channels = self.num_frames * 3  # 9 frames × 3 RGB channels = 27 channels

        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(channels, self.target_height, self.target_width),
            dtype=np.uint8,
        )

        print(f"⚡ SAMURAI SHOWDOWN RGB WRAPPER")
        print(f"   🎮 Action space: {self.env.action_space}")
        print(f"   🎮 Using MultiBinary action space")
        print(f"   🎯 Simple rewards: Damage opponent +1, Take damage -1")
        print(f"   📏 Episode length: {max_episode_steps} steps")
        print(f"   📊 Frame stack: {self.num_frames} frames")
        print(f"   🖼️  Frame size: {self.target_height}x{self.target_width}")
        print(f"   🌈 Input channels: {channels} (RGB only)")

    def _extract_game_state(self, info):
        """Extract basic game state information"""
        # Basic health information
        player_health = info.get("health", self.full_hp)
        opponent_health = info.get("enemy_health", self.full_hp)

        return {
            "player_health": player_health,
            "opponent_health": opponent_health,
        }

    def _process_frame(self, rgb_frame):
        """RGB-only frame processing for deep networks"""
        # Ensure we have RGB input
        if len(rgb_frame.shape) == 3 and rgb_frame.shape[2] == 3:
            # Already RGB
            frame = rgb_frame
        else:
            # Convert grayscale to RGB by repeating channels
            if len(rgb_frame.shape) == 2:
                frame = np.stack([rgb_frame, rgb_frame, rgb_frame], axis=2)
            else:
                frame = rgb_frame

        # RGB resizing
        if frame.shape[:2] != (self.target_height, self.target_width):
            h_ratio = frame.shape[0] / self.target_height
            w_ratio = frame.shape[1] / self.target_width
            h_indices = (np.arange(self.target_height) * h_ratio).astype(int)
            w_indices = (np.arange(self.target_width) * w_ratio).astype(int)

            # Handle RGB channels properly
            resized = np.zeros(
                (self.target_height, self.target_width, 3), dtype=np.uint8
            )
            for c in range(3):
                resized[:, :, c] = frame[np.ix_(h_indices, w_indices, [c])].squeeze()
            return resized
        else:
            return frame

    def _stack_observation(self):
        """Stack RGB frames for ultra-deep network input"""
        frames_list = list(self.frame_stack)

        # Fill missing frames with duplicates of the first frame
        while len(frames_list) < self.num_frames:
            if len(frames_list) > 0:
                frames_list.insert(0, frames_list[0].copy())
            else:
                dummy_frame = np.zeros(
                    (self.target_height, self.target_width, 3), dtype=np.uint8
                )
                frames_list.append(dummy_frame)

        # Stack RGB frames: shape will be (27, height, width)
        # Each frame is (height, width, 3), we want (9*3, height, width)
        stacked_frames = []
        for frame in frames_list:
            # Move channels to first dimension: (3, height, width)
            frame_chw = np.transpose(frame, (2, 0, 1))
            stacked_frames.append(frame_chw)

        # Concatenate along channel dimension: (27, height, width)
        stacked = np.concatenate(stacked_frames, axis=0)

        return stacked

    def _extract_health(self, info):
        """Extract health information"""
        player_health = info.get("health", self.full_hp)
        opponent_health = info.get("enemy_health", self.full_hp)
        return player_health, opponent_health

    def _calculate_reward(
        self,
        curr_player_health,
        curr_opponent_health,
    ):
        """Simple damage-based reward calculation"""
        reward = 0.0
        done = False

        # Check for round end
        if curr_player_health <= 0 or curr_opponent_health <= 0:
            self.total_rounds += 1

            if curr_opponent_health <= 0 and curr_player_health > 0:
                # reward = 100.0  # Big win reward
                self.wins += 1
                self.win_streak += 1
                self.best_win_streak = max(self.best_win_streak, self.win_streak)
                win_rate = self.wins / self.total_rounds if self.total_rounds > 0 else 0
                print(
                    f"🏆 WIN! Round {self.total_rounds} | {self.wins}W/{self.losses}L ({win_rate:.1%}) | Streak: {self.win_streak}"
                )

            elif curr_player_health <= 0 and curr_opponent_health > 0:
                # reward = -100.0  # Big loss penalty
                self.losses += 1
                self.win_streak = 0
                win_rate = self.wins / self.total_rounds if self.total_rounds > 0 else 0
                print(
                    f"💀 LOSS! Round {self.total_rounds} | {self.wins}W/{self.losses}L ({win_rate:.1%}) | Streak: 0"
                )

            if self.reset_round:
                done = True

            self._log_periodic_stats()

        else:
            # Simple damage-based rewards
            opponent_damage = self.prev_opponent_health - curr_opponent_health
            player_damage = self.prev_player_health - curr_player_health

            if opponent_damage > 0:
                reward = 1.0  # Reward for damaging opponent
            elif player_damage > 0:
                reward = -1.0  # Penalty for taking damage

        self.prev_player_health = curr_player_health
        self.prev_opponent_health = curr_opponent_health

        return reward, done

    def _log_periodic_stats(self):
        """Simple logging with basic statistics"""
        current_time = time.time()
        time_since_last_log = current_time - self.last_log_time

        if time_since_last_log >= self.log_interval:
            session_time = current_time - self.session_start

            if self.total_rounds > 0:
                win_rate = self.wins / self.total_rounds * 100
                avg_episode_reward = (
                    np.mean(self.episode_rewards) if self.episode_rewards else 0
                )
                avg_episode_length = (
                    np.mean(self.episode_lengths) if self.episode_lengths else 0
                )

                if win_rate >= 70:
                    performance = "🏆 EXCELLENT"
                elif win_rate >= 50:
                    performance = "⚔️ GOOD"
                elif win_rate >= 30:
                    performance = "📈 IMPROVING"
                else:
                    performance = "🎯 LEARNING"

                print(f"\n📊 RGB TRAINING STATS:")
                print(f"   {performance} | Win Rate: {win_rate:.1f}%")
                print(
                    f"   🎮 Rounds: {self.total_rounds} ({self.wins}W/{self.losses}L)"
                )
                print(
                    f"   🔥 Best Streak: {self.best_win_streak} | Current: {self.win_streak}"
                )
                print(f"   💰 Avg Reward: {avg_episode_reward:.2f}")
                print(f"   ⏱️  Avg Episode: {avg_episode_length:.0f} steps")
                print(f"   🕐 Session: {session_time/60:.1f} min")
                print(f"   📈 Total Steps: {self.total_steps:,}")
                print()

            self.last_log_time = current_time

    def reset(self, **kwargs):
        """Reset environment for new episode"""
        observation, info = self.env.reset(**kwargs)

        # Reset health tracking
        self.prev_player_health = self.full_hp
        self.prev_opponent_health = self.full_hp

        # Track episode statistics
        if hasattr(self, "episode_reward"):
            self.episode_rewards.append(self.episode_reward)
        if hasattr(self, "episode_steps"):
            self.episode_lengths.append(self.episode_steps)

        self.episode_steps = 0
        self.episode_reward = 0.0
        self.total_episodes += 1

        # Initialize frame stack
        self.frame_stack.clear()
        processed_frame = self._process_frame(observation)
        self.frame_stack.append(processed_frame.copy())

        # Fill frame stack with initial frame
        for _ in range(self.num_frames - 1):
            self.frame_stack.append(processed_frame.copy())

        stacked_obs = self._stack_observation()
        return stacked_obs, info

    def step(self, action):
        """Simple step with damage-based rewards and RGB processing"""

        # Simple action handling - expect MultiBinary format
        try:
            if isinstance(action, np.ndarray) and action.shape == (12,):
                # Correct MultiBinary format
                final_action = action.astype(np.int32)
            elif isinstance(action, np.ndarray) and action.size == 12:
                # Reshape if needed
                final_action = action.reshape(12).astype(np.int32)
            else:
                print(f"⚠️ Expected MultiBinary action with 12 elements, got: {action}")
                print(
                    f"   Type: {type(action)}, Shape: {getattr(action, 'shape', 'N/A')}"
                )
                # Use no-op action as fallback
                final_action = np.zeros(12, dtype=np.int32)

        except Exception as e:
            print(f"❌ Action processing failed: {e}")
            final_action = np.zeros(12, dtype=np.int32)

        # Execute action in environment
        try:
            result = self.env.step(final_action)

            # Handle gymnasium return format
            if len(result) == 5:
                observation, reward, terminated, truncated, info = result
                done = terminated or truncated
            elif len(result) == 4:
                observation, reward, done, info = result
                truncated = False
            else:
                raise ValueError(f"Unexpected step return length: {len(result)}")

        except Exception as e:
            print(f"❌ Environment step failed: {e}")
            print(f"   Action: {final_action} (type: {type(final_action)})")

            # Return emergency fallback values
            dummy_obs = np.zeros(
                (self.target_height, self.target_width, 3), dtype=np.uint8
            )
            self.frame_stack.append(dummy_obs)
            stacked_obs = self._stack_observation()
            return stacked_obs, -1.0, True, True, {"error": str(e)}

        # Extract health and calculate simple reward
        curr_player_health, curr_opponent_health = self._extract_health(info)
        custom_reward, custom_done = self._calculate_reward(
            curr_player_health,
            curr_opponent_health,
        )

        if custom_done:
            done = custom_done

        # Process frame and update stack
        processed_frame = self._process_frame(observation)
        self.frame_stack.append(processed_frame)
        stacked_obs = self._stack_observation()

        # Update episode tracking
        self.episode_steps += 1
        self.total_steps += 1
        self.episode_reward += custom_reward

        # Check for episode timeout
        if self.episode_steps >= self.max_episode_steps:
            truncated = True
            print(f"⏰ Episode timeout at {self.episode_steps} steps")

        return stacked_obs, custom_reward, done, truncated, info

    def close(self):
        """Clean shutdown with final statistics"""
        print(f"\n🏁 FINAL RGB STATISTICS:")

        if self.total_rounds > 0:
            final_win_rate = self.wins / self.total_rounds * 100
            session_time = time.time() - self.session_start

            print(f"   🎯 Final Win Rate: {final_win_rate:.1f}%")
            print(f"   🎮 Total Rounds: {self.total_rounds}")
            print(f"   🏆 Wins: {self.wins}")
            print(f"   💀 Losses: {self.losses}")
            print(f"   🔥 Best Win Streak: {self.best_win_streak}")
            print(f"   📊 Total Episodes: {self.total_episodes}")
            print(f"   📈 Total Steps: {self.total_steps:,}")
            print(f"   🕐 Session Time: {session_time/3600:.2f} hours")

            if self.episode_rewards:
                print(f"   💰 Avg Episode Reward: {np.mean(self.episode_rewards):.2f}")
                print(
                    f"   📏 Avg Episode Length: {np.mean(self.episode_lengths):.0f} steps"
                )

            if final_win_rate >= 70:
                summary = "🏆 EXCELLENT RGB PERFORMANCE!"
            elif final_win_rate >= 50:
                summary = "⚔️ GOOD RGB PERFORMANCE!"
            elif final_win_rate >= 30:
                summary = "📈 SOLID RGB IMPROVEMENT!"
            else:
                summary = "🎯 GOOD RGB LEARNING PROGRESS!"

            print(f"   {summary}")

        super().close()

    @property
    def current_stats(self):
        """Return current training statistics"""
        win_rate = self.wins / self.total_rounds if self.total_rounds > 0 else 0
        avg_reward = np.mean(self.episode_rewards) if self.episode_rewards else 0

        return {
            "win_rate": win_rate,
            "wins": self.wins,
            "losses": self.losses,
            "total_rounds": self.total_rounds,
            "win_streak": self.win_streak,
            "best_win_streak": self.best_win_streak,
            "avg_episode_reward": avg_reward,
            "total_episodes": self.total_episodes,
            "total_steps": self.total_steps,
        }
