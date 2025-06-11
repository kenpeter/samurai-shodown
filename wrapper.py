from collections import deque
import gymnasium as gym
import numpy as np
import time
from datetime import datetime
import math


class SamuraiShowdownCustomWrapper(gym.Wrapper):
    """Optimized wrapper for single environment ultra-deep training with MultiBinary action support"""

    def __init__(
        self,
        env,
        reset_round=True,
        rendering=False,
        max_episode_steps=12000,
        reward_coeff=1.0,
    ):
        super(SamuraiShowdownCustomWrapper, self).__init__(env)
        self.env = env

        # Frame processing - optimized for deep networks
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

        # Observation space for ultra-deep network
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(self.num_frames, self.target_height, self.target_width),
            dtype=np.uint8,
        )

        # Debug action space
        print(f"⚡ STABLE-RETRO + GYMNASIUM Wrapper")
        print(f"   🎮 Action space: {self.env.action_space}")
        print(f"   🎮 Action type: {type(self.env.action_space)}")

        # FIXED: Proper action space detection
        action_space_type = type(self.env.action_space).__name__
        print(f"   🎮 Action space class: {action_space_type}")

        if action_space_type == "Discrete":
            self.action_type = "discrete"
            print(f"   🎮 Discrete actions: {self.env.action_space.n}")
        elif action_space_type == "MultiBinary":
            self.action_type = "multibinary"
            print(f"   🎮 MultiBinary shape: {self.env.action_space.shape}")
        elif (
            hasattr(self.env.action_space, "shape")
            and len(self.env.action_space.shape) == 1
        ):
            # Fallback detection for MultiBinary-like spaces
            self.action_type = "multibinary"
            print(
                f"   🎮 Detected MultiBinary-like space: {self.env.action_space.shape}"
            )
        elif hasattr(self.env.action_space, "n"):
            self.action_type = "discrete"
            print(f"   🎮 Fallback discrete actions: {self.env.action_space.n}")
        else:
            # If all else fails, assume MultiBinary for retro games
            self.action_type = "multibinary"
            print(f"   ⚠️ Unknown action space, assuming MultiBinary for retro game")

        print(f"   🎯 Rewards: Win +1.0, Lose -1.0, Attack +0.05")
        print(f"   📏 Episode length: {max_episode_steps} steps")
        print(f"   📊 Frame stack: {self.num_frames} frames")
        print(f"   🖼️  Frame size: {self.target_height}x{self.target_width}")

    def _process_frame(self, rgb_frame):
        """Optimized frame processing for deep networks"""
        if len(rgb_frame.shape) == 3 and rgb_frame.shape[2] == 3:
            gray = np.dot(rgb_frame, [0.299, 0.587, 0.114])
            gray = gray.astype(np.uint8)
        else:
            gray = rgb_frame

        if gray.shape[:2] != (self.target_height, self.target_width):
            h_ratio = gray.shape[0] / self.target_height
            w_ratio = gray.shape[1] / self.target_width
            h_indices = (np.arange(self.target_height) * h_ratio).astype(int)
            w_indices = (np.arange(self.target_width) * w_ratio).astype(int)
            resized = gray[np.ix_(h_indices, w_indices)]
            return resized
        else:
            return gray

    def _stack_observation(self):
        """Stack frames for ultra-deep network input"""
        frames_list = list(self.frame_stack)

        while len(frames_list) < self.num_frames:
            if len(frames_list) > 0:
                frames_list.insert(0, frames_list[0].copy())
            else:
                dummy_frame = np.zeros(
                    (self.target_height, self.target_width), dtype=np.uint8
                )
                frames_list.append(dummy_frame)

        stacked = np.stack(frames_list, axis=0)
        return stacked

    def _extract_health(self, info):
        """Extract health information"""
        player_health = info.get("health", self.full_hp)
        opponent_health = info.get("enemy_health", self.full_hp)
        return player_health, opponent_health

    def _calculate_reward(self, curr_player_health, curr_opponent_health):
        """Optimized reward calculation"""
        reward = 0.0
        done = False

        if curr_player_health <= 0 or curr_opponent_health <= 0:
            self.total_rounds += 1

            if curr_opponent_health <= 0 and curr_player_health > 0:
                reward = 1.0
                self.wins += 1
                self.win_streak += 1
                self.best_win_streak = max(self.best_win_streak, self.win_streak)
                win_rate = self.wins / self.total_rounds if self.total_rounds > 0 else 0
                print(
                    f"🏆 WIN! Round {self.total_rounds} | {self.wins}W/{self.losses}L ({win_rate:.1%}) | Streak: {self.win_streak}"
                )

            elif curr_player_health <= 0 and curr_opponent_health > 0:
                reward = -1.0
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
            opponent_damage = self.prev_opponent_health - curr_opponent_health
            player_damage = self.prev_player_health - curr_player_health

            if opponent_damage > 0:
                reward = 0.05
            elif player_damage > 0:
                reward = -0.02
            else:
                reward = 0.0

        self.prev_player_health = curr_player_health
        self.prev_opponent_health = curr_opponent_health

        return reward, done

    def _log_periodic_stats(self):
        """Enhanced logging"""
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

                print(f"\n📊 TRAINING STATS:")
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

    def _discrete_to_multibinary(self, action_int):
        """Convert discrete action to MultiBinary button combination"""
        # Define common button combinations for fighting games
        # Based on typical Genesis controller: [B, Y, SELECT, START, UP, DOWN, LEFT, RIGHT, A, X, L, R]

        button_combinations = [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 0: No input
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 1: B (punch/attack)
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 2: Y (kick)
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # 3: UP (jump)
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],  # 4: DOWN (crouch)
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # 5: LEFT
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],  # 6: RIGHT
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],  # 7: A (strong punch)
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],  # 8: X (strong kick)
            [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # 9: B + UP (jump attack)
            [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],  # 10: B + DOWN (crouch attack)
            [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # 11: B + LEFT (attack left)
            [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],  # 12: B + RIGHT (attack right)
            [0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # 13: Y + UP (jump kick)
            [0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],  # 14: Y + DOWN (crouch kick)
            [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # 15: Y + LEFT (kick left)
            [0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],  # 16: Y + RIGHT (kick right)
            [0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0],  # 17: LEFT + A (strong left)
            [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0],  # 18: RIGHT + A (strong right)
            [0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0],  # 19: DOWN + A (special move)
            [0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],  # 20: UP + A (anti-air)
            [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],  # 21: DOWN + LEFT
            [0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0],  # 22: DOWN + RIGHT
            [0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0],  # 23: UP + LEFT
            [0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],  # 24: UP + RIGHT
        ]

        # Clamp action to valid range
        action_int = action_int % len(button_combinations)
        return np.array(button_combinations[action_int], dtype=np.int32)

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
        """FIXED: Handle both Discrete and MultiBinary action spaces"""

        # Debug action format (only for first few steps)
        if self.total_steps < 5:
            print(
                f"🔍 Raw action: type={type(action)}, shape={getattr(action, 'shape', 'N/A')}, value={action}"
            )

        # Handle different action formats based on action space type
        try:
            if self.action_type == "multibinary":
                # MultiBinary action space (button combinations)
                if isinstance(action, np.ndarray):
                    if action.shape == (12,):  # Correct MultiBinary format
                        action_array = action.astype(np.int32)
                        if self.total_steps < 5:
                            print(f"🎮 MultiBinary action: {action_array}")
                    elif action.ndim == 0:  # Single value - convert to MultiBinary
                        action_int = int(action.item())
                        action_array = self._discrete_to_multibinary(action_int)
                        if self.total_steps < 5:
                            print(
                                f"🎮 Converted {action_int} to MultiBinary: {action_array}"
                            )
                    elif action.size == 1:  # 1-element array
                        action_int = int(action.flatten()[0])
                        action_array = self._discrete_to_multibinary(action_int)
                        if self.total_steps < 5:
                            print(
                                f"🎮 Converted {action_int} to MultiBinary: {action_array}"
                            )
                    else:
                        # Handle unexpected shapes
                        if len(action.flatten()) >= 12:
                            action_array = action.flatten()[:12].astype(np.int32)
                        else:
                            action_int = int(action.flatten()[0])
                            action_array = self._discrete_to_multibinary(action_int)
                        if self.total_steps < 5:
                            print(f"🎮 Handled unexpected shape: {action_array}")

                elif isinstance(action, (int, np.integer)):
                    # Convert discrete action to MultiBinary
                    action_array = self._discrete_to_multibinary(int(action))
                    if self.total_steps < 5:
                        print(f"🎮 Converted {action} to MultiBinary: {action_array}")

                else:
                    print(f"⚠️ Unknown action type: {type(action)}, using no-op")
                    action_array = np.zeros(12, dtype=np.int32)

                # Ensure correct format
                if action_array.shape != (12,):
                    print(f"⚠️ Wrong action shape: {action_array.shape}, fixing")
                    if len(action_array) < 12:
                        action_array = np.pad(
                            action_array, (0, 12 - len(action_array)), "constant"
                        )
                    else:
                        action_array = action_array[:12]
                    action_array = action_array.astype(np.int32)

                final_action = action_array

            else:
                # Discrete action space
                if isinstance(action, np.ndarray):
                    if action.ndim == 0:
                        action_int = int(action.item())
                    elif action.size == 1:
                        action_int = int(action.flatten()[0])
                    else:
                        action_int = int(action.flatten()[0])
                        print(
                            f"⚠️ Multi-element array for discrete action: {action}, using first element"
                        )

                elif hasattr(action, "item"):
                    action_int = int(action.item())

                elif isinstance(action, (int, np.integer)):
                    action_int = int(action)

                else:
                    action_int = int(action)
                    print(
                        f"⚠️ Unusual action type: {type(action)}, converted to: {action_int}"
                    )

                # Validate discrete action range
                if hasattr(self.env.action_space, "n"):
                    if not (0 <= action_int < self.env.action_space.n):
                        print(
                            f"⚠️ Action {action_int} out of range [0, {self.env.action_space.n-1}], clipping"
                        )
                        action_int = np.clip(action_int, 0, self.env.action_space.n - 1)

                final_action = action_int
                if self.total_steps < 5:
                    print(f"🎮 Discrete action: {final_action}")

        except Exception as e:
            print(f"❌ Action processing failed: {e}")
            if self.action_type == "multibinary":
                final_action = np.zeros(12, dtype=np.int32)
            else:
                final_action = 0

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
                (self.target_height, self.target_width), dtype=np.uint8
            )
            self.frame_stack.append(dummy_obs)
            stacked_obs = self._stack_observation()
            return stacked_obs, -1.0, True, True, {"error": str(e)}

        # Extract health and calculate custom reward
        curr_player_health, curr_opponent_health = self._extract_health(info)
        custom_reward, custom_done = self._calculate_reward(
            curr_player_health, curr_opponent_health
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
        print(f"\n🏁 FINAL STATISTICS:")

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
                summary = "🏆 EXCELLENT PERFORMANCE!"
            elif final_win_rate >= 50:
                summary = "⚔️ GOOD PERFORMANCE!"
            elif final_win_rate >= 30:
                summary = "📈 SOLID IMPROVEMENT!"
            else:
                summary = "🎯 GOOD LEARNING PROGRESS!"

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
