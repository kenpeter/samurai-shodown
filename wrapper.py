import collections
import gymnasium as gym
import numpy as np
import time
from datetime import datetime
import math


class SamuraiShowdownCustomWrapper(gym.Wrapper):
    """Fixed wrapper - Clear win/loss tracking and rewards"""

    # Global tracking across all environments
    _global_stats = {
        "total_wins": 0,
        "total_losses": 0,
        "total_rounds": 0,
        "env_stats": {},
        "last_log_time": time.time(),
        "session_start": time.time(),
    }

    def __init__(
        self,
        env,
        reset_round=True,
        rendering=False,
        max_episode_steps=10000,
        reward_coeff=3.0,
    ):
        super(SamuraiShowdownCustomWrapper, self).__init__(env)
        self.env = env

        # Frame processing parameters
        self.resize_scale = 0.75
        self.num_frames = 9
        self.frame_stack = collections.deque(maxlen=self.num_frames)

        # Reward parameters
        self.reward_coeff = reward_coeff
        self.total_timesteps = 0

        # Rendering parameter
        self.rendering = rendering

        # Episode management
        self.max_episode_steps = max_episode_steps
        self.episode_steps = 0
        self.reset_round = reset_round

        # Health tracking - CRITICAL FIX
        self.full_hp = 128
        self.prev_player_health = self.full_hp
        self.prev_opponent_health = self.full_hp
        self.round_ended = False  # Track if round has ended

        # Environment tracking
        import random

        self.env_id = f"ENV-{random.randint(1000, 9999)}"
        self.wins = 0
        self.losses = 0
        self.total_rounds = 0

        # Initialize global tracking
        SamuraiShowdownCustomWrapper._global_stats["env_stats"][self.env_id] = {
            "wins": 0,
            "losses": 0,
            "total_rounds": 0,
        }

        # Logging configuration
        self.log_interval = 120

        # Get observation space dimensions
        dummy_obs = self.env.reset()
        if isinstance(dummy_obs, tuple):
            actual_obs = dummy_obs[0]
        else:
            actual_obs = dummy_obs

        original_height, original_width = actual_obs.shape[:2]
        self.target_height = int(original_height * self.resize_scale)
        self.target_width = int(original_width * self.resize_scale)

        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(self.num_frames, self.target_height, self.target_width),
            dtype=np.uint8,
        )

        print(f"🔧 {self.env_id} FIXED Wrapper - Clear Win/Loss Tracking")
        print(f"   🎯 Rewards: Progressive based on health + win/loss bonuses")
        print(f"   📏 Episode length: {max_episode_steps} steps")
        print(f"   🏥 Health tracking: Player vs Opponent")

    def _process_frame(self, rgb_frame):
        """Convert RGB frame to grayscale and resize"""
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
        """Stack frames in channels-first format"""
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

    def _log_periodic_stats(self):
        """Log win/loss stats periodically"""
        current_time = time.time()
        time_since_last_log = (
            current_time - SamuraiShowdownCustomWrapper._global_stats["last_log_time"]
        )

        if time_since_last_log >= self.log_interval:
            global_stats = SamuraiShowdownCustomWrapper._global_stats

            print(f"\n📊 WIN/LOSS STATS UPDATE:")
            print(
                f"   🌍 Global: {global_stats['total_wins']}W/{global_stats['total_losses']}L"
            )

            for env_id, stats in global_stats["env_stats"].items():
                if stats["total_rounds"] > 0:
                    win_rate = stats["wins"] / stats["total_rounds"] * 100
                    if win_rate >= 60:
                        emoji = "🏆"
                    elif win_rate >= 40:
                        emoji = "⚔️"
                    else:
                        emoji = "📈"
                    print(
                        f"   {emoji} {env_id}: {stats['wins']}W/{stats['losses']}L ({win_rate:.1f}%) - {stats['total_rounds']} rounds"
                    )
                else:
                    print(f"   ⏳ {env_id}: No completed rounds yet")

            print()
            global_stats["last_log_time"] = current_time

    def _calculate_reward_and_done(self, curr_player_health, curr_opponent_health):
        """FIXED: Calculate reward using your original reward logic"""
        done = False

        # Check for round end conditions
        player_dead = curr_player_health < 0
        opponent_dead = curr_opponent_health < 0

        if player_dead:
            if not self.round_ended:  # Only process once per round
                self.round_ended = True
                self.total_rounds += 1
                self.losses += 1

                # Update global stats
                SamuraiShowdownCustomWrapper._global_stats["total_rounds"] += 1
                SamuraiShowdownCustomWrapper._global_stats["total_losses"] += 1
                SamuraiShowdownCustomWrapper._global_stats["env_stats"][self.env_id][
                    "total_rounds"
                ] = self.total_rounds
                SamuraiShowdownCustomWrapper._global_stats["env_stats"][self.env_id][
                    "losses"
                ] = self.losses

                # Your original loss reward: penalty based on opponent's remaining health
                custom_reward = -math.pow(
                    self.full_hp, (curr_opponent_health + 1) / (self.full_hp + 1)
                )

                win_rate = (
                    self.wins / self.total_rounds * 100 if self.total_rounds > 0 else 0
                )
                print(
                    f"💀 {self.env_id} LOSS! Opponent: {curr_opponent_health}/{self.full_hp} | {self.wins}W/{self.losses}L ({win_rate:.1f}%)"
                )

                # End episode if reset_round is True
                if self.reset_round:
                    done = True

                return custom_reward, done

        elif opponent_dead:
            if not self.round_ended:  # Only process once per round
                self.round_ended = True
                self.total_rounds += 1
                self.wins += 1

                # Update global stats
                SamuraiShowdownCustomWrapper._global_stats["total_rounds"] += 1
                SamuraiShowdownCustomWrapper._global_stats["total_wins"] += 1
                SamuraiShowdownCustomWrapper._global_stats["env_stats"][self.env_id][
                    "total_rounds"
                ] = self.total_rounds
                SamuraiShowdownCustomWrapper._global_stats["env_stats"][self.env_id][
                    "wins"
                ] = self.wins

                # Your original win reward: reward based on player's remaining health
                custom_reward = (
                    math.pow(
                        self.full_hp, (curr_player_health + 1) / (self.full_hp + 1)
                    )
                ) * self.reward_coeff

                win_rate = self.wins / self.total_rounds * 100
                print(
                    f"🏆 {self.env_id} WIN! Health: {curr_player_health}/{self.full_hp} | {self.wins}W/{self.losses}L ({win_rate:.1f}%)"
                )

                # End episode if reset_round is True
                if self.reset_round:
                    done = True

                return custom_reward, done
        else:
            # Fighting continues - your original damage-based reward
            damage_dealt = self.prev_opponent_health - curr_opponent_health
            damage_received = self.prev_player_health - curr_player_health
            custom_reward = self.reward_coeff * damage_dealt - damage_received

        return custom_reward, done

    def reset(self, **kwargs):
        """Reset environment"""
        result = self.env.reset(**kwargs)
        if isinstance(result, tuple):
            observation, info = result
        else:
            observation = result
            info = {}

        # Reset health tracking
        self.prev_player_health = self.full_hp
        self.prev_opponent_health = self.full_hp
        self.episode_steps = 0
        self.round_ended = False  # Reset round flag

        # Reset frame stack
        self.frame_stack.clear()
        processed_frame = self._process_frame(observation)
        self.frame_stack.append(processed_frame.copy())

        for _ in range(self.num_frames - 1):
            zero_frame = np.zeros_like(processed_frame)
            self.frame_stack.append(zero_frame)

        stacked_obs = self._stack_observation()
        return stacked_obs, info

    def step(self, action):
        """FIXED step function with proper win/loss tracking"""
        # Take action in environment
        observation, original_reward, done, truncated, info = self.env.step(action)

        # Process frame
        processed_frame = self._process_frame(observation)
        self.frame_stack.append(processed_frame)

        # Render if needed
        if self.rendering:
            self.env.render()
            time.sleep(0.01)

        # Extract current health - CRITICAL: Handle missing health info
        curr_player_health = info.get("health", self.prev_player_health)
        curr_opponent_health = info.get("enemy_health", self.prev_opponent_health)

        # Ensure health values are valid
        if curr_player_health is None:
            curr_player_health = self.prev_player_health
        if curr_opponent_health is None:
            curr_opponent_health = self.prev_opponent_health

        # Calculate custom reward and check for round end
        custom_reward, custom_done = self._calculate_reward_and_done(
            curr_player_health, curr_opponent_health
        )

        # Update health tracking (only if round hasn't ended)
        if not self.round_ended:
            self.prev_player_health = curr_player_health
            self.prev_opponent_health = curr_opponent_health

        # Check episode step limit
        self.episode_steps += 1
        if self.episode_steps >= self.max_episode_steps:
            truncated = True

        # Use custom done flag
        if custom_done:
            done = True

        # Periodic logging
        self._log_periodic_stats()

        # Return stacked observation with custom reward
        stacked_obs = self._stack_observation()

        return stacked_obs, custom_reward, done, truncated, info

    @classmethod
    def print_final_stats(cls):
        """Print final statistics when training ends"""
        global_stats = cls._global_stats
        print(f"\n🏁 FINAL TRAINING STATS:")
        print(
            f"   🌍 Total: {global_stats['total_wins']}W/{global_stats['total_losses']}L"
        )

        for env_id, stats in global_stats["env_stats"].items():
            if stats["total_rounds"] > 0:
                win_rate = stats["wins"] / stats["total_rounds"] * 100
                print(
                    f"   📊 {env_id}: {stats['wins']}W/{stats['losses']}L ({win_rate:.1f}%)"
                )
