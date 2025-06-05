import collections
import gymnasium as gym
import numpy as np
import time
from datetime import datetime
import math


class SamuraiShowdownCustomWrapper(gym.Wrapper):
    """Simple wrapper - Win/Loss rewards only, optimized for large batch sizes"""

    # Global tracking across all environments
    _global_stats = {
        "total_wins": 0,
        "total_losses": 0,
        "total_rounds": 0,
        "env_stats": {},
        "last_log_time": time.time(),
        "session_start": time.time(),
    }

    def __init__(self, env, reset_round=True, rendering=False, max_episode_steps=10000):
        super(SamuraiShowdownCustomWrapper, self).__init__(env)
        self.env = env

        # Frame processing parameters
        self.resize_scale = 0.75
        self.num_frames = 9
        self.frame_stack = collections.deque(maxlen=self.num_frames)

        # Health tracking
        self.full_hp = 128
        self.prev_player_health = self.full_hp
        self.prev_opponent_health = self.full_hp

        # Episode management - LONGER episodes for larger batches
        self.max_episode_steps = max_episode_steps
        self.episode_steps = 0
        self.reset_round = reset_round

        # MINIMAL action filtering - prevent excessive jumping
        self.jump_cooldown = 0
        self.max_jump_cooldown = 180  # 3 seconds at 60 FPS (3 * 60 = 180 frames)
        self.jump_actions = [6, 7, 8]  # up, up-left, up-right

        # Environment tracking
        import random

        self.env_id = f"ENV-{random.randint(1000, 9999)}"
        self.wins = 0
        self.losses = 0
        self.total_rounds = 0

        # Initialize global tracking for this environment
        SamuraiShowdownCustomWrapper._global_stats["env_stats"][self.env_id] = {
            "wins": 0,
            "losses": 0,
            "total_rounds": 0,
        }

        # Logging configuration
        self.log_interval = 120  # Log every 2 minutes

        # Get frame dimensions and calculate target size
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

        print(f"⚡ {self.env_id} SIMPLE Wrapper - Win/Loss Only")
        print(f"   🎯 Rewards: +1 Win, -1 Loss, 0 everything else")
        print(f"   📏 Episode length: {max_episode_steps} steps (for large batches)")
        print(f"   🚫 Jump prevention: 3 seconds cooldown (180 frames)")
        print(f"   🚀 Optimized for fewer envs, larger batch sizes")

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

    def _extract_health(self, info):
        """Extract health from info"""
        player_health = info.get("health", self.full_hp)
        opponent_health = info.get("enemy_health", self.full_hp)
        return player_health, opponent_health

    def _log_periodic_stats(self):
        """Log simple win/loss stats"""
        current_time = time.time()
        time_since_last_log = (
            current_time - SamuraiShowdownCustomWrapper._global_stats["last_log_time"]
        )

        if time_since_last_log >= self.log_interval:
            global_stats = SamuraiShowdownCustomWrapper._global_stats

            print(f"\n📊 SIMPLE WIN/LOSS STATS:")

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
                        f"   {emoji} {env_id}: {stats['wins']}W/{stats['losses']}L ({win_rate:.1f}%)"
                    )
                else:
                    print(f"   ⏳ {env_id}: 0W/0L (0.0%)")

            print()
            global_stats["last_log_time"] = current_time

    def _calculate_reward(self, curr_player_health, curr_opponent_health):
        """SUPER SIMPLE - Only +1 win, -1 loss, 0 everything else"""
        reward = 0.0
        done = False

        # Check for round end
        if curr_player_health <= 0 or curr_opponent_health <= 0:
            self.total_rounds += 1
            SamuraiShowdownCustomWrapper._global_stats["total_rounds"] += 1

            if curr_opponent_health <= 0 and curr_player_health > 0:
                # WIN = +1
                self.wins += 1
                win_rate = self.wins / self.total_rounds
                SamuraiShowdownCustomWrapper._global_stats["total_wins"] += 1
                SamuraiShowdownCustomWrapper._global_stats["env_stats"][self.env_id][
                    "wins"
                ] = self.wins
                SamuraiShowdownCustomWrapper._global_stats["env_stats"][self.env_id][
                    "total_rounds"
                ] = self.total_rounds

                print(
                    f"🏆 {self.env_id} WIN! {self.wins}W/{self.losses}L ({win_rate:.1%})"
                )

                reward = 1.0  # Simple +1 for win

                self.prev_player_health = curr_player_health
                self.prev_opponent_health = curr_opponent_health
                return reward, True

            elif curr_player_health <= 0 and curr_opponent_health > 0:
                # LOSS = -1
                self.losses += 1
                win_rate = self.wins / self.total_rounds
                SamuraiShowdownCustomWrapper._global_stats["total_losses"] += 1
                SamuraiShowdownCustomWrapper._global_stats["env_stats"][self.env_id][
                    "losses"
                ] = self.losses
                SamuraiShowdownCustomWrapper._global_stats["env_stats"][self.env_id][
                    "total_rounds"
                ] = self.total_rounds

                print(
                    f"💀 {self.env_id} LOSS! {self.wins}W/{self.losses}L ({win_rate:.1%})"
                )

                reward = -1.0  # Simple -1 for loss

                self.prev_player_health = curr_player_health
                self.prev_opponent_health = curr_opponent_health
                return reward, True

            if self.reset_round:
                done = True

            self._log_periodic_stats()

        self.prev_player_health = curr_player_health
        self.prev_opponent_health = curr_opponent_health

        # Everything else = 0 reward
        return 0.0, done

    def reset(self, **kwargs):
        """Reset environment"""
        result = self.env.reset(**kwargs)
        if isinstance(result, tuple):
            observation, info = result
        else:
            observation = result
            info = {}

        self.prev_player_health = self.full_hp
        self.prev_opponent_health = self.full_hp
        self.episode_steps = 0
        self.jump_cooldown = 0

        self.frame_stack.clear()
        processed_frame = self._process_frame(observation)
        self.frame_stack.append(processed_frame.copy())

        for _ in range(self.num_frames - 1):
            zero_frame = np.zeros_like(processed_frame)
            self.frame_stack.append(zero_frame)

        stacked_obs = self._stack_observation()
        return stacked_obs, info

    def step(self, action):
        """MINIMAL action filtering - mostly let agent do what it wants"""
        # Convert action to int
        try:
            if hasattr(action, "shape") and action.shape == ():
                action_int = int(action)
            elif hasattr(action, "__len__") and len(action) == 1:
                action_int = int(action[0])
            elif hasattr(action, "item"):
                action_int = action.item()
            else:
                action_int = int(action)
        except (ValueError, IndexError):
            action_int = 0

        # ANTI-SPAM jump prevention - 3 seconds cooldown
        if self.jump_cooldown > 0:
            self.jump_cooldown -= 1

        # Prevent excessive jumping with 3-second cooldown
        if action_int in self.jump_actions and self.jump_cooldown > 0:
            action = 0  # Convert to neutral - no jumping allowed
        elif action_int in self.jump_actions:
            self.jump_cooldown = self.max_jump_cooldown  # Start 3-second cooldown

        observation, reward, done, truncated, info = self.env.step(action)

        curr_player_health, curr_opponent_health = self._extract_health(info)
        custom_reward, custom_done = self._calculate_reward(
            curr_player_health, curr_opponent_health
        )

        if custom_done:
            done = custom_done

        processed_frame = self._process_frame(observation)
        self.frame_stack.append(processed_frame)
        stacked_obs = self._stack_observation()

        self.episode_steps += 1

        if self.episode_steps >= self.max_episode_steps:
            truncated = True

        return stacked_obs, custom_reward, done, truncated, info

    @classmethod
    def print_final_stats(cls):
        """Print final statistics when training ends"""
        pass
