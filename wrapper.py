import collections
import gymnasium as gym
import numpy as np
import time
from datetime import datetime
import math


class SamuraiShowdownCustomWrapper(gym.Wrapper):
    """Samurai Showdown wrapper with normalized rewards for stable learning"""

    # Global tracking across all environments
    _global_stats = {
        "total_wins": 0,
        "total_losses": 0,
        "total_rounds": 0,
        "env_stats": {},
        "last_log_time": time.time(),
        "session_start": time.time(),
    }

    def __init__(self, env, reset_round=True, rendering=False, max_episode_steps=5000):
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

        # Episode management
        self.max_episode_steps = max_episode_steps
        self.episode_steps = 0
        self.reset_round = reset_round

        # NORMALIZED DEFENSE SYSTEM for stable learning + high win rate
        self.jump_cooldown = 0
        self.max_jump_cooldown = 10  # Very short cooldown
        self.jump_actions = [7, 8, 9]  # up, up-left, up-right

        # Comprehensive defensive actions
        self.defensive_actions = [4, 1, 3]  # back (4), down (1), down-back (3)
        self.safe_actions = [0, 1, 3, 4]  # neutral, down, down-back, back
        self.risky_actions = [2, 5, 6, 7, 8, 9]  # forward moves and jumps
        self.last_action = 0
        self.consecutive_blocks = 0
        self.max_consecutive_blocks = 8  # Allow much more defensive play

        # Enhanced tracking
        self.damage_taken_this_step = 0
        self.safe_action_streak = 0
        self.rounds_without_loss = 0

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
        self.log_interval = 300  # Log every 5 minutes instead of 1 minute

        # Get frame dimensions and calculate target size
        dummy_obs = self.env.reset()
        if isinstance(dummy_obs, tuple):
            actual_obs = dummy_obs[0]
        else:
            actual_obs = dummy_obs

        original_height, original_width = actual_obs.shape[:2]
        self.target_height = int(original_height * self.resize_scale)
        self.target_width = int(original_width * self.resize_scale)

        # FIXED: Proper observation space definition
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(self.num_frames, self.target_height, self.target_width),
            dtype=np.uint8,
        )

        print(f"🚀 {self.env_id} Samurai Showdown Wrapper initialized")
        print(f"   Original size: {original_height}x{original_width}")
        print(f"   Target size: {self.target_height}x{self.target_width}")
        print(f"   Observation shape: {self.observation_space.shape}")
        print(f"   Jump prevention: {self.max_jump_cooldown} frame cooldown")
        print(f"   Defense encouraged: block limit {self.max_consecutive_blocks}")
        print(f"   NORMALIZED REWARDS: Defense +0.3, Win +1.0-1.7, Loss -1.0")

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
        # FIXED: Proper frame stacking with padding
        frames_list = list(self.frame_stack)

        # If we don't have enough frames, pad with the first available frame
        while len(frames_list) < self.num_frames:
            if len(frames_list) > 0:
                frames_list.insert(0, frames_list[0].copy())  # Pad at beginning
            else:
                # Create zero frame if no frames available
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
        """Log all environment win rates periodically"""
        current_time = time.time()
        time_since_last_log = (
            current_time - SamuraiShowdownCustomWrapper._global_stats["last_log_time"]
        )

        if time_since_last_log >= self.log_interval:
            global_stats = SamuraiShowdownCustomWrapper._global_stats

            print(f"\n📊 ALL ENV WIN RATES:")

            # Show all environments
            for env_id, stats in global_stats["env_stats"].items():
                if stats["total_rounds"] > 0:
                    win_rate = stats["wins"] / stats["total_rounds"] * 100
                    if stats["wins"] > stats["losses"]:
                        emoji = "🏆"
                    elif stats["wins"] < stats["losses"]:
                        emoji = "💀"
                    else:
                        emoji = "⚖️"
                    print(
                        f"   {emoji} {env_id}: {stats['wins']}W/{stats['losses']}L ({win_rate:.1f}%)"
                    )
                else:
                    print(f"   ⏳ {env_id}: 0W/0L (0.0%)")

            print()  # Empty line after the list
            global_stats["last_log_time"] = current_time

    def _calculate_reward(self, curr_player_health, curr_opponent_health):
        """Calculate NORMALIZED reward with strong defensive bonuses for stable learning"""
        reward = 0.0
        done = False

        # Calculate damage taken this step
        self.damage_taken_this_step = self.prev_player_health - curr_player_health
        damage_dealt = self.prev_opponent_health - curr_opponent_health

        # NORMALIZED defensive rewards (0.0 to 1.0 range)
        if (
            self.damage_taken_this_step == 0
            and self.last_action in self.defensive_actions
        ):
            reward += 0.3  # Strong reward for successful defense

        # Reward for avoiding damage
        if self.damage_taken_this_step == 0:
            reward += 0.1
            self.safe_action_streak += 1

        # Streak bonus for consecutive safe play (capped)
        if self.safe_action_streak > 10:
            reward += min(0.05, self.safe_action_streak * 0.005)  # Small capped bonus

        # Reset streak if damage taken
        if self.damage_taken_this_step > 0:
            self.safe_action_streak = 0
            # Normalized damage penalty (0 to -0.5 max)
            damage_ratio = self.damage_taken_this_step / self.full_hp
            reward -= damage_ratio * 0.5

        # Small reward for dealing damage
        if damage_dealt > 0:
            damage_ratio = damage_dealt / self.full_hp
            reward += damage_ratio * 0.2

        # Health advantage bonus (small)
        health_advantage = curr_player_health - curr_opponent_health
        if health_advantage > 20:
            advantage_ratio = min(health_advantage / self.full_hp, 0.5)
            reward += advantage_ratio * 0.1

        # Check for round end
        if curr_player_health <= 0 or curr_opponent_health <= 0:
            self.total_rounds += 1
            SamuraiShowdownCustomWrapper._global_stats["total_rounds"] += 1

            if curr_opponent_health <= 0 and curr_player_health > 0:
                # WIN - normalized reward
                health_bonus = (curr_player_health / self.full_hp) * 0.5  # 0 to 0.5
                self.wins += 1
                self.rounds_without_loss += 1
                win_rate = self.wins / self.total_rounds

                # Win streak bonus (small and capped)
                if self.rounds_without_loss > 3:
                    streak_bonus = min(self.rounds_without_loss * 0.02, 0.2)  # Max 0.2
                else:
                    streak_bonus = 0

                SamuraiShowdownCustomWrapper._global_stats["total_wins"] += 1
                SamuraiShowdownCustomWrapper._global_stats["env_stats"][self.env_id][
                    "wins"
                ] = self.wins
                SamuraiShowdownCustomWrapper._global_stats["env_stats"][self.env_id][
                    "total_rounds"
                ] = self.total_rounds

                print(
                    f"🏆 {self.env_id} WIN! {self.wins}W/{self.losses}L ({win_rate:.1%}) HP:{curr_player_health} Streak:{self.rounds_without_loss}"
                )
                # Total win reward: 1.0 + health_bonus + streak_bonus (max ~1.7)
                reward += 1.0 + health_bonus + streak_bonus

                self.prev_player_health = curr_player_health
                self.prev_opponent_health = curr_opponent_health
                return reward, True

            elif curr_player_health <= 0 and curr_opponent_health > 0:
                # LOSS - normalized penalty
                self.losses += 1
                self.rounds_without_loss = 0  # Reset streak
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
                reward -= 1.0  # Fixed loss penalty

                self.prev_player_health = curr_player_health
                self.prev_opponent_health = curr_opponent_health
                return reward, True

            if self.reset_round:
                done = True

            # Check if it's time to log periodic stats
            self._log_periodic_stats()

        self.prev_player_health = curr_player_health
        self.prev_opponent_health = curr_opponent_health

        # Clip final reward to reasonable range
        reward = np.clip(reward, -1.5, 2.0)
        return reward, done

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

        # Reset defensive tracking
        self.jump_cooldown = 0
        self.last_action = 0
        self.consecutive_blocks = 0
        self.damage_taken_this_step = 0
        self.safe_action_streak = 0

        # FIXED: Proper frame stack initialization
        self.frame_stack.clear()
        processed_frame = self._process_frame(observation)

        # Start with the current frame
        self.frame_stack.append(processed_frame.copy())

        # Fill remaining slots with zero frames for proper motion detection
        for _ in range(self.num_frames - 1):
            zero_frame = np.zeros_like(processed_frame)
            self.frame_stack.append(zero_frame)

        stacked_obs = self._stack_observation()
        return stacked_obs, info

    def step(self, action):
        """Step function with NORMALIZED aggressive defense system"""
        # IMPROVED DEFENSE: Less restrictive action handling
        try:
            # Handle different action formats
            if hasattr(action, "shape") and action.shape == ():
                action_int = int(action)
            elif hasattr(action, "__len__") and len(action) == 1:
                action_int = int(action[0])
            elif hasattr(action, "item"):
                action_int = action.item()
            else:
                action_int = int(action)
        except (ValueError, IndexError):
            action_int = None

        # Apply AGGRESSIVE defense logic with normalized rewards
        if action_int is not None:
            # Decrease jump cooldown
            if self.jump_cooldown > 0:
                self.jump_cooldown -= 1

            # Track consecutive blocking
            if action_int in self.defensive_actions:
                if self.last_action in self.defensive_actions:
                    self.consecutive_blocks += 1
                else:
                    self.consecutive_blocks = 1
            else:
                self.consecutive_blocks = 0

            # AGGRESSIVE action filtering - heavily favor safe actions
            if action_int in self.risky_actions:
                # 80% chance to convert risky actions to safe ones
                if np.random.random() < 0.8:
                    if action_int in self.jump_actions:
                        action = 4  # Convert jumps to back
                    elif action_int in [2, 5, 6]:  # Forward actions
                        action = 4  # Convert forward to back

            # Prevent jumping during cooldown
            if action_int in self.jump_actions and self.jump_cooldown > 0:
                action = 4  # Always convert to back
            elif action_int in self.jump_actions:
                self.jump_cooldown = self.max_jump_cooldown

            # Allow long defensive sequences
            if (
                self.consecutive_blocks > self.max_consecutive_blocks
                and action_int in self.defensive_actions
            ):
                # Only occasionally break defensive play
                if np.random.random() < 0.1:  # 10% chance
                    action = 0  # Neutral action

            self.last_action = action_int

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
        # Remove final stats completely - just silent cleanup
        pass
