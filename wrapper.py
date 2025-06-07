import collections
import gymnasium as gym
import numpy as np
import time
from datetime import datetime
import math


# wrapper
class SamuraiShowdownCustomWrapper(gym.Wrapper):
    """Simple wrapper - Win/Loss rewards only, optimized for large batch sizes"""

    # total win, lose, round, stat, log, session start
    # Global tracking across all environments
    _global_stats = {
        "total_wins": 0,
        "total_losses": 0,
        "total_rounds": 0,
        "env_stats": {},
        "last_log_time": time.time(),
        "session_start": time.time(),
    }

    # init with super wrapper
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

        # NEW: Reward parameters
        self.reward_coeff = reward_coeff  # Reward coefficient for wins/damage
        self.total_timesteps = 0  # Track total timesteps

        # MISSING: Rendering parameter
        self.rendering = rendering  # Store rendering flag

        # Episode management
        self.max_episode_steps = max_episode_steps
        self.episode_steps = 0
        self.reset_round = reset_round

        # Health tracking
        self.full_hp = 128
        self.prev_player_health = self.full_hp
        self.prev_opponent_health = self.full_hp  # Fixed typo: oppont -> opponent

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

        # actual obs
        dummy_obs = self.env.reset()
        if isinstance(dummy_obs, tuple):
            actual_obs = dummy_obs[0]
        else:
            actual_obs = dummy_obs

        # 75% resize
        original_height, original_width = actual_obs.shape[:2]
        self.target_height = int(original_height * self.resize_scale)
        self.target_width = int(original_width * self.resize_scale)

        # obs
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(self.num_frames, self.target_height, self.target_width),
            dtype=np.uint8,
        )

        print(f"⚡ {self.env_id} SIMPLE Wrapper - Win/Loss Only")
        print(f"   🎯 Rewards: +1 Win, -1 Loss, 0 everything else")
        print(f"   📏 Episode length: {max_episode_steps} steps (for large batches)")
        print(f"   🚀 No action filtering - agent has full control")
        print(f"   🚀 Optimized for fewer envs, larger batch sizes")

    def _process_frame(self, rgb_frame):
        # conert to gray scale
        """Convert RGB frame to grayscale and resize"""
        if len(rgb_frame.shape) == 3 and rgb_frame.shape[2] == 3:
            gray = np.dot(rgb_frame, [0.299, 0.587, 0.114])
            gray = gray.astype(np.uint8)
        else:
            # default to rgb frame
            gray = rgb_frame

        # resize
        if gray.shape[:2] != (self.target_height, self.target_width):
            h_ratio = gray.shape[0] / self.target_height
            w_ratio = gray.shape[1] / self.target_width
            h_indices = (np.arange(self.target_height) * h_ratio).astype(int)
            w_indices = (np.arange(self.target_width) * w_ratio).astype(int)
            # gray then resize
            resized = gray[np.ix_(h_indices, w_indices)]
            return resized
        else:
            return gray

    # stack obs
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

    # get health
    def _extract_health(self, info):
        """Extract health from info"""
        player_health = info.get("health", self.full_hp)
        opponent_health = info.get("enemy_health", self.full_hp)
        return player_health, opponent_health

    # just log
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

    # super simple reward
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

    # reset
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

        self.frame_stack.clear()
        processed_frame = self._process_frame(observation)
        self.frame_stack.append(processed_frame.copy())

        for _ in range(self.num_frames - 1):
            zero_frame = np.zeros_like(processed_frame)
            self.frame_stack.append(zero_frame)

        stacked_obs = self._stack_observation()
        return stacked_obs, info

    def step(self, action):
        """Custom step function with detailed reward system"""
        custom_done = False

        # Single step with action (pass action directly, let retro handle conversion)
        observation, reward, done, truncated, info = self.env.step(action)

        # Process and add frame (no downsampling)
        processed_frame = self._process_frame(observation)
        self.frame_stack.append(processed_frame)

        # Render if needed
        if self.rendering:
            self.env.render()
            time.sleep(0.01)

        # Extract health from info
        curr_player_health = info.get("health", self.full_hp)
        curr_opponent_health = info.get("enemy_health", self.full_hp)

        # Update total timesteps
        if not hasattr(self, "total_timesteps"):
            self.total_timesteps = 0
        self.total_timesteps += 1

        # Calculate custom reward based on game state
        if curr_player_health < 0:
            # Player loses - penalty based on opponent's remaining health
            custom_reward = -math.pow(
                self.full_hp, (curr_opponent_health + 1) / (self.full_hp + 1)
            )
            custom_done = True

            # Update stats
            self.losses += 1
            self.total_rounds += 1
            win_rate = self.wins / self.total_rounds if self.total_rounds > 0 else 0
            print(
                f"💀 {self.env_id} LOSS! {self.wins}W/{self.losses}L ({win_rate:.1%})"
            )

        elif curr_opponent_health < 0:
            # Player wins - reward based on player's remaining health
            reward_coeff = getattr(
                self, "reward_coeff", 3.0
            )  # Default reward coefficient
            custom_reward = (
                math.pow(self.full_hp, (curr_player_health + 1) / (self.full_hp + 1))
            ) * reward_coeff  # convert to gray scale
            win_rate = self.wins / self.total_rounds
            print(f"🏆 {self.env_id} WIN! {self.wins}W/{self.losses}L ({win_rate:.1%})")

        else:
            # Fighting continues - reward based on damage dealt vs received
            reward_coeff = getattr(self, "reward_coeff", 3.0)
            damage_dealt = self.prev_opponent_health - curr_opponent_health
            damage_received = self.prev_player_health - curr_player_health
            custom_reward = reward_coeff * damage_dealt - damage_received

            # Update health tracking
            self.prev_player_health = curr_player_health
            self.prev_opponent_health = curr_opponent_health
            custom_done = False

        # Override done flag if reset_round is False
        if not self.reset_round:
            custom_done = False

        # Check episode step limit
        self.episode_steps += 1
        if self.episode_steps >= self.max_episode_steps:
            truncated = True

        # Apply custom done flag
        if custom_done:
            done = custom_done

        # Periodic logging
        self._log_periodic_stats()

        # Return with reward normalization (max reward ~1054, normalized to ~1.0)
        stacked_obs = self._stack_observation()
        normalized_reward = 0.001 * custom_reward

        return stacked_obs, normalized_reward, done, truncated, info

    @classmethod
    def print_final_stats(cls):
        """Print final statistics when training ends"""
        pass
