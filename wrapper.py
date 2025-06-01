import collections
import gymnasium as gym
import numpy as np


class SamuraiShowdownCustomWrapper(gym.Wrapper):
    """Custom Samurai Showdown wrapper - outputs channels-first format (C, H, W)"""

    def __init__(self, env, reset_round=True, rendering=False, max_episode_steps=5000):
        super(SamuraiShowdownCustomWrapper, self).__init__(env)
        self.env = env

        # Frame processing parameters
        self.resize_scale = 0.75  # Resize to 75% of original size
        self.num_frames = 9  # Frame stacking
        self.frame_stack = collections.deque(maxlen=self.num_frames)

        # Health tracking - ADJUSTED FOR SAMURAI SHOWDOWN
        # Based on found data.json: health values are 1-byte unsigned
        self.full_hp = 128  # Max health from memory address data
        self.prev_player_health = self.full_hp
        self.prev_opponent_health = self.full_hp

        # Episode management
        self.max_episode_steps = max_episode_steps
        self.episode_steps = 0
        self.reset_round = reset_round

        # Win tracking for each environment with unique ID
        import time
        import random

        self.env_id = f"ENV-{random.randint(1000, 9999)}"  # Unique environment ID
        self.wins = 0
        self.losses = 0
        self.total_rounds = 0
        self.rounds_this_session = 0

        # Get actual frame dimensions and calculate resized dimensions
        dummy_obs = self.env.reset()
        if isinstance(dummy_obs, tuple):
            actual_obs = dummy_obs[0]
        else:
            actual_obs = dummy_obs

        original_height, original_width = actual_obs.shape[:2]

        # Calculate new dimensions (75% of original)
        self.target_height = int(original_height * self.resize_scale)
        self.target_width = int(original_width * self.resize_scale)

        # Use channels-first format (C, H, W) to match model expectations
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(self.num_frames, self.target_height, self.target_width),  # C, H, W
            dtype=np.uint8,
        )

        print(f"🚀 Samurai Showdown Wrapper initialized (CHANNELS-FIRST FORMAT):")
        print(f"   Original frame shape: {actual_obs.shape}")
        print(
            f"   Resized to: ({self.target_height}, {self.target_width}) - {int(self.resize_scale*100)}% of original"
        )
        print(f"   Final observation shape: {self.observation_space.shape} (C, H, W)")
        print(f"   Max episode steps: {self.max_episode_steps}")
        print(f"   Frame stacking: {self.num_frames} grayscale frames")

    def _process_frame(self, rgb_frame):
        """Convert RGB frame to grayscale and resize"""
        # Convert to grayscale first
        if len(rgb_frame.shape) == 3 and rgb_frame.shape[2] == 3:
            # Standard grayscale conversion: 0.299*R + 0.587*G + 0.114*B
            gray = np.dot(rgb_frame, [0.299, 0.587, 0.114])
            gray = gray.astype(np.uint8)
        else:
            # Already grayscale or different format
            gray = rgb_frame

        # Resize using simple numpy interpolation (nearest neighbor for speed)
        if gray.shape[:2] != (self.target_height, self.target_width):
            # Simple resizing using array indexing (fast but basic)
            h_ratio = gray.shape[0] / self.target_height
            w_ratio = gray.shape[1] / self.target_width

            # Create index arrays for resizing
            h_indices = (np.arange(self.target_height) * h_ratio).astype(int)
            w_indices = (np.arange(self.target_width) * w_ratio).astype(int)

            # Resize by indexing
            resized = gray[np.ix_(h_indices, w_indices)]
            return resized
        else:
            return gray

    def _stack_observation(self):
        """Stack frames in channels-first format (C, H, W)"""
        # Fill stack if not enough frames
        while len(self.frame_stack) < self.num_frames:
            if len(self.frame_stack) > 0:
                self.frame_stack.append(self.frame_stack[-1].copy())
            else:
                # Create dummy frame with target dimensions
                dummy_frame = np.zeros(
                    (self.target_height, self.target_width), dtype=np.uint8
                )
                self.frame_stack.append(dummy_frame)

        # Stack frames along first dimension (channels-first)
        stacked = np.stack(list(self.frame_stack), axis=0)  # Shape: (C, H, W)
        return stacked

    def _extract_health(self, info):
        """Extract health from info - ONLY health and enemy_health"""

        # Only use the confirmed memory addresses
        player_health = info.get("health", self.full_hp)
        opponent_health = info.get("enemy_health", self.full_hp)

        return player_health, opponent_health

    def _calculate_reward(self, curr_player_health, curr_opponent_health):
        """Calculate reward based on damage dealt/received - SIMPLIFIED"""
        reward = 0.0
        done = False

        # Check for round end
        if curr_player_health <= 0 or curr_opponent_health <= 0:
            self.total_rounds += 1
            self.rounds_this_session += 1

            if curr_opponent_health <= 0 and curr_player_health > 0:
                # Win bonus
                self.wins += 1
                win_rate = self.wins / self.total_rounds
                print(
                    f"🏆 {self.env_id} WIN! W:{self.wins} L:{self.losses} Rate:{win_rate:.1%} [Round {self.rounds_this_session}]"
                )
                reward += 100  # Large win bonus
            elif curr_player_health <= 0 and curr_opponent_health > 0:
                # Loss penalty
                self.losses += 1
                win_rate = self.wins / self.total_rounds
                print(
                    f"💀 {self.env_id} LOSS! W:{self.wins} L:{self.losses} Rate:{win_rate:.1%} [Round {self.rounds_this_session}]"
                )
                reward -= 100  # Large loss penalty

            if self.reset_round:
                done = True

        # Damage-based reward: reward for dealing damage, penalty for taking damage
        damage_dealt = max(0, self.prev_opponent_health - curr_opponent_health)
        damage_received = max(0, self.prev_player_health - curr_player_health)

        # Reward system focused only on health changes
        reward += damage_dealt * 2  # +2 per damage dealt
        reward -= damage_received * 1  # -1 per damage received

        # Small time penalty to encourage active fighting
        reward -= 0.1

        # Update health tracking
        self.prev_player_health = curr_player_health
        self.prev_opponent_health = curr_opponent_health

        return reward, done

    def reset(self, **kwargs):
        """Reset environment"""
        result = self.env.reset(**kwargs)
        if isinstance(result, tuple):
            observation, info = result
        else:
            observation = result
            info = {}

        # Reset tracking
        self.prev_player_health = self.full_hp
        self.prev_opponent_health = self.full_hp
        self.episode_steps = 0

        # Initialize frame stack with processed frames
        self.frame_stack.clear()
        processed_frame = self._process_frame(observation)

        for _ in range(self.num_frames):
            self.frame_stack.append(processed_frame.copy())

        stacked_obs = self._stack_observation()
        return stacked_obs, info

    def step(self, action):
        """Step function"""
        # Convert action if needed
        if isinstance(action, np.ndarray) and action.ndim == 0:
            # Convert scalar to binary action
            binary_action = np.zeros(self.env.action_space.n, dtype=int)
            if 0 <= action < self.env.action_space.n:
                binary_action[action] = 1
            action = binary_action

        # Execute step
        observation, reward, done, truncated, info = self.env.step(action)

        # Extract health and calculate custom reward
        curr_player_health, curr_opponent_health = self._extract_health(info)
        custom_reward, custom_done = self._calculate_reward(
            curr_player_health, curr_opponent_health
        )

        if custom_done:
            done = custom_done

        # Update frame stack with processed frame
        processed_frame = self._process_frame(observation)
        self.frame_stack.append(processed_frame)
        stacked_obs = self._stack_observation()

        self.episode_steps += 1

        # End episode if too long
        if self.episode_steps >= self.max_episode_steps:
            truncated = True

        return stacked_obs, custom_reward, done, truncated, info
