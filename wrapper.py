import gymnasium as gym
from gymnasium import spaces
from collections import deque
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from typing import Dict, Tuple, Optional
import math


class SamuraiShowdownCustomWrapper(gym.Wrapper):
    """
    PRIME-optimized wrapper for Samurai Showdown fighting game with proper reward modeling
    UPDATED: 4 frames for memory efficiency with large batch training

    Features:
    - Frame stacking (4 frames) with RGB channels (12 total channels)
    - Dense process rewards compatible with PRIME methodology
    - Proper action tracking for credit assignment
    - Performance tracking with detailed statistics
    - Scaled reward system for gradient stability
    - Optimized for large batch sizes (2048+) and long trajectories (3000+)
    """

    def __init__(
        self,
        env,
        reset_round=True,
        rendering=False,
        max_episode_steps=15000,
        frame_stack=4,  # 4 frames for memory efficiency
        frame_skip=4,
        target_size=(180, 126),
    ):
        super().__init__(env)

        self.reset_round = reset_round
        self.rendering = rendering
        self.max_episode_steps = max_episode_steps
        self.frame_stack = frame_stack  # 4 frames
        self.frame_skip = frame_skip
        self.target_size = target_size

        # Health constants
        self.full_hp = 128

        # PRIME-specific tracking
        self.action_history = deque(maxlen=100)  # Track action sequences
        self.reward_history = deque(maxlen=1000)  # For reward normalization
        self.process_rewards = []  # Store step-by-step rewards
        self.outcome_reward = 0.0  # Final outcome reward

        # Performance tracking with enhanced metrics
        self.current_stats = {
            "wins": 0,
            "losses": 0,
            "total_rounds": 0,
            "win_rate": 0.0,
            "current_episode_reward": 0.0,
            "best_win_streak": 0,
            "current_win_streak": 0,
            "avg_episode_length": 0.0,
            "total_damage_dealt": 0,
            "total_damage_taken": 0,
            "damage_efficiency": 0.0,
        }

        # Episode tracking
        self.step_count = 0
        self.episode_count = 0
        self.episode_start_time = 0

        # Game state tracking with enhanced granularity
        self.prev_player_health = None
        self.prev_enemy_health = None
        self.prev_action = None
        self.consecutive_same_action = 0

        # Combat tracking
        self.combo_length = 0
        self.last_damage_step = 0
        self.damage_sequence = []

        # Reward normalization parameters
        self.reward_scale = 10.0  # Scale factor for gradients
        self.outcome_weight = 0.3  # 30% outcome, 70% process (PRIME recommendation)
        self.process_weight = 0.7

        # Frame buffer for stacking - 4 FRAMES
        self.frame_buffer = deque(maxlen=self.frame_stack)

        # Initialize frame buffer with zeros
        dummy_frame = np.zeros(
            (self.target_size[1], self.target_size[0], 3), dtype=np.uint8
        )
        for _ in range(self.frame_stack):
            self.frame_buffer.append(dummy_frame)

        # Update observation space for stacked RGB frames - 12 CHANNELS
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(
                self.frame_stack * 3,  # 4 * 3 = 12 channels
                self.target_size[0],
                self.target_size[1],
            ),
            dtype=np.uint8,
        )

        print(f"🎮 PRIME-OPTIMIZED SAMURAI SHOWDOWN WRAPPER (SIMPLE CNN):")
        print(f"   📊 Observation space: {self.observation_space.shape}")
        print(
            f"   🎨 Frame stacking: {self.frame_stack} frames × 3 RGB = {self.frame_stack * 3} channels"
        )
        print(f"   🎯 Target size: {self.target_size}")
        print(f"   ⏱️ Max episode steps: {self.max_episode_steps}")
        print(f"   🔄 Frame skip: {self.frame_skip}")
        print(f"   🧠 PRIME integration: Dense process rewards + outcome rewards")
        print(
            f"   ⚖️ Reward weighting: {self.process_weight:.1%} process + {self.outcome_weight:.1%} outcome"
        )
        print(f"   💾 Memory optimized for LARGE batch training (2048+)")
        print(f"   🚀 Optimized for Simple CNN + PRIME methodology")

    def _preprocess_frame(self, frame):
        """Preprocess frame: resize and maintain RGB channels"""
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            frame = cv2.resize(frame, self.target_size, interpolation=cv2.INTER_AREA)
        else:
            if len(frame.shape) == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            frame = cv2.resize(frame, self.target_size, interpolation=cv2.INTER_AREA)
        return frame.astype(np.uint8)

    def _get_stacked_observation(self):
        """Create stacked observation from frame buffer - 12 CHANNELS"""
        stacked = np.concatenate(list(self.frame_buffer), axis=2)  # (H, W, 12)
        stacked = np.transpose(stacked, (2, 0, 1))  # (12, H, W)
        stacked = np.transpose(stacked, (0, 2, 1))  # (12, W, H)
        return stacked

    def _extract_game_state(self, info):
        """Extract enhanced game state information for fighting games"""
        # Basic health information from available RAM addresses
        player_health = info.get("health", self.full_hp)
        opponent_health = info.get("enemy_health", self.full_hp)
        current_round = info.get("round", 1)
        score = info.get("score", 0)

        return {
            "player_health": player_health,
            "opponent_health": opponent_health,
            "current_round": current_round,
            "score": score,
        }

    # process reward. self, game info, action, info
    def _calculate_process_reward(self, game_info, action, info):
        """
        PRIME-style dense process reward calculation
        OPTIMIZED: For large batch training with simple CNN
        Returns step-by-step dense rewards for better credit assignment
        """

        # reward
        reward = 0.0

        # player health, opponent health
        player_health = game_info["player_health"]
        enemy_health = game_info["opponent_health"]

        # prev player health, prev opponent health
        if self.prev_player_health is None:
            self.prev_player_health = player_health
            self.prev_enemy_health = enemy_health
            self.prev_action = action
            return 0.0

        # 1. SCALED Health difference rewards (primary signal)

        # agent health diff. enemy health diff
        health_diff = self.prev_player_health - player_health
        enemy_health_diff = self.prev_enemy_health - enemy_health

        # Scale by reward_scale for better gradients

        # if enemy health got damaged.
        if enemy_health_diff > 0:
            normalized_damage = enemy_health_diff / self.full_hp
            damage_reward = (
                normalized_damage * 3.5 * self.reward_scale
            )  # Slightly increased for simple CNN
            reward += damage_reward

            # Combo tracking - optimized for 4-frame window
            if self.step_count - self.last_damage_step <= 4:
                self.combo_length += 1
                combo_bonus = (
                    min(self.combo_length * 0.15, 1.5) * self.reward_scale
                )  # Enhanced for simple CNN
                reward += combo_bonus
            else:
                self.combo_length = 1
            self.last_damage_step = self.step_count

        if health_diff > 0:
            normalized_damage_taken = health_diff / self.full_hp
            damage_penalty = normalized_damage_taken * 1.5 * self.reward_scale
            reward -= damage_penalty
            self.combo_length = 0  # Reset combo on taking damage

        # 2. Health advantage (scaled)
        health_advantage = (player_health - enemy_health) / self.full_hp
        advantage_reward = (
            health_advantage * 0.6 * self.reward_scale
        )  # Slightly increased
        reward += advantage_reward

        # 3. Action diversity reward (prevent button mashing) - enhanced for large batch
        if isinstance(action, np.ndarray):
            if action.ndim == 1:
                current_action = tuple(action)
            else:
                current_action = tuple(action.flatten())
        else:
            current_action = action

        prev_action = self.prev_action

        if prev_action is not None:
            try:
                if current_action == prev_action:
                    self.consecutive_same_action += 1
                    if self.consecutive_same_action > 3:
                        diversity_penalty = (
                            -0.12 * self.reward_scale
                        )  # Slightly increased penalty
                        reward += diversity_penalty
                else:
                    self.consecutive_same_action = 0
                    diversity_bonus = (
                        0.06 * self.reward_scale
                    )  # Slightly increased bonus
                    reward += diversity_bonus
            except (ValueError, TypeError):
                self.consecutive_same_action = 0
                diversity_bonus = 0.06 * self.reward_scale
                reward += diversity_bonus

        # 4. Combat engagement reward - enhanced for simple CNN
        if enemy_health_diff > 0 or health_diff > 0:
            engagement_bonus = 0.15 * self.reward_scale  # Increased for simple CNN
            reward += engagement_bonus

        # 5. Health preservation bonus
        player_health_ratio = player_health / self.full_hp
        if player_health_ratio > 0.8:
            preservation_bonus = 0.12 * self.reward_scale  # Slightly increased
            reward += preservation_bonus
        elif player_health_ratio < 0.2:
            low_health_penalty = -0.25 * self.reward_scale  # Slightly increased penalty
            reward += low_health_penalty

        # 6. Small time penalty (scaled)
        time_penalty = (
            -0.008 * self.reward_scale
        )  # Slightly reduced for longer trajectories
        reward += time_penalty

        # Update tracking
        self.prev_player_health = player_health
        self.prev_enemy_health = enemy_health
        self.prev_action = current_action

        return reward

    def _calculate_outcome_reward(self, game_info, info):
        """
        Calculate sparse outcome reward for round completion
        OPTIMIZED: For large batch training
        """
        player_health = game_info["player_health"]
        enemy_health = game_info["opponent_health"]

        # Check for round completion
        round_over = player_health <= 0 or enemy_health <= 0

        if round_over:
            if player_health > enemy_health:
                # Win with health-based scaling
                health_bonus = (
                    player_health / self.full_hp
                ) * 6.0  # Slightly increased
                win_reward = (
                    18.0 + health_bonus
                ) * self.reward_scale  # Increased for simple CNN

                # Update stats
                self.current_stats["wins"] += 1
                self.current_stats["current_win_streak"] += 1
                self.current_stats["best_win_streak"] = max(
                    self.current_stats["best_win_streak"],
                    self.current_stats["current_win_streak"],
                )
                return win_reward

            elif enemy_health > player_health:
                # Loss penalty with scaling
                health_disadvantage = (enemy_health - player_health) / self.full_hp
                loss_penalty = (
                    -(10.0 + health_disadvantage * 3.5) * self.reward_scale
                )  # Slightly increased

                # Update stats
                self.current_stats["losses"] += 1
                self.current_stats["current_win_streak"] = 0
                return loss_penalty
            else:
                # Draw
                return 0.0

        return 0.0

    def _update_statistics(self, process_reward, outcome_reward):
        """Update detailed statistics for monitoring"""
        total_reward = process_reward + outcome_reward
        self.current_stats["current_episode_reward"] += total_reward

        # Track reward history for normalization
        self.reward_history.append(total_reward)

        # Update round statistics
        if outcome_reward != 0:  # Round ended
            self.current_stats["total_rounds"] += 1
            if self.current_stats["total_rounds"] > 0:
                self.current_stats["win_rate"] = (
                    self.current_stats["wins"] / self.current_stats["total_rounds"]
                )

        # Update episode length tracking
        total_episodes = self.episode_count if self.episode_count > 0 else 1
        self.current_stats["avg_episode_length"] = (
            self.current_stats["avg_episode_length"] * (total_episodes - 1)
            + self.step_count
        ) / total_episodes

    def reset(self, **kwargs):
        """Reset environment and tracking variables"""
        obs, info = self.env.reset(**kwargs)

        # Reset episode tracking
        self.step_count = 0
        self.episode_count += 1

        # Reset game state tracking
        self.prev_player_health = None
        self.prev_enemy_health = None
        self.prev_action = None
        self.consecutive_same_action = 0

        # Reset combat tracking
        self.combo_length = 0
        self.last_damage_step = 0
        self.damage_sequence.clear()

        # Reset PRIME tracking
        self.process_rewards.clear()
        self.outcome_reward = 0.0
        self.action_history.clear()
        self.current_stats["current_episode_reward"] = 0.0

        # Reset frame buffer - 4 FRAMES
        processed_frame = self._preprocess_frame(obs)
        self.frame_buffer.clear()
        for _ in range(self.frame_stack):
            self.frame_buffer.append(processed_frame)

        stacked_obs = self._get_stacked_observation()
        return stacked_obs, info

    def step(self, action):
        """Execute action with PRIME-optimized reward calculation"""
        total_reward = 0.0
        info = {}

        # Execute action with frame skipping
        for _ in range(self.frame_skip):
            obs, reward, terminated, truncated, step_info = self.env.step(action)
            total_reward += reward
            info.update(step_info)
            if terminated or truncated:
                break

        # Process frame and add to buffer
        processed_frame = self._preprocess_frame(obs)
        self.frame_buffer.append(processed_frame)

        # Track action (gymnasium + stable-retro format)
        if isinstance(action, np.ndarray):
            action_scalar = tuple(action.flatten())
        else:
            action_scalar = action

        self.action_history.append(action_scalar)

        # Get game information
        game_info = self._extract_game_state(info)

        # ok we have process reward here
        process_reward = self._calculate_process_reward(game_info, action, info)
        outcome_reward = self._calculate_outcome_reward(game_info, info)

        # Store for PRIME algorithm
        self.process_rewards.append(process_reward)
        if outcome_reward != 0:
            self.outcome_reward = outcome_reward

        # Combine rewards with PRIME weighting
        combined_reward = (
            self.process_weight * process_reward + self.outcome_weight * outcome_reward
        )

        # Update statistics
        self._update_statistics(process_reward, outcome_reward)

        # Episode termination logic
        done = (
            self.step_count >= self.max_episode_steps
            or terminated
            or truncated
            or (game_info["player_health"] <= 0 and game_info["opponent_health"] <= 0)
        )

        # Get stacked observation
        stacked_obs = self._get_stacked_observation()
        self.step_count += 1

        # Enhanced info for PRIME algorithm
        info.update(
            {
                "game_info": game_info,
                "process_reward": process_reward,
                "outcome_reward": outcome_reward,
                "combined_reward": combined_reward,
                "episode_steps": self.step_count,
                "episode_reward": self.current_stats["current_episode_reward"],
                "action_history": list(self.action_history),
                "combo_length": self.combo_length,
                "win_rate": self.current_stats["win_rate"],
            }
        )

        return stacked_obs, combined_reward, done, False, info

    def render(self, mode="human"):
        """Render environment if rendering is enabled"""
        if self.rendering:
            return self.env.render()
        return None

    def close(self):
        """Close environment"""
        return self.env.close()


# Simple CNN Feature Extractor for PRIME
class SimplePRIMECNN(BaseFeaturesExtractor):
    """
    Simple CNN optimized for PRIME + Large Batch + Long Trajectories
    Memory efficient and fast - perfect for fighting games
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 512):
        super().__init__(observation_space, features_dim)

        n_input_channels = observation_space.shape[0]  # 12 channels for 4-frame

        print(f"🚀 Creating Simple PRIME CNN for LARGE BATCH training...")

        # Simple but effective CNN architecture
        self.cnn = nn.Sequential(
            # First block - capture basic features
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=2),
            nn.ReLU(inplace=True),
            # Second block - spatial patterns
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            # Third block - higher level features
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            # Fourth block - fighting game specific
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            # Global pooling
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )

        # Calculate CNN output size
        with torch.no_grad():
            sample = torch.zeros(1, n_input_channels, 126, 180)
            cnn_output_size = self.cnn(sample).shape[1]

        # Simple classifier
        self.classifier = nn.Sequential(
            nn.Linear(cnn_output_size, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(512, features_dim),
            nn.ReLU(inplace=True),
        )

        print(f"🧠 SIMPLE PRIME CNN:")
        print(f"   📊 Input: {observation_space.shape}")
        print(f"   🎨 Input channels: {n_input_channels} (4-frame)")
        print(f"   🏗️ CNN output: {cnn_output_size}")
        print(f"   🎯 Final features: {features_dim}")
        print(f"   💾 Memory efficient for LARGE batches")
        print(f"   🚀 Perfect for batch_size=2048+ and n_steps=3000+")

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # Simple normalization - much faster
        x = observations.float() / 255.0

        # CNN feature extraction
        features = self.cnn(x)

        # Final classification
        output = self.classifier(features)

        return output


# Utility function to create the wrapped environment
def make_samurai_env(
    game="SamuraiShodown-Genesis",
    state=None,
    reset_round=True,
    rendering=False,
    max_episode_steps=15000,
    **wrapper_kwargs,
):
    """
    Utility function to create a PRIME-optimized Samurai Showdown environment
    UPDATED: Uses 4 frames and optimized for large batch training
    """
    import retro

    env = retro.make(
        game=game,
        state=state,
        use_restricted_actions=retro.Actions.FILTERED,
        obs_type=retro.Observations.IMAGE,
        render_mode="human" if rendering else None,
    )

    # Force 4-frame setup for memory efficiency
    wrapper_kwargs.setdefault("frame_stack", 4)

    env = SamuraiShowdownCustomWrapper(
        env,
        reset_round=reset_round,
        rendering=rendering,
        max_episode_steps=max_episode_steps,
        **wrapper_kwargs,
    )

    return env


if __name__ == "__main__":
    # Test the PRIME-optimized wrapper with Simple CNN
    print("🧪 Testing PRIME-optimized SamuraiShowdownCustomWrapper (Simple CNN)...")

    try:
        env = make_samurai_env(rendering=False)

        print(f"✅ Environment created successfully")
        print(f"📊 Observation space: {env.observation_space}")
        print(f"🎮 Action space: {env.action_space}")

        # Test reset
        obs, info = env.reset()
        print(f"📸 Reset observation shape: {obs.shape}")

        # Test steps with PRIME features
        for i in range(5):
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)
            print(
                f"Step {i+1}: obs_shape={obs.shape}, reward={reward:.3f}, process={info.get('process_reward', 0):.3f}"
            )

            if done:
                break

        # Print enhanced stats
        print(f"📊 PRIME stats: {env.current_stats}")

        env.close()
        print("✅ PRIME-optimized wrapper test completed successfully (Simple CNN)!")

    except Exception as e:
        print(f"❌ Wrapper test failed: {e}")
        import traceback

        traceback.print_exc()
