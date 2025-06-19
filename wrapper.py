import gymnasium as gym
from gymnasium import spaces
from collections import deque
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from typing import Dict, Tuple, Optional
import math


class SamuraiShowdownCustomWrapper(gym.Wrapper):
    """
    PRIME-optimized wrapper for Samurai Showdown fighting game with proper reward modeling

    Features:
    - Frame stacking (9 frames) with RGB channels (27 total channels)
    - Dense process rewards compatible with PRIME methodology
    - Proper action tracking for credit assignment
    - Performance tracking with detailed statistics
    - Scaled reward system for gradient stability
    """

    def __init__(
        self,
        env,
        reset_round=True,
        rendering=False,
        max_episode_steps=15000,
        frame_stack=9,
        frame_skip=4,
        target_size=(180, 126),
    ):
        super().__init__(env)

        self.reset_round = reset_round
        self.rendering = rendering
        self.max_episode_steps = max_episode_steps
        self.frame_stack = frame_stack
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

        # Frame buffer for stacking
        self.frame_buffer = deque(maxlen=self.frame_stack)

        # Initialize frame buffer with zeros
        dummy_frame = np.zeros(
            (self.target_size[1], self.target_size[0], 3), dtype=np.uint8
        )
        for _ in range(self.frame_stack):
            self.frame_buffer.append(dummy_frame)

        # Update observation space for stacked RGB frames
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(
                self.frame_stack * 3,
                self.target_size[0],
                self.target_size[1],
            ),
            dtype=np.uint8,
        )

        print(f"🎮 PRIME-OPTIMIZED SAMURAI SHOWDOWN WRAPPER INITIALIZED:")
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
        """Create stacked observation from frame buffer"""
        stacked = np.concatenate(list(self.frame_buffer), axis=2)  # (H, W, 27)
        stacked = np.transpose(stacked, (2, 0, 1))  # (27, H, W)
        stacked = np.transpose(stacked, (0, 2, 1))  # (27, W, H)
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

    def _calculate_process_reward(self, game_info, action, info):
        """
        PRIME-style dense process reward calculation
        Returns step-by-step dense rewards for better credit assignment
        """
        reward = 0.0

        player_health = game_info["player_health"]
        enemy_health = game_info["opponent_health"]

        # Initialize previous values if first step
        if self.prev_player_health is None:
            self.prev_player_health = player_health
            self.prev_enemy_health = enemy_health
            self.prev_action = action
            return 0.0

        # 1. SCALED Health difference rewards (primary signal)
        health_diff = self.prev_player_health - player_health
        enemy_health_diff = self.prev_enemy_health - enemy_health

        # Scale by reward_scale for better gradients
        if enemy_health_diff > 0:
            normalized_damage = enemy_health_diff / self.full_hp
            damage_reward = normalized_damage * 3.0 * self.reward_scale
            reward += damage_reward

            # Combo tracking
            if self.step_count - self.last_damage_step <= 5:  # Within combo window
                self.combo_length += 1
                combo_bonus = min(self.combo_length * 0.1, 1.0) * self.reward_scale
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
        advantage_reward = health_advantage * 0.5 * self.reward_scale
        reward += advantage_reward

        # 3. Action diversity reward (prevent button mashing)
        # Handle gymnasium + stable-retro action format
        # Actions in retro are typically arrays representing controller buttons
        if isinstance(action, np.ndarray):
            # Convert multi-dimensional retro actions to a comparable format
            # For retro games, action is typically a 1D array of button states
            if action.ndim == 1:
                current_action = tuple(action)  # Convert to tuple for comparison
            else:
                current_action = tuple(action.flatten())  # Flatten if multi-dimensional
        else:
            current_action = action

        prev_action = self.prev_action

        if prev_action is not None:
            try:
                # Compare action tuples (works for retro controller inputs)
                if current_action == prev_action:
                    self.consecutive_same_action += 1
                    if self.consecutive_same_action > 3:
                        diversity_penalty = -0.1 * self.reward_scale
                        reward += diversity_penalty
                else:
                    self.consecutive_same_action = 0
                    diversity_bonus = 0.05 * self.reward_scale
                    reward += diversity_bonus
            except (ValueError, TypeError) as e:
                # If comparison fails, assume actions are different (safe fallback)
                self.consecutive_same_action = 0
                diversity_bonus = 0.05 * self.reward_scale
                reward += diversity_bonus

        # 4. Combat engagement reward
        if enemy_health_diff > 0 or health_diff > 0:
            engagement_bonus = 0.1 * self.reward_scale
            reward += engagement_bonus

        # 5. Health preservation bonus
        player_health_ratio = player_health / self.full_hp
        if player_health_ratio > 0.8:
            preservation_bonus = 0.1 * self.reward_scale
            reward += preservation_bonus
        elif player_health_ratio < 0.2:
            low_health_penalty = -0.2 * self.reward_scale
            reward += low_health_penalty

        # 6. Small time penalty (scaled)
        time_penalty = -0.01 * self.reward_scale
        reward += time_penalty

        # Update tracking
        self.prev_player_health = player_health
        self.prev_enemy_health = enemy_health
        self.prev_action = current_action  # Store scalar action

        return reward

    def _calculate_outcome_reward(self, game_info, info):
        """
        Calculate sparse outcome reward for round completion
        """
        player_health = game_info["player_health"]
        enemy_health = game_info["opponent_health"]

        # Check for round completion
        round_over = player_health <= 0 or enemy_health <= 0

        if round_over:
            if player_health > enemy_health:
                # Win with health-based scaling
                health_bonus = (player_health / self.full_hp) * 5.0
                win_reward = (15.0 + health_bonus) * self.reward_scale

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
                loss_penalty = -(8.0 + health_disadvantage * 3.0) * self.reward_scale

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

        # Reset frame buffer
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
            # For retro games, action is typically a 1D array of button states
            action_scalar = tuple(action.flatten())  # Convert to tuple for storage
        else:
            action_scalar = action

        self.action_history.append(action_scalar)

        # Get game information
        game_info = self._extract_game_state(info)

        # Calculate PRIME-style rewards
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


# CBAM (Convolutional Block Attention Module) - Optimized
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    def __init__(self, in_channels, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.channel_attention(x)
        x = x * self.spatial_attention(x)
        return x


# PRIME-compatible Multi-Head Attention
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads=8, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch_size, seq_len, d_model = x.size()

        Q = (
            self.w_q(x)
            .view(batch_size, seq_len, self.num_heads, self.d_k)
            .transpose(1, 2)
        )
        K = (
            self.w_k(x)
            .view(batch_size, seq_len, self.num_heads, self.d_k)
            .transpose(1, 2)
        )
        V = (
            self.w_v(x)
            .view(batch_size, seq_len, self.num_heads, self.d_k)
            .transpose(1, 2)
        )

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        context = torch.matmul(attention_weights, V)
        context = (
            context.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        )
        output = self.w_o(context)
        return output


# PRIME-Optimized EfficientNet-B3 Feature Extractor
class PRIMEOptimizedEfficientNetB3(BaseFeaturesExtractor):
    """
    PRIME-optimized EfficientNet-B3 feature extractor
    Designed for dense reward processing and stable gradients
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 512):
        super().__init__(observation_space, features_dim)

        n_input_channels = observation_space.shape[0]  # 27 channels

        print("🔄 Loading PRIME-optimized EfficientNet-B3 from ImageNet...")

        # Load pre-trained EfficientNet-B3
        efficientnet_b3 = torchvision.models.efficientnet_b3(weights="IMAGENET1K_V1")
        self.backbone_features = efficientnet_b3.features

        # Adapt first layer for 27 channels
        original_conv = self.backbone_features[0][0]
        self.backbone_features[0][0] = nn.Conv2d(
            n_input_channels,
            original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=original_conv.bias,
        )

        # Initialize with averaged pretrained weights
        with torch.no_grad():
            pretrained_weight = original_conv.weight
            new_weight = pretrained_weight.mean(dim=1, keepdim=True).repeat(
                1, n_input_channels, 1, 1
            )
            new_weight = new_weight / (n_input_channels / 3.0)
            self.backbone_features[0][0].weight.copy_(new_weight)

        # Strategic CBAM placement for fighting games
        self.attention_modules = nn.ModuleDict(
            {
                "stage_4": CBAM(48, reduction=12),  # Mid-level features
                "stage_8": CBAM(384, reduction=12),  # High-level features
            }
        )

        # Global feature processing
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()

        # PRIME-compatible attention processing
        self.feature_attention = nn.Sequential(
            nn.Linear(1536, 384),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(384, 1536),
            nn.Sigmoid(),
        )

        # Gradient-friendly classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(1536, 768),
            nn.ReLU(inplace=True),  # ReLU for stable gradients
            nn.Dropout(0.1),
            nn.Linear(768, features_dim),
            nn.ReLU(inplace=True),
        )

        print(f"🧠 PRIME-OPTIMIZED EFFICIENTNET-B3:")
        print(f"   📊 Input: {observation_space.shape}")
        print(f"   🎨 Input channels: {n_input_channels} (adapted from ImageNet)")
        print(f"   🏆 Pre-trained: ImageNet with fighting game optimization")
        print(f"   🔍 Strategic CBAM at key stages")
        print(f"   💾 Memory optimized for 11GB GPU")
        print(f"   🎯 Output features: {features_dim}")
        print(f"   🚀 PRIME-compatible with stable gradients!")

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # Normalize with gradient-friendly scaling
        x = observations.float() / 255.0

        # ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device, dtype=x.dtype)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device, dtype=x.dtype)

        # Apply normalization per RGB triplet
        batch_size, channels, height, width = x.shape
        x_reshaped = x.view(batch_size, -1, 3, height, width)

        for i in range(3):
            x_reshaped[:, :, i] = (x_reshaped[:, :, i] - mean[i]) / std[i]

        x = x_reshaped.view(batch_size, channels, height, width)

        # Forward through backbone with strategic attention
        for i, layer in enumerate(self.backbone_features):
            x = layer(x)

            # Apply attention at key stages
            if i == 4 and x.shape[1] == 48:
                x = self.attention_modules["stage_4"](x)
            elif i == 8 and x.shape[1] == 384:
                x = self.attention_modules["stage_8"](x)

        # Global pooling and feature processing
        features = self.global_pool(x)
        features = self.flatten(features)

        # Apply attention weighting
        attention_weights = self.feature_attention(features)
        attended_features = features * attention_weights

        # Final classification with stable gradients
        output = self.classifier(attended_features)
        return output


# Optional: Utility function to create the wrapped environment
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
    """
    import retro

    env = retro.make(
        game=game,
        state=state,
        use_restricted_actions=retro.Actions.FILTERED,
        obs_type=retro.Observations.IMAGE,
        render_mode="human" if rendering else None,
    )

    env = SamuraiShowdownCustomWrapper(
        env,
        reset_round=reset_round,
        rendering=rendering,
        max_episode_steps=max_episode_steps,
        **wrapper_kwargs,
    )

    return env


if __name__ == "__main__":
    # Test the PRIME-optimized wrapper
    print("🧪 Testing PRIME-optimized SamuraiShowdownCustomWrapper...")

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
        print("✅ PRIME-optimized wrapper test completed successfully!")

    except Exception as e:
        print(f"❌ Wrapper test failed: {e}")
        import traceback

        traceback.print_exc()
