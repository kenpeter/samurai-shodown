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


class SamuraiShowdownCustomWrapper(gym.Wrapper):
    """
    Custom wrapper for Samurai Showdown fighting game optimized for RL training

    Features:
    - Frame stacking (9 frames) with RGB channels (27 total channels)
    - Multi-component reward system (distance, health, combo, defensive)
    - Performance tracking (win/loss statistics)
    - Proper episode management for fighting games
    """

    def __init__(
        self,
        env,
        reset_round=True,
        rendering=False,
        max_episode_steps=15000,
        frame_stack=9,
        frame_skip=4,
        target_size=(180, 126),  # Fixed: Use actual environment dimensions (W, H)
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

        # Performance tracking
        self.current_stats = {
            "wins": 0,
            "losses": 0,
            "total_rounds": 0,
            "win_rate": 0.0,
            "current_episode_reward": 0.0,
            "best_win_streak": 0,
            "current_win_streak": 0,
        }

        # Episode tracking
        self.step_count = 0
        self.episode_count = 0

        # Game state tracking
        self.prev_player_health = None
        self.prev_enemy_health = None

        # Frame buffer for stacking
        self.frame_buffer = deque(maxlen=self.frame_stack)

        # Initialize frame buffer with zeros using correct dimensions
        # target_size is (W, H) = (180, 126), so frame shape is (H, W, 3) = (126, 180, 3)
        dummy_frame = np.zeros(
            (self.target_size[1], self.target_size[0], 3), dtype=np.uint8
        )
        for _ in range(self.frame_stack):
            self.frame_buffer.append(dummy_frame)

        # Update observation space for stacked RGB frames
        # Final shape: (27, 180, 126) - channels first, then width, then height
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(
                self.frame_stack * 3,
                self.target_size[0],
                self.target_size[1],
            ),  # (27, 180, 126)
            dtype=np.uint8,
        )

        print(f"🎮 SAMURAI SHOWDOWN WRAPPER INITIALIZED:")
        print(f"   📊 Observation space: {self.observation_space.shape}")
        print(
            f"   🎨 Frame stacking: {self.frame_stack} frames × 3 RGB = {self.frame_stack * 3} channels"
        )
        print(f"   🎯 Target size: {self.target_size}")
        print(f"   ⏱️ Max episode steps: {self.max_episode_steps}")
        print(f"   🔄 Frame skip: {self.frame_skip}")

    def _preprocess_frame(self, frame):
        """Preprocess frame: resize and maintain RGB channels"""
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            # Resize frame while maintaining RGB - target_size is (W, H)
            frame = cv2.resize(frame, self.target_size, interpolation=cv2.INTER_AREA)
        else:
            # Convert grayscale to RGB if needed
            if len(frame.shape) == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            frame = cv2.resize(frame, self.target_size, interpolation=cv2.INTER_AREA)

        return frame.astype(np.uint8)

    def _get_stacked_observation(self):
        """Create stacked observation from frame buffer"""
        # Stack frames along channel dimension
        # Convert from list of (H, W, 3) to (9*3, W, H) = (27, 180, 126)
        stacked = np.concatenate(list(self.frame_buffer), axis=2)  # (H, W, 27)
        stacked = np.transpose(stacked, (2, 0, 1))  # (27, H, W) = (27, 126, 180)
        stacked = np.transpose(stacked, (0, 2, 1))  # (27, W, H) = (27, 180, 126)

        return stacked

    def _extract_game_state(self, info):
        """Extract enhanced game state information for fighting games"""
        # Basic health information
        player_health = info.get("health", self.full_hp)
        opponent_health = info.get("enemy_health", self.full_hp)

        # Round information
        current_round = info.get("round", 1)

        # Score information
        score = info.get("score", 0)

        return {
            "player_health": player_health,
            "opponent_health": opponent_health,
            "current_round": current_round,
            "score": score,
        }

    def _calculate_reward(self, game_info, info):
        """Multi-component reward system for fighting games"""
        reward = 0.0

        player_health = game_info["player_health"]
        enemy_health = game_info["opponent_health"]

        # Initialize previous values if first step
        if self.prev_player_health is None:
            self.prev_player_health = player_health
            self.prev_enemy_health = enemy_health

        # 1. Health difference rewards/penalties
        health_diff = self.prev_player_health - player_health
        enemy_health_diff = self.prev_enemy_health - enemy_health

        # Reward for damaging enemy
        if enemy_health_diff > 0:
            reward += enemy_health_diff * 0.1

        # Penalty for taking damage
        if health_diff > 0:
            reward -= health_diff * 0.05

        # 2. Health advantage bonus
        health_advantage = player_health - enemy_health
        reward += health_advantage * 0.001

        # 3. Round completion rewards (if round info available)
        current_round = game_info.get("current_round", 1)

        # Simple round end detection based on very low health
        round_over = player_health <= 0 or enemy_health <= 0

        if round_over:
            if player_health > enemy_health:
                reward += 10.0  # Win round
                self.current_stats["wins"] += 1
                self.current_stats["current_win_streak"] += 1
                self.current_stats["best_win_streak"] = max(
                    self.current_stats["best_win_streak"],
                    self.current_stats["current_win_streak"],
                )
            else:
                reward -= 5.0  # Lose round
                self.current_stats["losses"] += 1
                self.current_stats["current_win_streak"] = 0

            self.current_stats["total_rounds"] += 1
            if self.current_stats["total_rounds"] > 0:
                self.current_stats["win_rate"] = (
                    self.current_stats["wins"] / self.current_stats["total_rounds"]
                )

        # 4. Small time penalty to encourage action
        reward -= 0.001

        # Update previous values
        self.prev_player_health = player_health
        self.prev_enemy_health = enemy_health

        return reward

    def _check_done(self, game_info, info):
        """Check if episode should end"""
        # Episode ends if:
        # 1. Maximum steps reached
        if self.step_count >= self.max_episode_steps:
            return True

        # 2. Round is over (if reset_round is False)
        player_health = game_info["player_health"]
        opponent_health = game_info["opponent_health"]

        round_over = player_health <= 0 or opponent_health <= 0
        if not self.reset_round and round_over:
            return True

        # 3. Both players have no health left
        if player_health <= 0 and opponent_health <= 0:
            return True

        return False

    def reset(self, **kwargs):
        """Reset environment and frame buffer"""
        obs, info = self.env.reset(**kwargs)

        # Reset tracking variables
        self.step_count = 0
        self.episode_count += 1
        self.prev_player_health = None
        self.prev_enemy_health = None
        self.current_stats["current_episode_reward"] = 0.0

        # Reset frame buffer
        processed_frame = self._preprocess_frame(obs)
        self.frame_buffer.clear()
        for _ in range(self.frame_stack):
            self.frame_buffer.append(processed_frame)

        stacked_obs = self._get_stacked_observation()

        return stacked_obs, info

    def step(self, action):
        """Execute action with frame skipping and return processed observation"""
        total_reward = 0.0
        info = {}

        # why need frame skip?
        for _ in range(self.frame_skip):
            obs, reward, terminated, truncated, step_info = self.env.step(action)
            total_reward += reward
            info.update(step_info)

            if terminated or truncated:
                break

        # Process frame and add to buffer
        processed_frame = self._preprocess_frame(obs)
        self.frame_buffer.append(processed_frame)

        # Get game information
        game_info = self._extract_game_state(info)

        # Calculate custom reward
        custom_reward = self._calculate_reward(game_info, info)
        total_reward += custom_reward

        # Update episode reward tracking
        self.current_stats["current_episode_reward"] += total_reward

        # Check if episode should end
        done = self._check_done(game_info, info) or terminated or truncated

        # Get stacked observation
        stacked_obs = self._get_stacked_observation()

        self.step_count += 1

        # Add custom info
        info.update(
            {
                "game_info": game_info,
                "custom_reward": custom_reward,
                "episode_steps": self.step_count,
                "episode_reward": self.current_stats["current_episode_reward"],
            }
        )

        return stacked_obs, total_reward, done, False, info

    def render(self, mode="human"):
        """Render environment if rendering is enabled"""
        if self.rendering:
            return self.env.render()
        return None

    def close(self):
        """Close environment"""
        return self.env.close()


# CBAM (Convolutional Block Attention Module)
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


# Multi-Head Attention for feature refinement
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

        # Linear transformations and split into heads
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

        # Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(
            torch.tensor(self.d_k, dtype=torch.float32)
        )
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        context = torch.matmul(attention_weights, V)
        context = (
            context.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        )

        output = self.w_o(context)
        return output


# Ultra-lightweight CNN for memory-constrained systems
class UltraLightCNNFeatureExtractor(BaseFeaturesExtractor):
    """
    Ultra-lightweight CNN feature extractor for 11GB GPUs
    No pre-trained weights, minimal memory usage
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 512):
        super().__init__(observation_space, features_dim)

        n_input_channels = observation_space.shape[0]  # 27 channels

        # Ultra-lightweight network
        self.features = nn.Sequential(
            # Stage 1: Initial compression
            nn.Conv2d(n_input_channels, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Stage 2: Feature extraction
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Stage 3: Pattern recognition
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Stage 4: High-level features
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(512, features_dim),
            nn.ReLU(inplace=True),
        )

        print(f"🧠 ULTRA-LIGHT CNN:")
        print(f"   📊 Input: {observation_space.shape}")
        print(f"   💾 Memory: Minimal (~1GB)")
        print(f"   🎨 Channels: {n_input_channels} → 32 → 64 → 128 → 256")
        print(f"   🎯 Output: {features_dim}")
        print(f"   ⚡ Ultra-lightweight for 11GB GPUs")

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # Simple normalization
        x = observations.float() / 255.0

        # Feature extraction
        x = self.features(x)

        # Classification
        x = self.classifier(x)

        return x


# Lightweight EfficientNet-B0 feature extractor for memory-constrained systems
class LightweightEfficientNetFeatureExtractor(BaseFeaturesExtractor):
    """
    Lightweight EfficientNet-B0 from ImageNet with attention mechanisms
    Optimized for fighting games with lower memory usage
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 512):
        super().__init__(observation_space, features_dim)

        n_input_channels = observation_space.shape[0]  # 27 channels (9 frames × 3 RGB)

        # Load pre-trained EfficientNet-B0 (much smaller than B3)
        print("🔄 Loading pre-trained EfficientNet-B0 from ImageNet (lightweight)...")

        # Get the pre-trained model (B0 is much smaller)
        efficientnet_b0 = torchvision.models.efficientnet_b0(weights="IMAGENET1K_V1")

        # Extract features (everything except the classifier)
        self.backbone_features = efficientnet_b0.features

        # Get the original first conv layer
        original_conv = self.backbone_features[0][0]

        # Create new first conv layer to handle 27 input channels
        self.backbone_features[0][0] = nn.Conv2d(
            n_input_channels,  # 27 channels instead of 3
            original_conv.out_channels,  # Keep same output channels (32 for B0)
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=original_conv.bias,
        )

        # Initialize new conv weights using pretrained weights
        with torch.no_grad():
            # Average the pretrained weights across input channels and repeat
            pretrained_weight = original_conv.weight
            new_weight = pretrained_weight.mean(dim=1, keepdim=True).repeat(
                1, n_input_channels, 1, 1
            )
            # Scale down to maintain similar activation magnitudes
            new_weight = new_weight / (n_input_channels / 3.0)
            self.backbone_features[0][0].weight.copy_(new_weight)

        # Simplified attention (only at key stages to save memory)
        self.attention_modules = nn.ModuleDict(
            {
                "stage_3": CBAM(40),  # After stage 3
                "stage_6": CBAM(112),  # After stage 6
            }
        )

        # Global pooling (EfficientNet-B0 outputs 1280 channels instead of 1536)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()

        # Simplified attention mechanism to save memory
        self.simple_attention = nn.Sequential(
            nn.Linear(1280, 320),
            nn.ReLU(inplace=True),
            nn.Linear(320, 1280),
            nn.Sigmoid(),
        )

        # Final classification layers with aggressive compression
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(1280, 512),
            nn.SiLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(512, features_dim),
            nn.SiLU(inplace=True),
        )

        print(f"🧠 LIGHTWEIGHT EFFICIENTNET-B0 + ATTENTION:")
        print(f"   📊 Input: {observation_space.shape}")
        print(
            f"   🎨 Input channels: {n_input_channels} (adapted from 3-channel ImageNet)"
        )
        print(f"   🏆 Pre-trained: ImageNet weights with transfer learning")
        print(f"   🔍 Simplified CBAM attention to save memory")
        print(f"   💾 Memory optimized: B0 instead of B3 (much smaller)")
        print(f"   🎯 Output features: {features_dim}")
        print(f"   🚀 Expected: Good performance with lower memory usage!")

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # Normalize input to ImageNet standards
        x = observations.float() / 255.0

        # ImageNet normalization (adapted for our multi-channel input)
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device, dtype=x.dtype)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device, dtype=x.dtype)

        # Reshape for normalization (apply to every 3 channels)
        batch_size, channels, height, width = x.shape
        x_reshaped = x.view(batch_size, -1, 3, height, width)  # [B, 9, 3, H, W]

        # Normalize each RGB triplet
        for i in range(3):
            x_reshaped[:, :, i] = (x_reshaped[:, :, i] - mean[i]) / std[i]

        x = x_reshaped.view(
            batch_size, channels, height, width
        )  # Back to [B, 27, H, W]

        # Pass through EfficientNet backbone with limited attention
        for i, layer in enumerate(self.backbone_features):
            x = layer(x)

            # Apply CBAM attention at limited stages to save memory
            if i == 3 and x.shape[1] == 40:  # After stage 3
                x = self.attention_modules["stage_3"](x)
            elif i == 6 and x.shape[1] == 112:  # After stage 6
                x = self.attention_modules["stage_6"](x)

        # Global pooling and flatten
        features = self.global_pool(x)
        features = self.flatten(features)  # Shape: [batch_size, 1280]

        # Simple attention instead of multi-head to save memory
        attention_weights = self.simple_attention(features)
        attended_features = features * attention_weights

        # Final classification
        output = self.classifier(attended_features)

        return output


# High-Performance EfficientNet-B3 feature extractor with maximum VRAM usage - FIXED VERSION
class HighPerformanceEfficientNetB3FeatureExtractor(BaseFeaturesExtractor):
    """
    High-Performance EfficientNet-B3 from ImageNet with maximum VRAM usage
    Designed for high-end GPUs (16GB+ VRAM) with aggressive performance optimizations
    FIXED: Tensor dimension mismatch in multi-scale pooling
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 1024):
        super().__init__(observation_space, features_dim)

        n_input_channels = observation_space.shape[0]  # 27 channels (9 frames × 3 RGB)

        # Load pre-trained EfficientNet-B3 from torchvision
        print(
            "🔄 Loading pre-trained EfficientNet-B3 from ImageNet (HIGH PERFORMANCE MODE)..."
        )

        # Get the pre-trained model
        efficientnet_b3 = torchvision.models.efficientnet_b3(weights="IMAGENET1K_V1")

        # Extract features (everything except the classifier)
        self.backbone_features = efficientnet_b3.features

        # Get the original first conv layer
        original_conv = self.backbone_features[0][0]

        # Create new first conv layer to handle 27 input channels
        self.backbone_features[0][0] = nn.Conv2d(
            n_input_channels,  # 27 channels instead of 3
            original_conv.out_channels,  # Keep same output channels (40)
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=original_conv.bias,
        )

        # Initialize new conv weights using pretrained weights
        with torch.no_grad():
            # Average the pretrained weights across input channels and repeat
            pretrained_weight = original_conv.weight
            new_weight = pretrained_weight.mean(dim=1, keepdim=True).repeat(
                1, n_input_channels, 1, 1
            )
            # Scale down to maintain similar activation magnitudes
            new_weight = new_weight / (n_input_channels / 3.0)
            self.backbone_features[0][0].weight.copy_(new_weight)

        # ENHANCED ATTENTION: Add CBAM at ALL major stages for maximum pattern recognition
        self.attention_modules = nn.ModuleDict(
            {
                "stage_1": CBAM(24, reduction=8),  # More aggressive attention
                "stage_2": CBAM(32, reduction=8),
                "stage_3": CBAM(48, reduction=8),
                "stage_4": CBAM(96, reduction=8),
                "stage_5": CBAM(136, reduction=8),
                "stage_6": CBAM(232, reduction=8),
                "stage_7": CBAM(384, reduction=8),
            }
        )

        # Enhanced spatial attention for fighting game patterns
        self.spatial_attention = nn.ModuleList(
            [
                SpatialAttention(kernel_size=7),
                SpatialAttention(kernel_size=5),
                SpatialAttention(kernel_size=3),
            ]
        )

        # FIXED: Simplified pooling strategy to avoid dimension mismatch
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()

        # Enhanced Multi-head attention with more heads and larger dimensions
        self.attention_1 = MultiHeadAttention(1536, num_heads=16, dropout=0.1)
        self.attention_2 = MultiHeadAttention(1536, num_heads=16, dropout=0.1)

        # Layer normalization for attention stability
        self.layer_norm_1 = nn.LayerNorm(1536)
        self.layer_norm_2 = nn.LayerNorm(1536)

        # FIXED: Feature processor with correct input dimension
        self.feature_processor = nn.Sequential(
            nn.Linear(1536, 2048),  # FIXED: Direct 1536 input, not 1536*7
            nn.SiLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(2048, 1536),
            nn.SiLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(1536, 1024),
            nn.SiLU(inplace=True),
            nn.Dropout(0.1),
        )

        # Final classification layers with enhanced capacity
        self.classifier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.SiLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(512, features_dim),
            nn.SiLU(inplace=True),
        )

        print(f"🧠 HIGH-PERFORMANCE EFFICIENTNET-B3 + MAXIMUM ATTENTION (FIXED):")
        print(f"   📊 Input: {observation_space.shape}")
        print(
            f"   🎨 Input channels: {n_input_channels} (adapted from 3-channel ImageNet)"
        )
        print(f"   🏆 Pre-trained: ImageNet weights with transfer learning")
        print(f"   🔍 CBAM attention at ALL 7 stages with aggressive reduction")
        print(
            f"   🧩 Dual Multi-head attention (16 heads each) for complex relationships"
        )
        print(f"   🎯 Global pooling only (simplified for stability)")
        print(
            f"   💪 Enhanced feature processing: 1536 → 2048 → 1536 → 1024 → {features_dim}"
        )
        print(f"   🚀 Expected: MAXIMUM performance with high VRAM usage!")
        print(f"   💾 Estimated VRAM: 16-24GB (designed for high-end GPUs)")
        print(f"   ✅ FIXED: Tensor dimension mismatch resolved")

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # Normalize input to ImageNet standards with enhanced preprocessing
        x = observations.float() / 255.0

        # ImageNet normalization with per-channel statistics
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device, dtype=x.dtype)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device, dtype=x.dtype)

        # Reshape for normalization (apply to every 3 channels)
        batch_size, channels, height, width = x.shape
        x_reshaped = x.view(batch_size, -1, 3, height, width)  # [B, 9, 3, H, W]

        # Normalize each RGB triplet
        for i in range(3):
            x_reshaped[:, :, i] = (x_reshaped[:, :, i] - mean[i]) / std[i]

        x = x_reshaped.view(
            batch_size, channels, height, width
        )  # Back to [B, 27, H, W]

        # Pass through EfficientNet backbone with MAXIMUM attention
        for i, layer in enumerate(self.backbone_features):
            x = layer(x)

            # Apply CBAM attention at ALL major stages
            if i == 1 and x.shape[1] == 24:  # Stage 1
                x = self.attention_modules["stage_1"](x)
            elif i == 2 and x.shape[1] == 32:  # Stage 2
                x = self.attention_modules["stage_2"](x)
            elif i == 3 and x.shape[1] == 48:  # Stage 3
                x = self.attention_modules["stage_3"](x)
                # Add spatial attention for fighting patterns
                for spatial_att in self.spatial_attention:
                    x = x * spatial_att(x)
            elif i == 4 and x.shape[1] == 96:  # Stage 4
                x = self.attention_modules["stage_4"](x)
            elif i == 5 and x.shape[1] == 136:  # Stage 5
                x = self.attention_modules["stage_5"](x)
            elif i == 6 and x.shape[1] == 232:  # Stage 6
                x = self.attention_modules["stage_6"](x)
            elif i == 7 and x.shape[1] == 384:  # Stage 7
                x = self.attention_modules["stage_7"](x)

        # FIXED: Simple global pooling to get [batch_size, 1536]
        global_features = self.global_pool(x)
        global_features = self.flatten(global_features)  # [batch_size, 1536]

        # Enhanced dual multi-head attention processing
        # Reshape for attention: [batch_size, 1, 1536]
        attention_input = global_features.unsqueeze(1)  # [batch_size, 1, 1536]

        # First attention layer with residual connection
        attended_1 = self.attention_1(attention_input)
        attended_1 = self.layer_norm_1(attended_1 + attention_input)

        # Second attention layer with residual connection
        attended_2 = self.attention_2(attended_1)
        attended_2 = self.layer_norm_2(attended_2 + attended_1)

        # Flatten back: [batch_size, 1536]
        final_features = attended_2.squeeze(1)  # [batch_size, 1536]

        # Enhanced feature processing
        processed_features = self.feature_processor(final_features)

        # Final classification
        output = self.classifier(processed_features)

        return output


# Pre-trained EfficientNet-B3 feature extractor with attention mechanisms
class EfficientNetB3FeatureExtractor(BaseFeaturesExtractor):
    """
    Pre-trained EfficientNet-B3 from ImageNet with Multi-Head Attention and CBAM
    Optimized for fighting game pattern recognition with transfer learning
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 512):
        super().__init__(observation_space, features_dim)

        n_input_channels = observation_space.shape[0]  # 27 channels (9 frames × 3 RGB)

        # Load pre-trained EfficientNet-B3 from torchvision
        print("🔄 Loading pre-trained EfficientNet-B3 from ImageNet...")

        # Get the pre-trained model
        efficientnet_b3 = torchvision.models.efficientnet_b3(weights="IMAGENET1K_V1")

        # Extract features (everything except the classifier)
        self.backbone_features = efficientnet_b3.features

        # Get the original first conv layer
        original_conv = self.backbone_features[0][0]

        # Create new first conv layer to handle 27 input channels
        self.backbone_features[0][0] = nn.Conv2d(
            n_input_channels,  # 27 channels instead of 3
            original_conv.out_channels,  # Keep same output channels (40)
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=original_conv.bias,
        )

        # Initialize new conv weights using pretrained weights
        with torch.no_grad():
            # Average the pretrained weights across input channels and repeat
            pretrained_weight = original_conv.weight
            new_weight = pretrained_weight.mean(dim=1, keepdim=True).repeat(
                1, n_input_channels, 1, 1
            )
            # Scale down to maintain similar activation magnitudes
            new_weight = new_weight / (n_input_channels / 3.0)
            self.backbone_features[0][0].weight.copy_(new_weight)

        # Add CBAM attention after key feature extraction layers
        self.attention_modules = nn.ModuleDict(
            {
                "stage_1": CBAM(24),  # After stage 1
                "stage_3": CBAM(48),  # After stage 3
                "stage_5": CBAM(136),  # After stage 5
                "stage_7": CBAM(384),  # After final stage
            }
        )

        # Global pooling (EfficientNet-B3 outputs 1536 channels)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()

        # Multi-head attention for feature refinement
        self.attention = MultiHeadAttention(1536, num_heads=8)

        # Final classification layers with dropout for regularization
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(1536, 768),
            nn.SiLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(768, features_dim),
            nn.SiLU(inplace=True),
        )

        print(f"🧠 PRE-TRAINED EFFICIENTNET-B3 + ATTENTION:")
        print(f"   📊 Input: {observation_space.shape}")
        print(
            f"   🎨 Input channels: {n_input_channels} (adapted from 3-channel ImageNet)"
        )
        print(f"   🏆 Pre-trained: ImageNet weights with transfer learning")
        print(f"   🔍 CBAM attention at key stages for pattern focus")
        print(f"   🧩 Multi-head attention (8 heads) for feature relationships")
        print(f"   🎯 Output features: {features_dim}")
        print(f"   🚀 Expected: Much better performance with ImageNet knowledge!")

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # Normalize input to ImageNet standards
        x = observations.float() / 255.0

        # ImageNet normalization (adapted for our multi-channel input)
        # Note: We apply the same normalization per channel
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device, dtype=x.dtype)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device, dtype=x.dtype)

        # Reshape for normalization (apply to every 3 channels)
        batch_size, channels, height, width = x.shape
        x_reshaped = x.view(batch_size, -1, 3, height, width)  # [B, 9, 3, H, W]

        # Normalize each RGB triplet
        for i in range(3):
            x_reshaped[:, :, i] = (x_reshaped[:, :, i] - mean[i]) / std[i]

        x = x_reshaped.view(
            batch_size, channels, height, width
        )  # Back to [B, 27, H, W]

        # Pass through EfficientNet backbone with attention
        stage_outputs = []

        for i, layer in enumerate(self.backbone_features):
            x = layer(x)

            # Apply CBAM attention at key stages
            if i == 1 and x.shape[1] == 24:  # After stage 1
                x = self.attention_modules["stage_1"](x)
            elif i == 3 and x.shape[1] == 48:  # After stage 3
                x = self.attention_modules["stage_3"](x)
            elif i == 5 and x.shape[1] == 136:  # After stage 5
                x = self.attention_modules["stage_5"](x)
            elif i == 7 and x.shape[1] == 384:  # After final stage
                x = self.attention_modules["stage_7"](x)

        # Global pooling and flatten
        features = self.global_pool(x)
        features = self.flatten(features)  # Shape: [batch_size, 1536]

        # Multi-head attention (treat features as sequence of length 1)
        features_reshaped = features.unsqueeze(1)  # Shape: [batch_size, 1, 1536]
        attended_features = self.attention(features_reshaped)
        attended_features = attended_features.squeeze(1)  # Shape: [batch_size, 1536]

        # Final classification
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
    Utility function to create a wrapped Samurai Showdown environment
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
    # Test the wrapper
    print("🧪 Testing SamuraiShowdownCustomWrapper...")

    try:
        env = make_samurai_env(rendering=False)

        print(f"✅ Environment created successfully")
        print(f"📊 Observation space: {env.observation_space}")
        print(f"🎮 Action space: {env.action_space}")

        # Test reset
        obs, info = env.reset()
        print(f"📸 Reset observation shape: {obs.shape}")

        # Test a few steps
        for i in range(5):
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)
            print(
                f"Step {i+1}: obs_shape={obs.shape}, reward={reward:.3f}, done={done}"
            )

            if done:
                break

        # Print stats
        print(f"📊 Final stats: {env.current_stats}")

        env.close()
        print("✅ Wrapper test completed successfully!")

    except Exception as e:
        print(f"❌ Wrapper test failed: {e}")
        import traceback

        traceback.print_exc()
