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
import logging

# Setup logging
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class SamuraiShowdownCustomWrapper(gym.Wrapper):
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
        self.current_stats = {
            "wins": 0,
            "losses": 0,
            "total_rounds": 0,
            "win_rate": 0.0,
            "current_episode_reward": 0.0,
            "best_win_streak": 0,
            "current_win_streak": 0,
        }
        self.step_count = 0
        self.episode_count = 0
        self.prev_player_health = None
        self.prev_enemy_health = None
        self.prev_player_x = None
        self.prev_enemy_x = None
        self.prev_distance = None
        self.frame_buffer = deque(maxlen=self.frame_stack)
        dummy_frame = np.zeros(
            (self.target_size[1], self.target_size[0], 3), dtype=np.uint8
        )
        for _ in range(self.frame_stack):
            self.frame_buffer.append(dummy_frame)
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
        logger.info(f"SAMURAI SHOWDOWN WRAPPER INITIALIZED:")
        logger.info(f"Observation space: {self.observation_space.shape}")
        logger.info(
            f"Frame stacking: {self.frame_stack} frames × 3 RGB = {self.frame_stack * 3} channels"
        )
        logger.info(f"Target size: {self.target_size}")
        logger.info(f"Max episode steps: {self.max_episode_steps}")
        logger.info(f"Frame skip: {self.frame_skip}")

    def _preprocess_frame(self, frame):
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            frame = cv2.resize(frame, self.target_size, interpolation=cv2.INTER_AREA)
        else:
            if len(frame.shape) == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            frame = cv2.resize(frame, self.target_size, interpolation=cv2.INTER_AREA)
        if np.any(np.isnan(frame)) or np.any(np.isinf(frame)):
            logger.warning(f"Invalid frame detected: {frame}")
        return frame.astype(np.uint8)

    def _get_stacked_observation(self):
        stacked = np.concatenate(list(self.frame_buffer), axis=2)
        stacked = np.transpose(stacked, (2, 0, 1))
        stacked = np.transpose(stacked, (0, 2, 1))
        if np.any(np.isnan(stacked)) or np.any(np.isinf(stacked)):
            logger.warning(f"Invalid stacked observation detected")
        return stacked

    def _extract_game_info(self, info=None):
        try:
            ram = self.env.unwrapped.data
            return {
                "player_health": 100,
                "enemy_health": 100,
                "player_x": 80,
                "enemy_x": 240,
                "round_over": False,
            }
        except Exception:
            return {
                "player_health": 100,
                "enemy_health": 100,
                "player_x": 80,
                "enemy_x": 240,
                "round_over": False,
            }

    def _calculate_reward(self, game_info, info):
        reward = 0.0
        player_health = game_info["player_health"]
        enemy_health = game_info["enemy_health"]
        player_x = game_info["player_x"]
        enemy_x = game_info["enemy_x"]
        if self.prev_player_health is None:
            self.prev_player_health = player_health
            self.prev_enemy_health = enemy_health
            self.prev_player_x = player_x
            self.prev_enemy_x = enemy_x
            self.prev_distance = abs(player_x - enemy_x)
        health_diff = self.prev_player_health - player_health
        enemy_health_diff = self.prev_enemy_health - enemy_health
        if enemy_health_diff > 0:
            reward += enemy_health_diff * 0.1
        if health_diff > 0:
            reward -= health_diff * 0.05
        current_distance = abs(player_x - enemy_x)
        distance_change = self.prev_distance - current_distance
        if distance_change > 0:
            reward += distance_change * 0.001
        screen_width = 320
        if player_x < 20 or player_x > screen_width - 20:
            reward -= 0.005
        if game_info["round_over"]:
            if player_health > enemy_health:
                reward += 10.0
                self.current_stats["wins"] += 1
                self.current_stats["current_win_streak"] += 1
                self.current_stats["best_win_streak"] = max(
                    self.current_stats["best_win_streak"],
                    self.current_stats["current_win_streak"],
                )
            else:
                reward -= 5.0
                self.current_stats["losses"] += 1
                self.current_stats["current_win_streak"] = 0
            self.current_stats["total_rounds"] += 1
            if self.current_stats["total_rounds"] > 0:
                self.current_stats["win_rate"] = (
                    self.current_stats["wins"] / self.current_stats["total_rounds"]
                )
        reward -= 0.001
        if np.isnan(reward) or np.isinf(reward):
            logger.warning(f"Invalid reward calculated: {reward}")
            reward = 0.0
        self.prev_player_health = player_health
        self.prev_enemy_health = enemy_health
        self.prev_player_x = player_x
        self.prev_enemy_x = enemy_x
        self.prev_distance = current_distance
        return reward

    def _check_done(self, game_info, info):
        if self.step_count >= self.max_episode_steps:
            return True
        if not self.reset_round and game_info["round_over"]:
            return True
        if game_info["player_health"] <= 0 and game_info["enemy_health"] <= 0:
            return True
        return False

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.step_count = 0
        self.episode_count += 1
        self.prev_player_health = None
        self.prev_enemy_health = None
        self.prev_player_x = None
        self.prev_enemy_x = None
        self.prev_distance = None
        self.current_stats["current_episode_reward"] = 0.0
        processed_frame = self._preprocess_frame(obs)
        self.frame_buffer.clear()
        for _ in range(self.frame_stack):
            self.frame_buffer.append(processed_frame)
        stacked_obs = self._get_stacked_observation()
        return stacked_obs, info

    def step(self, action):
        total_reward = 0.0
        info = {}
        for _ in range(self.frame_skip):
            obs, reward, terminated, truncated, step_info = self.env.step(action)
            if np.any(np.isnan(obs)) or np.any(np.isinf(obs)):
                logger.warning(f"Invalid observation from env.step: {obs}")
            total_reward += reward
            info.update(step_info)
            if terminated or truncated:
                break
        processed_frame = self._preprocess_frame(obs)
        self.frame_buffer.append(processed_frame)
        game_info = self._extract_game_info(info)
        custom_reward = self._calculate_reward(game_info, info)
        total_reward += custom_reward
        self.current_stats["current_episode_reward"] += total_reward
        done = self._check_done(game_info, info) or terminated or truncated
        stacked_obs = self._get_stacked_observation()
        self.step_count += 1
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
        if self.rendering:
            return self.env.render()
        return None

    def close(self):
        return self.env.close()


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
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
        super().__init__()
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
        super().__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.channel_attention(x)
        x = x * self.spatial_attention(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads=8, dropout=0.1):
        super().__init__()
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


class UltraLightCNNFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 512):
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        self.features = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(512, features_dim),
            nn.ReLU(inplace=True),
        )
        logger.info(
            f"ULTRA-LIGHT CNN: Input {observation_space.shape}, Output {features_dim}, Memory ~1GB"
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        x = observations.float() / 255.0
        if torch.any(torch.isnan(x)) or torch.any(torch.isinf(x)):
            logger.warning("Invalid observations in UltraLightCNNFeatureExtractor")
        x = self.features(x)
        x = self.classifier(x)
        return x


class LightweightEfficientNetFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 512):
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        efficientnet_b0 = torchvision.models.efficientnet_b0(weights="IMAGENET1K_V1")
        self.backbone_features = efficientnet_b0.features
        original_conv = self.backbone_features[0][0]
        self.backbone_features[0][0] = nn.Conv2d(
            n_input_channels,
            original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=original_conv.bias,
        )
        with torch.no_grad():
            pretrained_weight = original_conv.weight
            new_weight = pretrained_weight.mean(dim=1, keepdim=True).repeat(
                1, n_input_channels, 1, 1
            )
            new_weight = new_weight / (n_input_channels / 3.0)
            self.backbone_features[0][0].weight.copy_(new_weight)
        self.attention_modules = nn.ModuleDict(
            {"stage_3": CBAM(40), "stage_6": CBAM(112)}
        )
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.simple_attention = nn.Sequential(
            nn.Linear(1280, 320),
            nn.ReLU(inplace=True),
            nn.Linear(320, 1280),
            nn.Sigmoid(),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(1280, 512),
            nn.SiLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(512, features_dim),
            nn.SiLU(inplace=True),
        )
        logger.info(
            f"LIGHTWEIGHT EFFICIENTNET-B0 + ATTENTION: Input {observation_space.shape}, Output {features_dim}"
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        x = observations.float() / 255.0
        if torch.any(torch.isnan(x)) or torch.any(torch.isinf(x)):
            logger.warning(
                "Invalid observations in LightweightEfficientNetFeatureExtractor"
            )
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device, dtype=x.dtype)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device, dtype=x.dtype)
        batch_size, channels, height, width = x.shape
        x_reshaped = x.view(batch_size, -1, 3, height, width)
        for i in range(3):
            x_reshaped[:, :, i] = (x_reshaped[:, :, i] - mean[i]) / std[i]
        x = x_reshaped.view(batch_size, channels, height, width)
        for i, layer in enumerate(self.backbone_features):
            x = layer(x)
            if i == 3 and x.shape[1] == 40:
                x = self.attention_modules["stage_3"](x)
            elif i == 6 and x.shape[1] == 112:
                x = self.attention_modules["stage_6"](x)
        features = self.global_pool(x)
        features = self.flatten(features)
        attention_weights = self.simple_attention(features)
        attended_features = features * attention_weights
        output = self.classifier(attended_features)
        return output


class HighPerformanceEfficientNetB3FeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 1024):
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        efficientnet_b3 = torchvision.models.efficientnet_b3(weights="IMAGENET1K_V1")
        self.backbone_features = efficientnet_b3.features
        original_conv = self.backbone_features[0][0]
        self.backbone_features[0][0] = nn.Conv2d(
            n_input_channels,
            original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=original_conv.bias,
        )
        with torch.no_grad():
            pretrained_weight = original_conv.weight
            new_weight = pretrained_weight.mean(dim=1, keepdim=True).repeat(
                1, n_input_channels, 1, 1
            )
            new_weight = new_weight / (n_input_channels / 3.0)
            self.backbone_features[0][0].weight.copy_(new_weight)
        self.attention_modules = nn.ModuleDict(
            {
                "stage_3": CBAM(48, reduction=8),
                "stage_5": CBAM(136, reduction=8),
                "stage_7": CBAM(384, reduction=8),
            }
        )
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.attention_1 = MultiHeadAttention(1536, num_heads=8, dropout=0.1)
        self.layer_norm_1 = nn.LayerNorm(1536)
        self.feature_processor = nn.Sequential(
            nn.Linear(1536 * 3, 1536),
            nn.SiLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(1536, 1024),
            nn.SiLU(inplace=True),
            nn.Dropout(0.1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.SiLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(512, features_dim),
            nn.SiLU(inplace=True),
        )
        logger.info(
            f"HIGH-PERFORMANCE EFFICIENTNET-B3: Input {observation_space.shape}, Output {features_dim}, Estimated VRAM ~12GB"
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        x = observations.float() / 255.0
        if torch.any(torch.isnan(x)) or torch.any(torch.isinf(x)):
            logger.warning(
                "Invalid observations in HighPerformanceEfficientNetB3FeatureExtractor"
            )
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device, dtype=x.dtype)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device, dtype=x.dtype)
        batch_size, channels, height, width = x.shape
        x_reshaped = x.view(batch_size, -1, 3, height, width)
        for i in range(3):
            x_reshaped[:, :, i] = (x_reshaped[:, :, i] - mean[i]) / std[i]
        x = x_reshaped.view(batch_size, channels, height, width)
        for i, layer in enumerate(self.backbone_features):
            x = layer(x)
            if i == 3 and x.shape[1] == 48:
                x = self.attention_modules["stage_3"](x)
            elif i == 5 and x.shape[1] == 136:
                x = self.attention_modules["stage_5"](x)
            elif i == 7 and x.shape[1] == 384:
                x = self.attention_modules["stage_7"](x)
        features = self.global_pool(x)
        features = self.flatten(features)
        attention_input = features.view(batch_size, 1, 1536)
        attended_1 = self.attention_1(attention_input)
        attended_1 = self.layer_norm_1(attended_1 + attention_input)
        final_features = attended_1.view(batch_size, -1)
        final_features = torch.cat([final_features] * 3, dim=1)
        processed_features = self.feature_processor(final_features)
        output = self.classifier(processed_features)
        return output


class EfficientNetB3FeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 512):
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        efficientnet_b3 = torchvision.models.efficientnet_b3(weights="IMAGENET1K_V1")
        self.backbone_features = efficientnet_b3.features
        original_conv = self.backbone_features[0][0]
        self.backbone_features[0][0] = nn.Conv2d(
            n_input_channels,
            original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=original_conv.bias,
        )
        with torch.no_grad():
            pretrained_weight = original_conv.weight
            new_weight = pretrained_weight.mean(dim=1, keepdim=True).repeat(
                1, n_input_channels, 1, 1
            )
            new_weight = new_weight / (n_input_channels / 3.0)
            self.backbone_features[0][0].weight.copy_(new_weight)
        self.attention_modules = nn.ModuleDict(
            {
                "stage_1": CBAM(24),
                "stage_3": CBAM(48),
                "stage_5": CBAM(136),
                "stage_7": CBAM(384),
            }
        )
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.attention = MultiHeadAttention(1536, num_heads=8)
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(1536, 768),
            nn.SiLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(768, features_dim),
            nn.SiLU(inplace=True),
        )
        self.backbone_features.requires_grad_(True)
        logger.info(
            f"PRE-TRAINED EFFICIENTNET-B3 + ATTENTION: Input {observation_space.shape}, Output {features_dim}, Gradient checkpointing enabled"
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        x = observations.float() / 255.0
        if torch.any(torch.isnan(x)) or torch.any(torch.isinf(x)):
            logger.warning("Invalid observations in EfficientNetB3FeatureExtractor")
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device, dtype=x.dtype)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device, dtype=x.dtype)
        batch_size, channels, height, width = x.shape
        x_reshaped = x.view(batch_size, -1, 3, height, width)
        for i in range(3):
            x_reshaped[:, :, i] = (x_reshaped[:, :, i] - mean[i]) / std[i]
        x = x_reshaped.view(batch_size, channels, height, width)
        # Apply gradient checkpointing in forward pass
        x = torch.utils.checkpoint.checkpoint_sequential(
            self.backbone_features, segments=2, input=x
        )
        # Apply attention modules after backbone
        for i, _ in enumerate(self.backbone_features):
            if i == 1 and x.shape[1] == 24:
                x = self.attention_modules["stage_1"](x)
            elif i == 3 and x.shape[1] == 48:
                x = self.attention_modules["stage_3"](x)
            elif i == 5 and x.shape[1] == 136:
                x = self.attention_modules["stage_5"](x)
            elif i == 7 and x.shape[1] == 384:
                x = self.attention_modules["stage_7"](x)
        features = self.global_pool(x)
        features = self.flatten(features)
        features_reshaped = features.unsqueeze(1)
        attended_features = self.attention(features_reshaped)
        attended_features = attended_features.squeeze(1)
        output = self.classifier(attended_features)
        return output


def make_samurai_env(
    game="SamuraiShodown-Genesis",
    state=None,
    reset_round=True,
    rendering=False,
    max_episode_steps=15000,
    **wrapper_kwargs,
):
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
    logger.info("Testing SamuraiShowdownCustomWrapper...")
    try:
        env = make_samurai_env(rendering=False)
        logger.info(f"Environment created successfully")
        obs, info = env.reset()
        logger.info(f"Reset observation shape: {obs.shape}")
        for i in range(5):
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)
            logger.info(
                f"Step {i+1}: obs_shape={obs.shape}, reward={reward:.3f}, done={done}"
            )
            if done:
                break
        logger.info(f"Final stats: {env.current_stats}")
        env.close()
        logger.info("Wrapper test completed successfully!")
    except Exception as e:
        logger.error(f"Wrapper test failed: {e}")
        import traceback

        traceback.print_exc()
