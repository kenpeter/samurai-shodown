#!/usr/bin/env python3
"""
JEPA-Enhanced Samurai Showdown Wrapper - FIXED PREDICTION TIMING
"""
import math
import cv2
import torch
import numpy as np
import gymnasium as gym
import torch.nn as nn
from collections import deque
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from typing import Dict


# --- Transformer Components (No changes) ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


class JEPASimpleBinaryPredictor(nn.Module):
    def __init__(
        self, visual_dim=512, sequence_length=8, prediction_horizon=8, game_state_dim=8
    ):
        super().__init__()
        self.prediction_horizon = prediction_horizon
        self.binary_outcomes = [
            "will_opponent_attack",
            "will_opponent_take_damage",
            "will_player_take_damage",
            "will_round_end_soon",
        ]
        feature_dim = 128
        self.context_encoder = nn.Linear(visual_dim, feature_dim)
        self.game_state_encoder = nn.Linear(game_state_dim, feature_dim)
        d_model = feature_dim * 2
        self.pos_encoder = PositionalEncoding(d_model, dropout=0.1)
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=4,
            dim_feedforward=d_model * 2,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=3)
        self.binary_predictors = nn.ModuleDict(
            {
                outcome: nn.Sequential(
                    nn.Linear(d_model, 64), nn.ReLU(), nn.Linear(64, prediction_horizon)
                )
                for outcome in self.binary_outcomes
            }
        )

    def forward(self, visual_features_seq, game_state_features_seq):
        visual_context = self.context_encoder(visual_features_seq)
        game_context = self.game_state_encoder(game_state_features_seq)
        sequence_input = torch.cat([visual_context, game_context], dim=-1)
        sequence_input = self.pos_encoder(sequence_input)
        transformer_out = self.transformer_encoder(sequence_input)
        final_representation = transformer_out[:, -1, :]
        binary_predictions = {
            outcome: torch.sigmoid(predictor(final_representation))
            for outcome, predictor in self.binary_predictors.items()
        }
        return binary_predictions


# --- Main Environment Wrapper ---
class SamuraiJEPAWrapper(gym.Wrapper):
    def __init__(self, env, frame_stack=8, enable_jepa=True, **kwargs):
        super().__init__(env)
        self.frame_stack = frame_stack
        self.enable_jepa = enable_jepa
        self.state_history_length = 8
        self.target_size = (128, 180)

        obs_shape = (3 * frame_stack, self.target_size[1], self.target_size[0])
        self.observation_space = spaces.Box(
            low=0, high=255, shape=obs_shape, dtype=np.uint8
        )

        self.frame_buffer = deque(maxlen=self.frame_stack)
        self.prev_player_health = 176
        self.prev_enemy_health = 176
        self.player_wins_in_episode = 0
        self.enemy_wins_in_episode = 0

        self.current_stats = {"wins": 0, "losses": 0}
        self.strategic_stats = {"binary_predictions_made": 0, "overall_accuracy": 0.5}

        # FIXED: Add ready flag to track JEPA system status
        self.jepa_ready = False

        if self.enable_jepa:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.feature_extractor = None
            self.jepa_predictor = None
            self.visual_features_history = deque(maxlen=self.state_history_length)
            self.game_state_history = deque(maxlen=self.state_history_length)
            self.prediction_from_last_step = None

        print(
            f"🥷 JEPA Wrapper Initialized: FIXED PREDICTION TIMING, Frame Stack={self.frame_stack}"
        )

    def inject_feature_extractor(self, feature_extractor):
        """FIXED: Explicit method to inject feature extractor and initialize JEPA"""
        if self.enable_jepa:
            self.feature_extractor = feature_extractor
            # Initialize JEPA predictor immediately
            self._initialize_jepa_modules(feature_extractor.features_dim)
            self.jepa_ready = True
            print("   ✅ Feature extractor injected and JEPA predictor initialized!")

    def _initialize_jepa_modules(self, visual_dim):
        if self.jepa_predictor is None:
            self.jepa_predictor = JEPASimpleBinaryPredictor(
                visual_dim=visual_dim,
                sequence_length=self.state_history_length,
                prediction_horizon=self.frame_stack,
                game_state_dim=8,
            ).to(self.device)
            print(f"   🧠 JEPA Predictor initialized with visual_dim={visual_dim}")

    def _get_visual_features(self, observation):
        if self.feature_extractor is None:
            return None
        try:
            with torch.no_grad():
                obs_tensor = torch.tensor(
                    observation, dtype=torch.float32, device=self.device
                ).unsqueeze(0)
                return self.feature_extractor(obs_tensor).squeeze(0)
        except Exception as e:
            print(f"   ⚠️ Visual feature extraction failed: {e}")
            return torch.zeros(self.feature_extractor.features_dim, device=self.device)

    def _preprocess_frame(self, frame):
        return cv2.resize(frame, self.target_size, interpolation=cv2.INTER_AREA)

    def _get_stacked_observation(self):
        return np.concatenate(list(self.frame_buffer), axis=2).transpose(2, 0, 1)

    def _extract_game_state_vector(self, info):
        player_health = info.get("health", self.prev_player_health)
        enemy_health = info.get("enemy_health", self.prev_enemy_health)
        return np.array(
            [
                player_health / 176.0,
                enemy_health / 176.0,
                (player_health - enemy_health) / 176.0,
                info.get("round", 1) / 3.0,
                info.get("timer", 99) / 99.0,
                1.0 if player_health < 50 else 0.0,
                1.0 if enemy_health < 50 else 0.0,
                1.0,  # padding to make it 8-dimensional
            ],
            dtype=np.float32,
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.prev_player_health = info.get("health", 176)
        self.prev_enemy_health = info.get("enemy_health", 176)
        self.player_wins_in_episode = 0
        self.enemy_wins_in_episode = 0
        processed_frame = self._preprocess_frame(obs)
        self.frame_buffer.clear()
        for _ in range(self.frame_stack):
            self.frame_buffer.append(processed_frame)

        if self.enable_jepa and self.jepa_ready:
            self.prediction_from_last_step = None
            # Initialize with proper dimensions
            zero_vf = torch.zeros(
                self.feature_extractor.features_dim, device=self.device
            )
            zero_gs = np.zeros(8, dtype=np.float32)
            self.visual_features_history.clear()
            self.game_state_history.clear()
            for _ in range(self.state_history_length):
                self.visual_features_history.append(zero_vf)
                self.game_state_history.append(zero_gs)

        return self._get_stacked_observation(), info

    def step(self, action):
        # FIXED: Only make predictions if JEPA system is fully ready
        current_stacked_obs = self._get_stacked_observation()

        # --- 1. Make prediction if JEPA is ready ---
        if self.enable_jepa and self.jepa_ready:
            visual_features = self._get_visual_features(current_stacked_obs)
            if visual_features is not None:
                current_info = self.env.unwrapped.data.lookup_all()
                game_state_vec = self._extract_game_state_vector(current_info)
                self.visual_features_history.append(visual_features)
                self.game_state_history.append(game_state_vec)

                # Stack sequences for transformer
                vf_seq = torch.stack(list(self.visual_features_history)).unsqueeze(0)
                gs_seq = torch.tensor(
                    np.array(list(self.game_state_history)),
                    dtype=torch.float32,
                    device=self.device,
                ).unsqueeze(0)

                try:
                    with torch.no_grad():
                        self.prediction_from_last_step = self.jepa_predictor(
                            vf_seq, gs_seq
                        )
                except Exception as e:
                    print(f"   ⚠️ JEPA prediction failed: {e}")
                    self.prediction_from_last_step = None

        # --- 2. Step the environment ---
        obs, _, terminated, truncated, next_info = self.env.step(action)
        self.frame_buffer.append(self._preprocess_frame(obs))

        # --- 3. Evaluate prediction accuracy if we made one ---
        if self.prediction_from_last_step is not None:
            player_health_after = next_info.get("health", self.prev_player_health)
            enemy_health_after = next_info.get("enemy_health", self.prev_enemy_health)

            # Calculate actual outcomes
            ground_truth = {
                "will_opponent_attack": float(
                    self.prev_player_health - player_health_after > 1
                ),
                "will_opponent_take_damage": float(
                    self.prev_enemy_health - enemy_health_after > 1
                ),
                "will_player_take_damage": float(
                    self.prev_player_health - player_health_after > 1
                ),
                "will_round_end_soon": float(
                    player_health_after < 30 or enemy_health_after < 30
                ),
            }

            # Calculate accuracy
            total_correct, num_predictions = 0, 0
            for key, pred_tensor in self.prediction_from_last_step.items():
                if key in ground_truth:
                    predicted_prob = pred_tensor[0, 0].item()
                    predicted_label = 1 if predicted_prob > 0.5 else 0
                    actual_label = int(ground_truth[key])

                    if predicted_label == actual_label:
                        total_correct += 1
                    num_predictions += 1

            # Update running accuracy with exponential moving average
            if num_predictions > 0:
                current_accuracy = total_correct / num_predictions
                alpha = 0.001  # Learning rate for running average
                self.strategic_stats["overall_accuracy"] = (
                    1 - alpha
                ) * self.strategic_stats["overall_accuracy"] + alpha * current_accuracy
                self.strategic_stats["binary_predictions_made"] += 1

        # --- 4. Calculate Enhanced Reward ---
        current_player_health = next_info.get("health", self.prev_player_health)
        current_enemy_health = next_info.get("enemy_health", self.prev_enemy_health)

        damage_dealt = self.prev_enemy_health - current_enemy_health
        damage_taken = self.prev_player_health - current_player_health
        health_delta_reward = (damage_dealt * 1.0 - damage_taken * 1.2) * 0.1
        time_reward = 0.001
        win_loss_reward = 0.0

        # Check for round end
        if (current_player_health <= 0 or current_enemy_health <= 0) and (
            self.prev_player_health > 0 and self.prev_enemy_health > 0
        ):
            if current_player_health > current_enemy_health:
                win_loss_reward = 20.0
                self.player_wins_in_episode += 1
            else:
                win_loss_reward = -20.0
                self.enemy_wins_in_episode += 1

        enhanced_reward = health_delta_reward + time_reward + win_loss_reward

        self.prev_player_health = current_player_health
        self.prev_enemy_health = current_enemy_health

        # --- 5. Episode termination logic ---
        done = terminated or truncated
        if self.player_wins_in_episode >= 2 or self.enemy_wins_in_episode >= 2:
            done = True
            if self.player_wins_in_episode > self.enemy_wins_in_episode:
                self.current_stats["wins"] += 1
            else:
                self.current_stats["losses"] += 1

            total_matches = self.current_stats["wins"] + self.current_stats["losses"]
            if total_matches > 0:
                self.current_stats["win_rate"] = (
                    self.current_stats["wins"] / total_matches
                )

        next_info["current_stats"] = self.current_stats
        next_info["strategic_stats"] = self.strategic_stats

        return self._get_stacked_observation(), enhanced_reward, done, False, next_info


class JEPAEnhancedCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 512):
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            n_flatten = self.cnn(
                torch.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]
        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations / 255.0))
