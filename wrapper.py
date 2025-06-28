#!/usr/bin/env python3
"""
JEPA-Enhanced Samurai Showdown Wrapper - ACCURACY LOGGING FIX
"""
import math
import cv2
import torch
import numpy as np
import gymnasium as gym
import torch.nn as nn
from collections import deque
from gymnasium import spaces
import torch.nn.functional as F
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from typing import Dict, Tuple, Optional, List


# --- Transformer Components (No changes) ---
class PositionalEncoding(nn.Module):
    # ... (code is unchanged)
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
    # ... (code is unchanged)
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
        self.num_binary_outcomes = len(self.binary_outcomes)
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
        print(
            f"🔮 JEPA Transformer Predictor initialized: Horizon={prediction_horizon}, SeqLen={sequence_length}"
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
        self.prediction_horizon = frame_stack
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

        self.current_stats = {
            "wins": 0,
            "losses": 0,
            "total_rounds": 0,
            "win_rate": 0.0,
        }
        self.strategic_stats = {"binary_predictions_made": 0, "overall_accuracy": 0.5}

        if self.enable_jepa:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.feature_extractor = None
            self.jepa_predictor = None
            self.visual_features_history = deque(maxlen=self.state_history_length)
            self.game_state_history = deque(maxlen=self.state_history_length)

        print(
            f"🥷 JEPA Wrapper Initialized: Reward Shaping & ACCURACY FIX, Frame Stack={self.frame_stack}"
        )

    def _initialize_jepa_modules(self, visual_dim):
        if self.jepa_predictor is None:
            self.jepa_predictor = JEPASimpleBinaryPredictor(
                visual_dim=visual_dim,
                sequence_length=self.state_history_length,
                prediction_horizon=self.prediction_horizon,
                game_state_dim=8,
            ).to(self.device)
            print(
                f"✅ JEPA modules initialized on device '{self.device}' with visual_dim={visual_dim}"
            )

    def _get_visual_features(self, observation):
        if self.feature_extractor is None:
            return None
        try:
            with torch.no_grad():
                obs_tensor = torch.tensor(
                    observation, dtype=torch.float32, device=self.device
                ).unsqueeze(0)
                features = self.feature_extractor(obs_tensor)
                return features.squeeze(0)
        except Exception:
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
                1.0 if abs(player_health - enemy_health) < 20 else 0.0,
            ],
            dtype=np.float32,
        )

    def _get_ground_truth(self, current_info):
        player_health = current_info.get("health", self.prev_player_health)
        enemy_health = current_info.get("enemy_health", self.prev_enemy_health)
        damage_dealt = self.prev_enemy_health - enemy_health > 1
        damage_taken = self.prev_player_health - player_health > 1
        opponent_attacked = damage_dealt  # Simple heuristic
        round_ending = player_health < 30 or enemy_health < 30
        return {
            "will_opponent_attack": float(opponent_attacked),
            "will_opponent_take_damage": float(damage_dealt),
            "will_player_take_damage": float(damage_taken),
            "will_round_end_soon": float(round_ending),
        }

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
        if self.enable_jepa:
            zero_vf = torch.zeros(512, device=self.device)
            zero_gs = np.zeros(8, dtype=np.float32)
            self.visual_features_history.clear()
            self.game_state_history.clear()
            for _ in range(self.state_history_length):
                self.visual_features_history.append(zero_vf)
                self.game_state_history.append(zero_gs)
        return self._get_stacked_observation(), info

    def step(self, action):
        # *** NEW, ROBUST LOGIC FLOW ***

        # --- 1. PREDICT (based on current state) ---
        predicted_outcomes = None
        if self.enable_jepa and self.feature_extractor:
            if self.jepa_predictor is None:
                self._initialize_jepa_modules(self.feature_extractor.features_dim)

            # We need to get the feature vector for the *current* observation before we step
            current_stacked_obs = self._get_stacked_observation()
            visual_features = self._get_visual_features(current_stacked_obs)

            if visual_features is not None:
                current_info = self.env.unwrapped.data.lookup_all()
                game_state_vec = self._extract_game_state_vector(current_info)

                # Update history buffers
                self.visual_features_history.append(visual_features)
                self.game_state_history.append(game_state_vec)

                # Make prediction
                vf_seq = torch.stack(list(self.visual_features_history)).unsqueeze(0)
                gs_seq = torch.tensor(
                    np.array(list(self.game_state_history)),
                    dtype=torch.float32,
                    device=self.device,
                ).unsqueeze(0)
                with torch.no_grad():
                    predicted_outcomes = self.jepa_predictor(vf_seq, gs_seq)

        # --- 2. STEP THE ENVIRONMENT ---
        obs, _, terminated, truncated, next_info = self.env.step(action)
        self.frame_buffer.append(self._preprocess_frame(obs))
        stacked_obs = self._get_stacked_observation()

        # --- 3. CALCULATE ACCURACY (Compare prediction from state N with outcome in state N+1) ---
        if predicted_outcomes:
            ground_truth = self._get_ground_truth(next_info)
            total_correct, num_predictions = 0, 0
            for key, pred_tensor in predicted_outcomes.items():
                if key in ground_truth:
                    predicted_label = 1 if pred_tensor[0, 0].item() > 0.5 else 0
                    actual_label = int(ground_truth[key])
                    if predicted_label == actual_label:
                        total_correct += 1
                    num_predictions += 1
            if num_predictions > 0:
                current_accuracy = total_correct / num_predictions
                alpha = 0.001  # Moving average smoothing
                self.strategic_stats["overall_accuracy"] = (
                    1 - alpha
                ) * self.strategic_stats["overall_accuracy"] + alpha * current_accuracy
            self.strategic_stats["binary_predictions_made"] += 1

        # --- 4. CALCULATE REWARD ---
        current_player_health = next_info.get("health", self.prev_player_health)
        current_enemy_health = next_info.get("enemy_health", self.prev_enemy_health)
        damage_dealt = self.prev_enemy_health - current_enemy_health
        damage_taken = self.prev_player_health - current_player_health
        health_delta_reward = (damage_dealt * 1.0 - damage_taken * 1.2) * 0.1
        time_reward = 0.001
        win_loss_reward = 0.0
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

        # --- 5. BUNDLE INFO AND DONE SIGNAL ---
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

        # Pass all relevant info to the callback
        next_info["current_stats"] = self.current_stats
        next_info["strategic_stats"] = self.strategic_stats

        return stacked_obs, enhanced_reward, done, False, next_info


# (JEPAEnhancedCNN class remains unchanged)
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
        print(
            f"🧠 JEPA-Enhanced CNN created: {n_input_channels} channels -> {features_dim} features."
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations / 255.0))
