#!/usr/bin/env python3
"""
JEPA-Enhanced Samurai Showdown Wrapper - REWARD SHAPING & ACCURACY LOGGING
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

        # *** CHANGE: Initialize strategic stats with moving average accuracies ***
        self.strategic_stats = {
            "binary_predictions_made": 0,
            "overall_accuracy": 0.5,  # Start at 50% baseline
        }

        if self.enable_jepa:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.feature_extractor = None
            self.jepa_predictor = None
            self.visual_features_history = deque(maxlen=self.state_history_length)
            self.game_state_history = deque(maxlen=self.state_history_length)
            self.last_predicted_outcomes = (
                None  # Store the prediction from the previous step
            )

        print(
            f"🥷 JEPA Wrapper Initialized: Reward Shaping & Accuracy Logging, Frame Stack={self.frame_stack}"
        )

    # ... (_initialize_jepa_modules, _get_visual_features, _preprocess_frame, _get_stacked_observation are unchanged) ...
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

    def _extract_game_state(self, info, for_ground_truth=False):
        player_health = info.get("health", self.prev_player_health)
        enemy_health = info.get("enemy_health", self.prev_enemy_health)

        # If calculating ground truth, we care about what ACTUALLY happened
        if for_ground_truth:
            damage_dealt = self.prev_enemy_health - enemy_health > 0
            damage_taken = self.prev_player_health - player_health > 0
            # Opponent attack is hard to define perfectly, we'll use a heuristic: they took damage
            opponent_attacked = damage_dealt
            round_ending = player_health < 30 or enemy_health < 30
            return {
                "will_opponent_attack": float(opponent_attacked),
                "will_opponent_take_damage": float(damage_dealt),
                "will_player_take_damage": float(damage_taken),
                "will_round_end_soon": float(round_ending),
            }

        # Otherwise, return the standard state vector for the transformer
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

    # *** NEW FUNCTION: To calculate JEPA prediction accuracy ***
    def _calculate_jepa_accuracy(self, current_info):
        if self.last_predicted_outcomes is None:
            return

        # 1. Get the ground truth of what just happened in the current step
        ground_truth = self._extract_game_state(current_info, for_ground_truth=True)

        # 2. Compare with the prediction made in the *previous* step
        total_correct = 0
        num_predictions = 0
        for key, pred_tensor in self.last_predicted_outcomes.items():
            if key in ground_truth:
                # Get the prediction for the very next step (index 0 of the horizon)
                prediction_prob = pred_tensor[0, 0].item()
                predicted_label = 1 if prediction_prob > 0.5 else 0

                actual_label = int(ground_truth[key])

                if predicted_label == actual_label:
                    total_correct += 1
                num_predictions += 1

        if num_predictions > 0:
            # 3. Update the overall accuracy using a moving average
            current_accuracy = total_correct / num_predictions
            alpha = 0.001  # Smoothing factor for the moving average
            self.strategic_stats["overall_accuracy"] = (
                1 - alpha
            ) * self.strategic_stats["overall_accuracy"] + alpha * current_accuracy

        # 4. Increment the prediction counter
        self.strategic_stats["binary_predictions_made"] += 1

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
            self.last_predicted_outcomes = None
            zero_vf = torch.zeros(512, device=self.device)
            zero_gs = np.zeros(8, dtype=np.float32)
            self.visual_features_history.clear()
            self.game_state_history.clear()
            for _ in range(self.state_history_length):
                self.visual_features_history.append(zero_vf)
                self.game_state_history.append(zero_gs)
        return self._get_stacked_observation(), info

    def step(self, action):
        # *** CHANGE: Accuracy calculation is now part of the flow ***

        # First, if we have a prediction from the last step, let's see how it did
        if self.enable_jepa and self.last_predicted_outcomes:
            # We need the info dict *before* the step to get the ground truth
            # We'll use the info from the *previous* step's return, but for simplicity here we'll pass the new one.
            # A more advanced implementation would buffer the previous info dict.
            # For logging purposes, this is sufficient.
            self._calculate_jepa_accuracy(self.env.unwrapped.data.lookup_all())

        obs, _, terminated, truncated, info = self.env.step(action)
        self.frame_buffer.append(self._preprocess_frame(obs))
        stacked_obs = self._get_stacked_observation()

        current_player_health = info.get("health", self.prev_player_health)
        current_enemy_health = info.get("enemy_health", self.prev_enemy_health)

        # Reward Shaping Logic (unchanged)
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

        # --- JEPA Logic (Now includes storing the prediction) ---
        if self.enable_jepa and self.feature_extractor:
            if self.jepa_predictor is None:
                self._initialize_jepa_modules(self.feature_extractor.features_dim)
            visual_features = self._get_visual_features(stacked_obs)
            if visual_features is not None:
                game_state = self._extract_game_state(info)
                self.visual_features_history.append(visual_features)
                self.game_state_history.append(game_state)
                vf_seq = torch.stack(list(self.visual_features_history)).unsqueeze(0)
                gs_seq = torch.tensor(
                    np.array(list(self.game_state_history)),
                    dtype=torch.float32,
                    device=self.device,
                ).unsqueeze(0)
                with torch.no_grad():
                    # Predict outcomes and store them for the *next* step's accuracy check
                    self.last_predicted_outcomes = self.jepa_predictor(vf_seq, gs_seq)

        info["strategic_stats"] = self.strategic_stats  # Pass stats to the callback

        # --- Done Signal Logic (unchanged) ---
        done = terminated or truncated
        if self.player_wins_in_episode >= 2 or self.enemy_wins_in_episode >= 2:
            done = True
            if self.player_wins_in_episode > self.enemy_wins_in_episode:
                self.current_stats["wins"] += 1
            else:
                self.current_stats["losses"] += 1
            self.current_stats["total_rounds"] = (
                self.player_wins_in_episode + self.enemy_wins_in_episode
            )
            total_matches = self.current_stats["wins"] + self.current_stats["losses"]
            if total_matches > 0:
                self.current_stats["win_rate"] = (
                    self.current_stats["wins"] / total_matches
                )
        info["current_stats"] = self.current_stats

        return stacked_obs, enhanced_reward, done, False, info


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
