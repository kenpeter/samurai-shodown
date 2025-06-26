#!/usr/bin/env python3
"""
JEPA-Enhanced Samurai Showdown Wrapper - FIXED & UPGRADED VERSION

Key Fixes & Upgrades:
- Replaced LSTM with a Transformer Encoder in the JEPA predictor for superior temporal pattern recognition.
- Eliminated all harmful randomness:
    - Feature extraction fallbacks now use deterministic zeros instead of torch.randn.
    - JEPA state history is initialized with zeros for a stable starting point.
- Preserved beneficial randomness (PPO exploration, dropout).
- Code is streamlined for clarity and stability.
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

# --- Transformer Components for JEPA Predictor ---


class PositionalEncoding(nn.Module):
    """Adds positional information to the input sequence for the Transformer."""

    # pass self, model, drop 0.1, max len 5000
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        # init super
        super().__init__()
        # drop out
        self.dropout = nn.Dropout(p=dropout)

        # position, torch array range, unsqueeze insert at index 1. [5] -> [5, 1] (rows, cols), so it is col vector
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


# --- JEPA & Strategic Planner Models ---


class JEPASimpleBinaryPredictor(nn.Module):
    """
    JEPA-based binary outcome predictor using a Transformer Encoder.
    Predicts 4 simple binary outcomes based on a sequence of game states.
    - will_opponent_attack
    - will_opponent_take_damage
    - will_player_take_damage
    - will_round_end_soon
    """

    def __init__(
        self, visual_dim=512, sequence_length=8, prediction_horizon=6, game_state_dim=8
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

        # Feature encoders
        feature_dim = 128
        self.context_encoder = nn.Linear(visual_dim, feature_dim)
        self.game_state_encoder = nn.Linear(game_state_dim, feature_dim)

        # Transformer for temporal sequence modeling
        d_model = feature_dim * 2  # Combined context and game state

        # only encode position
        self.pos_encoder = PositionalEncoding(d_model, dropout=0.1)

        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=4,
            dim_feedforward=d_model * 2,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=3)

        # Prediction heads
        self.binary_predictors = nn.ModuleDict(
            {
                outcome: nn.Sequential(
                    nn.Linear(d_model, 64), nn.ReLU(), nn.Linear(64, prediction_horizon)
                )
                for outcome in self.binary_outcomes
            }
        )
        print(f"🔮 JEPA Transformer Predictor initialized:")
        print(f"   🧠 Model: Transformer Encoder (3 layers, 4 heads)")
        print(f"   🔮 Prediction horizon: {prediction_horizon}")
        print(f"   🎯 Binary outcomes: {self.num_binary_outcomes}")

    def forward(self, visual_features_seq, game_state_features_seq):
        """
        Args:
            visual_features_seq: (batch, seq_len, visual_dim)
            game_state_features_seq: (batch, seq_len, game_state_dim)
        """
        # Encode features for each step in the sequence
        visual_context = self.context_encoder(visual_features_seq)
        game_context = self.game_state_encoder(game_state_features_seq)

        # Combine features and add positional encoding
        sequence_input = torch.cat([visual_context, game_context], dim=-1)
        sequence_input = self.pos_encoder(sequence_input)

        # Process sequence with Transformer
        transformer_out = self.transformer_encoder(sequence_input)

        # Use the representation of the last element for prediction
        final_representation = transformer_out[:, -1, :]

        # Generate predictions
        binary_predictions = {
            outcome: torch.sigmoid(predictor(final_representation))
            for outcome, predictor in self.binary_predictors.items()
        }

        return binary_predictions


class SimpleBinaryResponsePlanner(nn.Module):
    """Plans agent responses based on binary outcome predictions."""

    def __init__(self, visual_dim=512, game_state_dim=8, agent_action_dim=12):
        super().__init__()
        self.num_binary_outcomes = 4

        # Input processing
        self.situation_analyzer = nn.Sequential(
            nn.Linear(visual_dim + game_state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )

        # Strategy generator
        self.strategy_generator = nn.Sequential(
            nn.Linear(128 + self.num_binary_outcomes, 256),
            nn.ReLU(),
            nn.Linear(256, agent_action_dim),
        )
        print("⚔️ Simple Binary Response Planner initialized.")

    def plan_response(self, visual_features, game_state, binary_predictions):
        """Plan response based on current state and predictions."""
        # Analyze current situation
        situation_input = torch.cat([visual_features, game_state], dim=-1)
        situation_features = self.situation_analyzer(situation_input)

        # Use the first timestep of the prediction horizon
        pred_probs = torch.cat(
            [preds[:, 0].unsqueeze(1) for preds in binary_predictions.values()], dim=1
        )

        # Combine situation and predictions to form a strategy
        strategy_input = torch.cat([situation_features, pred_probs], dim=-1)
        response_logits = self.strategy_generator(strategy_input)

        return response_logits


# --- Main Environment Wrapper ---


class SamuraiJEPAWrapper(gym.Wrapper):
    """
    Enhanced Samurai Showdown wrapper with JEPA-based opponent state prediction.
    FIXED VERSION: No harmful randomness.
    """

    def __init__(self, env, frame_stack=6, enable_jepa=True, **kwargs):
        super().__init__(env)
        self.frame_stack = frame_stack
        self.enable_jepa = enable_jepa
        self.prediction_horizon = frame_stack
        self.state_history_length = 8
        self.target_size = (128, 180)  # WxH

        # Observation space for stacked frames (C, H, W)
        obs_shape = (3 * frame_stack, self.target_size[1], self.target_size[0])
        self.observation_space = spaces.Box(
            low=0, high=255, shape=obs_shape, dtype=np.uint8
        )

        # State tracking
        self.frame_buffer = deque(maxlen=self.frame_stack)
        self.prev_player_health = 176
        self.prev_enemy_health = 176

        # Stats
        self.current_stats = {
            "wins": 0,
            "losses": 0,
            "total_rounds": 0,
            "win_rate": 0.0,
        }
        self.strategic_stats = {
            "binary_predictions_made": 0,
            "overall_prediction_accuracy": 0.0,
            "attack_prediction_accuracy": 0.0,
            "damage_prediction_accuracy": 0.0,
        }

        # JEPA Components
        if self.enable_jepa:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            # These are initialized on first use to get correct dimensions from the SB3 model
            self.feature_extractor = None
            self.jepa_predictor = None
            self.response_planner = None
            # History buffers for the transformer
            self.visual_features_history = deque(maxlen=self.state_history_length)
            self.game_state_history = deque(maxlen=self.state_history_length)
            self.predicted_binary_outcomes = None

        print(f"🥷 JEPA-ENHANCED SAMURAI WRAPPER (FIXED & UPGRADED):")
        print(f"   🧠 JEPA enabled: {self.enable_jepa}")
        if self.enable_jepa:
            print(f"   🤖 Predictor Model: Transformer Encoder")
        print(f"   ✅ HARMFUL RANDOMNESS: ELIMINATED")

    def _initialize_jepa_modules(self, visual_dim):
        """Initialize JEPA modules lazily once we have the visual feature dimension."""
        if self.jepa_predictor is None:
            self.jepa_predictor = JEPASimpleBinaryPredictor(
                visual_dim=visual_dim,
                sequence_length=self.state_history_length,
                prediction_horizon=self.prediction_horizon,
                game_state_dim=8,
            ).to(self.device)
            self.response_planner = SimpleBinaryResponsePlanner(
                visual_dim=visual_dim,
                game_state_dim=8,
                agent_action_dim=self.action_space.n,
            ).to(self.device)
            print(
                f"✅ JEPA modules initialized on device '{self.device}' with visual_dim={visual_dim}"
            )

    def _get_visual_features(self, observation):
        """Extract visual features using the PPO model's CNN feature extractor."""
        if self.feature_extractor is None:
            # This is a bit of a hack: we grab the feature extractor from the parent PPO model
            # This will be set from the training script after the model is created.
            return None

        try:
            with torch.no_grad():
                obs_tensor = torch.tensor(
                    observation, dtype=torch.float32, device=self.device
                ).unsqueeze(0)
                features = self.feature_extractor(obs_tensor)
                return features.squeeze(0)
        except Exception as e:
            print(
                f"⚠️ Visual feature extraction error: {e}. Using deterministic fallback."
            )
            # **KEY FIX**: Return deterministic zeros instead of random noise.
            # This prevents the model from learning on garbage data during errors.
            return torch.zeros(self.feature_extractor.features_dim, device=self.device)

    def _preprocess_frame(self, frame):
        """Resize and convert frame."""
        frame = cv2.resize(frame, self.target_size, interpolation=cv2.INTER_AREA)
        return frame  # H, W, C

    def _get_stacked_observation(self):
        """Stack frames from buffer and transpose to (C, H, W)."""
        stacked_frames = list(self.frame_buffer)
        # Transpose from (F, H, W, C) to (C, H, W)
        return np.concatenate(stacked_frames, axis=2).transpose(2, 0, 1)

    def _extract_game_state(self, info):
        """Extract a simple game state vector."""
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

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        # Reset state
        self.prev_player_health = info.get("health", 176)
        self.prev_enemy_health = info.get("enemy_health", 176)

        # Pre-fill buffers with the first frame
        processed_frame = self._preprocess_frame(obs)
        for _ in range(self.frame_stack):
            self.frame_buffer.append(processed_frame)

        if self.enable_jepa:
            # **KEY FIX**: Initialize history with deterministic zeros.
            zero_vf = torch.zeros(512, device=self.device)  # Assuming features_dim=512
            zero_gs = np.zeros(8, dtype=np.float32)
            self.visual_features_history.clear()
            self.game_state_history.clear()
            for _ in range(self.state_history_length):
                self.visual_features_history.append(zero_vf)
                self.game_state_history.append(zero_gs)

        return self._get_stacked_observation(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Process and stack frame
        self.frame_buffer.append(self._preprocess_frame(obs))
        stacked_obs = self._get_stacked_observation()

        # Extract game state and calculate reward
        current_player_health = info.get("health", self.prev_player_health)
        current_enemy_health = info.get("enemy_health", self.prev_enemy_health)

        damage_dealt = self.prev_enemy_health - current_enemy_health
        damage_taken = self.prev_player_health - current_player_health

        # better damage, so
        enhanced_reward = (damage_dealt - damage_taken) * 0.1

        self.prev_player_health = current_player_health
        self.prev_enemy_health = current_enemy_health

        # JEPA prediction logic
        if self.enable_jepa and self.feature_extractor:
            # Lazily initialize modules if they don't exist
            if self.jepa_predictor is None:
                self._initialize_jepa_modules(self.feature_extractor.features_dim)

            visual_features = self._get_visual_features(stacked_obs)
            game_state = self._extract_game_state(info)

            # Update history
            self.visual_features_history.append(visual_features)
            self.game_state_history.append(game_state)

            # Prepare sequences for the transformer
            vf_seq = torch.stack(list(self.visual_features_history)).unsqueeze(0)
            gs_seq = torch.tensor(
                np.array(list(self.game_state_history)),
                dtype=torch.float32,
                device=self.device,
            ).unsqueeze(0)

            with torch.no_grad():
                self.predicted_binary_outcomes = self.jepa_predictor(vf_seq, gs_seq)

            # Add JEPA info for logging/callbacks
            info["jepa_enabled"] = True
            info["strategic_stats"] = self.strategic_stats
            if self.predicted_binary_outcomes:
                info["predicted_outcomes_cpu"] = {
                    k: v.cpu().numpy()
                    for k, v in self.predicted_binary_outcomes.items()
                }

        # Win/loss tracking
        done = terminated or truncated
        if current_enemy_health <= 0 or current_player_health <= 0:
            done = True
            self.current_stats["total_rounds"] += 1
            if current_player_health > current_enemy_health:
                self.current_stats["wins"] += 1
            else:
                self.current_stats["losses"] += 1
            if self.current_stats["total_rounds"] > 0:
                self.current_stats["win_rate"] = (
                    self.current_stats["wins"] / self.current_stats["total_rounds"]
                )

        info["current_stats"] = self.current_stats

        return stacked_obs, enhanced_reward, done, False, info


class JEPAEnhancedCNN(BaseFeaturesExtractor):
    """
    CNN Feature Extractor for JEPA. This is the main network used by the PPO agent.
    """

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
