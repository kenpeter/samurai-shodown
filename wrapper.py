#!/usr/bin/env python3
"""
JEPA-Enhanced Samurai Showdown Wrapper - COMPREHENSIVE FIXES
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
from typing import Dict, Optional, Tuple


# --- Constants ---
MAX_HEALTH = 176
PREDICTION_THRESHOLD = 0.5
ACCURACY_LEARNING_RATE = 0.001

# Individual thresholds for better prediction accuracy
PREDICTION_THRESHOLDS = {
    "will_opponent_attack": 0.5,
    "will_opponent_take_damage": 0.5,
    "will_player_take_damage": 0.2,  # Lower threshold since model is too conservative
    "will_round_end_soon": 0.4,  # Slightly lower threshold
}


# --- Transformer Components ---
class PositionalEncoding(nn.Module):
    """Positional encoding for transformer with improved stability"""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        # Fixed: Correct dimensions - (max_len, d_model) not (max_len, 1, d_model)
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Add batch dimension for broadcasting: (1, max_len, d_model)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, d_model)
        # pe shape: (1, max_len, d_model)
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)


class JEPASimpleBinaryPredictor(nn.Module):
    """Enhanced JEPA predictor with improved architecture"""

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

        # Enhanced feature encoding
        feature_dim = 128
        self.context_encoder = nn.Sequential(
            nn.Linear(visual_dim, feature_dim), nn.LayerNorm(feature_dim), nn.ReLU()
        )

        self.game_state_encoder = nn.Sequential(
            nn.Linear(game_state_dim, feature_dim), nn.LayerNorm(feature_dim), nn.ReLU()
        )

        # Transformer architecture
        d_model = feature_dim * 2
        self.pos_encoder = PositionalEncoding(d_model, dropout=0.1)

        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=4,
            dim_feedforward=d_model * 2,
            dropout=0.1,
            batch_first=True,
            norm_first=True,  # Pre-norm for better stability
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=3)

        # Improved prediction heads
        self.binary_predictors = nn.ModuleDict(
            {
                outcome: nn.Sequential(
                    nn.Linear(d_model, 64),
                    nn.LayerNorm(64),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(64, prediction_horizon),
                )
                for outcome in self.binary_outcomes
            }
        )

    def forward(self, visual_features_seq, game_state_features_seq):
        """Forward pass with improved error handling and dimension checking"""
        try:
            batch_size, seq_len = visual_features_seq.shape[:2]

            # Encode features
            visual_context = self.context_encoder(visual_features_seq)
            game_context = self.game_state_encoder(game_state_features_seq)

            # Combine and process - shape: (batch_size, seq_len, d_model)
            sequence_input = torch.cat([visual_context, game_context], dim=-1)

            # Apply positional encoding - expects (batch_size, seq_len, d_model)
            sequence_input = self.pos_encoder(sequence_input)

            # Transformer expects batch_first=True format
            transformer_out = self.transformer_encoder(sequence_input)

            # Get final representation from last timestep
            final_representation = transformer_out[:, -1, :]  # (batch_size, d_model)

            # Generate predictions
            binary_predictions = {
                outcome: torch.sigmoid(predictor(final_representation))
                for outcome, predictor in self.binary_predictors.items()
            }

            return binary_predictions

        except Exception as e:
            print(f"⚠️ JEPA prediction error: {e}")
            print(
                f"   Visual seq shape: {visual_features_seq.shape if hasattr(visual_features_seq, 'shape') else 'Unknown'}"
            )
            print(
                f"   Game state seq shape: {game_state_features_seq.shape if hasattr(game_state_features_seq, 'shape') else 'Unknown'}"
            )

            # Return dummy predictions to maintain training stability
            device = visual_features_seq.device
            batch_size = visual_features_seq.size(0)
            return {
                outcome: torch.full(
                    (batch_size, self.prediction_horizon), 0.5, device=device
                )
                for outcome in self.binary_outcomes
            }


# --- Main Environment Wrapper ---
class SamuraiJEPAWrapper(gym.Wrapper):
    """Enhanced wrapper with comprehensive fixes"""

    def __init__(self, env, frame_stack=8, enable_jepa=True, **kwargs):
        super().__init__(env)

        # Core parameters
        self.frame_stack = frame_stack
        self.enable_jepa = enable_jepa
        self.state_history_length = 8
        self.target_size = (128, 180)

        # Setup observation space
        obs_shape = (3 * frame_stack, self.target_size[1], self.target_size[0])
        self.observation_space = spaces.Box(
            low=0, high=255, shape=obs_shape, dtype=np.uint8
        )

        # Initialize frame buffer
        self.frame_buffer = deque(maxlen=self.frame_stack)

        # Game state tracking
        self.prev_player_health = MAX_HEALTH
        self.prev_enemy_health = MAX_HEALTH
        self.player_wins_in_episode = 0
        self.enemy_wins_in_episode = 0

        # Statistics tracking (Fixed structure)
        self.current_stats = {
            "wins": 0,
            "losses": 0,
            "total_rounds": 0,  # Fixed: Added missing field
            "win_rate": 0.0,
        }

        self.strategic_stats = {
            "binary_predictions_made": 0,
            # Individual accuracy tracking only
            "individual_accuracies": {
                "will_opponent_attack": 0.5,
                "will_opponent_take_damage": 0.5,
                "will_player_take_damage": 0.5,
                "will_round_end_soon": 0.5,
            },
            "individual_predictions": {
                "will_opponent_attack": 0,
                "will_opponent_take_damage": 0,
                "will_player_take_damage": 0,
                "will_round_end_soon": 0,
            },
            "individual_correct": {
                "will_opponent_attack": 0,
                "will_opponent_take_damage": 0,
                "will_player_take_damage": 0,
                "will_round_end_soon": 0,
            },
        }

        # JEPA system initialization
        self.jepa_ready = False
        if self.enable_jepa:
            self._initialize_jepa_system()

        print(
            f"🥷 JEPA Wrapper Initialized: COMPREHENSIVE FIXES, Frame Stack={self.frame_stack}"
        )

    def _initialize_jepa_system(self):
        """Initialize JEPA-related components"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.feature_extractor: Optional[nn.Module] = None
        self.jepa_predictor: Optional[JEPASimpleBinaryPredictor] = None

        # History buffers with proper memory management
        self.visual_features_history = deque(maxlen=self.state_history_length)
        self.game_state_history = deque(maxlen=self.state_history_length)
        self.prediction_from_last_step: Optional[Dict] = None

        # Performance monitoring
        self.prediction_errors = []
        self.last_cleanup_time = 0

    def inject_feature_extractor(self, feature_extractor: nn.Module):
        """Inject feature extractor and initialize JEPA predictor"""
        if not self.enable_jepa:
            return

        try:
            self.feature_extractor = feature_extractor
            self._initialize_jepa_predictor(feature_extractor.features_dim)
            self.jepa_ready = True
            print("   ✅ Feature extractor injected and JEPA predictor initialized!")

        except Exception as e:
            print(f"   ❌ JEPA injection failed: {e}")
            self.jepa_ready = False

    def _initialize_jepa_predictor(self, visual_dim: int):
        """Initialize JEPA predictor with error handling"""
        if self.jepa_predictor is None:
            try:
                self.jepa_predictor = JEPASimpleBinaryPredictor(
                    visual_dim=visual_dim,
                    sequence_length=self.state_history_length,
                    prediction_horizon=self.frame_stack,
                    game_state_dim=8,
                ).to(self.device)
                print(f"   🧠 JEPA Predictor initialized with visual_dim={visual_dim}")

            except Exception as e:
                print(f"   ❌ JEPA predictor initialization failed: {e}")
                self.jepa_predictor = None

    def _get_visual_features(self, observation: np.ndarray) -> Optional[torch.Tensor]:
        """Extract visual features with proper error handling"""
        if self.feature_extractor is None:
            return None

        try:
            with torch.no_grad():
                # Ensure proper tensor conversion and device placement
                if isinstance(observation, np.ndarray):
                    obs_tensor = torch.from_numpy(observation).float()
                else:
                    obs_tensor = torch.tensor(observation, dtype=torch.float32)

                obs_tensor = obs_tensor.to(self.device).unsqueeze(0)
                features = self.feature_extractor(obs_tensor).squeeze(0)

                # Detach to prevent memory leaks
                return features.detach()

        except Exception as e:
            print(f"   ⚠️ Visual feature extraction failed: {e}")
            # Return zero tensor with correct dimensions
            if hasattr(self.feature_extractor, "features_dim"):
                return torch.zeros(
                    self.feature_extractor.features_dim, device=self.device
                )
            return torch.zeros(512, device=self.device)  # Fallback

    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess frame with error handling"""
        try:
            if frame is None:
                return np.zeros((*self.target_size, 3), dtype=np.uint8)
            return cv2.resize(frame, self.target_size, interpolation=cv2.INTER_AREA)
        except Exception as e:
            print(f"   ⚠️ Frame preprocessing failed: {e}")
            return np.zeros((*self.target_size, 3), dtype=np.uint8)

    def _get_stacked_observation(self) -> np.ndarray:
        """Get stacked observation with proper error handling"""
        try:
            if len(self.frame_buffer) == 0:
                # Return zero observation if buffer is empty
                return np.zeros(self.observation_space.shape, dtype=np.uint8)
            return np.concatenate(list(self.frame_buffer), axis=2).transpose(2, 0, 1)
        except Exception as e:
            print(f"   ⚠️ Frame stacking failed: {e}")
            return np.zeros(self.observation_space.shape, dtype=np.uint8)

    def _extract_game_state_vector(self, info: Dict) -> np.ndarray:
        """Extract game state with improved feature engineering"""
        try:
            player_health = info.get("health", self.prev_player_health)
            enemy_health = info.get("enemy_health", self.prev_enemy_health)
            round_num = info.get("round", 1)
            timer = info.get("timer", 99)

            # Normalize and extract meaningful features
            return np.array(
                [
                    player_health / MAX_HEALTH,  # Player health ratio
                    enemy_health / MAX_HEALTH,  # Enemy health ratio
                    (player_health - enemy_health) / MAX_HEALTH,  # Health difference
                    round_num / 3.0,  # Round progress
                    timer / 99.0,  # Time remaining
                    (
                        1.0 if player_health < (MAX_HEALTH * 0.3) else 0.0
                    ),  # Player critical
                    1.0 if enemy_health < (MAX_HEALTH * 0.3) else 0.0,  # Enemy critical
                    abs(player_health - enemy_health)
                    / MAX_HEALTH,  # Health gap magnitude
                ],
                dtype=np.float32,
            )

        except Exception as e:
            print(f"   ⚠️ Game state extraction failed: {e}")
            return np.zeros(8, dtype=np.float32)

    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict]:
        """Reset with comprehensive state initialization"""
        try:
            obs, info = self.env.reset(**kwargs)

            # Reset game state
            self.prev_player_health = info.get("health", MAX_HEALTH)
            self.prev_enemy_health = info.get("enemy_health", MAX_HEALTH)
            self.player_wins_in_episode = 0
            self.enemy_wins_in_episode = 0

            # Initialize frame buffer
            processed_frame = self._preprocess_frame(obs)
            self.frame_buffer.clear()
            for _ in range(self.frame_stack):
                self.frame_buffer.append(processed_frame)

            # Reset JEPA system
            if self.enable_jepa and self.jepa_ready:
                self._reset_jepa_state()

            return self._get_stacked_observation(), info

        except Exception as e:
            print(f"   ❌ Reset failed: {e}")
            # Return safe defaults
            return np.zeros(self.observation_space.shape, dtype=np.uint8), {}

    def _reset_jepa_state(self):
        """Reset JEPA state with proper memory management and dimension validation"""
        try:
            self.prediction_from_last_step = None

            # Clear histories with proper tensor cleanup
            if hasattr(self, "visual_features_history"):
                self.visual_features_history.clear()
            if hasattr(self, "game_state_history"):
                self.game_state_history.clear()

            # Initialize with zero states - ensure proper dimensions
            if self.feature_extractor is not None:
                feature_dim = getattr(self.feature_extractor, "features_dim", 512)
                zero_vf = torch.zeros(feature_dim, device=self.device)
                zero_gs = np.zeros(8, dtype=np.float32)

                # Fill histories to exact length needed
                for _ in range(self.state_history_length):
                    self.visual_features_history.append(zero_vf.clone())
                    self.game_state_history.append(zero_gs.copy())

                print(
                    f"   🔄 JEPA state reset: VF history length={len(self.visual_features_history)}, "
                    f"GS history length={len(self.game_state_history)}, "
                    f"feature_dim={feature_dim}"
                )

        except Exception as e:
            print(f"   ⚠️ JEPA state reset failed: {e}")
            import traceback

            print(f"   Full traceback: {traceback.format_exc()}")

    def step(self, action) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Enhanced step function with comprehensive fixes"""
        try:
            current_stacked_obs = self._get_stacked_observation()

            # 1. Make JEPA prediction if ready
            if self.enable_jepa and self.jepa_ready:
                self._make_jepa_prediction(current_stacked_obs)

            # 2. Execute environment step
            obs, _, terminated, truncated, next_info = self.env.step(action)
            self.frame_buffer.append(self._preprocess_frame(obs))

            # 3. Evaluate prediction accuracy
            if self.prediction_from_last_step is not None:
                self._evaluate_prediction_accuracy(next_info)

            # 4. Calculate enhanced reward
            enhanced_reward = self._calculate_enhanced_reward(next_info)

            # 5. Handle episode termination
            done = self._handle_episode_termination(terminated, truncated, next_info)

            # 6. Update info with stats
            next_info.update(
                {
                    "current_stats": self.current_stats,
                    "strategic_stats": self.strategic_stats,
                }
            )

            return (
                self._get_stacked_observation(),
                enhanced_reward,
                done,
                False,
                next_info,
            )

        except Exception as e:
            print(f"   ❌ Step failed: {e}")
            # Return safe defaults to maintain training stability
            return (
                np.zeros(self.observation_space.shape, dtype=np.uint8),
                0.0,
                True,
                False,
                {},
            )

    def _make_jepa_prediction(self, current_stacked_obs: np.ndarray):
        """Make JEPA prediction with error handling and dimension validation"""
        try:
            visual_features = self._get_visual_features(current_stacked_obs)
            if visual_features is None:
                return

            current_info = self.env.unwrapped.data.lookup_all()
            game_state_vec = self._extract_game_state_vector(current_info)

            # Update histories
            self.visual_features_history.append(visual_features.clone().detach())
            self.game_state_history.append(game_state_vec.copy())

            # Prepare sequences for transformer - ensure we have enough history
            if len(self.visual_features_history) >= self.state_history_length:
                # Stack visual features: (seq_len, feature_dim) -> (1, seq_len, feature_dim)
                vf_list = list(self.visual_features_history)
                vf_seq = torch.stack(vf_list, dim=0).unsqueeze(
                    0
                )  # (1, seq_len, visual_dim)

                # Stack game state features: (seq_len, 8) -> (1, seq_len, 8)
                gs_array = np.array(list(self.game_state_history))  # (seq_len, 8)
                gs_seq = torch.tensor(
                    gs_array, dtype=torch.float32, device=self.device
                ).unsqueeze(
                    0
                )  # (1, seq_len, 8)

                # Validate dimensions before prediction
                expected_seq_len = self.state_history_length
                if (
                    vf_seq.shape[1] != expected_seq_len
                    or gs_seq.shape[1] != expected_seq_len
                ):
                    print(
                        f"   ⚠️ Sequence length mismatch: VF={vf_seq.shape[1]}, GS={gs_seq.shape[1]}, expected={expected_seq_len}"
                    )
                    return

                # Make prediction
                with torch.no_grad():
                    self.prediction_from_last_step = self.jepa_predictor(vf_seq, gs_seq)

        except Exception as e:
            print(f"   ⚠️ JEPA prediction failed: {e}")
            import traceback

            print(f"   Full traceback: {traceback.format_exc()}")
            self.prediction_from_last_step = None

    def _evaluate_prediction_accuracy(self, next_info: Dict):
        """Evaluate individual JEPA prediction accuracies only"""
        try:
            player_health_after = next_info.get("health", self.prev_player_health)
            enemy_health_after = next_info.get("enemy_health", self.prev_enemy_health)

            # Calculate actual outcomes
            damage_to_enemy = max(0, self.prev_enemy_health - enemy_health_after)
            damage_to_player = max(0, self.prev_player_health - player_health_after)

            ground_truth = {
                "will_opponent_attack": float(damage_to_player > 1),
                "will_opponent_take_damage": float(damage_to_enemy > 1),
                "will_player_take_damage": float(damage_to_player > 1),
                "will_round_end_soon": float(
                    player_health_after < (MAX_HEALTH * 0.2)
                    or enemy_health_after < (MAX_HEALTH * 0.2)
                ),
            }

            # Calculate individual accuracy only
            num_predictions = 0
            individual_results = {}

            for key, pred_tensor in self.prediction_from_last_step.items():
                if key in ground_truth:
                    try:
                        predicted_prob = pred_tensor[0, 0].item()
                        # Use individual thresholds for better accuracy
                        threshold = PREDICTION_THRESHOLDS.get(key, PREDICTION_THRESHOLD)
                        predicted_label = 1 if predicted_prob > threshold else 0
                        actual_label = int(ground_truth[key])

                        is_correct = predicted_label == actual_label
                        individual_results[key] = {
                            "correct": is_correct,
                            "predicted": predicted_label,
                            "actual": actual_label,
                            "probability": predicted_prob,
                            "threshold_used": threshold,
                        }

                        if is_correct:
                            self.strategic_stats["individual_correct"][key] += 1

                        self.strategic_stats["individual_predictions"][key] += 1
                        num_predictions += 1

                        # Update individual accuracy with exponential moving average
                        current_individual_accuracy = float(is_correct)
                        self.strategic_stats["individual_accuracies"][key] = (
                            1 - ACCURACY_LEARNING_RATE
                        ) * self.strategic_stats["individual_accuracies"][
                            key
                        ] + ACCURACY_LEARNING_RATE * current_individual_accuracy

                    except Exception as e:
                        print(f"   ⚠️ Prediction evaluation error for {key}: {e}")
                        continue

            # Update prediction count
            if num_predictions > 0:
                self.strategic_stats["binary_predictions_made"] += 1

                # Debug logging every 100 predictions with more details
                if self.strategic_stats["binary_predictions_made"] % 100 == 0:
                    print(
                        f"\n   🎯 JEPA Individual Accuracies (after {self.strategic_stats['binary_predictions_made']} predictions):"
                    )
                    for outcome, accuracy in self.strategic_stats[
                        "individual_accuracies"
                    ].items():
                        total_preds = self.strategic_stats["individual_predictions"][
                            outcome
                        ]
                        total_correct = self.strategic_stats["individual_correct"][
                            outcome
                        ]
                        threshold = PREDICTION_THRESHOLDS.get(
                            outcome, PREDICTION_THRESHOLD
                        )
                        print(
                            f"      {outcome}: {accuracy*100:.1f}% ({total_correct}/{total_preds}) [thresh={threshold}]"
                        )

                    # Show current prediction details for debugging
                    if individual_results:
                        print("   🔍 Current Step Predictions:")
                        for key, result in individual_results.items():
                            prob = result["probability"]
                            pred = result["predicted"]
                            actual = result["actual"]
                            correct = "✅" if result["correct"] else "❌"
                            print(
                                f"      {key}: prob={prob:.3f} → pred={pred}, actual={actual} {correct}"
                            )

        except Exception as e:
            print(f"   ⚠️ Prediction accuracy evaluation failed: {e}")
            import traceback

            print(f"   Full traceback: {traceback.format_exc()}")

    def _calculate_enhanced_reward(self, next_info: Dict) -> float:
        """Calculate enhanced reward with improved balancing"""
        try:
            current_player_health = next_info.get("health", self.prev_player_health)
            current_enemy_health = next_info.get("enemy_health", self.prev_enemy_health)

            # Calculate damage components
            damage_dealt = max(0, self.prev_enemy_health - current_enemy_health)
            damage_taken = max(0, self.prev_player_health - current_player_health)

            # Base reward components
            health_delta_reward = (damage_dealt * 1.0 - damage_taken * 1.2) * 0.1
            time_reward = 0.001  # Small positive reward for survival
            win_loss_reward = 0.0

            # Bonus rewards
            combo_bonus = 0.0
            if damage_dealt > 10:  # Large damage combo
                combo_bonus = 0.5

            critical_health_penalty = 0.0
            if current_player_health < (MAX_HEALTH * 0.2):  # Critical health
                critical_health_penalty = -0.1

            # Check for round end
            round_ended = False
            if (current_player_health <= 0 or current_enemy_health <= 0) and (
                self.prev_player_health > 0 and self.prev_enemy_health > 0
            ):
                round_ended = True
                if current_player_health > current_enemy_health:
                    win_loss_reward = 20.0
                    self.player_wins_in_episode += 1
                else:
                    win_loss_reward = -20.0
                    self.enemy_wins_in_episode += 1

            # Combine all reward components
            enhanced_reward = (
                health_delta_reward
                + time_reward
                + win_loss_reward
                + combo_bonus
                + critical_health_penalty
            )

            # Update health tracking
            self.prev_player_health = current_player_health
            self.prev_enemy_health = current_enemy_health

            return float(enhanced_reward)

        except Exception as e:
            print(f"   ⚠️ Reward calculation failed: {e}")
            return 0.0

    def _handle_episode_termination(
        self, terminated: bool, truncated: bool, next_info: Dict
    ) -> bool:
        """Handle episode termination with proper statistics update"""
        try:
            done = terminated or truncated

            # Check for match end (best of 3)
            if self.player_wins_in_episode >= 2 or self.enemy_wins_in_episode >= 2:
                done = True

                # Update match statistics
                if self.player_wins_in_episode > self.enemy_wins_in_episode:
                    self.current_stats["wins"] += 1
                else:
                    self.current_stats["losses"] += 1

                # Update total rounds and win rate
                self.current_stats["total_rounds"] = (
                    self.current_stats["wins"] + self.current_stats["losses"]
                )

                if self.current_stats["total_rounds"] > 0:
                    self.current_stats["win_rate"] = (
                        self.current_stats["wins"] / self.current_stats["total_rounds"]
                    )

            return done

        except Exception as e:
            print(f"   ⚠️ Episode termination handling failed: {e}")
            return True  # Safe default

    def close(self):
        """Clean up resources"""
        try:
            if hasattr(self, "visual_features_history"):
                self.visual_features_history.clear()
            if hasattr(self, "game_state_history"):
                self.game_state_history.clear()
            if hasattr(self, "frame_buffer"):
                self.frame_buffer.clear()

            super().close()

        except Exception as e:
            print(f"   ⚠️ Cleanup failed: {e}")


class JEPAEnhancedCNN(BaseFeaturesExtractor):
    """Enhanced CNN feature extractor with improved architecture"""

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 512):
        super().__init__(observation_space, features_dim)

        # Store observation space for later use
        self.obs_space = observation_space
        n_input_channels = observation_space.shape[0]
        print(
            f"   🏗️ CNN Init: input_channels={n_input_channels}, target_features={features_dim}"
        )
        print(f"   📐 Input shape: {observation_space.shape}")

        # Enhanced CNN architecture
        self.cnn = nn.Sequential(
            # First conv block
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # Second conv block
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # Third conv block
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # Additional conv block for better feature extraction
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Calculate the number of features after convolution
        with torch.no_grad():
            sample_input = torch.as_tensor(observation_space.sample()[None]).float()
            cnn_output = self.cnn(sample_input)
            n_flatten = cnn_output.shape[1]
            print(f"   🔢 CNN output features: {n_flatten}")

        # Enhanced linear layers with dimension validation
        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim * 2),
            nn.LayerNorm(features_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(features_dim * 2, features_dim),
            nn.LayerNorm(features_dim),
            nn.ReLU(),
        )

        print(f"   ✅ CNN initialized successfully with {features_dim} output features")

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """Forward pass with proper normalization and dimension validation"""
        try:
            batch_size = observations.shape[0]

            # Validate input dimensions
            expected_shape = self.obs_space.shape  # Fixed: Use stored obs_space
            actual_shape = observations.shape[1:]  # Skip batch dimension

            if actual_shape != expected_shape:
                print(
                    f"   ⚠️ Input shape mismatch: expected {expected_shape}, got {actual_shape}"
                )
                # Try to reshape if possible
                if np.prod(actual_shape) == np.prod(expected_shape):
                    observations = observations.view(batch_size, *expected_shape)
                    print(f"   🔄 Reshaped input to {observations.shape}")
                else:
                    print(f"   ⚠️ Cannot reshape - proceeding with current shape")

            # Normalize observations to [0, 1]
            normalized_obs = observations.float() / 255.0

            # Apply CNN and linear layers
            cnn_features = self.cnn(normalized_obs)
            final_features = self.linear(cnn_features)

            # Validate output dimensions
            if final_features.shape[1] != self.features_dim:
                print(
                    f"   ⚠️ Output feature dimension mismatch: expected {self.features_dim}, got {final_features.shape[1]}"
                )

            return final_features

        except Exception as e:
            print(f"   ⚠️ CNN forward pass failed: {e}")
            print(f"   Input shape: {observations.shape}")
            print(f"   Expected shape: {getattr(self, 'obs_space', 'Unknown')}")
            import traceback

            print(f"   Full traceback: {traceback.format_exc()}")

            # Return zeros with correct shape to maintain training stability
            batch_size = observations.shape[0] if len(observations.shape) > 0 else 1
            return torch.zeros(
                batch_size, self.features_dim, device=observations.device
            )
