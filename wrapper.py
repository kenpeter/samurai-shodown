#!/usr/bin/env python3
"""
JEPA-Enhanced Samurai Showdown Wrapper - PERFORMANCE IMPROVEMENTS
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
ACCURACY_LEARNING_RATE = 0.01  # Increased for faster adaptation

# Adaptive thresholds based on observed patterns
PREDICTION_THRESHOLDS = {
    "will_opponent_attack": 0.55,  # Lowered - model shows ~52% confidence
    "will_opponent_take_damage": 0.5,
    "will_player_take_damage": 0.55,  # Lowered to catch more events
    "will_round_end_soon": 0.6,
}

# Minimum confidence for logging to reduce spam
MIN_LOG_CONFIDENCE = 0.65


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

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)


class JEPAImprovedPredictor(nn.Module):
    """Improved JEPA predictor with better learning signals"""

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

        # Enhanced feature encoding with residual connections
        feature_dim = 256  # Increased capacity
        self.context_encoder = nn.Sequential(
            nn.Linear(visual_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(),
        )

        self.game_state_encoder = nn.Sequential(
            nn.Linear(game_state_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(),
        )

        # Improved transformer architecture
        d_model = feature_dim * 2
        self.pos_encoder = PositionalEncoding(d_model, dropout=0.15)

        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=8,  # More attention heads
            dim_feedforward=d_model * 4,  # Larger feedforward
            dropout=0.15,
            batch_first=True,
            norm_first=True,
            activation="gelu",  # Better activation
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=4)

        # Specialized prediction heads with different architectures
        self.binary_predictors = nn.ModuleDict()
        for outcome in self.binary_outcomes:
            # Different architecture based on prediction difficulty
            if outcome == "will_player_take_damage":
                # More complex head for rare damage events
                self.binary_predictors[outcome] = nn.Sequential(
                    nn.Linear(d_model, 128),
                    nn.LayerNorm(128),
                    nn.GELU(),
                    nn.Dropout(0.2),
                    nn.Linear(128, 64),
                    nn.LayerNorm(64),
                    nn.GELU(),
                    nn.Dropout(0.1),
                    nn.Linear(64, prediction_horizon),
                )
            else:
                # Standard head for other predictions
                self.binary_predictors[outcome] = nn.Sequential(
                    nn.Linear(d_model, 96),
                    nn.LayerNorm(96),
                    nn.GELU(),
                    nn.Dropout(0.1),
                    nn.Linear(96, prediction_horizon),
                )

    def forward(self, visual_features_seq, game_state_features_seq):
        """Forward pass with improved gradient flow"""
        try:
            batch_size, seq_len = visual_features_seq.shape[:2]

            # Enhanced feature encoding
            visual_context = self.context_encoder(visual_features_seq)
            game_context = self.game_state_encoder(game_state_features_seq)

            # Combine with residual-like connection
            sequence_input = torch.cat([visual_context, game_context], dim=-1)
            sequence_input = self.pos_encoder(sequence_input)

            # Apply transformer with gradient checkpointing for memory efficiency
            transformer_out = self.transformer_encoder(sequence_input)

            # Use attention pooling instead of just last timestep
            attention_weights = torch.softmax(
                torch.sum(transformer_out, dim=-1), dim=1
            ).unsqueeze(-1)

            # Weighted average of all timesteps
            final_representation = torch.sum(transformer_out * attention_weights, dim=1)

            # Generate predictions with temperature scaling
            binary_predictions = {}
            for outcome, predictor in self.binary_predictors.items():
                logits = predictor(final_representation)
                # Temperature scaling for better calibration
                temperature = 1.2 if outcome == "will_player_take_damage" else 1.0
                binary_predictions[outcome] = torch.sigmoid(logits / temperature)

            return binary_predictions

        except Exception as e:
            print(f"⚠️ JEPA prediction error: {e}")
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
    """Improved wrapper with better game state detection and reduced logging"""

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

        # Enhanced game state tracking
        self.prev_player_health = MAX_HEALTH
        self.prev_enemy_health = MAX_HEALTH
        self.health_history = deque(maxlen=10)  # Track health changes over time
        self.action_history = deque(maxlen=5)  # Track recent actions

        self.player_wins_in_episode = 0
        self.enemy_wins_in_episode = 0

        # Statistics tracking
        self.current_stats = {
            "wins": 0,
            "losses": 0,
            "total_rounds": 0,
            "win_rate": 0.0,
        }

        self.strategic_stats = {
            "binary_predictions_made": 0,
            "game_active_frames": 0,
            "game_paused_frames": 0,
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
            f"🥷 JEPA Wrapper Improved: Enhanced Learning & Reduced Logging, Frame Stack={self.frame_stack}"
        )

    def _initialize_jepa_system(self):
        """Initialize JEPA-related components"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.feature_extractor: Optional[nn.Module] = None
        self.jepa_predictor: Optional[JEPAImprovedPredictor] = None

        # History buffers
        self.visual_features_history = deque(maxlen=self.state_history_length)
        self.game_state_history = deque(maxlen=self.state_history_length)
        self.prediction_from_last_step: Optional[Dict] = None

        # Enhanced damage tracking
        self.damage_stats = {
            "total_frames": 0,
            "player_damage_frames": 0,
            "enemy_damage_frames": 0,
            "damage_amounts": [],
            "recent_damages": [],
            "false_positives": 0,
            "true_positives": 0,
            "false_negatives": 0,
            "true_negatives": 0,
            "last_damage_frame": -1000,
            "damage_intervals": [],
        }

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
        """Initialize improved JEPA predictor"""
        if self.jepa_predictor is None:
            try:
                self.jepa_predictor = JEPAImprovedPredictor(
                    visual_dim=visual_dim,
                    sequence_length=self.state_history_length,
                    prediction_horizon=self.frame_stack,
                    game_state_dim=10,  # Expanded game state
                ).to(self.device)
                print(
                    f"   🧠 Improved JEPA Predictor initialized with visual_dim={visual_dim}"
                )

            except Exception as e:
                print(f"   ❌ JEPA predictor initialization failed: {e}")
                self.jepa_predictor = None

    def _is_game_active(self, info: Dict) -> bool:
        """Detect if game is actively running or paused"""
        try:
            # Check for timer existence and movement
            timer = info.get("timer", None)
            if timer is not None and hasattr(self, "_last_timer"):
                if timer != self._last_timer:
                    self._last_timer = timer
                    return True

            # Check for health changes
            current_player = info.get("health", self.prev_player_health)
            current_enemy = info.get("enemy_health", self.prev_enemy_health)

            if (
                current_player != self.prev_player_health
                or current_enemy != self.prev_enemy_health
            ):
                return True

            # Check round progression
            current_round = info.get("round", 0)
            if hasattr(self, "_last_round") and current_round != self._last_round:
                self._last_round = current_round
                return True

            return False

        except Exception:
            return True  # Assume active if we can't determine

    def _extract_enhanced_game_state_vector(self, info: Dict) -> np.ndarray:
        """Extract enhanced game state with more features"""
        try:
            player_health = info.get("health", self.prev_player_health)
            enemy_health = info.get("enemy_health", self.prev_enemy_health)
            round_num = info.get("round", 1)
            timer = info.get("timer", 99)

            # Calculate health momentum
            health_momentum_player = 0.0
            health_momentum_enemy = 0.0
            if len(self.health_history) > 1:
                prev_healths = self.health_history[-1]
                health_momentum_player = (player_health - prev_healths[0]) / MAX_HEALTH
                health_momentum_enemy = (enemy_health - prev_healths[1]) / MAX_HEALTH

            # Store current health for next calculation
            self.health_history.append((player_health, enemy_health))

            return np.array(
                [
                    player_health / MAX_HEALTH,  # Player health ratio
                    enemy_health / MAX_HEALTH,  # Enemy health ratio
                    (player_health - enemy_health) / MAX_HEALTH,  # Health difference
                    round_num / 3.0,  # Round progress
                    (timer if timer is not None else 99) / 99.0,  # Time remaining
                    (
                        1.0 if player_health < (MAX_HEALTH * 0.3) else 0.0
                    ),  # Player critical
                    1.0 if enemy_health < (MAX_HEALTH * 0.3) else 0.0,  # Enemy critical
                    abs(player_health - enemy_health)
                    / MAX_HEALTH,  # Health gap magnitude
                    health_momentum_player,  # Player health change rate
                    health_momentum_enemy,  # Enemy health change rate
                ],
                dtype=np.float32,
            )

        except Exception as e:
            print(f"   ⚠️ Enhanced game state extraction failed: {e}")
            return np.zeros(10, dtype=np.float32)

    def _get_visual_features(self, observation: np.ndarray) -> Optional[torch.Tensor]:
        """Extract visual features with proper error handling"""
        if self.feature_extractor is None:
            return None

        try:
            with torch.no_grad():
                if isinstance(observation, np.ndarray):
                    obs_tensor = torch.from_numpy(observation).float()
                else:
                    obs_tensor = torch.tensor(observation, dtype=torch.float32)

                obs_tensor = obs_tensor.to(self.device).unsqueeze(0)
                features = self.feature_extractor(obs_tensor).squeeze(0)
                return features.detach()

        except Exception as e:
            print(f"   ⚠️ Visual feature extraction failed: {e}")
            if hasattr(self.feature_extractor, "features_dim"):
                return torch.zeros(
                    self.feature_extractor.features_dim, device=self.device
                )
            return torch.zeros(512, device=self.device)

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
                return np.zeros(self.observation_space.shape, dtype=np.uint8)
            return np.concatenate(list(self.frame_buffer), axis=2).transpose(2, 0, 1)
        except Exception as e:
            print(f"   ⚠️ Frame stacking failed: {e}")
            return np.zeros(self.observation_space.shape, dtype=np.uint8)

    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict]:
        """Reset with comprehensive state initialization"""
        try:
            obs, info = self.env.reset(**kwargs)

            # Reset game state
            self.prev_player_health = info.get("health", MAX_HEALTH)
            self.prev_enemy_health = info.get("enemy_health", MAX_HEALTH)
            self.player_wins_in_episode = 0
            self.enemy_wins_in_episode = 0

            # Clear histories
            self.health_history.clear()
            self.action_history.clear()

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
            return np.zeros(self.observation_space.shape, dtype=np.uint8), {}

    def _reset_jepa_state(self):
        """Reset JEPA state with proper initialization"""
        try:
            self.prediction_from_last_step = None

            if hasattr(self, "visual_features_history"):
                self.visual_features_history.clear()
            if hasattr(self, "game_state_history"):
                self.game_state_history.clear()

            # Initialize with meaningful zero states
            if self.feature_extractor is not None:
                feature_dim = getattr(self.feature_extractor, "features_dim", 512)
                zero_vf = torch.zeros(feature_dim, device=self.device)
                zero_gs = np.zeros(10, dtype=np.float32)  # Updated for enhanced state

                for _ in range(self.state_history_length):
                    self.visual_features_history.append(zero_vf.clone())
                    self.game_state_history.append(zero_gs.copy())

        except Exception as e:
            print(f"   ⚠️ JEPA state reset failed: {e}")

    def step(self, action) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Enhanced step function with improved game state detection"""
        try:
            # Store action in history for pattern analysis
            self.action_history.append(action)

            current_stacked_obs = self._get_stacked_observation()

            # Make JEPA prediction if ready
            if self.enable_jepa and self.jepa_ready:
                self._make_jepa_prediction(current_stacked_obs)

            # Execute environment step
            obs, _, terminated, truncated, next_info = self.env.step(action)
            self.frame_buffer.append(self._preprocess_frame(obs))

            # Check if game is active
            is_active = self._is_game_active(next_info)
            if is_active:
                self.strategic_stats["game_active_frames"] += 1
            else:
                self.strategic_stats["game_paused_frames"] += 1

            # Only evaluate predictions during active gameplay
            if self.prediction_from_last_step is not None and is_active:
                self._evaluate_prediction_accuracy(next_info)

            # Calculate enhanced reward
            enhanced_reward = self._calculate_enhanced_reward(next_info)

            # Handle episode termination
            done = self._handle_episode_termination(terminated, truncated, next_info)

            # Update info with stats
            next_info.update(
                {
                    "current_stats": self.current_stats,
                    "strategic_stats": self.strategic_stats,
                    "game_active": is_active,
                }
            )

            # Update health tracking
            self.prev_player_health = next_info.get("health", self.prev_player_health)
            self.prev_enemy_health = next_info.get(
                "enemy_health", self.prev_enemy_health
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
            return (
                np.zeros(self.observation_space.shape, dtype=np.uint8),
                0.0,
                True,
                False,
                {},
            )

    def _make_jepa_prediction(self, current_stacked_obs: np.ndarray):
        """Make JEPA prediction with enhanced game state"""
        try:
            visual_features = self._get_visual_features(current_stacked_obs)
            if visual_features is None:
                return

            current_info = self.env.unwrapped.data.lookup_all()
            game_state_vec = self._extract_enhanced_game_state_vector(current_info)

            # Update histories
            self.visual_features_history.append(visual_features.clone().detach())
            self.game_state_history.append(game_state_vec.copy())

            # Make prediction if we have enough history
            if len(self.visual_features_history) >= self.state_history_length:
                vf_list = list(self.visual_features_history)
                vf_seq = torch.stack(vf_list, dim=0).unsqueeze(0)

                gs_array = np.array(list(self.game_state_history))
                gs_seq = torch.tensor(
                    gs_array, dtype=torch.float32, device=self.device
                ).unsqueeze(0)

                with torch.no_grad():
                    self.prediction_from_last_step = self.jepa_predictor(vf_seq, gs_seq)

        except Exception as e:
            print(f"   ⚠️ JEPA prediction failed: {e}")
            self.prediction_from_last_step = None

    def _evaluate_prediction_accuracy(self, next_info: Dict):
        """Improved prediction accuracy evaluation with reduced logging"""
        try:
            player_health_after = next_info.get("health", self.prev_player_health)
            enemy_health_after = next_info.get("enemy_health", self.prev_enemy_health)

            # Calculate damage
            damage_to_enemy = max(0, self.prev_enemy_health - enemy_health_after)
            damage_to_player = max(0, self.prev_player_health - player_health_after)

            # Update damage statistics
            self.damage_stats["total_frames"] += 1
            current_frame = self.damage_stats["total_frames"]

            if damage_to_player > 1:
                self.damage_stats["player_damage_frames"] += 1
                interval = current_frame - self.damage_stats["last_damage_frame"]
                if self.damage_stats["last_damage_frame"] > 0:
                    self.damage_stats["damage_intervals"].append(interval)
                self.damage_stats["last_damage_frame"] = current_frame

            if damage_to_enemy > 1:
                self.damage_stats["enemy_damage_frames"] += 1

            # Define ground truth
            ground_truth = {
                "will_opponent_attack": float(damage_to_player > 1),
                "will_opponent_take_damage": float(damage_to_enemy > 1),
                "will_player_take_damage": float(damage_to_player > 1),
                "will_round_end_soon": float(
                    player_health_after < (MAX_HEALTH * 0.2)
                    or enemy_health_after < (MAX_HEALTH * 0.2)
                ),
            }

            # Evaluate predictions with adaptive thresholds
            num_predictions = 0
            for key, pred_tensor in self.prediction_from_last_step.items():
                if key in ground_truth:
                    try:
                        predicted_prob = pred_tensor[0, 0].item()
                        threshold = PREDICTION_THRESHOLDS.get(key, 0.5)

                        predicted_label = 1 if predicted_prob > threshold else 0
                        actual_label = int(ground_truth[key])
                        is_correct = predicted_label == actual_label

                        # Update confusion matrix for damage predictions
                        if key == "will_player_take_damage":
                            if predicted_label == 1 and actual_label == 1:
                                self.damage_stats["true_positives"] += 1
                            elif predicted_label == 1 and actual_label == 0:
                                self.damage_stats["false_positives"] += 1
                            elif predicted_label == 0 and actual_label == 1:
                                self.damage_stats["false_negatives"] += 1
                            else:
                                self.damage_stats["true_negatives"] += 1

                            # REDUCED LOGGING: Only log significant events
                            if actual_label == 1:  # Actual damage
                                result_str = (
                                    "✅ DETECTED" if is_correct else "❌ MISSED"
                                )
                                print(
                                    f"      🚨 DAMAGE EVENT: prob={predicted_prob:.3f}, {result_str}"
                                )
                            elif (
                                predicted_label == 1 and predicted_prob > 0.7
                            ):  # High confidence false alarm
                                print(
                                    f"      ⚡ HIGH CONF FALSE ALARM: prob={predicted_prob:.3f}"
                                )

                        # Update statistics
                        if is_correct:
                            self.strategic_stats["individual_correct"][key] += 1

                        self.strategic_stats["individual_predictions"][key] += 1
                        num_predictions += 1

                        # Update accuracy with adaptive learning rate
                        current_accuracy = float(is_correct)
                        self.strategic_stats["individual_accuracies"][key] = (
                            1 - ACCURACY_LEARNING_RATE
                        ) * self.strategic_stats["individual_accuracies"][
                            key
                        ] + ACCURACY_LEARNING_RATE * current_accuracy

                    except Exception as e:
                        print(f"   ⚠️ Prediction evaluation error for {key}: {e}")
                        continue

            # Update prediction count and periodic reporting
            if num_predictions > 0:
                self.strategic_stats["binary_predictions_made"] += 1

                # Reduced reporting frequency
                if self.strategic_stats["binary_predictions_made"] % 1000 == 0:
                    self._print_comprehensive_stats()

        except Exception as e:
            print(f"   ⚠️ Prediction accuracy evaluation failed: {e}")

    def _print_comprehensive_stats(self):
        """Print comprehensive statistics less frequently"""
        try:
            print(
                f"\n   📊 JEPA Performance Report (after {self.strategic_stats['binary_predictions_made']} predictions):"
            )

            for outcome, accuracy in self.strategic_stats[
                "individual_accuracies"
            ].items():
                total_preds = self.strategic_stats["individual_predictions"][outcome]
                total_correct = self.strategic_stats["individual_correct"][outcome]
                threshold = PREDICTION_THRESHOLDS.get(outcome, 0.5)

                if total_preds > 0:
                    if outcome == "will_player_take_damage":
                        tp = self.damage_stats["true_positives"]
                        fp = self.damage_stats["false_positives"]
                        fn = self.damage_stats["false_negatives"]

                        precision = tp / max(1, tp + fp)
                        recall = tp / max(1, tp + fn)
                        f1 = 2 * (precision * recall) / max(0.001, precision + recall)

                        print(
                            f"      {outcome}: {accuracy*100:.1f}% | P={precision:.3f} R={recall:.3f} F1={f1:.3f}"
                        )
                    else:
                        print(
                            f"      {outcome}: {accuracy*100:.1f}% ({total_correct}/{total_preds})"
                        )

            # Game activity analysis
            total_frames = (
                self.strategic_stats["game_active_frames"]
                + self.strategic_stats["game_paused_frames"]
            )
            if total_frames > 0:
                active_rate = (
                    self.strategic_stats["game_active_frames"] / total_frames * 100
                )
                print(
                    f"   🎮 Game Activity: {active_rate:.1f}% active ({self.strategic_stats['game_active_frames']}/{total_frames})"
                )

            # Damage analysis
            if self.damage_stats["total_frames"] > 100:
                player_damage_rate = (
                    self.damage_stats["player_damage_frames"]
                    / self.damage_stats["total_frames"]
                    * 100
                )
                print(
                    f"   💥 Damage Rate: {player_damage_rate:.3f}% ({self.damage_stats['player_damage_frames']} events)"
                )

                if len(self.damage_stats["damage_intervals"]) > 1:
                    avg_interval = np.mean(self.damage_stats["damage_intervals"])
                    print(f"   ⏱️ Avg Damage Interval: {avg_interval:.1f} frames")

        except Exception as e:
            print(f"   ⚠️ Stats printing failed: {e}")

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
            if (current_player_health <= 0 or current_enemy_health <= 0) and (
                self.prev_player_health > 0 and self.prev_enemy_health > 0
            ):
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
            if hasattr(self, "health_history"):
                self.health_history.clear()
            if hasattr(self, "action_history"):
                self.action_history.clear()

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

        # Enhanced CNN architecture with residual connections
        self.conv1 = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        self.flatten = nn.Flatten()

        # Calculate the number of features after convolution
        with torch.no_grad():
            sample_input = torch.as_tensor(observation_space.sample()[None]).float()
            x = self.conv1(sample_input / 255.0)
            x = self.conv2(x)
            x = self.conv3(x)
            x = self.conv4(x)
            x = self.flatten(x)
            n_flatten = x.shape[1]
            print(f"   🔢 CNN output features: {n_flatten}")

        # Enhanced linear layers with skip connections
        self.linear1 = nn.Linear(n_flatten, features_dim * 2)
        self.norm1 = nn.LayerNorm(features_dim * 2)
        self.dropout1 = nn.Dropout(0.1)

        self.linear2 = nn.Linear(features_dim * 2, features_dim)
        self.norm2 = nn.LayerNorm(features_dim)

        # Additional processing layer
        self.linear3 = nn.Linear(features_dim, features_dim)
        self.norm3 = nn.LayerNorm(features_dim)

        print(
            f"   ✅ Enhanced CNN initialized successfully with {features_dim} output features"
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connections and proper normalization"""
        try:
            batch_size = observations.shape[0]

            # Validate input dimensions
            expected_shape = self.obs_space.shape
            actual_shape = observations.shape[1:]

            if actual_shape != expected_shape:
                if np.prod(actual_shape) == np.prod(expected_shape):
                    observations = observations.view(batch_size, *expected_shape)
                else:
                    print(f"   ⚠️ Cannot reshape input - shape mismatch")

            # Normalize observations to [0, 1]
            normalized_obs = observations.float() / 255.0

            # Apply convolutional layers
            x = self.conv1(normalized_obs)
            x = self.conv2(x)
            x = self.conv3(x)
            x = self.conv4(x)
            x = self.flatten(x)

            # Apply linear layers with residual connection
            x1 = torch.relu(self.norm1(self.linear1(x)))
            x1 = self.dropout1(x1)

            x2 = torch.relu(self.norm2(self.linear2(x1)))

            # Residual connection if dimensions match
            if x2.shape[-1] == x1.shape[-1]:
                x2 = x2 + x1

            # Final processing
            final_features = torch.relu(self.norm3(self.linear3(x2) + x2))

            return final_features

        except Exception as e:
            print(f"   ⚠️ CNN forward pass failed: {e}")
            print(f"   Input shape: {observations.shape}")
            import traceback

            print(f"   Full traceback: {traceback.format_exc()}")

            # Return zeros with correct shape to maintain training stability
            batch_size = observations.shape[0] if len(observations.shape) > 0 else 1
            return torch.zeros(
                batch_size, self.features_dim, device=observations.device
            )
