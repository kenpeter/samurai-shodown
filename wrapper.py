#!/usr/bin/env python3
"""
JEPA-Enhanced Samurai Showdown Wrapper - ANTI-SPAM DEFENSE FIXES
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
ACCURACY_LEARNING_RATE = 0.01
CRITICAL_HEALTH_THRESHOLD = MAX_HEALTH * 0.3

# FIXED: More restrictive thresholds to prevent spam
PREDICTION_THRESHOLDS = {
    "is_best_time_to_attack": 0.6,
    "is_best_time_to_defend": 0.75,  # INCREASED from 0.55 to 0.75 - much more selective
}

# Minimum confidence for logging to reduce spam
MIN_LOG_CONFIDENCE = 0.65

# NEW: Anti-spam constants
MAX_CONSECUTIVE_SAME_ACTION = 3  # Maximum allowed consecutive same actions
ACTION_VARIETY_REWARD_THRESHOLD = 0.1  # Reward threshold for action variety


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


class JEPAStrategicPredictor(nn.Module):
    """JEPA predictor focused on strategic attack/defense timing"""

    def __init__(
        self, visual_dim=512, sequence_length=8, prediction_horizon=8, game_state_dim=11
    ):
        super().__init__()
        self.prediction_horizon = prediction_horizon
        self.strategic_outcomes = [
            "is_best_time_to_attack",
            "is_best_time_to_defend",
        ]

        # Enhanced feature encoding with residual connections
        feature_dim = 256
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

        # Strategic transformer architecture
        d_model = feature_dim * 2
        self.pos_encoder = PositionalEncoding(d_model, dropout=0.15)

        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=8,
            dim_feedforward=d_model * 4,
            dropout=0.15,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=4)

        # Specialized prediction heads for strategic decisions
        self.strategic_predictors = nn.ModuleDict()

        # Attack timing predictor - more complex for aggressive decisions
        self.strategic_predictors["is_best_time_to_attack"] = nn.Sequential(
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

        # FIXED: Defense timing predictor - more restrictive architecture
        self.strategic_predictors["is_best_time_to_defend"] = nn.Sequential(
            nn.Linear(d_model, 96),
            nn.LayerNorm(96),
            nn.GELU(),
            nn.Dropout(0.2),  # INCREASED dropout from 0.15 to 0.2
            nn.Linear(96, 48),
            nn.LayerNorm(48),
            nn.GELU(),
            nn.Dropout(0.15),  # INCREASED dropout from 0.1 to 0.15
            nn.Linear(48, prediction_horizon),
        )

    def forward(self, visual_features_seq, game_state_features_seq):
        """Forward pass focused on strategic timing predictions"""
        try:
            batch_size, seq_len = visual_features_seq.shape[:2]

            # Enhanced feature encoding
            visual_context = self.context_encoder(visual_features_seq)
            game_context = self.game_state_encoder(game_state_features_seq)

            # Combine with residual-like connection
            sequence_input = torch.cat([visual_context, game_context], dim=-1)
            sequence_input = self.pos_encoder(sequence_input)

            # Apply transformer
            transformer_out = self.transformer_encoder(sequence_input)

            # Use attention pooling for strategic decisions
            attention_weights = torch.softmax(
                torch.sum(transformer_out, dim=-1), dim=1
            ).unsqueeze(-1)

            # Weighted average of all timesteps
            final_representation = torch.sum(transformer_out * attention_weights, dim=1)

            # Generate strategic predictions
            strategic_predictions = {}
            for outcome, predictor in self.strategic_predictors.items():
                logits = predictor(final_representation)
                # FIXED: Higher temperature for defense to make it more conservative
                if outcome == "is_best_time_to_attack":
                    temperature = 1.1
                else:  # is_best_time_to_defend
                    temperature = 1.3  # INCREASED from 1.0 to 1.3 - more conservative
                strategic_predictions[outcome] = torch.sigmoid(logits / temperature)

            return strategic_predictions

        except Exception as e:
            print(f"⚠️ JEPA strategic prediction error: {e}")
            device = visual_features_seq.device
            batch_size = visual_features_seq.size(0)
            return {
                outcome: torch.full(
                    (batch_size, self.prediction_horizon), 0.5, device=device
                )
                for outcome in self.strategic_outcomes
            }


# --- Main Environment Wrapper ---
class SamuraiJEPAWrapperImproved(gym.Wrapper):
    """Improved wrapper with anti-spam strategic attack/defense predictions"""

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
        self.prev_score = 0
        self.health_history = deque(maxlen=10)
        self.action_history = deque(maxlen=5)

        # Track recent damage for better defense detection
        self._recent_damage_history = deque(maxlen=5)
        self._last_action = 0  # Track last action for defense evaluation
        self._prev_action_for_reward = 0  # Initialize missing variable

        # NEW: Anti-spam tracking
        self._consecutive_action_count = 0
        self._last_repeated_action = None
        self._action_variety_history = deque(maxlen=10)  # Track action diversity
        self._spam_penalty_active = False

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
            "strategic_predictions_made": 0,
            "game_active_frames": 0,
            "game_paused_frames": 0,
            "strategic_accuracies": {
                "is_best_time_to_attack": 0.5,
                "is_best_time_to_defend": 0.5,
            },
            "strategic_predictions": {
                "is_best_time_to_attack": 0,
                "is_best_time_to_defend": 0,
            },
            "strategic_correct": {
                "is_best_time_to_attack": 0,
                "is_best_time_to_defend": 0,
            },
            # NEW: Anti-spam statistics
            "spam_penalties_applied": 0,
            "action_variety_rewards": 0,
            "defensive_spam_detected": 0,
        }

        # Strategic decision tracking
        self.strategic_decision_stats = {
            "attack_opportunities": 0,
            "successful_attacks": 0,
            "attack_success_rate": 0.0,
            "defense_opportunities": 0,
            "successful_defenses": 0,
            "defense_success_rate": 0.0,
            "recent_decisions": deque(maxlen=50),
        }

        # JEPA system initialization
        self.jepa_ready = False
        if self.enable_jepa:
            self._initialize_jepa_system()

        print(
            f"🥷 JEPA Wrapper Anti-Spam: Attack/Defense Timing + Variety Enforcement, Frame Stack={self.frame_stack}"
        )

    def _initialize_jepa_system(self):
        """Initialize JEPA-related components"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.feature_extractor: Optional[nn.Module] = None
        self.jepa_predictor: Optional[JEPAStrategicPredictor] = None

        # History buffers
        self.visual_features_history = deque(maxlen=self.state_history_length)
        self.game_state_history = deque(maxlen=self.state_history_length)
        self.prediction_from_last_step: Optional[Dict] = None

        # Strategic timing tracking
        self.strategic_timing_stats = {
            "total_frames": 0,
            "attack_frames": 0,
            "defense_frames": 0,
            "successful_attack_predictions": 0,
            "successful_defense_predictions": 0,
        }

    def inject_feature_extractor(self, feature_extractor: nn.Module):
        """Inject feature extractor and initialize JEPA predictor"""
        if not self.enable_jepa:
            return

        try:
            self.feature_extractor = feature_extractor
            self._initialize_jepa_predictor(feature_extractor.features_dim)
            self.jepa_ready = True
            print("   ✅ Strategic JEPA predictor initialized!")

        except Exception as e:
            print(f"   ❌ JEPA injection failed: {e}")
            self.jepa_ready = False

    def _initialize_jepa_predictor(self, visual_dim: int):
        """Initialize strategic JEPA predictor"""
        if self.jepa_predictor is None:
            try:
                self.jepa_predictor = JEPAStrategicPredictor(
                    visual_dim=visual_dim,
                    sequence_length=self.state_history_length,
                    prediction_horizon=self.frame_stack,
                    game_state_dim=11,  # Updated from 10 to 11 for score momentum
                ).to(self.device)
                print(
                    f"   🧠 Strategic JEPA Predictor initialized with visual_dim={visual_dim}, game_state_dim=11"
                )

            except Exception as e:
                print(f"   ❌ JEPA predictor initialization failed: {e}")
                self.jepa_predictor = None

    def _is_game_active(self, info: Dict) -> bool:
        """Simplified game activity detection"""
        try:
            current_player = info.get("health", self.prev_player_health)
            current_enemy = info.get("enemy_health", self.prev_enemy_health)

            if not hasattr(self, "_debug_counter"):
                self._debug_counter = 0
            self._debug_counter += 1

            is_active = current_player > 0 and current_enemy > 0

            if self._debug_counter % 2000 == 0:
                print(
                    f"   🎮 Game Status: P:{current_player}, E:{current_enemy}, Active:{is_active}"
                )

            return is_active

        except Exception as e:
            print(f"   ❌ Game activity detection error: {e}")
            return True

    def _extract_enhanced_game_state_vector(self, info: Dict) -> np.ndarray:
        """Extract strategic game state vector with score momentum (11 features)"""
        try:
            player_health = info.get("health", self.prev_player_health)
            enemy_health = info.get("enemy_health", self.prev_enemy_health)
            current_score = info.get("score", self.prev_score)
            round_num = info.get("round", 1)

            # Calculate score momentum (rate of score increase)
            score_momentum = 0.0
            if (
                hasattr(self, "_last_score_for_momentum")
                and self._last_score_for_momentum is not None
            ):
                score_delta = current_score - self._last_score_for_momentum
                if score_delta > 0:
                    # Normalize score momentum using tanh to keep in reasonable range
                    score_momentum = np.tanh(
                        score_delta / 1000.0
                    )  # Scale by 1000 points
                # Handle score resets (new round/game)
                elif score_delta < -1000:  # Large negative = reset
                    score_momentum = 0.0
            self._last_score_for_momentum = current_score

            # Strategic features for timing decisions
            f1_player_health_norm = player_health / MAX_HEALTH
            f2_enemy_health_norm = enemy_health / MAX_HEALTH
            f3_health_advantage = (player_health - enemy_health) / MAX_HEALTH
            f4_health_ratio = player_health / (enemy_health + 1e-6)
            f5_total_health_pool = (player_health + enemy_health) / (2 * MAX_HEALTH)
            f6_player_is_critical = (
                1.0 if player_health < CRITICAL_HEALTH_THRESHOLD else 0.0
            )
            f7_enemy_is_critical = (
                1.0 if enemy_health < CRITICAL_HEALTH_THRESHOLD else 0.0
            )
            f8_score_momentum = score_momentum  # Score momentum feature
            f9_is_round_1 = 1.0 if round_num == 1 else 0.0
            f10_is_round_2 = 1.0 if round_num == 2 else 0.0
            f11_is_round_3 = 1.0 if round_num >= 3 else 0.0

            return np.array(
                [
                    f1_player_health_norm,
                    f2_enemy_health_norm,
                    f3_health_advantage,
                    f4_health_ratio,
                    f5_total_health_pool,
                    f6_player_is_critical,
                    f7_enemy_is_critical,
                    f8_score_momentum,
                    f9_is_round_1,
                    f10_is_round_2,
                    f11_is_round_3,
                ],
                dtype=np.float32,
            )

        except Exception as e:
            print(f"   ⚠️ Game state extraction failed: {e}")
            return np.zeros(11, dtype=np.float32)

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

    def _log_round_summary(self):
        """Log a summary of the round wins before reset"""
        if hasattr(self, "player_wins_in_episode") and hasattr(
            self, "enemy_wins_in_episode"
        ):
            if self.player_wins_in_episode > 0 or self.enemy_wins_in_episode > 0:
                print(
                    f"   🎯 Episode Summary: Player rounds won: {self.player_wins_in_episode}, Enemy rounds won: {self.enemy_wins_in_episode}"
                )

    # FIXED: Anti-spam action tracking methods with proper multi-button handling
    def _track_action_variety(self, action):
        """Track action variety and detect spam patterns - FIXED for multi-button arrays"""
        try:
            # FIXED: Handle multi-button action arrays (e.g., [1. 0. 0. 0. 1. 0. 0. 1. 1. 1. 1. 0.])
            if hasattr(action, "__len__") and len(action) > 1:
                # Convert multi-button array to a unique action identifier
                # Method 1: Convert to tuple (hashable) for tracking

                # convert muti array button to unique action identifier
                action_tuple = tuple(int(x) for x in action)
                # now unique action identifier (basically using hash, multi -> discrete)
                action_hash = hash(action_tuple)

                # Method 2: Also create a readable action signature for logging
                pressed_buttons = [
                    i for i, pressed in enumerate(action) if pressed > 0.5
                ]
                action_signature = f"buttons_{'-'.join(map(str, pressed_buttons)) if pressed_buttons else 'none'}"

            elif hasattr(action, "item"):  # numpy scalar
                action_hash = int(action.item())
                action_signature = f"action_{action_hash}"
            elif (
                hasattr(action, "__len__") and len(action) == 1
            ):  # single element array
                action_hash = int(action[0])
                action_signature = f"action_{action_hash}"
            else:
                action_hash = int(action)  # already a scalar
                action_signature = f"action_{action_hash}"

            # Add action to variety history (use hash for comparison)
            self._action_variety_history.append(action_hash)

            # Check for consecutive same actions
            if self._last_repeated_action == action_hash:
                self._consecutive_action_count += 1
            else:
                self._consecutive_action_count = 1
                self._last_repeated_action = action_hash

            # Detect spam
            if self._consecutive_action_count > MAX_CONSECUTIVE_SAME_ACTION:
                if not self._spam_penalty_active:
                    self._spam_penalty_active = True
                    print(
                        f"   🚫 SPAM DETECTED: {action_signature} repeated {self._consecutive_action_count} times"
                    )
                    self.strategic_stats["spam_penalties_applied"] += 1

                    # Special tracking for defensive spam
                    # For multi-button: check if it's a defensive combination
                    is_defensive = self._is_defensive_action(action)
                    if is_defensive:
                        self.strategic_stats["defensive_spam_detected"] += 1
            else:
                self._spam_penalty_active = False

        except Exception as e:
            print(f"   ⚠️ Action tracking failed: {e}")
            print(
                f"   Action type: {type(action)}, Action shape: {getattr(action, 'shape', 'no shape')}"
            )

    def _is_defensive_action(self, action) -> bool:
        """Determine if an action is defensive based on button combination"""
        try:
            if hasattr(action, "__len__") and len(action) > 1:
                # For fighting games, defensive actions might be:
                # - No buttons pressed (blocking/waiting)
                # - Specific defensive button combinations
                pressed_buttons = [
                    i for i, pressed in enumerate(action) if pressed > 0.5
                ]

                # No buttons = defensive (blocking/waiting)
                if len(pressed_buttons) == 0:
                    return True

                # Single button presses might be defensive (e.g., block button)
                # This is game-specific - you may need to adjust based on your game
                if len(pressed_buttons) == 1:
                    # Buttons 0, 1 might be defensive in many fighting games
                    if pressed_buttons[0] in [0, 1]:
                        return True

                return False
            else:
                # Single action - assume 0, 1 are defensive
                action_int = int(action.item() if hasattr(action, "item") else action)
                return action_int in [0, 1]

        except Exception as e:
            print(f"   ⚠️ Defensive action check failed: {e}")
            return False

    def _calculate_action_variety_score(self) -> float:
        """Calculate action variety score (0 to 1, higher = more diverse) - FIXED for multi-button"""
        try:
            if len(self._action_variety_history) < 3:
                return 0.5  # Neutral score for insufficient data

            # Actions are already converted to hashable values in _track_action_variety
            # No need for further conversion
            action_hashes = list(self._action_variety_history)

            # Calculate unique actions in recent history
            unique_actions = len(set(action_hashes))
            total_actions = len(action_hashes)

            # Variety score (more unique actions = higher score)
            variety_score = unique_actions / total_actions

            # Bonus for avoiding consecutive repetition
            consecutive_penalty = min(self._consecutive_action_count - 1, 3) * 0.1
            variety_score = max(0.0, variety_score - consecutive_penalty)

            return variety_score

        except Exception as e:
            print(f"   ⚠️ Variety score calculation failed: {e}")
            print(
                f"   Action history sample: {list(self._action_variety_history)[-3:]}"
            )
            return 0.5

    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict]:
        """Reset with comprehensive state initialization"""
        try:
            self._log_round_summary()

            obs, info = self.env.reset(**kwargs)

            initial_health = info.get("health", MAX_HEALTH)
            initial_enemy_health = info.get("enemy_health", MAX_HEALTH)
            initial_score = info.get("score", 0)
            initial_round = info.get("round", 1)

            print(
                f"   🔄 RESET: P:{initial_health}, E:{initial_enemy_health}, Score:{initial_score}, Round:{initial_round}"
            )

            # Reset score momentum tracking
            self._last_score_for_momentum = initial_score

            # Reset game state
            self.prev_player_health = initial_health
            self.prev_enemy_health = initial_enemy_health
            self.prev_score = initial_score
            self.player_wins_in_episode = 0
            self.enemy_wins_in_episode = 0

            # Clear histories
            self.health_history.clear()
            self.action_history.clear()
            self._recent_damage_history.clear()
            self._last_action = 0
            self._prev_action_for_reward = 0

            # NEW: Reset anti-spam tracking
            self._consecutive_action_count = 0
            self._last_repeated_action = None
            self._action_variety_history.clear()
            self._spam_penalty_active = False

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

            if self.feature_extractor is not None:
                feature_dim = getattr(self.feature_extractor, "features_dim", 512)
                zero_vf = torch.zeros(feature_dim, device=self.device)
                zero_gs = np.zeros(11, dtype=np.float32)

                for _ in range(self.state_history_length):
                    self.visual_features_history.append(zero_vf.clone())
                    self.game_state_history.append(zero_gs.copy())

        except Exception as e:
            print(f"   ⚠️ JEPA state reset failed: {e}")

    def step(self, action) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Enhanced step function with anti-spam strategic predictions - FIXED multi-button handling"""
        try:
            # Store previous action for evaluation (action from LAST step)
            prev_action = self._last_action

            # FIXED: Handle multi-button action arrays properly
            # Don't convert to scalar - just track the action as-is
            self._track_action_variety(action)

            # Store action in history (keep original format for consistency)
            self.action_history.append(action)
            self._last_action = action  # Track for NEXT step's evaluation

            current_stacked_obs = self._get_stacked_observation()

            # Make strategic prediction if ready
            if self.enable_jepa and self.jepa_ready:
                self._make_strategic_prediction(current_stacked_obs)

            # Execute environment step - use original action directly
            obs, _, terminated, truncated, next_info = self.env.step(action)
            self.frame_buffer.append(self._preprocess_frame(obs))

            # Track damage history for better defense detection
            damage_taken = max(
                0,
                self.prev_player_health
                - next_info.get("health", self.prev_player_health),
            )
            self._recent_damage_history.append(damage_taken)

            # Check if game is active
            is_active = self._is_game_active(next_info)
            if is_active:
                self.strategic_stats["game_active_frames"] += 1
            else:
                self.strategic_stats["game_paused_frames"] += 1

            # Evaluate strategic predictions during active gameplay
            if self.prediction_from_last_step is not None and is_active:
                self._evaluate_strategic_accuracy(next_info, prev_action)

            # Calculate enhanced reward with anti-spam
            enhanced_reward = self._calculate_enhanced_reward(next_info, prev_action)

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
            self.prev_score = next_info.get("score", self.prev_score)
            self._last_score_for_momentum = next_info.get("score", self.prev_score)

            return (
                self._get_stacked_observation(),
                enhanced_reward,
                done,
                False,
                next_info,
            )

        except Exception as e:
            print(f"   ❌ Step failed: {e}")
            print(
                f"   Action type: {type(action)}, Action shape: {getattr(action, 'shape', 'no shape')}"
            )
            print(f"   Action value: {action}")
            return (
                np.zeros(self.observation_space.shape, dtype=np.uint8),
                0.0,
                True,
                False,
                {},
            )

    def _make_strategic_prediction(self, current_stacked_obs: np.ndarray):
        """Make strategic attack/defense predictions"""
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
            print(f"   ⚠️ Strategic prediction failed: {e}")
            self.prediction_from_last_step = None

    def _is_good_attack_timing(
        self, damage_dealt: int, action, game_state: Dict
    ) -> bool:
        """Determine if this was good attack timing"""
        try:
            player_health = game_state.get("health", self.prev_player_health)
            enemy_health = game_state.get("enemy_health", self.prev_enemy_health)

            # Good attack timing indicators:
            if damage_dealt > 5:  # Significant damage
                return True
            if (
                damage_dealt > 0 and enemy_health < CRITICAL_HEALTH_THRESHOLD
            ):  # Finish move
                return True
            if (
                damage_dealt > 0 and player_health > enemy_health * 1.3
            ):  # Health advantage
                return True
            if damage_dealt > 0 and enemy_health > MAX_HEALTH * 0.7:  # Early aggression
                return True

            return False

        except Exception:
            return False

    def _is_good_defense_timing(
        self, damage_taken: int, action, game_state: Dict
    ) -> bool:
        """
        FIXED: Much more restrictive defense timing to prevent spam
        Only reward truly exceptional defensive plays
        """
        try:
            player_health = game_state.get("health", self.prev_player_health)
            enemy_health = game_state.get("enemy_health", self.prev_enemy_health)

            # CRITICAL: Only reward defense in truly dangerous situations

            # Scenario 1: Perfect Avoidance in CRITICAL Danger Only
            if damage_taken == 0:
                # 1a: Successfully avoided damage while at VERY critical health (more restrictive)
                if (
                    player_health < CRITICAL_HEALTH_THRESHOLD * 0.7
                ):  # Even more critical than before
                    # Additional check: enemy must be healthy enough to be threatening
                    if enemy_health > CRITICAL_HEALTH_THRESHOLD:
                        print(
                            f"      🛡️ Exceptional Defense (Critical Avoid): Very low HP, avoided damage"
                        )
                        return True

                # 1b: Avoided damage while enemy has SIGNIFICANT health advantage (more restrictive)
                if enemy_health > player_health * 1.5:  # Increased from 1.3 to 1.5
                    # Must also have recent damage pattern indicating active combat
                    if (
                        hasattr(self, "_recent_damage_history")
                        and len(self._recent_damage_history) >= 2
                    ):
                        recent_damages = list(self._recent_damage_history)
                        recent_damage = sum(recent_damages[-2:])  # Last 2 steps only
                        if recent_damage > 8:  # Increased threshold from 5 to 8
                            print(
                                f"      🛡️ Exceptional Defense (Major Disadvantage): Enemy way ahead, recent combat"
                            )
                            return True

            # Scenario 2: REMOVED - No longer reward damage mitigation to prevent spam
            # This was too generous and led to spam

            # Scenario 3: Counter-Defense (enemy hurt themselves)
            if damage_taken == 0 and enemy_health < self.prev_enemy_health:
                enemy_damage = self.prev_enemy_health - enemy_health
                if enemy_damage > 3:  # Only if enemy took significant damage
                    print(
                        f"      🛡️ Exceptional Defense (Counter): Avoided damage, enemy took {enemy_damage}"
                    )
                    return True

            # NEW Scenario 4: Perfect defense during critical moments
            # Only when player is both low health AND being actively attacked
            if (
                damage_taken == 0
                and player_health < CRITICAL_HEALTH_THRESHOLD
                and hasattr(self, "_recent_damage_history")
            ):

                # Check if we were taking consistent damage recently (under attack)
                recent_damages = list(self._recent_damage_history)
                if len(recent_damages) >= 3:
                    recent_damage_total = sum(recent_damages[-3:])
                    # Only reward if we were taking significant damage recently but now avoided it
                    if recent_damage_total > 10:
                        print(
                            f"      🛡️ Exceptional Defense (Critical Recovery): Broke damage pattern"
                        )
                        return True

            return False

        except Exception as e:
            print(f"   ⚠️ Defense timing evaluation failed: {e}")
            return False

    def _evaluate_strategic_accuracy(self, next_info: Dict, action):
        """Evaluate strategic prediction accuracy"""
        try:
            player_health_after = next_info.get("health", self.prev_player_health)
            enemy_health_after = next_info.get("enemy_health", self.prev_enemy_health)

            damage_to_enemy = max(0, self.prev_enemy_health - enemy_health_after)
            damage_to_player = max(0, self.prev_player_health - player_health_after)

            # Determine ground truth for strategic decisions
            was_good_attack_timing = self._is_good_attack_timing(
                damage_to_enemy, action, next_info
            )
            was_good_defense_timing = self._is_good_defense_timing(
                damage_to_player, action, next_info
            )

            ground_truth = {
                "is_best_time_to_attack": float(was_good_attack_timing),
                "is_best_time_to_defend": float(was_good_defense_timing),
            }

            # Evaluate strategic predictions
            num_predictions = 0
            for key, pred_tensor in self.prediction_from_last_step.items():
                if key in ground_truth:
                    try:
                        predicted_prob = pred_tensor[0, 0].item()
                        threshold = PREDICTION_THRESHOLDS.get(key, 0.5)

                        predicted_label = 1 if predicted_prob > threshold else 0
                        actual_label = int(ground_truth[key])
                        is_correct = predicted_label == actual_label

                        # Log significant strategic events
                        if key == "is_best_time_to_attack" and actual_label == 1:
                            result_str = "✅ CORRECT" if is_correct else "❌ MISSED"
                            print(
                                f"      ⚔️ ATTACK OPPORTUNITY: {result_str}, prob={predicted_prob:.3f}"
                            )

                        elif key == "is_best_time_to_defend" and actual_label == 1:
                            result_str = "✅ CORRECT" if is_correct else "❌ MISSED"
                            print(
                                f"      🛡️ DEFENSE OPPORTUNITY: {result_str}, prob={predicted_prob:.3f}"
                            )

                        # Update statistics
                        if is_correct:
                            self.strategic_stats["strategic_correct"][key] += 1

                        self.strategic_stats["strategic_predictions"][key] += 1
                        num_predictions += 1

                        # Update accuracy with adaptive learning rate
                        current_accuracy = float(is_correct)
                        self.strategic_stats["strategic_accuracies"][key] = (
                            1 - ACCURACY_LEARNING_RATE
                        ) * self.strategic_stats["strategic_accuracies"][
                            key
                        ] + ACCURACY_LEARNING_RATE * current_accuracy

                    except Exception as e:
                        print(f"   ⚠️ Strategic evaluation error for {key}: {e}")
                        continue

            # Update prediction count and periodic reporting
            if num_predictions > 0:
                self.strategic_stats["strategic_predictions_made"] += 1

                if self.strategic_stats["strategic_predictions_made"] % 1000 == 0:
                    self._print_strategic_stats()

        except Exception as e:
            print(f"   ⚠️ Strategic accuracy evaluation failed: {e}")

    def _print_strategic_stats(self):
        """Print strategic prediction statistics including anti-spam metrics"""
        try:
            print(
                f"\n   📊 Strategic JEPA Report (after {self.strategic_stats['strategic_predictions_made']} predictions):"
            )

            for outcome, accuracy in self.strategic_stats[
                "strategic_accuracies"
            ].items():
                total_preds = self.strategic_stats["strategic_predictions"][outcome]
                total_correct = self.strategic_stats["strategic_correct"][outcome]

                if total_preds > 0:
                    readable_name = outcome.replace("is_best_time_to_", "").title()
                    print(
                        f"      {readable_name} Timing: {accuracy*100:.1f}% ({total_correct}/{total_preds})"
                    )

            # NEW: Anti-spam statistics
            print(f"   🚫 Anti-Spam Report:")
            print(
                f"      Spam Penalties: {self.strategic_stats['spam_penalties_applied']}"
            )
            print(
                f"      Defensive Spam Detected: {self.strategic_stats['defensive_spam_detected']}"
            )
            print(
                f"      Action Variety Rewards: {self.strategic_stats['action_variety_rewards']}"
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

        except Exception as e:
            print(f"   ⚠️ Strategic stats printing failed: {e}")

    def _calculate_enhanced_reward(self, next_info: Dict, prev_action: int) -> float:
        """FIXED: Calculate enhanced reward with anti-spam penalties and balanced defense"""
        try:
            my_health = next_info.get("health", self.prev_player_health)
            enemy_health = next_info.get("enemy_health", self.prev_enemy_health)
            my_score = next_info.get("score", self.prev_score)

            damage_dealt = max(0, self.prev_enemy_health - enemy_health)
            damage_taken = max(0, self.prev_player_health - my_health)
            score_delta = max(0, my_score - self.prev_score)

            # Base survival reward
            survival_reward = 0.01

            # Health-based rewards (balanced)
            health_reward = 0.0
            if damage_dealt > 0:
                health_reward += damage_dealt * 0.5 / MAX_HEALTH
            if damage_taken > 0:
                health_reward -= damage_taken * 0.3 / MAX_HEALTH

            # Score-based reward
            score_reward = score_delta * 0.001 if score_delta > 0 else 0

            # FIXED: Much more restrictive defense reward with better action handling
            defense_reward = 0.0
            was_good_defense = self._is_good_defense_timing(
                damage_taken, prev_action, next_info
            )
            if was_good_defense:
                defense_reward = 1.0  # REDUCED from 2.0 to 1.0

            # NEW: Action variety reward/penalty system
            variety_reward = 0.0
            action_variety_score = self._calculate_action_variety_score()

            # Reward variety above threshold
            if (
                action_variety_score > ACTION_VARIETY_REWARD_THRESHOLD + 0.3
            ):  # 0.4+ variety score
                variety_reward = 0.2  # Small bonus for good variety
                self.strategic_stats["action_variety_rewards"] += 1

            # NEW: Spam penalty
            spam_penalty = 0.0
            if self._spam_penalty_active:
                spam_penalty = -0.5  # Significant penalty for spamming
                # Additional penalty for defensive spam specifically
                # FIXED: Handle multi-button prev_action properly
                is_defensive = self._is_defensive_action(prev_action)
                if is_defensive:
                    spam_penalty -= 0.3  # Extra penalty for defensive spam

            # Strategic multipliers
            if damage_dealt > 0:
                health_ratio = my_health / (enemy_health + 1e-6)
                if health_ratio > 1.5:
                    health_reward *= 1.3
                if my_health < MAX_HEALTH * 0.4:
                    health_reward *= 1.5

            # Win/Loss Rewards - Keep existing logic
            win_loss_reward = 0.0
            if my_health <= 0 and self.prev_player_health > 0:
                win_loss_reward = -5.0
                self.enemy_wins_in_episode += 1
                print(f"   💀 PLAYER DIED! Enemy wins: {self.enemy_wins_in_episode}")
            elif enemy_health <= 0 and self.prev_enemy_health > 0:
                win_loss_reward = 10.0
                self.player_wins_in_episode += 1
                print(
                    f"   🏆 ENEMY DEFEATED! Player wins: {self.player_wins_in_episode}"
                )
            elif (my_health <= 0 or enemy_health <= 0) and (
                self.prev_player_health > 0 and self.prev_enemy_health > 0
            ):
                if my_health > enemy_health:
                    win_loss_reward = 10.0
                    self.player_wins_in_episode += 1
                    print(
                        f"   🏆 ROUND WON! Player wins: {self.player_wins_in_episode}"
                    )
                else:
                    win_loss_reward = -5.0
                    self.enemy_wins_in_episode += 1
                    print(f"   💀 ROUND LOST! Enemy wins: {self.enemy_wins_in_episode}")

            # FIXED: Total reward calculation with anti-spam components
            total_reward = (
                survival_reward
                + health_reward
                + score_reward
                + win_loss_reward
                + defense_reward
                + variety_reward
                + spam_penalty
            )

            # # Debug logging for significant events
            # if (
            #     win_loss_reward != 0
            #     or damage_dealt > 5
            #     or damage_taken > 5
            #     or defense_reward > 0
            #     or spam_penalty < 0
            #     or variety_reward > 0
            # ):
            #     print(
            #         f"   💰 Reward: {total_reward:.3f} (dmg_dealt:{damage_dealt}, dmg_taken:{damage_taken}, "
            #         f"defense:{defense_reward:.1f}, variety:{variety_reward:.2f}, spam:{spam_penalty:.2f}, win_loss:{win_loss_reward:.1f})"
            #     )

            return float(total_reward)

        except Exception as e:
            print(f"   ⚠️ Reward calculation failed: {e}")
            return 0.01

    def _handle_episode_termination(
        self, terminated: bool, truncated: bool, next_info: Dict
    ) -> bool:
        """Handle episode termination with proper win/loss detection"""
        try:
            done = terminated or truncated

            current_player = next_info.get("health", self.prev_player_health)
            current_enemy = next_info.get("enemy_health", self.prev_enemy_health)

            match_ended = False

            if self.player_wins_in_episode >= 2:
                match_ended = True
                self.current_stats["wins"] += 1
                print(
                    f"   🎉 MATCH WON! Player got {self.player_wins_in_episode} rounds. Total match wins: {self.current_stats['wins']}"
                )

            elif self.enemy_wins_in_episode >= 2:
                match_ended = True
                self.current_stats["losses"] += 1
                print(
                    f"   😞 MATCH LOST! Enemy got {self.enemy_wins_in_episode} rounds. Total match losses: {self.current_stats['losses']}"
                )

            elif done and not match_ended:
                if self.player_wins_in_episode > self.enemy_wins_in_episode:
                    self.current_stats["wins"] += 1
                    print(
                        f"   🎉 EPISODE WIN! Player rounds: {self.player_wins_in_episode} vs Enemy: {self.enemy_wins_in_episode}"
                    )
                elif self.enemy_wins_in_episode > self.player_wins_in_episode:
                    self.current_stats["losses"] += 1
                    print(
                        f"   😞 EPISODE LOSS! Player rounds: {self.player_wins_in_episode} vs Enemy: {self.enemy_wins_in_episode}"
                    )
                else:
                    if current_player > current_enemy:
                        self.current_stats["wins"] += 1
                        print(
                            f"   🎉 HEALTH WIN! P:{current_player} > E:{current_enemy}"
                        )
                    else:
                        self.current_stats["losses"] += 1
                        print(
                            f"   😞 HEALTH LOSS! P:{current_player} <= E:{current_enemy}"
                        )
                match_ended = True

            if match_ended:
                done = True
                self.current_stats["total_rounds"] = (
                    self.current_stats["wins"] + self.current_stats["losses"]
                )
                if self.current_stats["total_rounds"] > 0:
                    self.current_stats["win_rate"] = (
                        self.current_stats["wins"] / self.current_stats["total_rounds"]
                    )

                print(
                    f"   📊 UPDATED MATCH STATS: {self.current_stats['wins']}W/{self.current_stats['losses']}L (WR: {self.current_stats['win_rate']*100:.1f}%)"
                )

            return done

        except Exception as e:
            print(f"   ⚠️ Episode termination handling failed: {e}")
            return terminated or truncated

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
            # NEW: Clean up anti-spam tracking
            if hasattr(self, "_action_variety_history"):
                self._action_variety_history.clear()

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
