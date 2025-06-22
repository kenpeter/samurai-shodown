import gymnasium as gym
from gymnasium import spaces
from collections import deque
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from typing import Dict, Tuple, Optional
import math
import os  # Add this import


class NCSOFTMultiAgentWrapper(gym.Wrapper):
    """
    NCSOFT Multi-Agent Fighting Game Wrapper - BREAKTHROUGH SOLUTION

    Based on "Creating Pro-Level AI for a Real-Time Fighting Game Using Deep Reinforcement Learning"
    NCSOFT research that achieved 62% win rate against professional players

    Key Features:
    - Multi-style reward shaping (aggressive/defensive/balanced)
    - Data skipping techniques (passive no-op filtering, move maintenance)
    - Cross-style training support
    - Breakthrough-optimized reward system
    """

    def __init__(
        self,
        env,
        agent_style="balanced",  # NEW: Fighting style
        reset_round=True,
        rendering=False,
        max_episode_steps=15000,
        frame_stack=4,
        frame_skip=4,
        target_size=(180, 126),
        skip_passive_noop=True,  # NCSOFT: Skip forced no-ops
        maintain_move_ticks=10,  # NCSOFT: Maintain moves for consistency
        breakthrough_mode=True,  # Enhanced rewards for 30%+ breakthrough
    ):
        super().__init__(env)

        # Basic parameters
        self.reset_round = reset_round
        self.rendering = rendering
        self.max_episode_steps = max_episode_steps
        self.frame_stack = frame_stack
        self.frame_skip = frame_skip
        self.target_size = target_size

        # NCSOFT Multi-Agent Parameters
        self.agent_style = agent_style
        self.skip_passive_noop = skip_passive_noop
        self.maintain_move_ticks = maintain_move_ticks
        self.breakthrough_mode = breakthrough_mode

        # NCSOFT Move Maintenance
        self.move_countdown = 0
        self.last_move_action = None

        # Health and game constants
        self.full_hp = 128

        # NCSOFT Data Tracking
        self.forced_noop_count = 0
        self.strategic_noop_count = 0
        self.move_consistency_score = 0.0

        # Performance tracking
        self.current_stats = {
            "wins": 0,
            "losses": 0,
            "total_rounds": 0,
            "win_rate": 0.0,
            "current_episode_reward": 0.0,
            "best_win_streak": 0,
            "current_win_streak": 0,
            "avg_episode_length": 0.0,
            "breakthrough_progress": 0.0,
            "style_metrics": self._init_style_metrics(),
        }

        # Episode and combat tracking
        self.step_count = 0
        self.episode_count = 0
        self.prev_player_health = None
        self.prev_enemy_health = None
        self.prev_action = None
        self.consecutive_same_action = 0
        self.combo_length = 0
        self.last_damage_step = 0

        # NCSOFT Style-Specific Reward System
        self.reward_weights = self._get_ncsoft_reward_weights()
        self.reward_scale = 10.0

        # Enhanced reward weighting for breakthrough
        if self.breakthrough_mode:
            self.process_weight = 0.85
            self.outcome_weight = 0.15
        else:
            self.process_weight = 0.9
            self.outcome_weight = 0.1

        # Frame buffer setup
        self.frame_buffer = deque(maxlen=self.frame_stack)
        dummy_frame = np.zeros(
            (self.target_size[1], self.target_size[0], 3), dtype=np.uint8
        )
        for _ in range(self.frame_stack):
            self.frame_buffer.append(dummy_frame)

        # Observation space
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

        print(f"🥊 NCSOFT MULTI-AGENT WRAPPER - {self.agent_style.upper()} STYLE")
        print(f"   📊 Observation: {self.observation_space.shape}")
        print(f"   🚫 Skip passive no-ops: {self.skip_passive_noop}")
        print(f"   🔄 Move maintenance: {self.maintain_move_ticks} ticks")
        print(f"   💪 Breakthrough mode: {self.breakthrough_mode}")
        print(f"   🎯 Target: Break through 30% plateau → 50%+")

    def _init_style_metrics(self):
        """Initialize style-specific performance metrics"""
        return {
            "aggressive_score": 0.0,
            "defensive_score": 0.0,
            "consistency_score": 0.0,
            "adaptation_score": 0.0,
            "finishing_rate": 0.0,
        }

    def _get_ncsoft_reward_weights(self):
        """
        NCSOFT Style-Specific Reward Weights
        Based on the paper's reward shaping for different fighting styles
        """
        if self.agent_style == "aggressive":
            weights = {
                "time_penalty": 0.008,  # High urgency (NCSOFT paper)
                "damage_bonus": 7.0,  # Enhanced for breakthrough
                "combo_bonus": 0.4,  # High combo focus
                "finishing_bonus": 4.0,  # Big finishing rewards
                "distance_penalty": 0.003,  # Punish keeping distance
                "engagement_bonus": 0.5,  # Reward aggressive play
                "hp_self_weight": 5.0,
                "hp_enemy_weight": 6.0,  # Focus on enemy damage
            }
        elif self.agent_style == "defensive":
            weights = {
                "time_penalty": 0.0,  # No time pressure
                "damage_bonus": 5.0,  # Moderate damage focus
                "combo_bonus": 0.15,  # Lower combo focus
                "preservation_bonus": 0.4,  # HP preservation
                "defensive_bonus": 0.3,  # Reward defensive play
                "distance_bonus": 0.1,  # Reward distance control
                "hp_self_weight": 7.0,  # High self-HP priority
                "hp_enemy_weight": 4.0,
            }
        else:  # balanced
            weights = {
                "time_penalty": 0.004,  # Moderate time pressure
                "damage_bonus": 6.0,  # Balanced damage focus
                "combo_bonus": 0.25,  # Moderate combo focus
                "adaptive_bonus": 0.2,  # Situation adaptation
                "balance_bonus": 0.2,  # Reward balanced play
                "hp_self_weight": 5.5,  # Slight self-HP bias
                "hp_enemy_weight": 5.0,
            }

        # Breakthrough mode enhancements
        if self.breakthrough_mode:
            for key in weights:
                if "bonus" in key:
                    weights[key] *= 1.3  # 30% bonus increase

        return weights

    def _preprocess_frame(self, frame):
        """Preprocess frame maintaining RGB"""
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            frame = cv2.resize(frame, self.target_size, interpolation=cv2.INTER_AREA)
        else:
            if len(frame.shape) == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            frame = cv2.resize(frame, self.target_size, interpolation=cv2.INTER_AREA)
        return frame.astype(np.uint8)

    def _get_stacked_observation(self):
        """Create stacked observation from frame buffer"""
        stacked = np.concatenate(list(self.frame_buffer), axis=2)
        stacked = np.transpose(stacked, (2, 0, 1))
        stacked = np.transpose(stacked, (0, 2, 1))
        return stacked

    def _extract_game_state(self, info):
        """Extract game state information"""
        return {
            "player_health": info.get("health", self.full_hp),
            "opponent_health": info.get("enemy_health", self.full_hp),
            "current_round": info.get("round", 1),
            "score": info.get("score", 0),
        }

    def _detect_passive_noop(self, action, game_info):
        """
        NCSOFT Technique: Detect passive vs strategic no-ops
        Optimized for 12-dimensional fighting game action space
        """
        if not self.skip_passive_noop:
            return False

        # Check if action is no-op for 12-dimensional action space
        is_noop = False
        try:
            if isinstance(action, np.ndarray) and len(action) == 12:
                # No-op means all buttons are 0 (no inputs)
                is_noop = np.all(action < 0.1)  # Use threshold for floating point
            else:
                is_noop = (
                    np.allclose(action, 0)
                    if isinstance(action, np.ndarray)
                    else (action == 0)
                )
        except (ValueError, TypeError):
            is_noop = False

        if not is_noop:
            return False

        # Check if player is in forced state
        player_health = game_info.get("player_health", self.full_hp)
        if player_health <= 0:
            self.forced_noop_count += 1
            return True  # Forced no-op, skip it

        # Strategic no-op
        self.strategic_noop_count += 1
        return False

    def _apply_move_maintenance(self, action):
        """
        NCSOFT Technique: Maintain move decisions for consistency
        Optimized for 12-dimensional fighting game action space
        """
        if self.maintain_move_ticks <= 1:
            return action

        # Handle 12-dimensional action space (fighting game buttons)
        try:
            if isinstance(action, np.ndarray) and len(action) == 12:
                # For fighting games, movement is usually in first few dimensions
                # Assuming indices 0-3 might be directional movement
                current_move = action[:4]  # First 4 dimensions for movement

                # Apply move maintenance
                if self.move_countdown > 0:
                    if (
                        self.last_move_action is not None
                        and len(self.last_move_action) == 4
                    ):
                        action = action.copy()
                        action[:4] = self.last_move_action  # Maintain movement
                        self.move_consistency_score += 0.1
                    self.move_countdown -= 1
                else:
                    # Start new move sequence if there's significant movement
                    if np.any(current_move > 0.5):  # Any movement button pressed
                        self.move_countdown = self.maintain_move_ticks - 1
                        self.last_move_action = current_move.copy()

        except (IndexError, TypeError, ValueError):
            # If there's any issue, just return original action
            pass

        return action

    def _calculate_ncsoft_process_reward(self, game_info, action, info):
        """
        NCSOFT Multi-Style Process Reward System
        Enhanced for breakthrough training
        """
        reward = 0.0
        player_health = game_info["player_health"]
        enemy_health = game_info["opponent_health"]

        if self.prev_player_health is None:
            self.prev_player_health = player_health
            self.prev_enemy_health = enemy_health
            self.prev_action = action
            return 0.0

        health_diff = self.prev_player_health - player_health
        enemy_health_diff = self.prev_enemy_health - enemy_health

        # 1. ENHANCED Damage System
        if enemy_health_diff > 0:
            normalized_damage = enemy_health_diff / self.full_hp
            damage_reward = (
                normalized_damage
                * self.reward_weights["damage_bonus"]
                * self.reward_scale
            )
            reward += damage_reward

            # NCSOFT Combo System
            if self.step_count - self.last_damage_step <= 8:  # Extended combo window
                self.combo_length += 1
                combo_bonus = (
                    min(self.combo_length * self.reward_weights["combo_bonus"], 3.0)
                    * self.reward_scale
                )
                reward += combo_bonus

                # Update style metrics
                self.current_stats["style_metrics"]["aggressive_score"] += 0.2
            else:
                self.combo_length = 1
            self.last_damage_step = self.step_count

            # BREAKTHROUGH: Enhanced Finishing System
            if "finishing_bonus" in self.reward_weights and enemy_health <= 15:
                finishing_multiplier = (15 - enemy_health) / 15  # Scale with low health
                finishing_bonus = (
                    self.reward_weights["finishing_bonus"]
                    * finishing_multiplier
                    * self.reward_scale
                )
                reward += finishing_bonus
                self.current_stats["style_metrics"]["finishing_rate"] += 0.1

        # 2. Style-Specific Damage Penalties
        if health_diff > 0:
            normalized_damage_taken = health_diff / self.full_hp
            if self.agent_style == "defensive":
                damage_penalty = (
                    normalized_damage_taken * 2.5 * self.reward_scale
                )  # Higher penalty
                self.current_stats["style_metrics"]["defensive_score"] += 0.1
            else:
                damage_penalty = normalized_damage_taken * 1.5 * self.reward_scale
            reward -= damage_penalty
            self.combo_length = 0

        # 3. NCSOFT Health Advantage System
        health_advantage = (player_health - enemy_health) / self.full_hp
        self_hp_weight = self.reward_weights["hp_self_weight"]
        enemy_hp_weight = self.reward_weights["hp_enemy_weight"]

        advantage_reward = (
            health_advantage
            * (self_hp_weight - enemy_hp_weight)
            * 0.15
            * self.reward_scale
        )
        reward += advantage_reward

        # 4. Enhanced Action Diversity
        if isinstance(action, np.ndarray):
            current_action = tuple(action.flatten())
        else:
            current_action = action

        if self.prev_action is not None:
            try:
                # Safe comparison for arrays
                if isinstance(current_action, tuple) and isinstance(
                    self.prev_action, tuple
                ):
                    actions_equal = current_action == self.prev_action
                elif isinstance(current_action, np.ndarray) and isinstance(
                    self.prev_action, np.ndarray
                ):
                    actions_equal = np.array_equal(current_action, self.prev_action)
                else:
                    actions_equal = current_action == self.prev_action

                if actions_equal:
                    self.consecutive_same_action += 1
                    if self.consecutive_same_action > 2:
                        diversity_penalty = -0.25 * self.reward_scale
                        reward += diversity_penalty
                else:
                    self.consecutive_same_action = 0
                    diversity_bonus = 0.2 * self.reward_scale
                    reward += diversity_bonus
                    self.current_stats["style_metrics"]["adaptation_score"] += 0.05

            except (ValueError, TypeError) as e:
                # Handle any comparison errors gracefully
                self.consecutive_same_action = 0
                diversity_bonus = 0.1 * self.reward_scale
                reward += diversity_bonus

        # 5. Style-Specific Bonuses
        if "engagement_bonus" in self.reward_weights and (
            enemy_health_diff > 0 or health_diff > 0
        ):
            engagement_bonus = (
                self.reward_weights["engagement_bonus"] * self.reward_scale
            )
            reward += engagement_bonus

        if "preservation_bonus" in self.reward_weights:
            player_health_ratio = player_health / self.full_hp
            if player_health_ratio > 0.75:
                preservation_bonus = (
                    self.reward_weights["preservation_bonus"] * self.reward_scale
                )
                reward += preservation_bonus

        if "adaptive_bonus" in self.reward_weights:
            # Reward adaptation based on health difference
            adaptation_factor = (
                abs(health_advantage)
                * self.reward_weights["adaptive_bonus"]
                * self.reward_scale
            )
            reward += adaptation_factor

        # 6. NCSOFT Time Penalty (Style-Specific)
        time_penalty = -self.reward_weights["time_penalty"] * self.reward_scale
        reward += time_penalty

        # 7. BREAKTHROUGH: Pressure Bonus
        if self.breakthrough_mode and enemy_health < player_health:
            pressure_bonus = (
                ((player_health - enemy_health) / self.full_hp)
                * 0.3
                * self.reward_scale
            )
            reward += pressure_bonus

        # Update tracking
        self.prev_player_health = player_health
        self.prev_enemy_health = enemy_health
        self.prev_action = current_action

        return reward

    def _calculate_ncsoft_outcome_reward(self, game_info, info):
        """NCSOFT Enhanced Outcome Rewards for Breakthrough"""
        player_health = game_info["player_health"]
        enemy_health = game_info["opponent_health"]

        round_over = player_health <= 0 or enemy_health <= 0

        if round_over:
            if player_health > enemy_health:
                # BREAKTHROUGH Win Rewards
                health_bonus = (player_health / self.full_hp) * 10.0  # Increased
                perfect_bonus = 8.0 if player_health == self.full_hp else 0

                # Style-specific win bonuses
                style_bonus = 0.0
                if self.agent_style == "aggressive":
                    style_bonus = 5.0 + (
                        self.combo_length * 0.5
                    )  # Reward aggressive wins
                elif self.agent_style == "defensive":
                    style_bonus = health_bonus * 0.8  # Reward health preservation
                else:  # balanced
                    style_bonus = 3.0 + (health_bonus * 0.3)  # Balanced bonus

                # BREAKTHROUGH multiplier
                breakthrough_multiplier = 1.4 if self.breakthrough_mode else 1.0

                win_reward = (
                    (30.0 + health_bonus + perfect_bonus + style_bonus)
                    * breakthrough_multiplier
                    * self.reward_scale
                )

                # Update stats
                self.current_stats["wins"] += 1
                self.current_stats["current_win_streak"] += 1
                self.current_stats["best_win_streak"] = max(
                    self.current_stats["best_win_streak"],
                    self.current_stats["current_win_streak"],
                )

                # Update breakthrough progress
                self.current_stats["breakthrough_progress"] = min(
                    self.current_stats["breakthrough_progress"] + 0.02, 1.0
                )

                return win_reward

            elif enemy_health > player_health:
                # Reduced loss penalty to encourage risk-taking
                health_disadvantage = (enemy_health - player_health) / self.full_hp
                loss_penalty = -(10.0 + health_disadvantage * 2.5) * self.reward_scale

                # Update stats
                self.current_stats["losses"] += 1
                self.current_stats["current_win_streak"] = 0
                return loss_penalty
            else:
                return 3.0 * self.reward_scale  # Draw bonus

        return 0.0

    def _update_statistics(self, process_reward, outcome_reward):
        """Update comprehensive statistics"""
        total_reward = process_reward + outcome_reward
        self.current_stats["current_episode_reward"] += total_reward

        # Update round statistics
        if outcome_reward != 0:
            self.current_stats["total_rounds"] += 1
            if self.current_stats["total_rounds"] > 0:
                self.current_stats["win_rate"] = (
                    self.current_stats["wins"] / self.current_stats["total_rounds"]
                )

        # Update consistency metrics
        if self.step_count > 0:
            self.current_stats["style_metrics"]["consistency_score"] = (
                self.move_consistency_score / self.step_count
            )

    def reset(self, **kwargs):
        """Reset with NCSOFT tracking"""
        obs, info = self.env.reset(**kwargs)

        # Reset episode tracking
        self.step_count = 0
        self.episode_count += 1

        # Reset game state
        self.prev_player_health = None
        self.prev_enemy_health = None
        self.prev_action = None
        self.consecutive_same_action = 0

        # Reset combat tracking
        self.combo_length = 0
        self.last_damage_step = 0

        # Reset NCSOFT specific tracking
        self.move_countdown = 0
        self.last_move_action = None
        self.forced_noop_count = 0
        self.strategic_noop_count = 0
        self.move_consistency_score = 0.0

        # Reset episode reward
        self.current_stats["current_episode_reward"] = 0.0

        # Reset frame buffer
        processed_frame = self._preprocess_frame(obs)
        self.frame_buffer.clear()
        for _ in range(self.frame_stack):
            self.frame_buffer.append(processed_frame)

        return self._get_stacked_observation(), info

    def step(self, action):
        """Execute action with NCSOFT multi-agent techniques"""

        # Apply NCSOFT move maintenance
        action = self._apply_move_maintenance(action)

        total_reward = 0.0
        info = {}

        # Execute action with frame skipping
        for _ in range(self.frame_skip):
            obs, reward, terminated, truncated, step_info = self.env.step(action)
            total_reward += reward
            info.update(step_info)
            if terminated or truncated:
                break

        # Process frame
        processed_frame = self._preprocess_frame(obs)
        self.frame_buffer.append(processed_frame)

        # Get game state
        game_info = self._extract_game_state(info)

        # NCSOFT: Check for passive no-op
        is_passive_noop = self._detect_passive_noop(action, game_info)

        # Calculate NCSOFT rewards
        process_reward = self._calculate_ncsoft_process_reward(game_info, action, info)
        outcome_reward = self._calculate_ncsoft_outcome_reward(game_info, info)

        # Combine rewards
        combined_reward = (
            self.process_weight * process_reward + self.outcome_weight * outcome_reward
        )

        # Update statistics
        self._update_statistics(process_reward, outcome_reward)

        # Episode termination
        done = (
            self.step_count >= self.max_episode_steps
            or terminated
            or truncated
            or (game_info["player_health"] <= 0 and game_info["opponent_health"] <= 0)
        )

        # Get observation
        stacked_obs = self._get_stacked_observation()
        self.step_count += 1

        # Enhanced info for NCSOFT system
        info.update(
            {
                "game_info": game_info,
                "process_reward": process_reward,
                "outcome_reward": outcome_reward,
                "combined_reward": combined_reward,
                "agent_style": self.agent_style,
                "skip_training": is_passive_noop,
                "passive_noop": is_passive_noop,
                "win_rate": self.current_stats["win_rate"],
                "breakthrough_progress": self.current_stats["breakthrough_progress"],
                "style_metrics": self.current_stats["style_metrics"].copy(),
                "forced_noop_ratio": self.forced_noop_count / max(1, self.step_count),
                "strategic_noop_ratio": self.strategic_noop_count
                / max(1, self.step_count),
                "move_consistency": self.current_stats["style_metrics"][
                    "consistency_score"
                ],
            }
        )

        return stacked_obs, combined_reward, done, False, info


# Simple CNN Feature Extractor
class NCSOFTSimpleCNN(BaseFeaturesExtractor):
    """Simple CNN optimized for NCSOFT multi-agent training"""

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 512):
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]

        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )

        with torch.no_grad():
            sample = torch.zeros(1, n_input_channels, 126, 180)
            cnn_output_size = self.cnn(sample).shape[1]

        self.classifier = nn.Sequential(
            nn.Linear(cnn_output_size, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(512, features_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        x = observations.float() / 255.0
        features = self.cnn(x)
        return self.classifier(features)


# Utility function
def make_ncsoft_env(
    game="SamuraiShodown-Genesis",
    state=None,
    agent_style="balanced",
    reset_round=True,
    rendering=False,
    max_episode_steps=15000,
    breakthrough_mode=True,
    use_default_state=False,  # NEW: Add this parameter
    **wrapper_kwargs,
):
    """Create NCSOFT Multi-Agent environment with proper state handling"""
    import retro

    # Handle state file properly
    if not use_default_state:
        # Try to find samurai.state file
        possible_state_paths = [
            "samurai.state",
            "./samurai.state",
            "states/samurai.state",
            os.path.join(os.getcwd(), "samurai.state"),
        ]

        state_found = None
        for state_path in possible_state_paths:
            if os.path.exists(state_path):
                state_found = os.path.abspath(state_path)
                print(f"🎮 Found game state: {state_found}")
                break

        if state_found:
            state = state_found
        else:
            print(f"⚠️ samurai.state not found in common locations:")
            for path in possible_state_paths:
                print(f"   ❌ {path}")
            print(f"   🎮 Using default state (will start from menu)")
            state = None
    else:
        print(f"🎮 Using default game state")
        state = None

    env = retro.make(
        game=game,
        state=state,
        use_restricted_actions=retro.Actions.FILTERED,
        obs_type=retro.Observations.IMAGE,
        render_mode="human" if rendering else None,
    )

    env = NCSOFTMultiAgentWrapper(
        env,
        agent_style=agent_style,
        reset_round=reset_round,
        rendering=rendering,
        max_episode_steps=max_episode_steps,
        breakthrough_mode=breakthrough_mode,
        **wrapper_kwargs,
    )

    return env
