from collections import deque
import gymnasium as gym
import numpy as np
import time
from datetime import datetime
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple, Optional


class ImplicitPRM(nn.Module):
    """
    Implicit Process Reward Model for fighting games
    Adapted from PRIME methodology for dense action-level rewards
    """

    def __init__(self, feature_dim: int = 512, action_dim: int = 12):
        super().__init__()

        self.feature_dim = feature_dim
        self.action_dim = action_dim

        # Process reward head - maps features to action-level rewards
        self.reward_head = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),  # Single reward value per action
        )

        # Outcome prediction head for implicit training
        self.outcome_head = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
            nn.Sigmoid(),  # Win probability
        )

        print(f"🧠 Implicit PRM initialized:")
        print(f"   📊 Feature dim: {feature_dim}")
        print(f"   🎮 Action dim: {action_dim}")
        print(f"   🎯 Process reward head: {feature_dim} → 256 → 128 → 1")
        print(f"   🏆 Outcome head: {feature_dim} → 256 → 1")

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for implicit PRM
        Args:
            features: [batch_size, feature_dim] or [batch_size, seq_len, feature_dim]
        Returns:
            process_rewards: [batch_size] or [batch_size, seq_len]
            outcome_pred: [batch_size]
        """
        if features.dim() == 3:
            # Sequence of features
            batch_size, seq_len, _ = features.shape
            features_flat = features.view(-1, self.feature_dim)

            process_rewards = self.reward_head(features_flat).squeeze(-1)
            process_rewards = process_rewards.view(batch_size, seq_len)

            # Use last timestep for outcome prediction
            outcome_pred = self.outcome_head(features[:, -1, :]).squeeze(-1)
        else:
            # Single feature vector
            process_rewards = self.reward_head(features).squeeze(-1)
            outcome_pred = self.outcome_head(features).squeeze(-1)

        return process_rewards, outcome_pred


class SamuraiShowdownCustomWrapper(gym.Wrapper):
    """Enhanced wrapper with PRIME integration - USING ONLY REAL GAME DATA"""

    def __init__(
        self,
        env,
        reset_round=True,
        rendering=False,
        max_episode_steps=12000,
        reward_coeff=1.0,
        # PRIME specific parameters
        enable_prime=True,
        process_weight=0.3,
        outcome_weight=0.7,
        prm_lr=1e-6,
        device=None,
    ):
        super(SamuraiShowdownCustomWrapper, self).__init__(env)
        self.env = env

        # Frame processing - RGB only for deep networks
        self.resize_scale = 0.75
        self.num_frames = 9
        self.frame_stack = deque(maxlen=self.num_frames)

        # Health tracking - USING ONLY REAL GAME DATA
        self.full_hp = 128  # Based on typical fighting game health (8-bit value)
        self.prev_player_health = self.full_hp
        self.prev_opponent_health = self.full_hp

        # Episode management
        self.max_episode_steps = max_episode_steps
        self.episode_steps = 0
        self.reset_round = reset_round

        # Statistics tracking
        self.wins = 0
        self.losses = 0
        self.total_rounds = 0
        self.total_episodes = 0
        self.total_steps = 0

        # Performance tracking
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
        self.win_streak = 0
        self.best_win_streak = 0

        # PRIME Integration
        self.enable_prime = enable_prime
        self.process_weight = process_weight
        self.outcome_weight = outcome_weight
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        if self.enable_prime:
            # Initialize Implicit PRM
            self.implicit_prm = ImplicitPRM(feature_dim=512, action_dim=12).to(
                self.device
            )
            self.prm_optimizer = torch.optim.AdamW(
                self.implicit_prm.parameters(), lr=prm_lr
            )

            # CRITICAL: CNN Feature Extractor reference
            self.cnn_feature_extractor = None  # Will be set by training script

            # PRIME tracking
            self.episode_features = []
            self.episode_process_rewards = []
            self.reference_features = deque(maxlen=1000)
            self.last_features = None
            self.prm_update_frequency = 4  # Update every N episodes
            self.episodes_since_prm_update = 0

            print(
                f"🎯 PRIME enabled with weights: process={process_weight}, outcome={outcome_weight}"
            )
        else:
            print(f"🎯 PRIME disabled - using simplified rewards")

        # Logging
        self.last_log_time = time.time()
        self.log_interval = 60
        self.session_start = time.time()

        # Get frame dimensions
        dummy_obs, _ = self.env.reset()
        original_height, original_width = dummy_obs.shape[:2]
        self.target_height = int(original_height * self.resize_scale)
        self.target_width = int(original_width * self.resize_scale)

        # Observation space for RGB ultra-deep network
        channels = self.num_frames * 3  # 9 frames × 3 RGB channels = 27 channels

        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(channels, self.target_height, self.target_width),
            dtype=np.uint8,
        )

        print(f"⚡ SAMURAI SHODOWN WRAPPER (CORRECTED)")
        print(f"   🎮 Action space: {self.env.action_space}")
        print(f"   🎮 Using MultiBinary action space")
        if self.enable_prime:
            print(f"   🧠 PRIME Implicit PRM enabled")
            print(
                f"   🎯 Process/Outcome weights: {process_weight:.1f}/{outcome_weight:.1f}"
            )
        print(f"   📊 Available game data:")
        print(f"      • health: player health (8-bit)")
        print(f"      • enemy_health: opponent health (8-bit)")
        print(f"      • round: current round number")
        print(f"      • score: player score (32-bit)")
        print(f"   📏 Episode length: {max_episode_steps} steps")
        print(f"   📊 Frame stack: {self.num_frames} frames")
        print(f"   🖼️  Frame size: {self.target_height}x{self.target_width}")
        print(f"   🌈 Input channels: {channels} (RGB only)")

    def set_cnn_feature_extractor(self, feature_extractor):
        """
        CRITICAL: Set the CNN feature extractor for proper PRIME integration
        This connects the same CNN used by the policy to the PRIME system
        """
        self.cnn_feature_extractor = feature_extractor
        print(f"🔗 CNN Feature Extractor connected to PRIME!")

    def _extract_cnn_features(self, observation: np.ndarray) -> torch.Tensor:
        """
        Extract CNN features using the ACTUAL feature extractor from PPO
        """
        if not self.enable_prime:
            return None

        if self.cnn_feature_extractor is None:
            # Fallback: create simple features if CNN not connected yet
            return self._extract_simple_features(observation)

        try:
            # Convert observation to tensor (same format as policy expects)
            if isinstance(observation, np.ndarray):
                obs_tensor = (
                    torch.from_numpy(observation).unsqueeze(0).to(self.device).float()
                )
            else:
                obs_tensor = observation.unsqueeze(0).to(self.device).float()

            # Extract features using the SAME CNN as the policy
            with torch.no_grad():
                features = self.cnn_feature_extractor(obs_tensor)
                return features.squeeze()  # Remove batch dimension

        except Exception as e:
            print(f"⚠️ CNN feature extraction failed: {e}")
            return self._extract_simple_features(observation)

    def _extract_simple_features(self, observation: np.ndarray) -> torch.Tensor:
        """
        Fallback: Extract simple features from observation for PRIME
        Only used if CNN feature extractor is not available yet
        """
        # Simple feature extraction: flatten and reduce observation
        obs_flat = observation.flatten()

        # Create basic feature vector
        features = np.array(
            [
                np.mean(obs_flat),  # Average pixel value
                np.std(obs_flat),  # Pixel variance
                np.max(obs_flat),  # Max pixel value
                np.min(obs_flat),  # Min pixel value
                len(obs_flat),  # Size (constant)
            ]
            + [0.0] * 507
        )  # Pad to 512 dimensions

        return torch.tensor(features[:512], dtype=torch.float32).to(self.device)

    def _calculate_implicit_process_reward(
        self, current_features: torch.Tensor
    ) -> float:
        """
        Calculate implicit process reward using PRIME methodology
        """
        if not self.enable_prime or current_features is None:
            return 0.0

        try:
            with torch.no_grad():
                # Get process reward from implicit PRM
                process_reward_raw, _ = self.implicit_prm(current_features.unsqueeze(0))
                process_reward_raw = process_reward_raw.squeeze()

                # Calculate baseline from reference features
                if len(self.reference_features) > 10:
                    ref_features_stack = torch.stack(
                        list(self.reference_features)[-50:]
                    )  # Use last 50
                    ref_rewards, _ = self.implicit_prm(ref_features_stack)
                    baseline = torch.mean(ref_rewards)
                else:
                    baseline = 0.0

                # Implicit process reward = current - baseline
                implicit_reward = float(process_reward_raw - baseline)

                # Normalize to reasonable range for fighting games
                implicit_reward = np.clip(implicit_reward, -0.5, 0.5)

                return implicit_reward
        except Exception as e:
            print(f"⚠️ Process reward calculation failed: {e}")
            return 0.0

    def _update_implicit_prm(self, episode_outcome: float):
        """
        Update implicit PRM based on episode outcome
        """
        if not self.enable_prime or len(self.episode_features) == 0:
            return None

        try:
            # Stack episode features
            features_tensor = torch.stack(self.episode_features)
            outcome_tensor = torch.tensor([episode_outcome], dtype=torch.float32).to(
                self.device
            )

            # Forward pass through PRM
            process_rewards, outcome_pred = self.implicit_prm(features_tensor)

            # Calculate loss (cross-entropy for outcome prediction)
            outcome_loss = F.binary_cross_entropy(
                outcome_pred.mean().unsqueeze(0), outcome_tensor
            )

            # Optional: Add regularization on process rewards
            process_reg = 0.01 * torch.mean(process_rewards**2)

            total_loss = outcome_loss + process_reg

            # Gradient update
            self.prm_optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.implicit_prm.parameters(), 0.5)
            self.prm_optimizer.step()

            return {
                "prm_loss": total_loss.item(),
                "outcome_loss": outcome_loss.item(),
                "outcome_pred": outcome_pred.mean().item(),
                "actual_outcome": episode_outcome,
                "avg_process_reward": torch.mean(process_rewards).item(),
            }
        except Exception as e:
            print(f"⚠️ PRM update error: {e}")
            return None

    def _extract_game_state(self, info):
        """Extract ONLY the real game state from actual Samurai Shodown data"""

        # ONLY use the actual data available in your integration:
        player_health = info.get("health", self.full_hp)
        opponent_health = info.get("enemy_health", self.full_hp)
        current_round = info.get("round", 1)
        player_score = info.get("score", 0)

        return {
            "player_health": player_health,
            "opponent_health": opponent_health,
            "round": current_round,
            "score": player_score,
        }

    def _process_frame(self, rgb_frame):
        """RGB-only frame processing for deep networks"""
        # Ensure we have RGB input
        if len(rgb_frame.shape) == 3 and rgb_frame.shape[2] == 3:
            frame = rgb_frame
        else:
            if len(rgb_frame.shape) == 2:
                frame = np.stack([rgb_frame, rgb_frame, rgb_frame], axis=2)
            else:
                frame = rgb_frame

        # RGB resizing
        if frame.shape[:2] != (self.target_height, self.target_width):
            h_ratio = frame.shape[0] / self.target_height
            w_ratio = frame.shape[1] / self.target_width
            h_indices = (np.arange(self.target_height) * h_ratio).astype(int)
            w_indices = (np.arange(self.target_width) * w_ratio).astype(int)

            resized = np.zeros(
                (self.target_height, self.target_width, 3), dtype=np.uint8
            )
            for c in range(3):
                resized[:, :, c] = frame[np.ix_(h_indices, w_indices, [c])].squeeze()
            return resized
        else:
            return frame

    def _stack_observation(self):
        """Stack RGB frames for ultra-deep network input"""
        frames_list = list(self.frame_stack)

        while len(frames_list) < self.num_frames:
            if len(frames_list) > 0:
                frames_list.insert(0, frames_list[0].copy())
            else:
                dummy_frame = np.zeros(
                    (self.target_height, self.target_width, 3), dtype=np.uint8
                )
                frames_list.append(dummy_frame)

        stacked_frames = []
        for frame in frames_list:
            frame_chw = np.transpose(frame, (2, 0, 1))
            stacked_frames.append(frame_chw)

        stacked = np.concatenate(stacked_frames, axis=0)
        return stacked

    def _calculate_reward(
        self, curr_player_health, curr_opponent_health, game_state, info
    ):
        """
        Simple reward calculation using ONLY real game data
        No fabricated distance/position data
        """

        # Calculate health changes (damage dealt/received)
        opponent_damage = self.prev_opponent_health - curr_opponent_health
        player_damage = self.prev_player_health - curr_player_health

        # Base reward from actual game outcomes
        base_reward = 0.0
        done = False
        episode_outcome = 0.5  # Default neutral outcome

        # Reward for dealing damage to opponent
        if opponent_damage > 0:
            base_reward += min(opponent_damage / 10.0, 1.0)  # Normalize to max +1.0

        # Penalty for taking damage
        if player_damage > 0:
            base_reward -= min(player_damage / 10.0, 1.0)  # Normalize to max -1.0

        # Check for round end using ONLY real data
        if curr_player_health <= 0 or curr_opponent_health <= 0:
            self.total_rounds += 1

            if curr_opponent_health <= 0 and curr_player_health > 0:
                # Player won
                base_reward += 5.0  # Large win bonus
                episode_outcome = 1.0
                self.wins += 1
                self.win_streak += 1
                self.best_win_streak = max(self.best_win_streak, self.win_streak)

                win_rate = self.wins / self.total_rounds if self.total_rounds > 0 else 0
                print(
                    f"🏆 WIN! Round {self.total_rounds} | {self.wins}W/{self.losses}L ({win_rate:.1%}) | Streak: {self.win_streak}"
                )

            elif curr_player_health <= 0 and curr_opponent_health > 0:
                # Player lost
                base_reward -= 5.0  # Large loss penalty
                episode_outcome = 0.0
                self.losses += 1
                self.win_streak = 0

                win_rate = self.wins / self.total_rounds if self.total_rounds > 0 else 0
                print(
                    f"💀 LOSS! Round {self.total_rounds} | {self.wins}W/{self.losses}L ({win_rate:.1%}) | Streak: 0"
                )

            if self.reset_round:
                done = True

            # Update PRM if episode ended and PRIME is enabled
            if self.enable_prime:
                self.episodes_since_prm_update += 1
                if self.episodes_since_prm_update >= self.prm_update_frequency:
                    prm_info = self._update_implicit_prm(episode_outcome)
                    self.episodes_since_prm_update = 0
                    if prm_info and self.total_rounds % 20 == 0:
                        print(
                            f"🧠 PRM Update: Loss={prm_info['prm_loss']:.4f}, Pred={prm_info['outcome_pred']:.3f}, Actual={prm_info['actual_outcome']:.1f}"
                        )

            self._log_periodic_stats()

        # Add PRIME process reward if enabled
        if self.enable_prime and self.last_features is not None:
            process_reward = self._calculate_implicit_process_reward(self.last_features)
            self.episode_process_rewards.append(process_reward)

            # Combine base reward with process reward using PRIME weights
            total_reward = (
                self.outcome_weight * base_reward + self.process_weight * process_reward
            )
        else:
            total_reward = base_reward

        # Reasonable clipping for fighting games
        total_reward = np.clip(total_reward, -10.0, 10.0)

        # Update tracking variables
        self.prev_player_health = curr_player_health
        self.prev_opponent_health = curr_opponent_health

        return total_reward, done

    def _log_periodic_stats(self):
        """Enhanced logging with real game data only"""
        current_time = time.time()
        time_since_last_log = current_time - self.last_log_time

        if time_since_last_log >= self.log_interval:
            session_time = current_time - self.session_start

            if self.total_rounds > 0:
                win_rate = self.wins / self.total_rounds * 100
                avg_episode_reward = (
                    np.mean(self.episode_rewards) if self.episode_rewards else 0
                )
                avg_episode_length = (
                    np.mean(self.episode_lengths) if self.episode_lengths else 0
                )

                if win_rate >= 70:
                    performance = "🏆 EXCELLENT"
                elif win_rate >= 50:
                    performance = "⚔️ GOOD"
                elif win_rate >= 30:
                    performance = "📈 IMPROVING"
                else:
                    performance = "🎯 LEARNING"

                print(f"\n📊 SAMURAI SHODOWN TRAINING STATS:")
                print(f"   {performance} | Win Rate: {win_rate:.1f}%")
                print(
                    f"   🎮 Rounds: {self.total_rounds} ({self.wins}W/{self.losses}L)"
                )
                print(
                    f"   🔥 Best Streak: {self.best_win_streak} | Current: {self.win_streak}"
                )
                print(f"   💰 Avg Reward: {avg_episode_reward:.2f}")
                print(f"   ⏱️  Avg Episode: {avg_episode_length:.0f} steps")

                if self.enable_prime:
                    avg_process_reward = (
                        np.mean(self.episode_process_rewards)
                        if self.episode_process_rewards
                        else 0
                    )
                    cnn_connected = (
                        "✅" if self.cnn_feature_extractor is not None else "❌"
                    )
                    print(f"   🧠 Avg Process Reward: {avg_process_reward:.3f}")
                    print(f"   🔗 CNN Connected: {cnn_connected}")
                    print(f"   🎯 Reference Features: {len(self.reference_features)}")

                print(f"   🕐 Session: {session_time/60:.1f} min")
                print(f"   📈 Total Steps: {self.total_steps:,}")
                print()

            self.last_log_time = current_time

    def reset(self, **kwargs):
        """Reset environment for new episode"""
        observation, info = self.env.reset(**kwargs)

        # Reset health tracking using real data
        game_state = self._extract_game_state(info)
        self.prev_player_health = game_state["player_health"]
        self.prev_opponent_health = game_state["opponent_health"]

        # Reset PRIME tracking
        if self.enable_prime:
            # Update reference features with previous episode
            if len(self.episode_features) > 0:
                step_size = max(1, len(self.episode_features) // 20)
                for i in range(0, len(self.episode_features), step_size):
                    self.reference_features.append(self.episode_features[i].clone())

            self.episode_features = []
            self.episode_process_rewards = []

        # Track episode statistics
        if hasattr(self, "episode_reward"):
            self.episode_rewards.append(self.episode_reward)
        if hasattr(self, "episode_steps"):
            self.episode_lengths.append(self.episode_steps)

        self.episode_steps = 0
        self.episode_reward = 0.0
        self.total_episodes += 1

        # Initialize frame stack
        self.frame_stack.clear()
        processed_frame = self._process_frame(observation)
        self.frame_stack.append(processed_frame.copy())

        for _ in range(self.num_frames - 1):
            self.frame_stack.append(processed_frame.copy())

        stacked_obs = self._stack_observation()

        # Initialize PRIME features using CNN features
        if self.enable_prime:
            self.last_features = self._extract_cnn_features(stacked_obs)
            if self.last_features is not None:
                self.episode_features.append(self.last_features.clone())

        return stacked_obs, info

    def step(self, action):
        """Enhanced step using ONLY real game data"""

        # Action handling
        try:
            if isinstance(action, np.ndarray) and action.shape == (12,):
                final_action = action.astype(np.int32)
            elif isinstance(action, np.ndarray) and action.size == 12:
                final_action = action.reshape(12).astype(np.int32)
            else:
                print(f"⚠️ Expected MultiBinary action with 12 elements, got: {action}")
                final_action = np.zeros(12, dtype=np.int32)
        except Exception as e:
            print(f"❌ Action processing failed: {e}")
            final_action = np.zeros(12, dtype=np.int32)

        # Execute action in environment
        try:
            result = self.env.step(final_action)

            if len(result) == 5:
                observation, reward, terminated, truncated, info = result
                done = terminated or truncated
            elif len(result) == 4:
                observation, reward, done, info = result
                truncated = False
            else:
                raise ValueError(f"Unexpected step return length: {len(result)}")

        except Exception as e:
            print(f"❌ Environment step failed: {e}")
            dummy_obs = np.zeros(
                (self.target_height, self.target_width, 3), dtype=np.uint8
            )
            self.frame_stack.append(dummy_obs)
            stacked_obs = self._stack_observation()
            return stacked_obs, -1.0, True, True, {"error": str(e)}

        # Extract game state using ONLY real data
        game_state = self._extract_game_state(info)
        curr_player_health = game_state["player_health"]
        curr_opponent_health = game_state["opponent_health"]

        # Calculate reward using ONLY real game data
        custom_reward, custom_done = self._calculate_reward(
            curr_player_health, curr_opponent_health, game_state, info
        )

        if custom_done:
            done = custom_done

        # Process frame and update stack
        processed_frame = self._process_frame(observation)
        self.frame_stack.append(processed_frame)
        stacked_obs = self._stack_observation()

        # Update PRIME features using CNN features
        if self.enable_prime:
            current_features = self._extract_cnn_features(stacked_obs)
            if current_features is not None:
                self.episode_features.append(current_features.clone())
                self.last_features = current_features

        # Update episode tracking
        self.episode_steps += 1
        self.total_steps += 1
        self.episode_reward += custom_reward

        # Check for episode timeout
        if self.episode_steps >= self.max_episode_steps:
            truncated = True
            print(f"⏰ Episode timeout at {self.episode_steps} steps")

        # Add PRIME info to info dict
        if self.enable_prime:
            info.update(
                {
                    "prime_enabled": True,
                    "process_weight": self.process_weight,
                    "outcome_weight": self.outcome_weight,
                    "episode_features": len(self.episode_features),
                    "reference_features": len(self.reference_features),
                    "avg_process_reward": (
                        np.mean(self.episode_process_rewards)
                        if self.episode_process_rewards
                        else 0.0
                    ),
                    "cnn_connected": self.cnn_feature_extractor is not None,
                }
            )

        return stacked_obs, custom_reward, done, truncated, info

    def close(self):
        """Clean shutdown with real statistics"""
        print(f"\n🏁 FINAL SAMURAI SHODOWN STATISTICS:")

        if self.total_rounds > 0:
            final_win_rate = self.wins / self.total_rounds * 100
            session_time = time.time() - self.session_start

            print(f"   🎯 Final Win Rate: {final_win_rate:.1f}%")
            print(f"   🎮 Total Rounds: {self.total_rounds}")
            print(f"   🏆 Wins: {self.wins}")
            print(f"   💀 Losses: {self.losses}")
            print(f"   🔥 Best Win Streak: {self.best_win_streak}")
            print(f"   📊 Total Episodes: {self.total_episodes}")
            print(f"   📈 Total Steps: {self.total_steps:,}")
            print(f"   🕐 Session Time: {session_time/3600:.2f} hours")

            if self.episode_rewards:
                print(f"   💰 Avg Episode Reward: {np.mean(self.episode_rewards):.2f}")
                print(
                    f"   📏 Avg Episode Length: {np.mean(self.episode_lengths):.0f} steps"
                )

            if self.enable_prime:
                print(f"   🧠 PRIME Statistics:")
                print(f"      • Process weight: {self.process_weight}")
                print(f"      • Outcome weight: {self.outcome_weight}")
                print(f"      • Reference features: {len(self.reference_features)}")
                print(
                    f"      • CNN connected: {'✅' if self.cnn_feature_extractor is not None else '❌'}"
                )
                if self.episode_process_rewards:
                    print(
                        f"      • Avg process reward: {np.mean(self.episode_process_rewards):.4f}"
                    )

            if final_win_rate >= 70:
                summary = "🏆 EXCELLENT SAMURAI SHODOWN PERFORMANCE!"
            elif final_win_rate >= 50:
                summary = "⚔️ GOOD SAMURAI SHODOWN PERFORMANCE!"
            elif final_win_rate >= 30:
                summary = "📈 SOLID SAMURAI SHODOWN IMPROVEMENT!"
            else:
                summary = "🎯 GOOD SAMURAI SHODOWN LEARNING PROGRESS!"

            print(f"   {summary}")
            if self.enable_prime:
                print(f"   🧠 PRIME implicit PRM system active")
                print(f"   🎯 Dense process rewards implemented")
            print(f"   ⚔️ Reward system based on REAL game data only:")
            print(f"      • health: player health changes")
            print(f"      • enemy_health: opponent health changes")
            print(f"      • score: player score (if used)")
            print(f"      • round: round progression")

        super().close()

    @property
    def current_stats(self):
        """Return current training statistics"""
        win_rate = self.wins / self.total_rounds if self.total_rounds > 0 else 0
        avg_reward = np.mean(self.episode_rewards) if self.episode_rewards else 0

        stats = {
            "win_rate": win_rate,
            "wins": self.wins,
            "losses": self.losses,
            "total_rounds": self.total_rounds,
            "win_streak": self.win_streak,
            "best_win_streak": self.best_win_streak,
            "avg_episode_reward": avg_reward,
            "total_episodes": self.total_episodes,
            "total_steps": self.total_steps,
        }

        # Add PRIME-specific stats
        if self.enable_prime:
            stats.update(
                {
                    "prime_enabled": True,
                    "process_weight": self.process_weight,
                    "outcome_weight": self.outcome_weight,
                    "reference_features": len(self.reference_features),
                    "avg_process_reward": (
                        np.mean(self.episode_process_rewards)
                        if self.episode_process_rewards
                        else 0.0
                    ),
                    "episodes_since_prm_update": self.episodes_since_prm_update,
                    "cnn_connected": self.cnn_feature_extractor is not None,
                }
            )
        else:
            stats["prime_enabled"] = False

        return stats

    def get_implicit_prm(self):
        """Get the implicit PRM model for external access"""
        if self.enable_prime:
            return self.implicit_prm
        return None

    def set_prime_weights(self, process_weight: float, outcome_weight: float):
        """Update PRIME reward weights during training"""
        if self.enable_prime:
            self.process_weight = process_weight
            self.outcome_weight = outcome_weight
            print(
                f"🎯 Updated PRIME weights: process={process_weight:.1f}, outcome={outcome_weight:.1f}"
            )

    def get_prime_episode_data(self):
        """Get PRIME-specific data for the current episode"""
        if not self.enable_prime:
            return None

        return {
            "episode_features": (
                torch.stack(self.episode_features) if self.episode_features else None
            ),
            "episode_process_rewards": self.episode_process_rewards.copy(),
            "episode_length": len(self.episode_features),
            "total_process_reward": sum(self.episode_process_rewards),
            "reference_features_count": len(self.reference_features),
            "cnn_connected": self.cnn_feature_extractor is not None,
        }
