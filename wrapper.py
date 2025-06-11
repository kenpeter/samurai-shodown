import collections
import gymnasium as gym
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW


class FireKnifeDetector:
    """Detects fire knife startup using temporal visual patterns"""

    def __init__(self, frame_history_size=9, target_height=168, target_width=240):
        self.frame_history = collections.deque(maxlen=frame_history_size)
        self.target_height = target_height
        self.target_width = target_width

        # Visual pattern detection parameters
        self.brightness_threshold = 0.3  # Fire knife creates bright effects
        self.motion_threshold = 0.2  # Opponent movement patterns

        # Game state tracking
        self.opponent_action_history = collections.deque(maxlen=10)

    def analyze_frame(self, frame, opponent_health, player_health):
        """Analyze current frame for fire knife startup patterns"""

        self.frame_history.append(frame.copy())

        if len(self.frame_history) < 3:
            return 0.0  # Need at least 3 frames for temporal analysis

        threat_score = 0.0

        # 1. VISUAL PATTERN DETECTION
        threat_score += self._detect_visual_patterns()

        # 2. OPPONENT MOVEMENT ANALYSIS
        threat_score += self._detect_opponent_movement()

        # 3. BRIGHTNESS/EFFECT CHANGES
        threat_score += self._detect_brightness_changes()

        # 4. GAME STATE CONTEXT
        threat_score += self._analyze_game_context(opponent_health, player_health)

        # Normalize to 0-1 range
        threat_score = min(1.0, max(0.0, threat_score))

        return threat_score

    def _detect_visual_patterns(self):
        """Detect visual patterns that indicate fire knife startup"""
        if len(self.frame_history) < 3:
            return 0.0

        current_frame = self.frame_history[-1]
        prev_frame = self.frame_history[-2]

        # Look for sudden appearance of bright pixels (fire effects)
        current_bright = np.sum(current_frame > 200) / current_frame.size
        prev_bright = np.sum(prev_frame > 200) / prev_frame.size

        brightness_increase = current_bright - prev_bright

        if brightness_increase > self.brightness_threshold:
            return 0.4  # Strong visual indicator
        elif brightness_increase > self.brightness_threshold * 0.5:
            return 0.2  # Moderate indicator

        return 0.0

    def _detect_opponent_movement(self):
        """Detect opponent movement patterns that precede fire knife"""
        if len(self.frame_history) < 5:
            return 0.0

        # Analyze opponent region (assume right side of screen for now)
        opponent_region_width = self.target_width // 3

        movement_scores = []
        for i in range(len(self.frame_history) - 1):
            frame1 = self.frame_history[i][:, -opponent_region_width:]
            frame2 = self.frame_history[i + 1][:, -opponent_region_width:]

            # Calculate frame difference in opponent region
            diff = np.abs(frame2.astype(float) - frame1.astype(float))
            movement_score = np.mean(diff) / 255.0
            movement_scores.append(movement_score)

        # Look for specific movement pattern: stillness followed by action
        if len(movement_scores) >= 4:
            recent_movement = movement_scores[-2:]  # Last 2 frames
            earlier_movement = movement_scores[-4:-2]  # 2 frames before

            # Pattern: was still, now moving (charging up)
            if np.mean(earlier_movement) < 0.1 and np.mean(recent_movement) > 0.2:
                return 0.3

        return 0.0

    def _detect_brightness_changes(self):
        """Detect sudden brightness changes that indicate special moves"""
        if len(self.frame_history) < 3:
            return 0.0

        frames = list(self.frame_history)
        brightness_history = [np.mean(frame) for frame in frames]

        if len(brightness_history) >= 3:
            # Look for brightness spike (fire knife has bright effects)
            recent_avg = np.mean(brightness_history[-2:])
            baseline_avg = np.mean(brightness_history[:-2])

            brightness_ratio = recent_avg / (baseline_avg + 1e-6)

            if brightness_ratio > 1.3:  # 30% brightness increase
                return 0.3
            elif brightness_ratio > 1.15:  # 15% increase
                return 0.1

        return 0.0

    def _analyze_game_context(self, opponent_health, player_health):
        """Use game state to assess fire knife likelihood"""
        threat_score = 0.0

        # Opponents are more likely to use special moves when:
        # 1. They're losing (low health)
        if opponent_health < 64:  # Less than half health
            threat_score += 0.2

        # 2. Player is close (would need distance calculation)
        # For now, assume medium threat if player health is high (opponent getting desperate)
        if player_health > 90:
            threat_score += 0.1

        return threat_score


class SamuraiShowdownCustomWrapper(gym.Wrapper):
    """Enhanced wrapper with temporal fire knife detection"""

    _global_stats = {
        "total_wins": 0,
        "total_losses": 0,
        "total_rounds": 0,
        "env_stats": {},
        "last_log_time": time.time(),
        "session_start": time.time(),
    }

    def __init__(self, env, reset_round=True, rendering=False, max_episode_steps=15000):
        super(SamuraiShowdownCustomWrapper, self).__init__(env)
        self.env = env

        # Frame processing
        self.resize_scale = 0.75
        self.num_frames = 9
        self.frame_stack = collections.deque(maxlen=self.num_frames)

        # Health tracking
        self.full_hp = 128
        self.prev_player_health = self.full_hp
        self.prev_opponent_health = self.full_hp

        # Episode management
        self.max_episode_steps = max_episode_steps
        self.episode_steps = 0
        self.reset_round = reset_round

        # Aggression tracking
        self.consecutive_attacks = 0
        self.last_action = None

        # Environment tracking
        import random

        self.env_id = f"ENV-{random.randint(1000, 9999)}"
        self.wins = 0
        self.losses = 0
        self.total_rounds = 0

        # Initialize global tracking
        SamuraiShowdownCustomWrapper._global_stats["env_stats"][self.env_id] = {
            "wins": 0,
            "losses": 0,
            "total_rounds": 0,
        }

        self.log_interval = 120  # Log every 2 minutes

        # Get frame dimensions
        dummy_obs, _ = self.env.reset()
        original_height, original_width = dummy_obs.shape[:2]
        self.target_height = int(original_height * self.resize_scale)
        self.target_width = int(original_width * self.resize_scale)

        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(self.num_frames, self.target_height, self.target_width),
            dtype=np.uint8,
        )

        # Add fire knife detection
        self.fire_knife_detector = FireKnifeDetector(
            frame_history_size=self.num_frames,
            target_height=self.target_height,
            target_width=self.target_width,
        )

        # Track attack patterns
        self.attack_pattern_buffer = collections.deque(
            maxlen=20
        )  # Last 20 frames of analysis
        self.evasion_window = 0  # Countdown for when to dodge
        self.successful_evasions = 0  # Track successful evasions for logging

        # Handle action space conversion
        if hasattr(self.env.action_space, "n"):
            self.action_space = gym.spaces.Discrete(self.env.action_space.n)
            self._original_action_space = self.env.action_space
            print(
                f"   🔄 Converted MultiBinary({self.env.action_space.n}) to Discrete({self.action_space.n})"
            )
        else:
            self.action_space = self.env.action_space
            self._original_action_space = self.env.action_space

        print(f"🚀 {self.env_id} Enhanced Temporal Fire Knife Wrapper")
        print(
            f"   🎯 Smart rewards: Win/Loss terminal, damage/aggression/evasion continuous"
        )
        print(f"   📏 Episode length: {max_episode_steps} steps")
        print(f"   🔥 Temporal fire knife evasion enabled")
        print(f"   🧠 Visual pattern detection enabled")

    def _process_frame(self, rgb_frame):
        """Convert RGB to grayscale and resize"""
        if len(rgb_frame.shape) == 3 and rgb_frame.shape[2] == 3:
            gray = np.dot(rgb_frame, [0.299, 0.587, 0.114]).astype(np.uint8)
        else:
            gray = rgb_frame

        if gray.shape[:2] != (self.target_height, self.target_width):
            h_ratio = gray.shape[0] / self.target_height
            w_ratio = gray.shape[1] / self.target_width
            h_indices = (np.arange(self.target_height) * h_ratio).astype(int)
            w_indices = (np.arange(self.target_width) * w_ratio).astype(int)
            resized = gray[np.ix_(h_indices, w_indices)]
            return resized
        return gray

    def _stack_observation(self):
        """Stack frames efficiently"""
        frames_list = list(self.frame_stack)

        while len(frames_list) < self.num_frames:
            if len(frames_list) > 0:
                frames_list.insert(0, frames_list[0].copy())
            else:
                dummy_frame = np.zeros(
                    (self.target_height, self.target_width), dtype=np.uint8
                )
                frames_list.append(dummy_frame)

        return np.stack(frames_list, axis=0)

    def _extract_health(self, info):
        """Extract health from info"""
        player_health = info.get("health", self.full_hp)
        opponent_health = info.get("enemy_health", self.full_hp)
        return player_health, opponent_health

    def _detect_fire_knife_startup(
        self, processed_frame, opponent_health, player_health
    ):
        """Detect fire knife startup using visual and game state cues"""

        # Add current frame to detector
        threat_level = self.fire_knife_detector.analyze_frame(
            processed_frame, opponent_health, player_health
        )

        # Store analysis
        self.attack_pattern_buffer.append(
            {
                "frame": processed_frame.copy(),
                "threat_level": threat_level,
                "opponent_health": opponent_health,
                "step": self.episode_steps,
            }
        )

        return threat_level

    def _log_periodic_stats(self):
        """Log stats periodically"""
        current_time = time.time()
        time_since_last_log = (
            current_time - SamuraiShowdownCustomWrapper._global_stats["last_log_time"]
        )

        if time_since_last_log >= self.log_interval:
            global_stats = SamuraiShowdownCustomWrapper._global_stats
            print(f"\n📊 Training Stats:")

            for env_id, stats in global_stats["env_stats"].items():
                if stats["total_rounds"] > 0:
                    win_rate = stats["wins"] / stats["total_rounds"] * 100
                    emoji = "🏆" if win_rate >= 60 else "⚔️" if win_rate >= 40 else "📈"
                    print(
                        f"   {emoji} {env_id}: {stats['wins']}W/{stats['losses']}L ({win_rate:.1f}%)"
                    )

            # Show evasion stats
            if hasattr(self, "successful_evasions"):
                print(f"   🛡️ Successful evasions: {self.successful_evasions}")

            print()
            global_stats["last_log_time"] = current_time

    def _calculate_reward(self, curr_player_health, curr_opponent_health, action=None):
        """Enhanced reward with temporal fire knife detection"""
        reward = 0.0
        done = False

        # Get current processed frame for analysis
        if len(self.frame_stack) > 0:
            current_frame = list(self.frame_stack)[-1]

            # Detect fire knife startup
            threat_level = self._detect_fire_knife_startup(
                current_frame, curr_opponent_health, curr_player_health
            )

            # TEMPORAL EVASION REWARDS
            if threat_level > 0.7:  # High threat detected
                self.evasion_window = 3  # Should dodge in next 3 frames

                # Reward recognizing the threat
                reward += 0.2

                # Big reward for immediate evasive action
                evasive_actions = [1, 3, 2]  # Left, right, back
                if action in evasive_actions:
                    reward += 0.5
                    print(
                        f"🛡️ EVASION: High threat detected, rewarding dodge action {action}"
                    )

                # Penalty for attacking into danger
                attack_actions = [8, 9, 10, 11]
                if action in attack_actions:
                    reward -= 0.3

            elif self.evasion_window > 0:
                # Still in evasion window
                self.evasion_window -= 1

                evasive_actions = [1, 3, 2]
                if action in evasive_actions:
                    reward += 0.3
                    print(
                        f"🛡️ EVASION: Dodging in danger window (remaining: {self.evasion_window})"
                    )

            # SUCCESSFUL EVASION DETECTION
            # If we were in danger but didn't take big damage
            if len(self.attack_pattern_buffer) >= 5:
                recent_threats = [
                    a["threat_level"] for a in list(self.attack_pattern_buffer)[-5:]
                ]
                max_recent_threat = max(recent_threats)

                # If high threat was detected recently but we avoided damage
                if max_recent_threat > 0.7:
                    damage_taken = self.prev_player_health - curr_player_health
                    if damage_taken < 5:  # Successfully avoided major damage
                        reward += 1.0  # Big reward for successful evasion
                        self.successful_evasions += 1
                        print(
                            f"🏆 SUCCESSFUL EVASION: Avoided fire knife! Total evasions: {self.successful_evasions}"
                        )

        # 1. ATTACK BONUS - Encourage aggression (when not in danger)
        if (
            action is not None and self.evasion_window == 0
        ):  # Only reward aggression when safe
            attack_actions = [8, 9, 10, 11]  # Main attack buttons
            defensive_actions = [4, 5, 6, 7]  # Block/defensive moves
            movement_actions = [0, 1, 2, 3]  # Movement/jump

            if action in attack_actions:
                # Big bonus for attacking
                reward += 0.3
                self.consecutive_attacks += 1
                # Combo bonus: reward sustained aggression
                combo_bonus = min(self.consecutive_attacks * 0.1, 0.5)
                reward += combo_bonus
            elif action in defensive_actions:
                # Small penalty for defensive play (unless evading)
                reward -= 0.1
                self.consecutive_attacks = 0
            elif action in movement_actions:
                # Neutral for movement, but break attack combo
                self.consecutive_attacks = 0

        # 2. DAMAGE REWARDS - Big rewards for dealing damage
        damage_reward = 0.0
        if hasattr(self, "prev_opponent_health"):
            if curr_opponent_health < self.prev_opponent_health:
                damage_dealt = self.prev_opponent_health - curr_opponent_health
                damage_reward = damage_dealt * 0.05  # 5 points per HP damage
                reward += damage_reward

        # 3. DAMAGE PENALTY - Discourage taking damage
        damage_penalty = 0.0
        if hasattr(self, "prev_player_health"):
            if curr_player_health < self.prev_player_health:
                damage_taken = self.prev_player_health - curr_player_health
                damage_penalty = damage_taken * 0.02  # Small penalty for taking damage
                reward -= damage_penalty

                # Extra penalty for big damage (likely fire knife hit)
                if damage_taken >= 20:
                    reward -= 1.0
                    print(f"💀 FIRE KNIFE HIT: Major damage penalty")

        # 4. TERMINAL REWARDS - Simple win/loss
        if curr_player_health <= 0 or curr_opponent_health <= 0:
            self.total_rounds += 1
            SamuraiShowdownCustomWrapper._global_stats["total_rounds"] += 1

            if curr_opponent_health <= 0 and curr_player_health > 0:
                # WIN - Big reward
                self.wins += 1
                win_rate = self.wins / self.total_rounds
                SamuraiShowdownCustomWrapper._global_stats["total_wins"] += 1
                SamuraiShowdownCustomWrapper._global_stats["env_stats"][self.env_id][
                    "wins"
                ] = self.wins
                SamuraiShowdownCustomWrapper._global_stats["env_stats"][self.env_id][
                    "total_rounds"
                ] = self.total_rounds

                # Dominance bonus: extra reward for winning with high health
                health_ratio = curr_player_health / self.full_hp
                dominance_bonus = health_ratio * 0.5  # Up to +0.5 for perfect wins

                # Quick finish bonus: reward fast victories
                if self.episode_steps < 1000:
                    quick_bonus = 0.3
                else:
                    quick_bonus = 0.0

                # Evasion bonus: extra reward for successful evasions during match
                evasion_bonus = min(self.successful_evasions * 0.2, 1.0)

                total_win_reward = 2.0 + dominance_bonus + quick_bonus + evasion_bonus
                reward += total_win_reward

                print(
                    f"🏆 {self.env_id} WIN! Health: {curr_player_health}/{self.full_hp} "
                    f"Evasions: {self.successful_evasions} "
                    f"Reward: +{total_win_reward:.2f} ({self.wins}W/{self.losses}L - {win_rate:.1%})"
                )

                done = True

            elif curr_player_health <= 0 and curr_opponent_health > 0:
                # LOSS - Big penalty
                self.losses += 1
                win_rate = self.wins / self.total_rounds
                SamuraiShowdownCustomWrapper._global_stats["total_losses"] += 1
                SamuraiShowdownCustomWrapper._global_stats["env_stats"][self.env_id][
                    "losses"
                ] = self.losses
                SamuraiShowdownCustomWrapper._global_stats["env_stats"][self.env_id][
                    "total_rounds"
                ] = self.total_rounds

                reward -= 2.0  # Simple loss penalty

                print(
                    f"💀 {self.env_id} LOSS! Evasions: {self.successful_evasions} ({self.wins}W/{self.losses}L - {win_rate:.1%})"
                )
                done = True

            if self.reset_round:
                done = True

            self._log_periodic_stats()

        # Update health tracking
        self.prev_player_health = curr_player_health
        self.prev_opponent_health = curr_opponent_health

        # Clamp reward to reasonable bounds
        reward = np.clip(reward, -3.0, 4.0)  # Increased upper bound for evasion rewards

        return reward, done

    def reset(self, **kwargs):
        """Reset environment"""
        result = self.env.reset(**kwargs)
        if isinstance(result, tuple):
            observation, info = result
        else:
            observation = result
            info = {}

        self.prev_player_health = self.full_hp
        self.prev_opponent_health = self.full_hp
        self.episode_steps = 0
        self.consecutive_attacks = 0
        self.evasion_window = 0
        self.successful_evasions = 0

        # Reset fire knife detector
        self.fire_knife_detector = FireKnifeDetector(
            frame_history_size=self.num_frames,
            target_height=self.target_height,
            target_width=self.target_width,
        )
        self.attack_pattern_buffer.clear()

        self.frame_stack.clear()
        processed_frame = self._process_frame(observation)
        self.frame_stack.append(processed_frame.copy())

        for _ in range(self.num_frames - 1):
            zero_frame = np.zeros_like(processed_frame)
            self.frame_stack.append(zero_frame)

        stacked_obs = self._stack_observation()
        return stacked_obs, info

    def _bias_action_toward_attacks(self, action):
        """Enhanced action selection with evasion priority"""
        attack_actions = [8, 9, 10, 11]
        defensive_actions = [4, 5, 6, 7]
        evasive_actions = [1, 3]  # Left/right movement

        # PRIORITY 1: If in evasion window, force evasive movement
        if self.evasion_window > 0:
            if np.random.random() < 0.8:  # 80% chance to force evasion
                return np.random.choice(evasive_actions)

        # PRIORITY 2: Random evasion attempts (10% chance)
        if np.random.random() < 0.1:
            return np.random.choice(evasive_actions)

        # PRIORITY 3: Force aggression when safe (40% chance)
        if self.evasion_window == 0 and np.random.random() < 0.4:
            return np.random.choice(attack_actions)

        # Reduce jumping (often action 0)
        if action == 0 and np.random.random() > 0.1:
            return np.random.choice(attack_actions)

        # Replace blocking with attacks 60% of the time (when safe)
        if (
            action in defensive_actions
            and self.evasion_window == 0
            and np.random.random() < 0.6
        ):
            return np.random.choice(attack_actions)

        # 30% chance to force attack for any non-attack action (when safe)
        if (
            action not in attack_actions
            and self.evasion_window == 0
            and np.random.random() < 0.3
        ):
            return np.random.choice(attack_actions)

        return action

    def step(self, action):
        """Enhanced step with temporal fire knife detection"""
        # Convert action to proper format
        try:
            if hasattr(action, "shape") and action.shape == ():
                action_int = int(action.item())
            elif hasattr(action, "__len__") and len(action) == 1:
                action_int = int(action[0])
            elif hasattr(action, "item"):
                action_int = action.item()
            else:
                action_int = int(action)
        except (ValueError, IndexError, TypeError):
            action_int = 0

        # Apply enhanced action bias (includes evasion priority)
        action_int = self._bias_action_toward_attacks(action_int)

        # Convert for MultiBinary action space
        if hasattr(self._original_action_space, "n"):
            n_actions = self._original_action_space.n
            action_array = np.zeros(n_actions, dtype=np.int8)
            action_int = max(0, min(action_int, n_actions - 1))
            action_array[action_int] = 1
            action_to_use = action_array
        else:
            action_to_use = action_int

        # Execute step
        try:
            result = self.env.step(action_to_use)
            if len(result) == 5:
                observation, reward, done, truncated, info = result
            elif len(result) == 4:
                observation, reward, done, info = result
                truncated = False
            else:
                raise ValueError(f"Unexpected step return format: {len(result)} values")

        except Exception as e:
            print(f"❌ Step error: {e}")
            # Fallback
            if hasattr(self._original_action_space, "n"):
                fallback_action = np.zeros(self._original_action_space.n, dtype=np.int8)
            else:
                fallback_action = 0

            result = self.env.step(fallback_action)
            if len(result) == 5:
                observation, reward, done, truncated, info = result
            elif len(result) == 4:
                observation, reward, done, info = result
                truncated = False

        # Extract health and calculate ALL rewards in one place
        curr_player_health, curr_opponent_health = self._extract_health(info)

        # ALL REWARDS CALCULATED HERE (including temporal fire knife detection)
        total_reward, custom_done = self._calculate_reward(
            curr_player_health, curr_opponent_health, action_int
        )

        if custom_done:
            done = custom_done

        # Process frame
        processed_frame = self._process_frame(observation)
        self.frame_stack.append(processed_frame)
        stacked_obs = self._stack_observation()

        self.episode_steps += 1
        if self.episode_steps >= self.max_episode_steps:
            truncated = True

        # Store last action for combo tracking
        self.last_action = action_int

        return stacked_obs, total_reward, done, truncated, info


# Enhanced Decision Transformer with temporal processing
class EnhancedDecisionTransformer(nn.Module):
    """Enhanced Decision Transformer with better temporal processing for fire knife detection"""

    def __init__(
        self,
        observation_shape,
        action_dim,
        hidden_size=256,
        n_layer=4,
        n_head=4,
        max_ep_len=2000,
    ):
        super().__init__()
        self.observation_shape = observation_shape
        self.action_dim = action_dim
        self.hidden_size = hidden_size
        self.max_ep_len = max_ep_len

        # Enhanced CNN encoder with temporal awareness
        self.cnn_encoder = nn.Sequential(
            nn.Conv2d(observation_shape[0], 32, kernel_size=8, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
        )

        cnn_output_size = 256 * 4 * 4

        # Add temporal convolution to better process frame sequences
        self.temporal_conv = nn.Conv1d(
            in_channels=cnn_output_size,
            out_channels=hidden_size,
            kernel_size=3,
            padding=1,
        )

        # Embeddings
        self.state_encoder = nn.Linear(cnn_output_size, hidden_size)
        self.action_encoder = nn.Embedding(action_dim, hidden_size)
        self.return_encoder = nn.Linear(1, hidden_size)
        self.timestep_encoder = nn.Embedding(max_ep_len, hidden_size)

        # Transformer
        self.ln = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(0.1)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=n_head,
            dim_feedforward=4 * hidden_size,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layer)
        self.action_head = nn.Linear(hidden_size, action_dim)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, states, actions, returns_to_go, timesteps, attention_mask=None):
        batch_size, seq_len = states.shape[:2]

        # Process states through CNN
        states_flat = states.view(-1, *self.observation_shape)
        state_features = self.cnn_encoder(states_flat)  # [batch*seq, 256*4*4]

        # Reshape for temporal processing
        state_features = state_features.view(
            batch_size, seq_len, -1
        )  # [batch, seq, 256*4*4]

        # Apply temporal convolution for better sequence understanding
        state_features_temporal = state_features.transpose(
            1, 2
        )  # [batch, features, seq]
        state_features_temporal = self.temporal_conv(
            state_features_temporal
        )  # [batch, hidden, seq]
        state_features_temporal = state_features_temporal.transpose(
            1, 2
        )  # [batch, seq, hidden]

        # Use temporal features as state embeddings
        state_embeddings = state_features_temporal

        # Encode other inputs
        action_embeddings = self.action_encoder(actions)
        rtg_embeddings = self.return_encoder(returns_to_go.unsqueeze(-1))
        timesteps_clamped = torch.clamp(timesteps, 0, self.max_ep_len - 1)
        timestep_embeddings = self.timestep_encoder(timesteps_clamped)

        # Stack embeddings
        stacked_inputs = torch.stack(
            [rtg_embeddings, state_embeddings, action_embeddings], dim=2
        )
        stacked_inputs = stacked_inputs.view(batch_size, 3 * seq_len, self.hidden_size)

        # Apply transformer
        stacked_inputs = self.ln(stacked_inputs)
        stacked_inputs = self.dropout(stacked_inputs)
        transformer_outputs = self.transformer(stacked_inputs)

        # Extract action predictions
        action_outputs = transformer_outputs[:, 2::3]
        action_logits = self.action_head(action_outputs)

        return action_logits

    def get_action(self, states, actions, returns_to_go, timesteps, temperature=1.0):
        """Get action for inference with temperature control"""
        self.eval()
        with torch.no_grad():
            logits = self.forward(states, actions, returns_to_go, timesteps)
            last_logits = logits[0, -1] / temperature
            probs = F.softmax(last_logits, dim=-1)
            action = torch.multinomial(probs, 1).item()
        return action

    def save(self, path):
        """Simple save - just the state dict"""
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path, env=None, device="cpu"):
        """Load model - basic PyTorch loading"""
        obs_shape = env.observation_space.shape
        action_dim = env.action_space.n
        model = cls(obs_shape, action_dim)
        model.load_state_dict(torch.load(path, map_location=device))
        model.to(device)
        print(f"📂 Model loaded from: {path}")
        return model


# Keep existing classes for compatibility
DecisionTransformer = EnhancedDecisionTransformer  # Use enhanced version by default


class TrajectoryDataset(Dataset):
    """Dataset for Decision Transformer training"""

    def __init__(self, trajectories, context_length=30):
        self.trajectories = [
            t
            for t in trajectories
            if len(t["rewards"]) > context_length // 2 and len(t["states"]) > 0
        ]
        self.context_length = context_length

        print(f"📊 Dataset: {len(self.trajectories)} trajectories")
        if len(self.trajectories) > 0:
            avg_length = np.mean([len(t["rewards"]) for t in self.trajectories])
            total_reward = np.mean([sum(t["rewards"]) for t in self.trajectories])
            print(f"   Average length: {avg_length:.1f}")
            print(f"   Average reward: {total_reward:.2f}")

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        traj = self.trajectories[idx]
        states = np.array(traj["states"])
        actions = np.array(traj["actions"])
        rewards = np.array(traj["rewards"])

        # Ensure consistent lengths
        min_length = min(len(states), len(actions))
        states = states[:min_length]
        actions = actions[:min_length]
        rewards = rewards[:min_length]

        # Calculate returns-to-go
        returns_to_go = np.zeros_like(rewards, dtype=np.float32)
        running_return = 0
        gamma = 0.99
        for i in reversed(range(len(rewards))):
            running_return = rewards[i] + gamma * running_return
            returns_to_go[i] = running_return

        timesteps = np.arange(len(states))

        # Handle sequence length
        if len(states) <= self.context_length:
            # Pad sequence
            pad_length = self.context_length - len(states)
            if pad_length > 0:
                pad_states = np.zeros(
                    (pad_length, *states.shape[1:]), dtype=states.dtype
                )
                states = np.concatenate([pad_states, states])
                actions = np.concatenate(
                    [np.zeros(pad_length, dtype=actions.dtype), actions]
                )
                returns_to_go = np.concatenate(
                    [np.zeros(pad_length, dtype=np.float32), returns_to_go]
                )
                timesteps = np.concatenate(
                    [np.zeros(pad_length, dtype=timesteps.dtype), timesteps]
                )
        else:
            # Random crop
            start_idx = np.random.randint(0, len(states) - self.context_length + 1)
            states = states[start_idx : start_idx + self.context_length]
            actions = actions[start_idx : start_idx + self.context_length]
            returns_to_go = returns_to_go[start_idx : start_idx + self.context_length]
            timesteps = timesteps[start_idx : start_idx + self.context_length]

        return {
            "states": torch.from_numpy(states).float() / 255.0,
            "actions": torch.from_numpy(actions).long(),
            "returns_to_go": torch.from_numpy(returns_to_go).float(),
            "timesteps": torch.from_numpy(timesteps).long(),
        }


def train_decision_transformer(
    model,
    trajectories,
    epochs=50,
    batch_size=16,
    lr=1e-4,
    device="cuda",
    context_length=30,
):
    """Train Enhanced Decision Transformer with memory optimizations"""

    dataset = TrajectoryDataset(trajectories, context_length)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=1,
        persistent_workers=False,
    )

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4, betas=(0.9, 0.95))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    model.to(device)
    model.train()

    print(f"🚀 Training Enhanced Decision Transformer:")
    print(f"   Device: {device}")
    print(f"   Epochs: {epochs}")
    print(f"   Batch size: {batch_size}")
    print(f"   Learning rate: {lr}")
    print(f"   Context length: {context_length}")
    print(f"   Dataset size: {len(dataset)}")
    print(f"   🔥 Temporal fire knife detection enabled")

    scaler = torch.cuda.amp.GradScaler()
    best_loss = float("inf")

    for epoch in range(epochs):
        total_loss = 0
        num_batches = 0
        epoch_start_time = time.time()

        for batch_idx, batch in enumerate(dataloader):
            states = batch["states"].to(device, non_blocking=True)
            actions = batch["actions"].to(device, non_blocking=True)
            returns_to_go = batch["returns_to_go"].to(device, non_blocking=True)
            timesteps = batch["timesteps"].to(device, non_blocking=True)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                action_logits = model(states, actions, returns_to_go, timesteps)

                if action_logits.shape[1] > 1:
                    targets = actions[:, 1:]
                    predictions = action_logits[:, :-1]
                    loss = F.cross_entropy(
                        predictions.reshape(-1, predictions.shape[-1]),
                        targets.reshape(-1),
                    )
                else:
                    continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            num_batches += 1

        scheduler.step()
        epoch_time = time.time() - epoch_start_time
        avg_loss = total_loss / max(num_batches, 1)

        if avg_loss < best_loss:
            best_loss = avg_loss

        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(
                f"   Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}, "
                f"LR: {scheduler.get_last_lr()[0]:.6f}, Time: {epoch_time:.1f}s"
            )

            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**3
                cached = torch.cuda.memory_reserved() / 1024**3
                print(f"   GPU: {allocated:.2f}GB allocated, {cached:.2f}GB cached")

    print(f"✅ Enhanced Training complete! Best loss: {best_loss:.4f}")
    print(f"🔥 Model now includes temporal fire knife detection capabilities")
    return model
