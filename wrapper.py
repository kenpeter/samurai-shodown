from collections import deque
import gymnasium as gym
import numpy as np
import time
from datetime import datetime
import math


class SamuraiShowdownCustomWrapper(gym.Wrapper):
    """Enhanced wrapper with opponent pattern recognition and predictive rewards"""

    def __init__(
        self,
        env,
        reset_round=True,
        rendering=False,
        max_episode_steps=12000,
        reward_coeff=1.0,
        prediction_horizon=30,  # How many frames to look ahead
    ):
        super(SamuraiShowdownCustomWrapper, self).__init__(env)
        self.env = env

        # Frame processing - optimized for deep networks
        self.resize_scale = 0.75
        self.num_frames = 9
        self.frame_stack = deque(maxlen=self.num_frames)

        # Health tracking
        self.full_hp = 128
        self.prev_player_health = self.full_hp
        self.prev_opponent_health = self.full_hp

        # Reward system
        self.reward_coeff = reward_coeff

        # Episode management
        self.max_episode_steps = max_episode_steps
        self.episode_steps = 0
        self.reset_round = reset_round

        # PREDICTIVE COMPONENTS
        self.prediction_horizon = prediction_horizon

        # Opponent pattern tracking
        self.opponent_position_history = deque(maxlen=60)  # 1 second at 60fps
        self.opponent_action_history = deque(maxlen=30)  # Recent actions
        self.opponent_health_history = deque(maxlen=10)  # Health changes

        # Player state tracking for reaction analysis
        self.player_position_history = deque(maxlen=60)
        self.player_action_history = deque(maxlen=30)

        # Pattern recognition
        self.recognized_patterns = {}
        self.pattern_confidence = {}
        self.last_prediction = None
        self.prediction_accuracy = deque(maxlen=100)

        # Advanced reward tracking
        self.anticipation_rewards = 0.0
        self.prediction_rewards = 0.0
        self.reaction_rewards = 0.0

        # Distance and spatial analysis
        self.optimal_distances = {
            "close_range": (0, 40),  # Grappling/close attacks
            "mid_range": (40, 80),  # Normal attacks
            "long_range": (80, 120),  # Projectiles/long attacks
            "far_range": (120, 200),  # Safe distance
        }

        # Combat state detection
        self.combat_states = {
            "aggressive": 0,
            "defensive": 0,
            "neutral": 0,
            "retreating": 0,
        }

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

        # Logging
        self.last_log_time = time.time()
        self.log_interval = 60
        self.session_start = time.time()

        # Get frame dimensions
        dummy_obs, _ = self.env.reset()
        original_height, original_width = dummy_obs.shape[:2]
        self.target_height = int(original_height * self.resize_scale)
        self.target_width = int(original_width * self.resize_scale)

        # Observation space for ultra-deep network
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(self.num_frames, self.target_height, self.target_width),
            dtype=np.uint8,
        )

        # Debug action space
        print(f"⚡ SAMURAI SHOWDOWN ENHANCED WRAPPER")
        print(f"   🎮 Action space: {self.env.action_space}")
        print(f"   🎮 Action type: {type(self.env.action_space)}")
        print(f"   🔮 Prediction horizon: {prediction_horizon} frames")
        print(f"   🧠 Pattern recognition: ENABLED")
        print(f"   🎯 Predictive rewards: ENABLED")

        # FIXED: Proper action space detection
        action_space_type = type(self.env.action_space).__name__
        print(f"   🎮 Action space class: {action_space_type}")

        if action_space_type == "Discrete":
            self.action_type = "discrete"
            print(f"   🎮 Discrete actions: {self.env.action_space.n}")
        elif action_space_type == "MultiBinary":
            self.action_type = "multibinary"
            print(f"   🎮 MultiBinary shape: {self.env.action_space.shape}")
        elif (
            hasattr(self.env.action_space, "shape")
            and len(self.env.action_space.shape) == 1
        ):
            # Fallback detection for MultiBinary-like spaces
            self.action_type = "multibinary"
            print(
                f"   🎮 Detected MultiBinary-like space: {self.env.action_space.shape}"
            )
        elif hasattr(self.env.action_space, "n"):
            self.action_type = "discrete"
            print(f"   🎮 Fallback discrete actions: {self.env.action_space.n}")
        else:
            # If all else fails, assume MultiBinary for retro games
            self.action_type = "multibinary"
            print(f"   ⚠️ Unknown action space, assuming MultiBinary for retro game")

        print(
            f"   🎯 Enhanced rewards: Win +1.0, Lose -1.0, Prediction +0.15, Anticipation +0.1"
        )
        print(f"   📏 Episode length: {max_episode_steps} steps")
        print(f"   📊 Frame stack: {self.num_frames} frames")
        print(f"   🖼️  Frame size: {self.target_height}x{self.target_width}")

    def _extract_game_state(self, info):
        """Extract comprehensive game state information"""
        # Basic health information
        player_health = info.get("health", self.full_hp)
        opponent_health = info.get("enemy_health", self.full_hp)

        # Position information (if available)
        player_x = info.get("x", 100)  # Default values if not available
        player_y = info.get("y", 200)
        opponent_x = info.get("enemy_x", 200)
        opponent_y = info.get("enemy_y", 200)

        # Calculate distance
        distance = abs(player_x - opponent_x)

        # Determine range category
        range_category = "far_range"
        for range_name, (min_dist, max_dist) in self.optimal_distances.items():
            if min_dist <= distance < max_dist:
                range_category = range_name
                break

        return {
            "player_health": player_health,
            "opponent_health": opponent_health,
            "player_x": player_x,
            "player_y": player_y,
            "opponent_x": opponent_x,
            "opponent_y": opponent_y,
            "distance": distance,
            "range_category": range_category,
        }

    def _detect_opponent_pattern(self):
        """Analyze opponent movement/action patterns"""
        if len(self.opponent_position_history) < 10:
            return None, 0.0

        # Analyze recent position changes
        positions = list(self.opponent_position_history)[-10:]

        # Calculate movement pattern
        x_changes = [
            positions[i + 1]["x"] - positions[i]["x"] for i in range(len(positions) - 1)
        ]
        y_changes = [
            positions[i + 1]["y"] - positions[i]["y"] for i in range(len(positions) - 1)
        ]

        # Detect patterns
        avg_x_change = np.mean(x_changes) if x_changes else 0
        avg_y_change = np.mean(y_changes) if y_changes else 0

        # Pattern classification
        pattern = None
        confidence = 0.0

        if abs(avg_x_change) > 2:
            if avg_x_change > 0:
                pattern = "moving_right"
            else:
                pattern = "moving_left"
            confidence = min(abs(avg_x_change) / 5.0, 1.0)

        elif abs(avg_y_change) > 2:
            if avg_y_change > 0:
                pattern = "moving_down"
            else:
                pattern = "moving_up"
            confidence = min(abs(avg_y_change) / 5.0, 1.0)

        # Check for repetitive behavior
        if len(self.opponent_action_history) >= 5:
            recent_actions = list(self.opponent_action_history)[-5:]
            if len(set(recent_actions)) <= 2:  # Limited action variety
                pattern = "repetitive_behavior"
                confidence = 0.8

        # Health-based pattern detection
        if len(self.opponent_health_history) >= 3:
            health_changes = list(self.opponent_health_history)[-3:]
            if all(h < self.full_hp * 0.3 for h in health_changes):
                pattern = "low_health_desperate"
                confidence = 0.9

        return pattern, confidence

    def _predict_next_opponent_action(self, current_state):
        """Predict opponent's next likely action based on patterns"""
        pattern, confidence = self._detect_opponent_pattern()

        if pattern is None or confidence < 0.3:
            return None, 0.0

        # Store pattern for learning
        if pattern not in self.recognized_patterns:
            self.recognized_patterns[pattern] = []
            self.pattern_confidence[pattern] = []

        # Prediction logic based on recognized patterns
        prediction = None
        pred_confidence = confidence

        if pattern == "moving_right":
            prediction = "will_continue_right"
        elif pattern == "moving_left":
            prediction = "will_continue_left"
        elif pattern == "repetitive_behavior":
            prediction = "will_repeat_action"
        elif pattern == "low_health_desperate":
            prediction = "will_attack_aggressively"

        # Store prediction for accuracy tracking
        self.last_prediction = {
            "pattern": pattern,
            "prediction": prediction,
            "confidence": pred_confidence,
            "frame": self.episode_steps,
        }

        return prediction, pred_confidence

    def _calculate_predictive_reward(self, action, current_state, prev_state):
        """Calculate rewards based on predictive behavior and pattern recognition"""
        base_reward = 0.0
        prediction_reward = 0.0
        anticipation_reward = 0.0
        reaction_reward = 0.0

        # Get current prediction
        prediction, pred_confidence = self._predict_next_opponent_action(current_state)

        # Reward for making predictions (encourages pattern recognition)
        if prediction and pred_confidence > 0.5:
            prediction_reward = 0.05 * pred_confidence
            self.prediction_rewards += prediction_reward

        # Analyze if the agent's action shows anticipation
        if prediction and self.last_prediction:
            agent_action_type = self._classify_agent_action(action)

            # Reward anticipatory actions
            if self._is_anticipatory_action(
                agent_action_type, prediction, current_state
            ):
                anticipation_reward = 0.15 * pred_confidence
                self.anticipation_rewards += anticipation_reward

        # Reward quick reactions to opponent changes
        if prev_state and current_state:
            opponent_change = self._detect_opponent_state_change(
                prev_state, current_state
            )
            if opponent_change and self._is_appropriate_reaction(
                action, opponent_change
            ):
                reaction_reward = 0.08
                self.reaction_rewards += reaction_reward

        # Distance management rewards
        distance_reward = self._calculate_distance_reward(current_state, action)

        # Validate prediction accuracy (delayed reward)
        accuracy_reward = self._validate_prediction_accuracy(current_state)

        total_predictive_reward = (
            prediction_reward
            + anticipation_reward
            + reaction_reward
            + distance_reward
            + accuracy_reward
        )

        return total_predictive_reward

    def _classify_agent_action(self, action):
        """Classify the agent's action for analysis"""
        # Convert action to interpretable type
        if isinstance(action, np.ndarray) and len(action) >= 12:
            # MultiBinary action analysis
            if action[4]:  # UP
                return "jumping"
            elif action[5]:  # DOWN
                return "crouching"
            elif action[6]:  # LEFT
                return "moving_left"
            elif action[7]:  # RIGHT
                return "moving_right"
            elif action[0] or action[1]:  # B or Y (attacks)
                return "attacking"
            elif action[8] or action[9]:  # A or X (strong attacks)
                return "strong_attacking"
            else:
                return "neutral"
        else:
            # Discrete action - map to categories
            if isinstance(action, (int, np.integer)) or (
                isinstance(action, np.ndarray) and action.size == 1
            ):
                action_int = (
                    int(action)
                    if isinstance(action, (int, np.integer))
                    else int(action.item())
                )

                action_mapping = {
                    0: "neutral",
                    1: "attacking",
                    2: "attacking",
                    3: "jumping",
                    4: "crouching",
                    5: "moving_left",
                    6: "moving_right",
                    7: "strong_attacking",
                    8: "strong_attacking",
                }
                return action_mapping.get(action_int % 25, "neutral")

        return "neutral"

    def _is_anticipatory_action(self, agent_action, prediction, current_state):
        """Check if agent action shows anticipation of opponent behavior"""
        if prediction == "will_continue_right":
            # Good anticipation: move to intercept or prepare counter
            return agent_action in ["moving_left", "attacking", "strong_attacking"]

        elif prediction == "will_continue_left":
            return agent_action in ["moving_right", "attacking", "strong_attacking"]

        elif prediction == "will_attack_aggressively":
            # Good anticipation: defensive actions
            return agent_action in ["crouching", "moving_left", "moving_right"]

        elif prediction == "will_repeat_action":
            # Exploit repetitive behavior
            return agent_action in ["attacking", "strong_attacking"]

        return False

    def _detect_opponent_state_change(self, prev_state, current_state):
        """Detect significant changes in opponent state"""
        if not prev_state or not current_state:
            return None

        # Position changes
        x_change = abs(current_state["opponent_x"] - prev_state["opponent_x"])
        y_change = abs(current_state["opponent_y"] - prev_state["opponent_y"])

        if x_change > 5:
            return "position_change"

        # Health changes (opponent took damage)
        health_change = prev_state["opponent_health"] - current_state["opponent_health"]
        if health_change > 0:
            return "opponent_damaged"

        # Range changes
        if current_state["range_category"] != prev_state["range_category"]:
            return "range_change"

        return None

    def _is_appropriate_reaction(self, action, opponent_change):
        """Check if action is appropriate reaction to opponent change"""
        agent_action = self._classify_agent_action(action)

        if opponent_change == "opponent_damaged":
            # Good to continue pressure
            return agent_action in ["attacking", "strong_attacking", "moving_right"]

        elif opponent_change == "position_change":
            # Good to adjust position
            return agent_action in ["moving_left", "moving_right", "jumping"]

        elif opponent_change == "range_change":
            # Adjust strategy based on new range
            return agent_action in ["moving_left", "moving_right", "attacking"]

        return False

    def _calculate_distance_reward(self, current_state, action):
        """Reward optimal distance management"""
        distance = current_state["distance"]
        agent_action = self._classify_agent_action(action)

        # Reward maintaining optimal distances
        if current_state["range_category"] == "mid_range":
            if agent_action in ["attacking", "strong_attacking"]:
                return 0.02  # Good range for attacks

        elif current_state["range_category"] == "close_range":
            if agent_action in ["attacking", "crouching"]:
                return 0.02  # Close combat

        elif current_state["range_category"] == "far_range":
            if agent_action in ["moving_right", "moving_left"]:
                return 0.01  # Closing distance

        return 0.0

    def _validate_prediction_accuracy(self, current_state):
        """Check if previous predictions were accurate and reward accordingly"""
        if not self.last_prediction:
            return 0.0

        # Check if enough time has passed to validate
        frames_since_prediction = self.episode_steps - self.last_prediction["frame"]
        if frames_since_prediction < 5 or frames_since_prediction > 15:
            return 0.0

        prediction = self.last_prediction["prediction"]
        confidence = self.last_prediction["confidence"]

        # Validate prediction based on current state
        accurate = False

        if prediction == "will_continue_right":
            # Check if opponent continued moving right
            if len(self.opponent_position_history) >= 2:
                recent_x = [
                    pos["x"] for pos in list(self.opponent_position_history)[-2:]
                ]
                if len(recent_x) >= 2 and recent_x[-1] > recent_x[-2]:
                    accurate = True

        elif prediction == "will_continue_left":
            if len(self.opponent_position_history) >= 2:
                recent_x = [
                    pos["x"] for pos in list(self.opponent_position_history)[-2:]
                ]
                if len(recent_x) >= 2 and recent_x[-1] < recent_x[-2]:
                    accurate = True

        # Record accuracy
        self.prediction_accuracy.append(accurate)

        # Clear prediction after validation
        self.last_prediction = None

        # Reward accurate predictions
        if accurate:
            return 0.1 * confidence
        else:
            return -0.02 * confidence  # Small penalty for wrong predictions

        return 0.0

    def _process_frame(self, rgb_frame):
        """Optimized frame processing for deep networks"""
        if len(rgb_frame.shape) == 3 and rgb_frame.shape[2] == 3:
            gray = np.dot(rgb_frame, [0.299, 0.587, 0.114])
            gray = gray.astype(np.uint8)
        else:
            gray = rgb_frame

        if gray.shape[:2] != (self.target_height, self.target_width):
            h_ratio = gray.shape[0] / self.target_height
            w_ratio = gray.shape[1] / self.target_width
            h_indices = (np.arange(self.target_height) * h_ratio).astype(int)
            w_indices = (np.arange(self.target_width) * w_ratio).astype(int)
            resized = gray[np.ix_(h_indices, w_indices)]
            return resized
        else:
            return gray

    def _stack_observation(self):
        """Stack frames for ultra-deep network input"""
        frames_list = list(self.frame_stack)

        while len(frames_list) < self.num_frames:
            if len(frames_list) > 0:
                frames_list.insert(0, frames_list[0].copy())
            else:
                dummy_frame = np.zeros(
                    (self.target_height, self.target_width), dtype=np.uint8
                )
                frames_list.append(dummy_frame)

        stacked = np.stack(frames_list, axis=0)
        return stacked

    def _extract_health(self, info):
        """Extract health information"""
        player_health = info.get("health", self.full_hp)
        opponent_health = info.get("enemy_health", self.full_hp)
        return player_health, opponent_health

    def _calculate_reward(
        self,
        curr_player_health,
        curr_opponent_health,
        action,
        current_state,
        prev_state,
    ):
        """Enhanced reward calculation with predictive components"""
        reward = 0.0
        done = False

        # Basic combat rewards
        if curr_player_health <= 0 or curr_opponent_health <= 0:
            self.total_rounds += 1

            if curr_opponent_health <= 0 and curr_player_health > 0:
                reward = 1.0
                self.wins += 1
                self.win_streak += 1
                self.best_win_streak = max(self.best_win_streak, self.win_streak)
                win_rate = self.wins / self.total_rounds if self.total_rounds > 0 else 0
                print(
                    f"🏆 WIN! Round {self.total_rounds} | {self.wins}W/{self.losses}L ({win_rate:.1%}) | Streak: {self.win_streak}"
                )

            elif curr_player_health <= 0 and curr_opponent_health > 0:
                reward = -1.0
                self.losses += 1
                self.win_streak = 0
                win_rate = self.wins / self.total_rounds if self.total_rounds > 0 else 0
                print(
                    f"💀 LOSS! Round {self.total_rounds} | {self.wins}W/{self.losses}L ({win_rate:.1%}) | Streak: 0"
                )

            if self.reset_round:
                done = True

            self._log_periodic_stats()

        else:
            # Damage-based rewards
            opponent_damage = self.prev_opponent_health - curr_opponent_health
            player_damage = self.prev_player_health - curr_player_health

            if opponent_damage > 0:
                reward = 0.05
            elif player_damage > 0:
                reward = -0.02

            # ADD PREDICTIVE REWARDS
            predictive_reward = self._calculate_predictive_reward(
                action, current_state, prev_state
            )
            reward += predictive_reward

        self.prev_player_health = curr_player_health
        self.prev_opponent_health = curr_opponent_health

        return reward, done

    def _log_periodic_stats(self):
        """Enhanced logging with predictive statistics"""
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

                # Predictive statistics
                prediction_accuracy = (
                    np.mean(self.prediction_accuracy) * 100
                    if self.prediction_accuracy
                    else 0
                )

                if win_rate >= 70:
                    performance = "🏆 EXCELLENT"
                elif win_rate >= 50:
                    performance = "⚔️ GOOD"
                elif win_rate >= 30:
                    performance = "📈 IMPROVING"
                else:
                    performance = "🎯 LEARNING"

                print(f"\n📊 ENHANCED TRAINING STATS:")
                print(f"   {performance} | Win Rate: {win_rate:.1f}%")
                print(
                    f"   🎮 Rounds: {self.total_rounds} ({self.wins}W/{self.losses}L)"
                )
                print(
                    f"   🔥 Best Streak: {self.best_win_streak} | Current: {self.win_streak}"
                )
                print(f"   💰 Avg Reward: {avg_episode_reward:.2f}")
                print(f"   🔮 Prediction Accuracy: {prediction_accuracy:.1f}%")
                print(f"   🧠 Anticipation Rewards: {self.anticipation_rewards:.2f}")
                print(f"   🎯 Reaction Rewards: {self.reaction_rewards:.2f}")
                print(f"   📊 Recognized Patterns: {len(self.recognized_patterns)}")
                print(f"   ⏱️  Avg Episode: {avg_episode_length:.0f} steps")
                print(f"   🕐 Session: {session_time/60:.1f} min")
                print(f"   📈 Total Steps: {self.total_steps:,}")
                print()

            self.last_log_time = current_time

    def _discrete_to_multibinary(self, action_int):
        """Convert discrete action to MultiBinary button combination"""
        # Define common button combinations for fighting games
        # Based on typical Genesis controller: [B, Y, SELECT, START, UP, DOWN, LEFT, RIGHT, A, X, L, R]

        button_combinations = [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 0: No input
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 1: B (punch/attack)
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 2: Y (kick)
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # 3: UP (jump)
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],  # 4: DOWN (crouch)
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # 5: LEFT
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],  # 6: RIGHT
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],  # 7: A (strong punch)
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],  # 8: X (strong kick)
            [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # 9: B + UP (jump attack)
            [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],  # 10: B + DOWN (crouch attack)
            [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # 11: B + LEFT (attack left)
            [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],  # 12: B + RIGHT (attack right)
            [0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # 13: Y + UP (jump kick)
            [0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],  # 14: Y + DOWN (crouch kick)
            [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # 15: Y + LEFT (kick left)
            [0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],  # 16: Y + RIGHT (kick right)
            [0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0],  # 17: LEFT + A (strong left)
            [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0],  # 18: RIGHT + A (strong right)
            [0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0],  # 19: DOWN + A (special move)
            [0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],  # 20: UP + A (anti-air)
            [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],  # 21: DOWN + LEFT
            [0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0],  # 22: DOWN + RIGHT
            [0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0],  # 23: UP + LEFT
            [0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],  # 24: UP + RIGHT
        ]

        # Clamp action to valid range
        action_int = action_int % len(button_combinations)
        return np.array(button_combinations[action_int], dtype=np.int32)

    def reset(self, **kwargs):
        """Reset environment for new episode"""
        observation, info = self.env.reset(**kwargs)

        # Reset health tracking
        self.prev_player_health = self.full_hp
        self.prev_opponent_health = self.full_hp

        # Reset predictive components
        self.opponent_position_history.clear()
        self.opponent_action_history.clear()
        self.opponent_health_history.clear()
        self.player_position_history.clear()
        self.player_action_history.clear()
        self.last_prediction = None

        # Reset reward tracking
        self.anticipation_rewards = 0.0
        self.prediction_rewards = 0.0
        self.reaction_rewards = 0.0

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

        # Fill frame stack with initial frame
        for _ in range(self.num_frames - 1):
            self.frame_stack.append(processed_frame.copy())

        # Initialize game state tracking
        initial_state = self._extract_game_state(info)
        self.opponent_position_history.append(
            {"x": initial_state["opponent_x"], "y": initial_state["opponent_y"]}
        )
        self.player_position_history.append(
            {"x": initial_state["player_x"], "y": initial_state["player_y"]}
        )
        self.opponent_health_history.append(initial_state["opponent_health"])

        stacked_obs = self._stack_observation()
        return stacked_obs, info

    def step(self, action):
        """Enhanced step with predictive analysis"""

        # Store previous state for comparison
        prev_state = None
        if len(self.opponent_position_history) > 0:
            prev_state = {
                "opponent_x": self.opponent_position_history[-1]["x"],
                "opponent_y": self.opponent_position_history[-1]["y"],
                "opponent_health": (
                    self.opponent_health_history[-1]
                    if self.opponent_health_history
                    else self.full_hp
                ),
                "distance": 0,  # Will be calculated
                "range_category": "mid_range",
            }

        # Debug action format (only for first few steps)
        if self.total_steps < 5:
            print(
                f"🔍 Raw action: type={type(action)}, shape={getattr(action, 'shape', 'N/A')}, value={action}"
            )

        # Handle different action formats based on action space type
        try:
            if self.action_type == "multibinary":
                # MultiBinary action space (button combinations)
                if isinstance(action, np.ndarray):
                    if action.shape == (12,):  # Correct MultiBinary format
                        action_array = action.astype(np.int32)
                        if self.total_steps < 5:
                            print(f"🎮 MultiBinary action: {action_array}")
                    elif action.ndim == 0:  # Single value - convert to MultiBinary
                        action_int = int(action.item())
                        action_array = self._discrete_to_multibinary(action_int)
                        if self.total_steps < 5:
                            print(
                                f"🎮 Converted {action_int} to MultiBinary: {action_array}"
                            )
                    elif action.size == 1:  # 1-element array
                        action_int = int(action.flatten()[0])
                        action_array = self._discrete_to_multibinary(action_int)
                        if self.total_steps < 5:
                            print(
                                f"🎮 Converted {action_int} to MultiBinary: {action_array}"
                            )
                    else:
                        # Handle unexpected shapes
                        if len(action.flatten()) >= 12:
                            action_array = action.flatten()[:12].astype(np.int32)
                        else:
                            action_int = int(action.flatten()[0])
                            action_array = self._discrete_to_multibinary(action_int)
                        if self.total_steps < 5:
                            print(f"🎮 Handled unexpected shape: {action_array}")

                elif isinstance(action, (int, np.integer)):
                    # Convert discrete action to MultiBinary
                    action_array = self._discrete_to_multibinary(int(action))
                    if self.total_steps < 5:
                        print(f"🎮 Converted {action} to MultiBinary: {action_array}")

                else:
                    print(f"⚠️ Unknown action type: {type(action)}, using no-op")
                    action_array = np.zeros(12, dtype=np.int32)

                # Ensure correct format
                if action_array.shape != (12,):
                    print(f"⚠️ Wrong action shape: {action_array.shape}, fixing")
                    if len(action_array) < 12:
                        action_array = np.pad(
                            action_array, (0, 12 - len(action_array)), "constant"
                        )
                    else:
                        action_array = action_array[:12]
                    action_array = action_array.astype(np.int32)

                final_action = action_array

            else:
                # Discrete action space
                if isinstance(action, np.ndarray):
                    if action.ndim == 0:
                        action_int = int(action.item())
                    elif action.size == 1:
                        action_int = int(action.flatten()[0])
                    else:
                        action_int = int(action.flatten()[0])
                        print(
                            f"⚠️ Multi-element array for discrete action: {action}, using first element"
                        )

                elif hasattr(action, "item"):
                    action_int = int(action.item())

                elif isinstance(action, (int, np.integer)):
                    action_int = int(action)

                else:
                    action_int = int(action)
                    print(
                        f"⚠️ Unusual action type: {type(action)}, converted to: {action_int}"
                    )

                # Validate discrete action range
                if hasattr(self.env.action_space, "n"):
                    if not (0 <= action_int < self.env.action_space.n):
                        print(
                            f"⚠️ Action {action_int} out of range [0, {self.env.action_space.n-1}], clipping"
                        )
                        action_int = np.clip(action_int, 0, self.env.action_space.n - 1)

                final_action = action_int
                if self.total_steps < 5:
                    print(f"🎮 Discrete action: {final_action}")

        except Exception as e:
            print(f"❌ Action processing failed: {e}")
            if self.action_type == "multibinary":
                final_action = np.zeros(12, dtype=np.int32)
            else:
                final_action = 0

        # Execute action in environment
        try:
            result = self.env.step(final_action)

            # Handle gymnasium return format
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
            print(f"   Action: {final_action} (type: {type(final_action)})")

            # Return emergency fallback values
            dummy_obs = np.zeros(
                (self.target_height, self.target_width), dtype=np.uint8
            )
            self.frame_stack.append(dummy_obs)
            stacked_obs = self._stack_observation()
            return stacked_obs, -1.0, True, True, {"error": str(e)}

        # Extract current game state
        current_state = self._extract_game_state(info)

        # Update position and health histories
        self.opponent_position_history.append(
            {"x": current_state["opponent_x"], "y": current_state["opponent_y"]}
        )
        self.player_position_history.append(
            {"x": current_state["player_x"], "y": current_state["player_y"]}
        )
        self.opponent_health_history.append(current_state["opponent_health"])

        # Store action in history
        agent_action_type = self._classify_agent_action(final_action)
        self.player_action_history.append(agent_action_type)

        # Simple opponent action inference (could be enhanced with more game state info)
        if len(self.opponent_position_history) >= 2:
            prev_pos = self.opponent_position_history[-2]
            curr_pos = self.opponent_position_history[-1]
            if abs(curr_pos["x"] - prev_pos["x"]) > 2:
                opponent_action = "moving"
            elif (
                len(self.opponent_health_history) >= 2
                and self.opponent_health_history[-1] < self.opponent_health_history[-2]
            ):
                opponent_action = "damaged"
            else:
                opponent_action = "neutral"
            self.opponent_action_history.append(opponent_action)

        # Extract health and calculate enhanced reward with predictive components
        curr_player_health, curr_opponent_health = self._extract_health(info)
        custom_reward, custom_done = self._calculate_reward(
            curr_player_health,
            curr_opponent_health,
            final_action,
            current_state,
            prev_state,
        )

        if custom_done:
            done = custom_done

        # Process frame and update stack
        processed_frame = self._process_frame(observation)
        self.frame_stack.append(processed_frame)
        stacked_obs = self._stack_observation()

        # Update episode tracking
        self.episode_steps += 1
        self.total_steps += 1
        self.episode_reward += custom_reward

        # Check for episode timeout
        if self.episode_steps >= self.max_episode_steps:
            truncated = True
            print(f"⏰ Episode timeout at {self.episode_steps} steps")

        return stacked_obs, custom_reward, done, truncated, info

    def close(self):
        """Clean shutdown with final statistics including predictive metrics"""
        print(f"\n🏁 FINAL ENHANCED STATISTICS:")

        if self.total_rounds > 0:
            final_win_rate = self.wins / self.total_rounds * 100
            session_time = time.time() - self.session_start
            prediction_accuracy = (
                np.mean(self.prediction_accuracy) * 100
                if self.prediction_accuracy
                else 0
            )

            print(f"   🎯 Final Win Rate: {final_win_rate:.1f}%")
            print(f"   🎮 Total Rounds: {self.total_rounds}")
            print(f"   🏆 Wins: {self.wins}")
            print(f"   💀 Losses: {self.losses}")
            print(f"   🔥 Best Win Streak: {self.best_win_streak}")
            print(f"   📊 Total Episodes: {self.total_episodes}")
            print(f"   📈 Total Steps: {self.total_steps:,}")
            print(f"   🕐 Session Time: {session_time/3600:.2f} hours")

            # Predictive performance metrics
            print(f"\n🔮 ENHANCED PERFORMANCE:")
            print(f"   🧠 Prediction Accuracy: {prediction_accuracy:.1f}%")
            print(f"   🎯 Total Anticipation Rewards: {self.anticipation_rewards:.2f}")
            print(f"   ⚡ Total Reaction Rewards: {self.reaction_rewards:.2f}")
            print(f"   🔍 Total Prediction Rewards: {self.prediction_rewards:.2f}")
            print(f"   📊 Recognized Patterns: {len(self.recognized_patterns)}")

            if self.recognized_patterns:
                print(f"   📝 Pattern Types: {list(self.recognized_patterns.keys())}")

            if self.episode_rewards:
                print(f"   💰 Avg Episode Reward: {np.mean(self.episode_rewards):.2f}")
                print(
                    f"   📏 Avg Episode Length: {np.mean(self.episode_lengths):.0f} steps"
                )

            if final_win_rate >= 70:
                summary = "🏆 EXCELLENT ENHANCED PERFORMANCE!"
            elif final_win_rate >= 50:
                summary = "⚔️ GOOD ENHANCED PERFORMANCE!"
            elif final_win_rate >= 30:
                summary = "📈 SOLID ENHANCED IMPROVEMENT!"
            else:
                summary = "🎯 GOOD ENHANCED LEARNING PROGRESS!"

            print(f"   {summary}")

        super().close()

    @property
    def current_stats(self):
        """Return current training statistics including predictive metrics"""
        win_rate = self.wins / self.total_rounds if self.total_rounds > 0 else 0
        avg_reward = np.mean(self.episode_rewards) if self.episode_rewards else 0
        prediction_accuracy = (
            np.mean(self.prediction_accuracy) if self.prediction_accuracy else 0
        )

        return {
            "win_rate": win_rate,
            "wins": self.wins,
            "losses": self.losses,
            "total_rounds": self.total_rounds,
            "win_streak": self.win_streak,
            "best_win_streak": self.best_win_streak,
            "avg_episode_reward": avg_reward,
            "total_episodes": self.total_episodes,
            "total_steps": self.total_steps,
            "prediction_accuracy": prediction_accuracy,
            "anticipation_rewards": self.anticipation_rewards,
            "reaction_rewards": self.reaction_rewards,
            "prediction_rewards": self.prediction_rewards,
            "recognized_patterns": len(self.recognized_patterns),
            "pattern_types": list(self.recognized_patterns.keys()),
        }
