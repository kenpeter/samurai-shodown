import collections
import gymnasium as gym
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
import cv2


class FireKnifeDetector:
    """Ultra strict fire knife detection - avoid false positives"""

    def __init__(self):
        self.prev_frame = None
        self.detection_cooldown = 0  # Prevent spam detection

    def analyze_frame(self, frame, opponent_health, player_health):
        """Ultra strict detection - only major changes"""

        # Convert to uint8 if needed
        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8)

        # Long cooldown to prevent spam detection
        if self.detection_cooldown > 0:
            self.detection_cooldown -= 1
            self.prev_frame = frame.copy()
            return 0.0

        # Need previous frame for motion detection
        if self.prev_frame is None:
            self.prev_frame = frame.copy()
            return 0.0

        # Find frame difference (motion detection)
        frame_diff = cv2.absdiff(frame, self.prev_frame)

        # VERY high threshold to ignore almost everything
        _, new_bright = cv2.threshold(frame_diff, 120, 255, cv2.THRESH_BINARY)

        # Count new bright pixels
        new_bright_ratio = np.sum(new_bright > 0) / new_bright.size

        # Update previous frame
        self.prev_frame = frame.copy()

        # EXTREMELY strict thresholds
        if new_bright_ratio > 0.15:  # 15% of screen MASSIVELY changed
            self.detection_cooldown = 30  # Don't detect again for 30 frames (1 second)
            return 0.8  # High threat
        elif new_bright_ratio > 0.10:  # 10% changed dramatically
            self.detection_cooldown = 20  # 20 frame cooldown
            return 0.3  # Medium threat

        return 0.0  # No significant change


class SamuraiShowdownSimpleWrapper(gym.Wrapper):
    """SIMPLIFIED wrapper with basic fire knife evasion"""

    def __init__(self, env, reset_round=True, rendering=False, max_episode_steps=15000):
        super(SamuraiShowdownSimpleWrapper, self).__init__(env)
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

        # SIMPLE tracking
        self.wins = 0
        self.losses = 0
        self.total_rounds = 0

        # Fire knife detection (much more selective)
        self.fire_knife_detector = None  # Will initialize after getting frame dims
        self.evasion_window = 0  # Frames left to dodge
        self.successful_evasions = 0
        self.last_bright_pixels = 0  # Track brightness for evasion detection
        self.fire_detections = 0  # Track how many detections per episode

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

        # Handle action space conversion
        if hasattr(self.env.action_space, "n"):
            self.action_space = gym.spaces.Discrete(self.env.action_space.n)
            self._original_action_space = self.env.action_space
        else:
            self.action_space = self.env.action_space
            self._original_action_space = self.env.action_space

        # Initialize fire knife detector after getting dimensions
        self.fire_knife_detector = FireKnifeDetector()

        print(f"🎯 SIMPLE Wrapper with Motion+Brightness Fire Detection")
        print(f"   Main rewards: DAMAGE ±0.1, WIN +1, LOSE -1")
        print(f"   🔥 Motion-based fire detection (ignores background fire)")

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

    def _calculate_reward(self, curr_player_health, curr_opponent_health):
        """ULTRA SIMPLE rewards + OpenCV fire knife detection"""
        reward = 0.0
        done = False

        # Fire knife detection with OpenCV (very simple)
        if len(self.frame_stack) > 0:
            current_frame = list(self.frame_stack)[-1]
            threat_level = self.fire_knife_detector.analyze_frame(
                current_frame, curr_opponent_health, curr_player_health
            )

            # Set evasion window if high threat detected (SILENT MODE)
            if threat_level > 0.7:
                self.evasion_window = 3  # 3 frames to dodge
                self.fire_detections += 1
                # Only log occasionally to avoid spam
                if self.fire_detections % 5 == 1:  # Log every 5th detection
                    print(f"🔥 FIRE SWORD #{self.fire_detections}! (Logging every 5th)")
            elif self.evasion_window > 0:
                self.evasion_window -= 1

        # 1. DAMAGE OPPONENT = +0.1 per HP
        if hasattr(self, "prev_opponent_health"):
            if curr_opponent_health < self.prev_opponent_health:
                damage_dealt = self.prev_opponent_health - curr_opponent_health
                reward += damage_dealt * 0.1
                print(f"💥 DAMAGE: {damage_dealt} HP -> +{damage_dealt * 0.1:.1f}")

        # 2. GET INJURED = -0.1 per HP
        if hasattr(self, "prev_player_health"):
            if curr_player_health < self.prev_player_health:
                damage_taken = self.prev_player_health - curr_player_health
                reward -= damage_taken * 0.1
                print(f"💀 INJURED: {damage_taken} HP -> -{damage_taken * 0.1:.1f}")

        # 3. MOTION-BASED FIRE KNIFE EVASION (OpenCV Option 1)
        if hasattr(self, "prev_player_health"):
            damage_taken = self.prev_player_health - curr_player_health

            # Simple check: if we detected NEW bright areas but took little damage
            current_frame = (
                list(self.frame_stack)[-1] if len(self.frame_stack) > 0 else None
            )
            if (
                current_frame is not None
                and self.fire_knife_detector.prev_frame is not None
            ):
                # Use ULTRA strict motion detection
                frame_diff = cv2.absdiff(
                    current_frame.astype(np.uint8),
                    self.fire_knife_detector.prev_frame.astype(np.uint8),
                )
                _, new_bright = cv2.threshold(
                    frame_diff, 120, 255, cv2.THRESH_BINARY
                )  # VERY high threshold
                new_bright_ratio = np.sum(new_bright > 0) / new_bright.size

                # Only reward for MASSIVE visual changes
                if (
                    new_bright_ratio > 0.12 and damage_taken < 3
                ):  # 12% of screen changed dramatically
                    reward += 0.2  # Small evasion bonus
                    self.successful_evasions += 1
                    print(
                        f"🛡️ MAJOR EVASION! +0.2 (Huge change: {new_bright_ratio:.3f})"
                    )

        # 4. WIN/LOSE (simple)
        if curr_player_health <= 0 or curr_opponent_health <= 0:
            self.total_rounds += 1

            if curr_opponent_health <= 0 and curr_player_health > 0:
                # WIN = +1
                self.wins += 1
                win_rate = self.wins / self.total_rounds
                reward += 1.0
                print(
                    f"🏆 WIN! +1.0 Fire detections: {self.fire_detections} ({self.wins}W/{self.losses}L - {win_rate:.1%})"
                )
                done = True

            elif curr_player_health <= 0 and curr_opponent_health > 0:
                # LOSE = -1
                self.losses += 1
                win_rate = self.wins / self.total_rounds
                reward -= 1.0
                print(
                    f"💀 LOSE! -1.0 Fire detections: {self.fire_detections} ({self.wins}W/{self.losses}L - {win_rate:.1%})"
                )
                done = True

            if self.reset_round:
                done = True

        # Update health tracking
        self.prev_player_health = curr_player_health
        self.prev_opponent_health = curr_opponent_health

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
        self.evasion_window = 0
        self.successful_evasions = 0
        self.last_bright_pixels = 0
        self.fire_detections = 0

        # Reset fire knife detector (simple - no history needed)
        self.fire_knife_detector = FireKnifeDetector()

        self.frame_stack.clear()
        processed_frame = self._process_frame(observation)
        self.frame_stack.append(processed_frame.copy())

        for _ in range(self.num_frames - 1):
            zero_frame = np.zeros_like(processed_frame)
            self.frame_stack.append(zero_frame)

        stacked_obs = self._stack_observation()
        return stacked_obs, info

    def step(self, action):
        """SIMPLE step - no action modification!"""
        # Convert action to proper format (NO MODIFICATION)
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

        # NO ACTION BIAS - let the agent learn naturally!

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

        # Extract health and calculate SIMPLE rewards
        curr_player_health, curr_opponent_health = self._extract_health(info)

        # SIMPLE REWARDS ONLY
        total_reward, custom_done = self._calculate_reward(
            curr_player_health, curr_opponent_health
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

        return stacked_obs, total_reward, done, truncated, info


# Simple Decision Transformer (keep original name)
class DecisionTransformer(nn.Module):
    """Simple Decision Transformer - focus on learning, not complexity"""

    def __init__(
        self,
        observation_shape,
        action_dim,
        hidden_size=128,  # Smaller = faster learning
        n_layer=3,  # Fewer layers = simpler
        n_head=4,
        max_ep_len=1000,  # Shorter episodes
    ):
        super().__init__()
        self.observation_shape = observation_shape
        self.action_dim = action_dim
        self.hidden_size = hidden_size
        self.max_ep_len = max_ep_len

        # Simple CNN encoder
        self.cnn_encoder = nn.Sequential(
            nn.Conv2d(observation_shape[0], 32, kernel_size=8, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),  # Simpler
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
        )

        cnn_output_size = 64 * 4 * 4

        # Simple embeddings
        self.state_encoder = nn.Linear(cnn_output_size, hidden_size)
        self.action_encoder = nn.Embedding(action_dim, hidden_size)
        self.return_encoder = nn.Linear(1, hidden_size)
        self.timestep_encoder = nn.Embedding(max_ep_len, hidden_size)

        # Simple transformer
        self.ln = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(0.1)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=n_head,
            dim_feedforward=2 * hidden_size,  # Smaller
            dropout=0.1,
            activation="relu",  # Simpler than GELU
            batch_first=True,
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

        # Encode states - SIMPLE
        states_flat = states.view(-1, *self.observation_shape)
        state_features = self.cnn_encoder(states_flat)
        state_embeddings = self.state_encoder(state_features)
        state_embeddings = state_embeddings.view(batch_size, seq_len, self.hidden_size)

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
        """Get action for inference"""
        self.eval()
        with torch.no_grad():
            logits = self.forward(states, actions, returns_to_go, timesteps)
            last_logits = logits[0, -1] / temperature
            probs = F.softmax(last_logits, dim=-1)
            action = torch.multinomial(probs, 1).item()
        return action

    def save(self, path):
        """Simple save"""
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path, env=None, device="cpu"):
        """Load model"""
        obs_shape = env.observation_space.shape
        action_dim = env.action_space.n
        model = cls(obs_shape, action_dim)
        model.load_state_dict(torch.load(path, map_location=device))
        model.to(device)
        print(f"📂 Model loaded from: {path}")
        return model


# Keep original wrapper name
SamuraiShowdownCustomWrapper = SamuraiShowdownSimpleWrapper


# Keep existing dataset and training functions (they work fine)
class TrajectoryDataset(Dataset):
    """Dataset for Decision Transformer training"""

    def __init__(self, trajectories, context_length=20):  # Shorter context
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
    epochs=30,  # Fewer epochs
    batch_size=16,
    lr=1e-3,  # Higher learning rate for faster learning
    device="cuda",
    context_length=20,  # Shorter context
):
    """Train Simple Decision Transformer"""

    dataset = TrajectoryDataset(trajectories, context_length)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=1,
        persistent_workers=False,
    )

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    model.to(device)
    model.train()

    print(f"🚀 Training SIMPLE Decision Transformer:")
    print(f"   Device: {device}")
    print(f"   Epochs: {epochs}")
    print(f"   Batch size: {batch_size}")
    print(f"   Learning rate: {lr}")
    print(f"   Context length: {context_length}")
    print(f"   🎯 FOCUS: Only damage and wins matter")

    best_loss = float("inf")

    for epoch in range(epochs):
        total_loss = 0
        num_batches = 0

        for batch_idx, batch in enumerate(dataloader):
            states = batch["states"].to(device, non_blocking=True)
            actions = batch["actions"].to(device, non_blocking=True)
            returns_to_go = batch["returns_to_go"].to(device, non_blocking=True)
            timesteps = batch["timesteps"].to(device, non_blocking=True)

            optimizer.zero_grad()

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

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        scheduler.step()
        avg_loss = total_loss / max(num_batches, 1)

        if avg_loss < best_loss:
            best_loss = avg_loss

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"   Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

    print(f"✅ Simple training complete! Best loss: {best_loss:.4f}")
    return model
