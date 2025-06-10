import collections
import gymnasium as gym
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW


class SamuraiShowdownCustomWrapper(gym.Wrapper):
    """Simplified aggressive wrapper for Samurai Showdown training"""

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

        print(f"🚀 {self.env_id} Simplified Aggressive Wrapper")
        print(f"   🎯 Simple rewards: Win/Loss terminal, damage/aggression continuous")
        print(f"   📏 Episode length: {max_episode_steps} steps")
        print(f"   🔥 Anti-fire knife evasion enabled")

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

            print()
            global_stats["last_log_time"] = current_time

    def _calculate_reward(self, curr_player_health, curr_opponent_health, action=None):
        """SIMPLIFIED: All rewards calculated here"""
        reward = 0.0
        done = False

        # 1. ATTACK BONUS - Encourage aggression
        if action is not None:
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
                # Small penalty for defensive play
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

        # 4. FIRE KNIFE EVASION - Detect and reward avoiding big attacks
        # Look for sudden large damage spikes (fire knife does big damage)
        if hasattr(self, "prev_player_health"):
            damage_taken = self.prev_player_health - curr_player_health
            if damage_taken >= 20:  # Fire knife does significant damage
                # Extra penalty for getting hit by special attacks
                reward -= 0.5
            elif damage_taken == 0 and action in [1, 3]:  # Moved away successfully
                # Small bonus for evasive movement when not taking damage
                reward += 0.1

        # 5. TERMINAL REWARDS - Simple win/loss
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

                total_win_reward = 2.0 + dominance_bonus + quick_bonus
                reward += total_win_reward

                print(
                    f"🏆 {self.env_id} WIN! Health: {curr_player_health}/{self.full_hp} "
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
                    f"💀 {self.env_id} LOSS! ({self.wins}W/{self.losses}L - {win_rate:.1%})"
                )
                done = True

            if self.reset_round:
                done = True

            self._log_periodic_stats()

        # Update health tracking
        self.prev_player_health = curr_player_health
        self.prev_opponent_health = curr_opponent_health

        # Clamp reward to reasonable bounds
        reward = np.clip(reward, -3.0, 3.0)

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

        self.frame_stack.clear()
        processed_frame = self._process_frame(observation)
        self.frame_stack.append(processed_frame.copy())

        for _ in range(self.num_frames - 1):
            zero_frame = np.zeros_like(processed_frame)
            self.frame_stack.append(zero_frame)

        stacked_obs = self._stack_observation()
        return stacked_obs, info

    def _bias_action_toward_attacks(self, action):
        """Force more aggressive action selection"""
        attack_actions = [8, 9, 10, 11]
        defensive_actions = [4, 5, 6, 7]

        # FIRE KNIFE EVASION: If we detect potential incoming fire knife
        # (this would need game-specific detection, for now use heuristics)
        # Force evasive movement occasionally
        if np.random.random() < 0.1:  # 10% chance to evade
            evasive_actions = [1, 3]  # Left/right movement
            return np.random.choice(evasive_actions)

        # FORCE AGGRESSION: 40% chance to force attack
        if np.random.random() < 0.4:
            return np.random.choice(attack_actions)

        # Reduce jumping (often action 0)
        if action == 0 and np.random.random() > 0.1:
            return np.random.choice(attack_actions)

        # Replace blocking with attacks 60% of the time
        if action in defensive_actions and np.random.random() < 0.6:
            return np.random.choice(attack_actions)

        # 30% chance to force attack for any non-attack action
        if action not in attack_actions and np.random.random() < 0.3:
            return np.random.choice(attack_actions)

        return action

    def step(self, action):
        """Simplified step with all rewards in _calculate_reward"""
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

        # Apply aggression bias
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

        # ALL REWARDS CALCULATED HERE
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


# Keep the existing DecisionTransformer and training classes unchanged
class DecisionTransformer(nn.Module):
    """Decision Transformer for Samurai Showdown"""

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

        # CNN encoder
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

        # Encode states
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
        """Load model"""
        obs_shape = env.observation_space.shape
        action_dim = env.action_space.n
        model = cls(obs_shape, action_dim)
        model.load_state_dict(torch.load(path, map_location=device))
        model.to(device)
        print(f"📂 Model loaded from: {path}")
        return model


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
    """Train Decision Transformer with memory optimizations"""

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

    print(f"🚀 Training Decision Transformer:")
    print(f"   Device: {device}")
    print(f"   Epochs: {epochs}")
    print(f"   Batch size: {batch_size}")
    print(f"   Learning rate: {lr}")
    print(f"   Context length: {context_length}")
    print(f"   Dataset size: {len(dataset)}")

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

    print(f"✅ Training complete! Best loss: {best_loss:.4f}")
    return model
