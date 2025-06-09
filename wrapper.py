import collections
import gymnasium as gym
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
import json
import os


class SamuraiShowdownCustomWrapper(gym.Wrapper):
    """Simple wrapper - Win/Loss rewards only, optimized for CUDA training with large batch sizes"""

    # Global tracking across all environments
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

        # Frame processing parameters - OPTIMIZED FOR GPU
        self.resize_scale = 0.75
        self.num_frames = 9
        self.frame_stack = collections.deque(maxlen=self.num_frames)

        # Health tracking
        self.full_hp = 128
        self.prev_player_health = self.full_hp
        self.prev_opponent_health = self.full_hp

        # Episode management - LONGER episodes for larger batches
        self.max_episode_steps = max_episode_steps
        self.episode_steps = 0
        self.reset_round = reset_round

        # No action filtering - let agent learn naturally

        # Environment tracking
        import random

        self.env_id = f"ENV-{random.randint(1000, 9999)}"
        self.wins = 0
        self.losses = 0
        self.total_rounds = 0

        # Initialize global tracking for this environment
        SamuraiShowdownCustomWrapper._global_stats["env_stats"][self.env_id] = {
            "wins": 0,
            "losses": 0,
            "total_rounds": 0,
        }

        # Logging configuration
        self.log_interval = 120  # Log every 2 minutes

        # Get frame dimensions and calculate target size - FIXED FOR STABLE-RETRO
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

        # Update observation and action spaces for our wrapper
        if hasattr(self.env.action_space, "n"):
            # Convert MultiBinary to Discrete for our wrapper
            # This makes it easier for the Decision Transformer to handle
            self.action_space = gym.spaces.Discrete(self.env.action_space.n)
            self._original_action_space = self.env.action_space
            print(
                f"   🔄 Converted MultiBinary({self.env.action_space.n}) to Discrete({self.action_space.n})"
            )
        else:
            self.action_space = self.env.action_space
            self._original_action_space = self.env.action_space

        print(f"🚀 {self.env_id} CUDA-OPTIMIZED Wrapper")
        print(f"   🎯 Rewards: +1 Win, -1 Loss, 0 everything else")
        print(f"   📏 Episode length: {max_episode_steps} steps (for MASSIVE batches)")
        print(f"   💾 Memory optimized for 12GB VRAM")

    def _process_frame(self, rgb_frame):
        """Convert RGB frame to grayscale and resize - OPTIMIZED"""
        if len(rgb_frame.shape) == 3 and rgb_frame.shape[2] == 3:
            # Optimized grayscale conversion
            gray = np.dot(rgb_frame, [0.299, 0.587, 0.114])
            gray = gray.astype(np.uint8)
        else:
            gray = rgb_frame

        # Efficient resize using numpy indexing
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
        """Stack frames in channels-first format - MEMORY EFFICIENT"""
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
        """Extract health from info"""
        player_health = info.get("health", self.full_hp)
        opponent_health = info.get("enemy_health", self.full_hp)
        return player_health, opponent_health

    def _log_periodic_stats(self):
        """Log simple win/loss stats"""
        current_time = time.time()
        time_since_last_log = (
            current_time - SamuraiShowdownCustomWrapper._global_stats["last_log_time"]
        )

        if time_since_last_log >= self.log_interval:
            global_stats = SamuraiShowdownCustomWrapper._global_stats

            print(f"\n📊 CUDA TRAINING STATS:")

            for env_id, stats in global_stats["env_stats"].items():
                if stats["total_rounds"] > 0:
                    win_rate = stats["wins"] / stats["total_rounds"] * 100
                    if win_rate >= 60:
                        emoji = "🏆"
                    elif win_rate >= 40:
                        emoji = "⚔️"
                    else:
                        emoji = "📈"
                    print(
                        f"   {emoji} {env_id}: {stats['wins']}W/{stats['losses']}L ({win_rate:.1f}%)"
                    )
                else:
                    print(f"   ⏳ {env_id}: 0W/0L (0.0%)")

            print()
            global_stats["last_log_time"] = current_time

    def _calculate_reward(self, curr_player_health, curr_opponent_health):
        """SUPER SIMPLE - Only +1 win, -1 loss, 0 everything else"""
        reward = 0.0
        done = False

        # Check for round end
        if curr_player_health <= 0 or curr_opponent_health <= 0:
            self.total_rounds += 1
            SamuraiShowdownCustomWrapper._global_stats["total_rounds"] += 1

            if curr_opponent_health <= 0 and curr_player_health > 0:
                # WIN = +1
                self.wins += 1
                win_rate = self.wins / self.total_rounds
                SamuraiShowdownCustomWrapper._global_stats["total_wins"] += 1
                SamuraiShowdownCustomWrapper._global_stats["env_stats"][self.env_id][
                    "wins"
                ] = self.wins
                SamuraiShowdownCustomWrapper._global_stats["env_stats"][self.env_id][
                    "total_rounds"
                ] = self.total_rounds

                print(
                    f"🏆 {self.env_id} WIN! {self.wins}W/{self.losses}L ({win_rate:.1%})"
                )

                reward = 1.0  # Simple +1 for win

                self.prev_player_health = curr_player_health
                self.prev_opponent_health = curr_opponent_health
                return reward, True

            elif curr_player_health <= 0 and curr_opponent_health > 0:
                # LOSS = -1
                self.losses += 1
                win_rate = self.wins / self.total_rounds
                SamuraiShowdownCustomWrapper._global_stats["total_losses"] += 1
                SamuraiShowdownCustomWrapper._global_stats["env_stats"][self.env_id][
                    "losses"
                ] = self.losses
                SamuraiShowdownCustomWrapper._global_stats["env_stats"][self.env_id][
                    "total_rounds"
                ] = self.total_rounds

                print(
                    f"💀 {self.env_id} LOSS! {self.wins}W/{self.losses}L ({win_rate:.1%})"
                )

                reward = -1.0  # Simple -1 for loss

                self.prev_player_health = curr_player_health
                self.prev_opponent_health = curr_opponent_health
                return reward, True

            if self.reset_round:
                done = True

            self._log_periodic_stats()

        self.prev_player_health = curr_player_health
        self.prev_opponent_health = curr_opponent_health

        # Everything else = 0 reward
        return 0.0, done

    def reset(self, **kwargs):
        """Reset environment - FIXED FOR STABLE-RETRO"""
        result = self.env.reset(**kwargs)
        if isinstance(result, tuple):
            observation, info = result
        else:
            observation = result
            info = {}

        self.prev_player_health = self.full_hp
        self.prev_opponent_health = self.full_hp
        self.episode_steps = 0

        self.frame_stack.clear()
        processed_frame = self._process_frame(observation)
        self.frame_stack.append(processed_frame.copy())

        for _ in range(self.num_frames - 1):
            zero_frame = np.zeros_like(processed_frame)
            self.frame_stack.append(zero_frame)

        stacked_obs = self._stack_observation()
        return stacked_obs, info

    def step(self, action):
        """FIXED ACTION HANDLING for stable-retro + gymnasium with MultiBinary action space"""
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

        # For stable-retro with MultiBinary action space, convert discrete to binary array
        if hasattr(self._original_action_space, "n"):
            # MultiBinary action space - convert discrete action to binary array
            n_actions = self._original_action_space.n
            action_array = np.zeros(n_actions, dtype=np.int8)

            # Clamp action to valid range
            action_int = max(0, min(action_int, n_actions - 1))
            action_array[action_int] = 1
            action_to_use = action_array
        else:
            # Other action spaces (Discrete, etc.)
            action_to_use = action_int

        # Execute step
        try:
            result = self.env.step(action_to_use)

            # Handle different return formats between retro versions
            if len(result) == 5:
                observation, reward, done, truncated, info = result
            elif len(result) == 4:
                observation, reward, done, info = result
                truncated = False
            else:
                raise ValueError(f"Unexpected step return format: {len(result)} values")

        except Exception as e:
            print(f"❌ Step error with action {action_to_use}: {e}")
            print(f"   Original action space: {self._original_action_space}")
            print(f"   Action space type: {type(self._original_action_space)}")

            # Fallback: try with zero/neutral action
            if hasattr(self._original_action_space, "n"):
                fallback_action = np.zeros(self._original_action_space.n, dtype=np.int8)
            else:
                fallback_action = 0

            try:
                result = self.env.step(fallback_action)
                if len(result) == 5:
                    observation, reward, done, truncated, info = result
                elif len(result) == 4:
                    observation, reward, done, info = result
                    truncated = False
            except Exception as e2:
                print(f"❌ Even fallback action failed: {e2}")
                raise e2

        curr_player_health, curr_opponent_health = self._extract_health(info)
        custom_reward, custom_done = self._calculate_reward(
            curr_player_health, curr_opponent_health
        )

        if custom_done:
            done = custom_done

        processed_frame = self._process_frame(observation)
        self.frame_stack.append(processed_frame)
        stacked_obs = self._stack_observation()

        self.episode_steps += 1

        if self.episode_steps >= self.max_episode_steps:
            truncated = True

        return stacked_obs, custom_reward, done, truncated, info

    @classmethod
    def print_final_stats(cls):
        """Print final statistics when training ends"""
        pass


# === OPTIMIZED DECISION TRANSFORMER FOR CUDA ===
class DecisionTransformer(nn.Module):
    """CUDA-Optimized Decision Transformer for Samurai Showdown"""

    def __init__(
        self,
        observation_shape,
        action_dim,
        hidden_size=512,
        n_layer=8,
        n_head=8,
        max_ep_len=2000,
    ):
        super().__init__()
        self.observation_shape = observation_shape
        self.action_dim = action_dim
        self.hidden_size = hidden_size
        self.max_ep_len = max_ep_len

        # Optimized CNN encoder for CUDA
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

        # Calculate CNN output size
        cnn_output_size = 256 * 4 * 4

        # Embeddings
        self.state_encoder = nn.Linear(cnn_output_size, hidden_size)
        self.action_encoder = nn.Embedding(action_dim, hidden_size)
        self.return_encoder = nn.Linear(1, hidden_size)
        self.timestep_encoder = nn.Embedding(max_ep_len, hidden_size)

        # Optimized Transformer
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

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights for better training"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, states, actions, returns_to_go, timesteps, attention_mask=None):
        batch_size, seq_len = states.shape[:2]

        # Encode states through CNN
        states_flat = states.view(-1, *self.observation_shape)
        state_features = self.cnn_encoder(states_flat)
        state_embeddings = self.state_encoder(state_features)
        state_embeddings = state_embeddings.view(batch_size, seq_len, self.hidden_size)

        # Encode other inputs
        action_embeddings = self.action_encoder(actions)
        rtg_embeddings = self.return_encoder(returns_to_go.unsqueeze(-1))

        # Handle timesteps properly
        timesteps_clamped = torch.clamp(timesteps, 0, self.max_ep_len - 1)
        timestep_embeddings = self.timestep_encoder(timesteps_clamped)

        # Stack embeddings: (return, state, action)
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
        """Save model"""
        torch.save(self.state_dict(), path)
        print(f"💾 Model saved to: {path}")

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
    """CUDA-Optimized Dataset for Decision Transformer training"""

    def __init__(self, trajectories, context_length=50):
        self.trajectories = [
            t
            for t in trajectories
            if len(t["rewards"]) > context_length // 2 and len(t["states"]) > 0
        ]
        self.context_length = context_length

        print(f"📊 Dataset created with {len(self.trajectories)} trajectories")
        if len(self.trajectories) > 0:
            avg_length = np.mean([len(t["rewards"]) for t in self.trajectories])
            total_reward = np.mean([sum(t["rewards"]) for t in self.trajectories])
            print(f"   Average trajectory length: {avg_length:.1f}")
            print(f"   Average total reward: {total_reward:.2f}")

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        traj = self.trajectories[idx]
        states = np.array(traj["states"])
        actions = np.array(traj["actions"])
        rewards = np.array(traj["rewards"])

        # Ensure states and actions have same length
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


def collect_trajectories(env, num_episodes=500):
    """Collect trajectories for Decision Transformer training - SIMPLIFIED"""
    print(f"🎮 Collecting {num_episodes} trajectories for CUDA training...")
    trajectories = []

    for episode in range(num_episodes):
        trajectory = {"states": [], "actions": [], "rewards": []}
        obs, _ = env.reset()
        trajectory["states"].append(obs.copy())

        done = False
        truncated = False
        episode_reward = 0
        step_count = 0

        while not done and not truncated and step_count < 3000:
            # Simple random policy - no action filtering
            action = env.action_space.sample()

            try:
                obs, reward, done, truncated, _ = env.step(action)

                trajectory["actions"].append(action)
                trajectory["rewards"].append(reward)
                episode_reward += reward
                step_count += 1

                if not done and not truncated:
                    trajectory["states"].append(obs.copy())

            except Exception as e:
                print(f"⚠️ Step error in episode {episode}: {e}")
                break

        # Only keep trajectories that have some meaningful interaction
        if len(trajectory["actions"]) > 10:
            trajectories.append(trajectory)

        if (episode + 1) % 100 == 0:
            recent_rewards = [sum(t["rewards"]) for t in trajectories[-100:]]
            avg_reward = np.mean(recent_rewards) if recent_rewards else 0
            win_episodes = sum(1 for r in recent_rewards if r > 0)
            print(
                f"   Episode {episode + 1}/{num_episodes}, "
                f"Avg reward: {avg_reward:.2f}, "
                f"Wins: {win_episodes}/100"
            )

    print(f"✅ Collected {len(trajectories)} valid trajectories")
    return trajectories


def train_decision_transformer(
    model,
    trajectories,
    epochs=200,
    batch_size=64,
    lr=1e-4,
    device="cuda",
    context_length=50,
):
    """CUDA-Optimized Decision Transformer training"""

    # Create dataset and dataloader
    dataset = TrajectoryDataset(trajectories, context_length)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=4,
        persistent_workers=True,
    )

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4, betas=(0.9, 0.95))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    model.to(device)
    model.train()

    print(f"🚀 CUDA Training Decision Transformer:")
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
                f"   Epoch {epoch + 1}/{epochs}, "
                f"Loss: {avg_loss:.4f}, "
                f"LR: {scheduler.get_last_lr()[0]:.6f}, "
                f"Time: {epoch_time:.1f}s"
            )

            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**3
                cached = torch.cuda.memory_reserved() / 1024**3
                print(
                    f"   GPU Memory: {allocated:.2f}GB allocated, {cached:.2f}GB cached"
                )

    print(f"✅ CUDA Training complete! Best loss: {best_loss:.4f}")
    return model
