import collections
import gymnasium as gym
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
import os


class SamuraiShowdownCustomWrapper(gym.Wrapper):
    """Minimal wrapper - Win/Loss rewards with statistics"""

    def __init__(self, env, reset_round=True, rendering=False, max_episode_steps=5000):
        super().__init__(env)
        self.env = env
        self.resize_scale = 0.75
        self.num_frames = 9
        self.frame_stack = collections.deque(maxlen=self.num_frames)
        self.full_hp = 128
        self.prev_player_health = self.full_hp
        self.prev_opponent_health = self.full_hp
        self.max_episode_steps = max_episode_steps
        self.episode_steps = 0
        self.reset_round = reset_round

        # Win/Loss tracking
        self.wins = 0
        self.losses = 0
        self.total_rounds = 0

        # Get frame dimensions
        dummy_obs, _ = self.env.reset()
        original_height, original_width = dummy_obs.shape[:2]
        self.target_height = int(original_height * self.resize_scale)
        self.target_width = int(original_width * self.resize_scale)

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.num_frames, self.target_height, self.target_width), dtype=np.uint8
        )

        # Handle action space
        if hasattr(self.env.action_space, "n"):
            self.action_space = gym.spaces.Discrete(self.env.action_space.n)
            self._original_action_space = self.env.action_space
        else:
            self.action_space = self.env.action_space
            self._original_action_space = self.env.action_space

    def _process_frame(self, rgb_frame):
        """Convert to grayscale and resize"""
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
        """Stack frames"""
        frames_list = list(self.frame_stack)
        while len(frames_list) < self.num_frames:
            if len(frames_list) > 0:
                frames_list.insert(0, frames_list[0].copy())
            else:
                dummy_frame = np.zeros((self.target_height, self.target_width), dtype=np.uint8)
                frames_list.append(dummy_frame)
        return np.stack(frames_list, axis=0)

    def _calculate_reward(self, curr_player_health, curr_opponent_health):
        """Win/loss rewards with statistics tracking"""
        if curr_player_health <= 0 or curr_opponent_health <= 0:
            self.total_rounds += 1
            
            if curr_opponent_health <= 0 and curr_player_health > 0:
                # WIN
                self.wins += 1
                win_rate = self.wins / self.total_rounds * 100
                print(f"🏆 WIN! {self.wins}W/{self.losses}L ({win_rate:.1f}%)")
                return 1.0, True
                
            elif curr_player_health <= 0 and curr_opponent_health > 0:
                # LOSS
                self.losses += 1
                win_rate = self.wins / self.total_rounds * 100
                print(f"💀 LOSS! {self.wins}W/{self.losses}L ({win_rate:.1f}%)")
                return -1.0, True
                
            return 0.0, self.reset_round
        return 0.0, False

    def reset(self, **kwargs):
        """Reset environment"""
        result = self.env.reset(**kwargs)
        observation, info = result if isinstance(result, tuple) else (result, {})
        
        self.prev_player_health = self.full_hp
        self.prev_opponent_health = self.full_hp
        self.episode_steps = 0
        self.frame_stack.clear()
        
        processed_frame = self._process_frame(observation)
        self.frame_stack.append(processed_frame.copy())
        for _ in range(self.num_frames - 1):
            self.frame_stack.append(np.zeros_like(processed_frame))
            
        return self._stack_observation(), info

    def step(self, action):
        """Step environment"""
        # Convert action
        try:
            action_int = int(action.item()) if hasattr(action, "item") else int(action)
        except:
            action_int = 0

        # Handle action space
        if hasattr(self._original_action_space, "n"):
            n_actions = self._original_action_space.n
            action_array = np.zeros(n_actions, dtype=np.int8)
            action_int = max(0, min(action_int, n_actions - 1))
            action_array[action_int] = 1
            action_to_use = action_array
        else:
            action_to_use = action_int

        # Execute step
        result = self.env.step(action_to_use)
        if len(result) == 5:
            observation, reward, done, truncated, info = result
        else:
            observation, reward, done, info = result
            truncated = False

        # Extract health
        curr_player_health = info.get("health", self.full_hp)
        curr_opponent_health = info.get("enemy_health", self.full_hp)
        
        custom_reward, custom_done = self._calculate_reward(curr_player_health, curr_opponent_health)
        if custom_done:
            done = custom_done

        # Process observation
        processed_frame = self._process_frame(observation)
        self.frame_stack.append(processed_frame)
        stacked_obs = self._stack_observation()

        self.episode_steps += 1
        if self.episode_steps >= self.max_episode_steps:
            truncated = True

        return stacked_obs, custom_reward, done, truncated, info


class DecisionTransformer(nn.Module):
    """Minimal Decision Transformer"""

    def __init__(self, observation_shape, action_dim, hidden_size=256, n_layer=4, n_head=4, max_ep_len=2000):
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
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
        )

        cnn_output_size = 128 * 4 * 4

        # Embeddings
        self.state_encoder = nn.Linear(cnn_output_size, hidden_size)
        self.action_encoder = nn.Embedding(action_dim, hidden_size)
        self.return_encoder = nn.Linear(1, hidden_size)
        self.timestep_encoder = nn.Embedding(max_ep_len, hidden_size)

        # Transformer
        self.ln = nn.LayerNorm(hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=n_head, dim_feedforward=4 * hidden_size,
            dropout=0.1, activation="gelu", batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layer)
        self.action_head = nn.Linear(hidden_size, action_dim)

    def forward(self, states, actions, returns_to_go, timesteps):
        batch_size, seq_len = states.shape[:2]

        # Encode states
        states_flat = states.view(-1, *self.observation_shape)
        state_features = self.cnn_encoder(states_flat)
        state_embeddings = self.state_encoder(state_features).view(batch_size, seq_len, self.hidden_size)

        # Encode other inputs
        action_embeddings = self.action_encoder(actions)
        rtg_embeddings = self.return_encoder(returns_to_go.unsqueeze(-1))
        timesteps_clamped = torch.clamp(timesteps, 0, self.max_ep_len - 1)
        timestep_embeddings = self.timestep_encoder(timesteps_clamped)

        # Stack embeddings
        stacked_inputs = torch.stack([rtg_embeddings, state_embeddings, action_embeddings], dim=2)
        stacked_inputs = stacked_inputs.view(batch_size, 3 * seq_len, self.hidden_size)

        # Apply transformer
        stacked_inputs = self.ln(stacked_inputs)
        transformer_outputs = self.transformer(stacked_inputs)

        # Extract action predictions
        action_outputs = transformer_outputs[:, 2::3]
        return self.action_head(action_outputs)

    def save(self, path):
        """Save model"""
        torch.save(self.state_dict(), path)
        print(f"💾 Saved: {path}")


class TrajectoryDataset(Dataset):
    """Minimal dataset"""

    def __init__(self, trajectories, context_length=30):
        self.trajectories = [t for t in trajectories if len(t["rewards"]) > 10 and len(t["states"]) > 0]
        self.context_length = context_length

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        traj = self.trajectories[idx]
        states = np.array(traj["states"])
        actions = np.array(traj["actions"])
        rewards = np.array(traj["rewards"])

        # Ensure same length
        min_length = min(len(states), len(actions))
        states = states[:min_length]
        actions = actions[:min_length]
        rewards = rewards[:min_length]

        # Calculate returns-to-go
        returns_to_go = np.zeros_like(rewards, dtype=np.float32)
        running_return = 0
        for i in reversed(range(len(rewards))):
            running_return = rewards[i] + 0.99 * running_return
            returns_to_go[i] = running_return

        timesteps = np.arange(len(states))

        # Handle sequence length
        if len(states) <= self.context_length:
            pad_length = self.context_length - len(states)
            if pad_length > 0:
                pad_states = np.zeros((pad_length, *states.shape[1:]), dtype=states.dtype)
                states = np.concatenate([pad_states, states])
                actions = np.concatenate([np.zeros(pad_length, dtype=actions.dtype), actions])
                returns_to_go = np.concatenate([np.zeros(pad_length, dtype=np.float32), returns_to_go])
                timesteps = np.concatenate([np.zeros(pad_length, dtype=timesteps.dtype), timesteps])
        else:
            start_idx = np.random.randint(0, len(states) - self.context_length + 1)
            states = states[start_idx:start_idx + self.context_length]
            actions = actions[start_idx:start_idx + self.context_length]
            returns_to_go = returns_to_go[start_idx:start_idx + self.context_length]
            timesteps = timesteps[start_idx:start_idx + self.context_length]

        return {
            "states": torch.from_numpy(states).float() / 255.0,
            "actions": torch.from_numpy(actions).long(),
            "returns_to_go": torch.from_numpy(returns_to_go).float(),
            "timesteps": torch.from_numpy(timesteps).long(),
        }


def collect_trajectories(env, num_episodes=100):
    """Collect trajectories"""
    print(f"🎮 Collecting {num_episodes} episodes...")
    trajectories = []

    for episode in range(num_episodes):
        trajectory = {"states": [], "actions": [], "rewards": []}
        obs, _ = env.reset()
        trajectory["states"].append(obs.copy())

        done = False
        truncated = False

        while not done and not truncated:
            action = env.action_space.sample()
            obs, reward, done, truncated, _ = env.step(action)
            
            trajectory["actions"].append(action)
            trajectory["rewards"].append(reward)
            
            if not done and not truncated:
                trajectory["states"].append(obs.copy())

        if len(trajectory["actions"]) > 10:
            trajectories.append(trajectory)

    print(f"✅ Collected {len(trajectories)} trajectories")
    return trajectories


def train_decision_transformer(model, trajectories, epochs=100, batch_size=32, lr=1e-4, device="cuda", context_length=30, save_dir=None):
    """Training with 300k timestep checkpoints"""
    dataset = TrajectoryDataset(trajectories, context_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = AdamW(model.parameters(), lr=lr)
    
    model.to(device)
    model.train()

    print(f"🚀 Training {epochs} epochs, {len(dataset)} samples")
    if save_dir:
        print(f"💾 Checkpoints every 300,000 timesteps")

    timesteps_processed = 0
    last_checkpoint = 0

    for epoch in range(epochs):
        total_loss = 0
        num_batches = 0

        for batch in dataloader:
            states = batch["states"].to(device)
            actions = batch["actions"].to(device)
            returns_to_go = batch["returns_to_go"].to(device)
            timesteps_batch = batch["timesteps"].to(device)

            # Count timesteps processed
            batch_timesteps = states.shape[0] * states.shape[1]
            timesteps_processed += batch_timesteps

            optimizer.zero_grad()
            action_logits = model(states, actions, returns_to_go, timesteps_batch)

            if action_logits.shape[1] > 1:
                targets = actions[:, 1:]
                predictions = action_logits[:, :-1]
                loss = F.cross_entropy(predictions.reshape(-1, predictions.shape[-1]), targets.reshape(-1))
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                num_batches += 1

                # Save checkpoint every 300k timesteps
                if save_dir and timesteps_processed - last_checkpoint >= 300000:
                    checkpoint_dir = os.path.join(save_dir, "checkpoints")
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{timesteps_processed}_timesteps.pth")
                    torch.save(model.state_dict(), checkpoint_path)
                    print(f"💾 CHECKPOINT: {timesteps_processed:,} timesteps")
                    last_checkpoint = timesteps_processed

        if epoch % 10 == 0:
            avg_loss = total_loss / max(num_batches, 1)
            print(f"   Epoch {epoch}/{epochs}, Loss: {avg_loss:.4f}, Timesteps: {timesteps_processed:,}")

    return model