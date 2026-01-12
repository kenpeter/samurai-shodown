# Samurai Shodown AI Agent with JEPA

> Advanced Reinforcement Learning agent that masters the classic fighting game Samurai Shodown using Joint-Embedding Predictive Architecture (JEPA) and modern deep learning techniques.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.7.0-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This project implements a state-of-the-art reinforcement learning agent that learns to play Samurai Shodown through self-play. By combining **JEPA (Joint-Embedding Predictive Architecture)** with **Proximal Policy Optimization (PPO)**, the agent develops sophisticated fighting strategies and temporal understanding of game mechanics.

### Key Achievements

- **Self-Learning AI**: Agent learns complex fighting game mechanics purely from visual input
- **Advanced Architecture**: Custom CNN-Transformer hybrid with temporal sequence modeling
- **Predictive Understanding**: JEPA enables the agent to anticipate opponent moves and game outcomes
- **Robust Training**: Anti-spam mechanisms and comprehensive reward shaping for stable learning

## Technical Highlights

### Architecture

The system employs a sophisticated multi-stage pipeline:

```
Input Observation: (batch, channels, height, width)
         ↓
   CNN Features: (batch, feature_dim)
         ↓
Visual History: (seq_len, feature_dim) → Stack → (1, seq_len, feature_dim)
  Game History: (seq_len, 8) → Stack → (1, seq_len, 8)
         ↓
Transformer Input: (1, seq_len, d_model) where d_model = feature_dim + 8
         ↓
Positional Encoding: Adds (1, seq_len, d_model)
         ↓
Transformer Output: (1, seq_len, d_model)
         ↓
Final Representation: (1, d_model) [last timestep]
         ↓
Predictions: (1, prediction_horizon) for each outcome
```

### Core Technologies

- **Deep Reinforcement Learning**: PPO algorithm with custom policy networks
- **JEPA (Joint-Embedding Predictive Architecture)**: Self-supervised learning for visual representation
- **Transformer Networks**: Temporal attention mechanism for sequence modeling
- **Frame Stacking**: Multi-frame observation history for motion understanding
- **Reward Shaping**: Carefully designed reward signals for effective learning

### Features

- **Temporal Modeling**: 8-frame history with transformer-based sequence processing
- **Predictive Learning**: Forward prediction of game state and outcomes
- **Comprehensive Metrics**: Win rate tracking, damage analysis, and action spam prevention
- **Flexible Training**: Configurable hyperparameters and training modes
- **Evaluation Suite**: Detailed performance analysis with rendering support

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended for training)
- Samurai Shodown ROM file (see `rom.md` for details)

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/samurai-shodown.git
cd samurai-shodown

# Install dependencies
pip install -r requirements.txt

# Set up Retro environment with the ROM
# Follow instructions in rom.md
```

## Usage

### Training

Train the agent with JEPA-enhanced features:

```bash
python train.py \
    --total-timesteps 5000000 \
    --n-steps 4096 \
    --batch-size 512 \
    --lr 0.00025 \
    --ent-coef 0.01 \
    --frame-stack 8
```

Resume from a checkpoint:

```bash
python train.py \
    --total-timesteps 5000000 \
    --n-steps 4096 \
    --batch-size 512 \
    --lr 0.00025 \
    --ent-coef 0.01 \
    --frame-stack 8 \
    --resume trained_models/ppo_jepa_7400000_steps.zip
```

### Evaluation

Evaluate with visual rendering:

```bash
python eval.py trained_models/ppo_jepa_7400000_steps.zip --episodes 5 --render
```

Quick evaluation without rendering:

```bash
python eval.py trained_models/ppo_jepa_7400000_steps.zip --episodes 10
```

Compare JEPA vs standard CNN:

```bash
# With JEPA features
python eval.py trained_models/ppo_jepa_7400000_steps.zip --episodes 10

# Without JEPA (standard CNN)
python eval.py trained_models/ppo_jepa_7400000_steps.zip --no-jepa --episodes 10
```

Save evaluation results:

```bash
python eval.py trained_models/ppo_jepa_7400000_steps.zip --episodes 20 --output results.json
```

### Interactive Play

Watch the trained agent play:

```bash
python play.py trained_models/ppo_jepa_7400000_steps.zip
```

## Project Structure

```
samurai-shodown/
├── train.py              # Main training script with JEPA integration
├── eval.py               # Evaluation and performance analysis
├── play.py               # Interactive agent playback
├── wrapper.py            # Custom environment wrapper with JEPA
├── requirements.txt      # Python dependencies
├── rom.md               # ROM setup instructions
└── trained_models/      # Saved model checkpoints
```

## Technical Stack

| Component | Technology |
|-----------|------------|
| **RL Framework** | Stable Baselines3 (PPO) |
| **Deep Learning** | PyTorch 2.7.0 |
| **Game Environment** | OpenAI Retro (Gym) |
| **Computer Vision** | OpenCV, Custom CNN |
| **Architecture** | Transformer, JEPA |
| **Monitoring** | TensorBoard |

## Key Innovations

### 1. JEPA Integration
Self-supervised learning approach that enables the agent to build rich internal representations of the game state without explicit labels.

### 2. Temporal Transformer
Attention-based mechanism processes sequences of game frames to understand temporal dynamics and opponent patterns.

### 3. Anti-Spam Mechanisms
Sophisticated reward shaping prevents the agent from developing repetitive, exploitative strategies.

### 4. Reward Engineering
Multi-component reward function balancing offensive actions, defensive positioning, and strategic gameplay.

## Performance Metrics

The agent tracks comprehensive performance metrics:

- **Win Rate**: Percentage of rounds won
- **Damage Efficiency**: Damage dealt vs. damage received
- **Action Diversity**: Prevention of move spamming
- **Combo Execution**: Multi-hit attack sequences
- **Defense Rate**: Successful blocking and evasion

## Future Work

- [ ] Multi-agent self-play for advanced strategy evolution
- [ ] Transfer learning to other fighting games
- [ ] Human-AI cooperative training
- [ ] Real-time strategy adaptation
- [ ] Tournament-level performance optimization

## Research & Learning

This project demonstrates proficiency in:

- **Deep Reinforcement Learning**: PPO, policy gradients, value functions
- **Self-Supervised Learning**: JEPA, contrastive learning
- **Neural Architecture Design**: CNNs, Transformers, attention mechanisms
- **Training Optimization**: Hyperparameter tuning, stability techniques
- **Software Engineering**: Clean code, modular design, reproducibility

## License

MIT License - See LICENSE file for details

## Acknowledgments

- OpenAI Retro for game environment framework
- Stable Baselines3 for RL algorithms
- JEPA paper authors for architectural inspiration

---

**Built with** Python • PyTorch • Reinforcement Learning • Computer Vision

*Demonstrating advanced machine learning engineering and research capabilities*
