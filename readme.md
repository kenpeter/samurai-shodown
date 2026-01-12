# Samurai Shodown AI Agent with JEPA + PRIME

> Advanced Reinforcement Learning agent that masters the classic fighting game Samurai Shodown using Joint-Embedding Predictive Architecture (JEPA), Process Reward Incentive Modeling (PRIME), and LSTM-based opponent prediction.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.7.0-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

<div align="center">
  <img src="images/samurai-shodown-characters.webp" alt="Samurai Shodown Characters" width="600"/>
</div>

## Overview

This project implements a state-of-the-art reinforcement learning agent that learns to play Samurai Shodown through self-play. By combining **JEPA (Joint-Embedding Predictive Architecture)** and **PRIME (Process Reward Incentive Modeling)** with **Proximal Policy Optimization (PPO)**, the agent develops sophisticated fighting strategies, predicts opponent behavior, and learns optimal combat tactics through strategic reward modeling.

### Key Achievements

- **Self-Learning AI**: Agent learns complex fighting game mechanics purely from visual input
- **Hybrid JEPA + PRIME Architecture**: Combined predictive learning with process reward modeling
- **Binary Outcome Prediction**: 60-80% accuracy on opponent behavior anticipation
- **Strategic Response Planning**: Confidence-weighted action selection for adaptive gameplay
- **LSTM Temporal Modeling**: 2-layer bidirectional sequence processing for pattern recognition
- **Robust Training**: Enhanced reward shaping with prediction accuracy bonuses

## Gameplay Demonstration

<div align="center">
  <img src="images/gameplay-screenshot.jpg" alt="AI Agent Playing Samurai Shodown" width="700"/>
  <p><em>The AI agent in action - demonstrating learned combat strategies and timing</em></p>
</div>

## Technical Highlights

### Architecture

The system employs a sophisticated JEPA + PRIME pipeline:

```
Input Observation: (batch, 18, 180, 126) [6 frames × 3 channels]
         ↓
   CNN Feature Extractor: (batch, 512)
         ↓
Visual Context Encoder: (batch, 128)
  Game State Encoder: (batch, 32)  [health, distance, etc.]
Binary Outcome History: (batch, 8)  [previous predictions]
         ↓
LSTM Temporal Encoder: (batch, seq_len, 256) [2-layer bidirectional]
         ↓
Binary Outcome Predictors (4 parallel heads):
  • will_opponent_attack
  • will_opponent_take_damage
  • will_player_take_damage
  • will_round_end_soon
         ↓
Response Planner: Strategic action selection
         ↓
PRIME Process Reward Model: Reward shaping for learning
         ↓
Final Action + Value Estimation
```

### Core Technologies

- **Deep Reinforcement Learning**: PPO algorithm with custom policy networks
- **JEPA (Joint-Embedding Predictive Architecture)**: Binary outcome prediction for fighting game strategy
- **PRIME (Process Reward Incentive Modeling)**: Process reward modeling for strategic learning
- **LSTM Temporal Encoder**: 2-layer bidirectional LSTM for temporal pattern recognition
- **Binary Prediction**: 4 simple binary outcomes (60-80% accuracy target)
- **Response Planning**: Strategic action selection based on predicted outcomes
- **Enhanced Reward Shaping**: Combined base rewards with prediction accuracy bonuses

### Features

- **Binary Outcome Prediction**: 4 high-accuracy binary predictions instead of complex movement classification
- **LSTM Temporal Modeling**: 2-layer bidirectional LSTM for opponent pattern recognition
- **Strategic Response Planning**: Action selection based on predicted outcomes and confidence
- **PRIME Process Rewards**: Advanced reward shaping for strategic behavior learning
- **Prediction Confidence Estimation**: Per-outcome confidence scores with global correlation analysis
- **Comprehensive Metrics**: Win rate, prediction accuracy, damage efficiency, response success tracking
- **Flexible Training**: Configurable JEPA/PRIME parameters and training modes
- **Evaluation Suite**: Detailed performance analysis with rendering and comparison modes

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
├── train.py              # JEPA + PRIME training with PPO
├── eval.py               # Performance evaluation and analysis
├── play.py               # Interactive trained agent playback
├── wrapper.py            # JEPA wrapper with binary prediction
├── requirements.txt      # Python dependencies
├── rom.md               # ROM setup instructions
├── images/              # Screenshots and visual assets
└── trained_models/      # Model checkpoints
```

## Technical Stack

| Component | Technology |
|-----------|------------|
| **RL Framework** | Stable Baselines3 (PPO) |
| **Deep Learning** | PyTorch 2.7.0 |
| **Game Environment** | Stable-Retro (Gymnasium) |
| **Computer Vision** | OpenCV, Custom CNN |
| **Sequence Modeling** | 2-Layer Bidirectional LSTM |
| **Architecture** | JEPA + PRIME Hybrid |
| **Prediction** | Binary Outcome Classification |
| **Monitoring** | TensorBoard, Custom Metrics |

## Key Innovations

### 1. JEPA + PRIME Hybrid Architecture
Combines **Joint-Embedding Predictive Architecture (JEPA)** for opponent behavior prediction with **Process Reward Incentive Modeling (PRIME)** for strategic learning. This hybrid approach enables the agent to anticipate opponent actions while learning optimal process-level strategies.

### 2. Binary Outcome Prediction
Instead of complex movement classification, the system predicts 4 simple binary outcomes with 60-80% target accuracy:
- **will_opponent_attack**: Anticipates opponent offensive actions
- **will_opponent_take_damage**: Predicts successful agent attacks
- **will_player_take_damage**: Defensive awareness prediction
- **will_round_end_soon**: Strategic round-end planning

### 3. LSTM Temporal Pattern Recognition
2-layer bidirectional LSTM processes sequences of visual features, game states, and binary outcome history to recognize opponent behavioral patterns and temporal dynamics.

### 4. Strategic Response Planning
Confidence-weighted response planner selects actions based on predicted outcomes and their confidence scores, enabling adaptive counter-strategies and proactive gameplay.

### 5. Enhanced Reward Shaping with PRIME
Multi-component reward system combining:
- Base health differential rewards
- JEPA prediction accuracy bonuses
- Strategic response effectiveness rewards
- Anti-spam penalties for diverse action selection

## Performance Metrics

The agent tracks comprehensive performance metrics across multiple dimensions:

**Combat Performance:**
- **Win Rate**: Percentage of rounds won
- **Damage Efficiency**: Damage dealt vs. damage received ratio
- **Win Streak**: Best and current consecutive win streaks
- **Average Episode Length**: Tactical round duration

**JEPA Prediction Metrics:**
- **Binary Prediction Accuracy**: Overall accuracy across 4 outcome types
- **Attack Prediction Accuracy**: Anticipating opponent attacks
- **Damage Prediction Accuracy**: Predicting damage events
- **Round End Prediction Accuracy**: Strategic timing awareness
- **Average Prediction Confidence**: Model certainty levels

**Strategic Response Metrics:**
- **Successful Responses**: Effective actions following predictions
- **Response Success Rate**: Ratio of successful to total planned responses
- **Counter-Attack Effectiveness**: Damage following predicted opponent attacks

## Future Work

- [ ] Multi-agent self-play for advanced strategy evolution
- [ ] Transfer learning to other fighting games
- [ ] Human-AI cooperative training
- [ ] Real-time strategy adaptation
- [ ] Tournament-level performance optimization

## Research & Learning

This project demonstrates proficiency in:

- **Deep Reinforcement Learning**: PPO algorithm, policy gradients, value functions, custom reward shaping
- **Predictive Learning**: JEPA (Joint-Embedding Predictive Architecture) for opponent behavior modeling
- **Process Reward Modeling**: PRIME technique for strategic learning and process-level optimization
- **Sequence Modeling**: Bidirectional LSTM for temporal pattern recognition and behavioral analysis
- **Binary Classification**: High-accuracy outcome prediction with confidence estimation
- **Neural Architecture Design**: Custom CNNs, LSTM networks, multi-head predictors
- **Strategic AI**: Response planning, confidence-weighted decision making, adaptive strategies
- **Training Optimization**: Hyperparameter tuning, stability techniques, curriculum learning
- **Software Engineering**: Clean code, modular design, reproducibility, PyTorch optimization

## License

MIT License - See LICENSE file for details

## Acknowledgments

- Stable-Retro for game environment framework (Gymnasium compatibility)
- Stable Baselines3 for PPO RL implementation
- JEPA paper authors (Yann LeCun et al.) for predictive architecture concepts
- PRIME research for process reward modeling techniques

---

**Built with** Python • PyTorch • Reinforcement Learning • Computer Vision

*Demonstrating advanced machine learning engineering and research capabilities*
