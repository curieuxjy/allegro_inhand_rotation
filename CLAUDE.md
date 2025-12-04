# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

In-hand object rotation for Allegro Hand V4 using Rapid Motor Adaptation (RMA). Two-stage training pipeline:
- **Stage 1 (PPO)**: Train policy with privileged information (object state, contact forces)
- **Stage 2 (ProprioAdapt)**: Train adaptation module using only proprioceptive history

Based on HORA research (https://haozhi.io/hora/) with modifications for Allegro Hand V4.

## Environment Setup

Two separate conda environments required due to Python version conflicts:

```bash
# Training (Isaac Gym requires Python 3.6-3.8)
conda create -n hora python=3.8
pip install -r hora_requirements.txt

# Deployment (ROS2 Humble requires Python 3.10+)
conda create -n allegro python=3.10
pip install -r allegro_requirements.txt
```

## Common Commands

### Training Pipeline

```bash
# Stage 1: Train with privileged info
scripts/train_s1.sh

# Stage 2: Train adaptation module (uses stage1 checkpoint)
scripts/train_s2.sh

# Resume interrupted training
scripts/resume_s1.sh
scripts/resume_s2.sh
```

### Evaluation & Visualization

```bash
# Evaluate checkpoints (headless)
scripts/eval_s1.sh
scripts/eval_s2.sh

# Visualize in Isaac Gym
scripts/vis_s1.sh
scripts/vis_s2.sh
```

### Grasp Generation

```bash
scripts/gen_grasp.sh GPU_ID  # Generate grasp poses for training
```

### Hardware Deployment

```bash
scripts/deploy.sh CACHE_NAME  # Deploy to real Allegro Hand via ROS2
```

### Direct Python Usage

```bash
# Training with overrides
python train.py task=RightCorlAllegroHandHora train.algo=PPO headless=True

# Testing
python train.py task=RightCorlAllegroHandHora test=True checkpoint=path/to/model.pth
```

## Architecture

### Training Algorithms (`hora/algo/`)
- `ppo/ppo.py` - Stage 1: PPO with privileged information encoder
- `padapt/padapt.py` - Stage 2: Proprioceptive adaptation (freezes Stage 1 policy, trains only adaptation module)
- `models/models.py` - ActorCritic network with privileged info MLP and adaptation module

### Environments (`hora/tasks/`)
- `allegro_hand_hora.py` - Main in-hand rotation environment
- `allegro_hand_grasp.py` - Grasp pose generation environment
- Task map in `__init__.py` maps config names to classes (multiple configs can share same class)

### Configuration (`configs/`)
Uses Hydra for hierarchical config:
- `config.yaml` - Entry point (device, seed, wandb settings)
- `task/*.yaml` - Environment params (rewards, randomization, URDF paths)
- `train/*.yaml` - Training params (PPO hyperparameters, network architecture)

Task naming: `{Left|Right}{Corl|Tip|}AllegroHand{Hora|Grasp}`
- Hora = rotation task, Grasp = grasp generation
- Corl/Tip/blank = different fingertip geometries

### Output Structure

```
outputs/{TASK}/{CACHE}/
├── stage1_nn/          # Stage 1 checkpoints
│   ├── best.pth
│   └── ep_*.pth
├── stage1_tb/          # TensorBoard logs
├── stage2_nn/          # Stage 2 checkpoints
├── stage2_tb/
└── config_*.yaml       # Saved config
```

## Key Config Options

```yaml
# Training stage selection
train.algo: PPO           # Stage 1
train.algo: ProprioAdapt  # Stage 2

# Privileged info
train.ppo.priv_info: True
train.ppo.proprio_adapt: False  # Stage 1
train.ppo.proprio_adapt: True   # Stage 2

# Resume vs fresh start for Stage 2
train.ppo.resume: True   # Resume training state
train.ppo.resume: False  # Load weights only, reset epoch counter
```

## URDF Files

```
assets/allegro/
├── allegro_right.urdf      # Standard V4 right hand
├── allegro_left.urdf       # Standard V4 left hand
├── allegro_corl_right.urdf # CORL fingertip geometry
├── allegro_corl_left.urdf
├── allegro_tip_right.urdf  # Alternative tip geometry
└── allegro_tip_left.urdf
```
