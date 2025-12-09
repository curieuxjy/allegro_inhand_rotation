# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository implements in-hand object rotation for **Allegro Hand V4** (Right and Left) using Rapid Motor Adaptation (RMA). It combines Isaac Gym simulation for training with ROS2-based hardware deployment.

**Key Components:**
- **Training**: Two-stage RL training (teacher policy → student policy)
- **Simulation**: Isaac Gym for parallel environment training
- **Deployment**: ROS2 interface for real Allegro Hand hardware
- **Configuration**: Hydra-based hierarchical config system

## Development Setup

**Two Conda Environments Required:**

```bash
# Training environment (Isaac Gym requires Python 3.6-3.8)
conda create -n hora python=3.8
conda activate hora
pip install -r hora_requirements.txt

# Deployment environment (ROS2 requires Python 3.10+)
conda create -n allegro python=3.10
conda activate allegro
pip install -r allegro_requirements.txt
```

**Important**: Isaac Gym and ROS2 have incompatible Python requirements. Always use separate environments.

## Core Commands

All training/evaluation scripts are **interactive** and prompt for parameters (with defaults in brackets).

### Training

**Stage 1: Teacher Policy (with privileged info)**
```bash
./scripts/train_s1.sh
# Prompts: GPUS [0], TASK [RightCorlAllegroHandHora], SEED [42], CACHE [test], EXTRA_ARGS
```

**Stage 2: Student Policy (proprioceptive only)**
```bash
./scripts/train_s2.sh
# Prompts: GPUS [0], TASK [RightCorlAllegroHandHora], SEED [42], CACHE [test], EXTRA_ARGS
# Requires Stage 1 checkpoint at: outputs/{TASK}/{CACHE}/stage1_nn/best.pth
```

**Switch hands by entering task when prompted:**
- Right: `RightCorlAllegroHandHora`, `RightAllegroHandHora`, `RightTipAllegroHandHora`
- Left: `LeftCorlAllegroHandHora`, `LeftAllegroHandHora`, `LeftTipAllegroHandHora`

### Evaluation & Visualization

```bash
./scripts/eval_s1.sh  # Headless evaluation, 10240 envs
./scripts/eval_s2.sh  # Stage 2 evaluation
./scripts/vis_s1.sh   # Visual inspection, 64 envs
./scripts/vis_s2.sh   # Stage 2 visualization
```

### Grasp Generation

```bash
./scripts/gen_grasp.sh GPU_ID           # Single GPU
./scripts/gen_grasp_multigpus.sh 0 1 2  # Multiple GPUs
```

### Deployment (Real Hardware)

```bash
# Terminal 1: Start Allegro Hand ROS2 node
sudo ip link set can0 down
sudo ip link set can0 type can bitrate 1000000
sudo ip link set can0 up
ros2 launch allegro_hand_bringup allegro_hand.launch.py

# Terminal 2: Run deployment (interactive prompts)
./scripts/deploy_one_right.sh  # Single right hand
./scripts/deploy_one_left.sh   # Single left hand
./scripts/deploy_two_hands.sh  # Dual hand deployment
```

## Architecture

### Configuration System (Hydra)

**Hierarchy** (RightCorlAllegroHandHora is the root base config):
```
config.yaml (entry point, default task=RightCorlAllegroHandHora)
├── task/*.yaml (environment, rewards, URDF paths)
│   └── RightCorlAllegroHandHora.yaml (BASE)
│       ├── RightAllegroHandHora.yaml
│       ├── RightTipAllegroHandHora.yaml
│       ├── LeftCorlAllegroHandHora.yaml
│       │   ├── LeftAllegroHandHora.yaml
│       │   └── LeftTipAllegroHandHora.yaml
│       └── *Grasp.yaml variants (for grasp generation)
└── train/*.yaml (PPO params, mirrors task structure)
```

**Task Variants:**
- **Corl**: Original HORA fingertip geometry
- **Standard** (RightAllegroHandHora): Allegro V4 sphere fingertips
- **Tip**: Alternative fingertip configuration

**Task Name Mapping** (`hora/tasks/__init__.py`):
- All variants map to same base classes (`AllegroHandHora`, `AllegroHandGrasp`)
- Differentiation happens via config files (URDF paths, grasp caches)
- **Critical**: Each task config MUST define `name` field for proper wandb logging

**Override configs:**
```bash
python train.py task=RightAllegroHandHora \
    task.env.numEnvs=1024 \
    train.ppo.learning_rate=1e-4 \
    wandb.enabled=True
```

### Training Pipeline

**Stage 1 (PPO with privileged info):**
- Algorithm: `train.algo=PPO`
- Input: Proprioceptive + privileged observations (object dynamics, forces)
- Output: `outputs/{task}/{run_name}/stage1_nn/best.pth`
- Network: Actor-Critic with separate privileged info encoder

**Stage 2 (Proprioceptive Adaptation):**
- Algorithm: `train.algo=ProprioAdapt`
- Input: Only proprioceptive history (joint states)
- Requires: Stage 1 checkpoint via `checkpoint=` parameter
- Output: `outputs/{task}/{run_name}/stage2_nn/best.pth`
- Enables sim-to-real transfer

### Key Modules

**`hora/tasks/`**
- `allegro_hand_hora.py`: Main in-hand rotation task
- `allegro_hand_grasp.py`: Grasp pose generation task

**`hora/algo/ppo/`**
- `ppo.py`: PPO training loop, value/policy optimization
- `experience.py`: Replay buffer for batch collection

**`hora/algo/padapt/`**
- `padapt.py`: Proprioceptive adaptation module (Stage 2)

**`hora/algo/models/`**
- `models.py`: Actor-Critic architecture with optional privileged encoder
- `running_mean_std.py`: Observation normalization

**`hora/algo/deploy/`**
- `deploy_ros2_right.py`: Right hand deployment (RightHardwarePlayer)
- `deploy_ros2_left.py`: Left hand deployment (LeftHardwarePlayer)
- `deploy_ros2_two_hands.py`: Dual hand deployment
- `robots/allegro_ros2.py`: ROS2 interface wrapper

**`run.py`**: Unified deployment entry point that auto-detects hand side from checkpoint path

### URDF Files

**Location**: `assets/allegro/`
- `allegro_right.urdf` / `allegro_left.urdf`: Allegro Hand V4 (sphere fingertips)
- `allegro_corl_right.urdf` / `allegro_corl_left.urdf`: Original HORA fingertip geometry
- `allegro_tip_right.urdf` / `allegro_tip_left.urdf`: Alternative fingertip variant

**Key differences:**
- V4 standard: sphere fingertips (radius 0.012m)
- Corl: custom fingertip geometry with different contact properties (matches original HORA paper)

### Batch Size Configuration

**Critical**: `batch_size = num_actors × horizon_length` must be divisible by `minibatch_size`

**Default (full training):**
- `task.env.numEnvs=24576`
- `train.ppo.horizon_length=8`
- `train.ppo.minibatch_size=32768`
- Batch size: 24576 × 8 = 196608

**Debug mode (low memory):**
- `task.env.numEnvs=4`
- `train.ppo.horizon_length=8`
- `train.ppo.minibatch_size=32`
- Batch size: 4 × 8 = 32

### Wandb Integration

**Configuration** (`configs/config.yaml`):
```yaml
wandb:
  enabled: True/False
  entity: 'your-username'
  project: 'project-name'
```

**Run naming**: `{stage}_{task_name}_{timestamp}`
- Example: `stage1_RightAllegroHandHora_12201530`
- Ensure task configs define `name` field to avoid inheriting parent names

## File Structure

```
├── train.py                   # Main training entry point
├── run.py                     # Unified deployment entry point
├── gen_grasp.py              # Grasp pose generation
├── compare_hands.py          # URDF comparison visualization
├── allegro_right_left.py     # Right/Left hand visualization
├── configs/                  # Hydra configuration files
│   ├── config.yaml          # Main config (device, wandb, defaults)
│   ├── task/*.yaml          # Task-specific configs
│   └── train/*.yaml         # Training hyperparameters
├── scripts/                 # Interactive bash scripts
│   ├── train_s1.sh, train_s2.sh    # Training
│   ├── eval_s1.sh, eval_s2.sh      # Evaluation
│   ├── vis_s1.sh, vis_s2.sh        # Visualization
│   ├── gen_grasp.sh                 # Grasp generation
│   └── deploy_one_right.sh, deploy_one_left.sh, deploy_two_hands.sh
├── hora/                    # Core implementation
│   ├── tasks/              # IsaacGym task environments
│   └── algo/               # RL algorithms and models
│       ├── ppo/           # PPO implementation
│       ├── padapt/        # Proprioceptive adaptation
│       ├── models/        # Neural network architectures
│       └── deploy/        # ROS2 deployment code
├── assets/allegro/        # URDF files and meshes
├── cache/                 # Generated grasp poses (.npy)
└── outputs/              # Training outputs (checkpoints, logs)
```

## Common Workflows

### Starting a New Training Run

1. Generate grasp poses (if not already done):
   ```bash
   ./scripts/gen_grasp.sh 0
   ```

2. Train Stage 1 (interactive, press Enter for defaults or type values):
   ```bash
   ./scripts/train_s1.sh
   # GPUS [0]:
   # TASK [RightCorlAllegroHandHora]:
   # SEED [42]:
   # CACHE [test]: my_experiment
   ```

3. Train Stage 2 (after Stage 1 completes):
   ```bash
   ./scripts/train_s2.sh
   # Use same TASK and CACHE as Stage 1
   ```

4. Evaluate:
   ```bash
   ./scripts/eval_s2.sh
   # Enter same TASK and CACHE
   ```

### Switching Between Hands

When prompted for TASK, enter the desired variant:
- Right hand: `RightCorlAllegroHandHora` (default)
- Left hand: `LeftCorlAllegroHandHora`

Output paths automatically adjust: `outputs/{TASK}/{CACHE}/`

### Dual Hand CAN Bus Configuration

**Important**: For dual-hand setups, CAN bus mapping must match physical hardware connections.

In `allegro_hand_ros2/allegro_hand_bringup/config/v4/dual_hand/dual_hand.urdf.xacro`:
- Physical `can0` connection → `io_interface_descriptor="can:can0"`
- Physical `can1` connection → `io_interface_descriptor="can:can1"`

If mapping is wrong, commands will be sent to the wrong hand.

### Debugging Issues

**Assertion Error: batch_size % minibatch_size != 0**
- Ensure `num_actors × horizon_length` is divisible by `minibatch_size`
- For small `numEnvs`, reduce `minibatch_size` proportionally

**Wrong task name in wandb logs**
- Check that task config file has `name` field defined
- Task name comes from config, not from the config filename

**URDF mesh not loading**
- Verify mesh paths use correct format: `meshes/allegro/*.obj`
- Check that required .obj files exist in `assets/allegro/meshes/allegro/`

**Stage 2 fails to load checkpoint**
- Verify Stage 1 checkpoint exists at expected path
- Check `checkpoint=` parameter points to correct `.pth` file
- Ensure task name consistency between stages

**Isaac Gym libpython error**
```
ImportError: libpython3.8.so.1.0: cannot open shared object file
```
Fix: `export LD_LIBRARY_PATH=/path/to/conda/envs/hora/lib:$LD_LIBRARY_PATH`
