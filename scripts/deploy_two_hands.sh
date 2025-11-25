#!/bin/bash
# Deploy two hands with hand-specific checkpoints
# Interactive script that prompts for deployment parameters

# Source ROS2 environment
source /opt/ros/humble/setup.bash

echo "=== Two Hands Deployment ==="

read -p "TASK_RIGHT [RightAllegroHandHora]: " TASK_RIGHT
read -p "CACHE_RIGHT [test]: " CACHE_RIGHT
read -p "TASK_LEFT [LeftAllegroHandHora]: " TASK_LEFT
read -p "CACHE_LEFT [test]: " CACHE_LEFT
read -p "EXTRA_ARGS (optional): " EXTRA_ARGS

TASK_RIGHT=${TASK_RIGHT:-RightAllegroHandHora}
CACHE_RIGHT=${CACHE_RIGHT:-test}
TASK_LEFT=${TASK_LEFT:-LeftAllegroHandHora}
CACHE_LEFT=${CACHE_LEFT:-test}

CHECKPOINT_RIGHT="outputs/${TASK_RIGHT}/${CACHE_RIGHT}/stage2_nn/best.pth"
CHECKPOINT_LEFT="outputs/${TASK_LEFT}/${CACHE_LEFT}/stage2_nn/best.pth"

echo ""
echo "Right: TASK=${TASK_RIGHT}, CACHE=${CACHE_RIGHT}"
echo "Left:  TASK=${TASK_LEFT}, CACHE=${CACHE_LEFT}"
echo ""
echo "Checkpoint paths:"
echo "  Right: ${CHECKPOINT_RIGHT}"
echo "  Left:  ${CHECKPOINT_LEFT}"
echo "EXTRA_ARGS: ${EXTRA_ARGS}"
echo ""

# Verify checkpoints exist
if [ ! -f "${CHECKPOINT_RIGHT}" ]; then
    echo "Error: Right hand checkpoint not found: ${CHECKPOINT_RIGHT}"
    exit 1
fi

if [ ! -f "${CHECKPOINT_LEFT}" ]; then
    echo "Error: Left hand checkpoint not found: ${CHECKPOINT_LEFT}"
    exit 1
fi

echo "Both checkpoints found"
echo ""
echo "Starting deployment..."

python run.py +checkpoint_right="${CHECKPOINT_RIGHT}" +checkpoint_left="${CHECKPOINT_LEFT}" ${EXTRA_ARGS}
