#!/bin/bash
# Deploy single hand with checkpoint
# Interactive script that prompts for deployment parameters

# Source ROS2 environment
source /opt/ros/humble/setup.bash

echo "=== Single Hand Deployment ==="

read -p "TASK [LeftAllegroHandHora]: " TASK
read -p "CACHE [test]: " CACHE
read -p "EXTRA_ARGS (optional): " EXTRA_ARGS

TASK=${TASK:-LeftAllegroHandHora}
CACHE=${CACHE:-test}

CHECKPOINT="outputs/${TASK}/${CACHE}/stage2_nn/best.pth"

echo ""
echo "TASK: ${TASK}, CACHE: ${CACHE}"
echo "CHECKPOINT: ${CHECKPOINT}"
echo "EXTRA_ARGS: ${EXTRA_ARGS}"
echo ""

# Verify checkpoint exists
if [ ! -f "${CHECKPOINT}" ]; then
    echo "❌ Error: Checkpoint not found: ${CHECKPOINT}"
    exit 1
fi

echo "✓ Checkpoint found"
echo ""
echo "Starting deployment..."

python run.py checkpoint="${CHECKPOINT}" ${EXTRA_ARGS}
