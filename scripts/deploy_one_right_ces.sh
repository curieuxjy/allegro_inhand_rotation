#!/bin/bash
# Deploy single right hand with hardcoded CES demo checkpoint

# Source ROS2 environment
source /opt/ros/humble/setup.bash

echo "=== Single Right Hand Deployment (CES) ==="

TASK="RightAllegroHandHora"
CACHE="public"

CHECKPOINT="outputs/${TASK}/${CACHE}/stage2_nn/best.pth"

echo ""
echo "TASK: ${TASK}, CACHE: ${CACHE}"
echo "CHECKPOINT: ${CHECKPOINT}"
echo ""

# Verify checkpoint exists
if [ ! -f "${CHECKPOINT}" ]; then
    echo "Error: Checkpoint not found: ${CHECKPOINT}"
    exit 1
fi

echo "Checkpoint found"
echo ""
echo "Starting deployment..."

python run.py checkpoint="${CHECKPOINT}" "$@"
