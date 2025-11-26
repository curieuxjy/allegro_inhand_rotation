#!/bin/bash
# Deploy two hands with hardcoded CES demo checkpoints

# Source ROS2 environment
source /opt/ros/humble/setup.bash

echo "=== Two Hands Deployment (CES) ==="

TASK_RIGHT="RightAllegroHandHora"
CACHE_RIGHT="public"
TASK_LEFT="LeftTipAllegroHandHora"
CACHE_LEFT="left_1121"

CHECKPOINT_RIGHT="outputs/${TASK_RIGHT}/${CACHE_RIGHT}/stage2_nn/best.pth"
CHECKPOINT_LEFT="outputs/${TASK_LEFT}/${CACHE_LEFT}/stage2_nn/best.pth"

echo ""
echo "Right: TASK=${TASK_RIGHT}, CACHE=${CACHE_RIGHT}"
echo "Left:  TASK=${TASK_LEFT}, CACHE=${CACHE_LEFT}"
echo ""
echo "Checkpoint paths:"
echo "  Right: ${CHECKPOINT_RIGHT}"
echo "  Left:  ${CHECKPOINT_LEFT}"
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

python -m hora.algo.deploy.deploy_ros2_two_hands \
    --checkpoint-right="${CHECKPOINT_RIGHT}" \
    --checkpoint-left="${CHECKPOINT_LEFT}" \
    "$@"
