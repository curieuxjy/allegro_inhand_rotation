#!/bin/bash
# Weight Visualization Script
# Visualizes network weights from a checkpoint as heatmaps

echo "=== Weight Visualization ==="

read -p "TASK [RightCorlAllegroHandHora]: " TASK
read -p "CACHE [test]: " CACHE
read -p "STAGE (1 or 2) [1]: " STAGE
read -p "CHECKPOINT (e.g., best.pth or last.pth) [best.pth]: " CHECKPOINT
read -p "OUTPUT_DIR [weight_visualizations]: " OUTPUT_DIR

TASK=${TASK:-RightCorlAllegroHandHora}
CACHE=${CACHE:-test}
STAGE=${STAGE:-1}
CHECKPOINT=${CHECKPOINT:-best.pth}
OUTPUT_DIR=${OUTPUT_DIR:-weight_visualizations}

CHECKPOINT_PATH="outputs/${TASK}/${CACHE}/stage${STAGE}_nn/${CHECKPOINT}"

if [ ! -f "${CHECKPOINT_PATH}" ]; then
    echo "Error: Checkpoint not found at ${CHECKPOINT_PATH}"
    exit 1
fi

echo ""
echo "TASK: ${TASK}, CACHE: ${CACHE}, STAGE: ${STAGE}"
echo "CHECKPOINT_PATH: ${CHECKPOINT_PATH}"
echo "OUTPUT_DIR: ${OUTPUT_DIR}"

python visualize_weights.py --checkpoint ${CHECKPOINT_PATH} --output_dir ${OUTPUT_DIR}
