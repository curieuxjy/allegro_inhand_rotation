#!/bin/bash
# Stage 1 Visualization Script
# Visualizes PPO with privileged information

echo "=== Stage 1 Visualization ==="

read -p "TASK [RightCorlAllegroHandHora]: " TASK
read -p "CACHE [test]: " CACHE
read -p "EXTRA_ARGS (optional): " EXTRA_ARGS

TASK=${TASK:-RightCorlAllegroHandHora}
CACHE=${CACHE:-test}

echo ""
echo "TASK: ${TASK}, CACHE: ${CACHE}"
echo "EXTRA_ARGS: ${EXTRA_ARGS}"

python train.py task=${TASK} headless=False pipeline=gpu \
task.env.numEnvs=64 test=True \
train.algo=PPO \
task.env.randomization.randomizeMass=False \
task.env.randomization.randomizeCOM=False \
task.env.randomization.randomizeFriction=False \
task.env.randomization.randomizePDGains=False \
task.env.randomization.randomizeScale=True \
train.ppo.priv_info=True \
train.ppo.output_name=${TASK}/"${CACHE}" \
checkpoint=outputs/${TASK}/"${CACHE}"/stage1_nn/best.pth \
wandb.enabled=False \
${EXTRA_ARGS}
