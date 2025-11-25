#!/bin/bash
# Stage 1 Training Script
# Trains PPO with privileged information

echo "=== Stage 1 Training ==="

read -p "GPUS [0]: " GPUS
read -p "TASK [RightCorlAllegroHandHora]: " TASK
read -p "SEED [42]: " SEED
read -p "CACHE [test]: " CACHE
read -p "EXTRA_ARGS (optional): " EXTRA_ARGS

GPUS=${GPUS:-0}
TASK=${TASK:-RightCorlAllegroHandHora}
SEED=${SEED:-42}
CACHE=${CACHE:-test}

echo ""
echo "GPUS: ${GPUS}, TASK: ${TASK}, SEED: ${SEED}, CACHE: ${CACHE}"
echo "EXTRA_ARGS: ${EXTRA_ARGS}"

CUDA_VISIBLE_DEVICES=${GPUS} \
python train.py task=${TASK} headless=True seed=${SEED} \
task.env.numEnvs=4 train.ppo.minibatch_size=32 \
task.env.forceScale=2 task.env.randomForceProbScalar=0.25 \
train.algo=PPO \
train.ppo.priv_info=True train.ppo.proprio_adapt=False \
train.ppo.output_name=${TASK}/"${CACHE}" \
${EXTRA_ARGS}
