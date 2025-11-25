#!/bin/bash
# Stage 2 Evaluation Script
# Evaluates ProprioAdapt with proprioceptive adaptation

echo "=== Stage 2 Evaluation ==="

read -p "GPUS [0]: " GPUS
read -p "TASK [RightCorlAllegroHandHora]: " TASK
read -p "CACHE [test]: " CACHE
read -p "EXTRA_ARGS (optional): " EXTRA_ARGS

GPUS=${GPUS:-0}
TASK=${TASK:-RightCorlAllegroHandHora}
CACHE=${CACHE:-test}

echo ""
echo "GPUS: ${GPUS}, TASK: ${TASK}, CACHE: ${CACHE}"
echo "EXTRA_ARGS: ${EXTRA_ARGS}"

C=outputs/${TASK}/"${CACHE}"/stage2_nn/best.pth
CUDA_VISIBLE_DEVICES=${GPUS} \
python train.py task=${TASK} headless=True \
task.env.numEnvs=10240 test=True task.on_evaluation=True \
task.env.object.type=cylinder_default \
train.algo=ProprioAdapt \
task.env.randomization.randomizeMass=True \
task.env.randomization.randomizeCOM=True \
task.env.randomization.randomizeFriction=True \
task.env.randomization.randomizePDGains=True \
task.env.randomization.randomizeScale=True \
task.env.randomization.jointNoiseScale=0.005 \
task.env.reset_height_threshold=0.6 \
task.env.forceScale=2 task.env.randomForceProbScalar=0.25 \
train.ppo.priv_info=True train.ppo.proprio_adapt=True \
train.ppo.output_name=${TASK}/"${CACHE}" \
checkpoint="${C}" \
wandb.enabled=False \
${EXTRA_ARGS}
