#!/bin/bash
# Stage 2 Resume Training Script
# Resumes ProprioAdapt training from a checkpoint

echo "=== Stage 2 Resume Training ==="

read -p "GPUS [0]: " GPUS
read -p "TASK [RightCorlAllegroHandHora]: " TASK
read -p "SEED [42]: " SEED
read -p "CACHE [test]: " CACHE
read -p "CHECKPOINT (e.g., last.pth or best.pth) [last.pth]: " CHECKPOINT
read -p "EXTRA_ARGS (optional): " EXTRA_ARGS

GPUS=${GPUS:-0}
TASK=${TASK:-RightCorlAllegroHandHora}
SEED=${SEED:-42}
CACHE=${CACHE:-test}
CHECKPOINT=${CHECKPOINT:-last.pth}

CHECKPOINT_PATH="outputs/${TASK}/${CACHE}/stage2_nn/${CHECKPOINT}"

if [ ! -f "${CHECKPOINT_PATH}" ]; then
    echo "Error: Checkpoint not found at ${CHECKPOINT_PATH}"
    exit 1
fi

echo ""
echo "GPUS: ${GPUS}, TASK: ${TASK}, SEED: ${SEED}, CACHE: ${CACHE}"
echo "CHECKPOINT_PATH: ${CHECKPOINT_PATH}"
echo "EXTRA_ARGS: ${EXTRA_ARGS}"

CUDA_VISIBLE_DEVICES=${GPUS} \
python train.py task=${TASK} headless=True seed=${SEED} \
task.env.numEnvs=20000 \
task.env.forceScale=2 task.env.randomForceProbScalar=0.25 \
train.algo=ProprioAdapt \
train.ppo.priv_info=True train.ppo.proprio_adapt=True \
train.ppo.output_name=${TASK}/"${CACHE}" \
checkpoint=outputs/${TASK}/"${CACHE}"/stage1_nn/best.pth \
train.load_path=${CHECKPOINT_PATH} \
${EXTRA_ARGS}
