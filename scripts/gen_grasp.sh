#!/bin/bash
# CACHE can be some existing output folder, does not matter
# numEnvs=20000, headless=True, episodeLen=50 to save time
# see the object, check whether it's simple tennis ball or fancy balls
# pipeline need to be cpu to get the pairwise contact
# no custom PD because bug in CPU mode
# mass should be about 50g
# Usage: ./gen_grasp.sh GPU_NUMBER
# Automatically runs for scales 0.66 to 0.9 with 0.2 intervals

GPUS=$1

if [ -z "$GPUS" ]; then
    echo "Error: GPU number is required"
    echo "Usage: ./gen_grasp.sh GPU_NUMBER"
    exit 1
fi

# Loop through scales from 0.66 to 0.9 with 0.2 interval
for SCALE in 0.66 0.86; do
    echo "========================================="
    echo "Running with GPU ${GPUS} and SCALE ${SCALE}"
    echo "========================================="

    CUDA_VISIBLE_DEVICES=${GPUS} \
    python gen_grasp.py task=RightAllegroHandGrasp headless=True pipeline=cpu \
    task.env.numEnvs=20000 test=True \
    task.env.controller.controlFrequencyInv=8 task.env.episodeLength=50 \
    task.env.controller.torque_control=False task.env.genGrasps=True task.env.baseObjScale="${SCALE}" \
    task.env.object.type=simple_tennis_ball task.env.object.sampleProb=[1.0] \
    task.env.randomization.randomizeMass=True task.env.randomization.randomizeMassLower=0.05 task.env.randomization.randomizeMassUpper=0.051 \
    task.env.randomization.randomizeCOM=False \
    task.env.randomization.randomizeFriction=False \
    task.env.randomization.randomizePDGains=False \
    task.env.randomization.randomizeScale=False \
    train.ppo.priv_info=True

    echo "Completed SCALE ${SCALE}"
    echo ""
done

echo "All scales completed!"
