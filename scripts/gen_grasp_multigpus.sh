#!/bin/bash
# Multi-GPU grasp generation script
# Distributes scale generations across multiple GPUs in parallel
# Usage: ./gen_grasp_multigpus.sh GPU1 GPU2 [GPU3 ...]
# Example: ./gen_grasp_multigpus.sh 0 1

if [ $# -lt 1 ]; then
    echo "Error: At least one GPU number is required"
    echo "Usage: ./gen_grasp_multigpus.sh GPU1 GPU2 [GPU3 ...]"
    echo "Example: ./gen_grasp_multigpus.sh 0 1"
    exit 1
fi

GPUS=($@)
NUM_GPUS=${#GPUS[@]}
SCALES=(0.66 0.92)
NUM_SCALES=${#SCALES[@]}

echo "========================================="
echo "Multi-GPU Grasp Generation"
echo "GPUs: ${GPUS[@]}"
echo "Scales: ${SCALES[@]}"
echo "========================================="
echo ""

# Array to store background process PIDs
PIDS=()

# Launch jobs across GPUs in round-robin fashion
for i in "${!SCALES[@]}"; do
    SCALE=${SCALES[$i]}
    GPU=${GPUS[$((i % NUM_GPUS))]}

    echo "Launching SCALE ${SCALE} on GPU ${GPU}..."

    (
        echo "========================================="
        echo "GPU ${GPU}: Starting SCALE ${SCALE}"
        echo "========================================="

        CUDA_VISIBLE_DEVICES=${GPU} \
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

        echo ""
        echo "GPU ${GPU}: Completed SCALE ${SCALE}"
        echo ""
    ) > "gen_grasp_scale_${SCALE}_gpu_${GPU}.log" 2>&1 &

    PIDS+=($!)

    # Small delay to avoid race conditions on startup
    sleep 1
done

echo ""
echo "All jobs launched. Waiting for completion..."
echo "PIDs: ${PIDS[@]}"
echo "Logs: gen_grasp_scale_*.log"
echo ""

# Wait for all background jobs to complete
FAILED=0
for i in "${!PIDS[@]}"; do
    PID=${PIDS[$i]}
    SCALE=${SCALES[$i]}
    GPU=${GPUS[$((i % NUM_GPUS))]}

    wait $PID
    EXIT_CODE=$?

    if [ $EXIT_CODE -eq 0 ]; then
        echo "✓ SCALE ${SCALE} (GPU ${GPU}) completed successfully"
    else
        echo "✗ SCALE ${SCALE} (GPU ${GPU}) failed with exit code ${EXIT_CODE}"
        FAILED=$((FAILED + 1))
    fi
done

echo ""
echo "========================================="
if [ $FAILED -eq 0 ]; then
    echo "All scales completed successfully!"
else
    echo "Completed with ${FAILED} failure(s)"
    echo "Check gen_grasp_scale_*.log for details"
fi
echo "========================================="

exit $FAILED
