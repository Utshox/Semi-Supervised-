#!/bin/bash
# Script to check GPU usage in SLURM jobs

# Get the job ID (either from argument or most recent job)
if [ -z "$1" ]; then
    JOB_ID=$(squeue -u $USER -h -o "%i" | head -n 1)
    echo "Checking most recent job: $JOB_ID"
else
    JOB_ID=$1
    echo "Checking job: $JOB_ID"
fi

# Check if job exists
if ! squeue -j $JOB_ID &> /dev/null; then
    echo "Job $JOB_ID not found or not running!"
    exit 1
fi

echo "========================================================"
echo "Running nvidia-smi on compute node:"
srun --jobid=$JOB_ID --gres=gpu:0 nvidia-smi

echo "========================================================"
echo "Checking GPU processes:"
srun --jobid=$JOB_ID --gres=gpu:0 nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv

echo "========================================================"
echo "To monitor GPU usage in real-time, run:"
echo "srun --jobid=$JOB_ID --gres=gpu:0 watch -n 1 nvidia-smi"
echo "========================================================"