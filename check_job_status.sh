#!/bin/bash
# Script to check status of SLURM jobs

echo "========================================================"
echo "Current job status - $(date)"
echo "========================================================"
echo "Your running jobs:"
squeue -u $USER
echo ""

# Get node information for running jobs
RUNNING_NODES=$(squeue -u $USER -o "%N" -h | xargs)
if [ -n "$RUNNING_NODES" ]; then
    echo "Checking GPU usage on your nodes:"
    for NODE in $RUNNING_NODES; do
        echo "GPU utilization on $NODE:"
        srun --pty --jobid=$(squeue -u $USER -o "%i" -h) nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv 2>/dev/null || echo "Cannot connect to node for GPU stats"
    done
    echo ""
fi

echo "Completed/running jobs in the last 24 hours:"
sacct -u $USER -S $(date -d "24 hours ago" +%Y-%m-%dT%H:%M:%S) --format=JobID,JobName,State,Elapsed,NodeList,AllocGRES
echo ""

echo "GPU usage summary:"
sacct -u $USER -S $(date -d "24 hours ago" +%Y-%m-%dT%H:%M:%S) --format=JobID,JobName,State,Elapsed,AllocGRES | grep "gpu:"
echo ""

echo "Estimated total GPU hours used:"
sacct -u $USER -S $(date -d "30 days ago" +%Y-%m-%dT%H:%M:%S) --format=JobID,AllocGRES,Elapsed -n | grep "gpu:" | \
awk -F'[:=]' '{
    split($3, time, ":"); 
    if (NF >= 4) {
        gpus = $2;
        hours = time[1] + time[2]/60 + time[3]/3600;
        total += gpus * hours;
    }
} END {printf "%.2f GPU hours in the last 30 days\n", total}'
echo "========================================================"