#!/bin/bash
# Script to run all three learning methods in sequence

echo "========================================================"
echo "Submitting all learning methods - $(date)"
echo "========================================================"

# Submit supervised learning job and capture its job ID
SUPERVISED_JOB=$(sbatch run_supervised.sh | awk '{print $4}')
echo "Submitted supervised learning job with ID: $SUPERVISED_JOB"

# Submit mean teacher job with dependency on supervised job
MEAN_TEACHER_JOB=$(sbatch --dependency=afterany:$SUPERVISED_JOB run_mean_teacher.sh | awk '{print $4}')
echo "Submitted mean teacher learning job with ID: $MEAN_TEACHER_JOB (will run after job $SUPERVISED_JOB)"

# Submit mixmatch job with dependency on mean teacher job
MIXMATCH_JOB=$(sbatch --dependency=afterany:$MEAN_TEACHER_JOB run_mixmatch.sh | awk '{print $4}')
echo "Submitted mixmatch learning job with ID: $MIXMATCH_JOB (will run after job $MEAN_TEACHER_JOB)"

echo "========================================================"
echo "All jobs submitted. Check status with: squeue -u $USER"
echo "========================================================"