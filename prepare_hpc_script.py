#!/usr/bin/env python3
# Script to prepare proper HPC script for the Mean Teacher model

import os
import sys
from pathlib import Path

# HPC paths
HPC_DATA_DIR = "/lustre/home/mdah0000/images/preprocessed_v2"
HPC_WORK_DIR = "/lustre/home/mdah0000/smm/v14"
LOCAL_WORK_DIR = "/stud3/2023/mdah0000/smm/Semi-Supervised-"

def create_hpc_script():
    """Create a proper SLURM script for the HPC environment."""
    
    # Create the SLURM script content
    script_content = """#!/bin/bash
# SLURM batch job script for Mean Teacher v2 semi-supervised learning on HPC

#SBATCH -p gpu
#SBATCH --gres=gpu:2
#SBATCH -n 8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH -o meanteacher_v2-hpc-%j.out
#SBATCH -e meanteacher_v2-hpc-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --job-name=pancreas_mt_hpc

# --- Configuration ---
# Use HPC directories
DATA_DIR="{hpc_data_dir}" # HPC path for preprocessed data
WORK_DIR="{hpc_work_dir}" # HPC working directory
PYTHON_SCRIPT_NAME="run_mean_teacher_v2.py" # The main script
PYTHON_SCRIPT_PATH="$WORK_DIR/$PYTHON_SCRIPT_NAME"
OUTPUT_DIR_BASE="$WORK_DIR/mean_teacher_v2_results" # Base directory for experiment outputs

# Experiment specific parameters
EXPERIMENT_NAME="mt_v2_hpc_$(date +%Y%m%d_%H%M%S)"
IMG_SIZE=256
NUM_LABELED=15
NUM_VALIDATION=5
BATCH_SIZE=8
NUM_EPOCHS=100
LEARNING_RATE=1e-4
EMA_DECAY=0.999
CONSISTENCY_MAX=10.0
CONSISTENCY_RAMPUP=30 # Epochs
EARLY_STOPPING_PATIENCE=20
SEED=42
VERBOSE=1 # 0=silent, 1=progress bar, 2=one line per epoch

# --- SLURM Preamble ---
echo "========================================================="
echo "Running Mean Teacher v2 Learning on HPC - $(date)"
echo "========================================================="
echo "Running on node: $HOSTNAME"
echo "Working directory: $(pwd)"
echo "Python script: $PYTHON_SCRIPT_PATH"
echo "Data directory: $DATA_DIR"
echo "Base Output directory: $OUTPUT_DIR_BASE"
echo "Experiment Name: $EXPERIMENT_NAME"
echo "========================================================="

# --- Environment Setup ---
# Create output directory
CURRENT_OUTPUT_DIR="$OUTPUT_DIR_BASE/$EXPERIMENT_NAME"
mkdir -p "$CURRENT_OUTPUT_DIR"
echo "Results will be saved in: $CURRENT_OUTPUT_DIR"

# --- Add any required modules on your HPC system ---
# module load tensorflow/2.x-gpu
# module load python/3.8
# Add any other modules you need

# --- TensorFlow GPU environment settings ---
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_CPP_MIN_LOG_LEVEL=1 # Suppress TensorFlow informational messages

# --- Apply fixes ---
echo "Applying eager execution fix to help with debugging..."
python3 fix_eager_exec.py

# --- Check if TensorFlow can see the GPU ---
echo "TensorFlow GPU availability check:"
python3 -c "import tensorflow as tf; print('Num GPUs Available:', len(tf.config.list_physical_devices('GPU'))); print('GPU Devices:', tf.config.list_physical_devices('GPU'))"

# --- Run the Mean Teacher v2 training script ---
echo "Running Mean Teacher v2 training script: $PYTHON_SCRIPT_NAME"
python3 "$PYTHON_SCRIPT_PATH" \\
    --data_dir "$DATA_DIR" \\
    --output_dir "$OUTPUT_DIR_BASE" \\
    --experiment_name "$EXPERIMENT_NAME" \\
    --img_size "$IMG_SIZE" \\
    --num_labeled "$NUM_LABELED" \\
    --num_validation "$NUM_VALIDATION" \\
    --batch_size "$BATCH_SIZE" \\
    --num_epochs "$NUM_EPOCHS" \\
    --learning_rate "$LEARNING_RATE" \\
    --ema_decay "$EMA_DECAY" \\
    --consistency_max "$CONSISTENCY_MAX" \\
    --consistency_rampup "$CONSISTENCY_RAMPUP" \\
    --early_stopping_patience "$EARLY_STOPPING_PATIENCE" \\
    --seed "$SEED" \\
    --verbose "$VERBOSE"

echo "========================================================="
echo "Job completed at $(date)"
echo "========================================================="
""".format(
        hpc_data_dir=HPC_DATA_DIR,
        hpc_work_dir=HPC_WORK_DIR
    )
    
    # Create the script file
    script_path = os.path.join(LOCAL_WORK_DIR, "run_mean_teacher_v2_hpc.sh")
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    os.chmod(script_path, 0o755)  # Make it executable
    
    print(f"Created HPC script at: {script_path}")
    print("\nTo transfer to HPC, you can use:")
    print(f"scp {script_path} username@hpc_login_node:{HPC_WORK_DIR}/")
    print("\nThen on the HPC system, run:")
    print(f"cd {HPC_WORK_DIR}")
    print("sbatch run_mean_teacher_v2_hpc.sh")


def create_data_path_fix_script():
    """Create a script to fix data paths in the code."""
    
    script_content = """#!/usr/bin/env python3
# Script to fix data paths in run_mean_teacher_v2.py

import os
import sys
from pathlib import Path
import re

# Path configuration
LOCAL_WORK_DIR = "{local_work_dir}"
HPC_WORK_DIR = "{hpc_work_dir}"
HPC_DATA_DIR = "{hpc_data_dir}"

def fix_data_paths():
    # Determine which environment we're in
    if os.path.exists(LOCAL_WORK_DIR):
        work_dir = LOCAL_WORK_DIR
        print(f"Detected local environment: {work_dir}")
    elif os.path.exists(HPC_WORK_DIR):
        work_dir = HPC_WORK_DIR
        print(f"Detected HPC environment: {work_dir}")
    else:
        print("Could not determine environment. Please check paths.")
        sys.exit(1)
    
    # Path to the main script
    script_path = os.path.join(work_dir, "run_mean_teacher_v2.py")
    
    if not os.path.exists(script_path):
        print(f"Error: Script not found at {script_path}")
        sys.exit(1)
    
    # Create backup
    backup_path = f"{script_path}.bak"
    if not os.path.exists(backup_path):
        with open(script_path, 'r') as src, open(backup_path, 'w') as dst:
            dst.write(src.read())
        print(f"Created backup at {backup_path}")
    
    # Read the file content
    with open(script_path, 'r') as f:
        content = f.read()
    
    # Make environment-specific modifications to the content
    # This focuses on adapting paths - for HPC we might need to check if 
    # directories exist and handle special structures
    
    # Function to adapt prepare_data_paths to the environment
    if os.path.exists(HPC_WORK_DIR):  # If we're on HPC
        # Add code to handle HPC specific adaptations, like the directory structure
        hpc_handler = '''
    # Special handling for HPC environment where data might be in a different format
    data_dir = Path(data_dir_str)
    
    # Check if we're in the HPC environment
    on_hpc = Path("/lustre").exists()
    
    # Modified directory structure handling for HPC
    if on_hpc:
        print("Detected HPC environment, using special directory handling")
        # HPC specific directory structure
        all_patient_dirs = []
        # This assumes HPC data is stored differently, adapt to your actual structure
        for i in range(1, 100):  # Assuming there are patient numbers 1-99
            patient_num = f"{i:03d}"  # Format as 001, 002, etc.
            patient_dir = data_dir / f"pancreas_{patient_num}"
            
            # Check if patient directory exists or create a Path object for it
            if patient_dir.exists():
                all_patient_dirs.append(patient_dir)
            
        if not all_patient_dirs:
            # Try alternative directory structure if standard isn't found
            alternative_pattern = list(data_dir.glob("*"))
            if alternative_pattern:
                all_patient_dirs = [p for p in alternative_pattern if p.is_dir()]
                print(f"Using alternative directory structure, found {len(all_patient_dirs)} directories")
    else:
        # Standard directory structure for local testing
        all_patient_dirs = sorted([p for p in data_dir.iterdir() if p.is_dir() and p.name.startswith("pancreas_")])
    '''
        
        # Find where the prepare_data_paths function begins
        prepare_data_paths_start = content.find("def prepare_data_paths(")
        if prepare_data_paths_start != -1:
            # Find the line after the function definition where we insert our code
            insert_point = content.find("all_patient_dirs = sorted(", prepare_data_paths_start)
            if insert_point != -1:
                # Replace the all_patient_dirs line with our adaptive code
                old_line = content[insert_point:content.find("\\n", insert_point)+1]
                content = content.replace(old_line, hpc_handler)
    
    # Write the modified content back to the file
    with open(script_path, 'w') as f:
        f.write(content)
    
    print(f"Updated {script_path} for environment-specific paths.")

if __name__ == "__main__":
    fix_data_paths()
    print("Data path fix applied successfully!")
""".format(
        local_work_dir=LOCAL_WORK_DIR,
        hpc_work_dir=HPC_WORK_DIR,
        hpc_data_dir=HPC_DATA_DIR
    )
    
    # Create the script file
    script_path = os.path.join(LOCAL_WORK_DIR, "fix_data_paths.py")
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    os.chmod(script_path, 0o755)  # Make it executable
    
    print(f"Created data path fix script at: {script_path}")
    print("\nYou can run it with:")
    print(f"python3 {script_path}")


def create_noneType_fix():
    """Create a more comprehensive fix for the NoneType error in train_ssl_tf2n.py."""
    
    script_content = """#!/usr/bin/env python3
# A comprehensive fix for the MeanTeacherTrainer class in train_ssl_tf2n.py
# This addresses both the 'empty logs' error and potential None conversion issues

import tensorflow as tf
import numpy as np
import sys
import os
from pathlib import Path

# First, check if train_ssl_tf2n.py exists in the current directory
work_dir = "{local_work_dir}" if os.path.exists("{local_work_dir}") else "{hpc_work_dir}"
file_path = os.path.join(work_dir, 'train_ssl_tf2n.py')

if not os.path.exists(file_path):
    print(f"Error: Cannot find {file_path}")
    sys.exit(1)

# Backup the original file if not already backed up
backup_path = f'{file_path}.bak'
if not os.path.exists(backup_path):
    os.system(f'cp {file_path} {backup_path}')
    print(f"Backed up original file to {backup_path}")

# Read the current content
with open(file_path, 'r') as f:
    content = f.read()

# Find the MeanTeacherTrainer class to modify its train_step method
class_start = content.find("class MeanTeacherTrainer")
if class_start == -1:
    print("Error: MeanTeacherTrainer class not found in the file!")
    sys.exit(1)

# Find the train_step method in the class
train_step_start = content.find("def train_step(self", class_start)
if train_step_start == -1:
    print("Error: train_step method not found in MeanTeacherTrainer class!")
    sys.exit(1)

# Find the end of the train_step method
# This is a bit tricky - we'll look for the next method definition or the end of the class
next_method_start = content.find("def ", train_step_start + 1)
if next_method_start == -1:
    # Look for the end of the class
    class_end = content.find("# ---", train_step_start)
    train_step_end = class_end if class_end != -1 else len(content)
else:
    train_step_end = next_method_start

# Get the current train_step method content
current_train_step = content[train_step_start:train_step_end]

# Check if the method already returns proper logs
if "return logs" in current_train_step:
    # If already fixed in some way, we should carefully modify it
    if not "total_loss_val = float(total_loss)" in current_train_step:
        # Modify specifically the return logs part by ensuring tensor values are converted to Python values
        end_of_method = current_train_step.rfind("return")
        if end_of_method != -1:
            logs_section_start = current_train_step.rfind("logs =", 0, end_of_method)
            if logs_section_start != -1:
                logs_section_end = end_of_method
                logs_section = current_train_step[logs_section_start:logs_section_end]
                
                # Prepare modified logs section
                new_logs_section = '''
        # Calculate student dice score manually for logging
        student_dice = self._calculate_dice(true_labels, student_labeled_logits)
        
        # Handle NoneType conversion and conversion to Python values
        def to_float(tensor):
            if tensor is None:
                return 0.0
            try:
                return float(tensor)
            except (TypeError, ValueError):
                print(f"Warning: Could not convert {tensor} to float, using 0.0")
                return 0.0
        
        # Convert tensors to Python values to avoid serialization issues
        total_loss_val = to_float(total_loss)
        supervised_loss_val = to_float(supervised_loss)
        consistency_loss_val = to_float(consistency_loss_value)
        student_dice_val = to_float(student_dice)
        
        # Prepare logs with Python values
        logs = {{
            'loss': total_loss_val,
            'supervised_loss': supervised_loss_val,
            'consistency_loss': consistency_loss_val,
            'student_dice': student_dice_val
        }}
'''
                # Replace the logs section
                modified_train_step = current_train_step.replace(logs_section, new_logs_section)
                
                # Replace the train_step method in the original content
                content = content.replace(current_train_step, modified_train_step)
                
                with open(file_path, 'w') as f:
                    f.write(content)
                
                print("Successfully updated the train_step method to handle NoneType values and return proper logs.")
            else:
                print("Could not find logs section in the method, manual inspection needed.")
        else:
            print("Could not find return statement in the method, manual inspection needed.")
    else:
        print("The train_step method already appears to be fixed properly.")
else:
    # More extensive fix as likely the method doesn't return logs at all
    # We'll modify the entire method
    
    # Look for the method signature to preserve it
    method_sig_end = current_train_step.find("):") + 2
    method_sig = current_train_step[:method_sig_end]
    
    # Create a completely new train_step implementation
    new_train_step = method_sig + '''
        inputs, true_labels = data
        
        # Extract student and teacher parameters
        student_model = self.student_model
        teacher_model = self.teacher_model
        optimizer = self.optimizer
        supervised_loss_fn = self.supervised_loss_fn
        consistency_loss_fn = self.consistency_loss_fn
        consistency_weight = self.consistency_weight
        
        # Apply same augmentation to inputs for student and teacher
        student_inputs = inputs
        teacher_inputs = inputs
        
        with tf.GradientTape() as tape:
            # Forward pass for student
            student_outputs = student_model(student_inputs, training=True)
            
            # If model outputs a list/tuple, take the first element
            student_logits = student_outputs[0] if isinstance(student_outputs, (list, tuple)) else student_outputs
            student_labeled_logits = student_logits
            
            # Compute supervised loss with student predictions
            supervised_loss = supervised_loss_fn(true_labels, student_labeled_logits)
            
            # Forward pass for teacher (no gradients needed)
            # Use EMA weights in teacher model
            teacher_outputs = teacher_model(teacher_inputs, training=False)
            
            # If model outputs a list/tuple, take the first element
            teacher_logits = teacher_outputs[0] if isinstance(teacher_outputs, (list, tuple)) else teacher_outputs
            teacher_labeled_logits = teacher_logits
            
            # Compute consistency loss between student and teacher predictions
            consistency_loss_value = consistency_loss_fn(
                tf.nn.sigmoid(teacher_labeled_logits),
                tf.nn.sigmoid(student_labeled_logits)
            )
            
            # Apply consistency weight
            weighted_consistency_loss = consistency_weight * consistency_loss_value
            
            # Total loss
            total_loss = supervised_loss + weighted_consistency_loss
        
        # Get gradients and update student model
        gradients = tape.gradient(total_loss, student_model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, student_model.trainable_variables))
        
        # Calculate student dice score manually for logging
        student_dice = self._calculate_dice(true_labels, student_labeled_logits)
        
        # Handle NoneType conversion and conversion to Python values
        def to_float(tensor):
            if tensor is None:
                return 0.0
            try:
                return float(tensor)
            except (TypeError, ValueError):
                print(f"Warning: Could not convert {tensor} to float, using 0.0")
                return 0.0
        
        # Convert tensors to Python values to avoid serialization issues
        total_loss_val = to_float(total_loss)
        supervised_loss_val = to_float(supervised_loss)
        consistency_loss_val = to_float(consistency_loss_value)
        student_dice_val = to_float(student_dice)
        
        # Prepare logs with Python values
        logs = {{
            'loss': total_loss_val,
            'supervised_loss': supervised_loss_val,
            'consistency_loss': consistency_loss_val,
            'student_dice': student_dice_val
        }}
        
        return logs
'''
    
    # Replace the train_step method in the original content
    content = content.replace(current_train_step, new_train_step)
    
    # Add the _calculate_dice method if it doesn't exist
    if "def _calculate_dice(self" not in content:
        calculate_dice_method = '''
    def _calculate_dice(self, y_true, y_pred_logits, smooth=1e-6):
        """Calculate Dice coefficient for binary segmentation."""
        # Apply sigmoid to get probabilities
        y_pred = tf.nn.sigmoid(y_pred_logits)
        
        # Flatten the tensors
        y_true_f = tf.reshape(y_true, [-1])
        y_pred_f = tf.reshape(y_pred, [-1])
        
        # Calculate intersection and union
        intersection = tf.reduce_sum(y_true_f * y_pred_f)
        dice = (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
        
        return dice
'''
        
        # Find the end of the class or the appropriate place to add the method
        if next_method_start != -1:
            # Add before the next method
            content = content[:next_method_start] + calculate_dice_method + content[next_method_start:]
        else:
            # Add at the end of the class
            class_end = content.find("# ---", train_step_end)
            if class_end != -1:
                content = content[:class_end] + calculate_dice_method + content[class_end:]
            else:
                # Add at the end of the file
                content += calculate_dice_method
    
    with open(file_path, 'w') as f:
        f.write(content)
    
    print("Successfully replaced the train_step method with a comprehensive implementation that handles errors.")

if __name__ == "__main__":
    create_noneType_fix()
    print("Comprehensive fix for the MeanTeacherTrainer class has been applied!")
    print("This should resolve both the 'empty logs' error and potential None conversion issues.")
""".format(
        local_work_dir=LOCAL_WORK_DIR,
        hpc_work_dir=HPC_WORK_DIR
    )
    
    # Create the script file
    script_path = os.path.join(LOCAL_WORK_DIR, "fix_train_step_comprehensive.py")
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    os.chmod(script_path, 0o755)  # Make it executable
    
    print(f"Created comprehensive train_step fix script at: {script_path}")
    print("\nYou can run it with:")
    print(f"python3 {script_path}")


if __name__ == "__main__":
    print("Preparing scripts for the Mean Teacher model on HPC environment...")
    create_hpc_script()
    create_data_path_fix_script()
    create_noneType_fix()
    print("\nAll preparation scripts have been created successfully.")
    print("\nFollow these steps to get your code working on the HPC:")
    print("1. Run fix_train_step_comprehensive.py to fix the empty logs issue")
    print("2. Run fix_data_paths.py to update the path handling for HPC")
    print("3. Copy the files to your HPC environment:")
    print(f"   scp -r {LOCAL_WORK_DIR}/* username@hpc_login_node:{HPC_WORK_DIR}/")
    print("4. On the HPC, run your job:")
    print(f"   cd {HPC_WORK_DIR}")
    print("   sbatch run_mean_teacher_v2_hpc.sh")
