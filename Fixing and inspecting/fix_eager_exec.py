#!/usr/bin/env python3
# A more direct fix for the MeanTeacherTrainer class to ensure it works with eager execution

import tensorflow as tf
import numpy as np
import sys
import os
from pathlib import Path

# First, check if train_ssl_tf2n.py exists in the current directory
file_path = '/scratch/lustre/home/mdah0000/smm/v14/train_ssl_tf2n.py'
if not os.path.exists(file_path):
    print(f"Error: Cannot find {file_path}")
    sys.exit(1)

# Backup the original file if not already backed up
backup_path = f'{file_path}.bak'
if not os.path.exists(backup_path):
    os.system(f'cp {file_path} {backup_path}')
    print(f"Backed up original file to {backup_path}")

# Now create a simple patch to run the model with eager execution enabled
# This will add a block at the beginning of the file to ensure eager execution is enabled
with open(file_path, 'r') as f:
    content = f.read()

# Add eager execution to the beginning of the file after imports
import_section_end = content.find("def setup_gpu()")
if import_section_end == -1:
    import_section_end = content.find("def prepare_data_paths(")

insert_position = import_section_end
eager_execution_code = """
# Enable eager execution for better debugging
print("Enabling eager execution for MeanTeacher training...")
tf.config.run_functions_eagerly(True)

"""

# Only insert if it doesn't already exist
if "run_functions_eagerly" not in content[:import_section_end]:
    new_content = content[:insert_position] + eager_execution_code + content[insert_position:]
    
    # Write the modified content back to the file
    with open(file_path, 'w') as f:
        f.write(new_content)
    
    print(f"Added eager execution to {file_path}")
else:
    print("Eager execution is already enabled in the file.")

# Now let's also create a simplistic patch for train_step
# This entirely replaces the train_step method with a simpler version
try:
    # Define a replacement train_step method
    train_step_replacement = '''
    def train_step(self, data):
        labeled_data, unlabeled_data = data
        labeled_images, true_labels = labeled_data
        unlabeled_student_input, unlabeled_teacher_input = unlabeled_data

        with tf.GradientTape() as tape:
            # Supervised loss on labeled data
            student_labeled_logits = self.student_model(labeled_images, training=True)
            supervised_loss = self.compiled_loss(true_labels, student_labeled_logits)

            # Consistency loss on unlabeled data
            student_unlabeled_logits = self.student_model(unlabeled_student_input, training=True)
            teacher_unlabeled_logits = self.teacher_model(unlabeled_teacher_input, training=False)

            student_unlabeled_probs = tf.nn.sigmoid(student_unlabeled_logits)
            teacher_unlabeled_probs = tf.nn.sigmoid(teacher_unlabeled_logits)
            
            consistency_loss_value = self.consistency_loss_fn(teacher_unlabeled_probs, student_unlabeled_probs)
            
            total_loss = supervised_loss + self.consistency_weight * consistency_loss_value

        # Compute gradients for student model
        gradients = tape.gradient(total_loss, self.student_model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.student_model.trainable_variables))

        # Update teacher model using EMA
        self._update_teacher_model()
        
        # Calculate dice score manually
        student_dice = self._calculate_dice(true_labels, student_labeled_logits)

        # Return primitive Python values to avoid serialization issues
        return {
            "loss": float(total_loss),
            "supervised_loss": float(supervised_loss),
            "consistency_loss": float(consistency_loss_value),
            "student_dice": float(student_dice)
        }
'''

    # Use a simpler approach - we'll just replace the entire class implementation
    mean_teacher_class_start = content.find("class MeanTeacherTrainer(tf.keras.Model):")
    if mean_teacher_class_start != -1:
        # Find the end of train_step method
        train_step_start = content.find("    def train_step(self, data):", mean_teacher_class_start)
        if train_step_start != -1:
            # Find the next method after train_step
            next_method_start = content.find("    def test_step(self, data):", train_step_start)
            if next_method_start != -1:
                # Replace the train_step method with our new version
                new_content = content[:train_step_start] + train_step_replacement + content[next_method_start:]
                
                # Write the modified content back to the file
                with open(file_path, 'w') as f:
                    f.write(new_content)
                
                print(f"Replaced train_step method in {file_path}")
            else:
                print("Could not find the end of train_step method")
        else:
            print("Could not find train_step method")
    else:
        print("Could not find MeanTeacherTrainer class")

except Exception as e:
    print(f"Error modifying file: {e}")

print("Fixes applied. Now you can run the script with better error visibility.")
