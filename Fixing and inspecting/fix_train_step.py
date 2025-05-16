#!/usr/bin/env python3
# Modified train_step fix for MeanTeacherTrainer

import tensorflow as tf
import numpy as np
import sys
import os
from pathlib import Path

# Fix MeanTeacherTrainer train_step method
def fix_train_step():
    # Path to the train_ssl_tf2n.py file
    file_path = '/scratch/lustre/home/mdah0000/smm/v14/train_ssl_tf2n.py'
    
    # Backup the original file
    backup_path = f'{file_path}.bak'
    if not os.path.exists(backup_path):
        os.system(f'cp {file_path} {backup_path}')
        print(f"Backed up original file to {backup_path}")
    
    # Read the file content
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Find the train_step method in MeanTeacherTrainer class
    train_step_pattern = """    def train_step(self, data):
        labeled_data, unlabeled_data = data
        labeled_images, true_labels = labeled_data
        unlabeled_student_input, unlabeled_teacher_input = unlabeled_data

        with tf.GradientTape() as tape:
            # Supervised loss on labeled data
            student_labeled_logits = self.student_model(labeled_images, training=True)
            supervised_loss = self.compiled_loss(true_labels, student_labeled_logits) # Uses loss_fn from compile

            # Consistency loss on unlabeled data
            student_unlabeled_logits = self.student_model(unlabeled_student_input, training=True)
            teacher_unlabeled_logits = self.teacher_model(unlabeled_teacher_input, training=False) # Teacher not in training mode

            student_unlabeled_probs = tf.nn.sigmoid(student_unlabeled_logits)
            teacher_unlabeled_probs = tf.nn.sigmoid(teacher_unlabeled_logits)
            
            consistency_loss_value = self.consistency_loss_fn(teacher_unlabeled_probs, student_unlabeled_probs)
            
            total_loss = supervised_loss + self.consistency_weight * consistency_loss_value

        # Compute gradients for student model
        gradients = tape.gradient(total_loss, self.student_model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.student_model.trainable_variables))

        # Update teacher model using EMA
        self._update_teacher_model()

        # Update compiled metrics (e.g., student_dice on labeled data)
        self.compiled_metrics.update_state(true_labels, student_labeled_logits)

        # Prepare logs
        logs = {'loss': total_loss, 'supervised_loss': supervised_loss, 'consistency_loss': consistency_loss_value}
        for metric in self.metrics: # self.metrics includes compiled_metrics
            logs[metric.name] = metric.result()
        
        return logs"""
    
    # The fixed train_step method
    fixed_train_step = """    def train_step(self, data):
        labeled_data, unlabeled_data = data
        labeled_images, true_labels = labeled_data
        unlabeled_student_input, unlabeled_teacher_input = unlabeled_data

        with tf.GradientTape() as tape:
            # Supervised loss on labeled data
            student_labeled_logits = self.student_model(labeled_images, training=True)
            supervised_loss = self.compiled_loss(true_labels, student_labeled_logits) # Uses loss_fn from compile

            # Consistency loss on unlabeled data
            student_unlabeled_logits = self.student_model(unlabeled_student_input, training=True)
            teacher_unlabeled_logits = self.teacher_model(unlabeled_teacher_input, training=False) # Teacher not in training mode

            student_unlabeled_probs = tf.nn.sigmoid(student_unlabeled_logits)
            teacher_unlabeled_probs = tf.nn.sigmoid(teacher_unlabeled_logits)
            
            consistency_loss_value = self.consistency_loss_fn(teacher_unlabeled_probs, student_unlabeled_probs)
            
            total_loss = supervised_loss + self.consistency_weight * consistency_loss_value

        # Compute gradients for student model
        gradients = tape.gradient(total_loss, self.student_model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.student_model.trainable_variables))

        # Update teacher model using EMA
        self._update_teacher_model()
        
        # Calculate student dice score manually for logging
        student_dice = self._calculate_dice(true_labels, student_labeled_logits)

        # Update metrics
        self.compiled_metrics.update_state(true_labels, student_labeled_logits)
        
        # Convert tensors to Python values to avoid possible serialization issues
        total_loss_val = float(total_loss)
        supervised_loss_val = float(supervised_loss)
        consistency_loss_val = float(consistency_loss_value)
        student_dice_val = float(student_dice)
        
        # Prepare logs with Python values
        logs = {
            'loss': total_loss_val,
            'supervised_loss': supervised_loss_val,
            'consistency_loss': consistency_loss_val,
            'student_dice': student_dice_val
        }
        
        return logs"""
    
    # Replace the train_step method
    new_content = content.replace(train_step_pattern, fixed_train_step)
    
    # Write the modified content back to the file
    with open(file_path, 'w') as f:
        f.write(new_content)
    
    print(f"Updated train_step method in {file_path}")

if __name__ == "__main__":
    fix_train_step()
