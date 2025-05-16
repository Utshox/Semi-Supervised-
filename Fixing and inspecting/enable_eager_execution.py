
import tensorflow as tf
import sys

# When running the script with the empty logs issue, add this at the beginning of your run
print("Setting TensorFlow to run eagerly...")
tf.config.run_functions_eagerly(True)
print("TensorFlow is now running eagerly.")

# This will help diagnose what's happening with the train_step function
print("TensorFlow version:", tf.__version__)
print("Eager execution enabled:", tf.executing_eagerly())
