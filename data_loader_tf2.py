import tensorflow as tf
import numpy as np
from pathlib import Path
import nibabel as nib  # You might not need this if you are not using .nii files
import time
import logging
import sys  # Import sys for stderr
import math

# Define AUTOTUNE for tf.data
AUTOTUNE = tf.data.experimental.AUTOTUNE

class PancreasDataLoader:
    def __init__(self, config):
        self.config = config
        # Use config for target size, ensure y (height) is first, then x (width)
        self.target_size = (config.img_size_y, config.img_size_x)

    def preprocess_volume(self, image_path, label_path=None):
        """Loads and preprocesses a single volume.
        Assumes input .npy files are (Depth, Height, Width).
        Outputs volumes as (Depth, Height, Width)."""
        try:
            # Load the image data
            img_data = np.load(str(image_path))  # Expected shape (D, H, W)
            
            # Handle NaN values
            img_data = np.nan_to_num(img_data)
            
            # Normalize image to [0,1] range if not already normalized
            if img_data.max() > 1.0:
                img_data = (img_data - img_data.min()) / (img_data.max() - img_data.min())
            
            img_data = tf.convert_to_tensor(img_data, dtype=tf.float32)
            
            # Get dimensions
            shape = tf.shape(img_data)
            num_slices = shape[0]
            target_h, target_w = self.target_size
            
            # Process each slice
            processed_slices = []
            for i in range(num_slices):
                slice_2d = img_data[i]  # No need for :, : with TF
                current_shape = tf.shape(slice_2d)
                
                # Resize if needed
                if current_shape[0] != target_h or current_shape[1] != target_w:
                    # Add channel dim for resize, then remove
                    resized = tf.image.resize(
                        slice_2d[..., tf.newaxis],
                        [target_h, target_w],
                        method='bilinear'
                    )
                    resized = tf.squeeze(resized, axis=-1)
                else:
                    resized = slice_2d
                    
                processed_slices.append(resized)
            
            # Stack all processed slices
            final_img_data = tf.stack(processed_slices, axis=0)
            
            # Handle label if provided
            if label_path:
                try:
                    # Load and process label
                    label_data = np.load(str(label_path))  # Expected shape (D, H, W)
                    
                    # Handle NaN values in labels
                    label_data = np.nan_to_num(label_data)
                    
                    # Ensure binary labels (0 or 1)
                    label_data = (label_data > 0.5).astype(np.float32)
                    
                    label_data = tf.convert_to_tensor(label_data, dtype=tf.float32)
                    
                    label_shape = tf.shape(label_data)
                    num_label_slices = label_shape[0]
                    
                    # Verify label dimensions match image
                    if num_label_slices != num_slices:
                        raise ValueError(f"Label slices ({num_label_slices}) don't match image slices ({num_slices})")
                    
                    # Process each label slice
                    processed_label_slices = []
                    for i in range(num_label_slices):
                        label_slice = label_data[i]
                        current_shape = tf.shape(label_slice)
                        
                        # Resize if needed using nearest neighbor interpolation for labels
                        if current_shape[0] != target_h or current_shape[1] != target_w:
                            resized = tf.image.resize(
                                label_slice[..., tf.newaxis],
                                [target_h, target_w],
                                method='nearest'  # Use nearest neighbor for labels to preserve binary values
                            )
                            resized = tf.squeeze(resized, axis=-1)
                        else:
                            resized = label_slice
                            
                        # Re-binarize after resize to ensure binary values
                        resized = tf.cast(resized > 0.5, tf.float32)
                        processed_label_slices.append(resized)
                    
                    # Stack all processed label slices
                    final_label_data = tf.stack(processed_label_slices, axis=0)
                    
                    # Add debug output for label stats
                    tf.print("DEBUG Preprocess Volume: Label stats - ",
                            "sum:", tf.reduce_sum(final_label_data),
                            "min:", tf.reduce_min(final_label_data),
                            "max:", tf.reduce_max(final_label_data),
                            "shape:", tf.shape(final_label_data),
                            "unique values:", tf.unique(tf.reshape(final_label_data, [-1])),
                            output_stream=sys.stderr)
                    
                    # Final sanity check on label values
                    if tf.reduce_max(final_label_data) <= 0:
                        tf.print("WARNING: All-zero label detected!", output_stream=sys.stderr)
                    
                    return final_img_data, final_label_data
                    
                except Exception as e:
                    tf.print(f"ERROR processing label {label_path}: {str(e)}", output_stream=sys.stderr)
                    return None, None
            else:
                return final_img_data
                
        except Exception as e:
            tf.print(f"ERROR processing {image_path}: {str(e)}", output_stream=sys.stderr)
            return None if label_path is None else (None, None)

    def resize_slice(self, slice_2d):
        """Resizes a single 2D slice."""
        if slice_2d is None:
            return tf.zeros([self.config.img_size_x, self.config.img_size_y, 1], dtype=tf.float32)

        try:
            # Convert to tensor and add channel dimension if needed
            slice_2d = tf.convert_to_tensor(slice_2d, dtype=tf.float32)
            if len(slice_2d.shape) == 2:
                slice_2d = slice_2d[..., tf.newaxis]

            # Resize
            resized = tf.image.resize(
                slice_2d,
                [self.config.img_size_x, self.config.img_size_y],
                method='bilinear'
            )
            resized.set_shape([self.config.img_size_x, self.config.img_size_y, 1])
            return resized
            
        except Exception as e:
            logging.error(f"Resize error: {str(e)}")
            return tf.zeros([self.config.img_size_x, self.config.img_size_y, 1], dtype=tf.float32)

    def resize_volume(self, volume):
        """Resizes 3D volume slice by slice."""
        if volume is None:
            return tf.zeros([self.config.img_size_x, self.config.img_size_y, 1], dtype=tf.float32)

        try:
            # Convert to tensor
            volume = tf.convert_to_tensor(volume, dtype=tf.float32)
            
            # Handle 2D case
            if len(volume.shape) == 2:
                return self.resize_slice(volume)

            # Process each slice in the volume
            resized_slices = []
            depth = tf.shape(volume)[2]
            
            for i in range(depth):
                slice_2d = volume[:, :, i]
                resized = self.resize_slice(slice_2d)
                if resized is not None:
                    resized_slices.append(resized)

            if not resized_slices:
                return tf.zeros([self.config.img_size_x, self.config.img_size_y, 1], dtype=tf.float32)

            # Stack all slices
            resized_volume = tf.stack(resized_slices, axis=-1)
            return resized_volume
            
        except Exception as e:
            logging.error(f"Volume resize error: {str(e)}")
            return tf.zeros([self.config.img_size_x, self.config.img_size_y, 1], dtype=tf.float32)    
        
    @tf.function
    def augment_unlabeled(self, image, training=True):
        """Data augmentation function for unlabeled images."""
        if not training:
            return image

        try:
            # Convert to tensor if not already
            image = tf.convert_to_tensor(image)
            rank = tf.rank(image)

            # For single slice input [H, W, C]
            if rank == 3:
                # Process a single 2D slice
                slice_img = tf.ensure_shape(image, [self.config.img_size_y, self.config.img_size_x, 1])
                slice_img = tf.cast(slice_img, tf.float32)
                slice_img = self._augment_single_slice(slice_img)
                # Add batch dimension [1, H, W, C]
                return tf.expand_dims(slice_img, axis=0)

            # For volume input [D, H, W, C]
            elif rank == 4:
                # Extract first slice if shape is unknown or empty
                shape = tf.shape(image)
                if shape[0] == 0:
                    tf.print("WARNING: Empty volume in augment_unlabeled", output_stream=sys.stderr)
                    return tf.zeros([1, self.config.img_size_y, self.config.img_size_x, 1], dtype=tf.float32)

                # Process each slice in the volume
                try:
                    # Try to unstack volume into slices
                    slices = tf.unstack(image, axis=0)
                    processed_slices = []
                    
                    for slice_img in slices:
                        slice_img = tf.ensure_shape(slice_img, [self.config.img_size_y, self.config.img_size_x, 1])
                        slice_img = tf.cast(slice_img, tf.float32)
                        slice_img = self._augment_single_slice(slice_img)
                        processed_slices.append(slice_img)

                    # Stack processed slices if we have any
                    if len(processed_slices) > 0:
                        return tf.stack(processed_slices, axis=0)
                    else:
                        tf.print("WARNING: No valid slices processed", output_stream=sys.stderr)
                        return tf.zeros([1, self.config.img_size_y, self.config.img_size_x, 1], dtype=tf.float32)

                except (tf.errors.InvalidArgumentError, ValueError) as e:
                    tf.print("WARNING: Failed to process volume, falling back to first slice:", e, output_stream=sys.stderr)
                    # Extract and process first slice as fallback
                    slice_img = tf.ensure_shape(
                        tf.slice(image, [0, 0, 0, 0], [1, self.config.img_size_y, self.config.img_size_x, 1]),
                        [1, self.config.img_size_y, self.config.img_size_x, 1]
                    )
                    slice_img = tf.squeeze(slice_img, axis=0)
                    slice_img = tf.cast(slice_img, tf.float32)
                    slice_img = self._augment_single_slice(slice_img)
                    return tf.expand_dims(slice_img, axis=0)

            # Invalid rank, return zero tensor
            else:
                tf.print("WARNING: Invalid rank in augment_unlabeled:", rank, output_stream=sys.stderr)
                return tf.zeros([1, self.config.img_size_y, self.config.img_size_x, 1], dtype=tf.float32)

        except Exception as e:
            tf.print("ERROR in augment_unlabeled:", str(e), output_stream=sys.stderr)
            return tf.zeros([1, self.config.img_size_y, self.config.img_size_x, 1], dtype=tf.float32)
            
    def _augment_single_slice(self, slice_img):
        """Helper function to apply augmentations to a single slice."""
        # Random rotation
        if tf.random.uniform(()) > 0.3:
            angle = tf.random.uniform((), minval=-20, maxval=20) * math.pi / 180
            k = tf.cast(angle / (math.pi/2), tf.int32)
            slice_img = tf.image.rot90(slice_img, k=k)

        # Random flip
        if tf.random.uniform(()) > 0.5:
            slice_img = tf.image.flip_left_right(slice_img)

        # Random brightness/contrast
        if tf.random.uniform(()) > 0.3:
            slice_img = tf.image.random_brightness(slice_img, max_delta=0.2)
            slice_img = tf.image.random_contrast(slice_img, lower=0.7, upper=1.3)

        # Ensure values are in valid range
        return tf.clip_by_value(slice_img, 0.0, 1.0)

    @tf.function
    def augment_labeled(self, image, label, training=True):
        """Augmentation function for labeled data. 
        Can handle both 2D slices and 3D volumes with batch dimension.
        """
        if not training:
            return image, label
            
        def augment_single_slice(img_slice, lbl_slice):
            """Helper function to augment a single slice."""
            # Check and fix input shapes
            img_shape = tf.shape(img_slice)
            if tf.rank(img_slice) == 2:
                img_slice = tf.reshape(img_slice, [img_shape[0], self.config.img_size_x, 1])
            
            lbl_shape = tf.shape(lbl_slice)
            if tf.rank(lbl_slice) == 2:
                lbl_slice = tf.reshape(lbl_slice, [lbl_shape[0], self.config.img_size_x, 1])
            
            # Ensure proper shapes and types
            img_slice = tf.ensure_shape(img_slice, [self.config.img_size_y, self.config.img_size_x, 1])
            lbl_slice = tf.ensure_shape(lbl_slice, [self.config.img_size_y, self.config.img_size_x, 1])
            
            img_slice = tf.cast(img_slice, tf.float32)
            lbl_slice = tf.cast(lbl_slice, tf.float32)
            
            # Random rotation
            if tf.random.uniform(()) > 0.3:
                angle = tf.random.uniform((), minval=-20, maxval=20) * (math.pi / 180)
                k = tf.cast(angle / (math.pi/2), tf.int32)
                img_slice = tf.image.rot90(img_slice, k=k)
                lbl_slice = tf.image.rot90(lbl_slice, k=k)
            
            # Random flip
            if tf.random.uniform(()) > 0.5:
                img_slice = tf.image.flip_left_right(img_slice)
                lbl_slice = tf.image.flip_left_right(lbl_slice)
            
            # Random brightness/contrast (only for image)
            if tf.random.uniform(()) > 0.3:
                img_slice = tf.image.random_brightness(img_slice, max_delta=0.2)
                img_slice = tf.image.random_contrast(img_slice, lower=0.7, upper=1.3)
            
            # Ensure proper value ranges
            img_slice = tf.clip_by_value(img_slice, 0.0, 1.0)
            lbl_slice = tf.cast(lbl_slice > 0.5, tf.float32)  # Re-binarize label
            
            return img_slice, lbl_slice
        
        # Convert inputs to proper tensors and ensure correct shape
        image = tf.convert_to_tensor(image, dtype=tf.float32)
        label = tf.convert_to_tensor(label, dtype=tf.float32)
        
        # Get shape information and handle reshaping if needed
        shape = tf.shape(image)
        rank = tf.rank(image)
        
        # Fix shapes if needed
        if rank == 2:
            image = tf.reshape(image, [shape[0], self.config.img_size_x, 1])
            label = tf.reshape(label, [shape[0], self.config.img_size_x, 1])
            rank = 3  # Update rank after reshape
            shape = tf.shape(image)
        
        tf.print("DEBUG augment_labeled - Input:",
                "shape:", shape,
                "rank:", rank,
                "img sum:", tf.reduce_sum(image),
                "label sum:", tf.reduce_sum(label),
                output_stream=sys.stderr)
        
        # For single slice [H, W, C]
        if rank == 3:
            augmented_img, augmented_lbl = augment_single_slice(image, label)
            return tf.expand_dims(augmented_img, 0), tf.expand_dims(augmented_lbl, 0)
            
        # For batched data [B, H, W, C]
        elif rank == 4:
            # Use tf.map_fn to avoid InaccessibleTensorError
            def map_fn_fn(inputs):
                img_slice, lbl_slice = inputs
                return augment_single_slice(img_slice, lbl_slice)
            # Stack images and labels together for map_fn
            stacked = (image, label)
            mapped = tf.map_fn(map_fn_fn, stacked, fn_output_signature=(tf.float32, tf.float32))
            final_image, final_label = mapped
            tf.print("DEBUG augment_labeled - Output:",
                    "shape:", tf.shape(final_image),
                    "img sum:", tf.reduce_sum(final_image),
                    "label sum:", tf.reduce_sum(final_label),
                    output_stream=sys.stderr)
            return final_image, final_label
            
        else:
            tf.print(f"WARNING: Unexpected rank {rank}, returning unaugmented data", output_stream=sys.stderr)
            return image, label

     # Helper functions for zoom and shift (modified to handle cases where label is None)
    def random_zoom(self, image, label, zoom_range=(0.7, 1.3)):
        """Randomly zooms the image and label."""
        zoom_factor = tf.random.uniform((), minval=zoom_range[0], maxval=zoom_range[1])
        image_shape = tf.shape(image)

        # Resize the image using the zoom factor
        zoomed_image = tf.image.resize(image, (tf.cast(tf.cast(image_shape[0], tf.float32) * zoom_factor, tf.int32),
                                              tf.cast(tf.cast(image_shape[1], tf.float32) * zoom_factor, tf.int32)))

        # Crop or pad the zoomed image to the original size
        image = tf.image.resize_with_crop_or_pad(zoomed_image, image_shape[0], image_shape[1])

        if label is not None:
            label_shape = tf.shape(label)
            # Resize the label using the zoom factor (nearest neighbor for labels)
            zoomed_label = tf.image.resize(label, (tf.cast(tf.cast(label_shape[0], tf.float32) * zoom_factor, tf.int32),
                                                tf.cast(tf.cast(label_shape[1], tf.float32) * zoom_factor, tf.int32)),
                                        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            # Crop or pad the zoomed label to the original size
            label = tf.image.resize_with_crop_or_pad(zoomed_label, label_shape[0], label_shape[1])
            return image, label
        else:
            return image, None

    def random_shift(self, image, label, shift_range=0.1):
        """Randomly shifts the image and label."""
        shift_x = tf.random.uniform((), minval=-shift_range, maxval=shift_range) * tf.cast(self.config.img_size_x, tf.float32)
        shift_y = tf.random.uniform((), minval=-shift_range, maxval=shift_range) * tf.cast(self.config.img_size_y, tf.float32)

        # Shift the image
        shifted_image = tf.roll(image, shift=[tf.cast(shift_x, tf.int32), tf.cast(shift_y, tf.int32)], axis=[0, 1])

        if label is not None:
          # Shift the label
          shifted_label = tf.roll(label, shift=[tf.cast(shift_x, tf.int32), tf.cast(shift_y, tf.int32)], axis=[0, 1])
          return shifted_image, shifted_label
        else:
          return shifted_image, None        

    def create_unlabeled_dataset(self, image_paths, batch_size):
        """Creates an unlabeled dataset from image paths, yielding 2D slices."""
        tf.print("DEBUG: Setting up UNLABELED dataset", output_stream=sys.stderr)
        
        # Convert paths to tensor
        ds = tf.data.Dataset.from_tensor_slices(image_paths)
        
        # Map loading and preprocessing
        ds = ds.map(
            lambda x: tf.py_function(
                func=lambda p: self.preprocess_volume(p.numpy().decode('utf-8')),
                inp=[x],
                Tout=tf.float32
            ),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        
        # Filter out None values
        ds = ds.filter(lambda x: tf.reduce_all(tf.shape(x) > 0))
        
        # Add channel dimension if not present
        ds = ds.map(lambda x: tf.expand_dims(x, axis=-1))
        
        # Extract 2D slices from each volume
        def extract_slices(volume):
            num_slices = tf.shape(volume)[0]
            slice_ds = tf.data.Dataset.from_tensor_slices(
                tf.reshape(volume, [num_slices, self.config.img_size_y, self.config.img_size_x, 1])
            )
            return slice_ds
        ds = ds.flat_map(extract_slices)
        
        # Batch the dataset
        ds = ds.batch(batch_size)
        
        # Prefetch for performance
        ds = ds.prefetch(tf.data.AUTOTUNE)
        
        return ds

    def create_dataset(self, image_paths, label_paths=None, batch_size=8, shuffle=True, augment=True, cache=True):
        """Creates a TensorFlow dataset from image and label paths."""
        
        # If no labels, create unlabeled dataset
        if label_paths is None:
            tf.print("DEBUG: Creating UNLABELED dataset", output_stream=sys.stderr)
            return self.create_unlabeled_dataset(image_paths, batch_size)

        tf.print("DEBUG: Creating LABELED dataset", output_stream=sys.stderr)
            
        # Convert paths to tensors
        ds = tf.data.Dataset.from_tensor_slices((image_paths, label_paths))
        
        # Shuffle early if enabled
        if shuffle:
            ds = ds.shuffle(buffer_size=len(image_paths), reshuffle_each_iteration=True)
        
        # Map loading and preprocessing
        ds = ds.map(
            lambda x, y: tf.py_function(
                func=lambda img_p, lbl_p: self.preprocess_volume(
                    img_p.numpy().decode('utf-8'),
                    lbl_p.numpy().decode('utf-8')
                ),
                inp=[x, y],
                Tout=[tf.float32, tf.float32]
            ),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        
        # Filter out None values and check shapes
        ds = ds.filter(lambda x, y: tf.logical_and(
            tf.reduce_all(tf.shape(x) > 0),
            tf.reduce_all(tf.shape(y) > 0)
        ))
        
        # Add channel dimension if not present
        def add_channel_dim(x, y):
            x_rank = tf.rank(x)
            y_rank = tf.rank(y)
            
            x = tf.cond(
                tf.equal(x_rank, 3),
                lambda: tf.expand_dims(x, axis=-1),
                lambda: x
            )
            y = tf.cond(
                tf.equal(y_rank, 3),
                lambda: tf.expand_dims(y, axis=-1),
                lambda: y
            )
            return x, y
            
        ds = ds.map(add_channel_dim, num_parallel_calls=tf.data.AUTOTUNE)
        
        # Extract slices using tf.data.Dataset.flat_map
        def extract_slices(volume, label):
            # Get number of slices
            num_slices = tf.shape(volume)[0]
            
            # Create a dataset from the slices
            slice_ds = tf.data.Dataset.from_tensor_slices((
                tf.reshape(volume, [num_slices, self.config.img_size_y, self.config.img_size_x, 1]),
                tf.reshape(label, [num_slices, self.config.img_size_y, self.config.img_size_x, 1])
            ))
            
            return slice_ds
            
        # Flatten the volume into individual slices
        ds = ds.flat_map(extract_slices)
        
        # Cache if enabled (after preprocessing, before augmentation)
        if cache:
            ds = ds.cache()
            
        # Data augmentation if enabled
        if augment:
            ds = ds.map(
                lambda x, y: self.augment_labeled(x, y, training=True),
                num_parallel_calls=tf.data.AUTOTUNE
            )
        
        # Add debug output for label stats before batching
        ds = ds.map(lambda x, y: (
            x,
            tf.py_function(
                func=lambda l: tf.print(
                    "DEBUG Pre-batch Label Stats:",
                    "sum:", tf.reduce_sum(l),
                    "min:", tf.reduce_min(l),
                    "max:", tf.reduce_max(l),
                    "shape:", tf.shape(l),
                    "unique:", tf.unique(tf.reshape(l, [-1])),
                    output_stream=sys.stderr
                ) or l,
                inp=[y],
                Tout=tf.float32
            )
        ))
        
        # Batch the dataset
        ds = ds.batch(batch_size)
        
        # Add debug output for label stats after batching
        ds = ds.map(lambda x, y: (
            x,
            tf.py_function(
                func=lambda l: tf.print(
                    "DEBUG Post-batch Label Stats:",
                    "sum:", tf.reduce_sum(l),
                    "min:", tf.reduce_min(l),
                    "max:", tf.reduce_max(l),
                    "shape:", tf.shape(l),
                    "unique:", tf.unique(tf.reshape(l, [-1])),
                    output_stream=sys.stderr
                ) or l,
                inp=[y],
                Tout=tf.float32
            )
        ))
        
        # Prefetch for performance
        ds = ds.prefetch(tf.data.AUTOTUNE)
        
        return ds
        
    @tf.function
    def elastic_deformation(self, image, label=None, alpha=500, sigma=20):
        """Applies elastic deformation using pure TensorFlow operations."""

        def _bilinear_interpolate(image, x, y):
            """Performs bilinear interpolation for given coordinates."""
            x0 = tf.floor(x)
            x1 = x0 + 1
            y0 = tf.floor(y)
            y1 = y0 + 1

            x0 = tf.clip_by_value(x0, 0, tf.cast(self.config.img_size_x - 1, tf.float32))
            x1 = tf.clip_by_value(x1, 0, tf.cast(self.config.img_size_x - 1, tf.float32))
            y0 = tf.clip_by_value(y0, 0, tf.cast(self.config.img_size_y - 1, tf.float32))
            y1 = tf.clip_by_value(y1, 0, tf.cast(self.config.img_size_y - 1, tf.float32))

            x0_int = tf.cast(x0, tf.int32)
            x1_int = tf.cast(x1, tf.int32)
            y0_int = tf.cast(y0, tf.int32)
            y1_int = tf.cast(y1, tf.int32)

            Ia = tf.gather_nd(image, tf.stack([y0_int, x0_int], axis=-1))
            Ib = tf.gather_nd(image, tf.stack([y1_int, x0_int], axis=-1))
            Ic = tf.gather_nd(image, tf.stack([y0_int, x1_int], axis=-1))
            Id = tf.gather_nd(image, tf.stack([y1_int, x1_int], axis=-1))

            wa = (x1 - x) * (y1 - y)
            wb = (x1 - x) * (y - y0)
            wc = (x - x0) * (y1 - y)
            wd = (x - x0) * (y - y0)

            return wa * Ia + wb * Ib + wc * Ic + wd * Id

        image_shape = tf.shape(image)
        dx = tf.random.normal(image_shape, mean=0.0, stddev=sigma)
        dy = tf.random.normal(image_shape, mean=0.0, stddev=sigma)

        x, y = tf.meshgrid(tf.range(image_shape[1]), tf.range(image_shape[0]))
        x = tf.cast(x, tf.float32) + alpha * dx
        y = tf.cast(y, tf.float32) + alpha * dy

        warped_image = _bilinear_interpolate(image, x, y)

        if label is not None:
            warped_label = _bilinear_interpolate(label, x, y)
            return warped_image, warped_label

        return warped_image    

    def _gaussian_kernel(self, sigma, kernel_size=15):
            """Creates a Gaussian kernel for elastic deformation."""
            x = tf.range(-kernel_size // 2 + 1, kernel_size // 2 + 1, dtype=tf.float32)
            g = tf.math.exp(-(x ** 2) / (2 * sigma ** 2))
            g = g / tf.reduce_sum(g)
            return tf.reshape(g, [kernel_size, 1, 1, 1])    
    
    def _load_and_preprocess_mask(self, mask_path):
        """Load and preprocess a mask file."""
        try:
            # Load mask data
            mask = np.load(mask_path)
            # Convert to tensor and ensure float32
            mask = tf.convert_to_tensor(mask, dtype=tf.float32)
            # Add channel dimension
            mask = tf.expand_dims(mask, axis=-1)
            return mask
        except Exception as e:
            logging.error(f"Error loading mask {mask_path}: {e}")
            return None

    def resize_image_slice(self, slice_2d):
        # Ensure slice_2d is [H, W, C]
        if len(slice_2d.shape) == 2: # Grayscale, no channel
            slice_2d = slice_2d[..., tf.newaxis]
        
        # Check if channel dim is missing for grayscale and add it
        if slice_2d.shape[-1] != self.config.num_channels:
             if self.config.num_channels == 1 and len(slice_2d.shape) == 2: # H, W
                 slice_2d = slice_2d[..., tf.newaxis] # H, W, 1
             # Add more sophisticated channel handling if necessary

        resized = tf.image.resize_with_pad(
            slice_2d, self.config.img_size_y, self.config.img_size_x, method=tf.image.ResizeMethod.BILINEAR
        )
        return resized # Shape [target_y, target_x, C]

    def resize_label_slice(self, slice_2d):
        # Ensure slice_2d is [H, W, C]
        if len(slice_2d.shape) == 2: # Grayscale, no channel
            slice_2d = slice_2d[..., tf.newaxis]

        # Check if channel dim is missing for grayscale and add it
        if slice_2d.shape[-1] != 1: # Assuming label is single channel
             if len(slice_2d.shape) == 2: # H, W
                 slice_2d = slice_2d[..., tf.newaxis] # H, W, 1
        
        resized = tf.image.resize_with_pad(
            slice_2d, self.config.img_size_y, self.config.img_size_x, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
        )
        return tf.cast(resized, tf.float32) # Ensure labels are float for loss calculation if needed, or int

    def resize_volume(self, volume):
        # ...existing code...
        # This function might not be directly used if we process slice by slice after loading full volume
        # However, if used, ensure it handles channels correctly
        # For labels, it should use NEAREST_NEIGHBOR if it were to resize whole 3D volume labels
        original_shape = tf.shape(volume) # S, H, W, C
        num_slices = original_shape[0]
        
        resized_slices = []
        for i in tf.range(num_slices):
            current_slice = volume[i, :, :, :]
            if current_slice.shape[-1] != self.config.num_channels: # Ensure correct channels for resize_image_slice
                if self.config.num_channels == 1 and len(current_slice.shape) == 2:
                    current_slice = current_slice[..., tf.newaxis]
                elif current_slice.shape[-1] != self.config.num_channels and len(current_slice.shape) == 3 and current_slice.shape[-1] > 1 and self.config.num_channels == 1:
                     current_slice = current_slice[..., :1] # Take first channel if multi-channel and config expects 1
            
            resized_slice = self.resize_image_slice(current_slice)
            resized_slices.append(resized_slice)
        
        if resized_slices:
            return tf.stack(resized_slices, axis=0)
        else: # Handle empty volume case
            # Return an empty tensor with the correct target shape but 0 slices
            return tf.zeros([0, self.config.img_size_y, self.config.img_size_x, self.config.num_channels], dtype=volume.dtype)


    @tf.function
    def _augment_slice_and_label(self, image_slice, label_slice, seed):
        """Applies identical geometric augmentations to image and label slices."""
        # Ensure image_slice and label_slice are [H, W, C]
        # Stateless random flips
        seed_flip_lr = tf.random.experimental.stateless_split(seed, num=1)[0]
        image_slice = tf.image.stateless_random_flip_left_right(image_slice, seed=seed_flip_lr)
        label_slice = tf.image.stateless_random_flip_left_right(label_slice, seed=seed_flip_lr)

        seed_flip_ud = tf.random.experimental.stateless_split(seed_flip_lr, num=1)[0] # Chain seeds
        image_slice = tf.image.stateless_random_flip_up_down(image_slice, seed=seed_flip_ud)
        label_slice = tf.image.stateless_random_flip_up_down(label_slice, seed=seed_flip_ud)

        # Stateless random 90-degree rotations
        # tf.image.rot90 takes k (number of 90-degree rotations)
        # Generate a random k (0, 1, 2, or 3)
        seed_rot = tf.random.experimental.stateless_split(seed_flip_ud, num=1)[0]
        k_rot = tf.random.stateless_uniform(shape=[], seed=seed_rot, minval=0, maxval=4, dtype=tf.int32)
        image_slice = tf.image.rot90(image_slice, k=k_rot)
        label_slice = tf.image.rot90(label_slice, k=k_rot)
        
        # Optional: Add stateless random zoom here if a robust TF implementation is available/developed
        # For now, focusing on flips and rotations.

        return image_slice, label_slice

    @tf.function
    def _augment_single_image_slice(self, image_slice, strength='weak'):
        """Applies augmentations to a single image slice."""
        # Color augmentations (intensity-based)
        if strength == 'weak':
            image_slice = tf.image.random_brightness(image_slice, max_delta=0.1)
            image_slice = tf.image.random_contrast(image_slice, lower=0.9, upper=1.1)
        elif strength == 'strong':
            image_slice = tf.image.random_brightness(image_slice, max_delta=0.2)
            image_slice = tf.image.random_contrast(image_slice, lower=0.8, upper=1.2)
            
            # Gaussian Noise
            noise = tf.random.normal(shape=tf.shape(image_slice), mean=0.0, stddev=0.05, dtype=image_slice.dtype)
            image_slice = image_slice + noise
            
            # Simple Cutout: zero out a random square patch
            # Ensure image_slice is 4D for stateless_random_crop or manual patch setting
            # For simplicity, let's assume H, W, C and apply cutout manually
            if tf.random.uniform(shape=[]) < 0.3: # Apply cutout with 30% probability
                img_shape = tf.shape(image_slice)
                h, w = img_shape[0], img_shape[1]
                cutout_size_h = tf.cast(tf.cast(h, tf.float32) * 0.25, tf.int32) # 25% of height
                cutout_size_w = tf.cast(tf.cast(w, tf.float32) * 0.25, tf.int32) # 25% of width
                
                # Random top-left corner for the cutout
                offset_h = tf.random.uniform(shape=[], maxval=h - cutout_size_h + 1, dtype=tf.int32)
                offset_w = tf.random.uniform(shape=[], maxval=w - cutout_size_w + 1, dtype=tf.int32)
                
                # Create a mask for the cutout area
                indices_h = tf.range(offset_h, offset_h + cutout_size_h)
                indices_w = tf.range(offset_w, offset_w + cutout_size_w)
                
                # Create a dense tensor of zeros for the patch
                cutout_patch_zeros = tf.zeros([cutout_size_h, cutout_size_w, tf.shape(image_slice)[-1]], dtype=image_slice.dtype)

                # Use tf.tensor_scatter_nd_update to apply the cutout
                # This requires constructing indices for scatter_nd_update
                # A simpler way for a rectangular patch is to construct the full image with the patch
                # For now, a slightly less efficient but direct way:
                updates = tf.zeros([cutout_size_h, cutout_size_w, img_shape[2]], dtype=image_slice.dtype)
                
                # Create indices for scatter update
                idx_h = tf.repeat(tf.range(offset_h, offset_h + cutout_size_h), cutout_size_w)
                idx_w = tf.tile(tf.range(offset_w, offset_w + cutout_size_w), [cutout_size_h])
                indices = tf.stack([idx_h, idx_w], axis=1) # Shape [N_pixels_in_patch, 2]
                
                # For 3D tensor [H, W, C], need to update all channels
                # This is complex with scatter_nd. A mask multiplication is easier:
                mask = tf.ones_like(image_slice, dtype=tf.bool)
                padding_h_before = offset_h
                padding_h_after = h - (offset_h + cutout_size_h)
                padding_w_before = offset_w
                padding_w_after = w - (offset_w + cutout_size_w)

                cutout_area_mask = tf.zeros([cutout_size_h, cutout_size_w, img_shape[2]], dtype=tf.bool)
                
                # Pad the cutout_area_mask to match image_slice shape
                paddings = [
                    [padding_h_before, padding_h_after],
                    [padding_w_before, padding_w_after],
                    [0, 0] # No padding for channels
                ]
                full_mask_for_cutout = tf.pad(cutout_area_mask, paddings, "CONSTANT", constant_values=True)
                image_slice = tf.where(full_mask_for_cutout, image_slice, tf.zeros_like(image_slice))


        image_slice = tf.clip_by_value(image_slice, 0.0, 1.0)
        return image_slice

    # Helper functions for zoom and shift (modified to handle cases where label is None)
    def random_zoom(self, image, label, zoom_range=(0.7, 1.3)):
        # ...existing code...
        # This function is stateful and complex to integrate directly into tf.data.map
        # with stateless requirements for paired augmentation.
        # Consider replacing with tf.image.stateless_random_crop + resize or similar if needed.
        # For now, _augment_slice_and_label uses flips and rotations.
        if label is not None:
            # ...
            return image, label
        else:
            # ...
            return image, None


class DataPipeline:
    def __init__(self, config):
        self.config = config
        self.dataloader = PancreasDataLoader(config)

    def setup_training_data(self, labeled_paths, unlabeled_paths, val_paths, batch_size=8):
        """Sets up all datasets needed for training."""
        labeled_dataset = self.dataloader.create_dataset(
            labeled_paths['images'],
            labeled_paths['labels'],
            batch_size=batch_size,
            shuffle=True,
            augment=True
        )

        unlabeled_dataset = self.dataloader.create_unlabeled_dataset(
            unlabeled_paths['images'],
            batch_size=batch_size
        )

        val_dataset = self.dataloader.create_dataset(
            val_paths['images'],
            val_paths['labels'],
            batch_size=batch_size,
            shuffle=False,
            augment=False
        )

        return {
            'labeled': labeled_dataset,
            'unlabeled': unlabeled_dataset,
            'validation': val_dataset
        }

    def update_unlabeled_dataset(self, pseudo_labeled_data):
        """Updates the unlabeled dataset with new pseudo-labeled data."""
        if not pseudo_labeled_data:
            logging.warning("No pseudo-labeled data provided. Keeping the original unlabeled dataset.")
            return

        all_images = []
        all_pseudo_labels = []
        for images, pseudo_labels in pseudo_labeled_data:
            all_images.append(images)
            all_pseudo_labels.append(pseudo_labels)

        all_images = tf.concat(all_images, axis=0)
        all_pseudo_labels = tf.concat(all_pseudo_labels, axis=0)

        self.unlabeled_dataset = tf.data.Dataset.from_tensor_slices((all_images, all_pseudo_labels))

        self.unlabeled_dataset = self.unlabeled_dataset.map(
            self.preprocess_unlabeled, num_parallel_calls=tf.data.AUTOTUNE
        )
        self.unlabeled_dataset = self.unlabeled_dataset.batch(self.config.batch_size)
        self.unlabeled_dataset = self.unlabeled_dataset.prefetch(tf.data.AUTOTUNE)

    def get_num_batches(self, dataset_type):
        """Returns the number of batches in the specified dataset."""
        if dataset_type == 'labeled':
            dataset = self.labeled_dataset
        elif dataset_type == 'unlabeled':
            dataset = self.unlabeled_dataset
        else:
            raise ValueError(f"Invalid dataset type: {dataset_type}")

        if dataset is None:
            return 0
        else:
            return len(list(dataset))

    def preprocess_unlabeled(self, image, pseudo_label):
        """Preprocesses an unlabeled image and its pseudo-label."""
        image = tf.cast(image, tf.float32)
        pseudo_label = tf.cast(pseudo_label, tf.int32)
        return image, pseudo_label

    def _py_load_preprocess_volume(self, image_path, label_path=None):
        # Wrapper for preprocess_volume to be used in tf.py_function
        # preprocess_volume now returns image_data, label_data
        image_data, label_data = self.dataloader.preprocess_volume(image_path, label_path)
        
        if image_data is None: # Error case from preprocess_volume
            # Return appropriately shaped empty tensors or handle error
            # For supervised, need two outputs. For unlabeled, one.
            # This structure assumes preprocess_volume always returns two, label_data can be None.
            dummy_img_shape = [0, self.config.img_size_y, self.config.img_size_x, self.config.num_channels]
            dummy_lbl_shape = [0, self.config.img_size_y, self.config.img_size_x, 1] # Assuming label is single channel
            
            if label_path is not None: # Supervised case expects two outputs
                 return np.zeros(dummy_img_shape, dtype=np.float32), np.zeros(dummy_lbl_shape, dtype=np.int32)
            else: # Unlabeled case expects one output (image)
                 return np.zeros(dummy_img_shape, dtype=np.float32)


        if label_path is not None:
            return image_data.astype(np.float32), label_data.astype(np.int32)
        else:
            return image_data.astype(np.float32)


    def _parse_fn_supervised_train(self, image_path, label_path):
        # This function will be used with tf.data.Dataset.from_generator
        # It should yield individual 2D slices: (image_slice, label_slice)
        
        # Use tf.py_function to call the numpy-based loading logic
        image_vol, label_vol = tf.py_function(
            self._py_load_preprocess_volume,
            [image_path, label_path],
            [tf.float32, tf.int32] # Output types
        )

        # Set shapes if possible, helps TF optimize. Actual shape might vary per volume.
        # image_vol.set_shape([None, None, None, self.config.num_channels])
        # label_vol.set_shape([None, None, None, 1]) # Assuming label is single channel

        num_slices = tf.shape(image_vol)[0]
        for i in tf.range(num_slices):
            img_slice = image_vol[i]
            lbl_slice = label_vol[i]
            
            # Resize slices
            img_slice_resized = self.dataloader.resize_image_slice(img_slice)
            lbl_slice_resized = self.dataloader.resize_label_slice(lbl_slice)
            
            # Ensure correct shapes for augmentation
            img_slice_resized.set_shape([self.config.img_size_y, self.config.img_size_x, self.config.num_channels])
            lbl_slice_resized.set_shape([self.config.img_size_y, self.config.img_size_x, 1])

            yield img_slice_resized, lbl_slice_resized
            
    def _parse_fn_unlabeled_train(self, image_path):
        # This function will be used with tf.data.Dataset.from_generator
        # It should yield individual 2D image slices
        image_vol = tf.py_function(
            self._py_load_preprocess_volume,
            [image_path, None], # Pass None for label_path
            tf.float32 # Output type for image
        )
        # image_vol.set_shape([None, None, None, self.config.num_channels])

        num_slices = tf.shape(image_vol)[0]
        for i in tf.range(num_slices):
            img_slice = image_vol[i]
            img_slice_resized = self.dataloader.resize_image_slice(img_slice)
            img_slice_resized.set_shape([self.config.img_size_y, self.config.img_size_x, self.config.num_channels])
            yield img_slice_resized

    def _parse_fn_supervised_eval(self, image_path, label_path):
        # Similar to _parse_fn_supervised_train but without augmentation in mapping
        # Yields (image_slice_resized, label_slice_resized)
        image_vol, label_vol = tf.py_function(
            self._py_load_preprocess_volume,
            [image_path, label_path],
            [tf.float32, tf.int32]
        )
        num_slices = tf.shape(image_vol)[0]
        for i in tf.range(num_slices):
            img_slice = image_vol[i]
            lbl_slice = label_vol[i]
            img_slice_resized = self.dataloader.resize_image_slice(img_slice)
            lbl_slice_resized = self.dataloader.resize_label_slice(lbl_slice)
            img_slice_resized.set_shape([self.config.img_size_y, self.config.img_size_x, self.config.num_channels])
            lbl_slice_resized.set_shape([self.config.img_size_y, self.config.img_size_x, 1])
            yield img_slice_resized, lbl_slice_resized
            
    def build_labeled_dataset(self, image_paths, label_paths, batch_size, is_training=True):
        if not image_paths or not label_paths:
            return None # Or an empty dataset

        dataset = tf.data.Dataset.from_tensor_slices((image_paths, label_paths))
        
        # Shuffle file paths before loading and slicing
        if is_training:
            dataset = dataset.shuffle(buffer_size=len(image_paths))

        # Use interleave to process files one by one and flatten their slices into the dataset
        dataset = dataset.interleave(
            lambda img_path, lbl_path: tf.data.Dataset.from_generator(
                self._parse_fn_supervised_train,
                args=(img_path, lbl_path),
                output_signature=(
                    tf.TensorSpec(shape=(self.config.img_size_y, self.config.img_size_x, self.config.num_channels), dtype=tf.float32),
                    tf.TensorSpec(shape=(self.config.img_size_y, self.config.img_size_x, 1), dtype=tf.float32) # Labels are float32 after resize
                )
            ),
            cycle_length=tf.data.AUTOTUNE, # Number of parallel file processing
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=not is_training
        )
        
        # Filter out potential None slices from loading errors if preprocess_volume was modified to return them
        # dataset = dataset.filter(lambda img, lbl: tf.reduce_all(tf.shape(img) > 0))


        if is_training:
            # Augment after slices are generated
            # Need a seed for stateless augmentations
            # Create a seed dataset to zip with the main dataset
            seed_counter = tf.data.experimental.Counter()
            seed_dataset = tf.data.Dataset.zip((seed_counter, seed_counter)).map(lambda s1, s2: (s1,s2)) #dummy, just to get seeds

            dataset = tf.data.Dataset.zip((dataset, seed_dataset))
            dataset = dataset.map(
                lambda x, seed_pair_tf: self.dataloader._augment_slice_and_label(x[0], x[1], seed=(tf.cast(seed_pair_tf[0], tf.int32), tf.cast(seed_pair_tf[1], tf.int32))),
                num_parallel_calls=AUTOTUNE
            )
            dataset = dataset.shuffle(buffer_size=1000) # Shuffle slices

        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(buffer_size=AUTOTUNE)
        return dataset

    def _augment_for_mean_teacher(self, image_slice):
        # image_slice is an unaugmented, resized slice
        student_input = self.dataloader._augment_single_image_slice(image_slice, strength='strong')
        teacher_input = self.dataloader._augment_single_image_slice(image_slice, strength='weak')
        # Ensure shapes are set
        student_input.set_shape([self.config.img_size_y, self.config.img_size_x, self.config.num_channels])
        teacher_input.set_shape([self.config.img_size_y, self.config.img_size_x, self.config.num_channels])
        return student_input, teacher_input

    def build_unlabeled_dataset_for_mean_teacher(self, image_paths, batch_size, is_training=True):
        if not image_paths:
            return None

        dataset = tf.data.Dataset.from_tensor_slices(image_paths)

        if is_training:
            dataset = dataset.shuffle(buffer_size=len(image_paths)) # Shuffle file paths

        dataset = dataset.interleave(
            lambda img_path: tf.data.Dataset.from_generator(
                self._parse_fn_unlabeled_train,
                args=(img_path,),
                output_signature=tf.TensorSpec(shape=(self.config.img_size_y, self.config.img_size_x, self.config.num_channels), dtype=tf.float32)
            ),
            cycle_length=tf.data.AUTOTUNE,
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=not is_training
        )
        
        # dataset = dataset.filter(lambda img: tf.reduce_all(tf.shape(img) > 0))


        if is_training:
            dataset = dataset.map(self._augment_for_mean_teacher, num_parallel_calls=AUTOTUNE)
            dataset = dataset.shuffle(buffer_size=1000) # Shuffle augmented slices

        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(buffer_size=AUTOTUNE)
        return dataset

    def build_validation_dataset(self, image_paths, label_paths, batch_size):
        # Similar to build_labeled_dataset but uses _parse_fn_supervised_eval (no augmentation)
        if not image_paths or not label_paths:
            return None

        dataset = tf.data.Dataset.from_tensor_slices((image_paths, label_paths))
        
        # No shuffling of file paths for validation
        dataset = dataset.interleave(
            lambda img_path, lbl_path: tf.data.Dataset.from_generator(
                self._parse_fn_supervised_eval, # Use eval parser
                args=(img_path, lbl_path),
                output_signature=(
                    tf.TensorSpec(shape=(self.config.img_size_y, self.config.img_size_x, self.config.num_channels), dtype=tf.float32),
                    tf.TensorSpec(shape=(self.config.img_size_y, self.config.img_size_x, 1), dtype=tf.float32)
                )
            ),
            cycle_length=tf.data.AUTOTUNE,
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=True # Deterministic for validation
        )
        # dataset = dataset.filter(lambda img, lbl: tf.reduce_all(tf.shape(img) > 0))

        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(buffer_size=AUTOTUNE)
        return dataset

    def build_unlabeled_dataset(self, image_paths, batch_size, is_training=True, augment_fn=None):
        # Generic unlabeled dataset builder, can be used by other SSL methods if needed
        # For Mean Teacher, build_unlabeled_dataset_for_mean_teacher is more specific
        if not image_paths:
            return None

        dataset = tf.data.Dataset.from_tensor_slices(image_paths)

        if is_training:
            dataset = dataset.shuffle(buffer_size=len(image_paths))

        dataset = dataset.interleave(
            lambda img_path: tf.data.Dataset.from_generator(
                self._parse_fn_unlabeled_train,
                args=(img_path,),
                output_signature=tf.TensorSpec(shape=(self.config.img_size_y, self.config.img_size_x, self.config.num_channels), dtype=tf.float32)
            ),
            cycle_length=tf.data.AUTOTUNE,
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=not is_training
        )
        
        # dataset = dataset.filter(lambda img: tf.reduce_all(tf.shape(img) > 0))


        if is_training and augment_fn: # If a generic augmentation function is provided
            dataset = dataset.map(augment_fn, num_parallel_calls=AUTOTUNE)
        elif is_training: # Default single augmentation if no specific one for MT
             dataset = dataset.map(lambda x: self.dataloader._augment_single_image_slice(x, strength='weak'), num_parallel_calls=AUTOTUNE)


        if is_training:
            dataset = dataset.shuffle(buffer_size=1000)

        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(buffer_size=AUTOTUNE)
        return dataset