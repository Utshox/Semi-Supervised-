import tensorflow as tf
import numpy as np
from pathlib import Path
import nibabel as nib  # You might not need this if you are not using .nii files
import time
import logging

class PancreasDataLoader:
    def __init__(self, config):
        self.config = config
        self.target_size = (config.img_size_x, config.img_size_y)

    def preprocess_volume(self, image_path, label_path=None):
        """Loads and preprocesses a single volume."""
        try:
            # Load image
            img_data = np.load(str(image_path))

            # Normalize intensity between 0 and 1
            p1, p99 = np.percentile(img_data, (1, 99))
            img_data = np.clip(img_data, p1, p99)
            img_data = (img_data - p1) / (p99 - p1)

            # Process label if provided
            if label_path:
                try:
                    label_data = np.load(str(label_path))
                    label_data = (label_data > 0).astype(np.float32)
                    print(f"Successfully loaded and preprocessed {image_path} and {label_path}")
                    return img_data, label_data
                except Exception as e:
                    logging.error(f"Error processing label {label_path}: {e}")
                    return None, None
            else:
                print(f"Successfully loaded and preprocessed {image_path}")
                return img_data

        except FileNotFoundError:
            logging.error(f"File not found: {image_path if label_path is None else label_path}")
            return None if label_path is None else (None, None)
        except Exception as e:
            logging.error(f"Error processing {image_path}: {e}")
            return None if label_path is None else (None, None)

    def resize_slice(self, slice_2d):
        """Resizes a single 2D slice."""
        if slice_2d is None:
            return tf.zeros([self.config.img_size_x, self.config.img_size_y, 1], dtype=tf.float32)

        # Add channel dimension if needed
        if len(slice_2d.shape) == 2:
            slice_2d = slice_2d[..., tf.newaxis]

        try:
            resized = tf.image.resize(
                tf.convert_to_tensor(slice_2d, dtype=tf.float32),
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
            if len(volume.shape) == 2:
                return self.resize_slice(volume)

            resized_slices = []
            for i in range(volume.shape[2]):
                slice_2d = volume[:, :, i]
                resized = self.resize_slice(slice_2d)
                if resized is not None:
                    resized_slices.append(resized)

            if not resized_slices:
                return tf.zeros([self.config.img_size_x, self.config.img_size_y, 1], dtype=tf.float32)

            resized_volume = tf.stack(resized_slices, axis=-1)
            return resized_volume
        except Exception as e:
            logging.error(f"Volume resize error: {str(e)}")
            return tf.zeros([self.config.img_size_x, self.config.img_size_y, 1], dtype=tf.float32)    
        

    @tf.function
    def augment(self, image, label=None, training=True):
        """Data augmentation function."""
        if not training:
            if label is not None:
                return image, label
            return image

        # Ensure image is a tensor
        image = tf.convert_to_tensor(image)
        if label is not None:
            label = tf.convert_to_tensor(label)

        # Random rotation (using tf.image instead of tfa)
        if tf.random.uniform(()) > 0.3: # Apply with 70% probability
            angle = tf.random.uniform((), minval=-20, maxval=20) * np.pi / 180  # Increased rotation range
            image = tf.image.rot90(image, k=tf.cast(angle / (np.pi / 2), tf.int32))
            if label is not None:
                label = tf.image.rot90(label, k=tf.cast(angle / (np.pi / 2), tf.int32))

        # Random flip
        if tf.random.uniform(()) > 0.5:
            image = tf.image.flip_left_right(image)
            if label is not None:
                label = tf.image.flip_left_right(label)

        # Random brightness/contrast (only apply to image)
        if tf.random.uniform(()) > 0.3:  # Apply with 70% probability
            image = tf.image.random_brightness(image, max_delta=0.2)
            image = tf.image.random_contrast(image, lower=0.7, upper=1.3)

        # Random zoom and shift (only apply if label is not None)
        if label is not None:
            if tf.random.uniform(()) > 0.3:  # Apply with 70% probability
                image, label = self.random_zoom(image, label, zoom_range=(0.7, 1.3))  # Increased zoom range

            if tf.random.uniform(()) > 0.3:  # Apply with 70% probability
                image, label = self.random_shift(image, label, shift_range=0.2)  # Increased shift range

        # Ensure values are in valid range
        image = tf.clip_by_value(image, 0.0, 1.0)

        if label is not None:
            return image, label
        return image

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

    def create_dataset(self, image_paths, label_paths=None, batch_size=8, shuffle=True, augment=True, cache=True):
        """Creates a TensorFlow dataset from image and label paths with caching."""

        def load_and_preprocess(image_path, label_path=None):
            """Loads and preprocesses a single image and label pair."""
            image_path_str = image_path.numpy().decode()

            # Create cache path
            cache_dir = Path("preprocessed_cache")
            cache_dir.mkdir(exist_ok=True)

            image_cache_path = cache_dir / f"{Path(image_path_str).stem}_processed.npy"

            if cache and image_cache_path.exists():
                # Load from cache
                processed_images = np.load(str(image_cache_path))
                if processed_images.shape[1:] != (self.config.img_size_x, self.config.img_size_y, 1):
                  processed_images = None
            else:
                # Process image
                image_vol = self.preprocess_volume(image_path_str)
                if image_vol is None:
                    return None if label_path is None else (None, None)

                # Process slices
                processed_images = []
                for i in range(image_vol.shape[2]):
                    img_slice = image_vol[:, :, i]
                    img_slice = tf.image.resize(img_slice[..., tf.newaxis],
                                                [self.config.img_size_x, self.config.img_size_y])
                    img_slice = tf.cast(img_slice, tf.float32)
                    img_slice = (img_slice - tf.reduce_min(img_slice)) / (
                        tf.reduce_max(img_slice) - tf.reduce_min(img_slice) + 1e-6)
                    processed_images.append(img_slice)

                processed_images = np.stack(processed_images)

                if cache:
                    # Save to cache
                    np.save(str(image_cache_path), processed_images)

            if label_path is not None:
                label_path_str = label_path.numpy().decode()
                label_cache_path = cache_dir / f"{Path(label_path_str).stem}_processed.npy"

                if cache and label_cache_path.exists():
                    # Load from cache
                    processed_labels = np.load(str(label_cache_path))
                    if processed_labels.shape[1:] != (self.config.img_size_x, self.config.img_size_y, self.config.num_classes):
                      processed_labels = None
                else:
                    # Process label
                    _, label_vol = self.preprocess_volume(image_path_str, label_path_str)
                    if label_vol is None:
                        return None, None

                    # Process label slices
                    processed_labels = []
                    for i in range(label_vol.shape[2]):
                        label_slice = label_vol[:, :, i]
                        label_slice = tf.image.resize(label_slice[..., tf.newaxis],
                                                     [self.config.img_size_x, self.config.img_size_y],
                                                     method='nearest')
                        label_slice = tf.cast(label_slice > 0, tf.float32)
                        processed_labels.append(label_slice)

                    processed_labels = np.stack(processed_labels)
                    processed_labels = tf.one_hot(tf.cast(processed_labels[..., 0], tf.int32),
                                                 depth=self.config.num_classes)

                    if cache:
                        # Save to cache
                        np.save(str(label_cache_path), processed_labels)

                return processed_images, processed_labels

            return processed_images

        # Create dataset
        if label_paths is not None:
            dataset = tf.data.Dataset.from_tensor_slices((image_paths, label_paths))
            dataset = dataset.map(
                lambda x, y: tf.py_function(
                    load_and_preprocess, [x, y], [tf.float32, tf.float32]
                ),
                num_parallel_calls=tf.data.AUTOTUNE
            )
        else:
            dataset = tf.data.Dataset.from_tensor_slices(image_paths)
            dataset = dataset.map(
                lambda x: tf.py_function(
                    load_and_preprocess, [x], [tf.float32]
                ),
                num_parallel_calls=tf.data.AUTOTUNE
            )

        # Filter out None values
        dataset = dataset.filter(lambda *args: all(arg is not None for arg in args))

        # Cache the dataset after preprocessing but before augmentation
        if cache:
            dataset = dataset.cache()

        # Unbatch to get individual slices
        dataset = dataset.unbatch()

        # Explicitly set shape after unbatching
        def set_shape(*args):
            if len(args) == 1:
                x = args[0]
                x.set_shape([self.config.img_size_x, self.config.img_size_y, 1])
                return x
            else:
                x, y = args
                x.set_shape([self.config.img_size_x, self.config.img_size_y, 1])
                y.set_shape([self.config.img_size_x, self.config.img_size_y, self.config.num_classes])
                return x, y

        dataset = dataset.map(set_shape, num_parallel_calls=tf.data.AUTOTUNE)

        # Augmentation for training
        if augment and label_paths:
            dataset = dataset.map(
                lambda x, y: self.augment(x, y, training=True),
                num_parallel_calls=tf.data.AUTOTUNE
            )

        # Shuffle if requested
        if shuffle:
            dataset = dataset.shuffle(buffer_size=1000)

        # Resize to target size and add channel dimension (after augmentation)
        def resize_and_add_channel(image, label=None):
            image = tf.image.resize(image, [self.config.img_size_x, self.config.img_size_y])
            if label is not None:
                label = tf.image.resize(label, [self.config.img_size_x, self.config.img_size_y], method='nearest')
                return image, label
            else:
                return image

        if label_paths is not None:
            dataset = dataset.map(
                lambda x, y: resize_and_add_channel(x, y),
                num_parallel_calls=tf.data.AUTOTUNE
            )
        else:
            dataset = dataset.map(
                lambda x: resize_and_add_channel(x),
                num_parallel_calls=tf.data.AUTOTUNE
            )

        # Batch the dataset
        dataset = dataset.batch(batch_size)

        # Prefetch for better performance
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        return dataset
    
    def create_unlabeled_dataset(self, image_paths, batch_size=8):
        """Creates dataset from unlabeled images."""
        dataset = tf.data.Dataset.from_tensor_slices(image_paths)

        def load_and_preprocess(image_path):
            """Load and preprocess a single unlabeled image"""
            try:
                # Convert path to string and load .npy file
                image_path_str = image_path.numpy().decode('utf-8')
                img_data = np.load(image_path_str)

                # Process each slice
                processed_images = []
                for i in range(img_data.shape[2]):
                    img_slice = img_data[:, :, i]

                    # Add channel dimension and resize
                    img_slice = tf.image.resize(img_slice[..., tf.newaxis],
                                                [self.config.img_size_x, self.config.img_size_y])

                    # Normalize slice
                    img_slice = tf.cast(img_slice, tf.float32)
                    img_slice = (img_slice - tf.reduce_min(img_slice)) / (
                        tf.reduce_max(img_slice) - tf.reduce_min(img_slice) + 1e-6)

                    processed_images.append(img_slice)

                # Stack processed slices
                images = tf.stack(processed_images)

                # Set the shape explicitly after stacking
                images.set_shape([img_data.shape[2], self.config.img_size_x, self.config.img_size_y, 1])

                return images

            except Exception as e:
                logging.error(f"Error processing {image_path_str}: {e}")
                return None

        # Map loading function
        dataset = dataset.map(
            lambda x: tf.py_function(
                func=load_and_preprocess,
                inp=[x],
                Tout=tf.float32
            ),
            num_parallel_calls=tf.data.AUTOTUNE
        )

        # Filter out None values
        dataset = dataset.filter(lambda x: x is not None)

        # Unbatch to get individual slices
        dataset = dataset.unbatch()

        # Set shape after unbatching
        dataset = dataset.map(lambda x: tf.ensure_shape(x, [self.config.img_size_x, self.config.img_size_y, 1]), num_parallel_calls=tf.data.AUTOTUNE)

        # Apply augmentation to each slice (without label)
        dataset = dataset.map(
            lambda x: self.augment(x, training=True),  # Removed label argument
            num_parallel_calls=tf.data.AUTOTUNE
        )

        # Resize to target size and add channel dimension (after augmentation)
        def resize_and_add_channel(image):  # Removed label argument
            image = tf.image.resize(image, [self.config.img_size_x, self.config.img_size_y])
            return image

        dataset = dataset.map(
            lambda x: resize_and_add_channel(x),
            num_parallel_calls=tf.data.AUTOTUNE
        )

        # Batch and prefetch
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        return dataset
    
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

            # Get weights for bilinear interpolation
            wa = (x1 - x) * (y1 - y)
            wb = (x1 - x) * (y - y0)
            wc = (x - x0) * (y1 - y)
            wd = (x - x0) * (y - y0)

            # Add batch dimension if necessary
            wa = tf.expand_dims(wa, -1)
            wb = tf.expand_dims(wb, -1)
            wc = tf.expand_dims(wc, -1)
            wd = tf.expand_dims(wd, -1)

            # Gather pixel values and apply weights
            batch_size = tf.shape(image)[0]
            batch_idx = tf.range(batch_size)
            batch_idx = tf.reshape(batch_idx, [batch_size, 1, 1])
            batch_idx = tf.tile(batch_idx, [1, tf.shape(x0_int)[1], tf.shape(x0_int)[2]])

            idx_a = tf.stack([batch_idx, y0_int, x0_int], axis=-1)
            idx_b = tf.stack([batch_idx, y1_int, x0_int], axis=-1)
            idx_c = tf.stack([batch_idx, y0_int, x1_int], axis=-1)
            idx_d = tf.stack([batch_idx, y1_int, x1_int], axis=-1)

            Ia = tf.gather_nd(image, idx_a)
            Ib = tf.gather_nd(image, idx_b)
            Ic = tf.gather_nd(image, idx_c)
            Id = tf.gather_nd(image, idx_d)

            return wa * Ia + wb * Ib + wc * Ic + wd * Id

        # Generate random displacement fields
        batch_size = tf.shape(image)[0]
        dx = tf.random.normal([batch_size, self.config.img_size_x, self.config.img_size_y],
                              mean=0, stddev=sigma)
        dy = tf.random.normal([batch_size, self.config.img_size_x, self.config.img_size_y],
                              mean=0, stddev=sigma)

        # Create coordinate grid
        x_mesh, y_mesh = tf.meshgrid(
            tf.range(self.config.img_size_x, dtype=tf.float32),
            tf.range(self.config.img_size_y, dtype=tf.float32)
        )

        # Add batch dimension and displacement
        x_mesh = tf.tile(tf.expand_dims(x_mesh, 0), [batch_size, 1, 1])
        y_mesh = tf.tile(tf.expand_dims(y_mesh, 0), [batch_size, 1, 1])

        # Apply displacement
        x_new = x_mesh + alpha * dx
        y_new = y_mesh + alpha * dy

        # Interpolate
        warped_image = _bilinear_interpolate(image, x_new, y_new)

        if label is not None:
            # Use nearest neighbor interpolation for labels
            x_new_int = tf.cast(tf.round(x_new), tf.int32)
            y_new_int = tf.cast(tf.round(y_new), tf.int32)
            x_new_int = tf.clip_by_value(x_new_int, 0, self.config.img_size_x - 1)
            y_new_int = tf.clip_by_value(y_new_int, 0, self.config.img_size_y - 1)

            batch_idx = tf.range(batch_size)
            batch_idx = tf.reshape(batch_idx, [batch_size, 1, 1])
            batch_idx = tf.tile(batch_idx, [1, tf.shape(x_new_int)[1], tf.shape(x_new_int)[2]])

            idx = tf.stack([batch_idx, y_new_int, x_new_int], axis=-1)
            warped_label = tf.gather_nd(label, idx)

            return warped_image, warped_label

        return warped_image    

    def _gaussian_kernel(self, sigma, kernel_size=15):
            """Creates a Gaussian kernel for elastic deformation."""
            x = tf.range(-kernel_size // 2 + 1, kernel_size // 2 + 1, dtype=tf.float32)
            g = tf.math.exp(-(x ** 2) / (2 * sigma ** 2))
            g = g / tf.reduce_sum(g)
            return tf.reshape(g, [kernel_size, 1, 1, 1])    
    
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


# import tensorflow as tf
# import numpy as np
# from pathlib import Path
# import nibabel as nib
# #import tensorflow_addons as tfa
# import time
# import logging

# class PancreasDataLoader:
#     def __init__(self, config):
#         self.config = config
#         self.target_size = (config.img_size_x, config.img_size_y)

#     def preprocess_volume(self, image_path, label_path=None):
#         """Loads and preprocesses a single volume."""
#         try:
#             # Load image
#             img_data = np.load(str(image_path))

#             # Normalize intensity between 0 and 1
#             p1, p99 = np.percentile(img_data, (1, 99))
#             img_data = np.clip(img_data, p1, p99)
#             img_data = (img_data - p1) / (p99 - p1)

#             # Process label if provided
#             if label_path:
#                 try:
#                     label_data = np.load(str(label_path))
#                     label_data = (label_data > 0).astype(np.float32)
#                     print(f"Successfully loaded and preprocessed {image_path} and {label_path}")
#                     return img_data, label_data
#                 except Exception as e:
#                     logging.error(f"Error processing label {label_path}: {e}")
#                     return None, None
#             else:
#                 print(f"Successfully loaded and preprocessed {image_path}")
#                 return img_data
                
#         except FileNotFoundError:
#             logging.error(f"File not found: {image_path if label_path is None else label_path}")
#             return None if label_path is None else (None, None)
#         except Exception as e:
#             logging.error(f"Error processing {image_path}: {e}")
#             return None if label_path is None else (None, None)

#     def resize_slice(self, slice_2d):
#         """Resizes a single 2D slice."""
#         if slice_2d is None:
#             return tf.zeros([self.config.img_size_x, self.config.img_size_y, 1], dtype=tf.float32)

#         # Add channel dimension if needed
#         if len(slice_2d.shape) == 2:
#             slice_2d = slice_2d[..., tf.newaxis]

#         try:
#             resized = tf.image.resize(
#                 tf.convert_to_tensor(slice_2d, dtype=tf.float32),
#                 [self.config.img_size_x, self.config.img_size_y],
#                 method='bilinear'
#             )
#             resized.set_shape([self.config.img_size_x, self.config.img_size_y, 1])
#             return resized
#         except Exception as e:
#             logging.error(f"Resize error: {str(e)}")
#             return tf.zeros([self.config.img_size_x, self.config.img_size_y, 1], dtype=tf.float32)

#     def resize_volume(self, volume):
#         """Resizes 3D volume slice by slice."""
#         if volume is None:
#             return tf.zeros([self.config.img_size_x, self.config.img_size_y, 1], dtype=tf.float32)

#         try:
#             if len(volume.shape) == 2:
#                 return self.resize_slice(volume)

#             resized_slices = []
#             for i in range(volume.shape[2]):
#                 slice_2d = volume[:, :, i]
#                 resized = self.resize_slice(slice_2d)
#                 if resized is not None:
#                     resized_slices.append(resized)

#             if not resized_slices:
#                 return tf.zeros([self.config.img_size_x, self.config.img_size_y, 1], dtype=tf.float32)

#             resized_volume = tf.stack(resized_slices, axis=-1)
#             return resized_volume
#         except Exception as e:
#             logging.error(f"Volume resize error: {str(e)}")
#             return tf.zeros([self.config.img_size_x, self.config.img_size_y, 1], dtype=tf.float32)
    

#     @tf.function
#     def augment(self, image, label=None, training=True):
#         """Data augmentation function."""
#         if not training:
#             if label is not None:
#                 return image, label
#             return image

#         # Ensure image is a tensor
#         image = tf.convert_to_tensor(image)
        
#         # Random rotation (using tf.image instead of tfa)
#         if tf.random.uniform(()) > 0.5:
#             angle = tf.random.uniform((), minval=-20, maxval=20) * np.pi / 180
#             image = tf.image.rot90(image, k=tf.cast(angle / (np.pi/2), tf.int32))
#             if label is not None:
#                 label = tf.image.rot90(label, k=tf.cast(angle / (np.pi/2), tf.int32))

#         # Random flip
#         if tf.random.uniform(()) > 0.5:
#             image = tf.image.flip_left_right(image)
#             if label is not None:
#                 label = tf.image.flip_left_right(label)

#         # Random brightness/contrast (only apply to image)
#         image = tf.image.random_brightness(image, max_delta=0.1)
#         image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
        
#         # Ensure values are in valid range
#         image = tf.clip_by_value(image, 0.0, 1.0)

#         if label is not None:
#             return image, label
#         return image
#     def create_dataset(self, image_paths, label_paths=None, batch_size=8, shuffle=True, augment=True, cache=True):
#         """Creates a TensorFlow dataset from image and label paths with caching."""

#         def load_and_preprocess(image_path, label_path=None):
#             """Loads and preprocesses a single image and label pair."""
#             image_path_str = image_path.numpy().decode()
            
#             # Create cache path
#             cache_dir = Path("preprocessed_cache")
#             cache_dir.mkdir(exist_ok=True)
            
#             image_cache_path = cache_dir / f"{Path(image_path_str).stem}_processed.npy"
            
#             if cache and image_cache_path.exists():
#                 # Load from cache
#                 processed_images = np.load(str(image_cache_path))
#             else:
#                 # Process image
#                 image_vol = self.preprocess_volume(image_path_str)
#                 if image_vol is None:
#                     return None if label_path is None else (None, None)
                    
#                 # Process slices
#                 processed_images = []
#                 for i in range(image_vol.shape[2]):
#                     img_slice = image_vol[:, :, i]
#                     img_slice = tf.image.resize(img_slice[..., tf.newaxis],
#                                               [self.config.img_size_x, self.config.img_size_y])
#                     img_slice = tf.cast(img_slice, tf.float32)
#                     img_slice = (img_slice - tf.reduce_min(img_slice)) / (
#                         tf.reduce_max(img_slice) - tf.reduce_min(img_slice) + 1e-6)
#                     processed_images.append(img_slice)
                
#                 processed_images = np.stack(processed_images)
                
#                 if cache:
#                     # Save to cache
#                     np.save(str(image_cache_path), processed_images)

#             if label_path is not None:
#                 label_path_str = label_path.numpy().decode()
#                 label_cache_path = cache_dir / f"{Path(label_path_str).stem}_processed.npy"
                
#                 if cache and label_cache_path.exists():
#                     # Load from cache
#                     processed_labels = np.load(str(label_cache_path))
#                 else:
#                     # Process label
#                     _, label_vol = self.preprocess_volume(image_path_str, label_path_str)
#                     if label_vol is None:
#                         return None, None
                    
#                     # Process label slices
#                     processed_labels = []
#                     for i in range(label_vol.shape[2]):
#                         label_slice = label_vol[:, :, i]
#                         label_slice = tf.image.resize(label_slice[..., tf.newaxis],
#                                                     [self.config.img_size_x, self.config.img_size_y],
#                                                     method='nearest')
#                         label_slice = tf.cast(label_slice > 0, tf.float32)
#                         processed_labels.append(label_slice)
                    
#                     processed_labels = np.stack(processed_labels)
#                     processed_labels = tf.one_hot(tf.cast(processed_labels[..., 0], tf.int32), 
#                                               depth=self.config.num_classes)
                    
#                     if cache:
#                         # Save to cache
#                         np.save(str(label_cache_path), processed_labels)
                
#                 return processed_images, processed_labels
            
#             return processed_images

#         # Create dataset
#         if label_paths is not None:
#             dataset = tf.data.Dataset.from_tensor_slices((image_paths, label_paths))
#             dataset = dataset.map(
#                 lambda x, y: tf.py_function(
#                     load_and_preprocess, [x, y], [tf.float32, tf.float32]
#                 ),
#                 num_parallel_calls=tf.data.AUTOTUNE
#             )
#         else:
#             dataset = tf.data.Dataset.from_tensor_slices(image_paths)
#             dataset = dataset.map(
#                 lambda x: tf.py_function(
#                     load_and_preprocess, [x], [tf.float32]
#                 ),
#                 num_parallel_calls=tf.data.AUTOTUNE
#             )

#         # Filter out None values
#         dataset = dataset.filter(lambda *args: all(arg is not None for arg in args))

#         # Cache the dataset after preprocessing but before augmentation
#         if cache:
#             dataset = dataset.cache()

#         # Unbatch to get individual slices
#         dataset = dataset.unbatch()

#         # Augmentation for training
#         if augment and label_paths:
#             dataset = dataset.map(
#                 lambda x, y: self.augment(x, y, training=True),
#                 num_parallel_calls=tf.data.AUTOTUNE
#             )

#         # Shuffle if requested
#         if shuffle:
#             dataset = dataset.shuffle(buffer_size=1000)

#         # Batch the dataset
#         dataset = dataset.batch(batch_size)

#         # Prefetch for better performance
#         dataset = dataset.prefetch(tf.data.AUTOTUNE)

#         return dataset

#     def create_unlabeled_dataset(self, image_paths, batch_size=8):
#         """Creates dataset from unlabeled images."""
#         dataset = tf.data.Dataset.from_tensor_slices(image_paths)

#         def load_and_preprocess(image_path):
#             """Load and preprocess a single unlabeled image"""
#             try:
#                 # Convert path to string and load .npy file
#                 image_path_str = image_path.numpy().decode('utf-8')
#                 img_data = np.load(image_path_str)
                
#                 # Process each slice
#                 processed_images = []
#                 for i in range(img_data.shape[2]):
#                     img_slice = img_data[:, :, i]
                    
#                     # Add channel dimension and resize
#                     img_slice = tf.image.resize(img_slice[..., tf.newaxis],
#                                               [self.config.img_size_x, self.config.img_size_y])
                    
#                     # Normalize slice
#                     img_slice = tf.cast(img_slice, tf.float32)
#                     img_slice = (img_slice - tf.reduce_min(img_slice)) / (
#                         tf.reduce_max(img_slice) - tf.reduce_min(img_slice) + 1e-6)
                    
#                     processed_images.append(img_slice)

#                 # Stack processed slices
#                 images = tf.stack(processed_images)
#                 return images

#             except Exception as e:
#                 logging.error(f"Error processing {image_path_str}: {e}")
#                 return None

#         # Map loading function
#         dataset = dataset.map(
#             lambda x: tf.py_function(
#                 func=load_and_preprocess,
#                 inp=[x],
#                 Tout=tf.float32
#             ),
#             num_parallel_calls=tf.data.AUTOTUNE
#         )

#         # Filter out None values
#         dataset = dataset.filter(lambda x: x is not None)

#         # Unbatch to get individual slices
#         dataset = dataset.unbatch()

#         # Apply augmentation to each slice
#         dataset = dataset.map(
#             lambda x: self.augment(x, training=True),
#             num_parallel_calls=tf.data.AUTOTUNE
#         )

#         # Batch and prefetch
#         dataset = dataset.batch(batch_size)
#         dataset = dataset.prefetch(tf.data.AUTOTUNE)

#         return dataset
# #     def create_pseudo_label_dataset(self, image_paths, pseudo_labels, confidence_threshold=0.9):
# #         """Creates dataset with pseudo-labels."""
# #         dataset = tf.data.Dataset.from_tensor_slices((image_paths, pseudo_labels))

# #         # Filter by confidence (not used in current implementation)
# #         # dataset = dataset.filter(lambda x, y, confidence: confidence >= confidence_threshold)

# #         # Map loading function
# #         dataset = dataset.map(
# #             lambda x, y: tf.py_function(
# #                 self.preprocess_volume,
# #                 [x, y],
# #                 [tf.float32, tf.float32]
# #             ),
# #             num_parallel_calls=tf.data.AUTOTUNE
# #         )

# #         # Filter out None values
# #         dataset = dataset.filter(lambda x, y: x is not None and y is not None)

# #         # Batch and prefetch
# #         dataset = dataset.batch(self.config.batch_size).prefetch(tf.data.AUTOTUNE)

# #         return dataset        

# #     @tf.function
# #     def elastic_deformation(self, image, label=None, alpha=500, sigma=20):
# #         """Applies elastic deformation using pure TensorFlow operations."""
        
# #         def _bilinear_interpolate(image, x, y):
# #             """Performs bilinear interpolation for given coordinates."""
# #             x0 = tf.floor(x)
# #             x1 = x0 + 1
# #             y0 = tf.floor(y)
# #             y1 = y0 + 1

# #             x0 = tf.clip_by_value(x0, 0, tf.cast(self.config.img_size_x - 1, tf.float32))
# #             x1 = tf.clip_by_value(x1, 0, tf.cast(self.config.img_size_x - 1, tf.float32))
# #             y0 = tf.clip_by_value(y0, 0, tf.cast(self.config.img_size_y - 1, tf.float32))
# #             y1 = tf.clip_by_value(y1, 0, tf.cast(self.config.img_size_y - 1, tf.float32))

# #             x0_int = tf.cast(x0, tf.int32)
# #             x1_int = tf.cast(x1, tf.int32)
# #             y0_int = tf.cast(y0, tf.int32)
# #             y1_int = tf.cast(y1, tf.int32)

# #             # Get weights for bilinear interpolation
# #             wa = (x1 - x) * (y1 - y)
# #             wb = (x1 - x) * (y - y0)
# #             wc = (x - x0) * (y1 - y)
# #             wd = (x - x0) * (y - y0)

# #             # Add batch dimension if necessary
# #             wa = tf.expand_dims(wa, -1)
# #             wb = tf.expand_dims(wb, -1)
# #             wc = tf.expand_dims(wc, -1)
# #             wd = tf.expand_dims(wd, -1)

# #             # Gather pixel values and apply weights
# #             batch_size = tf.shape(image)[0]
# #             batch_idx = tf.range(batch_size)
# #             batch_idx = tf.reshape(batch_idx, [batch_size, 1, 1])
# #             batch_idx = tf.tile(batch_idx, [1, tf.shape(x0_int)[1], tf.shape(x0_int)[2]])

# #             idx_a = tf.stack([batch_idx, y0_int, x0_int], axis=-1)
# #             idx_b = tf.stack([batch_idx, y1_int, x0_int], axis=-1)
# #             idx_c = tf.stack([batch_idx, y0_int, x1_int], axis=-1)
# #             idx_d = tf.stack([batch_idx, y1_int, x1_int], axis=-1)

# #             Ia = tf.gather_nd(image, idx_a)
# #             Ib = tf.gather_nd(image, idx_b)
# #             Ic = tf.gather_nd(image, idx_c)
# #             Id = tf.gather_nd(image, idx_d)

# #             return wa * Ia + wb * Ib + wc * Ic + wd * Id

# #         # Generate random displacement fields
# #         batch_size = tf.shape(image)[0]
# #         dx = tf.random.normal([batch_size, self.config.img_size_x, self.config.img_size_y], 
# #                             mean=0, stddev=sigma)
# #         dy = tf.random.normal([batch_size, self.config.img_size_x, self.config.img_size_y], 
# #                             mean=0, stddev=sigma)

# #         # Create coordinate grid
# #         x_mesh, y_mesh = tf.meshgrid(
# #             tf.range(self.config.img_size_x, dtype=tf.float32),
# #             tf.range(self.config.img_size_y, dtype=tf.float32)
# #         )

# #         # Add batch dimension and displacement
# #         x_mesh = tf.tile(tf.expand_dims(x_mesh, 0), [batch_size, 1, 1])
# #         y_mesh = tf.tile(tf.expand_dims(y_mesh, 0), [batch_size, 1, 1])

# #         # Apply displacement
# #         x_new = x_mesh + alpha * dx
# #         y_new = y_mesh + alpha * dy

# #         # Interpolate
# #         warped_image = _bilinear_interpolate(image, x_new, y_new)

# #         if label is not None:
# #             # Use nearest neighbor interpolation for labels
# #             x_new_int = tf.cast(tf.round(x_new), tf.int32)
# #             y_new_int = tf.cast(tf.round(y_new), tf.int32)
# #             x_new_int = tf.clip_by_value(x_new_int, 0, self.config.img_size_x - 1)
# #             y_new_int = tf.clip_by_value(y_new_int, 0, self.config.img_size_y - 1)
            
# #             batch_idx = tf.range(batch_size)
# #             batch_idx = tf.reshape(batch_idx, [batch_size, 1, 1])
# #             batch_idx = tf.tile(batch_idx, [1, tf.shape(x_new_int)[1], tf.shape(x_new_int)[2]])
            
# #             idx = tf.stack([batch_idx, y_new_int, x_new_int], axis=-1)
# #             warped_label = tf.gather_nd(label, idx)
            
# #             return warped_image, warped_label

# #         return warped_image

# #     def _gaussian_kernel(self, sigma, kernel_size=15):
# #         """Creates a Gaussian kernel for elastic deformation."""
# #         x = tf.range(-kernel_size // 2 + 1, kernel_size // 2 + 1, dtype=tf.float32)
# #         g = tf.math.exp(-(x ** 2) / (2 * sigma ** 2))
# #         g = g / tf.reduce_sum(g)
# #         return tf.reshape(g, [kernel_size, 1, 1, 1])

# # class DataPipeline:
# #     def __init__(self, config):
# #         self.config = config
# #         self.dataloader = PancreasDataLoader(config)

# #     def setup_training_data(self, labeled_paths, unlabeled_paths, val_paths, batch_size=8):
# #         """Sets up all datasets needed for training."""
# #         labeled_dataset = self.dataloader.create_dataset(
# #             labeled_paths['images'],
# #             labeled_paths['labels'],
# #             batch_size=batch_size,
# #             shuffle=True,
# #             augment=True
# #         )

# #         unlabeled_dataset = self.dataloader.create_unlabeled_dataset(
# #             unlabeled_paths['images'],
# #             batch_size=batch_size
# #         )

# #         val_dataset = self.dataloader.create_dataset(
# #             val_paths['images'],
# #             val_paths['labels'],
# #             batch_size=batch_size,
# #             shuffle=False,
# #             augment=False
# #         )

# #         return {
# #             'labeled': labeled_dataset,
# #             'unlabeled': unlabeled_dataset,
# #             'validation': val_dataset
# #         }

# #     def update_unlabeled_dataset(self, pseudo_labeled_data):
# #         """Updates the unlabeled dataset with new pseudo-labeled data."""
# #         if not pseudo_labeled_data:
# #             logging.warning("No pseudo-labeled data provided. Keeping the original unlabeled dataset.")
# #             return

# #         all_images = []
# #         all_pseudo_labels = []
# #         for images, pseudo_labels in pseudo_labeled_data:
# #             all_images.append(images)
# #             all_pseudo_labels.append(pseudo_labels)

# #         all_images = tf.concat(all_images, axis=0)
# #         all_pseudo_labels = tf.concat(all_pseudo_labels, axis=0)

# #         self.unlabeled_dataset = tf.data.Dataset.from_tensor_slices((all_images, all_pseudo_labels))

# #         self.unlabeled_dataset = self.unlabeled_dataset.map(
# #             self.preprocess_unlabeled, num_parallel_calls=tf.data.AUTOTUNE
# #         )
# #         self.unlabeled_dataset = self.unlabeled_dataset.batch(self.config.batch_size)
# #         self.unlabeled_dataset = self.unlabeled_dataset.prefetch(tf.data.AUTOTUNE)

# #     def get_num_batches(self, dataset_type):
# #         """Returns the number of batches in the specified dataset."""
# #         if dataset_type == 'labeled':
# #             dataset = self.labeled_dataset
# #         elif dataset_type == 'unlabeled':
# #             dataset = self.unlabeled_dataset
# #         else:
# #             raise ValueError(f"Invalid dataset type: {dataset_type}")

# #         if dataset is None:
# #             return 0
# #         else:
# #             return len(list(dataset))

# #     def preprocess_unlabeled(self, image, pseudo_label):
# #         """Preprocesses an unlabeled image and its pseudo-label."""
# #         image = tf.cast(image, tf.float32)
# #         pseudo_label = tf.cast(pseudo_label, tf.int32)
# #         return image, pseudo_label