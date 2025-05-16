import tensorflow as tf
import numpy as np
from pathlib import Path
import nibabel as nib
import time
import logging
import sys
import math

# Define AUTOTUNE for tf.data
AUTOTUNE = tf.data.experimental.AUTOTUNE

class PancreasDataLoader:
    def __init__(self, config):
        self.config = config
        self.target_size_hw = (config.img_size_y, config.img_size_x) # (Height, Width)

# ... (imports and PancreasDataLoader.__init__ remain the same) ...

    def preprocess_volume(self, image_path_str_py, label_path_str_py=None):
        # tf.print(f"DEBUG PDL.preprocess_volume: START. Image: '{image_path_str_py}', Label: '{label_path_str_py}'", output_stream=sys.stderr)

        try:
            if not Path(image_path_str_py).exists():
                tf.print(f"ERROR PDL.preprocess_volume: Image file NOT FOUND: '{image_path_str_py}'", output_stream=sys.stderr)
                return None if label_path_str_py is None else (None, None)

            raw_img_data = np.load(image_path_str_py)
            # tf.print(f"DEBUG PDL.preprocess_volume: Loaded img '{Path(image_path_str_py).name}', raw_shape: {raw_img_data.shape}, raw_dtype: {raw_img_data.dtype}", output_stream=sys.stderr)

            img_slices_list_hw = []
            if raw_img_data.ndim == 2:
                img_slices_list_hw.append(raw_img_data)
            elif raw_img_data.ndim == 3:
                if raw_img_data.shape[0] == 1 and raw_img_data.shape[1] > 1 and raw_img_data.shape[2] > 1 : # (1, H, W)
                    img_slices_list_hw.append(raw_img_data[0])
                elif raw_img_data.shape[-1] == 1 and raw_img_data.shape[0] > 1 and raw_img_data.shape[1] > 1: # (H, W, 1)
                    img_slices_list_hw.append(np.squeeze(raw_img_data, axis=-1))
                elif raw_img_data.shape[0] > 1 and raw_img_data.shape[1] > 1 and raw_img_data.shape[2] > 1: # Assume (D, H, W)
                    for i in range(raw_img_data.shape[0]):
                        img_slices_list_hw.append(raw_img_data[i])
                else: # Other 3D shapes that are problematic (e.g. (256,256,1) interpreted as D=256, H=256, W=1)
                    tf.print(f"WARNING PDL.preprocess_volume: Image '{Path(image_path_str_py).name}' has ambiguous 3D shape: {raw_img_data.shape}. Attempting to interpret.", output_stream=sys.stderr)
                    # Heuristic: if one of the dimensions is 1, it's likely channel or a flattened slice dimension
                    if raw_img_data.shape[0] != 1 and raw_img_data.shape[1] != 1 and raw_img_data.shape[2] == 1 : # (H, W, 1)
                         img_slices_list_hw.append(np.squeeze(raw_img_data, axis=-1))
                    elif raw_img_data.shape[0] == 1 and raw_img_data.shape[1] != 1 and raw_img_data.shape[2] != 1 : # (1, H, W)
                         img_slices_list_hw.append(raw_img_data[0])
                    else: # Fallback if still ambiguous, treat as D,H,W and hope H,W are reasonable
                         for i in range(raw_img_data.shape[0]):
                            img_slices_list_hw.append(raw_img_data[i])

            else:
                tf.print(f"ERROR PDL.preprocess_volume: Image '{Path(image_path_str_py).name}' has unsupported ndim: {raw_img_data.ndim}. Skipping.", output_stream=sys.stderr)
                return None if label_path_str_py is None else (None, None)

            if not img_slices_list_hw:
                tf.print(f"ERROR PDL.preprocess_volume: No valid image slices extracted from '{Path(image_path_str_py).name}'. Skipping.", output_stream=sys.stderr)
                return None if label_path_str_py is None else (None, None)

            processed_img_volume_slices = []
            for idx, slice_hw in enumerate(img_slices_list_hw):
                slice_hw = np.nan_to_num(slice_hw).astype(np.float32)
                min_val, max_val = slice_hw.min(), slice_hw.max()
                if max_val > min_val: slice_hw = (slice_hw - min_val) / (max_val - min_val)
                else: slice_hw = np.zeros_like(slice_hw)
                
                slice_h_w_c = slice_hw[..., np.newaxis]
                
                # Defensive resize: use tf.image.resize (forces to target, may distort)
                # This avoids the tf.image.resize_with_pad internal errors.
                # Your preprocess_pancreas_v2.py already aimed to pad to 256x256.
                # If a slice isn't 256x256 now, it means preprocess_pancreas_v2.py failed for it or produced varied sizes.
                # Forcing resize here ensures all slices are exactly target_size_hw.
                try:
                    # tf.print(f"DEBUG PDL.preprocess_volume: Resizing img slice {idx} of {Path(image_path_str_py).name}. Orig shape: {slice_h_w_c.shape}, Target: {self.target_size_hw}", output_stream=sys.stderr)
                    resized_slice_tf = tf.image.resize(
                        images=tf.convert_to_tensor(slice_h_w_c, dtype=tf.float32),
                        size=self.target_size_hw, # [target_height, target_width]
                        method='bilinear' # or 'nearest' for labels
                    )
                    # Ensure shape after resize
                    resized_slice_tf = tf.ensure_shape(resized_slice_tf, [self.target_size_hw[0], self.target_size_hw[1], 1])
                    processed_img_volume_slices.append(resized_slice_tf.numpy())
                except Exception as e_resize: # Broader catch for any resize issue
                    tf.print(f"ERROR PDL.preprocess_volume: tf.image.resize failed for img slice {idx} of {Path(image_path_str_py).name}. Input shape to resize: {slice_h_w_c.shape}. Error: {str(e_resize)}", output_stream=sys.stderr)

            if not processed_img_volume_slices:
                 tf.print(f"ERROR PDL.preprocess_volume: No image slices successfully resized for '{image_path_str_py}'", output_stream=sys.stderr)
                 return None if label_path_str_py is None else (None, None)

            final_img_data_d_th_tw_c = np.stack(processed_img_volume_slices, axis=0)
            
            if self.config.num_channels == 1 and final_img_data_d_th_tw_c.shape[-1] != 1:
                 final_img_data_d_th_tw_c = np.mean(final_img_data_d_th_tw_c, axis=-1, keepdims=True)
            elif self.config.num_channels == 3 and final_img_data_d_th_tw_c.shape[-1] == 1:
                 final_img_data_d_th_tw_c = np.repeat(final_img_data_d_th_tw_c, 3, axis=-1)

            final_label_data_d_th_tw_c = None
            if label_path_str_py:
                if not Path(label_path_str_py).exists():
                    tf.print(f"ERROR PDL.preprocess_volume: Label file NOT FOUND: '{label_path_str_py}'", output_stream=sys.stderr)
                    return None, None 

                raw_label_data = np.load(label_path_str_py)
                label_slices_list_hw = []
                if raw_label_data.ndim == 2:
                    label_slices_list_hw.append(raw_label_data)
                elif raw_label_data.ndim == 3:
                    # Similar standardization as for images
                    if raw_label_data.shape[0] == 1 and raw_label_data.shape[1] > 1 and raw_label_data.shape[2] > 1 : # (1, H, W)
                        label_slices_list_hw.append(raw_label_data[0])
                    elif raw_label_data.shape[-1] == 1 and raw_label_data.shape[0] > 1 and raw_label_data.shape[1] > 1: # (H, W, 1)
                        label_slices_list_hw.append(np.squeeze(raw_label_data, axis=-1))
                    elif raw_label_data.shape[0] > 1 and raw_label_data.shape[1] > 1 and raw_label_data.shape[2] > 1: # Assume (D, H, W)
                        for i in range(raw_label_data.shape[0]):
                            label_slices_list_hw.append(raw_label_data[i])
                    else: # Ambiguous, try to handle (H,W,1) case
                        if raw_label_data.shape[0] != 1 and raw_label_data.shape[1] != 1 and raw_label_data.shape[2] == 1 : # (H, W, 1)
                            label_slices_list_hw.append(np.squeeze(raw_label_data, axis=-1))
                        elif raw_label_data.shape[0] == 1 and raw_label_data.shape[1] != 1 and raw_label_data.shape[2] != 1 : # (1, H, W)
                            label_slices_list_hw.append(raw_label_data[0])
                        else:
                             for i in range(raw_label_data.shape[0]):
                                label_slices_list_hw.append(raw_label_data[i])
                else:
                    tf.print(f"ERROR PDL.preprocess_volume: Label '{Path(label_path_str_py).name}' has unsupported ndim: {raw_label_data.ndim}. Skipping.", output_stream=sys.stderr)
                    return None, None
                
                if not label_slices_list_hw:
                    tf.print(f"ERROR PDL.preprocess_volume: No valid label slices extracted from '{Path(label_path_str_py).name}'. Skipping.", output_stream=sys.stderr)
                    return None, None

                processed_label_volume_slices = []
                for idx, slice_hw in enumerate(label_slices_list_hw):
                    slice_hw = np.nan_to_num(slice_hw)
                    slice_hw = (slice_hw > 0.5).astype(np.float32)
                    slice_h_w_c = slice_hw[..., np.newaxis]
                    try:
                        resized_slice_tf = tf.image.resize(
                            images=tf.convert_to_tensor(slice_h_w_c, dtype=tf.float32),
                            size=self.target_size_hw,
                            method='nearest' # Use nearest for labels
                        )
                        resized_slice_tf = tf.ensure_shape(resized_slice_tf, [self.target_size_hw[0], self.target_size_hw[1], 1])
                        processed_label_volume_slices.append(resized_slice_tf.numpy())
                    except Exception as e_resize:
                        tf.print(f"ERROR PDL.preprocess_volume: tf.image.resize failed for lbl slice {idx} of {Path(label_path_str_py).name}. Input shape to resize: {slice_h_w_c.shape}. Error: {str(e_resize)}", output_stream=sys.stderr)


                if not processed_label_volume_slices:
                    tf.print(f"ERROR PDL.preprocess_volume: No label slices successfully resized for '{label_path_str_py}'", output_stream=sys.stderr)
                    return None, None
                
                final_label_data_d_th_tw_c = np.stack(processed_label_volume_slices, axis=0)
                final_label_data_d_th_tw_c = (final_label_data_d_th_tw_c > 0.5).astype(np.float32)

                min_successful_slices = min(len(processed_img_volume_slices), len(processed_label_volume_slices))
                if min_successful_slices == 0:
                    tf.print(f"ERROR PDL.preprocess_volume: Zero successful slices for img or label after resize for '{Path(image_path_str_py).name}'.", output_stream=sys.stderr)
                    return None, None
                
                final_img_data_d_th_tw_c = final_img_data_d_th_tw_c[:min_successful_slices]
                final_label_data_d_th_tw_c = final_label_data_d_th_tw_c[:min_successful_slices]
                
                return final_img_data_d_th_tw_c, final_label_data_d_th_tw_c
            else: 
                if final_img_data_d_th_tw_c.shape[0] == 0:
                    tf.print(f"ERROR PDL.preprocess_volume: Zero depth for unlabeled img '{Path(image_path_str_py).name}'.", output_stream=sys.stderr)
                    return None
                return final_img_data_d_th_tw_c

        except FileNotFoundError as e_fnf: 
            tf.print(f"FATAL ERROR PDL.preprocess_volume: FileNotFoundError for '{image_path_str_py}' or '{label_path_str_py}': {str(e_fnf)}", output_stream=sys.stderr)
            return None if label_path_str_py is None else (None, None)
        except Exception as e:
            tf.print(f"ERROR PDL.preprocess_volume: Unexpected error for '{image_path_str_py}': {type(e).__name__} - {str(e)}", output_stream=sys.stderr)
            return None if label_path_str_py is None else (None, None)
        
    def create_unlabeled_dataset_for_mixmatch(self, image_paths, batch_size, shuffle=True):
        """
        Creates a dataset of unlabeled images for MixMatch.
        Yields batches of single-view image slices (raw or very minimally processed).
        The K-augmentations for pseudo-labeling and strong augmentation for student
        consistency input will be handled within the MixMatchTrainer's train_step.
        """
        if not image_paths:
            tf.print("WARNING PDL.create_unlabeled_dataset_for_mixmatch: received empty image_paths.", output_stream=sys.stderr)
            # Yield an empty tensor with correct rank but 0 batch size to avoid issues downstream if possible
            # Or make the dataset yield a single dummy batch if that's easier for trainer logic
            return tf.data.Dataset.from_tensor_slices(
                tf.zeros([0, self.config.img_size_y, self.config.img_size_x, self.config.num_channels], dtype=tf.float32)
            ).batch(batch_size)


        dataset = tf.data.Dataset.from_tensor_slices(image_paths)
        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(image_paths), reshuffle_each_iteration=True)

        dataset = dataset.interleave(
            lambda img_path_tensor: tf.data.Dataset.from_generator(
                self._parse_volume_to_slices_unlabeled, # Yields single raw slices from volume
                args=(img_path_tensor,),
                output_signature=tf.TensorSpec(shape=(self.config.img_size_y, self.config.img_size_x, self.config.num_channels), dtype=tf.float32)
            ),
            cycle_length=tf.data.AUTOTUNE, # Use AUTOTUNE
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=not shuffle
        )
        
        # Filter out any dummy/error slices if _parse_volume_to_slices_unlabeled might produce them
        # For MixMatch, it's crucial that we get valid images.
        # dataset = dataset.filter(lambda img_slice: tf.reduce_sum(tf.cast(tf.shape(img_slice) > 0, tf.int32)) == 3 ) # Ensure H,W,C > 0

        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
        return dataset
# ... (rest of DataPipeline and other functions/classes in data_loader_tf2.py remain the same) ...

    @tf.function
    def _augment_slice_and_label(self, image_slice, label_slice, seed_pair):
        """Applies identical geometric augmentations to image and label slices using stateless random ops."""
        # image_slice, label_slice: [H, W, C]
        
        # Ensure correct shapes
        image_slice = tf.ensure_shape(image_slice, [self.target_size_hw[0], self.target_size_hw[1], self.config.num_channels])
        label_slice = tf.ensure_shape(label_slice, [self.target_size_hw[0], self.target_size_hw[1], 1]) # Label is single channel

        # Stateless random flips
        seed_lr, seed_ud, seed_rot = tf.unstack(tf.random.experimental.stateless_split(seed_pair, num=3))

        if tf.random.stateless_uniform(shape=[], seed=seed_lr, minval=0, maxval=1) > 0.5:
            image_slice = tf.image.flip_left_right(image_slice)
            label_slice = tf.image.flip_left_right(label_slice)

        if tf.random.stateless_uniform(shape=[], seed=seed_ud, minval=0, maxval=1) > 0.5:
            image_slice = tf.image.flip_up_down(image_slice)
            label_slice = tf.image.flip_up_down(label_slice)
        
        # Stateless random 90-degree rotations
        k_rot = tf.random.stateless_uniform(shape=[], seed=seed_rot, minval=0, maxval=4, dtype=tf.int32)
        image_slice = tf.image.rot90(image_slice, k=k_rot)
        label_slice = tf.image.rot90(label_slice, k=k_rot)
        
        return image_slice, label_slice

    @tf.function
    def _augment_single_image_slice(self, image_slice, strength='weak'):
        """Applies color/intensity augmentations to a single image slice."""
        image_slice = tf.ensure_shape(image_slice, [self.target_size_hw[0], self.target_size_hw[1], self.config.num_channels])

        if strength == 'weak':
            if tf.random.uniform(shape=[]) > 0.5:
                 image_slice = tf.image.random_brightness(image_slice, max_delta=0.1)
            if tf.random.uniform(shape=[]) > 0.5:
                 image_slice = tf.image.random_contrast(image_slice, lower=0.9, upper=1.1)
        elif strength == 'strong':
            if tf.random.uniform(shape=[]) > 0.5:
                image_slice = tf.image.random_brightness(image_slice, max_delta=0.2)
            if tf.random.uniform(shape=[]) > 0.5:
                image_slice = tf.image.random_contrast(image_slice, lower=0.8, upper=1.2)
            
            if tf.random.uniform(shape=[]) > 0.3: # Gaussian Noise
                noise = tf.random.normal(shape=tf.shape(image_slice), mean=0.0, stddev=0.05, dtype=image_slice.dtype)
                image_slice = image_slice + noise
            
            # Simple Cutout (more robust implementation)
            if tf.random.uniform(shape=[]) < 0.3: 
                img_h, img_w = self.target_size_hw[0], self.target_size_hw[1]
                cutout_size_h = tf.cast(tf.cast(img_h, tf.float32) * 0.25, tf.int32)
                cutout_size_w = tf.cast(tf.cast(img_w, tf.float32) * 0.25, tf.int32)
                
                offset_h = tf.random.uniform(shape=[], maxval=img_h - cutout_size_h + 1, dtype=tf.int32)
                offset_w = tf.random.uniform(shape=[], maxval=img_w - cutout_size_w + 1, dtype=tf.int32)
                
                # Create a mask for the cutout area
                mask_row = tf.logical_and(tf.range(img_h)[:, tf.newaxis] >= offset_h, tf.range(img_h)[:, tf.newaxis] < offset_h + cutout_size_h)
                mask_col = tf.logical_and(tf.range(img_w) >= offset_w, tf.range(img_w) < offset_w + cutout_size_w)
                cutout_mask_2d = tf.logical_and(mask_row, mask_col) # [H, W]
                cutout_mask = cutout_mask_2d[..., tf.newaxis] # [H, W, 1]
                
                image_slice = tf.where(cutout_mask, tf.zeros_like(image_slice), image_slice)

        image_slice = tf.clip_by_value(image_slice, 0.0, 1.0)
        return image_slice


class DataPipeline:
    def __init__(self, config):
        self.config = config
        self.dataloader = PancreasDataLoader(config)

    def _py_load_preprocess_volume_wrapper(self, image_path_input, label_path_input=None):
        def decode_path(path_input):
            if isinstance(path_input, tf.Tensor): return path_input.numpy().decode('utf-8')
            elif isinstance(path_input, bytes): return path_input.decode('utf-8')
            elif isinstance(path_input, str): return path_input
            return None

        image_path_str = decode_path(image_path_input)
        label_path_str = decode_path(label_path_input) if label_path_input is not None else None
        
        if image_path_str is None: # Critical if image path cannot be decoded
            tf.print(f"ERROR _py_load_wrapper: Failed to decode image_path_input: {image_path_input}", output_stream=sys.stderr)
            # Return dummy data matching Tout signature
            dummy_depth = 1
            dummy_img_vol = np.zeros([dummy_depth, self.config.img_size_y, self.config.img_size_x, self.config.num_channels], dtype=np.float32)
            if label_path_input is not None: # Assuming supervised call if label_path_input was provided
                dummy_lbl_vol = np.zeros([dummy_depth, self.config.img_size_y, self.config.img_size_x, 1], dtype=np.float32)
                return dummy_img_vol, dummy_lbl_vol
            return dummy_img_vol

        result = self.dataloader.preprocess_volume(image_path_str, label_path_str)

        is_supervised_call = label_path_str is not None
        dummy_depth = 1
        dummy_img_vol = np.zeros([dummy_depth, self.config.img_size_y, self.config.img_size_x, self.config.num_channels], dtype=np.float32)

        if is_supervised_call:
            img_data, lbl_data = result if result is not None and isinstance(result, tuple) and len(result) == 2 else (None, None)
            if img_data is None or lbl_data is None: 
                dummy_lbl_vol = np.zeros([dummy_depth, self.config.img_size_y, self.config.img_size_x, 1], dtype=np.float32)
                return dummy_img_vol, dummy_lbl_vol
            return img_data.astype(np.float32), lbl_data.astype(np.float32)
        else:
            img_data = result
            if img_data is None: return dummy_img_vol
            return img_data.astype(np.float32)

    def _parse_volume_to_slices_supervised(self, image_path_tensor, label_path_tensor):
        img_vol_np, lbl_vol_np = self._py_load_preprocess_volume_wrapper(image_path_tensor, label_path_tensor)
        for i in range(img_vol_np.shape[0]): yield img_vol_np[i], lbl_vol_np[i]

    def _parse_volume_to_slices_unlabeled(self, image_path_tensor):
        img_vol_np = self._py_load_preprocess_volume_wrapper(image_path_tensor, None)
        for i in range(img_vol_np.shape[0]): yield img_vol_np[i]

    # --- NEW: Stateless geometric augmentation for a single image slice (for unlabeled data) ---
    @tf.function
    def _stateless_geometric_augment_single_slice(self, image_slice, seed_pair):
        # image_slice: [H, W, C]
        # Unpack seeds from the pair for stateless operations
        seed1, seed2 = seed_pair[0], seed_pair[1] 
        s_lr = tf.random.experimental.stateless_split(tf.stack([seed1, seed2]), num=1)[0, :]
        s_ud = tf.random.experimental.stateless_split(s_lr, num=1)[0, :] 
        s_rot = tf.random.experimental.stateless_split(s_ud, num=1)[0, :]

        if tf.random.stateless_uniform(shape=[], seed=s_lr, minval=0, maxval=1) > 0.5:
            image_slice = tf.image.flip_left_right(image_slice)
        if tf.random.stateless_uniform(shape=[], seed=s_ud, minval=0, maxval=1) > 0.5:
            image_slice = tf.image.flip_up_down(image_slice)
        
        k_rot = tf.random.stateless_uniform(shape=[], seed=s_rot, minval=0, maxval=4, dtype=tf.int32)
        image_slice = tf.image.rot90(image_slice, k=k_rot)
        return image_slice
    # --- END NEW ---

    # --- MODIFIED: To use the new geometric augmentation for UNLABELED data ---
    def _augment_for_mean_teacher(self, image_slice, seed_pair_student_geom, seed_pair_teacher_geom):
        # Now it directly accepts the three arguments passed by dataset.map()
        # No need to unpack a tuple here.
        
        # Student: distinct geometric + strong intensity
        student_view_geom = self._stateless_geometric_augment_single_slice(image_slice, seed_pair_student_geom)
        student_view = self.dataloader._augment_single_image_slice(student_view_geom, strength='strong')

        # Teacher: distinct geometric + weak intensity
        teacher_view_geom = self._stateless_geometric_augment_single_slice(image_slice, seed_pair_teacher_geom)
        teacher_view = self.dataloader._augment_single_image_slice(teacher_view_geom, strength='weak')
        
        student_view = tf.ensure_shape(student_view, [self.config.img_size_y, self.config.img_size_x, self.config.num_channels])
        teacher_view = tf.ensure_shape(teacher_view, [self.config.img_size_y, self.config.img_size_x, self.config.num_channels])
        return student_view, teacher_view
    # --- END MODIFIED ---

    def build_labeled_dataset(self, image_paths, label_paths, batch_size, is_training=True):
        if not image_paths or not label_paths:
            tf.print("WARNING: build_labeled_dataset empty paths.", output_stream=sys.stderr)
            return tf.data.Dataset.from_tensor_slices(([], [])).batch(batch_size)

        dataset = tf.data.Dataset.from_tensor_slices((image_paths, label_paths))
        if is_training: dataset = dataset.shuffle(buffer_size=len(image_paths))

        dataset = dataset.interleave(
            lambda img_p, lbl_p: tf.data.Dataset.from_generator(
                self._parse_volume_to_slices_supervised, args=(img_p, lbl_p),
                output_signature=(
                    tf.TensorSpec(shape=(self.config.img_size_y, self.config.img_size_x, self.config.num_channels), dtype=tf.float32),
                    tf.TensorSpec(shape=(self.config.img_size_y, self.config.img_size_x, 1), dtype=tf.float32))),
            cycle_length=tf.data.AUTOTUNE, num_parallel_calls=AUTOTUNE, deterministic=not is_training)
        
        if is_training:
            # Seed dataset for LABELED data geometric augmentations
            seed_dataset_labeled_geom = tf.data.Dataset.counter().map(lambda x: (x, x + tf.cast(4e5, tf.int64))) # Different seed range

            dataset = tf.data.Dataset.zip((dataset, seed_dataset_labeled_geom))
            dataset = dataset.map(
                lambda data_pair, seed_pair: self.dataloader._augment_slice_and_label(data_pair[0], data_pair[1], seed_pair),
                num_parallel_calls=AUTOTUNE)
            # Intensity augmentations for LABELED data (can be added here if needed, or if _augment_slice_and_label is only geometric)
            # For now, assuming _augment_slice_and_label includes all necessary augmentations for labeled data.
            # Or, if PancreasDataLoader._augment_single_image_slice should also be applied:
            # dataset = dataset.map(
            #     lambda img, lbl: (self.dataloader._augment_single_image_slice(img, strength='strong'), lbl),
            #     num_parallel_calls=AUTOTUNE
            # )
            dataset = dataset.shuffle(buffer_size=1000)

        dataset = dataset.batch(batch_size); dataset = dataset.prefetch(buffer_size=AUTOTUNE)
        return dataset

    # --- MODIFIED: To provide distinct seeds for student and teacher geometric augmentations ---
    def build_unlabeled_dataset_for_mean_teacher(self, image_paths, batch_size, is_training=True):
        if not image_paths:
            tf.print("WARNING: build_unlabeled_dataset_for_mean_teacher empty paths.", output_stream=sys.stderr)
            # Return a dataset that yields pairs, even if empty, to satisfy zip
            return tf.data.Dataset.from_tensor_slices(([], [])).batch(batch_size) 

        dataset = tf.data.Dataset.from_tensor_slices(image_paths)
        if is_training: dataset = dataset.shuffle(buffer_size=len(image_paths))

        dataset = dataset.interleave(
            lambda img_p: tf.data.Dataset.from_generator(
                self._parse_volume_to_slices_unlabeled, args=(img_p,),
                output_signature=tf.TensorSpec(shape=(self.config.img_size_y, self.config.img_size_x, self.config.num_channels), dtype=tf.float32)),
            cycle_length=tf.data.AUTOTUNE, num_parallel_calls=AUTOTUNE, deterministic=not is_training)
        
        if is_training:
            # Create two independent seed datasets for student and teacher geometric augmentations
            # Using tf.data.Dataset.random() for better seed generation if available and TF version supports it,
            # otherwise, tf.data.Dataset.counter() with different offsets is a good approximation.
            seed_dataset_student_geom = tf.data.Dataset.counter().map(lambda x: (x, x + tf.cast(1e5, tf.int64))) 
            seed_dataset_teacher_geom = tf.data.Dataset.counter().map(lambda x: (x + tf.cast(2e5, tf.int64), x + tf.cast(3e5, tf.int64)))

            dataset_with_seeds = tf.data.Dataset.zip((dataset, seed_dataset_student_geom, seed_dataset_teacher_geom))
            
            dataset = dataset_with_seeds.map(
                self._augment_for_mean_teacher, # This now expects (image_slice, seed_stud_geom, seed_teach_geom)
                num_parallel_calls=AUTOTUNE
            )
            dataset = dataset.shuffle(buffer_size=1000)

        dataset = dataset.batch(batch_size); dataset = dataset.prefetch(buffer_size=AUTOTUNE)
        return dataset
    # --- END MODIFIED ---

    def build_validation_dataset(self, image_paths, label_paths, batch_size):
        if not image_paths or not label_paths:
            tf.print("WARNING: build_validation_dataset empty paths.", output_stream=sys.stderr)
            return tf.data.Dataset.from_tensor_slices(([], [])).batch(batch_size)

        dataset = tf.data.Dataset.from_tensor_slices((image_paths, label_paths))
        dataset = dataset.interleave(
            lambda img_p, lbl_p: tf.data.Dataset.from_generator(
                self._parse_volume_to_slices_supervised, args=(img_p, lbl_p), # No augmentation for validation
                output_signature=(
                    tf.TensorSpec(shape=(self.config.img_size_y, self.config.img_size_x, self.config.num_channels), dtype=tf.float32),
                    tf.TensorSpec(shape=(self.config.img_size_y, self.config.img_size_x, 1), dtype=tf.float32))),
            cycle_length=tf.data.AUTOTUNE, num_parallel_calls=AUTOTUNE, deterministic=True)
        
        dataset = dataset.batch(batch_size); dataset = dataset.prefetch(buffer_size=AUTOTUNE)
        return dataset


    def build_unlabeled_dataset_for_mixmatch(self, image_paths, batch_size, is_training=True):
        if not image_paths:
            # tf.print("WARNING DP.build_unlabeled_for_MixMatch: empty image_paths.", output_stream=sys.stderr)
            return tf.data.Dataset.from_tensor_slices(
                 (tf.zeros([0, self.config.img_size_y, self.config.img_size_x, self.config.num_channels], dtype=tf.float32),) # Tuple with one element
            ).batch(batch_size)

        dataset = tf.data.Dataset.from_tensor_slices(image_paths)
        if is_training:
            dataset = dataset.shuffle(buffer_size=len(image_paths), reshuffle_each_iteration=True)

        dataset = dataset.interleave(
            lambda img_p: tf.data.Dataset.from_generator(
                self._parse_volume_to_slices_unlabeled, # Correct: uses DataPipeline's method
                args=(img_p,),
                output_signature=tf.TensorSpec(shape=(self.config.img_size_y, self.config.img_size_x, self.config.num_channels), dtype=tf.float32)
            ),
            cycle_length=AUTOTUNE, num_parallel_calls=AUTOTUNE, deterministic=not is_training
        )
        dataset = dataset.filter(lambda img: tf.shape(img)[0] == self.config.img_size_y) # Filter out empty/failed loads

        # For MixMatch, the train_step expects a batch of "raw" (or weakly augmented at most) unlabeled images.
        # The K-augmentations for teacher and the strong augmentation for student consistency happen INSIDE the train_step.
        # So, here we just ensure the output is a tuple (img_slice,) as expected by the trainer's zipping logic.
        dataset = dataset.map(lambda x: (x,), num_parallel_calls=AUTOTUNE)

        if is_training:
            dataset = dataset.shuffle(buffer_size=1024) # Shuffle individual slices

        dataset = dataset.batch(batch_size, drop_remainder=is_training)
        dataset = dataset.prefetch(buffer_size=AUTOTUNE)
        # tf.print(f"DEBUG: build_unlabeled_dataset_for_mixmatch, element_spec: {dataset.element_spec}", output_stream=sys.stderr)
        return dataset

    def build_validation_dataset(self, image_paths, label_paths, batch_size):
        if not image_paths or not label_paths:
            # tf.print("WARNING DP.build_validation_dataset: empty image_paths or label_paths.", output_stream=sys.stderr)
            return tf.data.Dataset.from_tensor_slices((
                tf.zeros([0, self.config.img_size_y, self.config.img_size_x, self.config.num_channels]),
                tf.zeros([0, self.config.img_size_y, self.config.img_size_x, 1])
            )).batch(batch_size)

        dataset = tf.data.Dataset.from_tensor_slices((image_paths, label_paths))
        dataset = dataset.interleave(
            lambda img_p, lbl_p: tf.data.Dataset.from_generator(
                self._parse_volume_to_slices_supervised, # Correct
                args=(img_p, lbl_p),
                output_signature=(
                    tf.TensorSpec(shape=(self.config.img_size_y, self.config.img_size_x, self.config.num_channels), dtype=tf.float32),
                    tf.TensorSpec(shape=(self.config.img_size_y, self.config.img_size_x, 1), dtype=tf.float32)
                )
            ),
            cycle_length=AUTOTUNE, num_parallel_calls=AUTOTUNE, deterministic=True
        )
        dataset = dataset.filter(lambda img, lbl: tf.shape(img)[0] == self.config.img_size_y and tf.shape(lbl)[0] == self.config.img_size_y)

        dataset = dataset.batch(batch_size) # No drop_remainder for validation
        dataset = dataset.prefetch(buffer_size=AUTOTUNE)
        return dataset