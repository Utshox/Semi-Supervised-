# README v2 - Context for New Chat Session

This file contains context from a previous chat session to help continue the work.

**Date:** May 5, 2025

**Workspace Structure:**

```
analyze_results.py
check_job_status.sh
config.py
data_loader_tf2.py
debug_dice_scores.py
debug_dice.py
generate_visualizations.py
main.py
models_tf2.py
README_HIGHPERF_OPTIMIZATIONS.md
README.md
requirements.txt
run_debug_dice.sh
run_mean_teacher_enhanced.sh
run_mean_teacher_fixed.sh
run_mean_teacher_highperf.sh
run_mean_teacher_optimized.sh
run_mean_teacher_ultralow.sh
run_mean_teacher.sh
run_mixmatch_fixed.sh
run_mixmatch_highperf.sh
run_mixmatch_optimized.sh
run_mixmatch_ultralow.sh
run_mixmatch.sh
run_supervised_enhanced.py
run_supervised_enhanced.sh
run_supervised_highperf.sh
run_supervised_optimized.sh
run_supervised.sh
train_ssl_tf2n.py
visualization.py
visualize_mixmatch_history.py
visualize_mixmatch_results.py
visualize_supervised_results.py
```
*(Note: This structure might be truncated)*

**Last Action Summary:**

*   **File Edited:** `/stud3/2023/mdah0000/smm/Semi-Supervised-/run_mean_teacher_enhanced.py`
*   **Function Modified:** `_dice_metric` within the `AdvancedMeanTeacherTrainer` class (or potentially a standalone `DiceMetric` class depending on the exact file state after edits).
*   **Change Description:** The `_dice_metric` function was modified to calculate the Dice score for each image in the batch individually. It then averages these scores *only* for the images where the ground truth mask (`y_true`) is not empty (i.e., contains at least one positive pixel). This prevents images without any positive labels from diluting the average Dice score. Debug prints were also added.

**Next Steps:**

Continue the development or debugging process based on this context.

**Additional Context & Potential Issues:**

*   **Environment:** The project is being developed and run in a High-Performance Computing (HPC) environment. This might influence path structures, resource availability, and job submission methods (e.g., using SLURM or similar schedulers).
*   **File Paths:** There might be ongoing issues related to hardcoded or incorrect file paths within the scripts. Ensure paths are correctly configured, possibly using relative paths or environment variables, especially considering the HPC environment.
*   **Training Scripts:** The workspace contains multiple training scripts for different approaches:
    *   `run_supervised_*.py/.sh`: Standard supervised training.
    *   `run_mean_teacher_*.py/.sh`: Semi-supervised learning using the Mean Teacher method.
    *   `run_mixmatch_*.py/.sh`: Semi-supervised learning using the MixMatch method.
    *   Various versions exist (e.g., `_enhanced`, `_fixed`, `_highperf`, `_optimized`, `_ultralow`), indicating different stages of development, optimization, or debugging.
*   **Script Generation:** The `.sh` scripts are often used to configure and generate the corresponding `.py` training scripts before execution, allowing for parameterization and environment setup.
*   **Project Goal:** This project is part of a Master's thesis. High-quality results, including clear visualizations and plots, are crucial for the final output.
*   **Unresolved Issues (from previous session):**
    *   There might have been confusion regarding which file to edit (`run_mean_teacher_enhanced.sh` vs. `run_mean_teacher_enhanced.py`). The edit to `_dice_metric` was intended for the Python script.
    *   The previous assistant might have removed lines unintentionally during edits. Careful review of changes is recommended.
    *   The `tensorflow` import resolution error reported by the `get_errors` tool was likely an environment issue with the tool itself and not necessarily a problem in the HPC execution environment, but it's worth noting.
    *   **Path Confusion:** There was previous confusion regarding the correct output/data directory, specifically involving `/scratch/lustre/home/mdah0000/smm/v14`. This path needs careful verification in scripts.
    *   **Edit Stability:** Previous attempts to modify code sometimes introduced new, unrelated issues. Edits should be carefully checked and validated.
    *   **Session Continuity:** This README was created to ensure context is passed to a new chat session, as the previous session might have encountered limitations. The goal is to seamlessly continue the work.

**Next Steps:**

Continue the development or debugging process based on this expanded context. Pay close attention to file paths (especially `/scratch/lustre/home/mdah0000/smm/v14`), the specific training script being worked on, the need for robust visualization, and carefully validate any code changes.

## Mean Teacher Enhancement & Debugging Journey

This section documents the iterative process of enhancing the Mean Teacher model and debugging issues encountered, forming part of a thesis project.

### Next Major Strategy for Mean Teacher: Leveraging Supervised Pre-training (May 10, 2025)

*   **Context**: Previous Mean Teacher experiments (detailed in `README_MeanTeacher.md` and summarized below) showed that both student and teacher models failed to achieve good performance when trained from scratch or with various warmup/stabilization strategies. The teacher model consistently degraded, and the student's validation Dice scores remained low.
*   **Key Achievement**: A separate supervised training run (`run_supervised_enhanced.sh`) achieved a strong baseline validation Dice score of **0.8577**. The best checkpoint is saved at `/scratch/lustre/home/mdah0000/smm/v14/supervised_advanced_results/checkpoints/final_20250510_014845`.
*   **Proposed Strategy**: The primary next step for the Mean Teacher approach is to initialize the **student model** within the `run_mean_teacher_enhanced.py` script using the weights from this successful supervised checkpoint.
*   **Rationale**:
    1.  Starting the student model from a strong, pre-trained state should provide a much better foundation for the teacher model (whose weights are an EMA of the student's).
    2.  This should allow the consistency loss mechanism to be more effective, as it will be guiding a competent student based on a (hopefully) more competent teacher.
    3.  This approach aims to overcome the model collapse issues observed in previous Mean Teacher iterations.
*   **Immediate Next Action**: Modify the `run_mean_teacher_enhanced.py` script to implement the loading of these pre-trained supervised weights for the student model at the beginning of its training process. The teacher model's initialization should then follow (e.g., copied from the pre-trained student initially, then updated via EMA).

### Iteration 1: Initial Run and Validation Dice Issue

*   **Observation**: After an initial run of `run_mean_teacher_enhanced.sh`, the validation Dice scores (both student and teacher) were extremely low (e.g., Student Val Dice: 0.0341, Teacher Val Dice: 0.0000 after Epoch 1).
*   **Log Analysis**:
    *   Debug prints within the `_dice_metric` function (e.g., `Dice Metric y_true_sliced sum (before resize): 0`, `Dice Metric Valid Mask (non-empty labels): [0 0]`) indicated that for many validation batches, the ground truth label for the pancreas (channel 1) was being interpreted as entirely zero *before* any resizing occurred within the metric.
    *   This suggested an issue with how validation labels were being loaded or processed, leading to most validation samples being skipped or evaluated against an empty ground truth.
*   **Hypothesis**: The problem likely lies in the data loading pipeline, specifically how label files are read, binarized, or one-hot encoded.
*   **Debugging Step 1**: Added detailed print statements to `data_loader_tf2.py` in the `preprocess_volume` and `load_and_preprocess` functions. The goal is to inspect:
    1.  Unique values and sum of the raw `label_data` immediately after loading.
    2.  Sum of `resized_label` after binarization (`resized_label > 0`).
    3.  Sum of the foreground channel (channel 1) of `label_slice` after one-hot encoding.
    These prints are directed to `sys.stderr` for visibility in the SLURM output logs.
*   **Next Action**: Re-run the `run_mean_teacher_enhanced.sh` script to capture these new debug logs and analyze the label values at each stage.

### Iteration 2: Analyzing Enhanced Debug Logs

*   **Observation**: The new debug logs from `data_loader_tf2.py` and the `validate` method in `run_mean_teacher_enhanced.py` (after ensuring the debug prints in `validate` occur for every batch) provided more clarity.
*   **Log Analysis (Error File Snippets)**:
    *   `DEBUG_LOAD_SLICE: File mask_cropped.npy, Slice 15, Sum of label_slice_orig (binary H,W before one-hot): 1213.0`
    *   `DEBUG: One-hot label_slice foreground (channel 1) sum: 1213.0`
    *   `DEBUG: load_and_preprocess: For volume mask_cropped.npy, final label_slices_np ch1 (FG) sum: 36903.0, ch0 (BG) sum: 22245336.0`
        *   These lines confirm that the data loader *is* correctly reading `.npy` label files, identifying foreground pixels, and the one-hot encoding process within `load_and_preprocess` seems to preserve these foreground pixels in channel 1.
    *   `--- Validate Function Debug (Batch 67) ---`
    *   `Raw labels shape: [2 512 512 2] dtype: tf.float32 sum: 524288`
    *   `Raw labels channel 1 sum: 0`
        *   This, and similar messages for other validation batches (68, 69, 71, 86), shows that many batches *entering the `validate` function* have a sum of 0 for the foreground channel.
    *   `Epoch 1 summary: ... Student Val Dice: 0.0079, Teacher Val Dice: 0.0000`
    *   `Epoch 2 summary: ... Student Val Dice: 0.0058, Teacher Val Dice: 0.0000`
        *   The Dice scores remain extremely low.
*   **Interpretation**:
    *   The data loader correctly processes label files and creates one-hot encoded labels with foreground pixels.
    *   However, the validation dataset, when batched, results in many batches containing only slices with no foreground labels. This is expected behavior if the validation set contains many "empty" slices, as the `_dice_metric` correctly calculates a score of 0 for these and only averages scores from non-empty masks.
    *   The core issue is **not** that all validation labels are empty, but that the **model is performing very poorly even on slices that do contain labels**. The low `Train Dice` (e.g., 0.0501, 0.1477) further supports this. The model isn't learning effectively.
*   **Hypothesis**: The problem is less about the data pipeline's ability to *provide* correct labels (it seems to be doing that) and more about the model's ability to *learn* from them.
*   **Next Steps & Recommendations**:
    1.  **Verify Data Integrity**: Manually inspect a sample of image and label `.npy` files from both training and validation sets to ensure the masks are sensible and correctly aligned with images.
    2.  **Simplify Training Task**:
        *   Train on a very small, verified subset of data (e.g., 1-2 volumes or even a few specific slices known to have good masks).
        *   Initially, disable all data augmentations in the data loader and run in a purely supervised mode (i.e., set `consistency_weight` to 0 or remove the unlabeled data component entirely from the `train_step`).
        *   The goal is to see if the model can overfit this tiny, clean dataset. If it can't achieve a high Dice score on this simplified task, it points to a more fundamental issue in the model architecture, loss calculation, or the basic training loop.
    3.  **Inspect Model Predictions**: For a validation batch that *does* contain labels (e.g., batch 62 or 63 from the logs, which had non-zero `Raw labels channel 1 sum`), save and visualize the actual raw output (logits) of the model. This will show if the model is predicting all zeros, noise, or something else.
    4.  **Review Model and Loss Functions**:
        *   Double-check the `PancreasSeg` model architecture, particularly the final activation layer. Ensure it's outputting logits as expected by the loss functions.
        *   Review the `_dice_loss` and `_weighted_bce_loss`. Ensure the slicing and resizing operations are correct and that they are appropriate for the data characteristics (e.g., class imbalance, nature of segmentation task). The `pos_weight` in BCE might need tuning.
    5.  **Hyperparameter Tuning**: If the model shows signs of learning on a simplified task, then gradually reintroduce complexity (more data, augmentations, semi-supervised components) and experiment with learning rates, optimizer settings, or the weighting of Dice vs. BCE loss.

### Iteration 3: Main.py Update and Persistent Validation Label Issue

*   **Code Update**: Modified `main.py` to use `image.npy` and `mask.npy` instead of `img_cropped.npy` and `mask_cropped.npy` for data loading. This change was successful, and the scripts are running with these new filenames.
*   **Preprocessing Status**: The preprocessing step, which generated these `.npy` files, is considered complete.
*   **Persistent Issue**: Despite the data loader (`data_loader_tf2.py`) correctly reading and processing label files (including one-hot encoding and identifying foreground pixels), the `validate` function in `run_mean_teacher_enhanced.py` continues to receive validation batches where the foreground channel (channel 1) of the labels often sums to zero.
*   **Current Focus**: The primary challenge is to understand why the foreground information, confirmed to be present during the initial loading and preprocessing by `data_loader_tf2.py`, is lost or zeroed out by the time it forms batches and reaches the `validate` method.

**Next Steps:**

Investigate the data pipeline between the `load_and_preprocess` function in `data_loader_tf2.py` and the input to the `validate` method in `run_mean_teacher_enhanced.py`. Specifically, examine how TensorFlow datasets batch and prefetch the data, and if any intermediate steps might be altering or misinterpreting the label data for the validation set.

## Supervised Model Performance (run_supervised_enhanced.sh) - May 10, 2025

*   **Script Used**: `run_supervised_enhanced.sh` (which generates `run_supervised_enhanced.py`)
*   **Key Configuration**: Standard Dice Loss, BCE Loss, Adam optimizer.
*   **Outcome**: Training completed with early stopping.
    *   Best Validation Dice Score: 0.8577
    *   Early stopping triggered at Epoch 74 (Patience: 15/15).
    *   The final checkpoint was saved to `/scratch/lustre/home/mdah0000/smm/v14/supervised_advanced_results/checkpoints/final_20250510_014845`.
*   **Significance**: This provides a strong baseline for the supervised performance on the dataset, which will be a reference point when evaluating semi-supervised approaches like Mean Teacher.
