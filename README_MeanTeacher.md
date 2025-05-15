# Mean Teacher for Pancreas Segmentation - Experiment Log

## Project Overview

This project aims to implement and optimize a Mean Teacher semi-supervised learning approach for pancreas segmentation from CT scans. The primary dataset used is the Pancreas-CT dataset (described in `Task07_Pancreas/dataset.json`), which involves segmenting the pancreas (label 1) and cancerous regions (label 2) against a background (label 0). The goal is to leverage unlabeled data to improve segmentation performance beyond what is achievable with a limited set of labeled data.

## Methodology: Mean Teacher

The Mean Teacher model consists of two models with identical architectures: a student model and a teacher model.
- The **student model** is trained using standard backpropagation. It learns from labeled data via a supervised loss (e.g., a combination of Dice loss and Binary Cross-Entropy). It also generates predictions on unlabeled data, which are used to compute a consistency loss.
- The **teacher model**'s weights are an exponential moving average (EMA) of the student model's weights. The teacher is not trained directly by backpropagation. Instead, its weights are updated slowly based on the student's progression.
- A **consistency loss** is applied, encouraging the student model's predictions on augmented versions of unlabeled data to be consistent with the teacher model's predictions on differently augmented versions of the same unlabeled data.

The core idea is that the teacher, by averaging student weights over time, can provide more stable and accurate pseudo-labels for the unlabeled data, guiding the student's learning process in the absence of ground truth for these samples.

## Experimentation Log and Learnings

This log details the iterative process of developing, debugging, and attempting to optimize the Mean Teacher implementation.

### Initial Setup and Baseline (Prior to detailed iterative logging)

- A U-Net like architecture was chosen for both student and teacher models.
- The supervised loss for labeled data was a combination of Dice loss and weighted Binary Cross-Entropy.
- The initial Mean Teacher setup included a consistency loss (e.g., Mean Squared Error between student and teacher probability outputs on unlabeled data) and an EMA update rule for the teacher model.

### Challenge 1: Teacher Validation Dice Stuck at 0.0

**Observation:**
In the very early runs, the `Teacher Validation Dice` score consistently remained at 0.0 throughout training, even when the `Student Validation Dice` showed some signs of learning. This indicated a fundamental issue: the teacher model was not learning or its weights were not being effectively updated from the student.

**Hypothesis:**
If the teacher model is initialized with random weights (or another poor initialization) and the EMA decay rate (`ema_decay`) is high from the beginning (e.g., 0.99 or 0.999), the teacher's weights change too slowly. The term `(1-ema_decay) * student_weights` would be too small to pull the teacher out of its initial bad state.

### Experiment 1: Warmup with Direct Weight Copy

**Objective:**
To ensure the teacher model starts from a reasonable state by directly copying the student's weights to the teacher for an initial number of epochs.

**Changes Implemented (Conceptual, reflected in `run_mean_teacher_enhanced.py` logic):**
- Introduced `warmup_epochs` (e.g., 5 epochs).
- During these initial `warmup_epochs`:
    - The teacher model's weights were set to be identical to the student model's weights at the end of each of these epochs. This is equivalent to an EMA update where the decay rate makes the teacher fully adopt the student's weights.
    - Consistency loss was typically kept at 0 during this phase, as the teacher is still being initialized.

**Results & Learnings (Example from a run on May 9, 2025):**
- **Epoch 1 (Warmup):** Student Val Dice: 0.0494, Teacher Val Dice: 0.0000
- **Epoch 2 (Warmup):** Student Val Dice: 0.0000, **Teacher Val Dice: 0.0497** (Teacher starts making non-zero predictions)
- **Epoch 5 (Warmup):** Student Val Dice: 0.0772, **Teacher Val Dice: 0.0782**
- **Conclusion:** The direct weight copy during warmup successfully "kickstarted" the teacher model. The `Teacher Val Dice` was no longer stuck at 0.0 after this phase and could produce meaningful predictions.

### Challenge 2: Teacher Performance Degrades After Warmup Phase

**Observation:**
While the warmup phase successfully initialized the teacher, its performance (as measured by `Teacher Val Dice`) often started to degrade significantly in the epochs immediately following the warmup. This degradation occurred when the EMA updates (with a standard high decay) and consistency loss were introduced. The teacher's Dice score would often fall back towards 0.0.

**Example from a run (after 5 warmup epochs, May 9, 2025):**
- **Epoch 6 (EMA starts, Consistency loss begins to ramp):** Teacher Val Dice: 0.0786
- **Epoch 7:** Student Val Dice: 0.0931, Teacher Val Dice: 0.0751
- **Epoch 10:** Student Val Dice: 0.0714, Teacher Val Dice: 0.0212
- **Epoch 13:** Student Val Dice: 0.0441, Teacher Val Dice: 0.0000

**Hypothesis:**
1.  The student model, in the epochs immediately following warmup, might still be too unstable or its learned features might be noisy.
2.  When the teacher model starts updating via EMA (e.g., `ema_decay = 0.99`), it incorporates a small percentage (1% in this case) of the student's potentially unstable weights. Over several steps, this can corrupt the teacher.
3.  Introducing the consistency loss too early, thereby forcing the student to match a teacher that is itself becoming unstable or degrading, could exacerbate the problem, leading to a downward spiral for the teacher.

### Experiment 2: Phased EMA and Consistency Ramping

**Objective:**
To provide a more stable learning environment for the teacher by introducing a dedicated "EMA Stabilization" phase *before* ramping up the consistency loss.

**Changes Implemented (in `run_mean_teacher_enhanced.py`):**
- **Phase 1: `warmup_epochs` (e.g., 5 epochs, 0-4):**
    - Teacher: Direct weight copy from student.
    - `EMA decay` (for logging): `self.ema_decay_start` (e.g., 0.95).
    - `Consistency weight`: 0.0.
- **Phase 2: `ema_stabilization_duration` (e.g., 10 epochs, making it epochs 5-14):**
    - Teacher: Updated via EMA with a fixed, high decay rate (`self.ema_stabilization_decay = 0.99`).
    - `Consistency weight`: 0.0.
    - *Goal: Allow teacher to stabilize based on a more mature student, using a gentle EMA, without consistency pressure.*
- **Phase 3: EMA & Consistency Ramp (e.g., epochs 15 to `self.consistency_rampup_epochs`=79):**
    - `EMA decay`: Ramps from `self.ema_stabilization_decay` (0.99) up to `self.ema_decay_end` (e.g., 0.999).
    - `Consistency weight`: Ramps from 0 up to `self.consistency_weight_max` (e.g., 20.0).

**Results & Learnings (Example from a run on May 9, 2025):**
- **Warmup (Epochs 1-5):** Teacher initialized as expected (e.g., Student Dice 0.0886, Teacher Dice 0.0492 at Epoch 5).
- **EMA Stabilization (Epochs 6-16 with `EMA decay = 0.990000`, `Consistency weight = 0.0000`):**
    - **Teacher Val Dice still degraded:** Started around 0.0521 (Epoch 6) but steadily declined to 0.0009 by Epoch 16.
    - Student Val Dice showed fluctuations, hitting a peak of 0.0944 (Epoch 11).
- **Consistency Ramp (Epochs 17+):**
    - As consistency weight started ramping up (e.g., 0.3077 at Epoch 17), the `Teacher Val Dice` immediately dropped to 0.0001 and then to 0.0000 for subsequent logged epochs.
- **Conclusion:** The EMA decay of 0.99 during the stabilization phase was insufficient to prevent teacher degradation. The teacher was already in a very poor state (Dice near 0) by the time the consistency loss began to be applied.

### Experiment 3: Enhanced Teacher Stability with Prolonged Stabilization and Higher Fixed EMA Decay (Latest Strategy)

**Objective:**
To make the teacher model extremely stable ("conservative") after the initial warmup. This is achieved by using a very high, fixed EMA decay rate and extending the stabilization period, giving the student more time to mature before it significantly influences the teacher and before consistency pressure is applied.

**Changes Implemented (in `_setup_training_params` within `run_mean_teacher_enhanced.py`):**
- **Phase 1: `warmup_epochs = 5` (Epochs 0-4):**
    - Teacher: Direct weight copy from student.
    - `EMA decay` (for logging during this phase): `self.ema_decay_start` (0.95).
    - `Consistency weight`: 0.0.
- **Phase 2: Extended EMA Stabilization (Epochs 5 through 19):**
    - `self.ema_stabilization_duration = 15` (making this phase last for 15 epochs, e.g., epochs 5 to 5+15-1 = 19).
    - Teacher: Updated via EMA with a *very high, fixed* decay: `self.ema_stabilization_decay = 0.999`.
    - `self.ema_decay_end` also set to `0.999`. This means after warmup, the EMA decay jumps to 0.999 and *stays there* for the rest of training (no separate EMA ramp phase).
    - `Consistency weight`: 0.0.
- **Phase 3: Consistency Ramp (Epochs 20 through `self.consistency_rampup_epochs`-1, e.g., 20-79):**
    - `Consistency weight`: Ramps from 0 up to `self.consistency_weight_max` (20.0). The ramp starts at epoch `self.warmup_epochs + self.ema_stabilization_duration` (i.e., 5 + 15 = 20).
    - `EMA decay`: Remains fixed at 0.999.

**Rationale for Experiment 3:**
- An EMA decay of 0.999 means the teacher model incorporates only 0.1% of the student's weights at each update step post-warmup. This should make the teacher very resistant to rapid changes from a potentially noisy student.
- The student model has a significantly longer period (epochs 5-19) to train and stabilize using only the supervised loss before its weights can meaningfully influence the (very stable) teacher, and critically, before any consistency loss is applied.

**Current Status (As of May 9, 2025):**
- The changes for Experiment 3 have been implemented in the `run_mean_teacher_enhanced.sh` script (which calls `run_mean_teacher_enhanced.py`).
- A new training run with these settings has been initiated.
- Logs from this specific experiment are pending.

**Expected Outcome from Experiment 3:**
- The primary hope is that the `Teacher Val Dice` will maintain a respectable value (established during the warmup phase) throughout the extended EMA stabilization period (epochs 5-19).
- Close observation will be needed to see how both student and teacher Dice scores evolve once the consistency loss begins its ramp-up from epoch 20 onwards, with the teacher maintained by a 0.999 EMA.

## Key Parameters in `run_mean_teacher_enhanced.py` (Central to these Experiments)

- `self.warmup_epochs`: Number of initial epochs where the teacher's weights are directly copied from the student.
- `self.ema_decay_start`: The EMA decay value logged during the warmup phase (actual update is a direct copy).
- `self.ema_stabilization_duration`: The number of epochs immediately following warmup, during which the EMA decay is fixed (at `ema_stabilization_decay`) and consistency loss is kept at 0.
- `self.ema_stabilization_decay`: The fixed, high EMA decay rate used during the stabilization phase. In Experiment 3, this is set to 0.999.
- `self.ema_decay_end`: The target EMA decay rate for later stages of training. In Experiment 3, this is also set to 0.999, effectively making the EMA decay constant after warmup.
- `self.consistency_weight_max`: The maximum value that the consistency loss coefficient will reach.
- `self.consistency_rampup_epochs`: The epoch index by which the consistency weight completes its ramp from 0 to `consistency_weight_max`. The *start* of this ramp is determined by `warmup_epochs + ema_stabilization_duration`.

## Next Steps / Future Work

- Meticulously analyze the training logs (Student Val Dice, Teacher Val Dice, loss components, consistency weight, EMA decay values per epoch) from Experiment 3.
- If the teacher model demonstrates improved stability and the introduction of consistency loss leads to benefits for either or both models, continue the training run.
- If issues with teacher degradation persist even with the 0.999 EMA decay:
    - Re-evaluate the student model's learning rate schedule and overall stability. Perhaps the student itself is too erratic.
    - Investigate the impact of data augmentations – are they too strong for the student initially, or is the difference in augmentation between student and teacher problematic?
    - Consider alternative formulations for the consistency loss.
    - Further adjust the durations and specific values for the warmup, stabilization, and ramp-up phases.
- Systematically plot and compare learning curves across all major experiments to visually diagnose issues and confirm improvements.

- **Phase 3: Consistency Ramp-up (Epochs 21-25, 20-24 zero-indexed - `EMA decay = 0.999`, `Consistency Weight` ramping from 0):**
    - Consistency weight began its ramp as scheduled (e.g., 0.0 at Epoch 21, 0.3333 at Epoch 22, up to 1.3333 by Epoch 25).
    - As consistency loss was introduced, the `Teacher Val Dice`, already at a low 0.0464, declined rapidly to 0.0178 by Epoch 25.
    - The `Student Val Dice` hovered around 0.09-0.10 during these epochs.
    - *Conclusion for Phase 3:* Applying consistency loss to a weak and declining teacher accelerates its collapse.

- **Further into Consistency Ramp-up (Epochs 26-27, `EMA decay = 0.999`):**
    - `Consistency Weight` continued to ramp (1.6667 at Epoch 26, 2.0000 at Epoch 27).
    - `Teacher Val Dice` continued its decline, reaching 0.0137 by Epoch 27.
    - `Student Val Dice` also remained low, around 0.0826 at Epoch 27.
    - *The trend of teacher collapse and low student performance persists and worsens as consistency pressure increases with an unstable teacher.*

- **Overall Conclusion for Experiment 3 (Initial Epochs):**
