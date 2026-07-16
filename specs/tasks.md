# Implementation Plan:

## Overview

This plan implements the Oral Disease Image Classification notebook in 15 sequential tasks. The notebook is a single `.py` file with `# %%` cell markers. Tasks are ordered by dependency: configuration first, then data pipeline, then shared utilities, then model families (Custom CNN → EfficientNetB0 → ResNet50), and finally comparison and conclusion.

## Tasks

- [ ] 1. Create notebook file with imports and global configuration. Create `oral_disease_classification.py` with title markdown cell, imports organized by stdlib/third-party/tensorflow, configuration constants (DATASET_PATH, SEED=42, IMAGE_SIZE=(224,224), BATCH_SIZE=32, NUM_CLASSES=6, EPOCHS_BASELINE=100, EPOCHS_PRETRAINED=50, EPOCHS_TUNER=30), seed setting (random.seed, np.random.seed, tf.random.set_seed), and checkpoints directory creation. **Requirements: 1, 16**
- [ ] 2. Implement data loading and exploration. Add Section 1 markdown header. Implement load_dataset() that walks DATASET_PATH, skips folders with "yolo" (case-insensitive), collects image paths (.jpg/.jpeg/.png/.bmp) and labels into DataFrame. Populate CLASS_NAMES and LABEL_TO_INDEX. Add error handling (FileNotFoundError, ValueError). Print df.head(), total count, value_counts(). **Requirements: 2**
- [ ] 3. Implement stratified data splitting. Implement split_data() with two-stage sklearn train_test_split: 70/30 then 50/50 on temp, both with stratify and random_state=47. Print split sizes and per-class distribution. **Requirements: 3**
- [ ] 4. Implement tf.data.Dataset pipeline. Implement process_path() (read_file → decode_jpeg → resize → float32/255 → one_hot) and build_dataset() (from_tensor_slices → map → shuffle(1000) if training → batch(32) → prefetch(AUTOTUNE)). Build train_ds, val_ds, test_ds. **Requirements: 4**
- [ ] 5. Implement data visualization. Display 3×3 sample grid with class labels, 9 augmented versions of one image, and class distribution bar chart with axis labels and titles. **Requirements: 5**
- [ ] 6. Implement shared utility functions. Implement get_augmentation_layers() (RandomFlip + RandomRotation(0.05)), get_callbacks(checkpoint_path) (ReduceLROnPlateau, EarlyStopping, ModelCheckpoint), and evaluate_model() (accuracy/loss curves with best epoch marker, model.evaluate on all splits, classification_report, confusion matrix heatmap with coolwarm colormap). **Requirements: 13**
- [ ] 7. Implement Custom CNN C1 — build, train, evaluate. Add Section 2 markdown header. Build 4-block CNN (64/128/256/256 filters, dropout 0.2/0.3/0.4/0.4) with GAP → Dense(1024) → Drop(0.5) → Dense(512) → Drop(0.3) → Dense(6,softmax). All Conv2D: 3×3, relu, same, L2(1e-4). Train EPOCHS_BASELINE, save cnn_c1_best.h5. Evaluate. **Requirements: 6**
- [ ] 8. Implement Custom CNN C2 — build, train, evaluate. Same 4 blocks as C1 plus 5th block (512 filters, Drop(0.5)), reduced head: GAP → Dense(512,relu,L2(1e-4)) → Drop(0.3) → Dense(6,softmax). Train EPOCHS_BASELINE, save cnn_c2_best.h5. Evaluate. **Requirements: 7**
- [ ] 9. Implement EfficientNetB0 C1 — build, train, evaluate. Add Section 3 markdown header. EfficientNetB0(unfrozen, imagenet) → GAP → BN → Dense(256,relu,L2(0.0005)) → Drop(0.4) → Dense(6,softmax). Train EPOCHS_PRETRAINED, save eff_c1_best.h5. Evaluate. **Requirements: 8**
- [ ] 10. Implement EfficientNetB0 C2 — freeze/unfreeze training. EfficientNetB0(frozen) → GAP → BN → Dense(512) → Drop(0.5) → Dense(256) → Drop(0.3) → Dense(6,softmax). Phase 1: frozen 10 epochs. Phase 2: unfrozen, recompile, 40 epochs. Concatenate histories. Save eff_c2_best.h5. Evaluate. **Requirements: 9**
- [ ] 11. Implement EfficientNetB0 C3 — Keras Tuner search and retrain. Model builder with frozen EfficientNetB0 and search space (lr, dense_units, dropout, l2). BayesianOptimization max_trials=10, epochs=EPOCHS_TUNER. Print best params. Retrain best for EPOCHS_PRETRAINED. Save eff_c3_best.h5. Evaluate. **Requirements: 10**
- [ ] 12. Implement ResNet50 C1 — build, train, evaluate. Add Section 4 markdown header. ResNet50(unfrozen, imagenet) → GAP → BN → Dense(256,relu,L2(0.0005)) → Drop(0.4) → Dense(6,softmax). Train EPOCHS_PRETRAINED, save res_c1_best.h5. Evaluate. **Requirements: 11**
- [ ] 13. Implement ResNet50 C2 — freeze/unfreeze training. ResNet50(frozen) → GAP → BN → Dense(512) → Drop(0.5) → Dense(256) → Drop(0.3) → Dense(6,softmax). Phase 1: frozen 10 epochs. Phase 2: unfrozen, recompile, 40 epochs. Concatenate histories. Save res_c2_best.h5. Evaluate. **Requirements: 12**
- [ ] 14. Implement model comparison table. Add Section 5 markdown header. Build DataFrame from all 7 results with columns: Model, Config, Params(M), Train/Val/Test Acc, Train/Val/Test Loss, Epochs Run. Format percentages and decimals. Highlight best test accuracy row with yellow background. Handle ties via val_loss tiebreaker. **Requirements: 14**
- [ ] 15. Write conclusion markdown cell. Add Section 6 markdown header. State best config per family with test accuracy. Recommend best model for deployment citing 2+ metrics. List 3+ improvement suggestions (augmentation, larger models, ensemble, class balancing). **Requirements: 15**

## Task Dependency Graph

```json
{
  "waves": [
    {"wave": 1, "tasks": [1]},
    {"wave": 2, "tasks": [2]},
    {"wave": 3, "tasks": [3]},
    {"wave": 4, "tasks": [4]},
    {"wave": 5, "tasks": [5, 6]},
    {"wave": 6, "tasks": [7]},
    {"wave": 7, "tasks": [8]},
    {"wave": 8, "tasks": [9]},
    {"wave": 9, "tasks": [10]},
    {"wave": 10, "tasks": [11]},
    {"wave": 11, "tasks": [12]},
    {"wave": 12, "tasks": [13]},
    {"wave": 13, "tasks": [14]},
    {"wave": 14, "tasks": [15]}
  ]
}
```

Tasks 1-6 form the foundation (config, data, utilities). Tasks 7-13 are the model training pipeline (sequential because each appends to the results list). Task 14 depends on all model tasks completing. Task 15 depends on the comparison table.

## Notes

- The notebook is implemented as a single `.py` file using `# %%` cell markers for VS Code / Jupyter compatibility
- All model checkpoints use unique filenames to prevent overwrites
- The results list is accumulated across tasks 7-13 and consumed by task 14
- Two-phase training (tasks 10, 13) requires history concatenation for continuous plotting
- Keras Tuner (task 11) creates a `keras_tuner_dir/` directory with trial logs
