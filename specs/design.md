# Technical Design Document

## Overview

This document describes the technical design for the Oral Disease Image Classification experimental notebook. The notebook is implemented as a single Python script (`.py` with `# %%` cell markers, convertible to `.ipynb`) that trains 7 model configurations across 3 families and produces a comparative evaluation. The design prioritizes reproducibility, GPU-efficient data pipelines, and consistent evaluation methodology.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                     oral_disease_classification.py                    │
├─────────────────────────────────────────────────────────────────────┤
│  Cell 0: Imports & Configuration                                     │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │ DATASET_PATH, SEED, IMAGE_SIZE, BATCH_SIZE, NUM_CLASSES,    │    │
│  │ EPOCHS_BASELINE, EPOCHS_PRETRAINED, EPOCHS_TUNER            │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                      │
│  Cell 1-4: Data Pipeline                                             │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │ load_dataset() → DataFrame                                   │    │
│  │ split_data() → train_df, val_df, test_df                    │    │
│  │ process_path() → (image_tensor, one_hot_label)              │    │
│  │ build_dataset() → tf.data.Dataset (batched, prefetched)     │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                      │
│  Cell 5-6: Data Visualization                                        │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │ plot_sample_grid(), plot_augmented(), plot_distribution()    │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                      │
│  Cell 7-8: Shared Utilities                                          │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │ get_augmentation_layers() → Sequential                       │    │
│  │ get_callbacks(checkpoint_path) → [callbacks]                 │    │
│  │ evaluate_model(model, history, ...) → metrics_dict           │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                      │
│  Cell 9-14: Custom CNN (C1, C2 + evaluation)                        │
│  Cell 15-22: EfficientNetB0 (C1, C2, C3 + evaluations)             │
│  Cell 23-28: ResNet50 (C1, C2 + evaluation)                        │
│  Cell 29: Comparison Table                                           │
│  Cell 30: Conclusion (markdown)                                      │
└─────────────────────────────────────────────────────────────────────┘
```

## Data Flow Diagram

```
┌──────────────┐     ┌──────────────┐     ┌─────────────────┐
│  DATASET_PATH │────▶│ load_dataset │────▶│   DataFrame     │
│  (filesystem) │     │  (os.walk)   │     │ filepath, label │
└──────────────┘     └──────────────┘     └────────┬────────┘
                                                    │
                                                    ▼
                                          ┌─────────────────┐
                                          │  train_test_split│
                                          │  (stratified)    │
                                          └───┬───┬───┬─────┘
                                              │   │   │
                                    ┌─────────┘   │   └─────────┐
                                    ▼             ▼             ▼
                             ┌──────────┐  ┌──────────┐  ┌──────────┐
                             │ train_df │  │  val_df  │  │ test_df  │
                             └────┬─────┘  └────┬─────┘  └────┬─────┘
                                  │             │             │
                                  ▼             ▼             ▼
                         ┌─────────────────────────────────────────┐
                         │        tf.data.Dataset.from_tensor_slices│
                         │              ↓ map(process_path)         │
                         │              ↓ shuffle (train only)      │
                         │              ↓ batch(32)                 │
                         │              ↓ prefetch(AUTOTUNE)        │
                         └─────────────────────────────────────────┘
                                  │             │             │
                                  ▼             ▼             ▼
                            train_ds        val_ds        test_ds
                                  │             │             │
                                  └──────┬──────┘             │
                                         ▼                    ▼
                                  ┌─────────────┐     ┌─────────────┐
                                  │   model.fit │     │model.evaluate│
                                  │(train, val) │     │   (test)     │
                                  └─────────────┘     └─────────────┘
```

## Components and Interfaces

### Component 1: Configuration Module

**Purpose:** Centralize all hyperparameters and paths in one location for reproducibility.

**Data Model:**
```python
# Constants (immutable after definition)
DATASET_PATH: Path          # Root directory of oral disease dataset
SEED: int = 42              # Global random seed
IMAGE_SIZE: tuple = (224, 224)  # Target image dimensions (H, W)
BATCH_SIZE: int = 32        # Training batch size
NUM_CLASSES: int = 6        # Number of disease categories
EPOCHS_BASELINE: int = 100  # Max epochs for custom CNN
EPOCHS_PRETRAINED: int = 50 # Max epochs for transfer learning models
EPOCHS_TUNER: int = 30      # Max epochs per Keras Tuner trial

# Label mapping (populated at runtime)
CLASS_NAMES: list[str]      # Sorted list of class folder names
LABEL_TO_INDEX: dict        # {class_name: integer_index}
```

**Interface:**
- All constants are module-level variables accessible throughout the notebook
- `CLASS_NAMES` and `LABEL_TO_INDEX` are populated by `load_dataset()`

---

### Component 2: Data Pipeline

**Purpose:** Load images from disk, apply stratified splitting, and construct GPU-optimized tf.data pipelines.

**Functions:**

```python
def load_dataset(dataset_path: Path) -> pd.DataFrame:
    """
    Walk dataset directory, collect image paths and labels.
    Skips folders containing 'yolo' (case-insensitive).
    
    Returns:
        DataFrame with columns ['filepath', 'label']
    Raises:
        FileNotFoundError: if dataset_path doesn't exist
        ValueError: if no valid images found
    """

def split_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Stratified 70/15/15 split using two-stage train_test_split.
    
    Stage 1: 70% train, 30% temp (random_state=47, stratify=df['label'])
    Stage 2: 50% val, 50% test from temp (random_state=47, stratify=temp['label'])
    
    Returns:
        (train_df, val_df, test_df)
    """

def process_path(filepath: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    """
    TensorFlow mapping function for tf.data pipeline.
    
    Steps:
        1. tf.io.read_file(filepath)
        2. tf.io.decode_jpeg(img, channels=3)
        3. tf.image.resize(img, IMAGE_SIZE)
        4. tf.cast(img, tf.float32) / 255.0
        5. tf.one_hot(label, depth=NUM_CLASSES)
    
    Returns:
        (normalized_image [224,224,3], one_hot_label [6])
    """

def build_dataset(df: pd.DataFrame, is_training: bool = False) -> tf.data.Dataset:
    """
    Construct tf.data.Dataset from DataFrame.
    
    Pipeline:
        from_tensor_slices((filepaths, encoded_labels))
        → map(process_path, num_parallel_calls=AUTOTUNE)
        → shuffle(1000) [if is_training]
        → batch(BATCH_SIZE)
        → prefetch(AUTOTUNE)
    
    Returns:
        Batched, prefetched tf.data.Dataset
    """
```

**Design Decisions:**
- Augmentation is NOT applied in the data pipeline — it's embedded as Keras layers inside each model. This ensures augmentation is only active during training (layer behavior differs between `training=True` and `training=False`).
- `process_path` uses `tf.io.decode_jpeg` which handles both JPEG and PNG via TensorFlow's internal codec detection.
- Labels are integer-encoded by `LABEL_TO_INDEX` before being passed to `from_tensor_slices`, then one-hot encoded inside `process_path` to keep the pipeline pure TensorFlow ops (no Python callbacks in the graph).

---

### Component 3: Augmentation Layer

**Purpose:** Provide a consistent augmentation block embedded in all model architectures.

```python
def get_augmentation_layers() -> tf.keras.Sequential:
    """
    Returns a Sequential model with:
        - RandomFlip("horizontal")
        - RandomRotation(0.05)
    
    These layers are active only during training (model.fit).
    During inference (model.predict/evaluate), they pass through unchanged.
    """
```

**Design Decision:** Augmentation inside the model (not in tf.data pipeline) ensures:
1. Augmentation automatically disabled during evaluation/inference
2. Model is self-contained for deployment (augmentation behavior is serialized with the model)
3. GPU-accelerated augmentation when using GPU training

---

### Component 4: Callback Factory

**Purpose:** Generate fresh callback instances for each training run with unique checkpoint paths.

```python
def get_callbacks(checkpoint_path: str) -> list:
    """
    Returns:
        [
            ReduceLROnPlateau(monitor="val_accuracy", factor=0.1, patience=8),
            EarlyStopping(monitor="val_accuracy", patience=10, restore_best_weights=True),
            ModelCheckpoint(filepath=checkpoint_path, save_best_only=True,
                          monitor="val_accuracy", mode="max")
        ]
    
    Note: Fresh instances required for multi-phase training (C2 configs)
    to reset patience counters between freeze/unfreeze phases.
    """
```

---

### Component 5: Model Builders

**Purpose:** Construct each model architecture as a Keras Functional or Sequential model.

#### 5.1 Custom CNN C1

```python
def build_cnn_c1() -> tf.keras.Model:
    """
    Architecture:
        Input(224,224,3) → Augmentation →
        [Conv2D(64)×2 + BN + MaxPool + Drop(0.2)] →
        [Conv2D(128)×2 + BN + MaxPool + Drop(0.3)] →
        [Conv2D(256)×2 + BN + MaxPool + Drop(0.4)] →
        [Conv2D(256)×2 + BN + MaxPool + Drop(0.4)] →
        GAP → Dense(1024) → Drop(0.5) → Dense(512) → Drop(0.3) →
        Dense(6, softmax)
    
    All Conv2D: kernel_size=3, activation='relu', padding='same', L2(1e-4)
    All Dense (except output): activation='relu', L2(1e-4)
    
    Returns:
        Compiled Keras Model
    """
```

#### 5.2 Custom CNN C2

```python
def build_cnn_c2() -> tf.keras.Model:
    """
    Same as C1 but adds 5th block and reduces dense head:
        ... (same 4 blocks as C1) →
        [Conv2D(512)×2 + BN + MaxPool + Drop(0.5)] →
        GAP → Dense(512) → Drop(0.3) → Dense(6, softmax)
    
    Returns:
        Compiled Keras Model
    """
```

#### 5.3 EfficientNetB0 C1

```python
def build_efficientnet_c1() -> tf.keras.Model:
    """
    Architecture:
        Input(224,224,3) → Augmentation →
        EfficientNetB0(include_top=False, weights='imagenet', trainable=True) →
        GAP → BN → Dense(256, relu, L2(0.0005)) → Drop(0.4) →
        Dense(6, softmax)
    
    Note: input_tensor is used to pass augmented input directly to base model.
    Base model is fully unfrozen (replicating reference notebook behavior).
    
    Returns:
        Compiled Keras Model
    """
```

#### 5.4 EfficientNetB0 C2

```python
def build_efficientnet_c2() -> tuple[tf.keras.Model, tf.keras.Model]:
    """
    Architecture:
        Input(224,224,3) → Augmentation →
        EfficientNetB0(include_top=False, weights='imagenet', trainable=False) →
        GAP → BN → Dense(512, relu, L2(1e-4)) → Drop(0.5) →
        Dense(256, relu, L2(1e-4)) → Drop(0.3) → Dense(6, softmax)
    
    Training Strategy:
        Phase 1: base.trainable=False, train for 10 epochs
        Phase 2: base.trainable=True, recompile, train for remaining epochs
    
    Returns:
        (model, base_model) — base_model reference needed for unfreezing
    """
```

#### 5.5 EfficientNetB0 C3 (Keras Tuner)

```python
def build_efficientnet_c3(hp) -> tf.keras.Model:
    """
    Keras Tuner model-building function.
    
    Search Space:
        learning_rate: Choice([1e-3, 5e-4, 1e-4, 5e-5])
        dense_units: Choice([128, 256, 512])
        dropout_rate: Float(0.2, 0.6, step=0.1)
        l2_rate: Choice([1e-3, 5e-4, 1e-4])
    
    Architecture:
        Input(224,224,3) → Augmentation →
        EfficientNetB0(frozen) → GAP → BN →
        Dense(dense_units, relu, L2(l2_rate)) → Drop(dropout_rate) →
        Dense(6, softmax)
    
    Returns:
        Compiled Keras Model (compiled inside function with hp.learning_rate)
    """
```

#### 5.6 ResNet50 C1

```python
def build_resnet_c1() -> tf.keras.Model:
    """
    Same head pattern as EfficientNetB0 C1 but with ResNet50 backbone.
    Base model fully unfrozen.
    
    Architecture:
        Input(224,224,3) → Augmentation →
        ResNet50(include_top=False, weights='imagenet', trainable=True) →
        GAP → BN → Dense(256, relu, L2(0.0005)) → Drop(0.4) →
        Dense(6, softmax)
    
    Returns:
        Compiled Keras Model
    """
```

#### 5.7 ResNet50 C2

```python
def build_resnet_c2() -> tuple[tf.keras.Model, tf.keras.Model]:
    """
    Same freeze-then-unfreeze strategy as EfficientNetB0 C2.
    
    Architecture:
        Input(224,224,3) → Augmentation →
        ResNet50(include_top=False, weights='imagenet', trainable=False) →
        GAP → BN → Dense(512, relu, L2(1e-4)) → Drop(0.5) →
        Dense(256, relu, L2(1e-4)) → Drop(0.3) → Dense(6, softmax)
    
    Returns:
        (model, base_model)
    """
```

---

### Component 6: Training Orchestrator

**Purpose:** Manage the training loop for each configuration, handling freeze/unfreeze phases.

```python
def train_standard(model, train_ds, val_ds, epochs, checkpoint_path) -> tf.keras.callbacks.History:
    """
    Standard single-phase training.
    Used by: CNN C1, CNN C2, EfficientNetB0 C1, ResNet50 C1, EfficientNetB0 C3 retrain.
    
    Returns:
        Training history object
    """

def train_freeze_unfreeze(model, base_model, train_ds, val_ds, 
                          total_epochs, freeze_epochs, checkpoint_path) -> tf.keras.callbacks.History:
    """
    Two-phase training: frozen head training then full fine-tuning.
    Used by: EfficientNetB0 C2, ResNet50 C2.
    
    Phase 1: base_model.trainable = False, train for freeze_epochs
    Phase 2: base_model.trainable = True, recompile, train for (total_epochs - freeze_epochs)
    
    Histories from both phases are concatenated for plotting.
    
    Returns:
        Combined history (dict with concatenated metric lists)
    """
```

**Design Decision:** History concatenation for two-phase training allows the evaluation block to plot continuous curves across both phases without special-casing the plotting logic.

---

### Component 7: Evaluation Block

**Purpose:** Produce consistent evaluation outputs for every trained model.

```python
def evaluate_model(model, history, train_ds, val_ds, test_ds, 
                   class_names, model_name) -> dict:
    """
    Full evaluation pipeline:
    
    1. Plot accuracy curves (train vs val) with best epoch marker
    2. Plot loss curves (train vs val) with best epoch marker
    3. model.evaluate() on train, val, test → print results
    4. Generate test predictions → classification_report
    5. Confusion matrix heatmap (seaborn, coolwarm, annotated)
    
    Returns:
        {
            'model_name': str,
            'params_m': float,           # model.count_params() / 1e6
            'train_acc': float,
            'val_acc': float,
            'test_acc': float,
            'train_loss': float,
            'val_loss': float,
            'test_loss': float,
            'epochs_run': int            # len(history['accuracy'])
        }
    """
```

**Plotting Details:**
- Best epoch identified as `np.argmax(history['val_accuracy'])`
- Scatter point plotted at `(best_epoch, best_val_accuracy)` and `(best_epoch, best_val_loss)`
- Classification report uses `sklearn.metrics.classification_report` with `target_names=class_names`
- Confusion matrix uses `sklearn.metrics.confusion_matrix` + `sns.heatmap(annot=True, cmap='coolwarm')`

---

### Component 8: Comparison Table Builder

**Purpose:** Aggregate results from all models into a formatted summary.

```python
def build_comparison_table(results: list[dict]) -> pd.DataFrame:
    """
    Input: List of 7 result dicts from evaluate_model()
    
    Output DataFrame columns:
        | Model | Config | Params (M) | Train Acc | Val Acc | Test Acc |
        | Train Loss | Val Loss | Test Loss | Epochs Run |
    
    Formatting:
        - Params rounded to 2 decimals
        - Accuracy as percentage (×100, 2 decimals)
        - Loss to 4 decimal places
    
    Highlighting:
        - Row with max Test Acc gets yellow background via Styler
        - Tiebreaker: lowest Val Loss among tied rows
    
    Returns:
        Styled pandas DataFrame
    """
```

---

## Data Models

### Dataset DataFrame Schema

| Column | Type | Description |
|--------|------|-------------|
| filepath | str | Absolute path to image file on disk |
| label | str | Class name derived from parent folder name |

### Label Encoding

| Structure | Type | Description |
|-----------|------|-------------|
| CLASS_NAMES | list[str] | Sorted alphabetical list of class folder names (length = NUM_CLASSES) |
| LABEL_TO_INDEX | dict[str, int] | Maps class name → integer index (0 to NUM_CLASSES-1) |

### tf.data.Dataset Element Spec

| Element | Shape | Dtype | Description |
|---------|-------|-------|-------------|
| image | (224, 224, 3) | float32 | Normalized pixel values in [0, 1] |
| label | (6,) | float32 | One-hot encoded class vector |

### Batched Dataset Element Spec

| Element | Shape | Dtype | Description |
|---------|-------|-------|-------------|
| images | (32, 224, 224, 3) | float32 | Batch of normalized images |
| labels | (32, 6) | float32 | Batch of one-hot labels |

### Training History Schema

```python
history: dict = {
    'accuracy': list[float],        # Per-epoch training accuracy
    'val_accuracy': list[float],    # Per-epoch validation accuracy
    'loss': list[float],            # Per-epoch training loss
    'val_loss': list[float],        # Per-epoch validation loss
    'lr': list[float]               # Per-epoch learning rate
}
```

### Evaluation Results Schema

```python
results_entry: dict = {
    'model_name': str,       # e.g. "Custom CNN"
    'config': str,           # e.g. "C1"
    'params_m': float,       # model.count_params() / 1e6
    'train_acc': float,      # Training accuracy (0-1 scale)
    'val_acc': float,        # Validation accuracy (0-1 scale)
    'test_acc': float,       # Test accuracy (0-1 scale)
    'train_loss': float,     # Training loss
    'val_loss': float,       # Validation loss
    'test_loss': float,      # Test loss
    'epochs_run': int        # Actual epochs trained (may be < max due to early stopping)
}
```

## File Structure

```
ODL_ASSIGNMENT/
├── oral_disease_classification.py    # Main notebook (# %% cell format)
├── checkpoints/                       # Created at runtime
│   ├── cnn_c1_best.h5
│   ├── cnn_c2_best.h5
│   ├── eff_c1_best.h5
│   ├── eff_c2_best.h5
│   ├── eff_c3_best.h5
│   ├── res_c1_best.h5
│   └── res_c2_best.h5
├── keras_tuner_dir/                   # Keras Tuner trial logs
│   └── efficientnet_c3/
└── .kiro/
    └── specs/
        └── oral-disease-classification-notebook/
            ├── .config.kiro
            ├── requirements.md
            ├── design.md
            └── tasks.md
```

## Notebook Cell Layout

| Cell # | Type | Content | Req |
|--------|------|---------|-----|
| 0 | markdown | # Oral Disease Image Classification | 16 |
| 1 | code | Imports (stdlib → third-party → tf/keras) | 16 |
| 2 | code | Configuration constants + seed setting | 1 |
| 3 | markdown | ## Section 1 — Data Pipeline | 16 |
| 4 | code | load_dataset() + print stats | 2 |
| 5 | code | split_data() + print split sizes | 3 |
| 6 | code | process_path() + build_dataset() | 4 |
| 7 | code | Visualization: sample grid | 5 |
| 8 | code | Visualization: augmented images | 5 |
| 9 | code | Visualization: class distribution bar chart | 5 |
| 10 | code | Utility functions (augmentation, callbacks, evaluate) | 13 |
| 11 | markdown | ## Section 2 — Custom CNN | 16 |
| 12 | code | build_cnn_c1() + train | 6 |
| 13 | code | evaluate CNN C1 | 13 |
| 14 | code | build_cnn_c2() + train | 7 |
| 15 | code | evaluate CNN C2 | 13 |
| 16 | markdown | ## Section 3 — EfficientNetB0 | 16 |
| 17 | code | build_efficientnet_c1() + train | 8 |
| 18 | code | evaluate EfficientNetB0 C1 | 13 |
| 19 | code | build_efficientnet_c2() + freeze/unfreeze train | 9 |
| 20 | code | evaluate EfficientNetB0 C2 | 13 |
| 21 | code | build_efficientnet_c3() + Keras Tuner search + retrain | 10 |
| 22 | code | evaluate EfficientNetB0 C3 | 13 |
| 23 | markdown | ## Section 4 — ResNet50 | 16 |
| 24 | code | build_resnet_c1() + train | 11 |
| 25 | code | evaluate ResNet50 C1 | 13 |
| 26 | code | build_resnet_c2() + freeze/unfreeze train | 12 |
| 27 | code | evaluate ResNet50 C2 | 13 |
| 28 | markdown | ## Section 5 — Model Comparison | 16 |
| 29 | code | build_comparison_table() + display | 14 |
| 30 | markdown | ## Section 6 — Conclusion | 15 |

## Key Design Decisions

### 1. Augmentation Placement (Inside Model vs. Data Pipeline)

**Choice:** Inside the model as Keras preprocessing layers.

**Rationale:** 
- Automatically disabled during `model.evaluate()` and `model.predict()` (no manual flag needed)
- Model is deployment-ready as-is (augmentation serialized with weights)
- GPU-accelerated when training on GPU
- Consistent with the reference notebook approach

### 2. Two-Stage Split Strategy

**Choice:** `train_test_split` called twice (70/30 then 50/50 on temp).

**Rationale:**
- Sklearn's `train_test_split` only produces binary splits
- Two-stage ensures exact 70/15/15 proportions
- Both calls use same `random_state=47` for reproducibility
- Stratification applied at both stages preserves class ratios

### 3. Label Encoding Strategy

**Choice:** Integer encode labels via `LABEL_TO_INDEX` dict, pass integers to `from_tensor_slices`, then one-hot encode inside `process_path`.

**Rationale:**
- `from_tensor_slices` can't directly serialize string labels efficiently
- One-hot encoding inside the TF graph (via `tf.one_hot`) keeps the pipeline GPU-friendly
- Integer mapping is deterministic because `CLASS_NAMES` is sorted alphabetically

### 4. Checkpoint File Organization

**Choice:** Store all `.h5` files in a `checkpoints/` subdirectory with unique names.

**Rationale:**
- Prevents accidental overwrites between configs
- Easy to identify which model produced which checkpoint
- Directory created with `os.makedirs(exist_ok=True)` at runtime

### 5. History Concatenation for Two-Phase Training

**Choice:** Merge Phase 1 and Phase 2 history dicts by concatenating metric lists.

**Rationale:**
- Evaluation block plots continuous curves without branching logic
- `epochs_run` correctly reflects total epochs across both phases
- Best epoch identification works on the full concatenated sequence

### 6. Keras Tuner Search Architecture

**Choice:** Frozen backbone during search, only tune the classification head.

**Rationale:**
- Dramatically reduces search time (frozen backbone = fewer trainable params per trial)
- Hyperparameter search focuses on head architecture decisions
- Best head config is then retrained with standard callbacks for full epochs
- Aligns with common practice in the literature for efficient NAS on medical images

## Dependencies

| Package | Version Requirement | Purpose |
|---------|-------------------|---------|
| tensorflow | ≥2.12 | Core DL framework, data pipeline, model training |
| keras-tuner | ≥1.3 | Bayesian hyperparameter optimization |
| numpy | ≥1.23 | Array operations, seed setting |
| pandas | ≥1.5 | DataFrame for dataset management, comparison table |
| matplotlib | ≥3.6 | Plotting (sample grids, training curves) |
| seaborn | ≥0.12 | Confusion matrix heatmaps |
| scikit-learn | ≥1.2 | train_test_split, classification_report, confusion_matrix |

## Error Handling

| Scenario | Handling |
|----------|----------|
| DATASET_PATH doesn't exist | Raise `FileNotFoundError` with clear message |
| No images found in dataset | Raise `ValueError("No valid images found...")` |
| Empty class folder | Print warning, skip class, continue |
| GPU not available | TensorFlow auto-falls back to CPU (no special handling) |
| Checkpoint directory missing | `os.makedirs("checkpoints", exist_ok=True)` before training |
| Keras Tuner directory exists | `overwrite=True` parameter in tuner constructor |
| Corrupted image file | `tf.io.decode_jpeg` raises error — handled by TF pipeline skip logic |
| Out of memory during training | Reduce BATCH_SIZE (user action); documented in config cell comments |

## Correctness Properties

### Property 1: Reproducibility
Given identical SEED, DATASET_PATH, and hardware, all training runs produce identical results (within TF non-determinism bounds on GPU).

**Validates: Requirements 1.2**

### Property 2: Stratification Preservation
Class distribution in train/val/test sets matches the original dataset distribution within ±2 percentage points per class.

**Validates: Requirements 3.3**

### Property 3: Augmentation Isolation
Augmentation layers are only active during `model.fit()` — `model.evaluate()` and `model.predict()` operate on unaugmented images.

**Validates: Requirements 4.4**

### Property 4: Checkpoint Integrity
Each model's best weights are saved to a unique file path — no cross-contamination between experiments.

**Validates: Requirements 1.8**

### Property 5: Evaluation Consistency
All 7 models are evaluated using the same test set, same metrics, and same reporting format.

**Validates: Requirements 13.3**

### Property 6: History Continuity
For two-phase training (C2 configs), concatenated history represents the complete training trajectory without gaps or duplicates.

**Validates: Requirements 9.4, 12.4**

## Testing Strategy

Since this is a notebook (not a library), verification is done through:

1. **Data Pipeline Verification**: Print dataset sizes after split, verify they sum to total. Print class distribution per split to confirm stratification.
2. **Model Shape Verification**: `model.summary()` printed for each architecture to confirm layer structure matches spec.
3. **Augmentation Verification**: Visual inspection via the 9-augmented-images plot (Section 1).
4. **Training Verification**: Callbacks enforce best-model selection. Training curves plotted for visual anomaly detection (divergence, instability).
5. **Evaluation Verification**: `model.evaluate()` on all three splits provides sanity check (train acc ≥ val acc ≥ test acc under normal conditions). Classification report shows per-class breakdown.
6. **Comparison Verification**: All 7 rows populated in comparison table. Params column cross-checked with `model.count_params()`.

## Performance Considerations

1. **tf.data.AUTOTUNE**: Lets TensorFlow dynamically tune prefetch buffer size and parallel map calls based on available resources
2. **Shuffle buffer (1000)**: Balances memory usage vs. randomization quality — 1000 is sufficient for datasets of this size (~few thousand images)
3. **num_parallel_calls=AUTOTUNE**: In the `map(process_path)` call, enables parallel image decoding/resizing
4. **Mixed precision**: Not included (reference notebook doesn't use it), but could be a future optimization
5. **Checkpoint save_best_only**: Reduces disk I/O by only writing when val_accuracy improves
