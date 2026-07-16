# Hyperparameters — Explanation & Tuning Strategy

## What Are Hyperparameters?

Hyperparameters are configuration values set **before** training begins (not learned from data). They control model architecture, training behavior, and regularization strength.

---

## Hyperparameters Used in This Experiment

### Architecture Hyperparameters

| Parameter | Description | Values Used |
|---|---|---|
| **Number of Conv blocks** | Depth of the CNN feature extractor | 2–4 (C3 search), 3 (C1), 4 (C2) |
| **Filters per block** | Number of feature maps per convolutional layer | 32, 64, 128, 256, 512 |
| **Dense units** | Number of neurons in fully-connected head | 128, 256, 512, 1024 |
| **Kernel size** | Spatial extent of convolution filter | 3×3 (all models) |
| **Padding** | Whether to preserve spatial dimensions | 'valid' (C1), 'same' (C2/C3) |

### Training Hyperparameters

| Parameter | Description | Values Used |
|---|---|---|
| **Learning rate** | Step size for gradient descent updates | 0.001 (C1), 1e-4 (C2), [1e-3, 5e-4, 1e-4, 5e-5] (C3 search) |
| **Batch size** | Number of samples per gradient update | 32 (fixed) |
| **Epochs** | Maximum training iterations over full dataset | 20 (C1), 100 (C2 CNN), 50 (C2 transfer), 30 (C3 search) |
| **Optimizer** | Algorithm for weight updates | Adam (all models) |

### Regularization Hyperparameters

| Parameter | Description | Values Used |
|---|---|---|
| **Dropout rate** | Fraction of neurons randomly zeroed during training | 0.2–0.5 (progressive), [0.2–0.6] (C3 search) |
| **L2 regularization (weight decay)** | Penalty on large weights to prevent overfitting | 1e-4, 5e-4, 1e-3 |
| **Batch Normalization** | Normalizes layer inputs for stable training | Used in C2/C3 configs |

### Callback Hyperparameters

| Parameter | Description | Value |
|---|---|---|
| **EarlyStopping patience** | Epochs to wait before stopping if no improvement | 10 |
| **ReduceLROnPlateau patience** | Epochs to wait before reducing learning rate | 8 |
| **ReduceLROnPlateau factor** | Multiplier to reduce LR by | 0.1 (LR × 0.1) |
| **ModelCheckpoint monitor** | Metric to track for saving best model | val_accuracy |

### Transfer Learning Hyperparameters

| Parameter | Description | Values Used |
|---|---|---|
| **Base model frozen/unfrozen** | Whether pretrained layers are updated during training | Frozen (C1, C3), Unfrozen (C2) |
| **Freeze-unfreeze epochs** | How long to train head before unfreezing base | 10 epochs (C2 EfficientNet) |
| **Pretrained weights** | Source of initial backbone weights | ImageNet |

---

## Tuning Strategy: Keras Tuner (BayesianOptimization)

### Why Bayesian Optimization?

- More efficient than Grid Search (exponential combinations) or Random Search (no learning)
- Builds a probabilistic model of the objective function (val_accuracy)
- Each trial is informed by previous results → converges faster
- With 10 trials, explores the search space effectively for our dataset size

### Search Space (C3 Configurations)

| Parameter | Type | Range |
|---|---|---|
| `learning_rate` | Choice | [1e-3, 5e-4, 1e-4, 5e-5] |
| `dense_units` | Choice | [128, 256, 512] |
| `dropout_rate` | Float | 0.2 to 0.6 (step 0.1) |
| `l2_rate` | Choice | [1e-3, 5e-4, 1e-4] |
| `num_blocks` (CNN only) | Int | 2 to 4 |
| `filters_start` (CNN only) | Choice | [32, 64] |

### Tuning Configuration

```
Tuner: BayesianOptimization
Objective: val_accuracy (maximize)
Max trials: 10
Executions per trial: 1
Epochs per trial: 30
```

After search completes, the best hyperparameter combination is retrained for full epochs (100 for CNN, 50 for transfer learning) with standard callbacks.
