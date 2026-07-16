# Requirements Document

## Introduction

This document defines the requirements for a full experimental Jupyter notebook that performs Oral Disease Image Classification using the Kaggle dataset "salmansajid05/oral-diseases". The notebook trains and evaluates multiple deep learning model families (Custom CNN, EfficientNetB0, ResNet50) across different configurations (exact replica, improved architecture, hyperparameter-tuned) to identify the best model for oral disease classification across 6 classes. The notebook is designed for reproducibility, research alignment, and deployment readiness assessment.

## Glossary

- **Notebook**: The Jupyter notebook (.ipynb) that contains all code, visualizations, and markdown documentation for the experiment
- **Dataset**: The Kaggle dataset "salmansajid05/oral-diseases" containing oral disease images organized in class-named subfolders
- **Data_Pipeline**: The module responsible for loading, splitting, augmenting, and serving image data as tf.data.Dataset objects
- **Custom_CNN**: A from-scratch convolutional neural network model family with multiple configurations
- **EfficientNetB0_Model**: A transfer learning model family using the EfficientNetB0 pretrained backbone
- **ResNet50_Model**: A transfer learning model family using the ResNet50 pretrained backbone
- **Configuration_C1**: Exact replica of the reference notebook architecture for a given model family
- **Configuration_C2**: An improved architecture variant for a given model family
- **Configuration_C3**: A Keras Tuner hyperparameter-optimized variant derived from the best C1/C2 winner
- **Evaluation_Block**: A reusable set of evaluation outputs including accuracy/loss curves, classification report, and confusion matrix
- **Comparison_Table**: A summary DataFrame comparing all trained models on key metrics
- **DATASET_PATH**: A configurable variable at the top of the notebook pointing to the root directory of the dataset
- **Keras_Tuner**: The hyperparameter optimization library used for Configuration_C3 searches
- **Callbacks**: A set of Keras callback objects (ReduceLROnPlateau, EarlyStopping, ModelCheckpoint) applied during training

## Requirements

### Requirement 1: Global Configuration and Reproducibility

**User Story:** As a researcher, I want all global settings defined in a single cell at the top of the notebook, so that I can reproduce experiments and modify parameters easily.

#### Acceptance Criteria

1. THE Notebook SHALL define the following global constants in the first code cell: SEED=42, IMAGE_SIZE=(224,224), BATCH_SIZE=32, NUM_CLASSES=6, EPOCHS_BASELINE=100, EPOCHS_PRETRAINED=50, EPOCHS_TUNER=30, and DATASET_PATH as a configurable Path variable
2. THE Notebook SHALL set random seeds via random.seed(SEED), np.random.seed(SEED), and tf.random.set_seed(SEED) immediately after imports to ensure reproducible results
3. THE Notebook SHALL use train/validation/test split ratios of 70/15/15 with stratified sampling and random_state=47
4. THE Notebook SHALL normalize all pixel values by dividing by 255.0 within the process_path function
5. THE Notebook SHALL one-hot encode all class labels using tf.one_hot with depth=NUM_CLASSES
6. THE Notebook SHALL use Adam optimizer with learning_rate=1e-4 and categorical_crossentropy loss for all model training
7. THE Notebook SHALL apply the following callbacks to all training runs: ReduceLROnPlateau(monitor="val_accuracy", factor=0.1, patience=8), EarlyStopping(monitor="val_accuracy", patience=10, restore_best_weights=True), ModelCheckpoint(save_best_only=True, monitor="val_accuracy", mode="max")
8. THE Notebook SHALL generate unique checkpoint filenames for each model configuration using the pattern {model_family}_{config}_best.h5 (e.g., cnn_c1_best.h5, eff_c2_best.h5)

### Requirement 2: Dataset Loading and Filtering

**User Story:** As a researcher, I want the notebook to automatically load images from the dataset directory structure while skipping irrelevant folders, so that I have a clean dataset without manual intervention.

#### Acceptance Criteria

1. WHEN the Data_Pipeline walks the dataset directory, THE Data_Pipeline SHALL collect image file paths (extensions: .jpg, .jpeg, .png, .bmp) and corresponding class labels from the immediate class-named subfolders of DATASET_PATH into a Pandas DataFrame with columns "filepath" and "label"
2. WHEN the Data_Pipeline encounters a folder whose name contains "yolo" (case-insensitive match), THE Data_Pipeline SHALL skip that folder entirely and exclude its contents from the dataset
3. WHEN the Data_Pipeline completes loading the DataFrame, THE Data_Pipeline SHALL print df.head(), the total image count, and per-class value_counts()
4. IF the DATASET_PATH does not exist or contains no files matching the accepted image extensions (.jpg, .jpeg, .png, .bmp), THEN THE Data_Pipeline SHALL raise an error message indicating the dataset location issue
5. IF the Data_Pipeline encounters a subfolder containing zero valid image files after filtering, THEN THE Data_Pipeline SHALL exclude that class from the DataFrame and print a warning indicating the empty class name

### Requirement 3: Data Splitting

**User Story:** As a researcher, I want a stratified train/validation/test split so that class distributions are preserved across all subsets.

#### Acceptance Criteria

1. THE Data_Pipeline SHALL split the dataset into train (70%), validation (15%), and test (15%) subsets using sklearn train_test_split with stratify parameter and random_state=47
2. THE Data_Pipeline SHALL perform the split in two stages: first split 70/30 for train vs. temp, then split temp 50/50 for validation vs. test (achieving 15/15)
3. THE Data_Pipeline SHALL preserve the class distribution ratio across all three subsets within a tolerance of ±2 percentage points per class

### Requirement 4: tf.data.Dataset Pipeline Construction

**User Story:** As a researcher, I want efficient data pipelines using tf.data.Dataset so that training is performant and GPU-optimized.

#### Acceptance Criteria

1. THE Data_Pipeline SHALL construct tf.data.Dataset objects for train, validation, and test sets using a process_path function that reads the file from disk, decodes it as JPEG, resizes to IMAGE_SIZE, casts to float32, divides by 255.0, and one-hot encodes the label
2. THE Data_Pipeline SHALL resize all images to IMAGE_SIZE (224, 224) during the process_path function
3. WHILE processing the training dataset, THE Data_Pipeline SHALL apply runtime augmentation consisting of RandomFlip("horizontal") and RandomRotation(0.05) as Keras preprocessing layers inside the model
4. THE Data_Pipeline SHALL NOT apply runtime augmentation to validation or test datasets
5. THE Data_Pipeline SHALL shuffle the training dataset with a buffer size of at least 1000 samples before batching
6. THE Data_Pipeline SHALL batch all datasets with BATCH_SIZE=32 and apply prefetching with buffer size set to tf.data.AUTOTUNE

### Requirement 5: Data Visualization

**User Story:** As a researcher, I want to visually inspect the data and augmentation effects so that I can verify the pipeline is working correctly.

#### Acceptance Criteria

1. THE Notebook SHALL display a 3x3 grid of 9 randomly selected sample images from the training set with their corresponding class labels displayed as titles above each image
2. THE Notebook SHALL display a 3x3 grid of 9 augmented versions of a single randomly selected training image to demonstrate the augmentation pipeline effects, with each augmented output shown as a separate subplot
3. THE Notebook SHALL display a bar chart showing the number of images per class across the full dataset, with class names on the x-axis and image counts on the y-axis
4. THE Notebook SHALL include axis labels and a figure title on all visualization plots

### Requirement 6: Custom CNN Configuration C1

**User Story:** As a researcher, I want an exact replica of the reference notebook's custom CNN architecture so that I can establish a baseline for comparison.

#### Acceptance Criteria

1. THE Custom_CNN Configuration_C1 SHALL implement exactly 4 convolutional blocks with the following filter progression: Block 1 (64 filters), Block 2 (128 filters), Block 3 (256 filters), Block 4 (256 filters), each block containing two Conv2D(kernel_size=3×3, activation="relu", padding="same", kernel_regularizer=L2(1e-4)) layers followed by BatchNormalization, MaxPooling2D(2×2), and Dropout
2. THE Custom_CNN Configuration_C1 SHALL use dropout rates of 0.2, 0.3, 0.4, 0.4 for blocks 1 through 4 respectively
3. THE Custom_CNN Configuration_C1 SHALL use GlobalAveragePooling2D followed by Dense(1024, activation="relu", kernel_regularizer=L2(1e-4)) → Dropout(0.5) → Dense(512, activation="relu", kernel_regularizer=L2(1e-4)) → Dropout(0.3) → Dense(NUM_CLASSES, activation="softmax")
4. THE Custom_CNN Configuration_C1 SHALL accept input shape (224, 224, 3) and apply the augmentation layers before the convolutional blocks
5. THE Custom_CNN Configuration_C1 SHALL train for a maximum of EPOCHS_BASELINE (100) epochs with the defined callbacks and save to cnn_c1_best.h5

### Requirement 7: Custom CNN Configuration C2

**User Story:** As a researcher, I want an improved custom CNN architecture so that I can evaluate whether additional depth improves performance.

#### Acceptance Criteria

1. THE Custom_CNN Configuration_C2 SHALL retain all 4 convolutional blocks from Configuration_C1 and add a 5th convolutional block containing two Conv2D(512, kernel_size=3×3, activation="relu", padding="same", kernel_regularizer=L2(1e-4)) layers followed by BatchNormalization, MaxPooling2D(2×2), and Dropout(0.5)
2. THE Custom_CNN Configuration_C2 SHALL use GlobalAveragePooling2D followed by Dense(512, activation="relu", kernel_regularizer=L2(1e-4)) → Dropout(0.3) → Dense(NUM_CLASSES, activation="softmax") as the classification head
3. THE Custom_CNN Configuration_C2 SHALL train for a maximum of EPOCHS_BASELINE (100) epochs with the defined callbacks and save to cnn_c2_best.h5

### Requirement 8: EfficientNetB0 Configuration C1

**User Story:** As a researcher, I want a transfer learning model using EfficientNetB0 with all layers unfrozen so that I can evaluate full fine-tuning performance.

#### Acceptance Criteria

1. THE EfficientNetB0_Model Configuration_C1 SHALL load EfficientNetB0 pretrained on ImageNet with include_top=False and input_shape=(224, 224, 3)
2. THE EfficientNetB0_Model Configuration_C1 SHALL keep all base layers unfrozen (trainable=True) for full fine-tuning
3. THE EfficientNetB0_Model Configuration_C1 SHALL attach a head consisting of GlobalAveragePooling2D → BatchNormalization → Dense(256, activation="relu", kernel_regularizer=L2(0.0005)) → Dropout(0.4) → Dense(NUM_CLASSES, activation="softmax")
4. THE EfficientNetB0_Model Configuration_C1 SHALL compile with Adam(learning_rate=1e-4), categorical_crossentropy loss, and train for a maximum of EPOCHS_PRETRAINED (50) epochs, saving to eff_c1_best.h5

### Requirement 9: EfficientNetB0 Configuration C2

**User Story:** As a researcher, I want a freeze-then-unfreeze training strategy for EfficientNetB0 so that I can evaluate whether gradual unfreezing improves convergence.

#### Acceptance Criteria

1. THE EfficientNetB0_Model Configuration_C2 SHALL load EfficientNetB0 pretrained on ImageNet with include_top=False, freeze all base layers (trainable=False), and train only the classification head for 10 epochs in the first phase
2. THE EfficientNetB0_Model Configuration_C2 SHALL unfreeze all base layers (trainable=True), recompile the model with Adam(learning_rate=1e-4), and continue training for the remaining epochs (up to 40 epochs) in the second phase
3. THE EfficientNetB0_Model Configuration_C2 SHALL use an expanded classification head consisting of GlobalAveragePooling2D → BatchNormalization → Dense(512, activation="relu", kernel_regularizer=L2(1e-4)) → Dropout(0.5) → Dense(256, activation="relu", kernel_regularizer=L2(1e-4)) → Dropout(0.3) → Dense(NUM_CLASSES, activation="softmax")
4. THE EfficientNetB0_Model Configuration_C2 SHALL apply freshly instantiated callbacks (ReduceLROnPlateau, EarlyStopping, ModelCheckpoint) at the start of each training phase and save to eff_c2_best.h5

### Requirement 10: EfficientNetB0 Configuration C3

**User Story:** As a researcher, I want a hyperparameter-tuned EfficientNetB0 configuration so that I can find an optimal architecture within a defined search space.

#### Acceptance Criteria

1. THE EfficientNetB0_Model Configuration_C3 SHALL use Keras Tuner BayesianOptimization with max_trials=10, executions_per_trial=1, and objective="val_accuracy"
2. THE EfficientNetB0_Model Configuration_C3 SHALL search over: learning_rate from [1e-3, 5e-4, 1e-4, 5e-5], dense_units from [128, 256, 512], dropout_rate as Float(0.2, 0.6, step=0.1), and l2_rate from [1e-3, 5e-4, 1e-4]
3. THE EfficientNetB0_Model Configuration_C3 SHALL use a frozen EfficientNetB0 base → GlobalAveragePooling2D → BatchNormalization → Dense(dense_units, activation="relu", kernel_regularizer=L2(l2_rate)) → Dropout(dropout_rate) → Dense(NUM_CLASSES, activation="softmax") as the trial architecture
4. THE EfficientNetB0_Model Configuration_C3 SHALL train each trial for a maximum of EPOCHS_TUNER (30) epochs with the standard callbacks
5. THE EfficientNetB0_Model Configuration_C3 SHALL print the best hyperparameters after search, retrain the best configuration for EPOCHS_PRETRAINED (50) epochs with standard callbacks, and save to eff_c3_best.h5

### Requirement 11: ResNet50 Configuration C1

**User Story:** As a researcher, I want a transfer learning model using ResNet50 with all layers unfrozen so that I can compare its performance against EfficientNetB0.

#### Acceptance Criteria

1. THE ResNet50_Model Configuration_C1 SHALL load ResNet50 pretrained on ImageNet with include_top=False and input_shape=(224, 224, 3)
2. THE ResNet50_Model Configuration_C1 SHALL keep all base layers unfrozen (trainable=True) for full fine-tuning
3. THE ResNet50_Model Configuration_C1 SHALL attach a head consisting of GlobalAveragePooling2D → BatchNormalization → Dense(256, activation="relu", kernel_regularizer=L2(0.0005)) → Dropout(0.4) → Dense(NUM_CLASSES, activation="softmax")
4. THE ResNet50_Model Configuration_C1 SHALL compile with Adam(learning_rate=1e-4), categorical_crossentropy loss, and train for a maximum of EPOCHS_PRETRAINED (50) epochs, saving to res_c1_best.h5

### Requirement 12: ResNet50 Configuration C2

**User Story:** As a researcher, I want a freeze-then-unfreeze training strategy for ResNet50 so that I can evaluate whether gradual unfreezing improves convergence compared to full fine-tuning.

#### Acceptance Criteria

1. THE ResNet50_Model Configuration_C2 SHALL load ResNet50 pretrained on ImageNet with include_top=False, freeze all base layers (trainable=False), and train only the classification head for 10 epochs in the first phase
2. WHEN the first phase of 10 epochs completes, THE ResNet50_Model Configuration_C2 SHALL unfreeze all base layers (trainable=True), recompile the model with Adam(learning_rate=1e-4), and continue training for the remaining epochs (up to 40 epochs) in the second phase
3. THE ResNet50_Model Configuration_C2 SHALL use an expanded classification head consisting of GlobalAveragePooling2D → BatchNormalization → Dense(512, activation="relu", kernel_regularizer=L2(1e-4)) → Dropout(0.5) → Dense(256, activation="relu", kernel_regularizer=L2(1e-4)) → Dropout(0.3) → Dense(NUM_CLASSES, activation="softmax")
4. THE ResNet50_Model Configuration_C2 SHALL apply freshly instantiated callbacks (ReduceLROnPlateau, EarlyStopping, ModelCheckpoint) at the start of each training phase and save to res_c2_best.h5

### Requirement 13: Model Evaluation Block

**User Story:** As a researcher, I want a consistent evaluation procedure applied to every trained model so that I can fairly compare results across configurations.

#### Acceptance Criteria

1. WHEN a model finishes training, THE Evaluation_Block SHALL load the best checkpoint weights and plot training and validation accuracy curves with the best epoch (highest val_accuracy) marked with a scatter point
2. WHEN a model finishes training, THE Evaluation_Block SHALL plot training and validation loss curves with the best epoch marked with a scatter point
3. WHEN a model finishes training, THE Evaluation_Block SHALL run model.evaluate() on train, validation, and test sets and print the resulting accuracy and loss values
4. WHEN a model finishes training, THE Evaluation_Block SHALL generate predictions on the test set using argmax decoding on both predictions and true labels, then produce a sklearn classification_report showing per-class precision, recall, and F1-score
5. WHEN a model finishes training, THE Evaluation_Block SHALL display a confusion matrix as a seaborn heatmap with "coolwarm" colormap, annotations enabled, and class names on both axes

### Requirement 14: Model Comparison Table

**User Story:** As a researcher, I want a summary comparison table so that I can quickly identify the best-performing model configuration.

#### Acceptance Criteria

1. THE Comparison_Table SHALL include columns for: Model family, Configuration, Parameters (in millions rounded to 2 decimal places), Train Accuracy, Validation Accuracy, Test Accuracy, Train Loss, Validation Loss, Test Loss, and Epochs run, with all accuracy values displayed as percentages to 2 decimal places and all loss values displayed to 4 decimal places
2. THE Comparison_Table SHALL include rows for all 7 trained model configurations (Custom CNN C1, C2; EfficientNetB0 C1, C2, C3; ResNet50 C1, C2)
3. THE Comparison_Table SHALL visually distinguish the row with the highest test accuracy by applying bold formatting or a distinct background color visible in Jupyter Notebook
4. IF two or more model configurations share the same highest test accuracy, THEN THE Comparison_Table SHALL highlight all tied rows and use the lowest validation loss as the tiebreaker to indicate the recommended best model

### Requirement 15: Conclusion Section

**User Story:** As a researcher, I want a concluding markdown cell summarizing the experiment so that readers can quickly understand the key findings.

#### Acceptance Criteria

1. THE Notebook SHALL include a markdown cell stating the best configuration for each of the 3 model families (Custom CNN, EfficientNetB0, ResNet50) by name and its corresponding test accuracy percentage
2. THE Notebook SHALL include a markdown cell recommending the best overall model for deployment, justified by referencing at least 2 metrics from the Comparison_Table (e.g., test accuracy, parameter count, validation loss, or generalization gap)
3. THE Notebook SHALL include a markdown cell listing at least 3 specific improvement suggestions for future work (e.g., additional architectures, data collection, training strategies)
4. THE Notebook SHALL place all conclusion markdown cells after the Comparison_Table section

### Requirement 16: Notebook Code Quality and Structure

**User Story:** As a researcher, I want well-structured, documented code so that the notebook is readable, maintainable, and reproducible.

#### Acceptance Criteria

1. THE Notebook SHALL use one `# %%` code cell per logical block with a markdown comment header (# %% [markdown]) before each major section
2. THE Notebook SHALL include markdown header cells (## level) before each major section: Data Pipeline, Custom CNN, EfficientNetB0, ResNet50, Comparison, Conclusion
3. THE Notebook SHALL include inline comments on non-obvious lines such as AUTOTUNE usage, shuffle buffer rationale, process_path operations, and callback parameter choices
4. THE Notebook SHALL define DATASET_PATH = Path("YOUR_PATH_HERE") as the first configurable variable in the configuration cell
5. THE Notebook SHALL generate unique checkpoint file paths using the naming convention: cnn_c1_best.h5, cnn_c2_best.h5, eff_c1_best.h5, eff_c2_best.h5, eff_c3_best.h5, res_c1_best.h5, res_c2_best.h5
6. THE Notebook SHALL organize all imports in a single cell at the top of the notebook, grouped by standard library, third-party, and TensorFlow/Keras modules
