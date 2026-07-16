# Evaluation Metrics — Description & Justification

## Metrics Used

### 1. Accuracy (Training, Validation, Test)

**Definition:** The proportion of correctly classified samples out of total samples.

$$\text{Accuracy} = \frac{\text{Correct Predictions}}{\text{Total Predictions}}$$

**Why used:** Primary metric for overall model performance. Monitored during training (val_accuracy) for callbacks (EarlyStopping, ModelCheckpoint).

**Limitation:** Can be misleading with imbalanced classes — a model predicting the majority class achieves high accuracy without learning minority classes.

---

### 2. Loss (Categorical Crossentropy)

**Definition:** Measures the difference between predicted probability distribution and true one-hot labels. Lower = better.

$$\text{Loss} = -\sum_{i=1}^{C} y_i \log(\hat{y}_i)$$

**Why used:** The optimization objective for all models. Training loss shows learning progress; the gap between train and val loss reveals overfitting.

---

### 3. Precision (Per-class)

**Definition:** Of all samples predicted as class X, what proportion actually belongs to class X.

$$\text{Precision} = \frac{\text{True Positives}}{\text{True Positives + False Positives}}$$

**Why used:** Important in medical diagnosis — high precision means fewer false alarms (e.g., incorrectly diagnosing a healthy tooth as having caries).

---

### 4. Recall (Per-class)

**Definition:** Of all actual class X samples, what proportion was correctly identified.

$$\text{Recall} = \frac{\text{True Positives}}{\text{True Positives + False Negatives}}$$

**Why used:** Critical in healthcare — high recall means fewer missed diagnoses (e.g., not missing an actual case of gingivitis).

---

### 5. F1-Score (Per-class)

**Definition:** Harmonic mean of precision and recall, balancing both concerns.

$$\text{F1} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

**Why used:** Single metric that balances precision and recall, especially useful when classes are imbalanced.

---

### 6. Confusion Matrix

**Definition:** An N×N matrix (N = number of classes) showing the count of predictions vs. true labels for every class pair.

**Why used:** Reveals which specific classes are being confused with each other. For oral diseases, shows if the model confuses visually similar conditions (e.g., caries vs. calculus).

---

## Metric Selection Justification

| Metric | Purpose | When It Matters Most |
|---|---|---|
| Accuracy | Overall performance | Balanced datasets |
| Precision | Minimize false positives | When false alarms are costly |
| Recall | Minimize false negatives | When missed diagnoses are dangerous |
| F1-Score | Balance precision/recall | Imbalanced datasets |
| Confusion Matrix | Error analysis | Understanding model weaknesses |
| Loss curves | Training dynamics | Detecting overfitting/underfitting |

All metrics are computed using scikit-learn (`classification_report`, `confusion_matrix`) and TensorFlow (`model.evaluate()`).
