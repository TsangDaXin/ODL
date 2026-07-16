# Conclusion & Critical Analysis

## Best Model Per Family

| Model Family | Best Config | Test Accuracy | Parameters |
|---|---|---|---|
| Custom CNN | *(fill: C1/C2/C3)* | *(fill)%* | *(fill) M* |
| EfficientNetB0 | *(fill: C1/C2/C3)* | *(fill)%* | *(fill) M* |
| ResNet50 | *(fill: C1/C2/C3)* | *(fill)%* | *(fill) M* |

## Overall Best Model

**Recommended model:** *(fill)*

**Justification:**
- Test accuracy: *(fill)*
- Parameter efficiency: *(fill)*
- Generalization gap (train acc - test acc): *(fill)*
- Validation loss: *(fill)*

---

## Critical Analysis

### Why C2/C3 Outperformed C1 (if applicable)

- *(fill: regularization prevented overfitting)*
- *(fill: more training epochs allowed better convergence)*
- *(fill: augmentation improved generalization)*
- *(fill: batch normalization stabilized training)*

### Why Transfer Learning Outperformed Custom CNN (if applicable)

- *(fill: pretrained features capture universal visual patterns)*
- *(fill: ImageNet pretraining provides strong low-level features)*
- *(fill: fewer epochs needed due to warm-start initialization)*

### Limitations

- *(fill: dataset is relatively small for deep learning)*
- *(fill: class imbalance affects minority class recall)*
- *(fill: images collected from various sources may have inconsistent quality)*
- *(fill: augmentation is limited to flip/rotation — more aggressive methods may help)*

---

## Comparison with Literature

| Study | Model | Dataset Size | Accuracy | Our Result |
|---|---|---|---|---|
| [Author 1] | ResNet50 | [X] images | [X]% | *(fill)%* |
| [Author 2] | EfficientNet | [X] images | [X]% | *(fill)%* |
| [Author 3] | Custom CNN | [X] images | [X]% | *(fill)%* |

**Analysis:** *(fill: explain if your results are comparable, better, or worse than literature and why)*

---

## Future Improvements

1. **Larger and more diverse dataset** — Collect additional images from clinical settings to improve model robustness and class balance.

2. **Advanced augmentation** — Implement CutMix, MixUp, or GAN-based augmentation to synthetically expand the training set.

3. **Ensemble methods** — Combine predictions from top-performing models (soft voting) to reduce individual model weaknesses.

4. **Attention mechanisms** — Add SE-blocks or CBAM attention to help models focus on disease-relevant regions.

5. **Cross-validation** — Use k-fold cross-validation instead of single train/val/test split for more robust performance estimates.

6. **Explainability** — Apply Grad-CAM or LIME to visualize which image regions drive model predictions, increasing clinical trust.

7. **Deployment readiness** — Convert best model to TFLite or ONNX for mobile/edge deployment in dental clinics.
