# Dataset & Algorithm Selection

## Dataset: Oral Diseases (salmansajid05/oral-diseases)

### Description
- **Source:** Kaggle (salmansajid05/oral-diseases)
- **Type:** Image classification (multi-class)
- **Classes (6):** Calculus, Data caries, Gingivitis, Hypodontia, Mouth Ulcer, Tooth Discoloration
- **Total images:** ~5,000+ (varies per class)
- **Image format:** JPEG/PNG, varying resolutions (resized to 224×224 for training)

### Why This Dataset Is Suitable

1. **Sufficient size for deep learning** — The dataset contains thousands of images across 6 classes, meeting the threshold for effective CNN and transfer learning training.

2. **Real-world clinical relevance** — Oral disease detection is a practical healthcare problem where automated classification can assist dental professionals in early diagnosis.

3. **Multi-class structure** — Having 6 distinct classes allows evaluation of model discriminative ability across visually similar conditions (e.g., caries vs. calculus).

4. **Transfer learning applicability** — Medical image classification benefits significantly from pretrained ImageNet features, making this dataset ideal for comparing custom CNN vs. transfer learning approaches.

5. **Class imbalance present** — The natural class imbalance provides a realistic scenario for evaluating model robustness and the effect of augmentation strategies.

---

## Algorithm Selection

### Model 1: Custom CNN (from scratch)

**Justification:**
- Serves as the baseline to understand if a simple architecture can capture oral disease features
- Allows full control over architecture design (number of layers, filters, regularization)
- Demonstrates understanding of fundamental CNN building blocks
- Reference notebook uses a 3-block CNN, providing a replication target

### Model 2: EfficientNetB0 (Transfer Learning)

**Justification:**
- EfficientNet family achieves state-of-the-art accuracy with fewer parameters through compound scaling
- EfficientNetB0 is the smallest variant — suitable for a dataset of this size (avoids overfitting from oversized models)
- Pretrained on ImageNet (1.2M images, 1000 classes), providing rich low-level feature representations transferable to medical imaging
- Commonly used in medical image classification literature with strong results

### Model 3: ResNet50 (Transfer Learning)

**Justification:**
- ResNet50 introduced skip connections that solve the vanishing gradient problem in deep networks
- Well-established benchmark architecture in image classification
- 25.6M parameters provide a contrast to EfficientNetB0's 4M parameters — allows comparison of model capacity vs. performance
- Widely used in oral/dental disease classification papers, enabling direct comparison with literature
