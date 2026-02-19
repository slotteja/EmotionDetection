# EmotionDetection

Custom ResNet with CBAM attention and hand-crafted kernel initialization for facial emotion recognition — **62% multiclass accuracy** on FER2013's 7-class benchmark, near human-level agreement (~65%).

Four architectures were progressively built and compared.

---

## Results

| Model | Test Accuracy |
|-------|--------------|
| Baseline CNN | 58.26% |
| CNN w/ Custom Kernels | 59.72% |
| SimpleResNet | 53.13% |
| **Custom CNN + CBAM + Kernels** | **62.18%** |



---

## Architecture (Best Model)

- **Custom kernel initialization** — fixed filters targeting eyes, nose, mouth, and jaw to pre-encode facial structure before learned convolutions
- **CBAM attention** — applied after each conv block (64→128→256 filters) to focus on emotionally salient regions like brow position and lip curvature
- **L2 regularization** — dropped validation loss from 1.8 → ~1.4 by epoch 20, controlling overfitting

---

## GradCAM Visualizations

Attention maps confirm the model learned meaningful facial regions per class. See `FER_CNN_GradCam_ValLoss.ipynb` for full outputs.

<img width="1374" height="462" alt="Training and validation metrics" src="https://github.com/user-attachments/assets/d2fa2513-2bd4-4a5a-ab65-9fa1b52589e5" />

---

## Notebooks

- `FER_CNN.ipynb` — model architecture, training, and evaluation
- `FER_CNN_GradCam_ValLoss.ipynb` — model + GradCAM visualizations and validation loss analysis

---

## Dataset

[FER2013](https://www.kaggle.com/datasets/msambare/fer2013) — 35,887 grayscale 48×48 images across 7 emotion classes. Significant class imbalance with "Happy" overrepresented and "Disgust" severely underrepresented.

---

## Tech Stack

`Python` · `PyTorch` · `ResNet` · `CBAM` · `GradCAM` · `Google Colab`

---

## Future Work

- Explore transformer-based models (Vision Transformers) as a stronger baseline
- Address class imbalance more aggressively via oversampling or synthetic augmentation
- Extend to video-based emotion recognition using temporal data
