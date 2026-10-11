# Kellgren-Lawrence Knee Osteoarthritis Grading with Deep Learning

A machine learning project comparing classical machine learning and deep convolutional architectures for automated Kellgren-Lawrence (KL) grading of knee X-rays on a small medical dataset.

---

## Overview

Automated grading of knee osteoarthritis from plain radiography can support clinical workflows, but training models on limited data presents severe overfitting and generalisation challenges. This project evaluates four model architectures across increasing levels of complexity to classify knee X-rays into the 5 standard Kellgren-Lawrence grades (0 to 4):

* **Grade 0 (Normal):** No radiographic features of osteoarthritis.


* **Grade 1 (Doubtful):** Possible joint space narrowing and minute osteophytes.


* **Grade 2 (Mild):** Definite osteophytes, possible joint space narrowing.


* **Grade 3 (Moderate):** Moderate multiple osteophytes, definite joint space narrowing, some sclerosis.


* **Grade 4 (Severe):** Large osteophytes, marked joint space narrowing, severe sclerosis, definite bone deformity.



---

## Dataset & Preprocessing

The raw dataset contains approximately 350 knee radiograph images across class folders. Following clean-up and preprocessing, the dataset comprises **313 images**.

### Preprocessing Pipeline

1. **Deduplication:** Removed all images containing `"Osteopenia"` in their filename, as these were duplicated across multiple categories.


2. **Bilateral Split / Cropping:** Filtered two-knee views (detected where $\text{width} > 2 \times \text{height}$) by cropping in half to isolate a single knee joint and preserve class balance.


3. **Bit-Depth Normalisation:** Standardised 16-bit (`I;16`) images by scaling pixel values down (`p / 256`) to match standard 8-bit grayscale (`L`) dynamic range and prevent blown-out contrast.


4. **Square Padding & Resizing:** Padded images to square aspect ratio using edge/centred padding and resampled to **224 × 224** pixels.


5. **Pixel Scaling:** Flattened pixel intensities normalized to $[0, 1]$.



### Class Distribution (Post-Preprocessing)

| Class | Label Index | Count |
| --- | --- | --- |
| **0Normal** | 0 | 70

 |
| **1Doubtful** | 1 | 58

 |
| **2Mild** | 2 | 50

 |
| **3Moderate** | 3 | 73

 |
| **4Severe** | 4 | 62

 |
| **Total** |  | **313**<br> |

---

## Experimental Setup

* **Train / Test Split:** Stratified 80% train / 20% test split held identical across all models for fair comparison.


* **Train / Validation Split:** The training split is subdivided into 80% train / 20% validation (64% / 16% of total data) for neural network optimization.


* **Model Checkpointing:** Best-performing iterations on validation accuracy were checkpointed and evaluated on the test set to avoid overfitted solutions.



---

## Model Architectures & Results

### 1. Random Forest (Baseline)

* **Architecture:** 100 estimators, flattened 1D pixel vector input ($224 \times 224$), default parameters (`min_samples_leaf=1`, `max_features='sqrt'`).


* **Accuracy:** **0.54**

* **Notes:** Captures coarse pixel intensity patterns, but lacks spatial awareness. Frequently confuses Grade 0 with Grade 3.



### 2. Fully Connected Network (FCN)

* **Architecture:** Flattened input $\to$ 3 Dense layers (64 neurons each, ReLU) $\to$ Softmax output (5 classes).


* **Hyperparameters:** 250 epochs, batch size 32.


* **Accuracy:** **0.59**

* **Notes:** Marginally outperforms Random Forest; generalises better along the diagonal, but struggles significantly on Grade 1 (20% class accuracy) due to lack of 2D convolutional feature extraction.



### 3. Convolutional Neural Network (CNN)

* **Architecture:** Conv2D (32 filters, 3×3) + MaxPool2D (2×2) $\to$ Conv2D (64 filters, 3×3) + MaxPool2D (2×2) $\to$ Flatten $\to$ 2 Dense layers (64 neurons, ReLU) $\to$ Softmax.


* **Hyperparameters:** 60 epochs, batch size 32.


* **Accuracy:** **0.71**

* **Notes:** Detects structural local features (joint space narrowing, marginal edge changes), substantially improving accuracy over dense networks.



### 4. MobileNetV2 (Pre-trained Transfer Learning)

* **Architecture:** MobileNetV2 base (ImageNet weights, frozen backbone) $\to$ Global Average Pooling (GAP) $\to$ 2 Dense layers (64 neurons, ReLU) $\to$ Softmax.


* **Hyperparameters:** 80 epochs, batch size 32.


* **Accuracy:** **0.78**

* **Notes:** Best overall performance. Feature extraction learned on large-scale datasets prevents early memorisation and provides superior generalization on minimal sample sizes.



---

### Comparative Performance

| Model | Input Type | Epochs | Overall Accuracy |
| --- | --- | --- | --- |
| **Random Forest** | Flattened 1D

 | —

 | 54%

 |
| **FCN** | Flattened 1D

 | 250

 | 59%

 |
| **CNN** | 224 × 224 Grayscale

 | 60

 | 71%

 |
| **MobileNetV2 (PTN)** | 224 × 224 Feature maps

 | 80

 | **78%**<br> |

---

## Requirements & Environment

To run the notebook or training scripts:

```bash
pip install numpy pillow matplotlib scikit-learn tensorflow

```

---

## Directory Structure

```text
.
├── knees/
│   ├── 0Normal/
│   ├── 1Doubtful/
│   ├── 2Mild/
│   ├── 3Moderate/
│   └── 4Severe/
├── Kellgren_Lawrence_Grading.ipynb
└── README.md

```

---

## References

1. S. Olsson, E. Akbarian, A. Lind, A. S. Razavian, and M. Gordon, "Automating classification of osteoarthritis according to Kellgren-Lawrence in the knee using deep learning in an unfiltered adult population," *BMC Musculoskelet. Disord.*, vol. 22, no. 1, p. 844, 2021.


2. L. Brigato, B. Barz, L. Iocchi, and J. Denzler, "Image classification with small datasets: Overview and benchmark," *IEEE Access*, vol. 10, pp. 49233–49250, 2022.


3. A. J. Metcalfe, M. L. Andersson, and R. Goodfellow et al., "Is knee osteoarthritis a symmetrical disease? Analysis of a 12 year prospective cohort study," *BMC Musculoskelet. Disord.*, vol. 13, p. 153, 2012.


4. M. Sandler, A. Howard, M. Zhu, A. Zhmoginov, and L.-C. Chen, "MobileNetV2: Inverted residuals and linearized bottlenecks," in *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2018, pp. 4510–4520.