# 🧠 Head Injury Detection using PyTorch

This repository contains two deep learning pipelines using PyTorch:
- **Classification**: Binary classification model to detect presence of brain hemorrhage in CT scans.
- **Segmentation**: U-Net based model to localize and segment brain hemorrhages in CT scan images.

---

## 📁 Project Structure

```
HEAD-INJURY-DETECTION-PYTORCH/
│
├── data-augmented/
│   ├── images/                  # Contains augmented CT images
│   ├── train_labels.csv         # Training labels (0 or 1 for hemorrhage)
│   ├── test_labels.csv          # Testing labels
│   └── expanded_labels.csv      # Extended label file (if used)
│
├── notebook/
│   └── brain-hemorrhage-detector.ipynb   # Classification model notebook
│
├── segmentation/
│   ├── Data/
│   │   ├── dataset/
│   │   │   ├── images/         # CT images
│   │   │   ├── masks/          # Corresponding segmentation masks
│   │   │   └── augmented/      # Optional: augmented data
│   │   └── test_set/           # Test images + masks for evaluation
│   └── Notebook/
│       └── brain-hemorrhage-segmentation.ipynb   # Segmentation model notebook
```

---

## 🧪 1. Classification Model

### 🔍 Objective
Classify CT images as either:
- `0` → No Hemorrhage
- `1` → Hemorrhage present

### 📘 Dataset
CSV files (`train_labels.csv`, `test_labels.csv`) contain image paths and binary labels. Images are preprocessed and augmented.

### 🚀 How to Run
Use the Jupyter notebook:
```
notebook/brain-hemorrhage-detector.ipynb
```

Includes:
- Dataset preprocessing
- CNN model architecture
- Training loop
- Evaluation metrics (Accuracy, Loss)
- Visualization of training progress

---

## 🧠 2. Segmentation Model

### 🔍 Objective
Segment the region of hemorrhage in CT images using a U-Net architecture.

### 📘 Dataset
- `images/` → Raw CT images
- `masks/` → Corresponding segmentation masks (binary)
- `test_set/` → Test images with masks for inference and visualization

### 🚀 How to Run
Use the Jupyter notebook:
```
segmentation/Notebook/brain-hemorrhage-segmentation.ipynb
```

Includes:
- Custom PyTorch Dataset and DataLoader
- U-Net model definition
- Dice and IoU metric calculation
- Training and validation loops
- Side-by-side visualization of input image, ground truth mask, predicted mask, and overlay

---

## 📊 Evaluation Metrics

- **Classification**:
  - Accuracy
  - Loss Curve
  
- **Segmentation**:
  - Dice Coefficient
  - Intersection over Union (IoU)
  - Overlay visualizations

---

## ⚙️ Requirements

Install the dependencies using:

```bash
pip install -r requirements.txt
```

If no `requirements.txt` is present, install the major ones manually:

```bash
pip install torch torchvision matplotlib pandas opencv-python tqdm
```

---

## 📌 Notes

- Trained on a GPU for faster convergence.
- Normalization and resizing are applied to all images.
- Data is loaded using PyTorch's `DataLoader`.

---

## Contributors
- **Vishwesh Patidar** ([GitHub Profile](https://github.com/VishweshPatidar))
- **Abhigyan Shrivastava** ([GitHub Profile](https://github.com/abhiigyan))
- **Yuvika Yadav**


## ⭐ Star this repo if you found it helpful!
