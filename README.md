# 🩺 PNEUMONIA DETECTION SYSTEM

This project detects **Pneumonia from chest X-ray images** using **Transfer Learning (ResNet50)** and provides an **interactive Streamlit dashboard** for real-time predictions.

---

## Features
- Data preprocessing including resizing, normalization, and data augmentation.
- Class balancing with `compute_class_weight` for accurate training.
- Transfer Learning using **ResNet50 pretrained on ImageNet**.
- Fine-tuning the last layers of ResNet50 to improve accuracy.
- Streamlit dashboard with image upload and real-time predictions.
- Visualizations including confidence bar and pie chart.

---

## Description
This project implements a complete pipeline for Pneumonia detection:

- `chest_xray/` : Dataset folder with `train`, `val`, and `test` directories.
- `main.py` : Streamlit application for real-time predictions.
- `PneumoniaDetection_model.py` : Model training script with transfer learning and fine-tuning.
- `requirements.txt` : Python dependencies.
- `images/` : Screenshots of the Streamlit dashboard.
- `.gitignore` : Excludes model files, datasets, and IDE configurations.

> ⚠️ Note: The trained model (`resnet50_pneumonia_model.h5`) is **not included** due to GitHub size limits. You can train it locally using the provided script.

---

## Usage

1. Install dependencies:
```bash
pip install -r requirements.txt
```
2. Run the Streamlit app:
 ```bash
streamlit run main.py
```
4. Upload a chest X-ray image in the app to get predictions. The dashboard shows:
- Predicted class: Normal or Pneumonia

- Confidence percentage

- Pie chart visualization
## Notes

- Model uses Transfer Learning + Fine-Tuning, reducing training time and improving accuracy.
- Data augmentation helps the model generalize better.
- Large files like .h5 models are excluded from GitHub; train locally to generate them.
