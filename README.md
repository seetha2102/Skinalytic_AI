# 🤝🏻 🤝🏼 Skinalytic AI  🤝🏽 🤝🏾 
### AI-Based Skincare Analysis With Explainable AI (XAI)

Skinalytic AI is a deep-learning project that classifies skin conditions specifically **acne** and **pigmentation** using Convolutional Neural Networks (CNNs).  
The system integrates **Explainable AI (XAI)** methods such as **SHAP** and **Grad-CAM** to reveal *why* the model makes its predictions, improving trust, transparency, and reliability.

---

## ✨ Features

- 📸 **Skin Image Classification**  
  Classifies images into *acne* or *pigmentation* using three CNN models (Hybrid CNN, ResNet50V2, MobileNetV2).

- 🔍 **Explainable AI (XAI)**  
  - **SHAP** — Identifies pixel contributions to predictions  
  - **Grad-CAM** — Generates heatmaps showing important regions  
  Helps users understand the model’s decision-making process.

- 🧪 **Sensitivity Analysis**  
  Measures model robustness by adding noise levels and observing prediction changes.

- 🌐 **Web Interface**  
  A simple Flask-based interface that allows users to upload images and view results + explanations.

---

## 🧠 Models Used

### 🔹 Hybrid CNN (ResNet50V2 + MobileNetV2)
A custom-built convolutional network optimized for skin-image features.

### 🔹 ResNet50V2  
A pre-trained deep residual model (Transfer Learning) fine-tuned for binary classification.

### 🔹 MobileNetV2  
A lightweight, mobile-friendly architecture suitable for deployment on edge devices.

---

## 🔍 Explainable AI Methods

### **SHAP**
- Highlights which pixels increase or decrease prediction confidence.  
- Provides a full-pixel importance explanation.

### **Grad-CAM**
- Produces a heatmap showing the most influential regions.  
- Useful for visual validation of model attention.

---

## 📈 Sensitivity Analysis

The project evaluates:
- How model confidence changes after adding 4 different noise levels (0%,1%,5%,10%).  
This tests how much changes in the input affects the output

A stable model should maintain most of its confidence even after noise is added.

---

## 📂 Dataset

Skinalytic AI is trained and evaluated using two benchmark skin-image datasets:

- **SD-198**: A large dataset covering 198 skin disease categories.  
- **Fitzpatrick17k**: A diverse dataset containing skin tones across the Fitzpatrick scale.

These datasets help ensure robust and fair model performance.

---

## 🔗 Links to Models & Dataset
Models: https://drive.google.com/drive/folders/1TPxbxQFCO0n1aTGLkO5FewT0fCSpfuL9?usp=sharing <br>
Dataset: https://drive.google.com/drive/folders/14rlC1O4XhaGtsraDSKWxyO2yfyJagEWS?usp=sharing
