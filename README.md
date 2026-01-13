# Steel Surface Defect Detection System

An end-to-end **industrial computer vision inspection system** built around a **two-stage anomaly-first pipeline**:
Autoencoder-based anomaly detection → CNN-based defect classification → visual explainability → interactive Streamlit dashboard.

---

## 🔍 System Overview (High-Level)

![System Overview](results_%26_assets/System_Overview.PNG)

**Pipeline Flow**  
Raw steel image → Anomaly Detection → Defect Classification → Explainability → Operator Dashboard

This system is designed to simulate a real-world industrial quality control workflow.

---

## 🧠 System Architecture

### Overall Pipeline Architecture
![Pipeline Architecture](results_%26_assets/Pipeline_Architecture.PNG)

**Explanation**
- Autoencoder learns normal steel texture using defect-free samples only  
- Reconstruction error is used as an anomaly score  
- Only anomalous samples are forwarded to the classifier  
- CNN predicts defect category  
- Grad-CAM visualizes regions influencing the prediction  

---

### Module-Level Structure
![Module Structure](results_%26_assets/Structure.PNG)

**Explanation**
- Clear separation of anomaly detection, classification, explainability, and UI  
- Modular design for training, inference, and visualization  

---

## 📊 Model Training & Evaluation Results

### 1️⃣ CNN Training Behavior
![Training Curves](results_%26_assets/training_curves.png)

Shows convergence behavior and training stability.

---

### 2️⃣ Defect Classification Performance
![Confusion Matrix](results_%26_assets/confusion_matrix.png)

Demonstrates class-wise performance across industrial defect categories.

---

### 3️⃣ Model Explainability (Grad-CAM)
![Grad-CAM Example](results_%26_assets/gradcam_output.png)

Verifies that the CNN focuses on defect-relevant regions. Given Image: "Scratch" on a steel surface

---

## 🖥️ Interactive Streamlit Dashboard

### 1️⃣ Dashboard Home & Image Upload
![Dashboard Upload](results_%26_assets/Streamlit_Upload.png)

User uploads steel surface images for inspection.

---

### 2️⃣ Anomaly Detection Output & Defect Classification Result
![Dashboard Anomaly](results_%26_assets/Streamlit_Anomaly_Classification(1).png)

Displays anomaly score, normal/anomalous decision, predicted defect class, and confidence scores.

---

### 3️⃣ Explainability View (Grad-CAM)
![Dashboard GradCAM]((results_%26_assets/Streamlit_Grad_CAM.png)

Visual justification of CNN decision for operator trust.

---

## ⭐ Key Project Highlights

- Two-stage anomaly-first inspection pipeline  
- Autoencoder-based unsupervised anomaly detection  
- MobileNetV2 transfer learning for defect classification  
- Grad-CAM explainability integration  
- Interactive Streamlit dashboard  
- Modular, production-style project structure  

---

## 📁 Project Structure

```
steel-surface-defect-detector/
├── anomaly_detection/
├── defect_classification/
├── explainability/
├── streamlit_app/
├── data/
├── models/        # not tracked (large files)
├── results_&_assets/
├── docs/
├── README.md
└── LICENSE
```

---

## 🧠 Key Design Decisions

- Anomaly-first filtering to reduce false positives  
- Autoencoder chosen to avoid exhaustive defect labeling  
- Grad-CAM used for interpretability, not decision logic  
- UI separated from model logic  
- Model binaries excluded to keep repository lightweight  

---

## 🛠️ Technologies Used

TensorFlow · Keras · Autoencoders · CNNs · MobileNetV2 · Grad-CAM · OpenCV · Streamlit · NumPy · Matplotlib · Scikit-learn

---

## 👤 Author

**Akshat Jani**  
M.Sc. Mechatronics — University of Siegen  
Focus: Computer Vision · Industrial AI · Robotics
