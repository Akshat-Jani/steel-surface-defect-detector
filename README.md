#  Steel Surface Defect Detection System

A computer vision project for detecting and classifying surface defects on steel products using deep learning and traditional techniques. Built as a complete end-to-end pipeline — from dataset collection to model deployment — this project simulates a real-world quality control system in industrial settings.

---

## 🔍 Project Goal

To develop a vision-based quality control system capable of identifying surface defects in steel sheets using:
- Image classification (CNN)
- Segmentation (U-Net)
- Anomaly detection (Autoencoders / PatchCore)
- Real-time inference (OpenCV)
- Web app deployment (Streamlit/FastAPI)

---

## 🧱 Project Roadmap (10 Phases)

| Phase | Description |
| 1 | Project Setup, Planning, Dataset Collection |
| 2 | Data Annotation & Preprocessing |
| 3 | Image Classification with CNN |
| 4 | Image Segmentation (U-Net / DeepLabV3+) |
| 5 | Anomaly Detection (Autoencoder, PatchCore) |
| 6 | Evaluation & Explainability (Grad-CAM, F1, Confusion Matrix) |
| 7 | Real-time Inference with OpenCV |
| 8 | Web Dashboard (Streamlit or FastAPI) |
| 9 | Edge Deployment (Jetson Nano/Raspberry Pi) |
| 10 | Documentation & GitHub Portfolio Polish |

---

## 🗂️ Folder Structure
steel-surface-defect-detector/
├── data/ # Raw and processed dataset
├── notebooks/ # Jupyter notebooks for exploration and modeling
├── src/ # Python modules (models, preprocessing, utils)
├── models/ # Trained model files
├── results/ # Output images, evaluation results
├── streamlit_app/ # Dashboard code (Phase 8)
├── requirements.txt
├── README.md

---

## 📊 Dataset

Using the [NEU Surface Defect Database](https://github.com/abin24/Surface-Defect-Detection), which includes:
- **6 types of steel defects**: Crazing, Inclusion, Patches, Pitted Surface, Rolled-in Scale, Scratches
- ~1,800 grayscale images (200 per class)

---

## ⚙️ Tech Stack

- **Languages:** Python
- **Libraries:** OpenCV, PyTorch, scikit-learn, NumPy, Matplotlib, Pillow
- **DL Models:** CNN, U-Net, Autoencoder
- **Tools:** Jupyter, Streamlit, Git, GitHub

---

## 📦 Installation

```bash
git clone https://github.com/YOUR_USERNAME/steel-surface-defect-detector.git
cd steel-surface-defect-detector
pip install -r requirements.txt

👤 Author
Akshat
M.Sc. Student in Germany | Mechanical + Machine Vision
Seeking internship opportunities in Computer Vision and AI
