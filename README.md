# 🩺 Chest Cancer Classification using Deep Learning  

![Python](https://img.shields.io/badge/Python-3.9-blue?style=for-the-badge&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Deep%20Learning-FF6F00?style=for-the-badge&logo=tensorflow)
![Keras](https://img.shields.io/badge/Keras-Neural%20Networks-D00000?style=for-the-badge&logo=keras)
![Flask](https://img.shields.io/badge/Flask-Web%20App-black?style=for-the-badge&logo=flask)
![DVC](https://img.shields.io/badge/DVC-MLOps-purple?style=for-the-badge&logo=dvc)
![Docker](https://img.shields.io/badge/Docker-Containerization-2496ED?style=for-the-badge&logo=docker)
![AWS](https://img.shields.io/badge/AWS-Cloud-232F3E?style=for-the-badge&logo=amazon-aws)

---

## 📌 Project Overview  

This project implements an **AI-powered image classification system** to detect and classify **chest cancer from medical images**.  

It leverages:  
- 🧠 **Convolutional Neural Networks (CNNs)**  
- 🔄 **Transfer Learning (VGG16, ResNet, etc.)**  
- ⚡ **MLOps practices with DVC & CI/CD**  

The goal is to assist **radiologists** in:  
- 🩻 Early diagnosis  
- 🏥 Improved treatment decisions  
- ⏱️ Reducing manual diagnostic effort  

The project follows a **modular, production-ready ML pipeline** with configurable components and cloud deployment support.  

---

## 🔄 Workflow  

### 1️⃣ Data Ingestion  
- 📂 Load chest X-ray dataset  
- 🔀 Split into training / validation / testing sets  
- ✅ Ensure reproducibility & data quality  

### 2️⃣ Base Model Preparation  
- ⚙️ Initialize CNN with **transfer learning** (e.g., VGG16/ResNet)  
- 🧩 Freeze base layers, add custom classification head  
- 🏗️ Compile with optimizer & loss function  

### 3️⃣ Model Training  
- 🎯 Train CNN on chest cancer dataset  
- 📊 Log metrics (**accuracy, loss**)  
- 💾 Save trained model → `model.h5`  

### 4️⃣ Model Evaluation  
- 🧪 Test on unseen data  
- 📑 Save metrics → `scores.json`  
- 🛡️ Ensure reliability in medical context  

### 5️⃣ Pipeline Orchestration  
- 🔄 Modular pipelines → `src/cnnClassifier/pipeline/`  
- ⚡ Config-driven (`config.yaml`, `params.yaml`) for reproducibility  

### 6️⃣ Deployment (Flask Web App)  
- 🌐 `app.py` provides an interface to **upload chest images**  
- 🤖 Model predicts **Cancer / No Cancer** in real time  
- 🎨 UI served with `templates/index.html`  

### 7️⃣ MLOps with DVC  
- 📦 Track datasets, models & experiments  
- 🔁 Reproduce results with **versioned pipelines**  
- 📝 `dvc.yaml` defines ML workflow  

---

## 📊 Project Pipeline  

```mermaid
flowchart TD
    A[📂 Dataset<br>(Chest X-rays/CT scans)] --> B[🔄 Data Ingestion<br>(Split Train/Val/Test)]
    B --> C[⚙️ Base Model Preparation<br>(Transfer Learning - VGG16/ResNet)]
    C --> D[🧠 Model Training<br>(Fine-tuning CNN)]
    D --> E[📈 Model Evaluation<br>(Accuracy, Loss, Metrics)]
    E --> F[💾 Save Model<br>(model.h5 + scores.json)]
    F --> G[🌐 Flask Web App<br>(Upload & Predict)]
    G --> H[📦 MLOps with DVC<br>(Data & Model Versioning)]
    H --> I[☁️ Deployment<br>(Docker + Cloud Hosting)]

🛠 Tech Stack

Language: Python 🐍

Deep Learning: TensorFlow / Keras 🧠

Data Handling: NumPy, Pandas 📊

Visualization: Matplotlib, Seaborn 📈

Web Framework: Flask 🌐

MLOps Tools: DVC, GitHub Actions ⚡

Deployment (optional): Docker 🐳, AWS/Azure/GCP ☁️

📈 Results

✅ Achieved high accuracy in classifying chest cancer images

🚫 Reduced false negatives → improves trust in medical use

📌 Demonstrated a scalable, reproducible ML pipeline

🌟 Highlights

✔️ 91%+ Accuracy on test dataset
✔️ Transfer Learning (VGG16/ResNet) for robust performance
✔️ DVC-powered reproducibility
✔️ User-friendly Flask Web App for real-time predictions
✔️ Deployment-ready with Docker + Cloud

🔮 Future Improvements

✅ Multi-class classification (different cancer types)

✅ Integrate explainable AI (Grad-CAM) for all predictions

✅ Deploy as a full-stack web app (React + Flask backend)

✅ Use cloud-based inference API for scalability
