
# 🩺 Chest Cancer Classification using Deep Learning

## 📌 Project Overview

This project focuses on building an AI-powered image classification system to detect and classify chest cancer from medical images. The model leverages Convolutional Neural Networks (CNNs) and transfer learning techniques to assist radiologists in early diagnosis, improve treatment decisions, and reduce manual diagnostic effort.
The project is designed with a modular pipeline, production-ready ML system with configurable components and cloud deployment support.

## 🔄 Workflow

### 1. Data Ingestion

Loads chest X-ray dataset, splits into training/validation/testing.

Ensures data quality and reproducibility.

### 2. Base Model Preparation

Initializes a CNN (transfer learning with models like VGG16/ResNet).

Freezes base layers, adds classification head.

Compiles model with optimizer & loss function.

### 3. Model Training

Trains the CNN on chest cancer images.

Logs metrics (accuracy, loss).

Saves trained model as model.h5.

### 4. Model Evaluation

Evaluates on test data.

Generates metrics stored in scores.json.

Ensures reliability in medical context.

### 5. Pipeline Orchestration

Modular pipelines (src/cnnClassifier/pipeline) run each stage.

Config & parameters controlled by config.yaml and params.yaml.

### 6. Deployment (Flask Web App)

app.py provides an interface for users to upload chest images.

Model predicts cancer vs non-cancer in real-time.

Frontend served via templates/index.html.

### 7. MLOps with DVC

Tracks datasets, models, and experiments.

Ensures reproducibility and versioning.

dvc.yaml defines pipelines for training & evaluation.

## 🛠 Tech Stack

Programming Language: Python

Deep Learning: TensorFlow / Keras

Data Handling: NumPy, Pandas

Visualization: Matplotlib, Seaborn

Web Framework: Flask

MLOps Tools: DVC, GitHub Actions (CI/CD)

Deployment (optional): Docker, AWS/Azure/GCP
