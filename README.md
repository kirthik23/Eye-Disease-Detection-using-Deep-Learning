# Eye-Disease-Detection-using-Deep-Learning
This project implements a deep learning-based system (ResNet 50) for detecting eye diseases from retinal fundus images. It utilizes Convolutional Neural Networks (CNNs) and other advanced techniques to automatically classify various eye conditions such as glaucoma, diabetic retinopathy, and cataract from retinal images.
Note : This project is totally run and compiled in Jupyter Notebook

## 📘 Project Overview

Early and accurate diagnosis of eye diseases is crucial to prevent permanent vision loss. Traditional manual screening is time-consuming and prone to human error. This project addresses these challenges by leveraging deep learning for automated detection, increasing accuracy, speed, and consistency.

## 🧠 Motivation

- Rapid increase in eye-related diseases globally.
- Need for cost-effective, scalable, and automated diagnostic tools.
- Bridging the gap in rural and underdeveloped regions with limited access to ophthalmologists.

## 🧪 Technologies Used

- Python
- TensorFlow / Keras
- OpenCV
- Deep Learning (CNN, ResNet50, etc.)
- Tkinter (for GUI)
- PDF generation libraries (e.g., ReportLab or FPDF)


## 📂 Dataset [download](https://www.kaggle.com/datasets/gunavenkatdoddi/eye-diseases-classification)

We use annotated datasets of retinal fundus images (such as ODIR-5K) with labeled eye conditions including:
- Normal
- Diabetic Retinopathy
- Cataract
- Glaucoma
- Hypertension
- Age-related Macular Degeneration (AMD)

## 🎯 Features

- Automatic classification of retinal images into disease categories.
- User-friendly GUI to upload images and display predictions.
- PDF report generation after prediction.
- Trained deep learning model (.h5 or .keras) for deployment.

## 🚀 How to Run

1. Clone the repository:
   bash
   git clone https://github.com/your-username/eye-disease-detection.git
   cd eye-disease-detection


2. Install Dependencies
   bash
   pip install -r requirements.txt

3. Run the GUI:
   ```bash
   python app.py

4. Upload a retinal image to get a prediction and generate a PDF report


   ![Glaucoma](https://github.com/user-attachments/assets/37163c78-11fe-45ef-98f9-f09d24f473b0)



 ## 📈 Model Training

The model was trained using a custom CNN and/or ResNet50 architecture with data augmentation techniques to reduce overfitting.  
Performance metrics include *accuracy, **precision, **recall, and **F1-score*.

## 📄 Report

Includes:
- Literature review
- Methodology
- Data preprocessing
- Model architecture
- Results and analysis

Refer to chapter 1.pdf for the detailed project introduction and background.

## 🤝 Acknowledgements

- Open-source datasets like *ODIR-5K*
- *TensorFlow* and *Keras* communities
- Scientific literature on *medical image classification*

## 📌 Future Scope

- Expand classification to more diseases
- Improve prediction accuracy with *ensemble models*
- Deploy on *mobile platforms* for field diagnostics
