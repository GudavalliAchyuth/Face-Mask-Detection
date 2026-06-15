# 😷 Face Mask Detection System

An AI-powered computer vision system that detects in real-time whether a person is wearing a face mask. Built using deep learning and OpenCV, this project aims to provide a fast and efficient solution for monitoring safety compliance in public spaces.

## 📄 Abstract
*(Prepared for Major/Minor Project Submission)*

The global necessity for public health compliance has driven the need for automated safety monitoring systems. This project presents a Face Mask Detection System utilizing Convolutional Neural Networks (CNNs) and Computer Vision techniques. A custom deep learning model is trained on an image dataset of faces categorized into "With Mask" and "Without Mask" classes. The system pipeline deploys OpenCV for real-time face localization through a webcam stream, subsequently passing the extracted Region of Interest (ROI) to the trained CNN for binary classification. The model successfully identifies face masks under varying lighting conditions and angles. This lightweight system demonstrates the practical application of edge-deployment AI in enforcing public health protocols effectively.

## 🛠️ Technologies Used
* **Deep Learning Framework:** TensorFlow / Keras
* **Computer Vision:** OpenCV
* **Data Processing:** NumPy, Pandas
* **Machine Learning Tools:** Scikit-Learn

## ⚠️ Prerequisites (Crucial for TensorFlow)
Because this project utilizes TensorFlow 2.x, it requires a specific Python environment to run successfully.
* **Required Python Version:** Python 3.8, 3.9, 3.10, or 3.11.
* **Warning:** If you are using Python 3.12 or newer, the `pip install` command will fail because TensorFlow has not released stable binaries for those versions yet. 

**Recommended Setup:**
If you have a newer version of Python installed, please create an isolated Python 3.10 virtual environment before installing the requirements:
```bash
py -3.10 -m venv venv
.\venv\Scripts\activate

## ⚙️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/GudavalliAchyuth/Face-Mask-Detection.git
   cd Face-Mask-Detection

2. Install required dependencies:
    pip install -r requirements.txt

3.Usage & Execution:
   Phase 1: Training the Model:
      python scripts/train_mask_detector.py

   Phase 2: Running the Live Detection:
      python scripts/detect_mask_video.py
