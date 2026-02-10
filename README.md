# MNIST Digit Recognition 

A deep learning project that trains a Convolutional Neural Network (CNN) on the MNIST dataset using TensorFlow/Keras, and provides an interactive digit recognition app with OpenCV (canvas + webcam modes).

---

##  Overview
- **Dataset:** MNIST (handwritten digits 0–9)
- **Frameworks:** TensorFlow/Keras, OpenCV, NumPy, Matplotlib, Seaborn
- **Features:**
  - CNN model for digit classification
  - Training notebook with evaluation metrics
  - Interactive app: draw digits on canvas or use webcam for real-time recognition
  - Confusion matrix and classification report for model performance

---

##  Installation
Clone the repository and install dependencies:

```bash
git clone https://github.com/yourusername/mnist-digit-recognition.git
cd mnist-digit-recognition
pip install -r requirements.txt

```
---

##  Usage

### 1. Train the CNN model
Open the training notebook in Jupyter or VS Code:

```bash
jupyter notebook mnist_digit_classifier.ipynb
```
This notebook will:
- Load and preprocess the MNIST dataset
- Train the CNN model
- Display accuracy/loss plots, confusion matrix, and classification report
- Save the trained model as mnist_cnn.h5

### 2. Run the interactive recognition app
Launch the app from the terminal:
```bash
python digit_recognition_app.ipynb
```
You’ll see a menu with options:
- Press 1 → Canvas Mode
Draw digits with your mouse. Press:
- p → Predict drawn digits
- c → Clear canvas
- q → Quit
- Press 2 → Webcam Mode
Show handwritten digits to your webcam. The app will detect and classify them in real time.
- Press q → Quit
Exit the application.

---


##  Results

The CNN model trained on the MNIST dataset achieves **~99% accuracy** on the test set.  
Key evaluation outputs include:
Confusion Matrix
<img width="797" height="701" alt="image" src="https://github.com/user-attachments/assets/c23e69d8-d4cb-44d1-891b-a48e144f3c28" />

Training & Validation Accuracy/Loss
<img width="671" height="682" alt="image" src="https://github.com/user-attachments/assets/cf4d3ae8-113b-4891-9f0b-b9f20675f02a" />

Demo (Canvas Mode)
<img width="2558" height="1598" alt="image" src="https://github.com/user-attachments/assets/4096042d-d91e-4b72-9811-63e9c0d8f2d5" />

Demo (Webcam Mode)
<img width="1839" height="1544" alt="image" src="https://github.com/user-attachments/assets/ca143b3a-0b42-42cb-830c-097b024357fa" />


---

##  Project Structure

mnist-digit-recognition/
│── mnist_digit_classifier.ipynb   # Notebook for training & evaluation
│── digit_recognition_app.ipynb    # Interactive app (canvas + webcam)
│── requirements.txt               # Dependencies
│── README.md                      # Project documentation
│── images/                        # Screenshots (confusion matrix, demos)

---

##  Acknowledgements

This project was made possible thanks to the following:

- **MNIST Dataset** — Created by Yann LeCun, Corinna Cortes, and Christopher J.C. Burges, widely used for benchmarking handwritten digit recognition.
- **TensorFlow/Keras** — For building and training the Convolutional Neural Network (CNN).
- **OpenCV** — For enabling interactive digit recognition through canvas drawing and real-time webcam input.
- **Matplotlib & Seaborn** — For visualizing training progress, accuracy/loss curves, and confusion matrices.
- **Scikit-learn** — For generating evaluation metrics such as the confusion matrix and classification report.
- The broader **open-source community** — For providing tools, libraries, and inspiration that made this project possible.

---
##  License

This project is licensed under the **MIT License** — you are free to use, modify, and distribute this code for personal or commercial purposes, provided that proper credit is given.

