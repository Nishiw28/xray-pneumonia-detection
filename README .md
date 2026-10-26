# Chest X-ray Classifier (Normal vs Pneumonia)

A simple web app that takes a chest X-ray image and predicts whether it looks **Normal** or shows **Pneumonia**. It uses a **CNN (Convolutional Neural Network)** built with TensorFlow/Keras and a **Streamlit** interface.

> ⚠️ This is an educational project. It is **not** a medical diagnosis tool.

## 🚀 Run Locally

See [How to Run Locally](#how-to-run-locally) below. (Not deployed online yet.)

## How It Works

```
Upload X-ray Image
        ↓
Convert to RGB + Resize to 128 x 128
        ↓
Scale pixel values (divide by 255)
        ↓
CNN Model (xray.keras)
        ↓
Normal / Pneumonia
```

1. **Upload** – the user uploads a JPG or PNG X-ray image.
2. **Preprocessing** – the image is converted to RGB, resized to 128 × 128 and scaled to 0–1, exactly like during training.
3. **CNN model** – 3 convolution + max-pooling blocks, followed by dense layers and a sigmoid output.
4. **Prediction** – the sigmoid output is the probability of Pneumonia. Above 0.5 → Pneumonia, otherwise → Normal.

## Technologies Used

- Python
- TensorFlow / Keras (CNN)
- Streamlit
- Pillow and NumPy

## Dataset

The model was trained in `Practice.ipynb` on the [COVID-19 Xray Dataset (Train & Test Sets)](https://www.kaggle.com/datasets/khoongweihao/covid19-xray-dataset-train-test-sets) from Kaggle. It has two classes (`NORMAL` and `PNEUMONIA`): **148 training images** and **40 test images**.

The dataset is small, so the results in the notebook should not be treated as real-world medical performance. The dataset is not included in this repository; please download it from Kaggle if you want to retrain the model.

## Features

- Upload an X-ray image (JPG / PNG)
- Image is resized and scaled automatically
- Get a Normal / Pneumonia prediction with probabilities
- Friendly messages for invalid files or a missing model

## Project Structure

```
Xray-Pneumonia-Detection/
├── app.py
├── xray.keras
├── Practice.ipynb
├── requirements.txt
├── README.md
└── .gitignore
```

- `Practice.ipynb` – training notebook
- `xray.keras` – the saved trained model
- `app.py` – the Streamlit app

## How to Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

This opens the app in your browser (usually http://localhost:8501).
