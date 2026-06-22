# ✍️ Urdu Handwriting Recognition

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)](https://streamlit.io/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)

An end-to-end Machine Learning project showcasing a **Convolutional Neural Network (CNN)** capable of recognizing **40 handwritten Urdu characters**. The project features a clean, interactive **Streamlit web application** with a live drawable canvas and a file uploader for handwriting prediction.

Developed as part of the course **AIC270 — Programming for AI** (SP24 BAI) at **COMSATS University**.

---

## 🌟 Key Features

*   **Real-time Drawing Canvas:** Write characters directly on the screen using your mouse or touch screen.
*   **File Uploader:** Upload `.png`, `.jpg`, or `.jpeg` images of handwritten characters for instant inference.
*   **Top 5 Predictions:** Interactive bar chart visualizing the model's confidence across the top 5 most likely classes.
*   **Robust CNN Architecture:** Custom-built model trained on handwriting samples, achieving high accuracy.
*   **OOP Wrapper Class (`UrduOCR`):** Clean API encapsulating model reconstruction, weight loading, image preprocessing, and inference.

---

## 📂 Project Structure

```text
Urdu-Handwriting-Recognition/
│
├── reports/                          # Training metrics and visualization plots
│   ├── .gitkeep                      # Keeps directory tracked by Git
│   └── confusion_matrix.png          # Visual representation of model classification accuracy
│
├── scripts/                          # Utility python scripts for data and training
│   ├── explore_data.py               # Dataset exploration, distribution plotting & integrity checks
│   └── train_model.py                # Model design, training loop, evaluation & plot exporter
│
├── app.py                            # Streamlit web application front-end & UI logic
├── urdu_ocr.py                       # OOP wrapper for model reconstruction & prediction pipeline
├── urdu_model.h5                     # Saved trained model weights
├── requirements.txt                  # List of Python dependencies
├── .gitignore                        # Standard Git ignore configurations
└── README.md                         # Project documentation
```

> [!NOTE]
> The dataset is expected to be placed under a `data/` directory (e.g. `data/processed/` with `train/`, `val/`, and `test/` splits) when executing exploration or training scripts.

---

## 🚀 Setup & Installation

Follow these steps to set up and run the application locally:

### 1. Clone & Navigate
```bash
git clone https://github.com/your-username/Urdu-Handwriting-Recognition.git
cd Urdu-Handwriting-Recognition
```

### 2. Install Dependencies
Ensure you have Python 3.9+ installed. Install the required libraries:
```bash
pip install -r requirements.txt
```

> [!TIP]
> If you encounter Keras-specific version issues during runtime, install `tf-keras`:
> ```bash
> pip install tf-keras
> ```

### 3. Run the Streamlit App
Launch the web interface locally:
```bash
streamlit run app.py
```
Once started, open your web browser and navigate to **`http://localhost:8501`**.

---

## 🛠️ Model Architecture

The `UrduOCR` model uses a customized Convolutional Neural Network built with TensorFlow/Keras. The details of the layers are structured below:

| Layer | Configuration | Output Shape | Parameters |
| :--- | :--- | :--- | :--- |
| **Input** | Grayscale Image | `(64, 64, 1)` | — |
| **Conv2D #1** | 32 filters, 3x3 kernel, ReLU | `(62, 62, 32)` | 320 |
| **MaxPooling2D #1** | 2x2 pool | `(31, 31, 32)` | 0 |
| **Conv2D #2** | 64 filters, 3x3 kernel, ReLU | `(29, 29, 64)` | 18,496 |
| **MaxPooling2D #2** | 2x2 pool | `(14, 14, 64)` | 0 |
| **Conv2D #3** | 128 filters, 3x3 kernel, ReLU | `(12, 12, 128)` | 73,856 |
| **MaxPooling2D #3** | 2x2 pool | `(6, 6, 128)` | 0 |
| **Flatten** | Reshape | `(4608)` | 0 |
| **Dense** | 128 units, ReLU | `(128)` | 589,952 |
| **Dropout** | 50% rate | `(128)` | 0 |
| **Output (Dense)** | 40 units (classes), Softmax | `(40)` | 5,160 |

---

## 📊 Evaluation & Metrics

The training results, graphs, and performance metrics are exported to the `reports/` folder.

### Confusion Matrix
The confusion matrix showcases the classification precision across the 40 distinct Urdu handwritten characters:
*   Located at: `reports/confusion_matrix.png`

---

## 🔍 Troubleshooting

> [!WARNING]
> **Blank White Screen on App Start**
> The model might take 10–20 seconds to compile on first launch. If the screen remains blank, wait for the terminal logs to show loading complete, then press **R** in your browser to reload the interface.

> [!IMPORTANT]
> **Keras `batch_shape` Compatibility Errors**
> The model was trained with Keras 3.x saved weight syntax but the local environment runs TensorFlow 2.15 (using `tf-keras`). The class wrapper (`urdu_ocr.py`) includes a custom H5 weight-loading utility that dynamically rebuilds the model structure and loads the weights. If you run into issues, verify that `tf-keras` and `h5py` are installed properly.

> [!TIP]
> **Canvas Doesn't Show / Draw Option Fails**
> Verify you have the correct canvas package version installed:
> ```bash
> pip install streamlit-drawable-canvas
> ```

---

## 📜 License
This project is for educational purposes. All rights belong to the contributors.
CUniversity — SP24 BAI Coursework.
