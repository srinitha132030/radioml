# RadioML AMC - Automatic Modulation Classification System

A Flask-based web application for classifying radio signal modulations using the **RadioML 2016.10A** dataset from DeepSig.

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Application
```bash
python app.py
```

Open your browser at: **http://localhost:5000**

---

## 🔐 Default Credentials

| Role  | Username | Password |
|-------|----------|----------|
| Admin | `admin`  | `admin123` |
| User  | `user1`  | `user123` |

---

## 📁 Project Structure

```
radioml_app/
├── app.py                  # Main Flask application
├── requirements.txt        # Python dependencies
├── instance/
│   └── radioml.db          # SQLite database (auto-created)
├── static/
│   ├── uploads/            # User signal uploads
│   ├── models/             # Trained model files
│   └── datasets/           # Uploaded datasets
└── templates/
    ├── base.html           # Base layout with sidebar
    ├── login.html
    ├── register.html
    ├── admin/
    │   ├── dashboard.html
    │   ├── upload_dataset.html
    │   ├── datasets.html
    │   ├── preprocess.html
    │   ├── visualize.html
    │   ├── train.html
    │   ├── models.html
    │   ├── evaluate.html
    │   └── users.html
    └── user/
        ├── dashboard.html
        ├── predict.html
        └── history.html
```

---

## 🧩 Modules

### 👑 Admin Module
| Feature | Description |
|---------|-------------|
| **Dataset Upload** | Upload RadioML 2016.10A `.pkl` file (up to 500 MB) |
| **Preprocessing** | Parse pickle, extract I/Q samples, detect mod classes & SNR levels |
| **Visualization** | Heatmap, pie chart, I/Q waveforms, constellation diagrams |
| **Model Training** | Train CNN, LSTM, CNN-LSTM, or ResNet with configurable epochs |
| **Evaluation** | Training curves, per-class accuracy, SNR analysis, confusion matrix |
| **User Management** | View and delete users |

### 👤 User Module
| Feature | Description |
|---------|-------------|
| **Signal Prediction** | Upload signal file or generate synthetic I/Q signal |
| **File Formats** | `.pkl`, `.npy`, `.csv`, `.txt` |
| **Visualizations** | I/Q time domain, constellation diagram, probability bars |
| **History** | Full prediction history with statistics |

---

## 📶 Dataset: RadioML 2016.10A

**Download from:** https://www.kaggle.com/datasets/nolasthitnotomorrow/radioml2016-deepsigcom/data

- **File:** `RML2016.10a_dict.pkl`
- **Format:** Python dict with keys `(modulation_type, snr_db)`
- **Signal shape:** `(1000, 2, 128)` — 1000 samples × 2 channels × 128 time steps
- **Modulations:** 8PSK, AM-DSB, AM-SSB, BPSK, CPFSK, GFSK, PAM4, QAM16, QAM64, QPSK, WBFM
- **SNR Range:** -20 dB to +18 dB (step: 2 dB)

---

## 📤 User Signal File Formats

### `.pkl` (Pickle)
```python
# RadioML dict format
data = {('BPSK', 0): np.array(shape=(1000,2,128)), ...}
# Or plain numpy array
data = np.array(shape=(N, 2, 128))
```

### `.npy` (NumPy)
```python
np.save('signal.npy', array)  # shape: (N, 2, 128) or (2, 128)
```

### `.csv` (CSV)
```
# 256 columns: first 128 = I channel, last 128 = Q channel
0.12, 0.34, ..., -0.56, 0.78, ...
```

### `.txt` (Text)
```
# 256 space-separated values: first 128 = I, last 128 = Q
0.12 0.34 ... -0.56 0.78 ...
```

---

## 🛠️ Technologies Used

- **Backend:** Flask, SQLite3 (via Python's built-in sqlite3)
- **Frontend:** Bootstrap 5, Chart.js, Plotly.js, Bootstrap Icons
- **ML/Data:** NumPy (signal processing simulation)
- **Auth:** Werkzeug password hashing

---

## ⚙️ Configuration

In `app.py`, you can adjust:
- `MAX_CONTENT_LENGTH`: Max upload size (default 500 MB)
- `UPLOAD_FOLDER`, `MODEL_FOLDER`, `DATASET_FOLDER`: Storage paths
- `app.secret_key`: Change for production
