# 🫀 Heart Disease Prediction (ML Zoomcamp Midterm Project)

## 📌 Project Overview

This project predicts the **risk of heart disease** using machine learning.

It follows the full ML Zoomcamp pipeline:

* Select a real dataset
* Perform EDA (Exploratory Data Analysis)
* Train multiple models
* Tune the best model
* Export training code to `train.py`
* Serve the model with a web service (`service.py`)
* Test predictions using a client script (`client.py`)
* Containerize with Docker

This project demonstrates how machine learning can help identify high-risk patients based on simple clinical features.

---

## 📊 Dataset

**Dataset:** Heart Disease Cleveland (UCI Repository)

The dataset contains **303 rows** and **14 columns**:

* **13 clinical features** (age, sex, chest pain type, blood pressure, cholesterol, etc.)
* **1 target column:** `target`

  * `1` → heart disease present
  * `0` → no heart disease

In this repository, the dataset is stored at:

```text
data/Heart_disease_cleveland_new.csv
```

---

## 🔍 EDA

EDA is done in:

```text
notebooks/01_eda.ipynb
```

Main checks performed:

* Data types and missing values (none found)
* Target distribution (slightly more positives than negatives)
* Feature distributions (histograms)
* Correlation heatmap
* Relationship between important features and target (e.g., `oldpeak`, `thalach`, `cp`, `ca`, `thal`)

---

## 🤖 Model Training

Modeling is done in:

```text
notebooks/02_models.ipynb
```

and exported to:

```text
scripts/train.py
```

Models trained and evaluated:

* Logistic Regression
* Random Forest (baseline)
* Random Forest with hyperparameter tuning (GridSearchCV)

The **final model** used in production is the **tuned Random Forest**.

The trained model and scaler are stored together in:

```text
models/model.bin
```

This file is created by running the training script.

---

## 🧪 Training Script

**Script:**

```text
scripts/train.py
```

What it does:

* Loads the dataset from `data/Heart_disease_cleveland_new.csv`
* Splits into train and test sets
* Scales features with `StandardScaler`
* Trains Logistic Regression, Random Forest, and a tuned Random Forest
* Prints metrics (accuracy and ROC AUC)
* Saves the final scaler + model to `models/model.bin`

Run it with:

```bash
python scripts/train.py
```

---

## 🌐 Web Service (Flask API)

**File:**

```text
web/service.py
```

This script:

* Loads `models/model.bin`
* Starts a Flask app
* Exposes a `/predict` endpoint that accepts POST requests with JSON

Run the service locally:

```bash
python web/service.py
```

The service will listen on:

```text
http://127.0.0.1:9696/predict
```

Example input JSON:

```json
{
  "age": 63,
  "sex": 1,
  "cp": 3,
  "trestbps": 145,
  "chol": 233,
  "fbs": 1,
  "restecg": 0,
  "thalach": 150,
  "exang": 0,
  "oldpeak": 2.3,
  "slope": 0,
  "ca": 0,
  "thal": 1
}
```

---

## 🧪 Client Script

**File:**

```text
web/client.py
```

This small script sends a test request to the API to verify it is working:

```bash
python web/client.py
```

---

## 🐳 Running with Docker

You can run the prediction service inside a Docker container to ensure a consistent environment.

### 1. Build the image

From the project root directory:

```bash
docker build -t heart-disease-service .
```

### 2. Run the container

```bash
docker run -it --rm -p 9696:9696 heart-disease-service
```

The service will be available at:

```text
http://127.0.0.1:9696/predict
```

### 3. Test it

From your host machine (in a new terminal), run the client script:

```bash
python web/client.py
```

---

## 📦 Dependencies

Dependencies are listed in:

```text
requirements.txt
```

Install them with:

```bash
pip install -r requirements.txt
```

---

## 📁 Repository Structure

```text
.
├── data/
│   └── Heart_disease_cleveland_new.csv
├── models/
│   └── model.bin
├── notebooks/
│   ├── 01_eda.ipynb
│   └── 02_models.ipynb
├── scripts/
│   └── train.py
├── web/
│   ├── service.py
│   └── client.py
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## ✅ ML Zoomcamp Rubric Coverage

* Problem description: ✔
* EDA: ✔
* Model training & tuning: ✔
* Exported training script: ✔
* Reproducibility (data + code): ✔
* Model deployment (Flask): ✔
* Dependency management (`requirements.txt`): ✔
* Containerization (Dockerfile): ✔
