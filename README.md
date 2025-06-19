# 🌍 Hazardous Near-Earth Object (NEO) Classification Using Machine Learning

This repository presents a machine learning pipeline to classify Near-Earth Objects (NEOs) as potentially hazardous or not, based on their physical and orbital characteristics. This project is part of an undergraduate thesis and serves as a machine learning portfolio.

---

## 📚 Abstract

Near-Earth Objects (NEOs) are space objects that orbit the Earth and have the potential to cause significant harm. This study applies Supervised Machine Learning algorithms to predict the danger level of NEOs using features like **absolute magnitude**, **estimated diameter**, **relative velocity**, **miss distance**, and **MOID**.

Machine Learning models used:
- Logistic Regression
- Random Forest
- XGBoost
- Support Vector Classifier (SVC)
- K-Nearest Neighbors (KNN)
- Neural Network (MLPClassifier)

The best-performing model was **Random Forest** with **99.96% accuracy**, showing strong potential for automated threat detection from space objects.

---

## 📁 Project Structure

hazardous-neo-classification/
│
├── notebooks/
│   ├── preprocessing_data.ipynb
│   ├── baseline/
│   │   ├── hazardous-neos-prediction-1a.ipynb
│   │   └── hazardous-neos-prediction-1b.ipynb
│   ├── hyperparameter_tuning/
│   │   ├── hazardous-neos-prediction-2a.ipynb
│   │   ├── hazardous-neos-prediction-2b.ipynb
│   │   ├── hazardous-neos-prediction-2c.ipynb
│   │   └── hazardous-neos-prediction-2d.ipynb
│
├── figures/
│   ├── baseline/
│   │   ├── LG/
│   │   ├── RF/
│   │   ├── XGB/
│   │   ├── SVC/
│   │   ├── KNN/
│   │   └── MLP/
│   ├── hyperparameter_tuning/
│   │   ├── LG/
│   │   ├── RF/
│   │   ├── XGB/
│   │   ├── SVC/
│   │   ├── KNN/
│   │   └── MLP/
│
├── models/
│   ├── baseline/
│   │   ├── model_svc_baseline.pkl
│   │   ├── model_rf_baseline.pkl
│   │   └── ...
│   ├── hyperparameter_tuning/
│   │   ├── best_svc_model.pkl
│   │   ├── best_rf_model.pkl
│   │   └── ...
├── .gitignore
├── README.md
└── requirements.txt

---

## 📊 Features Used

- `absolute_magnitude`
- `estimated_diameter_min`
- `estimated_diameter_max`
- `relative_velocity`
- `miss_distance`
- `moid`

---

## 🧠 Machine Learning Models

- Logistic Regression
- Random Forest
- XGBoost
- Support Vector Classifier (SVC)
- K-Nearest Neighbors (KNN)
- Neural Network (MLPClassifier)

---

## 📈 Evaluation Metrics

- Precision
- Recall
- F1-Score
- ROC AUC

---

## 🏆 Results Summary

| Model                 | Accuracy (%) |
|----------------------|--------------|
| Random Forest         | 99.96        |
| XGBoost               | 99.92        |
| Neural Network (MLP)  | 99.57        |
| SVC                   | 97.41        |
| Logistic Regression   | 97.33        |
| K-Nearest Neighbors   | 96.44        |

---

## 📊 Model Evaluation Visuals

Evaluation results for each model — including **ROC Curve**, **Confusion Matrix**, and **Classification Report visualizations** — are stored under:

- `figures/baseline/` → Initial results for each model (default parameters)
- `figures/hyperparameter_tuning/` → Results after hyperparameter tuning

Each of these contains subfolders for the respective models:

- `LG` → Logistic Regression  
- `RF` → Random Forest  
- `XGB` → XGBoost  
- `SVC` → Support Vector Classifier  
- `KNN` → K-Nearest Neighbors  
- `MLP` → Neural Network (MLPClassifier)  

These visualizations help compare how model performance improves after tuning.

---

## 💾 Saved Models

Trained machine learning models are organized as follows:

- `models/baseline/`: Contains untuned models trained using default parameters  
  e.g., `model_rf_baseline.pkl`, `model_svc_baseline.pkl`

- `models/hyperparameter_tuning/`: Contains best models after hyperparameter tuning  
  e.g., `best_rf_model.pkl`, `best_svc_model.pkl`

These models can be loaded directly for inference or further evaluation without re-training.

---

## 💻 Execution Environment

Notebooks in this project were executed and tested using multiple platforms to provide flexibility and leverage computational support:

✅ **Google Colab** – for running experiments with free access to GPU/TPU  
✅ **Kaggle Notebook** – for seamless integration with the original dataset  
✅ **VSCode (Jupyter)** – for local development, preprocessing, and modular scripting

Each notebook is self-contained and can be run independently in any of the environments above, provided the dependencies in `requirements.txt` are installed.

---

## 🔗 Dataset Sources

- [Kaggle - NASA Near-Earth Objects](https://www.kaggle.com/datasets/ivansher/nasa-nearest-earth-objects-1910-2024)
- [NASA JPL Small-Body Database Query](https://ssd.jpl.nasa.gov/tools/sbdb_query.html)

---

## 📦 Installation

To install the required dependencies:

```bash
pip install -r requirements.txt


👩‍💻 Author
Wilma Nur Fatimah
Machine Learning Enthusiast
