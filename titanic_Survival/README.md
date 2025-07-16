# Titanic Survival Prediction 🚢

This project is a machine learning pipeline that predicts whether a passenger survived the Titanic disaster using structured data. It is built using the Titanic dataset and applies modern data preprocessing, feature engineering, and model tuning techniques.

## 📁 Folder: `titanic_Survival`

Author: **Ashutosh Kumar**

---

## 📊 Features Used

- Passenger Class (`Pclass`)
- Gender (`Sex`)
- Age (`Age`)
- Fare (`Fare`)
- Embarkation Port (`Embarked`)
- Number of Siblings/Spouses (`SibSp`)
- Number of Parents/Children (`Parch`)

---

## 🔧 ML Pipeline Overview

- 📥 **Data Loading** with `pandas`
- 📈 **Exploratory Data Analysis (EDA)** with `seaborn` & `matplotlib`
- 🧹 **Preprocessing Pipeline**:
  - Missing value handling
  - One-hot encoding for categorical variables
  - Standard scaling for numerical features
- 🤖 **Model Training**:
  - `RandomForestClassifier` with `GridSearchCV` for hyperparameter tuning
- 📊 **Evaluation**:
  - Accuracy Score
  - Confusion Matrix
  - Classification Report
- 📌 **Feature Importance** Visualization (Top 10)

---

## 📂 Files Included

- `titanic.csv` – Dataset
- `titanic.py` – Main Python script
- `README.md` – Project documentation

---

## 🚀 How to Run

1. Make sure you have the required libraries installed:
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn
   ```

2. Run the Python script:
   ```bash
   python titanic.py
   ```

---

## ✅ Output

- Survival prediction accuracy
- Confusion matrix heatmap
- Top 10 important features used by the model

---

## 🧠 Model Used

**Random Forest Classifier** with Grid Search for best hyperparameters.

---

## 🧑‍💻 Author

**Ashutosh Kumar**

GitHub: [@Ashutosh2245](https://github.com/Ashutosh2245)

---

## 📌 Note

This project is created as a part of internship task under `CODSOFT` and demonstrates clean end-to-end supervised learning pipeline.
