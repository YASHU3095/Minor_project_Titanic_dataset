# Titanic Survivor Data Preparation Pipeline

An end-to-end Machine Learning data preprocessing pipeline designed for the Titanic Survivor dataset. This project performs robust data cleaning, validation, feature engineering, missing value imputation, outlier handling, categorical encoding, and feature scaling to produce high-quality training and testing datasets ready for model training.

---

## 📌 Project Overview

- **Dataset**: Titanic Survival Dataset (`titanic.csv` or loaded via Seaborn)
- **Goal**: Clean, transform, and structure raw survival data into model-ready features.
- **Language**: Python 3.x
- **Key Libraries**: Pandas, NumPy, Scikit-learn, Seaborn

---

## 🛠️ Pipeline Architecture

1. **Data Loading**: Loads the dataset from Seaborn or local CSV.
2. **Data Validation**: Checks for essential features (`survived`, `pclass`, `sex`, `age`, `sibsp`, `parch`, `fare`, `embarked`) using custom exceptions.
3. **Handling Missing Values**:
   - Imputes numerical features (e.g. median age by passenger class / sex).
   - Imputes categorical features using mode values.
4. **Outlier Detection & Treatment**:
   - Caps extreme fare and numerical outliers using IQR or percentile capping.
5. **Feature Engineering**:
   - Creates family size indicators (`FamilySize = sibsp + parch + 1`).
   - Generates solo traveler flag (`IsAlone`).
   - Extracts titles and status where applicable.
6. **Train/Test Splitting**: Stratified split preserving target distribution.
7. **Categorical Encoding**: One-Hot / Target encoding for nominal variables.
8. **Feature Scaling**: Standardizes features using `StandardScaler`.
9. **Export**: Saves processed feature matrices and target vectors.

---

## 📁 Repository Structure

```text
├── main.py             # Primary end-to-end data preparation pipeline
├── requirements.txt    # Project dependencies
├── .gitignore          # Ignored cache & temporary files
├── titanic.csv         # Raw Titanic dataset
├── X_train.csv         # Processed training features
├── X_test.csv          # Processed testing features
├── y_train.csv         # Target labels for training
└── y_test.csv          # Target labels for testing
```

---

## 🚀 Getting Started

### 1. Prerequisites
Ensure you have Python 3.9+ installed on your system.

### 2. Clone the Repository
```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Pipeline
```bash
python main.py
```
After successful execution, the transformed datasets (`X_train.csv`, `X_test.csv`, `y_train.csv`, `y_test.csv`) will be generated.

