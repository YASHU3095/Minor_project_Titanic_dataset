
# MINOR PROJECT: TITANIC SURVIVOR DATA PREPARATION PIPELINE
# Python 3.11

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# CUSTOM EXCEPTION

class DataValidationError(Exception):
    pass

# DISPLAY PROCESSED DATA AFTER EACH STEP

def show_data(title, df):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)

    if isinstance(df, pd.DataFrame):
        print(df.head())
        print("\nDataset Shape:", df.shape)
        print("\nMissing Values:")
        print(df.isnull().sum())

    elif isinstance(df, pd.Series):
        print(df.head())
        print("\nTotal Records:", len(df))

# 1. LOAD DATA

def load_data():
    print("\nLoading Titanic dataset...")
    df = sns.load_dataset("titanic")
    return df

# 2. VALIDATE DATA

def validate_data(df):
    required_columns = [
        "survived", "pclass", "sex", "age",
        "sibsp", "parch", "fare", "embarked"
    ]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise DataValidationError(
            f"Missing columns: {missing_columns}"
        )

    if df.empty:
        raise DataValidationError("Dataset is empty.")

    print("Dataset validation successful.")
    return df

# 3. CLEAN TEXT

def clean_text(df):
    df = df.copy()

    for col in df.select_dtypes(
        include=["object", "category", "string"]
    ).columns:

        df[col] = (
            df[col]
            .astype("string")
            .str.strip()
            .str.lower()
            .replace("", pd.NA)
            .astype(object)
        )

    print("Text cleaning completed.")
    return df

# 4. DROP USELESS COLUMNS

def drop_useless(df):
    df = df.copy()

    # Remove target leakage and unnecessary columns
    useless_columns = [
        "alive",
        "class",
        "who",
        "adult_male",
        "deck",
        "embark_town"
    ]

    df.drop(
        columns=useless_columns,
        errors="ignore",
        inplace=True
    )

    # Remove columns with more than 50% missing values
    high_missing = [
        col for col in df.columns
        if col != "survived"
        and df[col].isnull().mean() > 0.50
    ]

    df.drop(
        columns=high_missing,
        errors="ignore",
        inplace=True
    )

    print("Unnecessary columns removed.")
    return df

# 5. HANDLE MISSING VALUES

def handle_missing(X_train, X_test):
    X_train = X_train.copy()
    X_test = X_test.copy()

    numeric_columns = X_train.select_dtypes(
        include=np.number
    ).columns.tolist()

    categorical_columns = [
        col for col in X_train.columns
        if col not in numeric_columns
    ]

    # Numeric columns: fill using training median
    for col in numeric_columns:
        median_value = X_train[col].median()

        X_train[col] = X_train[col].fillna(median_value)
        X_test[col] = X_test[col].fillna(median_value)

    # Categorical columns: fill using training mode
    for col in categorical_columns:
        modes = X_train[col].mode(dropna=True)

        if not modes.empty:
            mode_value = modes.iloc[0]
            X_train[col] = X_train[col].fillna(mode_value)
            X_test[col] = X_test[col].fillna(mode_value)

    print("Missing values handled.")
    return X_train, X_test

# 6. ENCODE CATEGORICAL VARIABLES
# Binary mapping + ordinal encoding + one-hot encoding

def encode_categoricals(X_train, X_test):
    X_train = X_train.copy()
    X_test = X_test.copy()

    # Binary encoding: male = 0, female = 1
    if "sex" in X_train.columns:
        X_train["sex"] = X_train["sex"].map(
            {"male": 0, "female": 1}
        )
        X_test["sex"] = X_test["sex"].map(
            {"male": 0, "female": 1}
        )

    # Ordinal encoding: passenger class is already 1, 2, 3
    if "pclass" in X_train.columns:
        X_train["pclass"] = X_train["pclass"].astype(int)
        X_test["pclass"] = X_test["pclass"].astype(int)

    # One-hot encode remaining categorical columns
    categorical_columns = X_train.select_dtypes(
        include=["object", "category", "string"]
    ).columns.tolist()

    X_train = pd.get_dummies(
        X_train,
        columns=categorical_columns,
        drop_first=True,
        dtype=int
    )

    X_test = pd.get_dummies(
        X_test,
        columns=categorical_columns,
        drop_first=True,
        dtype=int
    )

    # Match test columns to training columns.
    # Unseen test categories are safely ignored.
    X_test = X_test.reindex(
        columns=X_train.columns,
        fill_value=0
    )

    print("Categorical encoding completed.")
    return X_train, X_test

# 7. SPLIT DATA
# Stratified 80/20 training and testing split

def split_data(df):
    df = df.dropna(subset=["survived"]).copy()

    X = df.drop(columns=["survived"])
    y = df["survived"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print("Stratified 80/20 split completed.")

    return X_train, X_test, y_train, y_test



# 8. SCALE FEATURES
# StandardScaler fitted on training data only


def scale_features(X_train, X_test):
    X_train = X_train.copy()
    X_test = X_test.copy()

    scaler = StandardScaler()

    numeric_columns = X_train.select_dtypes(
        include=np.number
    ).columns.tolist()

    # Fit on training data; transform both sets
    X_train[numeric_columns] = scaler.fit_transform(
        X_train[numeric_columns]
    )

    X_test[numeric_columns] = scaler.transform(
        X_test[numeric_columns]
    )

    print("Feature scaling completed.")

    return X_train, X_test

# MAIN FUNCTION

def main():

    try:
        # 1. Load data
        df = load_data()
        show_data("1. AFTER LOADING DATA", df)

        # 2. Validate data
        df = validate_data(df)
        show_data("2. AFTER VALIDATING DATA", df)

        # 3. Clean text
        df = clean_text(df)
        show_data("3. AFTER CLEANING TEXT", df)

        # 4. Drop useless columns
        df = drop_useless(df)
        show_data("4. AFTER DROPPING USELESS COLUMNS", df)

        # 5. Split BEFORE fitting preprocessing transformations
        X_train, X_test, y_train, y_test = split_data(df)

        show_data("5. TRAINING FEATURES AFTER SPLITTING", X_train)
        show_data("5. TESTING FEATURES AFTER SPLITTING", X_test)

        print("\nTraining target:")
        print(y_train.head())

        print("\nTesting target:")
        print(y_test.head())

        # 6. Handle missing values
        X_train, X_test = handle_missing(X_train, X_test)

        show_data("6. AFTER HANDLING MISSING VALUES - TRAIN", X_train)
        show_data("6. AFTER HANDLING MISSING VALUES - TEST", X_test)

        # 7. Encode categorical variables
        X_train, X_test = encode_categoricals(
            X_train, X_test
        )

        show_data("7. AFTER ENCODING - TRAIN", X_train)
        show_data("7. AFTER ENCODING - TEST", X_test)

        # 8. Scale features
        X_train, X_test = scale_features(
            X_train, X_test
        )

        show_data("8. AFTER SCALING - TRAIN", X_train)
        show_data("8. AFTER SCALING - TEST", X_test)

        # Save final processed datasets
        X_train.to_csv("X_train.csv", index=False)
        X_test.to_csv("X_test.csv", index=False)

        y_train.to_csv("y_train.csv", index=False)
        y_test.to_csv("y_test.csv", index=False)

        print("\n" + "=" * 70)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 70)

        print("\nFinal Training Shape:", X_train.shape)
        print("Final Testing Shape:", X_test.shape)

        print("\nFiles saved successfully:")
        print("1. X_train.csv")
        print("2. X_test.csv")
        print("3. y_train.csv")
        print("4. y_test.csv")

    except DataValidationError as error:
        print("Validation Error:", error)

    except Exception as error:
        print("Error:", error)

# PROGRAM ENTRY POINT

if __name__ == "__main__":
    main()
