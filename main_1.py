import warnings
warnings.filterwarnings("ignore")

import seaborn as sns
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------
# Custom Exception
# ---------------------------------------------------------
class MissingColumnsError(Exception):
    """Raised when essential columns are missing from the dataset."""


# ---------------------------------------------------------
# 1. Load Titanic Dataset
# ---------------------------------------------------------
def load_data():
    """Load the Titanic dataset from Seaborn."""
    df = sns.load_dataset("titanic")
    print("Dataset loaded successfully.")
    print("Original shape:", df.shape)
    return df


# ---------------------------------------------------------
# 2. Validate Required Columns
# ---------------------------------------------------------
def validate_data(df):
    """Check whether all essential columns exist."""
    required_columns = [
        "survived", "pclass", "sex", "age",
        "sibsp", "parch", "fare", "embarked"
    ]

    missing_columns = [
        col for col in required_columns if col not in df.columns
    ]

    if missing_columns:
        raise MissingColumnsError(
            f"Missing required columns: {missing_columns}"
        )

    print("Data validation successful.")
    return df


# ---------------------------------------------------------
# 3. Clean Text Columns
# ---------------------------------------------------------
def clean_text(df):
    """Remove extra spaces and normalize text values."""
    df = df.copy()

    for col in df.select_dtypes(include=["object", "category"]).columns:
        df[col] = df[col].apply(
            lambda value: value.strip().lower()
            if isinstance(value, str) else value
        )

    print("Text columns cleaned.")
    return df


# ---------------------------------------------------------
# 4. Drop Useless or Leakage-Prone Columns
# ---------------------------------------------------------
def drop_useless(df):
    """Remove target leakage, redundant, and mostly empty columns."""
    df = df.copy()

    # 'alive' directly reveals the target 'survived'.
    # 'class' duplicates the information in 'pclass'.
    # 'embark_town' duplicates 'embarked'.
    # 'deck' has a large proportion of missing values.
    columns_to_drop = [
        "alive", "class", "embark_town", "deck"
    ]

    df.drop(
        columns=columns_to_drop,
        errors="ignore",
        inplace=True
    )

    # General rule: remove columns with more than 50% missing data.
    mostly_empty = [
        col for col in df.columns
        if col != "survived"
        and df[col].isna().mean() > 0.50
    ]

    df.drop(columns=mostly_empty, inplace=True)

    print("Unnecessary columns removed.")
    print("Remaining columns:", df.columns.tolist())
    return df


# ---------------------------------------------------------
# 5. Handle Missing Values
# ---------------------------------------------------------
def handle_missing(train_df, test_df):
    """
    Impute missing values using training-set statistics only.
    Apply the same values to the test set.
    """
    train_df = train_df.copy()
    test_df = test_df.copy()

    numeric_cols = train_df.select_dtypes(
        include=np.number
    ).columns.tolist()

    categorical_cols = train_df.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    for col in numeric_cols:
        median_value = train_df[col].median()
        train_df[col] = train_df[col].fillna(median_value)
        test_df[col] = test_df[col].fillna(median_value)

    for col in categorical_cols:
        mode_values = train_df[col].mode(dropna=True)

        if not mode_values.empty:
            mode_value = mode_values.iloc[0]
            train_df[col] = train_df[col].fillna(mode_value)
            test_df[col] = test_df[col].fillna(mode_value)

    print("Missing values handled using training statistics.")
    return train_df, test_df


# ---------------------------------------------------------
# 6. Encode Categorical Features
# ---------------------------------------------------------
def encode_categoricals(train_df, test_df):
    """
    Apply binary mapping, ordinal encoding, and one-hot encoding.
    Align test columns with training columns.
    """
    train_df = train_df.copy()
    test_df = test_df.copy()

    # Binary encoding
    binary_maps = {
        "sex": {"male": 0, "female": 1},
        "adult_male": {False: 0, True: 1},
        "alone": {False: 0, True: 1}
    }

    for col, mapping in binary_maps.items():
        if col in train_df.columns:
            train_df[col] = train_df[col].map(mapping)
            test_df[col] = test_df[col].map(mapping)

    # Ordinal encoding for passenger class
    # pclass already represents First=1, Second=2, Third=3.
    # No additional encoding is required for pclass.

    # One-hot encode nominal categorical columns.
    one_hot_cols = [
        col for col in ["embarked", "who"]
        if col in train_df.columns
    ]

    train_df = pd.get_dummies(
        train_df,
        columns=one_hot_cols,
        dtype=int
    )

    test_df = pd.get_dummies(
        test_df,
        columns=one_hot_cols,
        dtype=int
    )

    # Ensure test data has exactly the training feature columns.
    test_df = test_df.reindex(
        columns=train_df.columns,
        fill_value=0
    )

    # Convert any remaining Boolean columns to integers.
    train_df = train_df.astype(
        {col: int for col in train_df.select_dtypes("bool").columns}
    )
    test_df = test_df.astype(
        {col: int for col in test_df.select_dtypes("bool").columns}
    )

    print("Categorical encoding completed.")
    return train_df, test_df


# ---------------------------------------------------------
# 7. Split into Training and Testing Sets
# ---------------------------------------------------------
def split_data(df):
    """Split data into stratified 80% training and 20% testing sets."""
    X = df.drop(columns=["survived"])
    y = df["survived"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print("Stratified train-test split completed.")
    print("Training features:", X_train.shape)
    print("Testing features:", X_test.shape)

    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------
# 8. Scale Numerical Features
# ---------------------------------------------------------
def scale_features(X_train, X_test):
    """Fit StandardScaler on training data and transform both sets."""
    X_train = X_train.copy()
    X_test = X_test.copy()

    scaler = StandardScaler()

    # Fit only on training data to prevent data leakage.
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_train = pd.DataFrame(
        X_train_scaled,
        columns=X_train.columns,
        index=X_train.index
    )

    X_test = pd.DataFrame(
        X_test_scaled,
        columns=X_test.columns,
        index=X_test.index
    )

    print("Feature scaling completed.")
    return X_train, X_test, scaler


# ---------------------------------------------------------
# Main Function
# ---------------------------------------------------------
def main():
    try:
        # Load and validate
        df = load_data()
        df = validate_data(df)

        # Clean and remove unnecessary columns
        df = clean_text(df)
        df = drop_useless(df)

        # Split before fitting any data-dependent preprocessing.
        X_train, X_test, y_train, y_test = split_data(df)

        # Handle missing values using training statistics.
        X_train, X_test = handle_missing(X_train, X_test)

        # Encode categorical columns consistently.
        X_train, X_test = encode_categoricals(X_train, X_test)

        # Scale using training data only.
        X_train, X_test, scaler = scale_features(
            X_train, X_test
        )

        print("\nPipeline completed successfully.")
        print("Final training shape:", X_train.shape)
        print("Final testing shape:", X_test.shape)
        print("Training target shape:", y_train.shape)
        print("Testing target shape:", y_test.shape)

        print("\nRemaining missing values:")
        print("Training:", X_train.isna().sum().sum())
        print("Testing:", X_test.isna().sum().sum())

        print("\nSample of prepared training data:")
        print(X_train.head())

        # Return the prepared data for downstream use.
        return X_train, X_test, y_train, y_test, scaler

    except MissingColumnsError as error:
        print("Validation Error:", error)

    except Exception as error:
        print("Pipeline failed:", error)


# ---------------------------------------------------------
# Execute Script
# ---------------------------------------------------------
if __name__ == "__main__":
    main()