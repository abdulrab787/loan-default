import pandas as pd
import numpy as np
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# 1️⃣ Define target
TARGET_COL = "default"

# 2️⃣ Feature cleaning & engineering
def basic_feature_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Standardize text columns (lowercase)
    text_cols = df.select_dtypes(include=["object"]).columns
    for c in text_cols:
        df[c] = df[c].astype(str).str.lower()

    # Clean employment length -> convert to numeric
    if "emp_length" in df.columns:
        df["emp_length"] = (
            df["emp_length"]
            .str.replace("+ years", "", regex=False)
            .str.replace("< 1 year", "0", regex=False)
            .str.replace(" years", "", regex=False)
            .str.replace(" year", "", regex=False)
        )
        df["emp_length"] = pd.to_numeric(df["emp_length"], errors="coerce")

    return df

# 3️⃣ Build preprocessing pipeline
def build_preprocessor(df: pd.DataFrame):
    df = df.copy()

    # Separate target
    y = df[TARGET_COL]
    X = df.drop(columns=[TARGET_COL])

    # Identify column types
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()

    # Drop columns that are "leaky" or post-loan (credit risk rule)
    drop_cols = [
        "loan_status",
        "recoveries",
        "collection_recovery_fee",
        "last_pymnt_amnt",
        "total_rec_int",
        "total_rec_prncp",
        "total_pymnt",
        "total_pymnt_inv"
    ]

    drop_cols = [c for c in drop_cols if c in X.columns]

    for c in drop_cols:
        if c in num_cols:
            num_cols.remove(c)
        if c in cat_cols:
            cat_cols.remove(c)

    # Numeric pipeline
    num_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # Categorical pipeline
    cat_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    # ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, num_cols),
            ("cat", cat_pipeline, cat_cols),
            ("drop", "drop", drop_cols)
        ]
    )

    return preprocessor, X, y

# 4️⃣ Fit + Save artifacts
def fit_and_save_preprocessor(df: pd.DataFrame, save_dir: str = "../models"):
    df = basic_feature_cleaning(df)

    preprocessor, X, y = build_preprocessor(df)

    # Train/validation split (same everywhere)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Fit preprocessor on TRAIN only
    X_train_processed = preprocessor.fit_transform(X_train)

    # Transform validation
    X_val_processed = preprocessor.transform(X_val)

    # Save artifacts
    joblib.dump(preprocessor, f"{save_dir}/preprocessor.pkl")
    joblib.dump(X_train_processed, f"{save_dir}/X_train_processed.pkl")
    joblib.dump(X_val_processed, f"{save_dir}/X_val_processed.pkl")
    joblib.dump(y_train, f"{save_dir}/y_train.pkl")
    joblib.dump(y_val, f"{save_dir}/y_val.pkl")

    print("Preprocessing complete and saved:")
    print(f"- {save_dir}/preprocessor.pkl")
    print(f"- {save_dir}/X_train_processed.pkl")
    print(f"- {save_dir}/X_val_processed.pkl")
    print(f"- {save_dir}/y_train.pkl")
    print(f"- {save_dir}/y_val.pkl")

    return preprocessor
