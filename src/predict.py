import pandas as pd
import joblib
import scipy.sparse as sp

def load_artifacts():
    preprocessor = joblib.load("../models/preprocessor.pkl")
    model_bundle = joblib.load("../models/ensemble_model.pkl")
    return preprocessor, model_bundle

def predict(test_path, output_path):
    print("Loading artifacts...")
    preprocessor, model_bundle = load_artifacts()

    print("Reading test data...")
    df = pd.read_csv(test_path)

    # Keep ID column if exists
    if "id" in df.columns:
        ids = df["id"]
    else:
        ids = None

    print("Preprocessing...")
    X_processed = preprocessor.transform(df)
    X_sparse = sp.csr_matrix(X_processed)

    logreg = model_bundle["logreg"]
    lgb = model_bundle["lightgbm"]
    threshold = model_bundle["threshold"]

    print("Predicting...")
    logreg_probs = logreg.predict_proba(X_processed)[:, 1]
    lgb_probs = lgb.predict_proba(X_sparse)[:, 1]

    ensemble_probs = (logreg_probs + lgb_probs) / 2
    final_preds = (ensemble_probs >= threshold).astype(int)

    submission = pd.DataFrame({
        "prediction": final_preds
    })

    if ids is not None:
        submission.insert(0, "id", ids)

    submission.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")

if __name__ == "__main__":
    test_file = "../data/raw/test.csv"
    output_file = "../data/processed/submission.csv"
    predict(test_file, output_file)
