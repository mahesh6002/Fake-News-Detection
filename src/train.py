import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report
import joblib
import numpy as np
import sys
import os

# Ensure src is in sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from preprocessing import preprocess_dataframe
from features import build_pipeline, save_pipeline


def clean_and_merge_data(fake_path="data/raw/Fake.csv", true_path="data/raw/True.csv", output_path="data/processed/cleaned.csv"):
    print("Cleaning and merging raw datasets...")
    fake_df = pd.read_csv(fake_path)
    true_df = pd.read_csv(true_path)

    fake_df["label"] = 0
    true_df["label"] = 1

    df = pd.concat([fake_df, true_df], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle

    # Combine title + text
    df["content"] = df["title"].fillna("") + " " + df["text"].fillna("")

    # Drop rows with no content
    df.dropna(subset=["content"], inplace=True)

    # Drop duplicates
    before = len(df)
    df.drop_duplicates(subset=["content"], inplace=True)
    print(f"Dropped {before - len(df):,} duplicate articles")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df[["content", "label"]].to_csv(output_path, index=False)
    print("Saved cleaned dataset to", output_path)


def train(data_path: str = "data/processed/cleaned.csv",
          model_out: str = "models/pipeline.pkl"):

    # If processed data doesn't exist, create it from raw files
    if not os.path.exists(data_path):
        clean_and_merge_data()

    # 1. Load data
    print("Loading data...")
    df = pd.read_csv(data_path)
    print(f"  Loaded {len(df):,} articles")

    # 2. Preprocess
    print("Preprocessing text...")
    df = preprocess_dataframe(df, text_col="content")

    X = df["clean_text"]
    y = df["label"]

    # 3. Split — hold out test set FIRST before any fitting
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  Train: {len(X_train):,} | Test: {len(X_test):,}")

    # 4. Build pipeline
    pipeline = build_pipeline()

    # 5. Cross-validation on training set
    print("\nRunning 5-fold cross-validation...")
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring="f1")
    print(f"  CV F1 scores: {cv_scores}")
    print(f"  Mean CV F1: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")

    # 6. Fit on full training set
    print("\nTraining final model on full training set...")
    pipeline.fit(X_train, y_train)

    # 7. Evaluate on held-out test set
    print("\n--- Test Set Evaluation ---")
    y_pred = pipeline.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=["Fake", "Real"]))

    # 8. Save
    os.makedirs(os.path.dirname(model_out), exist_ok=True)
    save_pipeline(pipeline, model_out)
    print(f"\nModel saved to {model_out}")


if __name__ == "__main__":
    train()
