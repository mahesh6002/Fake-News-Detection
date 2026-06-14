from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt
import os
import pandas as pd
import sys

# Ensure src is in sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from preprocessing import preprocess_dataframe
from features import load_pipeline


def evaluate_model(pipeline, X_test, y_test, save_plots=True):
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    # Classification report
    print("=" * 50)
    print("CLASSIFICATION REPORT")
    print("=" * 50)
    print(classification_report(y_test, y_pred, target_names=["Fake", "Real"]))

    # ROC-AUC
    auc = roc_auc_score(y_test, y_prob)
    print(f"ROC-AUC Score: {auc:.4f}")

    if save_plots:
        os.makedirs("docs", exist_ok=True)
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(cm, display_labels=["Fake", "Real"])
        fig, ax = plt.subplots(figsize=(6, 5))
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title("Confusion Matrix")
        plt.tight_layout()
        plt.savefig("docs/confusion_matrix.png", dpi=150)
        plt.close()

        # ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, color="#2ecc71", label=f"AUC = {auc:.3f}")
        plt.plot([0, 1], [0, 1], "k--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend()
        plt.tight_layout()
        plt.savefig("docs/roc_curve.png", dpi=150)
        plt.close()

    return {"auc": auc, "predictions": y_pred}


if __name__ == "__main__":
    from sklearn.model_selection import train_test_split
    
    model_path = "models/pipeline.pkl"
    data_path = "data/processed/cleaned.csv"
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}. Please run train.py first.")
        sys.exit(1)
        
    if not os.path.exists(data_path):
        print(f"Error: Processed data not found at {data_path}.")
        sys.exit(1)
        
    print("Loading model...")
    pipeline = load_pipeline(model_path)
    
    print("Loading data...")
    df = pd.read_csv(data_path)
    
    print("Preprocessing text...")
    df = preprocess_dataframe(df, text_col="content")
    
    X = df["clean_text"]
    y = df["label"]
    
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print("Evaluating model...")
    evaluate_model(pipeline, X_test, y_test, save_plots=True)
