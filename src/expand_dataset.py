import os
import re
import json
import urllib.request
import pandas as pd

def setup_kaggle_credentials():
    # Attempt to load credentials from the Downloads folder
    downloads_json = r"C:\Users\srvar\Downloads\kaggle.json"
    home_kaggle_dir = os.path.expanduser("~/.kaggle")
    home_json = os.path.join(home_kaggle_dir, "kaggle.json")
    
    # Check if credentials file exists in Downloads and not in home dir
    if os.path.exists(downloads_json):
        try:
            with open(downloads_json, "r") as f:
                creds = json.load(f)
            os.environ["KAGGLE_USERNAME"] = creds.get("username", "")
            os.environ["KAGGLE_KEY"] = creds.get("key", "")
            print("Loaded Kaggle credentials from Downloads folder.")
            
            # Also write it to ~/.kaggle/kaggle.json just in case some tools look there
            os.makedirs(home_kaggle_dir, exist_ok=True)
            with open(home_json, "w") as f:
                json.dump(creds, f)
        except Exception as e:
            print(f"Warning: Failed to set up Kaggle credentials: {e}")
    else:
        print("Kaggle credentials not found in Downloads. Relying on default path or environment variables.")

def download_kaggle_dataset():
    import kagglehub
    setup_kaggle_credentials()
    
    print("Downloading clmentbisaillon/fake-and-real-news-dataset via kagglehub...")
    path = kagglehub.dataset_download('clmentbisaillon/fake-and-real-news-dataset')
    print(f"Kaggle dataset downloaded to: {path}")
    return path

def download_liar_dataset(dest_dir="data/raw/liar"):
    os.makedirs(dest_dir, exist_ok=True)
    urls = {
        "train.tsv": "https://raw.githubusercontent.com/thiagorainmaker77/liar_dataset/master/train.tsv",
        "test.tsv": "https://raw.githubusercontent.com/thiagorainmaker77/liar_dataset/master/test.tsv",
        "valid.tsv": "https://raw.githubusercontent.com/thiagorainmaker77/liar_dataset/master/valid.tsv"
    }
    
    downloaded_paths = {}
    for filename, url in urls.items():
        dest_path = os.path.join(dest_dir, filename)
        if not os.path.exists(dest_path):
            print(f"Downloading {filename} from LIAR dataset...")
            urllib.request.urlretrieve(url, dest_path)
            print(f"Downloaded to {dest_path}")
        else:
            print(f"{filename} already exists at {dest_path}")
        downloaded_paths[filename] = dest_path
    return downloaded_paths

def load_and_merge_datasets():
    # 1. Download datasets
    kaggle_dir = download_kaggle_dataset()
    liar_paths = download_liar_dataset()
    
    # 2. Load Kaggle Dataset
    print("\nProcessing Kaggle dataset...")
    kaggle_fake_path = os.path.join(kaggle_dir, "Fake.csv")
    kaggle_true_path = os.path.join(kaggle_dir, "True.csv")
    
    kaggle_fake = pd.read_csv(kaggle_fake_path)
    kaggle_true = pd.read_csv(kaggle_true_path)
    
    # Kaggle fields: title, text, subject, date. Combine title + text as content
    kaggle_fake["content"] = kaggle_fake["title"].fillna("") + " " + kaggle_fake["text"].fillna("")
    kaggle_fake["label"] = 0
    
    kaggle_true["content"] = kaggle_true["title"].fillna("") + " " + kaggle_true["text"].fillna("")
    kaggle_true["label"] = 1
    
    kaggle_combined = pd.concat([
        kaggle_fake[["content", "label"]],
        kaggle_true[["content", "label"]]
    ], ignore_index=True)
    
    print(f"  Kaggle loaded: {len(kaggle_combined):,} articles")
    
    # 3. Load LIAR Dataset
    print("\nProcessing LIAR dataset...")
    liar_dfs = []
    for filename, filepath in liar_paths.items():
        # LIAR format is tab-separated, no headers
        df = pd.read_csv(filepath, sep="\t", header=None, on_bad_lines='skip')
        # Col 1 (index 1) is label, Col 2 (index 2) is statement
        df = df[[1, 2]].rename(columns={1: "raw_label", 2: "content"})
        liar_dfs.append(df)
        
    liar_combined = pd.concat(liar_dfs, ignore_index=True)
    print(f"  LIAR loaded: {len(liar_combined):,} statements raw")
    
    # Convert LIAR labels:
    # true/mostly-true -> REAL (1), false/barely-true/pants-fire -> FAKE (0)
    real_labels = ["true", "mostly-true"]
    fake_labels = ["false", "barely-true", "pants-fire", "pants-on-fire"]
    
    liar_combined["label"] = -1
    liar_combined.loc[liar_combined["raw_label"].str.lower().str.strip().isin(real_labels), "label"] = 1
    liar_combined.loc[liar_combined["raw_label"].str.lower().str.strip().isin(fake_labels), "label"] = 0
    
    # Discard half-true or other labels
    liar_combined = liar_combined[liar_combined["label"].isin([0, 1])].reset_index(drop=True)
    print(f"  LIAR mapped binary size: {len(liar_combined):,} statements")
    
    # 4. Combine all
    merged_df = pd.concat([
        kaggle_combined[["content", "label"]],
        liar_combined[["content", "label"]]
    ], ignore_index=True)
    
    print(f"\nTotal merged size (before cleaning): {len(merged_df):,}")
    
    # Drop rows with empty content or label
    merged_df.dropna(subset=["content", "label"], inplace=True)
    merged_df = merged_df[merged_df["content"].str.strip() != ""].reset_index(drop=True)
    
    # Deduplicate on content
    before_dedup = len(merged_df)
    merged_df.drop_duplicates(subset=["content"], inplace=True)
    print(f"Deduplication removed {before_dedup - len(merged_df):,} duplicate articles/statements")
    
    # Check class balance
    class_counts = merged_df["label"].value_counts()
    print("\nClass distribution before balancing:")
    for lbl, cnt in class_counts.items():
        name = "REAL" if lbl == 1 else "FAKE"
        print(f"  {name}: {cnt:,} ({cnt/len(merged_df)*100:.2f}%)")
        
    fake_count = class_counts.get(0, 0)
    real_count = class_counts.get(1, 0)
    total = len(merged_df)
    
    imbalance_ratio = max(fake_count, real_count) / total
    if imbalance_ratio > 0.55:
        print(f"Imbalance ratio {imbalance_ratio:.4f} exceeds 0.55. Applying random undersampling...")
        majority_class = 0 if fake_count > real_count else 1
        minority_class = 1 - majority_class
        
        majority_df = merged_df[merged_df["label"] == majority_class]
        minority_df = merged_df[merged_df["label"] == minority_class]
        
        # Undersample majority class to match minority class size
        majority_sampled = majority_df.sample(n=len(minority_df), random_state=42)
        merged_df = pd.concat([minority_df, majority_sampled], ignore_index=True)
        # Shuffle
        merged_df = merged_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        print("\nClass distribution after balancing:")
        new_counts = merged_df["label"].value_counts()
        for lbl, cnt in new_counts.items():
            name = "REAL" if lbl == 1 else "FAKE"
            print(f"  {name}: {cnt:,} ({cnt/len(merged_df)*100:.2f}%)")
            
    # Save final dataset
    dest_cleaned = "data/processed/cleaned.csv"
    os.makedirs(os.path.dirname(dest_cleaned), exist_ok=True)
    merged_df.to_csv(dest_cleaned, index=False)
    print(f"\nSaved final merged dataset to {dest_cleaned}. Total articles: {len(merged_df):,}")

if __name__ == "__main__":
    load_and_merge_datasets()
