from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
import joblib

def build_pipeline(classifier=None):
    """Build a Scikit-Learn pipeline: TF-IDF + classifier."""
    if classifier is None:
        classifier = LogisticRegression(
            max_iter=1000,
            C=1.0,
            solver="lbfgs",
            random_state=42
        )

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=20_000,     # Adjusted to 20,000 per user constraint
            ngram_range=(1, 2),      # unigrams + bigrams
            sublinear_tf=True,       # apply log normalization
            min_df=2,                # ignore very rare terms
            max_df=0.95,             # ignore very common terms
        )),
        ("clf", classifier)
    ])
    return pipeline

def save_pipeline(pipeline, path="models/pipeline.pkl"):
    joblib.dump(pipeline, path)
    print(f"Pipeline saved to {path}")

def load_pipeline(path="models/pipeline.pkl"):
    return joblib.load(path)
