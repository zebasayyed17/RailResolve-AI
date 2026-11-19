
from pathlib import Path

import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

# ---------------------------------------
# Project paths
# ---------------------------------------

TRAIN_FILE = Path("data/complaints.csv")
EVALUATION_FILE = Path("data/classification_evaluation.csv")

ARTIFACT_DIR = Path("artifacts")
MODEL_FILE = ARTIFACT_DIR / "complaint_classifier.joblib"

RESULTS_FILE = ARTIFACT_DIR / "classification_predictions.csv"
REPORT_FILE = ARTIFACT_DIR / "classification_report.txt"
CONFUSION_FILE = ARTIFACT_DIR / "classification_confusion_matrix.csv"


def load_datasets():
    """Load and validate training and evaluation datasets."""

    if not TRAIN_FILE.exists():
        raise FileNotFoundError(f"Training file missing: {TRAIN_FILE}")

    if not EVALUATION_FILE.exists():
        raise FileNotFoundError(
            f"Evaluation file missing: {EVALUATION_FILE}"
        )

    train_df = pd.read_csv(TRAIN_FILE)
    eval_df = pd.read_csv(EVALUATION_FILE)

    required_train = {"complaint_text", "category"}
    required_eval = {"complaint_text", "expected_category"}

    if not required_train.issubset(train_df.columns):
        raise ValueError("Training dataset has missing columns.")

    if not required_eval.issubset(eval_df.columns):
        raise ValueError("Evaluation dataset has missing columns.")

    train_df = train_df.dropna(
        subset=["complaint_text", "category"]
    )

    eval_df = eval_df.dropna(
        subset=["complaint_text", "expected_category"]
    )

    train_df["complaint_text"] = (
        train_df["complaint_text"].astype(str).str.strip()
    )

    eval_df["complaint_text"] = (
        eval_df["complaint_text"].astype(str).str.strip()
    )

    if train_df.empty or eval_df.empty:
        raise ValueError("Training or evaluation dataset is empty.")

    # Prevent duplicate text leakage between datasets.
    training_texts = {
        text.lower()
        for text in train_df["complaint_text"]
    }

    evaluation_texts = {
        text.lower()
        for text in eval_df["complaint_text"]
    }

    overlap = training_texts.intersection(evaluation_texts)

    if overlap:
        raise ValueError(
            f"Found {len(overlap)} overlapping complaint texts."
        )

    if set(eval_df["expected_category"]) - set(train_df["category"]):
        raise ValueError(
            "Evaluation contains categories missing from training."
        )

    return train_df, eval_df


def build_classifier():
    """Create a TF-IDF and Logistic Regression pipeline."""

    features = FeatureUnion([
        (
            "word_features",
            TfidfVectorizer(
                ngram_range=(1, 2),
                min_df=1,
                sublinear_tf=True,
            ),
        ),
        (
            "character_features",
            TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(3, 5),
                min_df=1,
                sublinear_tf=True,
            ),
        ),
    ])

    classifier = Pipeline([
        ("features", features),
        (
            "model",
            LogisticRegression(
                max_iter=3000,
                class_weight="balanced",
                random_state=42,
            ),
        ),
    ])

    return classifier


def train_and_evaluate():
    """Train the model and evaluate unseen complaints."""

    print("\nLoading complaint datasets...")

    train_df, eval_df = load_datasets()

    X_train = train_df["complaint_text"]
    y_train = train_df["category"]

    X_eval = eval_df["complaint_text"]
    y_eval = eval_df["expected_category"]

    print("\n========== DATASET SUMMARY ==========")
    print("Training records:", len(train_df))
    print("Evaluation records:", len(eval_df))
    print("Training categories:", train_df["category"].nunique())
    print("Training/evaluation overlap: 0")

    # Build model
    model = build_classifier()

    print("\nTraining TF-IDF + Logistic Regression...")

    model.fit(X_train, y_train)

    print("Model training completed!")

    # Evaluate on separate unseen records
    print("\nPredicting evaluation complaints...")

    predictions = model.predict(X_eval)

    accuracy = accuracy_score(y_eval, predictions)

    macro_f1 = f1_score(
        y_eval,
        predictions,
        average="macro",
        zero_division=0,
    )

    # Classification report
    report = classification_report(
        y_eval,
        predictions,
        labels=model.classes_,
        target_names=model.classes_,
        zero_division=0,
    )

    # Create prediction output
    prediction_df = pd.DataFrame({
        "complaint_text": X_eval,
        "actual_category": y_eval,
        "predicted_category": predictions,
        "correct_prediction": (
            y_eval.to_numpy() == predictions
        ),
    })

    # Create confusion matrix
    matrix = confusion_matrix(
        y_eval,
        predictions,
        labels=model.classes_,
    )

    confusion_df = pd.DataFrame(
        matrix,
        index=model.classes_,
        columns=model.classes_,
    )

    # Save artifacts
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, MODEL_FILE)

    prediction_df.to_csv(
        RESULTS_FILE,
        index=False,
    )

    REPORT_FILE.write_text(
        report,
        encoding="utf-8",
    )

    confusion_df.to_csv(CONFUSION_FILE)

    # Display results
    print("\n========== MODEL EVALUATION ==========")

    print(f"Accuracy: {accuracy:.2%}")
    print(f"Macro F1-score: {macro_f1:.4f}")

    print("\nClassification Report:")
    print(report)

    print("\nPredictions:")
    print(prediction_df.to_string(index=False))

    print("\n========== SAVED FILES ==========")
    print("Model:", MODEL_FILE)
    print("Predictions:", RESULTS_FILE)
    print("Classification report:", REPORT_FILE)
    print("Confusion matrix:", CONFUSION_FILE)

    print("=================================")

    return model


def predict_complaint(complaint_text):
    """Predict the category of a new passenger complaint."""

    if not isinstance(complaint_text, str) or not complaint_text.strip():
        raise ValueError("Please enter a valid complaint.")

    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            "Trained model not found. Train the classifier first."
        )

    model = joblib.load(MODEL_FILE)

    prediction = model.predict([complaint_text.strip()])[0]

    return prediction


if __name__ == "__main__":
    train_and_evaluate()
