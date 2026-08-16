import json
from collections import OrderedDict
from pathlib import Path

import joblib
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from model.data_pipeline import (
    PROJECT_ROOT,
    RANDOM_STATE,
    build_preprocessor,
    load_source_data,
    split_data,
)


ARTIFACTS_DIR = PROJECT_ROOT / "model" / "artifacts"
METRICS_PATH = ARTIFACTS_DIR / "model_metrics.csv"
DETAILS_PATH = ARTIFACTS_DIR / "evaluation_details.json"


def build_classifiers() -> OrderedDict[str, ClassifierMixin]:
    return OrderedDict(
        [
            ("Logistic Regression", LogisticRegression(max_iter=2_000)),
            ("Decision Tree", DecisionTreeClassifier(random_state=RANDOM_STATE)),
            ("kNN", KNeighborsClassifier(n_neighbors=5)),
            ("Naive Bayes", GaussianNB()),
            (
                "Random Forest",
                RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE),
            ),
            (
                "Support Vector Machine",
                CalibratedClassifierCV(
                    SVC(random_state=RANDOM_STATE),
                    method="sigmoid",
                    cv=5,
                    ensemble=False,
                ),
            ),
        ]
    )


def artifact_name(model_name: str) -> str:
    return model_name.lower().replace(" ", "_") + ".joblib"


def evaluate_model(model: Pipeline, test_features, test_target) -> tuple[dict, dict]:
    predictions = model.predict(test_features)
    probabilities = model.predict_proba(test_features)
    classes = model.classes_

    metrics = {
        "Accuracy": accuracy_score(test_target, predictions),
        "AUC": roc_auc_score(
            test_target,
            probabilities,
            labels=classes,
            multi_class="ovr",
            average="weighted",
        ),
        "Precision": precision_score(
            test_target, predictions, average="weighted", zero_division=0
        ),
        "Recall": recall_score(
            test_target, predictions, average="weighted", zero_division=0
        ),
        "F1": f1_score(test_target, predictions, average="weighted", zero_division=0),
        "MCC": matthews_corrcoef(test_target, predictions),
    }
    details = {
        "classes": classes.tolist(),
        "confusion_matrix": confusion_matrix(
            test_target, predictions, labels=classes
        ).tolist(),
        "classification_report": classification_report(
            test_target,
            predictions,
            labels=classes,
            output_dict=True,
            zero_division=0,
        ),
    }
    return metrics, details


def train_and_evaluate() -> pd.DataFrame:
    data = load_source_data()
    train_features, test_features, train_target, test_target = split_data(data)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    metric_rows = []
    evaluation_details = {}

    for model_name, classifier in build_classifiers().items():
        pipeline = Pipeline(
            [
                ("preprocessor", build_preprocessor()),
                ("classifier", classifier),
            ]
        )
        pipeline.fit(train_features, train_target)
        metrics, details = evaluate_model(pipeline, test_features, test_target)

        metric_rows.append({"ML Model Name": model_name, **metrics})
        evaluation_details[model_name] = details
        joblib.dump(pipeline, ARTIFACTS_DIR / artifact_name(model_name))
        print(f"Trained {model_name}")

    metrics_table = pd.DataFrame(metric_rows)
    metrics_table.to_csv(METRICS_PATH, index=False)
    DETAILS_PATH.write_text(json.dumps(evaluation_details, indent=2), encoding="utf-8")
    return metrics_table


if __name__ == "__main__":
    results = train_and_evaluate()
    print("\nWeighted multiclass test metrics:")
    print(results.to_string(index=False, float_format=lambda value: f"{value:.4f}"))