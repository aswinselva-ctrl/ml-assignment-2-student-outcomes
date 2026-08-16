from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
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


PROJECT_ROOT = Path(__file__).resolve().parent
ARTIFACTS_DIR = PROJECT_ROOT / "model" / "artifacts"
TEST_DATA_PATH = PROJECT_ROOT / "test_data.csv"
TARGET_COLUMN = "Target"
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "kNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Random Forest": "random_forest.joblib",
    "Support Vector Machine": "support_vector_machine.joblib",
}


st.set_page_config(
    page_title="Student Outcome Model Lab",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Newsreader:opsz,wght@6..72,600&display=swap');

    :root {
        --ink: #172421;
        --forest: #175b4d;
        --coral: #c85d3f;
        --mist: #eef4f1;
        --line: #cad8d2;
    }
    html, body, [class*="st-"] { font-family: "DM Sans", sans-serif; }
    [data-testid="stIconMaterial"] {
        font-family: "Material Symbols Rounded" !important;
    }
    h1, h2, h3 { color: var(--ink); letter-spacing: 0; }
    h1 { font-family: "Newsreader", serif; font-size: 2.15rem !important; }
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(140deg, #f7faf8 0%, #edf5f1 55%, #f8f4f1 100%);
    }
    [data-testid="stSidebar"] { background: #173f35; }
    [data-testid="stSidebar"] * { color: #f5faf7; }
    [data-testid="stSidebar"] [data-baseweb="select"] * { color: var(--ink); }
    [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {
        background: #245447;
        border-color: #6e9b8f;
        border-radius: 6px;
    }
    [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button,
    [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button * {
        color: var(--ink) !important;
    }
    div[data-testid="stHorizontalBlock"]:has([data-testid="stMetric"]) {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 0.75rem;
    }
    div[data-testid="stHorizontalBlock"]:has([data-testid="stMetric"]) > div {
        width: 100% !important;
        min-width: 0;
    }
    [data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.72);
        border: 1px solid var(--line);
        border-top: 3px solid var(--forest);
        border-radius: 6px;
        padding: 0.85rem 1rem;
    }
    [data-testid="stMetricValue"] { color: var(--forest); }
    .dataset-strip {
        border-left: 4px solid var(--coral);
        padding: 0.35rem 0 0.35rem 0.9rem;
        margin: 0.2rem 0 1.4rem;
        color: #42514d;
    }
    .block-container { max-width: 1220px; padding-top: 2.5rem; }
    div[data-testid="stDataFrame"] { border: 1px solid var(--line); border-radius: 6px; }
    .stButton > button, .stDownloadButton > button { border-radius: 6px; }
    @media (min-width: 1200px) {
        div[data-testid="stHorizontalBlock"]:has([data-testid="stMetric"]) {
            grid-template-columns: repeat(6, minmax(0, 1fr));
        }
    }
    @media (max-width: 560px) {
        div[data-testid="stHorizontalBlock"]:has([data-testid="stMetric"]) {
            grid-template-columns: repeat(2, minmax(0, 1fr));
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model(model_name: str):
    return joblib.load(ARTIFACTS_DIR / MODEL_FILES[model_name])


@st.cache_data
def load_default_data() -> pd.DataFrame:
    return pd.read_csv(TEST_DATA_PATH)


@st.cache_data
def load_reference_metrics() -> pd.DataFrame:
    return pd.read_csv(ARTIFACTS_DIR / "model_metrics.csv")


def prepare_data(data: pd.DataFrame, model):
    normalized = data.copy()
    normalized.columns = normalized.columns.str.strip()
    expected_features = list(model.feature_names_in_)
    missing_features = sorted(set(expected_features) - set(normalized.columns))

    if missing_features:
        raise ValueError("Missing required columns: " + ", ".join(missing_features))

    features = normalized[expected_features]
    target = normalized[TARGET_COLUMN] if TARGET_COLUMN in normalized else None
    return features, target


def calculate_metrics(target, predictions, probabilities, classes) -> dict:
    try:
        auc = roc_auc_score(
            target,
            probabilities,
            labels=classes,
            multi_class="ovr",
            average="weighted",
        )
    except ValueError:
        auc = np.nan

    return {
        "Accuracy": accuracy_score(target, predictions),
        "AUC": auc,
        "Precision": precision_score(
            target, predictions, average="weighted", zero_division=0
        ),
        "Recall": recall_score(target, predictions, average="weighted", zero_division=0),
        "F1": f1_score(target, predictions, average="weighted", zero_division=0),
        "MCC": matthews_corrcoef(target, predictions),
    }


with st.sidebar:
    st.subheader("Evaluation controls")
    selected_model = st.selectbox("Model", options=list(MODEL_FILES))
    uploaded_file = st.file_uploader("Upload test CSV", type="csv")
    st.caption("UCI Student Dropout and Academic Success")

st.title("Student Outcome Model Lab")
st.markdown(
    '<div class="dataset-strip">Three-class prediction: Dropout, Enrolled, and Graduate</div>',
    unsafe_allow_html=True,
)

model = load_model(selected_model)
try:
    evaluation_data = (
        pd.read_csv(uploaded_file) if uploaded_file is not None else load_default_data()
    )
    features, target = prepare_data(evaluation_data, model)
except (ValueError, pd.errors.ParserError, UnicodeDecodeError) as error:
    st.error(str(error))
    st.stop()

predictions = model.predict(features)
probabilities = model.predict_proba(features)
classes = model.classes_

source_label = "Uploaded test data" if uploaded_file is not None else "Bundled test data"
st.caption(f"{source_label} | {len(features):,} rows | {len(features.columns)} features")

overview_tab, selected_tab, predictions_tab = st.tabs(
    ["Model comparison", "Selected model", "Predictions"]
)

with overview_tab:
    st.subheader("Reference test-set performance")
    reference_metrics = load_reference_metrics().set_index("ML Model Name")
    st.dataframe(
        reference_metrics.style.format("{:.3f}").highlight_max(
            axis=0, color="#cfe5dc"
        ),
        width="stretch",
    )

with selected_tab:
    st.subheader(selected_model)
    if target is None:
        st.info("Add a Target column to the CSV to display evaluation metrics.")
    else:
        unknown_labels = sorted(set(target) - set(classes))
        if unknown_labels:
            st.error("Unknown Target values: " + ", ".join(map(str, unknown_labels)))
            st.stop()

        metrics = calculate_metrics(target, predictions, probabilities, classes)
        metric_columns = st.columns(6)
        for column, (metric_name, value) in zip(metric_columns, metrics.items()):
            display_value = "N/A" if np.isnan(value) else f"{value:.3f}"
            column.metric(metric_name, display_value)

        st.markdown("### Confusion matrix")
        matrix = confusion_matrix(target, predictions, labels=classes)
        figure, axis = plt.subplots(figsize=(7.5, 4.5))
        sns.heatmap(
            matrix,
            annot=True,
            fmt="d",
            cmap=sns.light_palette("#175b4d", as_cmap=True),
            xticklabels=classes,
            yticklabels=classes,
            cbar=False,
            ax=axis,
        )
        axis.set_xlabel("Predicted outcome")
        axis.set_ylabel("Actual outcome")
        figure.tight_layout()
        st.pyplot(figure, width="stretch")
        plt.close(figure)

        st.markdown("### Classification report")
        report = classification_report(
            target,
            predictions,
            labels=classes,
            output_dict=True,
            zero_division=0,
        )
        report_table = pd.DataFrame(report).transpose()
        st.dataframe(
            report_table.style.format("{:.3f}"),
            width="stretch",
        )

with predictions_tab:
    prediction_table = pd.DataFrame({"Predicted outcome": predictions})
    if target is not None:
        prediction_table.insert(0, "Actual outcome", target.to_numpy())
        prediction_table["Correct"] = (
            prediction_table["Actual outcome"] == prediction_table["Predicted outcome"]
        )
    for index, class_name in enumerate(classes):
        prediction_table[f"Probability: {class_name}"] = probabilities[:, index]

    st.dataframe(prediction_table, width="stretch", hide_index=True)
    st.download_button(
        "Download predictions",
        data=prediction_table.to_csv(index=False).encode("utf-8"),
        file_name="student_outcome_predictions.csv",
        mime="text/csv",
    )