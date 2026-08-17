from pathlib import Path
from html import escape

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
PERFORMANCE_OBSERVATIONS = [
    {
        "ML Model Name": "Logistic Regression",
        "Observation about model performance": (
            "Best overall model. It achieved the highest Accuracy (0.771), weighted "
            "F1 (0.759), and MCC (0.619). Its Enrolled recall was 0.390, showing that "
            "the minority class remains the hardest outcome to identify."
        ),
    },
    {
        "ML Model Name": "Decision Tree",
        "Observation about model performance": (
            "Lower generalization than the leading models, with 0.663 Accuracy and "
            "0.459 MCC. It recovered 35.8% of Enrolled students but made more errors "
            "across all three classes."
        ),
    },
    {
        "ML Model Name": "kNN",
        "Observation about model performance": (
            "Moderate performance with 0.696 Accuracy and 0.680 weighted F1. Its "
            "Enrolled recall was only 0.277, while Graduate recall was much stronger "
            "at 0.887."
        ),
    },
    {
        "ML Model Name": "Naive Bayes",
        "Observation about model performance": (
            "Weakest overall result, with 0.244 Accuracy and 0.124 MCC. It identified "
            "96.2% of Enrolled cases but only 1.1% of Graduate cases, indicating that "
            "its independence assumptions are unsuitable for these related features."
        ),
    },
    {
        "ML Model Name": "Random Forest (Ensemble)",
        "Observation about model performance": (
            "Best AUC (0.904) and strong Dropout and Graduate recall (0.768 and "
            "0.941). Its low Enrolled recall (0.245) reduced weighted F1 to 0.735."
        ),
    },
    {
        "ML Model Name": "Support Vector Machine",
        "Observation about model performance": (
            "Competitive second-tier result with 0.764 Accuracy, 0.751 weighted F1, "
            "and 0.608 MCC. It was close to Logistic Regression but had a lower AUC "
            "and slightly lower Enrolled recall."
        ),
    },
    {
        "ML Model Name": "Overall winner",
        "Observation about model performance": (
            "Logistic Regression, because it leads Accuracy, weighted F1, and MCC. "
            "Random Forest is the strongest alternative when AUC is the priority."
        ),
    },
]


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
        --ink: #18332d;
        --forest: #146354;
        --deep-forest: #103f37;
        --coral: #c65338;
        --mist: #e8f2ee;
        --paper: #fbfdfc;
        --line: #bed2ca;
    }
    html, body, [class*="st-"] { font-family: "DM Sans", sans-serif; }
    [data-testid="stIconMaterial"] {
        font-family: "Material Symbols Rounded" !important;
    }
    h1, h2, h3 { color: var(--ink); letter-spacing: 0; }
    h1 { font-family: "Newsreader", serif; font-size: 2.15rem !important; }
    [data-testid="stAppViewContainer"] {
        background-color: #f4f7f5;
        background-image:
            linear-gradient(rgba(20, 99, 84, 0.035) 1px, transparent 1px),
            linear-gradient(90deg, rgba(20, 99, 84, 0.035) 1px, transparent 1px),
            linear-gradient(135deg, #f8fbf9 0%, #edf4f1 58%, #f8f2ef 100%);
        background-size: 32px 32px, 32px 32px, 100% 100%;
    }
    [data-testid="stSidebar"] {
        background: var(--deep-forest);
        border-right: 1px solid #2f6a5f;
    }
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
        color: #f7fbf9 !important;
    }
    [data-testid="stSidebar"] h3 {
        border-bottom: 2px solid #e28b72;
        padding-bottom: 0.75rem;
    }
    [data-testid="stSidebar"] [role="group"]:has([role="combobox"]) {
        background: var(--paper);
        border: 2px solid #79ad9f;
        border-radius: 6px;
        box-shadow: 0 5px 16px rgba(5, 29, 24, 0.2);
    }
    [data-testid="stSidebar"] [role="combobox"] {
        color: var(--ink) !important;
        -webkit-text-fill-color: var(--ink) !important;
        font-weight: 700;
    }
    [data-testid="stSidebar"] button[aria-label="Open"] {
        color: var(--forest) !important;
    }
    [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {
        background: var(--mist);
        border: 2px dashed #79ad9f;
        border-radius: 6px;
        padding: 0.75rem;
    }
    [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] * {
        color: var(--ink) !important;
    }
    [data-testid="stSidebar"] [data-testid="stFileChip"] {
        background: var(--paper);
        border: 1px solid #a8c7bd;
        box-shadow: 0 3px 10px rgba(16, 63, 55, 0.12);
    }
    [data-testid="stSidebar"] [data-testid="stFileChipName"] {
        color: var(--ink) !important;
        font-weight: 700;
    }
    [data-testid="stSidebar"] [data-testid="stFileChipDeleteBtn"] button {
        color: var(--coral) !important;
    }
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {
        color: #b8d6cd !important;
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
        background: rgba(255, 255, 255, 0.9);
        border: 1px solid var(--line);
        border-top: 3px solid var(--forest);
        border-radius: 6px;
        padding: 0.85rem 1rem;
        box-shadow: 0 7px 20px rgba(24, 51, 45, 0.07);
    }
    [data-testid="stMetricValue"] { color: var(--forest); }
    .dataset-strip {
        border-left: 4px solid var(--coral);
        background: rgba(255, 255, 255, 0.68);
        padding: 0.55rem 0.9rem;
        margin: 0.2rem 0 1.4rem;
        color: #42514d;
    }
    .run-context {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 1rem;
        flex-wrap: wrap;
        background: var(--deep-forest);
        color: #f7fbf9;
        border-left: 5px solid var(--coral);
        padding: 0.8rem 1rem;
        margin: 0 0 1.2rem;
        box-shadow: 0 8px 22px rgba(16, 63, 55, 0.14);
    }
    .run-context strong { color: #ffd8cc; }
    [role="tab"][aria-selected="true"] {
        color: var(--forest) !important;
        border-bottom-color: var(--coral) !important;
        font-weight: 700;
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

    if normalized.empty:
        raise ValueError("The uploaded CSV contains no data rows.")
    if normalized.columns.duplicated().any():
        duplicate_columns = normalized.columns[normalized.columns.duplicated()].tolist()
        raise ValueError("Duplicate columns: " + ", ".join(duplicate_columns))
    if missing_features:
        raise ValueError("Missing required columns: " + ", ".join(missing_features))
    missing_cells = int(normalized[expected_features].isna().sum().sum())
    if missing_cells:
        raise ValueError(
            f"The uploaded CSV contains {missing_cells} missing feature value(s)."
        )

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

try:
    predictions = model.predict(features)
    probabilities = model.predict_proba(features)
except (TypeError, ValueError):
    st.error(
        "The uploaded CSV contains feature values that are incompatible with "
        "the trained model. Check the column data types and values."
    )
    st.stop()
classes = model.classes_

data_name = uploaded_file.name if uploaded_file is not None else TEST_DATA_PATH.name
st.markdown(
    '<div class="run-context">'
    f'<span><strong>Data</strong> {escape(data_name)} &middot; '
    f'{len(features):,} rows &middot; {len(features.columns)} features</span>'
    f'<span><strong>Active model</strong> {escape(selected_model)}</span>'
    "</div>",
    unsafe_allow_html=True,
)

selected_tab, overview_tab, observations_tab, predictions_tab = st.tabs(
    [
        "Selected model",
        "Model comparison",
        "Performance observations",
        "Predictions",
    ]
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

with observations_tab:
    st.subheader("Observations on model performance")
    st.caption("Results are based on the same untouched 885-row test set.")
    st.table(pd.DataFrame(PERFORMANCE_OBSERVATIONS).set_index("ML Model Name"))

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