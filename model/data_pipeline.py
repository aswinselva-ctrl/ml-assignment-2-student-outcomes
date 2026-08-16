from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "student_dropout_success.csv"
TEST_DATA_PATH = PROJECT_ROOT / "test_data.csv"
TARGET_COLUMN = "Target"
RANDOM_STATE = 42

CATEGORICAL_FEATURES = [
    "Marital status",
    "Application mode",
    "Course",
    "Daytime/evening attendance",
    "Previous qualification",
    "Nacionality",
    "Mother's qualification",
    "Father's qualification",
    "Mother's occupation",
    "Father's occupation",
    "Displaced",
    "Educational special needs",
    "Debtor",
    "Tuition fees up to date",
    "Gender",
    "Scholarship holder",
    "International",
]


def load_source_data(path: Path = SOURCE_DATA_PATH) -> pd.DataFrame:
    data = pd.read_csv(path, sep=";")
    data.columns = data.columns.str.strip()

    if TARGET_COLUMN not in data.columns:
        raise ValueError(f"Required target column '{TARGET_COLUMN}' was not found.")
    if data.shape[1] - 1 != 36:
        raise ValueError(f"Expected 36 features, found {data.shape[1] - 1}.")
    if data.isna().any().any():
        raise ValueError("The source dataset contains missing values.")

    return data


def split_data(data: pd.DataFrame):
    features = data.drop(columns=TARGET_COLUMN)
    target = data[TARGET_COLUMN]
    return train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=target,
    )


def build_preprocessor() -> ColumnTransformer:
    numerical_features = [
        column for column in load_source_data().columns
        if column not in CATEGORICAL_FEATURES + [TARGET_COLUMN]
    ]
    return ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
            ("numerical", StandardScaler(), numerical_features),
        ]
    )


def create_test_data(output_path: Path = TEST_DATA_PATH) -> Path:
    data = load_source_data()
    _, test_features, _, test_target = split_data(data)
    test_data = test_features.copy()
    test_data[TARGET_COLUMN] = test_target
    test_data.to_csv(output_path, index=False)
    return output_path


if __name__ == "__main__":
    created_path = create_test_data()
    print(f"Created {created_path}")