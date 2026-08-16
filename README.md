# Student Outcome Model Lab

## A. Problem Statement

Student dropout is a significant concern for higher-education institutions. This project compares six classification models that predict whether a student will **Dropout**, remain **Enrolled**, or **Graduate**. The trained models are presented through an interactive Streamlit application where evaluators can upload test data, select a model, inspect evaluation metrics, review its confusion matrix and classification report, and download predictions.

## B. Dataset Description

This project uses the UCI Machine Learning Repository dataset **Predict Students' Dropout and Academic Success**. Each row represents one student and combines information available at enrollment with first- and second-semester academic performance.

| Property | Value |
|---|---|
| Source | [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success) |
| Instances | 4,424 |
| Input features | 36 |
| Target | `Target` |
| Classes | Dropout, Enrolled, Graduate |
| Missing values | None |
| License | CC BY 4.0 |

The dataset exceeds the assignment requirements of at least 500 instances and 12 features. Its class distribution is 2,209 Graduate, 1,421 Dropout, and 794 Enrolled records.

An 80/20 stratified split with `random_state=42` produces 3,539 training rows and 885 test rows. Encoded categorical features are one-hot encoded, while numerical features are standardized. Every preprocessing step is fitted only on training data inside each scikit-learn pipeline to prevent data leakage.

**Dataset citation:** Realinho, V., Vieira Martins, M., Machado, J., & Baptista, L. (2021). *Predict Students' Dropout and Academic Success* [Dataset]. UCI Machine Learning Repository. [https://doi.org/10.24432/C5MC89](https://doi.org/10.24432/C5MC89)

## C. GitHub Repository Link

**Repository:** [github.com/aswinselva-ctrl/ml-assignment-2-student-outcomes](https://github.com/aswinselva-ctrl/ml-assignment-2-student-outcomes)

## D. Models Used

The five models named in the assignment are implemented. Support Vector Machine was added as the sixth model because the assignment requires six models while explicitly listing only five.

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors Classifier
4. Gaussian Naive Bayes Classifier
5. Random Forest Classifier
6. Support Vector Machine with probability calibration

### Model Comparison

All values below are measured on the same untouched 885-row test set. Precision, Recall, and F1 use weighted multiclass averaging. AUC uses weighted one-vs-rest multiclass averaging.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.771 | 0.898 | 0.757 | 0.771 | 0.759 | 0.619 |
| Decision Tree | 0.663 | 0.745 | 0.672 | 0.663 | 0.667 | 0.459 |
| kNN | 0.696 | 0.816 | 0.679 | 0.696 | 0.680 | 0.491 |
| Naive Bayes | 0.244 | 0.712 | 0.532 | 0.244 | 0.167 | 0.124 |
| Random Forest | 0.760 | **0.904** | 0.739 | 0.760 | 0.735 | 0.601 |
| Support Vector Machine | 0.764 | 0.879 | 0.750 | 0.764 | 0.751 | 0.608 |

### Performance Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Best overall model. It achieved the highest Accuracy (0.771), weighted F1 (0.759), and MCC (0.619). Its Enrolled recall was 0.390, showing that the minority class remains the hardest outcome to identify. |
| Decision Tree | Lower generalization than the leading models, with 0.663 Accuracy and 0.459 MCC. It recovered 35.8% of Enrolled students but made more errors across all three classes. |
| kNN | Moderate performance with 0.696 Accuracy and 0.680 weighted F1. Its Enrolled recall was only 0.277, while Graduate recall was much stronger at 0.887. |
| Naive Bayes | Weakest overall result, with 0.244 Accuracy and 0.124 MCC. It identified 96.2% of Enrolled cases but only 1.1% of Graduate cases, indicating that its independence assumptions are unsuitable for these related academic features. |
| Random Forest | Best AUC (0.904) and strong Dropout and Graduate recall (0.768 and 0.941). Its low Enrolled recall (0.245) reduced weighted F1 to 0.735. |
| Support Vector Machine | Competitive second-tier result with 0.764 Accuracy, 0.751 weighted F1, and 0.608 MCC. It was close to Logistic Regression but had a lower AUC and slightly lower Enrolled recall. |
| **Overall winner** | **Logistic Regression**, because it leads the most threshold-based metrics: Accuracy, weighted F1, and MCC. Random Forest is the strongest alternative when ranking quality measured by AUC is the priority. |

## Streamlit Application

**Live application:** Pending deployment in Phase 6.

The application includes:

- CSV test-data upload
- Selection dropdown for all six models
- Reference comparison table
- Accuracy, AUC, Precision, Recall, F1, and MCC
- Confusion matrix and classification report
- Row-level predictions and class probabilities
- Downloadable prediction results

The bundled [`test_data.csv`](test_data.csv) loads by default, so the deployed application displays results immediately. An uploaded CSV must contain the same 36 feature columns. Including a `Target` column enables evaluation metrics; without it, the application provides predictions only.

## Repository Structure

```text
.
|-- app.py
|-- README.md
|-- requirements.txt
|-- test_data.csv
|-- data/
|   `-- raw/student_dropout_success.csv
`-- model/
    |-- data_pipeline.py
    |-- train_models.py
    `-- artifacts/
        |-- model_metrics.csv
        |-- evaluation_details.json
        `-- six trained .joblib pipelines
```

## Local Execution

Python 3.12 is recommended.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app.py
```

Open the local URL printed by Streamlit, normally `http://localhost:8501`.

To reproduce the split and retrain all models:

```bash
python -m model.data_pipeline
python -m model.train_models
```

## BITS Virtual Lab Execution

```bash
sudo dnf install -y git
git clone https://github.com/aswinselva-ctrl/ml-assignment-2-student-outcomes.git
cd ml-assignment-2-student-outcomes
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app.py
```

Open the displayed localhost URL in the BITS Lab browser. Keep PyCharm and the running application visible side by side and capture one screenshot for the submission PDF.

## Streamlit Community Cloud Deployment

1. Sign in at [Streamlit Community Cloud](https://streamlit.io/cloud) with GitHub.
2. Create a new app and select this repository.
3. Select the `main` branch and `app.py` entry point.
4. Choose Python 3.12 in Advanced settings.
5. Deploy and add the resulting URL to this README and the submission PDF.