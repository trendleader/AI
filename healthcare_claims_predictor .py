"""
Healthcare Claims Predictive Model using Random Forest
======================================================
Predicts whether a healthcare claim will be:
  - Approved
  - Denied
  - Flagged for fraud review

Features used:
  - Patient demographics (age, gender, region)
  - Claim details (amount, diagnosis code category, procedure type)
  - Provider info (provider type, specialty)
  - Historical claim behavior (prior claims, denial history)
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    ConfusionMatrixDisplay,
)
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import warnings

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


# ---------------------------------------------------------------------------
# 1. Synthetic Data Generation
# ---------------------------------------------------------------------------

def generate_claims_data(n_samples: int = 5000) -> pd.DataFrame:
    """Generate realistic synthetic healthcare claims data."""

    diagnosis_categories = [
        "Cardiovascular", "Musculoskeletal", "Respiratory",
        "Mental Health", "Gastrointestinal", "Neurological",
        "Oncology", "Endocrine", "Preventive",
    ]
    procedure_types = [
        "Inpatient", "Outpatient", "Emergency", "Imaging",
        "Lab/Path", "Therapy", "Surgical", "DME",
    ]
    provider_types = ["Hospital", "Physician", "Specialist", "Urgent Care", "Telehealth"]
    regions = ["Northeast", "Southeast", "Midwest", "Southwest", "West"]
    genders = ["M", "F", "Other"]

    age = np.random.randint(18, 90, n_samples)
    gender = np.random.choice(genders, n_samples, p=[0.48, 0.50, 0.02])
    region = np.random.choice(regions, n_samples)
    diagnosis = np.random.choice(diagnosis_categories, n_samples)
    procedure = np.random.choice(procedure_types, n_samples)
    provider_type = np.random.choice(provider_types, n_samples)

    # Claim amount influenced by procedure type
    base_amount = {
        "Inpatient": 15000, "Outpatient": 2500, "Emergency": 5000,
        "Imaging": 1200, "Lab/Path": 400, "Therapy": 600,
        "Surgical": 20000, "DME": 800,
    }
    claim_amount = np.array([
        max(50, np.random.lognormal(
            np.log(base_amount[p]), 0.6
        ))
        for p in procedure
    ])

    prior_claims_12mo = np.random.poisson(3, n_samples)
    prior_denials_12mo = np.random.binomial(prior_claims_12mo, 0.12)
    days_since_last_claim = np.random.exponential(45, n_samples).astype(int)
    is_preauthorized = np.random.choice([0, 1], n_samples, p=[0.25, 0.75])
    duplicate_flag = np.random.choice([0, 1], n_samples, p=[0.97, 0.03])
    out_of_network = np.random.choice([0, 1], n_samples, p=[0.80, 0.20])

    # ---------- Outcome logic (simulates real-world approval rules) ----------
    # Base denial probability
    denial_prob = np.full(n_samples, 0.10)
    denial_prob[~is_preauthorized.astype(bool)] += 0.20
    denial_prob[out_of_network.astype(bool)] += 0.15
    denial_prob += (prior_denials_12mo / (prior_claims_12mo + 1)) * 0.20
    denial_prob[claim_amount > 30000] += 0.10
    denial_prob = np.clip(denial_prob, 0, 0.75)

    # Fraud flag probability
    fraud_prob = np.full(n_samples, 0.03)
    fraud_prob[duplicate_flag.astype(bool)] += 0.35
    fraud_prob[claim_amount > np.percentile(claim_amount, 95)] += 0.10
    fraud_prob += (prior_denials_12mo / (prior_claims_12mo + 1)) * 0.10
    fraud_prob = np.clip(fraud_prob, 0, 0.60)

    outcome = []
    for i in range(n_samples):
        r = np.random.random()
        if r < fraud_prob[i]:
            outcome.append("Fraud Review")
        elif r < fraud_prob[i] + denial_prob[i]:
            outcome.append("Denied")
        else:
            outcome.append("Approved")

    return pd.DataFrame({
        "age": age,
        "gender": gender,
        "region": region,
        "diagnosis_category": diagnosis,
        "procedure_type": procedure,
        "provider_type": provider_type,
        "claim_amount": claim_amount.round(2),
        "prior_claims_12mo": prior_claims_12mo,
        "prior_denials_12mo": prior_denials_12mo,
        "days_since_last_claim": days_since_last_claim,
        "is_preauthorized": is_preauthorized,
        "duplicate_flag": duplicate_flag,
        "out_of_network": out_of_network,
        "outcome": outcome,
    })


# ---------------------------------------------------------------------------
# 2. Preprocessing + Model Pipeline
# ---------------------------------------------------------------------------

CATEGORICAL_FEATURES = [
    "gender", "region", "diagnosis_category", "procedure_type", "provider_type"
]
NUMERIC_FEATURES = [
    "age", "claim_amount", "prior_claims_12mo", "prior_denials_12mo",
    "days_since_last_claim", "is_preauthorized", "duplicate_flag", "out_of_network",
]
TARGET = "outcome"


def build_pipeline() -> Pipeline:
    """Construct a sklearn Pipeline with preprocessing and Random Forest."""
    categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    numeric_transformer = StandardScaler()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ]
    )

    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,          # grow full trees; regularized by min_samples_leaf
        min_samples_leaf=5,
        max_features="sqrt",     # standard for classification
        class_weight="balanced", # handles class imbalance
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    return Pipeline(steps=[("preprocessor", preprocessor), ("classifier", rf)])


# ---------------------------------------------------------------------------
# 3. Evaluation Helpers
# ---------------------------------------------------------------------------

def plot_confusion_matrix(y_true, y_pred, classes, title="Confusion Matrix"):
    fig, ax = plt.subplots(figsize=(7, 5))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=confusion_matrix(y_true, y_pred, labels=classes),
        display_labels=classes,
    )
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    plt.show()
    print("Saved → confusion_matrix.png")


def plot_feature_importance(pipeline: Pipeline, top_n: int = 20):
    """Extract and plot feature importances from the fitted pipeline."""
    rf: RandomForestClassifier = pipeline.named_steps["classifier"]
    pre: ColumnTransformer = pipeline.named_steps["preprocessor"]

    num_names = NUMERIC_FEATURES
    cat_encoder: OneHotEncoder = pre.named_transformers_["cat"]
    cat_names = list(cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES))
    all_names = num_names + cat_names

    importances = pd.Series(rf.feature_importances_, index=all_names)
    top = importances.nlargest(top_n).sort_values()

    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.barh(top.index, top.values, color=plt.cm.RdYlGn(top.values / top.values.max()))
    ax.set_xlabel("Mean Decrease in Impurity (MDI)", fontsize=11)
    ax.set_title(f"Top {top_n} Feature Importances — Random Forest", fontsize=13, fontweight="bold")
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
    plt.tight_layout()
    plt.savefig("feature_importance.png", dpi=150)
    plt.show()
    print("Saved → feature_importance.png")


def plot_class_distribution(y: pd.Series, title: str = "Outcome Distribution"):
    counts = y.value_counts()
    colors = {"Approved": "#2ecc71", "Denied": "#e74c3c", "Fraud Review": "#f39c12"}
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(counts.index, counts.values,
                  color=[colors.get(c, "#95a5a6") for c in counts.index])
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 20,
                f"{val:,}\n({val/len(y)*100:.1f}%)", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Count")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_ylim(0, counts.max() * 1.2)
    plt.tight_layout()
    plt.savefig("class_distribution.png", dpi=150)
    plt.show()
    print("Saved → class_distribution.png")


# ---------------------------------------------------------------------------
# 4. Main Training & Evaluation
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("  Healthcare Claims Predictive Model — Random Forest")
    print("=" * 60)

    # -- Data
    print("\n[1/5] Generating synthetic claims data...")
    df = generate_claims_data(n_samples=5000)
    print(f"      {len(df):,} claims  |  columns: {list(df.columns)}")
    print(f"\n      Outcome distribution:\n{df[TARGET].value_counts().to_string()}\n")
    plot_class_distribution(df[TARGET])

    # -- Split
    print("[2/5] Splitting data (80/20 stratified)...")
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )
    print(f"      Train: {len(X_train):,}  |  Test: {len(X_test):,}")

    # -- Train
    print("\n[3/5] Training Random Forest pipeline...")
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)
    print("      Done.")

    # -- Cross-validation
    print("\n[4/5] 5-fold stratified cross-validation (accuracy)...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=-1)
    print(f"      CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # -- Evaluate on test set
    print("\n[5/5] Test-set evaluation:")
    y_pred = pipeline.predict(X_test)
    classes = pipeline.classes_

    print("\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=classes))

    # AUC (one-vs-rest)
    y_prob = pipeline.predict_proba(X_test)
    auc = roc_auc_score(y_test, y_prob, multi_class="ovr", average="weighted")
    print(f"  Weighted OVR ROC-AUC: {auc:.4f}")

    # -- Visualise
    plot_confusion_matrix(y_test, y_pred, classes)
    plot_feature_importance(pipeline, top_n=20)

    # -- Sample predictions
    print("\n--- Sample Predictions (first 8 test rows) ---")
    sample = X_test.iloc[:8].copy()
    sample["actual"] = y_test.iloc[:8].values
    sample["predicted"] = y_pred[:8]
    probs = pd.DataFrame(y_prob[:8], columns=classes).round(3)
    display_cols = ["age", "claim_amount", "procedure_type", "is_preauthorized",
                    "duplicate_flag", "actual", "predicted"]
    print(pd.concat([sample[display_cols].reset_index(drop=True), probs], axis=1).to_string())

    print("\n" + "=" * 60)
    print("  Model training complete.")
    print("  Outputs: confusion_matrix.png, feature_importance.png, class_distribution.png")
    print("=" * 60)


if __name__ == "__main__":
    main()
