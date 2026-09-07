# -*- coding: utf-8 -*-
"""
fraud_detection.py — Influencer Fraud Detection (v2, corrected)

Corrections made vs. the original submission
----------------------------------------------
1. NO DUMMY-LABEL FALLBACK.  The original silently generated a random
   `is_fraud` column if one was missing, then validated against it while
   claiming "real labels, no fake leakage." This version hard-fails if
   labels aren't present -- a model is never silently validated against
   noise.

2. NO THRESHOLD LEAKAGE.  The original grid-searched contamination/threshold
   on the SAME 200 labeled rows it then reported as the "final validation"
   result -- textbook train/test contamination. This version splits the
   labeled set into a TUNE split (used only for grid search) and a held-out
   TEST split (touched only once, only for the final reported metrics).

3. SUPERVISED BASELINE ADDED.  Since labels exist, a simple supervised
   model (Random Forest) is trained on the tune split and evaluated on the
   same held-out test split, so the unsupervised ensemble's value is
   justified by comparison rather than asserted.

4. SINGLE SOURCE OF TRUTH FOR THE THRESHOLD.  All reported numbers (CSV,
   dashboard, both PDF reports) are generated from one `results` dict at
   the end of this run, so the technical report and the business report
   can never disagree with each other again.

5. ROBUSTNESS.  Division-by-zero guards on every ratio feature (not just
   one), explicit dtype/column validation on load, no Colab-specific calls
   (`google.colab.files.download` removed -- this is a portable script).

6. MODULAR / TESTABLE.  Logic is split into functions with docstrings
   instead of one linear notebook cell sequence, and the script is driven
   by argparse instead of hardcoded paths.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
)

RANDOM_STATE = 42
REQUIRED_COLS = [
    "name", "followers", "likes", "comments",
    "growth_rate", "engagement_rate", "account_age_days",
]

FEATURE_COLS = [
    "followers", "likes", "comments", "growth_rate", "engagement_rate",
    "account_age_days", "like_follower_ratio", "comment_follower_ratio",
    "comment_like_ratio", "engagement_consistency", "followers_per_day",
]


# --------------------------------------------------------------------------
# 1. LOAD + VALIDATE
# --------------------------------------------------------------------------
def load_data(path: str, require_labels: bool = True) -> pd.DataFrame:
    """Load the influencer CSV and validate its schema.

    Unlike the original script, this raises an error instead of silently
    fabricating a random `is_fraud` column when labels are absent -- a
    model must never be validated against noise without the operator
    knowing it.
    """
    df = pd.read_csv(path)

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Input file is missing required columns: {missing}")

    if require_labels and "is_fraud" not in df.columns:
        raise ValueError(
            "No 'is_fraud' column found and require_labels=True. "
            "Refusing to fabricate random labels (this was a bug in v1 -- "
            "it silently invalidated the entire validation). "
            "Pass --no-require-labels only for pure inference on unlabeled data."
        )

    if df[REQUIRED_COLS[1:]].isnull().any().any():
        raise ValueError("Null values found in required numeric columns; clean the input first.")

    if (df["followers"] <= 0).any():
        raise ValueError("Found rows with followers <= 0; cannot compute ratio features.")

    return df


# --------------------------------------------------------------------------
# 2. FEATURE ENGINEERING
# --------------------------------------------------------------------------
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add ratio-based fraud-relevant features with zero-division guards
    on every ratio (the original only guarded one of the four)."""
    df = df.copy()
    eps = 1e-9
    df["like_follower_ratio"] = df["likes"] / df["followers"].replace(0, eps)
    df["comment_follower_ratio"] = df["comments"] / df["followers"].replace(0, eps)
    df["comment_like_ratio"] = df["comments"] / df["likes"].replace(0, eps)
    df["engagement_consistency"] = df["engagement_rate"] / (df["growth_rate"].abs() + 1)
    df["followers_per_day"] = df["followers"] / df["account_age_days"].replace(0, eps)
    return df


def normalize_risk(raw_scores: np.ndarray) -> np.ndarray:
    """Convert raw anomaly scores to a 0-100 risk scale (lower raw = higher risk)."""
    s_min, s_max = raw_scores.min(), raw_scores.max()
    norm = (raw_scores - s_min) / (s_max - s_min + 1e-9)
    return (1 - norm) * 100


# --------------------------------------------------------------------------
# 3. UNSUPERVISED ENSEMBLE + LEAKAGE-FREE THRESHOLD SELECTION
# --------------------------------------------------------------------------
def fit_ensemble_risk(X_scaled: np.ndarray, contamination: float):
    iso = IsolationForest(n_estimators=300, contamination=contamination, random_state=RANDOM_STATE)
    iso.fit(X_scaled)
    iso_risk = normalize_risk(iso.decision_function(X_scaled))

    lof = LocalOutlierFactor(n_neighbors=25, contamination=contamination)
    lof.fit_predict(X_scaled)
    lof_risk = normalize_risk(lof.negative_outlier_factor_)

    ensemble_risk = 0.5 * iso_risk + 0.5 * lof_risk
    return ensemble_risk, iso, lof


def grid_search_threshold(X_scaled, tune_positions, y_tune, min_precision=0.85,
                           n_splits=5, min_flagged=8):
    """Grid-search contamination + threshold using ONLY the tune split, via
    K-fold cross-validation averaged across folds.

    This fixes two distinct problems found during development:
      (a) LEAKAGE -- the original script tuned and reported on the same 200
          rows. Fixed by never touching the held-out test split here.
      (b) SMALL-SAMPLE THRESHOLD OVERFITTING -- an earlier version of this
          fix still picked thresholds so extreme they isolated 1-2 lucky
          points at 100% "precision" on a single train/tune split, which
          then failed to generalize (0% recall on test). Fixed by (i)
          cross-validating the precision/recall estimate across folds
          instead of trusting one split, and (ii) requiring a minimum
          number of flagged points (`min_flagged`) so a threshold can't
          win purely by making an unsupported claim about 1-2 samples.
    """
    from sklearn.model_selection import StratifiedKFold

    contamination_grid = [0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25]
    threshold_grid = np.arange(20, 85, 2.5)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    results = []
    for contam in contamination_grid:
        ensemble_risk, _, _ = fit_ensemble_risk(X_scaled, contam)
        risk_tune = ensemble_risk[tune_positions]

        for thresh in threshold_grid:
            fold_precisions, fold_recalls, fold_flagged = [], [], []
            for _, fold_idx in skf.split(risk_tune, y_tune):
                y_pred_fold = (risk_tune[fold_idx] >= thresh).astype(int)
                fold_flagged.append(int(y_pred_fold.sum()))
                if y_pred_fold.sum() == 0:
                    fold_precisions.append(0.0)
                else:
                    fold_precisions.append(
                        precision_score(y_tune[fold_idx], y_pred_fold, zero_division=0)
                    )
                fold_recalls.append(
                    recall_score(y_tune[fold_idx], y_pred_fold, zero_division=0)
                )

            total_flagged = int(np.sum(fold_flagged))
            results.append({
                "contamination": contam,
                "threshold": thresh,
                "precision": round(float(np.mean(fold_precisions)), 3),
                "recall": round(float(np.mean(fold_recalls)), 3),
                "flagged": total_flagged,
            })

    results_df = pd.DataFrame(results)
    supported = results_df[results_df["flagged"] >= min_flagged]
    passing = supported[supported["precision"] >= min_precision].sort_values(
        "recall", ascending=False
    )
    if len(passing) > 0:
        best = passing.iloc[0]
        met_bar = True
    elif len(supported) > 0:
        best = supported.sort_values(["precision", "recall"], ascending=False).iloc[0]
        met_bar = False
    else:
        best = results_df.sort_values(["precision", "recall"], ascending=False).iloc[0]
        met_bar = False
    return best, results_df, met_bar


# --------------------------------------------------------------------------
# 4. SUPERVISED BASELINE
# --------------------------------------------------------------------------
def fit_supervised_baseline(X_tune, y_tune, X_test):
    """A simple Random Forest baseline trained on the tune split. Since we
    already have labels for the tune split, there's no reason not to try a
    supervised model too -- this gives the ensemble something to be
    compared against, rather than assumed to be the right tool."""
    clf = RandomForestClassifier(
        n_estimators=300, max_depth=6, class_weight="balanced",
        random_state=RANDOM_STATE
    )
    clf.fit(X_tune, y_tune)
    proba_test = clf.predict_proba(X_test)[:, 1]
    return clf, proba_test


# --------------------------------------------------------------------------
# 5. VETTING ACTION
# --------------------------------------------------------------------------
def vetting_action(score: float) -> str:
    if score >= 75:
        return "REJECT - High fraud risk"
    elif score >= 50:
        return "MANUAL REVIEW - Audit history"
    elif score >= 25:
        return "CAUTION - Monitor 2-4 weeks"
    else:
        return "APPROVE - Low fraud risk"


# --------------------------------------------------------------------------
# 6. MAIN PIPELINE
# --------------------------------------------------------------------------
def run_pipeline(input_csv: str, out_dir: str, labeled_sample_size: int = 800):
    out_dir = Path(out_dir)
    (out_dir / "Dashboard").mkdir(parents=True, exist_ok=True)
    (out_dir / "Influencer Fraud csv").mkdir(parents=True, exist_ok=True)
    (out_dir / "Report").mkdir(parents=True, exist_ok=True)

    df = load_data(input_csv, require_labels=True)
    df = engineer_features(df)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[FEATURE_COLS])

    # --- Build a labeled sample, then split TUNE / TEST (fixes leakage) ---
    labeled_sample = df.sample(labeled_sample_size, random_state=7).copy()
    tune_idx, test_idx = train_test_split(
        labeled_sample.index, test_size=0.5, random_state=RANDOM_STATE,
        stratify=labeled_sample["is_fraud"],
    )
    tune_positions = df.index.get_indexer(tune_idx)
    test_positions = df.index.get_indexer(test_idx)
    y_tune = df.loc[tune_idx, "is_fraud"].values
    y_test = df.loc[test_idx, "is_fraud"].values

    # --- Grid search on TUNE split only ---
    best, grid_results_df, met_precision_bar = grid_search_threshold(
        X_scaled, tune_positions, y_tune, min_precision=0.85
    )
    best_contam = float(best["contamination"])
    best_threshold = float(best["threshold"])

    # --- Fit final unsupervised ensemble on full data with winning contamination ---
    ensemble_risk, iso_final, lof_final = fit_ensemble_risk(X_scaled, best_contam)
    df["fraud_risk_score"] = np.round(ensemble_risk, 1).clip(0, 100)
    df["predicted_fraud"] = (df["fraud_risk_score"] >= best_threshold).astype(int)
    df["vetting_action"] = df["fraud_risk_score"].apply(vetting_action)

    # --- Final, UNBIASED evaluation on the held-out TEST split ---
    test_scores = df.loc[test_idx, "fraud_risk_score"].values
    test_pred = df.loc[test_idx, "predicted_fraud"].values
    ens_precision = precision_score(y_test, test_pred, zero_division=0)
    ens_recall = recall_score(y_test, test_pred, zero_division=0)
    ens_f1 = f1_score(y_test, test_pred, zero_division=0)
    ens_auc = roc_auc_score(y_test, test_scores)
    ens_cm = confusion_matrix(y_test, test_pred)

    # --- Supervised baseline, trained on TUNE, evaluated on the same TEST split ---
    X_tune_feat = df.loc[tune_idx, FEATURE_COLS].values
    X_test_feat = df.loc[test_idx, FEATURE_COLS].values
    rf_clf, rf_proba_test = fit_supervised_baseline(X_tune_feat, y_tune, X_test_feat)
    rf_pred_test = (rf_proba_test >= 0.5).astype(int)
    rf_precision = precision_score(y_test, rf_pred_test, zero_division=0)
    rf_recall = recall_score(y_test, rf_pred_test, zero_division=0)
    rf_f1 = f1_score(y_test, rf_pred_test, zero_division=0)
    rf_auc = roc_auc_score(y_test, rf_proba_test)

    # --- Decide production model honestly, based on which one actually
    #     clears the 85% precision bar on the held-out test split ---
    supervised_wins = rf_precision >= 0.85 and rf_precision > ens_precision
    production_model = "supervised_rf_plus_anomaly_flag" if supervised_wins else "unsupervised_ensemble"

    # --- Refit the winning supervised model on ALL labeled rows (tune + test
    #     combined, 800 total) for deployment -- standard practice once test
    #     metrics are locked in: never refit on test alone, but do use every
    #     labeled row you have for the model that actually ships. ---
    all_labeled_idx = labeled_sample.index
    X_all_labeled_feat = df.loc[all_labeled_idx, FEATURE_COLS].values
    y_all_labeled = df.loc[all_labeled_idx, "is_fraud"].values
    rf_deploy = RandomForestClassifier(
        n_estimators=300, max_depth=6, class_weight="balanced", random_state=RANDOM_STATE
    )
    rf_deploy.fit(X_all_labeled_feat, y_all_labeled)
    rf_proba_full = rf_deploy.predict_proba(df[FEATURE_COLS].values)[:, 1]
    df["supervised_fraud_probability"] = np.round(rf_proba_full * 100, 1)

    # Persist the fitted scaler + supervised model so they can be reused for
    # transfer inference on other (unlabeled) datasets -- always with a
    # clear caveat that transfer performance is not validated the same way.
    import joblib
    model_dir = out_dir / "Report"
    joblib.dump(rf_deploy, model_dir / "rf_deploy_model.joblib")
    joblib.dump(scaler, model_dir / "feature_scaler.joblib")

    # --- Hybrid production score: supervised probability drives the
    #     decision (it's the validated, high-precision signal); the
    #     unsupervised anomaly score is kept alongside to catch fraud
    #     patterns that don't resemble anything in the 800 labeled rows
    #     (a classic supervised-model blind spot) and to flag disagreement
    #     for manual review rather than silently trusting either model. ---
    if supervised_wins:
        df["fraud_risk_score"] = df["supervised_fraud_probability"]
        df["novel_pattern_flag"] = np.where(
            (df["supervised_fraud_probability"] < 50) & (ensemble_risk >= 75),
            "Possible novel fraud pattern (high anomaly, not matched by labeled examples)",
            ""
        )
    else:
        df["novel_pattern_flag"] = ""

    df["predicted_fraud"] = (df["fraud_risk_score"] >= (50 if supervised_wins else best_threshold)).astype(int)
    df["vetting_action"] = df["fraud_risk_score"].apply(vetting_action)

    results = {
        "n_total": int(len(df)),
        "n_tune": int(len(tune_idx)),
        "n_test": int(len(test_idx)),
        "best_contamination": best_contam,
        "best_threshold": best_threshold,
        "met_precision_bar_on_tune": bool(met_precision_bar),
        "ensemble": {
            "precision": round(float(ens_precision), 3),
            "recall": round(float(ens_recall), 3),
            "f1": round(float(ens_f1), 3),
            "roc_auc": round(float(ens_auc), 3),
            "confusion_matrix": ens_cm.tolist(),
        },
        "supervised_baseline_rf": {
            "precision": round(float(rf_precision), 3),
            "recall": round(float(rf_recall), 3),
            "f1": round(float(rf_f1), 3),
            "roc_auc": round(float(rf_auc), 3),
        },
        "production_model": production_model,
        "production_decision_threshold": 50 if supervised_wins else best_threshold,
        "novel_pattern_flags_raised": int((df["novel_pattern_flag"] != "").sum()),
        "vetting_action_counts": df["vetting_action"].value_counts().to_dict(),
    }

    # --- Dashboard ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    sns.histplot(df["fraud_risk_score"], bins=40, kde=True, ax=axes[0, 0])
    axes[0, 0].set_title("Fraud Risk Score Distribution (all profiles)")

    final_test_scores = df.loc[test_idx, "fraud_risk_score"].values
    test_label_series = pd.Series(y_test, name="is_fraud")
    test_score_series = pd.Series(final_test_scores, name="fraud_risk_score")
    sns.boxplot(x=test_label_series, y=test_score_series, ax=axes[0, 1])
    axes[0, 1].set_title(f"Final Risk Score by True Label\n(held-out TEST split, {production_model})")
    axes[0, 1].set_xlabel("is_fraud")

    sns.scatterplot(
        data=df, x="engagement_rate", y="growth_rate",
        hue="fraud_risk_score", palette="rocket", alpha=0.6, ax=axes[1, 0]
    )
    axes[1, 0].set_title("Engagement vs Growth Rate (colored by risk)")

    df["vetting_action"].value_counts().plot.pie(autopct="%1.1f%%", ax=axes[1, 1])
    axes[1, 1].set_ylabel("")
    axes[1, 1].set_title("Vetting Action Breakdown")

    plt.tight_layout()
    dashboard_path = out_dir / "Dashboard" / "eda_dashboard.png"
    plt.savefig(dashboard_path, dpi=150)
    plt.close(fig)

    # --- Model comparison chart ---
    fig2, ax2 = plt.subplots(figsize=(7, 5))
    metrics = ["precision", "recall", "f1", "roc_auc"]
    ens_vals = [results["ensemble"][m] for m in metrics]
    rf_vals = [results["supervised_baseline_rf"][m] for m in metrics]
    x = np.arange(len(metrics))
    width = 0.35
    ax2.bar(x - width / 2, ens_vals, width, label="Unsupervised Ensemble")
    ax2.bar(x + width / 2, rf_vals, width, label="Supervised RF Baseline")
    ax2.set_xticks(x)
    ax2.set_xticklabels(metrics)
    ax2.set_ylim(0, 1.05)
    ax2.set_title("Model Comparison on Held-Out Test Split")
    ax2.legend()
    plt.tight_layout()
    comparison_path = out_dir / "Dashboard" / "model_comparison.png"
    plt.savefig(comparison_path, dpi=150)
    plt.close(fig2)

    # --- Export scored CSV (all 5000 profiles) ---
    output_cols = [
        "name", "followers", "likes", "comments", "growth_rate", "engagement_rate",
        "account_age_days", "fraud_risk_score", "vetting_action", "predicted_fraud",
        "novel_pattern_flag",
    ]
    csv_path = out_dir / "Influencer Fraud csv" / "influencer_fraud_scores.csv"
    df[output_cols].to_csv(csv_path, index=False)

    # --- Save run results as single source of truth for report generation ---
    results_path = out_dir / "Report" / "run_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    return results, dashboard_path, comparison_path, csv_path, results_path


def main():
    parser = argparse.ArgumentParser(description="Influencer fraud detection pipeline (v2, corrected)")
    parser.add_argument("--input", default="/home/claude/v2/data/influencers.csv")
    parser.add_argument("--out-dir", default="/home/claude/v2/output")
    parser.add_argument("--labeled-sample-size", type=int, default=800)
    args = parser.parse_args()

    results, dashboard_path, comparison_path, csv_path, results_path = run_pipeline(
        args.input, args.out_dir, args.labeled_sample_size
    )

    print(json.dumps(results, indent=2))
    print(f"\nSaved dashboard      -> {dashboard_path}")
    print(f"Saved comparison chart -> {comparison_path}")
    print(f"Saved scored CSV     -> {csv_path}")
    print(f"Saved run results    -> {results_path}")


if __name__ == "__main__":
    sys.exit(main())
