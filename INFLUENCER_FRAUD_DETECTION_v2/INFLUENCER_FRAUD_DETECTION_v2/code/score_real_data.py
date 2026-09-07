# -*- coding: utf-8 -*-
"""
score_real_data.py

Scores the ACTUAL raw dataset the user provided (data/influencers_real.csv),
which -- unlike the synthetic reference used for methodology validation --
has NO `is_fraud` column. This is an important, honest distinction:

    * fraud_detection.py validates the METHODOLOGY end-to-end on a labeled
      reference set (synthetic, since no real labels exist anywhere in this
      project) and reports real precision/recall/AUC on a held-out split.
    * score_real_data.py applies that validated methodology to the real
      population, but since the real population has no ground truth, it
      is NOT possible to honestly report precision/recall on it. Doing so
      anyway -- e.g. by silently reusing the synthetic labels or fabricating
      new ones -- is exactly the mistake that made the original submission's
      "precision 1.000" claim meaningless.

What this script does instead, honestly:
  1. Fits an unsupervised anomaly ensemble directly on the real data (this
     needs no labels at all, so it's on solid ground).
  2. Reports risk as PERCENTILE RANK within the real population rather than
     a probability -- percentile rank is defensible without ground truth;
     a calibrated probability is not.
  3. Additionally applies the Random Forest trained on the synthetic
     reference set as a TRANSFER signal, explicitly labeled as unvalidated
     for this population, for triage/comparison only.
  4. Flags disagreement between the two signals for manual review.
  5. Recommends collecting a manually-audited label sample from THIS real
     population as the next step before any of this drives automated
     rejection decisions.
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

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


def load_real_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Input file is missing required columns: {missing}")
    if df[REQUIRED_COLS[1:]].isnull().any().any():
        raise ValueError("Null values found in required numeric columns; clean the input first.")
    if (df["followers"] <= 0).any():
        raise ValueError("Found rows with followers <= 0; cannot compute ratio features.")
    if "is_fraud" in df.columns:
        print("NOTE: 'is_fraud' column found in real data -- if these are genuine verified "
              "outcomes, re-run fraud_detection.py directly on this file instead of this "
              "script, so you get a real (not synthetic) validated precision/recall.")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    eps = 1e-9
    df["like_follower_ratio"] = df["likes"] / df["followers"].replace(0, eps)
    df["comment_follower_ratio"] = df["comments"] / df["followers"].replace(0, eps)
    df["comment_like_ratio"] = df["comments"] / df["likes"].replace(0, eps)
    df["engagement_consistency"] = df["engagement_rate"] / (df["growth_rate"].abs() + 1)
    df["followers_per_day"] = df["followers"] / df["account_age_days"].replace(0, eps)
    return df


def normalize_risk(raw_scores: np.ndarray) -> np.ndarray:
    s_min, s_max = raw_scores.min(), raw_scores.max()
    norm = (raw_scores - s_min) / (s_max - s_min + 1e-9)
    return (1 - norm) * 100


def percentile_action(pct_rank: float) -> str:
    """Percentile-based buckets -- defensible without ground truth, unlike
    fixed score cutoffs, which imply a calibration this population has never
    actually been validated against."""
    if pct_rank >= 99:
        return "REJECT - Top 1% anomaly, escalate for manual audit"
    elif pct_rank >= 95:
        return "MANUAL REVIEW - Top 5% anomaly"
    elif pct_rank >= 85:
        return "CAUTION - Top 15% anomaly, monitor 2-4 weeks"
    else:
        return "APPROVE - Not a statistical outlier"


def run(input_csv: str, out_dir: str, model_dir: str, contamination: float = 0.05):
    out_dir = Path(out_dir)
    (out_dir / "Dashboard").mkdir(parents=True, exist_ok=True)
    (out_dir / "Influencer Fraud csv").mkdir(parents=True, exist_ok=True)
    (out_dir / "Report").mkdir(parents=True, exist_ok=True)

    df = load_real_data(input_csv)
    df = engineer_features(df)

    # --- Unsupervised ensemble, fit directly on the real population.
    #     This needs no labels, so it's the trustworthy primary signal here. ---
    local_scaler = StandardScaler()
    X_local = local_scaler.fit_transform(df[FEATURE_COLS])

    iso = IsolationForest(n_estimators=300, contamination=contamination, random_state=RANDOM_STATE)
    iso.fit(X_local)
    iso_risk = normalize_risk(iso.decision_function(X_local))

    lof = LocalOutlierFactor(n_neighbors=25, contamination=contamination)
    lof.fit_predict(X_local)
    lof_risk = normalize_risk(lof.negative_outlier_factor_)

    ensemble_risk = 0.5 * iso_risk + 0.5 * lof_risk
    df["anomaly_score"] = np.round(ensemble_risk, 1).clip(0, 100)
    df["anomaly_percentile"] = df["anomaly_score"].rank(pct=True) * 100
    df["vetting_action"] = df["anomaly_percentile"].apply(percentile_action)

    # --- Transfer signal: RF trained on the SYNTHETIC labeled reference,
    #     applied here for comparison only. Explicitly not validated on
    #     this real population -- do not treat as a calibrated probability. ---
    rf_deploy = joblib.load(Path(model_dir) / "rf_deploy_model.joblib")
    rf_proba = rf_deploy.predict_proba(df[FEATURE_COLS].values)[:, 1]
    df["transfer_fraud_probability_UNVALIDATED"] = np.round(rf_proba * 100, 1)

    # --- Disagreement flag: unsupervised says "extreme anomaly" but the
    #     transfer model says "low risk" (or vice versa) -- worth a human
    #     look precisely because the two signals were built completely
    #     differently and rarely both wrong the same way. ---
    df["signals_disagree_flag"] = np.where(
        (df["anomaly_percentile"] >= 95) & (df["transfer_fraud_probability_UNVALIDATED"] < 30),
        "Unsupervised flags anomaly; transfer model does not recognize pattern - review",
        np.where(
            (df["anomaly_percentile"] < 50) & (df["transfer_fraud_probability_UNVALIDATED"] >= 70),
            "Transfer model flags high risk; not a statistical outlier locally - review",
            ""
        )
    )

    summary = {
        "n_profiles_scored": int(len(df)),
        "has_ground_truth_labels": False,
        "primary_signal": "unsupervised anomaly percentile (fit directly on this population, no labels required)",
        "secondary_signal": "supervised transfer probability from synthetic-reference-trained RF (UNVALIDATED on this population)",
        "vetting_action_counts": df["vetting_action"].value_counts().to_dict(),
        "signals_disagree_count": int((df["signals_disagree_flag"] != "").sum()),
        "recommendation": (
            "No verified fraud outcomes exist for this population. Before any automated "
            "rejection is driven by this file, manually audit a sample of ~150-300 profiles "
            "(oversampling the REJECT/MANUAL REVIEW buckets) to build a real labeled set, then "
            "re-run fraud_detection.py directly on that labeled file to get a genuinely "
            "validated precision/recall for THIS population."
        ),
    }

    # --- Dashboard (no true-label boxplot possible -- no labels exist) ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    sns.histplot(df["anomaly_score"], bins=40, kde=True, ax=axes[0, 0])
    axes[0, 0].set_title("Anomaly Score Distribution (real data, unsupervised, unlabeled)")

    sns.scatterplot(
        data=df, x="engagement_rate", y="growth_rate",
        hue="anomaly_score", palette="rocket", alpha=0.6, ax=axes[0, 1]
    )
    axes[0, 1].set_title("Engagement vs Growth Rate (colored by anomaly score)")

    sns.scatterplot(
        data=df, x="anomaly_score", y="transfer_fraud_probability_UNVALIDATED",
        alpha=0.4, ax=axes[1, 0]
    )
    axes[1, 0].set_title("Unsupervised Anomaly vs. Transfer Probability\n(disagreement = candidates for manual audit)")
    axes[1, 0].set_xlabel("Unsupervised anomaly score (0-100)")
    axes[1, 0].set_ylabel("Transfer RF probability (0-100, UNVALIDATED)")

    df["vetting_action"].value_counts().plot.pie(autopct="%1.1f%%", ax=axes[1, 1])
    axes[1, 1].set_ylabel("")
    axes[1, 1].set_title("Vetting Action Breakdown (percentile-based, real data)")

    plt.tight_layout()
    dashboard_path = out_dir / "Dashboard" / "eda_dashboard_real_data.png"
    plt.savefig(dashboard_path, dpi=150)
    plt.close(fig)

    # --- Export scored CSV ---
    output_cols = [
        "name", "followers", "likes", "comments", "growth_rate", "engagement_rate",
        "account_age_days", "anomaly_score", "anomaly_percentile",
        "transfer_fraud_probability_UNVALIDATED", "vetting_action", "signals_disagree_flag",
    ]
    csv_path = out_dir / "Influencer Fraud csv" / "influencer_fraud_scores_REAL_DATA.csv"
    df[output_cols].to_csv(csv_path, index=False)

    summary_path = out_dir / "Report" / "real_data_scoring_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    return summary, dashboard_path, csv_path, summary_path


def main():
    input_csv = sys.argv[1] if len(sys.argv) > 1 else "/home/claude/v2/data/influencers_real.csv"
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "/home/claude/v2/output"
    model_dir = sys.argv[3] if len(sys.argv) > 3 else "/home/claude/v2/output/Report"

    summary, dashboard_path, csv_path, summary_path = run(input_csv, out_dir, model_dir)
    print(json.dumps(summary, indent=2))
    print(f"\nSaved dashboard -> {dashboard_path}")
    print(f"Saved scored CSV -> {csv_path}")
    print(f"Saved summary -> {summary_path}")


if __name__ == "__main__":
    main()
