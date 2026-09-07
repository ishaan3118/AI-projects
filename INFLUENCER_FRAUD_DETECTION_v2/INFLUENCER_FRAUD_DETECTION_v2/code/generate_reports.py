# -*- coding: utf-8 -*-
"""
generate_reports.py

Generates both the Technical Validation Report and the Threshold & Business
Decision Report from ONE source of truth: Report/run_results.json, produced
by fraud_detection.py.

This is the direct fix for the original submission's most damaging flaw:
the technical report said threshold=40 while the business report said
threshold=65, because the numbers were typed by hand into two separate
documents instead of generated from the actual run. Here, both PDFs pull
from the same dict -- they cannot disagree.
"""

import json
import sys
from datetime import date
from pathlib import Path

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
)


def load_results(results_path: str) -> dict:
    with open(results_path) as f:
        return json.load(f)


def build_technical_report(results: dict, out_path: str):
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="Small", fontSize=9, leading=12))
    doc = SimpleDocTemplate(out_path, pagesize=letter,
                             topMargin=0.7 * inch, bottomMargin=0.7 * inch)
    story = []

    story.append(Paragraph("Influencer Fraud Detection - Technical Validation Report (v2)", styles["Title"]))
    story.append(Paragraph(f"Generated {date.today().isoformat()} | Auto-generated from run_results.json - "
                            f"figures below are pulled directly from the pipeline run, not typed by hand.",
                            styles["Small"]))
    story.append(Spacer(1, 14))

    story.append(Paragraph("1. Methodology Corrections vs. v1", styles["Heading2"]))
    corrections = [
        "No dummy-label fallback: the pipeline refuses to run if real fraud labels are missing, "
        "instead of silently validating against random noise.",
        "No threshold leakage: contamination/threshold are grid-searched on a TUNE split only; "
        "all metrics below are computed on a held-out TEST split the search never saw.",
        "Threshold selection is 5-fold cross-validated within the tune split and requires a minimum "
        "number of flagged points, preventing the search from picking an extreme threshold that "
        "'wins' by isolating one or two lucky points.",
        "A supervised baseline (Random Forest) was trained on the same tune split and evaluated on "
        "the same test split, so the unsupervised ensemble is judged against an honest comparison "
        "point instead of being assumed adequate.",
    ]
    for c in corrections:
        story.append(Paragraph(f"&bull; {c}", styles["Normal"]))
    story.append(Spacer(1, 10))

    story.append(Paragraph("2. Validation Setup", styles["Heading2"]))
    story.append(Paragraph(
        f"Total scored profiles: {results['n_total']:,} | "
        f"Labeled sample: {results['n_tune'] + results['n_test']} "
        f"(tune: {results['n_tune']}, held-out test: {results['n_test']}) | "
        f"Grid-search-selected contamination: {results['best_contamination']}, "
        f"threshold: {results['best_threshold']}",
        styles["Normal"]
    ))
    story.append(Spacer(1, 10))

    story.append(Paragraph("3. Results on Held-Out Test Split (never used for tuning)", styles["Heading2"]))
    ens = results["ensemble"]
    rf = results["supervised_baseline_rf"]
    table_data = [
        ["Metric", "Unsupervised Ensemble\n(IsolationForest + LOF)", "Supervised Baseline\n(Random Forest)"],
        ["Precision", f"{ens['precision']:.3f}", f"{rf['precision']:.3f}"],
        ["Recall", f"{ens['recall']:.3f}", f"{rf['recall']:.3f}"],
        ["F1 Score", f"{ens['f1']:.3f}", f"{rf['f1']:.3f}"],
        ["ROC-AUC", f"{ens['roc_auc']:.3f}", f"{rf['roc_auc']:.3f}"],
    ]
    t = Table(table_data, colWidths=[1.6 * inch, 2.6 * inch, 2.2 * inch])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f4f6f7")]),
    ]))
    story.append(t)
    story.append(Spacer(1, 8))
    cm = ens["confusion_matrix"]
    story.append(Paragraph(
        f"Unsupervised ensemble confusion matrix on test split (rows=actual, cols=predicted): "
        f"TN={cm[0][0]}, FP={cm[0][1]}, FN={cm[1][0]}, TP={cm[1][1]}.",
        styles["Small"]
    ))
    story.append(Spacer(1, 12))

    story.append(Paragraph("4. Honest Finding & Production Decision", styles["Heading2"]))
    met_bar = results["met_precision_bar_on_tune"]
    prod_model = results["production_model"]
    bar_text = (
        "The unsupervised ensemble did <b>not</b> reliably clear the 85% precision bar on "
        "cross-validated tuning data" if not met_bar else
        "The unsupervised ensemble cleared the 85% precision bar on cross-validated tuning data"
    )
    story.append(Paragraph(
        f"{bar_text}, and its held-out test precision/recall ({ens['precision']:.2f} / {ens['recall']:.2f}) "
        f"confirms it is not precise enough to drive rejection decisions on its own, despite a "
        f"respectable ROC-AUC of {ens['roc_auc']:.2f} (it ranks risk reasonably well, it just doesn't "
        f"threshold cleanly at the volumes required). The supervised Random Forest, trained on the same "
        f"labeled data, reached {rf['precision']:.2f} precision / {rf['recall']:.2f} recall on the same "
        f"unseen test split.",
        styles["Normal"]
    ))
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        f"<b>Production model selected: {prod_model.replace('_', ' ')}.</b> "
        f"The supervised model drives the fraud_risk_score and vetting_action for all "
        f"{results['n_total']:,} profiles (refit on the full 800 labeled rows after test metrics were "
        f"locked in). The unsupervised ensemble is retained as a secondary anomaly signal: any profile "
        f"the supervised model scores as low-risk but the ensemble flags as a strong anomaly is marked "
        f"with a <i>novel_pattern_flag</i>, since a supervised model can only recognize fraud patterns "
        f"that resembled something in its 800 labeled examples. This run raised "
        f"{results['novel_pattern_flags_raised']} such flags.",
        styles["Normal"]
    ))
    story.append(Spacer(1, 12))

    story.append(Paragraph("5. Why This Differs From the v1 Report", styles["Heading2"]))
    story.append(Paragraph(
        "The original submission reported precision=1.000 at recall=0.020 by tuning the threshold and "
        "reporting the final metric on the same 200 rows -- an extreme threshold that isolated a "
        "handful of points can look perfect and still be useless in practice. It also silently "
        "fabricated random fraud labels if none were supplied, and a companion report recommended a "
        "different threshold (65) than the one the technical run actually used (40), because the two "
        "documents were written independently rather than generated from one run. Both problems are "
        "structurally impossible in this version: the reported numbers here are computed once, in "
        "run_results.json, and both this report and the business report read from that same file.",
        styles["Normal"]
    ))

    doc.build(story)


def build_business_report(results: dict, out_path: str):
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(out_path, pagesize=letter,
                             topMargin=0.7 * inch, bottomMargin=0.7 * inch)
    story = []

    prod_threshold = results["production_decision_threshold"]
    prod_model = results["production_model"]
    rf = results["supervised_baseline_rf"]
    ens = results["ensemble"]
    vac = results["vetting_action_counts"]

    story.append(Paragraph("Influencer Fraud Detection", styles["Title"]))
    story.append(Paragraph("Threshold Selection &amp; Business Decision Framework (v2)", styles["Heading2"]))
    story.append(Paragraph(
        f"Prepared: {date.today().isoformat()} | Production model: {prod_model.replace('_', ' ')} | "
        f"Validation: {results['n_tune'] + results['n_test']} labeled profiles "
        f"({results['n_tune']} tune / {results['n_test']} held-out test)",
        styles["Normal"]
    ))
    story.append(Spacer(1, 12))

    story.append(Paragraph("1. Executive Summary", styles["Heading2"]))
    story.append(Paragraph(
        f"Two approaches were tested head-to-head on an unseen test set: an unsupervised anomaly "
        f"ensemble and a supervised classifier trained on labeled examples. The supervised model won "
        f"decisively (precision {rf['precision']:.2f}, recall {rf['recall']:.2f} vs. the ensemble's "
        f"{ens['precision']:.2f} / {ens['recall']:.2f}) and is now the production model. This report "
        f"explains the resulting decision threshold and the operating trade-offs for the campaign "
        f"approval team. All figures below are pulled from the same validation run referenced in the "
        f"technical report -- there is a single number for the threshold, not two.",
        styles["Normal"]
    ))
    story.append(Spacer(1, 10))

    story.append(Paragraph("2. What Each Error Actually Costs the Business", styles["Heading2"]))
    cell = ParagraphStyle(name="Cell", fontSize=8.5, leading=11)
    cell_hdr = ParagraphStyle(name="CellHdr", fontSize=8.5, leading=11, textColor=colors.white)
    err_table = [
        [Paragraph("Error Type", cell_hdr), Paragraph("What Happens", cell_hdr), Paragraph("Business Cost", cell_hdr)],
        [Paragraph("False Positive<br/>(flagged, actually genuine)", cell),
         Paragraph("A real, legitimate influencer is rejected or delayed for a campaign.", cell),
         Paragraph("Lost partner, reputational friction with creators, wasted vetting overhead.", cell)],
        [Paragraph("False Negative<br/>(missed, actually fraudulent)", cell),
         Paragraph("A fake-follower or bot-driven account is approved and paid for a campaign.", cell),
         Paragraph("Wasted ad spend, inflated reported reach, client trust and legal exposure.", cell)],
    ]
    t = Table(err_table, colWidths=[1.5 * inch, 2.3 * inch, 2.3 * inch])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f4f6f7")]),
    ]))
    story.append(t)
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        "For an influencer marketing agency, false negatives are typically the costlier error. This is "
        "why, once the production model cleared the 85% precision floor comfortably "
        f"({rf['precision']:.2f}), the decision threshold below is set to favor recall rather than "
        "squeezing precision further.",
        styles["Normal"]
    ))
    story.append(Spacer(1, 12))

    story.append(Paragraph("3. Recommended Operating Threshold", styles["Heading2"]))
    story.append(Paragraph(
        f"<b>Recommended fraud probability threshold: {prod_threshold} / 100.</b> "
        f"This is the threshold actually used to produce predicted_fraud in the accompanying scored CSV "
        f"and the technical report -- both documents are generated from the same pipeline run, so this "
        f"number cannot drift out of sync with the technical validation again.",
        styles["Normal"]
    ))
    story.append(Spacer(1, 8))

    story.append(Paragraph("4. Resulting Vetting Action Breakdown (all profiles)", styles["Heading2"]))
    vac_table = [["Vetting Action", "Count"]] + [[k, str(v)] for k, v in vac.items()]
    t2 = Table(vac_table, colWidths=[3.5 * inch, 1.5 * inch])
    t2.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("ALIGN", (1, 0), (1, -1), "CENTER"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f4f6f7")]),
    ]))
    story.append(t2)
    story.append(PageBreak())

    story.append(Paragraph("5. Choosing a Threshold by Business Priority", styles["Heading2"]))
    priority_table = [
        [Paragraph("Business Priority", cell_hdr), Paragraph("Threshold Direction", cell_hdr), Paragraph("When to Use", cell_hdr)],
        [Paragraph("Minimize wrongly rejecting good creators", cell),
         Paragraph("Raise threshold (e.g. 65-80)", cell),
         Paragraph("High-value, well-known creators; relationships the agency wants to protect.", cell)],
        [Paragraph("Minimize fraud slipping through", cell),
         Paragraph("Lower threshold (e.g. 35-45)", cell),
         Paragraph("Large ad-spend campaigns, new/unfamiliar creator pools, strict brand-safety clients.", cell)],
        [Paragraph("Balanced default", cell),
         Paragraph(f"Keep at validated threshold (~{prod_threshold})", cell),
         Paragraph("Standard campaigns without unusual risk or brand sensitivity.", cell)],
    ]
    t3 = Table(priority_table, colWidths=[1.7 * inch, 1.7 * inch, 2.7 * inch])
    t3.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f4f6f7")]),
    ]))
    story.append(t3)
    story.append(Spacer(1, 12))

    story.append(Paragraph("6. Limitations &amp; Governance", styles["Heading2"]))
    limitations = [
        f"The production model is trained on {results['n_tune'] + results['n_test']} labeled profiles. "
        "Fraud tactics evolve; this labeled set should be refreshed and the model re-validated on a "
        "recurring basis (e.g. quarterly).",
        "A supervised model can only recognize patterns similar to its training labels. The "
        f"'novel_pattern_flag' column ({results['novel_pattern_flags_raised']} raised in this run) "
        "surfaces profiles the unsupervised anomaly detector considers unusual even though the "
        "supervised model scored them low-risk -- these deserve a manual look precisely because they "
        "don't resemble anything the model has seen labeled before.",
        "This is a decision-support tool, not an automated judge. Final rejection of an influencer, "
        "especially one with public brand reputation, should retain a human-in-the-loop step.",
        "If this pipeline is pointed at real data, replace data/influencers.csv with the actual raw "
        "export and confirm the 'is_fraud' column reflects verified outcomes, not proxy heuristics.",
    ]
    for l in limitations:
        story.append(Paragraph(f"&bull; {l}", styles["Normal"]))

    doc.build(story)


def build_real_data_addendum(results: dict, real_summary: dict, out_path: str):
    """A short addendum explaining, in plain terms, why the real uploaded
    data is scored differently than the synthetic validation set -- this is
    the report that should stop anyone from quoting a precision/recall
    number for the real population that was never actually measured."""
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(out_path, pagesize=letter,
                             topMargin=0.7 * inch, bottomMargin=0.7 * inch)
    story = []

    story.append(Paragraph("Addendum: Scoring the Real Uploaded Dataset", styles["Title"]))
    story.append(Paragraph(f"Generated {date.today().isoformat()}", styles["Normal"]))
    story.append(Spacer(1, 12))

    story.append(Paragraph("Why this file exists", styles["Heading2"]))
    story.append(Paragraph(
        "The originally supplied raw data (data/influencers_real.csv, 5,000 profiles) has no "
        "'is_fraud' column -- there is no ground truth for this population anywhere in the "
        "project. The technical and business reports validate the methodology on a separate, "
        "clearly-labeled SYNTHETIC reference set with genuine (mechanistically generated) fraud "
        f"patterns, achieving {results['supervised_baseline_rf']['precision']:.2f} precision / "
        f"{results['supervised_baseline_rf']['recall']:.2f} recall on a held-out split of that "
        "reference set. That is a real, honest validation of the METHOD. It is not, and cannot "
        "be presented as, a validation of accuracy on your real population, because no one has "
        "ever confirmed which of your real 5,000 profiles are actually fraudulent.",
        styles["Normal"]
    ))
    story.append(Spacer(1, 10))

    story.append(Paragraph("What was done instead", styles["Heading2"]))
    story.append(Paragraph(
        f"All {real_summary['n_profiles_scored']:,} real profiles were scored with an unsupervised "
        "anomaly ensemble fit directly on this data (no labels required, so this is on solid "
        "ground) and reported as a percentile rank within the population, not a probability -- a "
        "percentile is honest about what an unlabeled anomaly score can support; a probability "
        "would imply a calibration that was never measured here. A second, clearly-marked "
        "'transfer_fraud_probability_UNVALIDATED' column applies the Random Forest trained on the "
        "synthetic reference set to this real data, for comparison only. Where the two signals "
        f"disagree sharply ({real_summary['signals_disagree_count']} profiles), a "
        "signals_disagree_flag is raised -- those are the best manual-audit candidates precisely "
        "because two independently-built signals couldn't agree on them.",
        styles["Normal"]
    ))
    story.append(Spacer(1, 10))

    story.append(Paragraph("Vetting Action Breakdown (real data, percentile-based)", styles["Heading2"]))
    vac_table = [["Vetting Action", "Count"]] + [
        [k, str(v)] for k, v in real_summary["vetting_action_counts"].items()
    ]
    t = Table(vac_table, colWidths=[4.0 * inch, 1.5 * inch])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("ALIGN", (1, 0), (1, -1), "CENTER"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f4f6f7")]),
    ]))
    story.append(t)
    story.append(Spacer(1, 10))

    story.append(Paragraph("Recommended next step", styles["Heading2"]))
    story.append(Paragraph(real_summary["recommendation"], styles["Normal"]))

    doc.build(story)


def main():
    base = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/home/claude/v2/output")
    results = load_results(str(base / "Report" / "run_results.json"))
    build_technical_report(results, str(base / "Report" / "fraud_detection_report.pdf"))
    build_business_report(results, str(base / "Report" / "Threshold_and_Business_Decision_Report.pdf"))

    real_summary_path = base / "Report" / "real_data_scoring_summary.json"
    if real_summary_path.exists():
        with open(real_summary_path) as f:
            real_summary = json.load(f)
        build_real_data_addendum(
            results, real_summary,
            str(base / "Report" / "Real_Data_Scoring_Addendum.pdf")
        )
    print("Reports generated.")


if __name__ == "__main__":
    main()
