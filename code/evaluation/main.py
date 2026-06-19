"""
evaluation/main.py — Evaluation pipeline for the damage claim verification system.
Runs on sample_claims.csv which has both inputs AND expected outputs.
Compares Strategy A (first image only) vs Strategy B (all images).
Writes evaluation/evaluation_report.md with full metrics.
"""

import csv
import os
import sys
import time

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Add code/ directory to sys.path so we can import sibling modules
CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, CODE_DIR)

from image_utils import load_all_images
from loader import enrich_claims, load_claims, load_evidence_requirements, load_user_history
from risk import assess_risk
from validator import validate_and_fix
from vlm import analyze_claim, print_stats

console = Console()

REPO_ROOT = os.path.dirname(CODE_DIR)
DATASET_DIR = os.path.join(REPO_ROOT, "dataset")
EVAL_DIR = os.path.dirname(os.path.abspath(__file__))

EXPECTED_COLUMNS = [
    "claim_status",
    "issue_type",
    "severity",
    "evidence_standard_met",
]

SAFE_DEFAULTS = {
    "evidence_standard_met": "false",
    "evidence_standard_met_reason": "Processing error",
    "risk_flags": "none",
    "issue_type": "unknown",
    "object_part": "unknown",
    "claim_status": "not_enough_information",
    "claim_status_justification": "Processing error",
    "supporting_image_ids": "none",
    "valid_image": "false",
    "severity": "unknown",
}


def run_single_claim(claim: dict, image_mode: str = "all") -> dict:
    """
    Run the full pipeline for one claim.
    image_mode: "all" → send all images; "first" → send only first image.
    Returns output dict with predictions.
    """
    user_id = claim.get("user_id", "")
    claim_text = claim.get("user_claim", "")
    claim_object = claim.get("claim_object", "").strip().lower()
    image_paths_str = claim.get("image_paths", "")
    user_history = claim.get("_user_history", {})
    evidence_requirements = claim.get("_evidence_requirements", [])

    try:
        images = load_all_images(image_paths_str, base_dir=DATASET_DIR)
        extra_risk_flags = assess_risk(user_history)
        valid_images = [img for img in images if img["valid"]]

        if image_mode == "first" and valid_images:
            valid_images = valid_images[:1]

        vlm_result = analyze_claim(
            claim_text=claim_text,
            claim_object=claim_object,
            images=valid_images,
            evidence_requirements=evidence_requirements,
            extra_risk_flags=extra_risk_flags,
        )

        # Merge risk flags
        existing = vlm_result.get("risk_flags", [])
        if isinstance(existing, str):
            existing = [f.strip() for f in existing.split(";") if f.strip() and f.strip() != "none"]
        vlm_result["risk_flags"] = list(dict.fromkeys(existing + extra_risk_flags))

        # --- Post-processing risk flag rules ---
        rf_list = vlm_result["risk_flags"]
        rf_set = {str(f).strip().lower() for f in rf_list}
        
        if "wrong_object" in rf_set:
            vlm_result["claim_status"] = "contradicted"
            
        if "damage_not_visible" in rf_set and ("wrong_angle" in rf_set or "cropped_or_obstructed" in rf_set):
            vlm_result["claim_status"] = "not_enough_information"
        # ---------------------------------------

        fixed = validate_and_fix(vlm_result, claim_object)
        return {
            "user_id": user_id,
            "claim_object": claim_object,
            "claim_status": fixed.get("claim_status", "not_enough_information"),
            "issue_type": fixed.get("issue_type", "unknown"),
            "severity": fixed.get("severity", "unknown"),
            "evidence_standard_met": fixed.get("evidence_standard_met", "false"),
            "claim_status_justification": fixed.get("claim_status_justification", ""),
            "risk_flags": fixed.get("risk_flags", "none"),
            "object_part": fixed.get("object_part", "unknown"),
        }

    except Exception as e:
        console.print(f"[red][eval] Error on claim {user_id}: {e}[/red]")
        return {
            "user_id": user_id,
            "claim_object": claim_object,
            **SAFE_DEFAULTS,
        }


def run_strategy(claims: list, image_mode: str, label: str) -> tuple:
    """
    Run all claims under a given image strategy sequentially.
    Returns (predictions_list, elapsed_seconds).
    A 2s delay is inserted between claims to avoid rate limits.
    """
    console.print(f"\n[bold cyan]Running {label} ({image_mode} image mode)...[/bold cyan]")
    predictions = []
    start = time.time()
    total = len(claims)

    for i, claim in enumerate(claims):
        result = run_single_claim(claim, image_mode)
        predictions.append(result)
        completed = i + 1
        console.print(
            f"  [{completed}/{total}] user={result.get('user_id','')} "
            f"status={result.get('claim_status','')}",
            highlight=False,
        )
        # Delay between claims to avoid rate limits during evaluation
        if completed < total:
            time.sleep(2.0)

    elapsed = time.time() - start
    return predictions, elapsed


def compute_metrics(predictions: list, ground_truth: list) -> dict:
    """
    Compute exact-match accuracy for each tracked field.
    Returns dict of {field: accuracy_pct}.
    """
    metrics = {col: {"correct": 0, "total": 0} for col in EXPECTED_COLUMNS}

    for pred, gt in zip(predictions, ground_truth):
        for col in EXPECTED_COLUMNS:
            pred_val = str(pred.get(col, "")).strip().lower()
            gt_val = str(gt.get(col, "")).strip().lower()
            if gt_val:  # only score when ground truth exists
                metrics[col]["total"] += 1
                if pred_val == gt_val:
                    metrics[col]["correct"] += 1

    result = {}
    for col, counts in metrics.items():
        if counts["total"] > 0:
            result[col] = round(100.0 * counts["correct"] / counts["total"], 1)
        else:
            result[col] = 0.0
    return result


def analyze_errors(predictions: list, ground_truth: list):
    cm = {
        "supported": {"supported": 0, "contradicted": 0, "not_enough_information": 0},
        "contradicted": {"supported": 0, "contradicted": 0, "not_enough_information": 0},
        "not_enough_information": {"supported": 0, "contradicted": 0, "not_enough_information": 0},
    }
    wrong_preds = []
    error_counts = {
        "claim_status": 0,
        "issue_type": 0,
        "severity": 0,
        "evidence_standard_met": 0
    }
    
    for p, gt in zip(predictions, ground_truth):
        user_id = p.get("user_id")
        claim_object = p.get("claim_object", "")
        
        expected_cs = gt.get("claim_status", "")
        pred_cs = p.get("claim_status", "")
        
        if expected_cs in cm and pred_cs in cm[expected_cs]:
            cm[expected_cs][pred_cs] += 1
            
        is_wrong = False
        row_wrong = {"user_id": user_id, "claim_object": claim_object}
        
        for field in EXPECTED_COLUMNS:
            e_val = str(gt.get(field, "")).strip().lower()
            p_val = str(p.get(field, "")).strip().lower()
            row_wrong[f"expected_{field}"] = e_val
            row_wrong[f"predicted_{field}"] = p_val
            if e_val and e_val != p_val:
                is_wrong = True
                error_counts[field] += 1
                
        if is_wrong:
            wrong_preds.append(row_wrong)
            
    return cm, wrong_preds, error_counts


def print_metrics_table(title: str, metrics: dict):
    table = Table(title=title, show_header=True, header_style="bold bright_blue", border_style="dim")
    table.add_column("Metric", style="bold")
    table.add_column("Accuracy (%)", justify="right")

    for field, acc in metrics.items():
        color = "green" if acc >= 70 else ("yellow" if acc >= 50 else "red")
        table.add_row(field, f"[{color}]{acc:.1f}%[/{color}]")
    console.print(table)


def print_comparison_table(metrics_a: dict, metrics_b: dict):
    table = Table(
        title="Strategy A (first image) vs Strategy B (all images)",
        show_header=True,
        header_style="bold bright_blue",
        border_style="dim",
    )
    table.add_column("Metric", style="bold")
    table.add_column("Strategy A (%)", justify="right")
    table.add_column("Strategy B (%)", justify="right")
    table.add_column("Winner", justify="center")

    for field in EXPECTED_COLUMNS:
        a = metrics_a.get(field, 0.0)
        b = metrics_b.get(field, 0.0)
        if b > a:
            winner = "[green]B[/green]"
        elif a > b:
            winner = "[cyan]A[/cyan]"
        else:
            winner = "[dim]Tie[/dim]"
        table.add_row(field, f"{a:.1f}%", f"{b:.1f}%", winner)
    console.print(table)


def choose_strategy(metrics_a: dict, metrics_b: dict) -> str:
    """Return 'A' or 'B' based on claim_status accuracy (primary signal)."""
    a_cs = metrics_a.get("claim_status", 0)
    b_cs = metrics_b.get("claim_status", 0)
    return "B" if b_cs >= a_cs else "A"


def write_report(
    metrics_a: dict,
    metrics_b: dict,
    chosen: str,
    chosen_metrics: dict,
    total_claims: int,
    total_images: int,
    elapsed_a: float,
    elapsed_b: float,
    api_calls_total: int,
    cm: dict,
    error_counts: dict,
):
    report_path = os.path.join(EVAL_DIR, "evaluation_report.md")

    def fmt_table(metrics: dict) -> str:
        lines = ["| Metric | Accuracy (%) |", "|--------|-------------|"]
        for k, v in metrics.items():
            lines.append(f"| {k} | {v:.1f}% |")
        return "\n".join(lines)

    def fmt_comparison(ma: dict, mb: dict) -> str:
        lines = ["| Metric | Strategy A (%) | Strategy B (%) | Winner |",
                 "|--------|---------------|---------------|--------|"]
        for field in EXPECTED_COLUMNS:
            a = ma.get(field, 0.0)
            b = mb.get(field, 0.0)
            winner = "B" if b > a else ("A" if a > b else "Tie")
            lines.append(f"| {field} | {a:.1f}% | {b:.1f}% | {winner} |")
        return "\n".join(lines)

    def fmt_cm(cm_dict: dict) -> str:
        lines = [
            "| Expected \\ Predicted | supported | contradicted | not_enough_information |",
            "|----------------------|-----------|--------------|------------------------|"
        ]
        for exp in ["supported", "contradicted", "not_enough_information"]:
            lines.append(f"| **{exp}** | {cm_dict[exp]['supported']} | {cm_dict[exp]['contradicted']} | {cm_dict[exp]['not_enough_information']} |")
        return "\n".join(lines)

    def fmt_errors(ec: dict) -> str:
        lines = []
        for k, v in sorted(ec.items(), key=lambda x: x[1], reverse=True):
            lines.append(f"- wrong {k}: {v}")
        return "\n".join(lines)

    report = f"""# Evaluation Report — Damage Claim Verification System

Generated: {time.strftime("%Y-%m-%dT%H:%M:%S")}

## Chosen Strategy: Strategy {chosen}

### Metrics for Strategy {chosen}

{fmt_table(chosen_metrics)}

### Error Analysis (Chosen Strategy)

#### Confusion Matrix (claim_status)
{fmt_cm(cm)}

#### Top Error Categories
{fmt_errors(error_counts)}

---

## Strategy A vs Strategy B Comparison

**Strategy A:** Send only the first image per claim to the VLM.  
**Strategy B:** Send all images per claim to the VLM.

{fmt_comparison(metrics_a, metrics_b)}

### Which Strategy Was Chosen and Why

Strategy **{chosen}** was selected based on higher `claim_status` accuracy,
which is the primary evaluation signal. Sending {"all images provides richer visual context" if chosen == "B" else "only the first image reduces token usage while matching or exceeding accuracy"}.

---

## Operational Analysis

| Metric | Value |
|--------|-------|
| Total claims evaluated | {total_claims} |
| Total images processed | {total_images} |
| Total API calls made | {api_calls_total} |
| Strategy A runtime | {elapsed_a:.1f}s |
| Strategy B runtime | {elapsed_b:.1f}s |
| Combined runtime | {elapsed_a + elapsed_b:.1f}s |
| Estimated cost | **Free** (llama-3.2-90b-vision-preview on Groq free tier) |

### Rate Limit Handling

- **Inter-call delay:** 0.5 seconds between every API call (`time.sleep(0.5)` in `vlm.py`)
- **Max workers:** 3 concurrent threads (`ThreadPoolExecutor(max_workers=3)`)
- **Retry policy:** Up to 3 retries on primary model, 1 attempt on fallback model
- **Fallback model:** `llama-3.2-11b-vision-preview` used if primary fails all retries

### Model Information

- **Primary:** `llama-3.2-90b-vision-preview` (Groq, free tier)
- **Fallback:** `llama-3.2-11b-vision-preview` (Groq, free tier)
- **Provider:** [Groq](https://console.groq.com) — no cost on free tier
"""

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    console.print(f"\n[bold green]✓ Evaluation report written to:[/bold green] {report_path}")


def main():
    console.print(
        Panel(
            "[bold cyan]Damage Claim Evaluation Pipeline[/bold cyan]\n"
            "[dim]Comparing Strategy A vs B on sample_claims.csv[/dim]",
            border_style="bright_blue",
            padding=(1, 4),
        )
    )

    sample_path = os.path.join(DATASET_DIR, "sample_claims.csv")
    user_history_path = os.path.join(DATASET_DIR, "user_history.csv")
    evidence_req_path = os.path.join(DATASET_DIR, "evidence_requirements.csv")

    console.print("[bold]Loading sample data...[/bold]")
    claims = load_claims(sample_path)
    user_history = load_user_history(user_history_path)
    evidence_requirements = load_evidence_requirements(evidence_req_path)
    claims = enrich_claims(claims, user_history, evidence_requirements)

    # Extract ground truth from sample CSV (before enrichment mutates the dicts)
    ground_truth = []
    for claim in claims:
        ground_truth.append({
            "claim_status": str(claim.get("claim_status", "")).strip().lower(),
            "issue_type": str(claim.get("issue_type", "")).strip().lower(),
            "severity": str(claim.get("severity", "")).strip().lower(),
            "evidence_standard_met": str(claim.get("evidence_standard_met", "")).strip().lower(),
        })

    total_claims = len(claims)
    console.print(f"[bold]{total_claims} sample claims loaded.[/bold]\n")

    # Count total images
    total_images = 0
    for claim in claims:
        paths_str = claim.get("image_paths", "")
        if paths_str:
            total_images += len([p for p in paths_str.split(";") if p.strip()])

    # --- Strategy A: first image only ---
    preds_a, elapsed_a = run_strategy(claims, "first", "Strategy A (first image)")
    metrics_a = compute_metrics(preds_a, ground_truth)

    # --- Strategy B: all images ---
    preds_b, elapsed_b = run_strategy(claims, "all", "Strategy B (all images)")
    metrics_b = compute_metrics(preds_b, ground_truth)

    # Print metrics tables
    console.print("\n")
    print_metrics_table("Strategy A Metrics (first image only)", metrics_a)
    console.print()
    print_metrics_table("Strategy B Metrics (all images)", metrics_b)
    console.print()
    print_comparison_table(metrics_a, metrics_b)

    # Choose winner
    chosen = choose_strategy(metrics_a, metrics_b)
    chosen_metrics = metrics_b if chosen == "B" else metrics_a
    console.print(
        f"\n[bold green]Chosen Strategy: {chosen}[/bold green] "
        f"(claim_status accuracy: {chosen_metrics.get('claim_status', 0):.1f}%)"
    )

    # Print VLM stats
    print_stats()

    # Get total API call count from vlm module
    import vlm as vlm_module
    api_calls_total = vlm_module._total_api_calls

    # Error analysis on chosen strategy
    chosen_preds = preds_b if chosen == "B" else preds_a
    cm, wrong_preds, error_counts = analyze_errors(chosen_preds, ground_truth)

    # Print error analysis
    console.print("\n[bold magenta]=== Error Analysis ===[/bold magenta]")
    console.print("[bold]Confusion Matrix (claim_status)[/bold]")
    console.print("Expected -> Predicted\n")
    for expected in ["supported", "contradicted", "not_enough_information"]:
        for predicted in ["supported", "contradicted", "not_enough_information"]:
            console.print(f"{expected} -> {predicted}: {cm[expected][predicted]}")
        console.print()
            
    console.print("[bold]Top error categories[/bold]")
    for k, v in sorted(error_counts.items(), key=lambda x: x[1], reverse=True):
        console.print(f"- wrong {k}: {v}")
    console.print()

    # Save wrong predictions
    wp_path = os.path.join(EVAL_DIR, "wrong_predictions.csv")
    with open(wp_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["user_id", "claim_object"] + \
            [f"expected_{c}" for c in EXPECTED_COLUMNS] + \
            [f"predicted_{c}" for c in EXPECTED_COLUMNS]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(wrong_preds)
    console.print(f"[bold green]✓ Wrong predictions CSV written to:[/bold green] {wp_path}")

    # Write report
    write_report(
        metrics_a=metrics_a,
        metrics_b=metrics_b,
        chosen=chosen,
        chosen_metrics=chosen_metrics,
        total_claims=total_claims,
        total_images=total_images,
        elapsed_a=elapsed_a,
        elapsed_b=elapsed_b,
        api_calls_total=api_calls_total,
        cm=cm,
        error_counts=error_counts,
    )

    console.print(
        Panel(
            f"[bold]Evaluation complete![/bold]\n"
            f"Strategy {chosen} selected with "
            f"claim_status accuracy: [green]{chosen_metrics.get('claim_status', 0):.1f}%[/green]",
            border_style="bright_green",
            padding=(1, 4),
        )
    )


if __name__ == "__main__":
    main()
