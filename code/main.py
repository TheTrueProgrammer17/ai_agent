"""
main.py — Entry point for the damage claim verification pipeline.
Uses rich for terminal UI, ThreadPoolExecutor for concurrency.
"""

import argparse
import csv
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

from rich.columns import Columns
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TimeElapsedColumn,
)
from rich.table import Table
from rich.text import Text

# Add code/ directory to path so sibling imports work when run from repo root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from image_utils import load_all_images
from loader import enrich_claims, load_claims, load_evidence_requirements, load_user_history
from risk import assess_risk
from validator import validate_and_fix
from vlm import analyze_claim, print_stats

console = Console()

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(REPO_ROOT, "dataset")

OUTPUT_COLUMNS = [
    "user_id", "image_paths", "user_claim", "claim_object",
    "evidence_standard_met", "evidence_standard_met_reason",
    "risk_flags", "issue_type", "object_part", "claim_status",
    "claim_status_justification", "supporting_image_ids",
    "valid_image", "severity",
]

SAFE_DEFAULTS = {
    "evidence_standard_met": "false",
    "evidence_standard_met_reason": "Processing error",
    "risk_flags": "none",
    "issue_type": "unknown",
    "object_part": "unknown",
    "claim_status": "not_enough_information",
    "claim_status_justification": "Processing error — no analysis available",
    "supporting_image_ids": "none",
    "valid_image": "false",
    "severity": "unknown",
}


def print_banner():
    banner = Panel(
        "[bold cyan]HackerRank Orchestrate — Damage Claim Verification System[/bold cyan]\n"
        "[dim]Multimodal evidence review powered by Groq LLaMA Vision[/dim]",
        border_style="bright_blue",
        padding=(1, 4),
    )
    console.print(banner)
    console.print()


def status_color(claim_status: str) -> Text:
    colors = {
        "supported": "bold green",
        "contradicted": "bold red",
        "not_enough_information": "bold yellow",
    }
    style = colors.get(claim_status, "white")
    return Text(claim_status, style=style)


def deduplicate_claims(claims: list) -> list:
    """Remove duplicate claims by exact user_claim string match. Keep first occurrence."""
    seen = set()
    unique = []
    for claim in claims:
        key = claim.get("user_claim", "").strip()
        if key not in seen:
            seen.add(key)
            unique.append(claim)
        else:
            console.print(f"[dim yellow][dedup] Skipping duplicate claim for user {claim.get('user_id')}[/dim yellow]")
    return unique


def process_claim(claim: dict, idx: int) -> dict:
    """Process a single claim end-to-end. Returns output row dict."""
    user_id = claim.get("user_id", "")
    claim_text = claim.get("user_claim", "")
    claim_object = claim.get("claim_object", "").strip().lower()
    image_paths_str = claim.get("image_paths", "")
    user_history = claim.get("_user_history", {})
    evidence_requirements = claim.get("_evidence_requirements", [])

    try:
        # Load images
        images = load_all_images(image_paths_str, base_dir=DATASET_DIR)

        # Assess risk from user history and claim text
        extra_risk_flags = assess_risk(user_history, user_claim=claim_text)

        # Only send valid images to VLM
        valid_images = [img for img in images if img["valid"]]

        # Extract image quality flags
        quality_flags = []
        for img in valid_images:
            quality_flags.extend(img.get("flags", []))
        extra_risk_flags.extend(quality_flags)

        # Check evidence requirements deterministically
        evidence_met_pre_vlm = None
        evidence_reason_pre_vlm = None
        if evidence_requirements:
            for req in evidence_requirements:
                if "minimum_images" in req:
                    try:
                        min_imgs = int(req["minimum_images"])
                        if len(valid_images) < min_imgs:
                            evidence_met_pre_vlm = "false"
                            evidence_reason_pre_vlm = f"Requires at least {min_imgs} images, only provided {len(valid_images)} valid image(s)."
                    except ValueError:
                        pass

        # Analyze with VLM
        vlm_result = analyze_claim(
            claim_text=claim_text,
            claim_object=claim_object,
            images=valid_images,
            evidence_requirements=evidence_requirements,
            extra_risk_flags=extra_risk_flags,
        )

        # Override if deterministic rule failed
        if evidence_met_pre_vlm == "false":
            vlm_result["evidence_standard_met"] = "false"
            if "evidence_standard_met_reason" not in vlm_result or not vlm_result["evidence_standard_met_reason"]:
                vlm_result["evidence_standard_met_reason"] = evidence_reason_pre_vlm

        # Merge extra_risk_flags into result risk_flags
        existing_flags = vlm_result.get("risk_flags", [])
        if isinstance(existing_flags, str):
            existing_flags = [f.strip() for f in existing_flags.split(";") if f.strip() and f.strip() != "none"]
        merged_flags = list(dict.fromkeys(existing_flags + extra_risk_flags))
        vlm_result["risk_flags"] = merged_flags

        # If any image was invalid, note it
        if any(not img["valid"] for img in images):
            vlm_result.setdefault("risk_flags", [])
            if isinstance(vlm_result["risk_flags"], list):
                vlm_result["risk_flags"].append("damage_not_visible")

        # Validate and fix all values
        fixed = validate_and_fix(vlm_result, claim_object)

        # Build output row
        row = {
            "user_id": user_id,
            "image_paths": image_paths_str,
            "user_claim": claim_text,
            "claim_object": claim_object,
            "evidence_standard_met": fixed.get("evidence_standard_met", "false"),
            "evidence_standard_met_reason": fixed.get("evidence_standard_met_reason", ""),
            "risk_flags": fixed.get("risk_flags", "none"),
            "issue_type": fixed.get("issue_type", "unknown"),
            "object_part": fixed.get("object_part", "unknown"),
            "claim_status": fixed.get("claim_status", "not_enough_information"),
            "claim_status_justification": fixed.get("claim_status_justification", ""),
            "supporting_image_ids": fixed.get("supporting_image_ids", "none"),
            "valid_image": fixed.get("valid_image", "false"),
            "severity": fixed.get("severity", "unknown"),
        }
        return row

    except Exception as e:
        console.print(f"[bold red][error] Claim {idx} ({user_id}) failed: {e}[/bold red]")
        return {
            "user_id": user_id,
            "image_paths": image_paths_str,
            "user_claim": claim_text,
            "claim_object": claim_object,
            **SAFE_DEFAULTS,
        }


def build_live_table(recent_rows: list) -> Table:
    table = Table(
        show_header=True,
        header_style="bold bright_blue",
        border_style="dim",
        expand=True,
    )
    table.add_column("#", style="dim", width=4)
    table.add_column("User ID", width=10)
    table.add_column("Object", width=10)
    table.add_column("Status", width=26)
    table.add_column("Severity", width=8)
    table.add_column("Risk Flags", overflow="fold")

    for r in recent_rows[-10:]:
        cs = r.get("claim_status", "")
        table.add_row(
            str(r.get("_idx", "")),
            r.get("user_id", ""),
            r.get("claim_object", ""),
            status_color(cs),
            r.get("severity", ""),
            r.get("risk_flags", "none"),
        )
    return table


def write_output_csv(rows: list, output_path: str):
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    console.print(f"\n[bold green]✓ Output written to:[/bold green] {output_path}")


def main():
    print_banner()

    parser = argparse.ArgumentParser(description="Damage Claim Verification System")
    parser.add_argument("--sample", action="store_true", help="Run on sample_claims.csv")
    parser.add_argument("--tickets", type=int, default=None, metavar="N", help="Run on first N claims only")
    args = parser.parse_args()

    # Resolve paths
    if args.sample:
        claims_path = os.path.join(DATASET_DIR, "sample_claims.csv")
        output_path = os.path.join(REPO_ROOT, "output_sample.csv")
        console.print("[cyan]Mode: SAMPLE[/cyan] (sample_claims.csv)\n")
    else:
        claims_path = os.path.join(DATASET_DIR, "claims.csv")
        output_path = os.path.join(REPO_ROOT, "output.csv")
        console.print("[cyan]Mode: FULL TEST[/cyan] (claims.csv)\n")

    user_history_path = os.path.join(DATASET_DIR, "user_history.csv")
    evidence_req_path = os.path.join(DATASET_DIR, "evidence_requirements.csv")

    # Load data
    console.print("[bold]Loading data...[/bold]")
    claims = load_claims(claims_path)
    user_history = load_user_history(user_history_path)
    evidence_requirements = load_evidence_requirements(evidence_req_path)
    claims = enrich_claims(claims, user_history, evidence_requirements)

    # Deduplicate
    before = len(claims)
    claims = deduplicate_claims(claims)
    after = len(claims)
    if before != after:
        console.print(f"[yellow]Deduplicated {before - after} duplicate claim(s).[/yellow]")

    # Limit tickets if requested
    if args.tickets is not None:
        claims = claims[: args.tickets]
        console.print(f"[yellow]Limiting to first {args.tickets} claim(s).[/yellow]")

    total = len(claims)
    console.print(f"[bold]Processing {total} claim(s)...[/bold]\n")

    results = [None] * total
    recent_rows = []
    start_time = time.time()

    progress = Progress(
        SpinnerColumn(),
        "[progress.description]{task.description}",
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
    )
    task_id = progress.add_task("[cyan]Analyzing claims...", total=total)

    with Live(console=console, refresh_per_second=4) as live:
        # max_workers=1: sequential processing prevents parallel API calls
        # burning through rate limits simultaneously
        with ThreadPoolExecutor(max_workers=1) as executor:
            future_to_idx = {
                executor.submit(process_claim, claim, i + 1): i
                for i, claim in enumerate(claims)
            }

            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    row = future.result()
                except Exception as e:
                    claim = claims[idx]
                    row = {
                        "user_id": claim.get("user_id", ""),
                        "image_paths": claim.get("image_paths", ""),
                        "user_claim": claim.get("user_claim", ""),
                        "claim_object": claim.get("claim_object", ""),
                        **SAFE_DEFAULTS,
                    }

                row["_idx"] = idx + 1
                results[idx] = row
                recent_rows.append(row)

                progress.advance(task_id)

                # Build combined live display
                table = build_live_table(recent_rows)
                live.update(
                    Panel(
                        Columns([progress, table]),
                        title="[bold cyan]Claim Verification Progress[/bold cyan]",
                        border_style="bright_blue",
                    )
                )

    elapsed = time.time() - start_time

    # Compute summary stats
    supported = sum(1 for r in results if r and r.get("claim_status") == "supported")
    contradicted = sum(1 for r in results if r and r.get("claim_status") == "contradicted")
    nei = sum(1 for r in results if r and r.get("claim_status") == "not_enough_information")
    with_flags = sum(
        1 for r in results if r and r.get("risk_flags", "none") not in ("none", "", None)
    )

    # Count fallback results (API rate limit exhausted)
    fallbacks = sum(
        1 for r in results
        if r and "API rate limit exceeded" in r.get("claim_status_justification", "")
    )

    summary = (
        f"[bold]Total Claims:[/bold] {total}\n"
        f"[bold green]Supported:[/bold green] {supported}\n"
        f"[bold red]Contradicted:[/bold red] {contradicted}\n"
        f"[bold yellow]Not Enough Info:[/bold yellow] {nei}\n"
        f"[bold magenta]With Risk Flags:[/bold magenta] {with_flags}\n"
        f"[bold]Runtime:[/bold] {elapsed:.1f}s"
    )
    console.print("\n")
    console.print(Panel(summary, title="[bold]Pipeline Summary[/bold]", border_style="bright_green", padding=(1, 4)))

    console.print("\n[bold cyan]Consistency Corrections Applied:[/bold cyan]")
    from validator import CORRECTION_COUNTS
    for k, v in CORRECTION_COUNTS.items():
        if v > 0:
            console.print(f"  {k}: {v}")

    # Warn if fallbacks occurred due to rate limits
    if fallbacks > 0:
        console.print(
            f"[bold red]⚠ WARNING: {fallbacks} claims used "
            f"fallback due to API rate limits. "
            f"Rerun when limits reset.[/bold red]"
        )

    # Write output
    clean_results = [{k: v for k, v in r.items() if not k.startswith("_")} for r in results if r]
    write_output_csv(clean_results, output_path)

    # Print VLM stats
    print_stats()


if __name__ == "__main__":
    main()
