# Damage Claim Verification System

Multimodal evidence review pipeline for the HackerRank Orchestrate June 2026 hackathon.

Verifies damage claims by analyzing submitted images against claim conversations using
the Groq LLaMA Vision API.

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set your Groq API key

```bash
export GROQ_API_KEY="your_key_here"
```

Get your key at [console.groq.com](https://console.groq.com).

---

## Running

### Run on sample data (development / evaluation)

```bash
python code/main.py --sample
```

Outputs: `output_sample.csv`

### Run on full test data

```bash
python code/main.py
```

Outputs: `output.csv`

### Run on first N claims only (quick test)

```bash
python code/main.py --tickets 5
```

### Run evaluation (Strategy A vs B comparison)

```bash
python code/evaluation/main.py
```

Outputs: `code/evaluation/evaluation_report.md`

---

## Architecture Diagram

`Loader` → `Risk Engine` → `Image Processing` → `Cache` → `VLM` → `Validator` → `Output`

## Architecture

```
code/
├── main.py              # Entry point — orchestrates the full pipeline
├── loader.py            # CSV loading + claim enrichment
├── image_utils.py       # PIL image loading, base64 encoding
├── risk.py              # Pure-logic risk assessment + prompt injection detection
├── validator.py         # Validates VLM output & applies consistency rules
├── vlm.py               # Groq vision model integration (primary + fallback)
└── evaluation/
    └── main.py          # Strategy eval, confusion matrix & error analysis
```

### Pipeline flow (per claim)

1. **Loader**: Load claims and enrich with user history and evidence requirements.
2. **Risk Engine**: Assess risk flags from user history + prompt injection detection (no LLM).
3. **Image Processing**: Extract images via Pillow and perform lightweight image quality checks (blur, lighting).
4. **Cache**: Check file-based JSON cache to skip redundant VLM calls.
5. **VLM**: Send images, claim text, and object-specific prompts to Groq LLaMA Vision.
6. **Validator**: Validate values, apply auto-correction consistency rules, and compute confidence score.
7. **Output**: Write row to `output.csv` (fallback row generated on failure to guarantee 1:1 mapping).

### Key Design Decisions

- **Why images are primary evidence**: Text claims can be easily manipulated or hallucinated. Visual evidence is the ground truth. If the damage isn't clearly visible in the image, the claim is not fully supported, even if the user provides a detailed text description.
- **Why deterministic validation exists**: Certain checks (like minimum required images or prompt injection detection) are straightforward boolean logic. Handling these deterministically before or alongside the VLM is faster, avoids hallucinations, and reduces prompt complexity.
- **Why caching was added**: Caching avoids redundant API calls for duplicate claims and heavily retried identical images. This speeds up processing, reduces API costs/rate limits, and makes the evaluation pipeline faster.
- **How confidence works**: An internal confidence score is calculated based on image quality, the number of supporting images, and the clarity of the VLM's decision. If confidence is below 0.5, a `manual_review_required` risk flag is automatically added, ensuring uncertain claims are escalated to human reviewers.

### Models used

| Model | Role |
|-------|------|
| `meta-llama/llama-4-scout-17b-16e-instruct` | Primary (up to 3 retries) |
| `qwen/qwen3.6-27b` | Fallback (1 attempt) |

---

## Output format

`output.csv` has exactly one row per input claim with these columns (in order):

```
user_id, image_paths, user_claim, claim_object,
evidence_standard_met, evidence_standard_met_reason,
risk_flags, issue_type, object_part, claim_status,
claim_status_justification, supporting_image_ids,
valid_image, severity
```

---

## Rate limiting & concurrency

- `time.sleep(0.5)` between every API call
- `ThreadPoolExecutor(max_workers=1)` for sequential processing to avoid simultaneous rate limit exhaustion
- Groq free tier — no cost incurred

---

## Environment variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GROQ_API_KEY` | ✅ Yes | Your Groq API key |
