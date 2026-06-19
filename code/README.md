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

## Architecture

```
code/
├── main.py              # Entry point — orchestrates the full pipeline
├── loader.py            # CSV loading + claim enrichment
├── image_utils.py       # PIL image loading, base64 encoding
├── risk.py              # Pure-logic risk flag assessment from user history
├── validator.py         # Validates & fixes VLM output to allowed values
├── vlm.py               # Groq vision model integration (primary + fallback)
└── evaluation/
    └── main.py          # Strategy A vs B evaluation on sample claims
```

### Pipeline flow (per claim)

1. Load images → base64 via Pillow
2. Assess risk flags from user history (no LLM)
3. Send images + claim text to Groq VLaMA Vision
4. Merge risk flags, validate all output values
5. Write row to `output.csv`

### Models used

| Model | Role |
|-------|------|
| `llama-3.2-90b-vision-preview` | Primary (up to 3 retries) |
| `llama-3.2-11b-vision-preview` | Fallback (1 attempt) |

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
- `ThreadPoolExecutor(max_workers=3)` for concurrency
- Groq free tier — no cost incurred

---

## Environment variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GROQ_API_KEY` | ✅ Yes | Your Groq API key |
