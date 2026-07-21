# Description

Builds a site to benchmark human reasoning against LLMs using a Rasch psychometric model. 

---

# Requirements

- **Python 3.9+** 

| Package | Needed by | Why |
|---|---|---|
| `certifi` | `build_item_bank.py`, `score_openrouter.py` | A working CA bundle for HTTPS. Fixes the common macOS "CERTIFICATE_VERIFY_FAILED" error. |
| `datasets` | `build_item_bank.py` | Only to fetch CommonsenseQA from HuggingFace. Avoidable — see `--csqa-local`. |

```bash
pip install certifi datasets
```
- An **OpenRouter API key** 
```bash
export OPENROUTER_API_KEY=your_key_xxx
```
---

# Reproducing the calibration

### 1. Build the item bank

```bash
python build_item_bank.py --seed 7 --out item_bank.json
```
Pulls from GSM8K and BIG-Bench Hard (GitHub) and CommonsenseQA (HuggingFace).
Optional extra sources:

### 2. Score the models

```bash

python score_openrouter.py --bank item_bank.json --models-file models.txt \
    --out results.json --csv matrix.csv
```
Or score local models with LMStudio using ```--local```

```bash
python score_openrouter.py --bank item_bank.json --models "the-lmstudio-id" \
    --local --out results.json --csv matrix.csv
```

The models included on the sample website cost about $5 to test

### 3. Build the matrix

```bash
python make_matrix.py --results results.json --bank item_bank.json --grading strict
```
### 4. Fit the Rasch model

```bash
python rasch_analysis.py --matrix matrix_strict.csv --bank item_bank.json \
    --out rasch_calibration.json
```

### 5. Build the site

```bash
python build_site.py --bank item_bank.json --calibration rasch_calibration.json \
    --results results.json --variety 0.5
```

Produces a self-contained `index.html`. Flags:

- `--variety 0..1` — repeat-play variation vs. measurement precision
  (0 = always the most informative item; higher = more varied).

Only items with a finite Rasch difficulty are included, so dropped or
all-correct items are omitted automatically.

---

