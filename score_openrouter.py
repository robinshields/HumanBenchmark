#!/usr/bin/env python3
"""
Score an item bank against a set of OpenRouter models and build a model x item
correct/incorrect matrix for Rasch calibration.

Design goals (per project):
  * OpenRouter only (full model variety), OpenAI-compatible endpoint.
  * Incremental: appends to an existing results file; re-running a model
    OVERWRITES that model's row (so changing the item set just means re-running).
  * The binary matrix is the real deliverable; results.json is the source of
    truth (raw responses, per-item grade, per-model cost), matrix.csv is derived.
  * Cost tally on the one-time scoring run.

Only depends on the Python standard library.

Quick start:
    # Provide the key EITHER via a .env file (see .env.example) in the working
    # directory, OR by exporting it. A real env var overrides the .env file.
    export OPENROUTER_API_KEY=sk-or-...
    # smoke test: 3 cheap models, first 5 items only
    python score_openrouter.py --bank item_bank_50.json \
        --models "openai/gpt-4o-mini,meta-llama/llama-3.1-8b-instruct,google/gemini-flash-1.5" \
        --limit 5
    # full run
    python score_openrouter.py --bank item_bank_50.json --models-file models.txt

    # offline check with no API/key:
    python score_openrouter.py --bank item_bank_50.json --models "fake/a,fake/b" --dry-run
"""
import argparse, csv, hashlib, json, os, random, re, ssl, sys, time
import urllib.request, urllib.error

CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"
MODELS_URL = "https://openrouter.ai/api/v1/models"
SSL_CTX = None  # set in run(); a CA-verifying context (uses certifi if available)


def make_ssl_context(insecure=False):
    """Build a TLS context with a working CA bundle. Prefers certifi's bundle,
    which fixes the common macOS 'CERTIFICATE_VERIFY_FAILED' error where Python
    has no system CA certs. `insecure` disables verification (last resort)."""
    if insecure:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        return ctx, "verification DISABLED (insecure)"
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where()), f"certifi ({certifi.where()})"
    except ImportError:
        return ssl.create_default_context(), "system default CA store"


# ----------------------------- .env loading ---------------------------------
def load_dotenv(path):
    """Minimal .env loader (no dependency). KEY=VALUE per line; supports an
    optional 'export ' prefix and quoted values. Does NOT overwrite variables
    already set in the real environment, so `export FOO=...` still wins."""
    if not path or not os.path.exists(path):
        return
    for line in open(path):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):]
        if "=" not in line:
            continue
        key, val = line.split("=", 1)
        key, val = key.strip(), val.strip()
        if len(val) >= 2 and val[0] == val[-1] and val[0] in "\"'":
            val = val[1:-1]
        os.environ.setdefault(key, val)


# ----------------------------- item bank ------------------------------------
def load_bank(path):
    items = json.load(open(path))
    for it in items:
        assert "item_id" in it and "format" in it and "answer" in it, "bad item schema"
    return items


def bank_hash(items):
    key = json.dumps([[it["item_id"], it["answer"]] for it in items], sort_keys=True)
    return hashlib.sha256(key.encode()).hexdigest()[:16]


# ----------------------------- prompting ------------------------------------
SYS = ("You are taking a short reasoning quiz. Solve each question, then give "
       "your final answer as a JSON object on the LAST line, in exactly this form:\n"
       '{"answer": "X"}\n'
       "where X is the option letter for a multiple-choice question, or the number "
       "for a numeric question. Output nothing after that JSON object.")


def build_messages(item):
    if item["format"] == "mc":
        opts = "\n".join(f"({chr(65+i)}) {c}" for i, c in enumerate(item["choices"]))
        user = (f"{item['question']}\n\nOptions:\n{opts}\n\n"
                'Reply with the option letter, e.g. {"answer": "A"}')
    else:
        user = (f"{item['question']}\n\n"
                'Reply with a single number, e.g. {"answer": "42"}')
    return [{"role": "system", "content": SYS}, {"role": "user", "content": user}]


# ----------------------------- grading --------------------------------------
def extract_mc(text, n_options, choices=None):
    """Pull a choice letter out of messy model output.

    Models fail the requested format in many ways: markdown bold (**B**), a
    restated answer word instead of a letter, or a trailing sentence. Each
    fallback below is ordered from most to least reliable.
    """
    if not text:
        return None
    letters = {chr(65 + i) for i in range(n_options)}
    t = text.strip()

    # 1) explicit "ANSWER: X", tolerating markdown/brackets around it
    for cand in reversed(re.findall(
            r"answer\s*(?:is)?\s*[:=\-]?\s*[\*_`\[\(]{0,3}\s*([A-Za-z])\b", t, re.I)):
        if cand.upper() in letters:
            return cand.upper()
    # 2) bolded single letter, e.g. **C**
    for cand in reversed(re.findall(r"\*\*\s*\(?([A-Za-z])\)?\s*\*\*", t)):
        if cand.upper() in letters:
            return cand.upper()
    # 3) parenthesised letter, e.g. (C)
    for cand in reversed(re.findall(r"\(([A-Za-z])\)", t)):
        if cand.upper() in letters:
            return cand.upper()
    # 4) the final line is just a letter
    lines = [l.strip(" .*_`#-") for l in t.splitlines() if l.strip()]
    if lines and len(lines[-1]) == 1 and lines[-1].upper() in letters:
        return lines[-1].upper()
    # 5) the model restated the option TEXT rather than its letter
    if choices:
        tail = t[-400:].lower()
        hits = [(tail.rfind(str(c).strip().lower()), i)
                for i, c in enumerate(choices)
                if str(c).strip() and len(str(c).strip()) >= 4
                and str(c).strip().lower() in tail]
        if hits:
            return chr(65 + max(hits)[1])
    # 6) last standalone capital letter in range
    for cand in reversed(re.findall(r"\b([A-Z])\b", t)):
        if cand in letters:
            return cand
    return None


def extract_num(text):
    if not text:
        return None
    cand = None
    m = re.findall(r"answer\s*(?:is)?\s*[:=\-]?\s*[\*_`]{0,2}\s*\$?\s*(-?[\d,]+(?:\.\d+)?)",
                   text, re.I)
    if not m:
        m = re.findall(r"\\boxed\{\s*\$?(-?[\d,]+(?:\.\d+)?)", text)
    if m:
        cand = m[-1]
    else:
        nums = re.findall(r"-?\$?[\d,]+(?:\.\d+)?", text)
        cand = nums[-1] if nums else None
    if cand is None:
        return None
    c = cand.replace("$", "").replace(",", "").replace("%", "").strip()
    try:
        return float(c)
    except ValueError:
        return None


def parse_json_answer(text):
    """Pull {"answer": ...} out of the reply. Returns the raw answer string.

    Scans from the end so a JSON object in the model's reasoning does not
    beat the final one. Tolerates markdown fences around the object.
    """
    if not text:
        return None
    for m in reversed(list(re.finditer(r"\{[^{}]*\}", text))):
        blob = m.group(0)
        try:
            obj = json.loads(blob)
        except ValueError:
            continue
        for k in ("answer", "Answer", "ANSWER", "final_answer"):
            if k in obj and obj[k] is not None:
                return str(obj[k]).strip()
    return None


def grade(item, raw):
    """Grade a reply. Returns (correct, pred, tier).

    tier is how the answer was recovered:
      "strict"  - a well-formed JSON object, exactly as instructed
      "lenient" - recovered from prose by the fallback parsers
      "none"    - no answer could be recovered at all
    Keeping these separate lets format compliance be measured on its own,
    and lets the correct/incorrect policy be changed later without re-running.
    """
    js = parse_json_answer(raw)
    if js is not None:
        if item["format"] == "mc":
            js_s = js.strip()
            letter = None
            # only read it as a letter if it LOOKS like one ("B", "(B)", "B.")
            m = re.fullmatch(r"\(?([A-Za-z])\)?[.):]?", js_s)
            if m:
                letter = m.group(1).upper()
            elif item.get("choices"):
                low = [str(c).strip().lower() for c in item["choices"]]
                if js_s.lower() in low:
                    letter = chr(65 + low.index(js_s.lower()))
            if letter and ord(letter) - 65 < item["n_options"]:
                return int(letter == item["answer"]), letter, "strict"
        else:
            v = extract_num(js)
            if v is not None:
                ok = abs(v - float(item["answer"])) < 1e-6
                return int(ok), (str(int(v)) if v.is_integer() else str(v)), "strict"
    correct, pred = _grade_lenient(item, raw)
    return correct, pred, ("lenient" if pred else "none")


def _grade_lenient(item, raw):
    if item["format"] == "mc":
        pred = extract_mc(raw, item["n_options"], item.get("choices"))
        return int(pred == item["answer"]), (pred if pred else "")
    predf = extract_num(raw)
    goldf = float(item["answer"])
    ok = predf is not None and abs(predf - goldf) < 1e-6
    return int(ok), ("" if predf is None else (str(int(predf)) if predf.is_integer() else str(predf)))


# ----------------------------- chat I/O (OpenAI-compatible) -----------------
def _post(url, api_key, payload, timeout):
    data = json.dumps(payload).encode()
    headers = {"Content-Type": "application/json",
               "HTTP-Referer": "https://localhost/human-vs-llm",
               "X-Title": "human-vs-llm-bench"}
    if api_key:                      # omitted for local servers (LM Studio)
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(url, data=data, method="POST", headers=headers)
    with urllib.request.urlopen(req, timeout=timeout, context=SSL_CTX) as r:
        return json.loads(r.read().decode())


def call_model(url, api_key, model, messages, max_tokens, temperature, timeout,
               max_retries, json_mode=False):
    payload = {"model": model, "messages": messages, "max_tokens": max_tokens,
               "temperature": temperature, "usage": {"include": True}}
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    delay = 2.0
    for attempt in range(max_retries + 1):
        try:
            return _post(url, api_key, payload, timeout)
        except urllib.error.HTTPError as e:
            code = e.code
            body = e.read().decode(errors="ignore")[:200]
            if code in (408, 429, 500, 502, 503, 504) and attempt < max_retries:
                time.sleep(delay); delay = min(delay * 2, 30); continue
            raise RuntimeError(f"HTTP {code}: {body}")
        except (urllib.error.URLError, TimeoutError) as e:
            if attempt < max_retries:
                time.sleep(delay); delay = min(delay * 2, 30); continue
            raise RuntimeError(f"network error: {e}")
    raise RuntimeError("exhausted retries")


def fetch_pricing(api_key):
    try:
        req = urllib.request.Request(MODELS_URL,
                                     headers={"Authorization": f"Bearer {api_key}"})
        data = json.loads(urllib.request.urlopen(req, timeout=30, context=SSL_CTX).read().decode())
    except Exception as e:
        print(f"  (could not fetch pricing: {e}; cost will use response usage only)",
              file=sys.stderr)
        return {}
    price = {}
    for m in data.get("data", []):
        p = m.get("pricing", {}) or {}
        try:
            price[m["id"]] = {"in": float(p.get("prompt", 0) or 0),
                              "out": float(p.get("completion", 0) or 0)}
        except (TypeError, ValueError):
            pass
    return price


def response_cost(data, model, pricing):
    usage = data.get("usage", {}) or {}
    if usage.get("cost") is not None:      # OpenRouter opt-in cost
        try:
            return float(usage["cost"]), usage
        except (TypeError, ValueError):
            pass
    pt = usage.get("prompt_tokens", 0) or 0
    ct = usage.get("completion_tokens", 0) or 0
    pr = pricing.get(model)
    if pr:
        return pt * pr["in"] + ct * pr["out"], usage
    return 0.0, usage


# ----------------------------- dry-run responder ----------------------------
def dry_run_response(item, model, rng):
    """Fake an OpenRouter response for offline testing. ~70% right, some noise."""
    right = rng.random() < 0.7
    if item["format"] == "mc":
        pick = item["answer"] if right else rng.choice(
            [chr(65 + i) for i in range(item["n_options"]) if chr(65 + i) != item["answer"]])
        style = rng.random()          # mimic real-world format compliance
        if style < 0.75:  content = 'Reasoning...\n{"answer": "%s"}' % pick
        elif style < 0.95: content = "Reasoning... the answer is **%s**" % pick
        else:             content = "Long ramble with no conclusion at all"
    else:
        val = item["answer"] if right else str(int(float(item["answer"])) + rng.choice([-2, -1, 1, 3]))
        style = rng.random()
        if style < 0.75:  content = 'Let me compute.\n{"answer": "%s"}' % val
        elif style < 0.95: content = "Let me compute. Final answer: %s" % val
        else:             content = "I cannot determine this"
    return {"choices": [{"message": {"content": content}}],
            "usage": {"prompt_tokens": 120, "completion_tokens": 20, "cost": 0.0}}


# ----------------------------- persistence ----------------------------------
def keep_raw(raw, cap):
    """Store the reply for later inspection. Kept in full by default (cap=0):
    disk is cheap and the raw text is the only evidence for why an item was
    graded the way it was. A positive cap keeps the HEAD and the TAIL."""
    raw = raw or ""
    if cap <= 0 or len(raw) <= cap:
        return raw
    head = cap // 3
    return raw[:head] + f"\n...[{len(raw)-cap} chars omitted]...\n" + raw[-(cap - head):]


def atomic_write(path, obj):
    tmp = path + ".tmp"
    json.dump(obj, open(tmp, "w"), ensure_ascii=False, indent=1)
    os.replace(tmp, path)


def write_matrix(results, items, path, grading="lenient"):
    cur_hash = results["item_bank"]["hash"]
    item_ids = [it["item_id"] for it in items]
    rows, skipped = [], []
    for model, rec in results["models"].items():
        if rec.get("item_hash") != cur_hash:
            skipped.append(model); continue
        row = {"model": model}
        for iid in item_ids:
            r = rec["responses"].get(iid)
            if r is None:
                row[iid] = ""
            elif grading == "strict":
                # a reply that ignored the output format counts as incorrect
                row[iid] = r.get("correct_strict", r["correct"])
            else:
                row[iid] = r["correct"]
        row["score"] = sum(v for k, v in row.items()
                           if k != "model" and isinstance(v, int))
        row["n"] = rec["n_items"]
        rows.append(row)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model"] + item_ids + ["score", "n"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    if skipped:
        print(f"  note: {len(skipped)} model(s) excluded from matrix as stale "
              f"(run against a different item set): {', '.join(skipped)}", file=sys.stderr)
    return len(rows)


# ----------------------------- main run -------------------------------------
def score_item(item, model, chat_url, api_key, pricing, args, rng):
    """Score one item. Returns the response record dict (same shape stored in
    results.json). Raises nothing - API failures come back as parse='error'."""
    try:
        if args.dry_run:
            data = dry_run_response(item, model, rng)
        else:
            data = call_model(chat_url, api_key, model, build_messages(item),
                              args.max_tokens, args.temperature,
                              args.timeout, args.max_retries, args.json_mode)
        raw = (data["choices"][0]["message"].get("content") or "")
        correct, pred, tier = grade(item, raw)
        fin = (data["choices"][0].get("finish_reason") or "").lower()
        if (not pred) and fin in ("length", "max_tokens") and not args.dry_run:
            data = call_model(chat_url, api_key, model, build_messages(item),
                              args.max_tokens * 3, args.temperature,
                              args.timeout, args.max_retries, args.json_mode)
            raw = (data["choices"][0]["message"].get("content") or "")
            correct, pred, tier = grade(item, raw)
            fin = (data["choices"][0].get("finish_reason") or "").lower()
        cost, _ = response_cost(data, model, pricing)
        return {"correct": correct, "correct_strict": int(correct and tier == "strict"),
                "parse": tier, "pred": pred, "gold": item["answer"],
                "raw": keep_raw(raw, args.raw_chars),
                "cost": round(cost, 6), "err": None, "finish": fin}
    except Exception as e:
        return {"correct": 0, "correct_strict": 0, "parse": "error",
                "pred": "", "gold": item["answer"],
                "raw": "", "cost": 0.0, "err": str(e)[:180], "finish": ""}


def recount(rec):
    """Recompute a model record's tallies from its stored responses."""
    resp = rec.get("responses") or {}
    rec["n_correct"] = sum(r["correct"] for r in resp.values())
    rec["n_correct_strict"] = sum(r.get("correct_strict", 0) for r in resp.values())
    rec["n_strict"] = sum(1 for r in resp.values() if r.get("parse") == "strict")
    rec["n_lenient"] = sum(1 for r in resp.values() if r.get("parse") == "lenient")
    rec["n_unparsed"] = sum(1 for r in resp.values() if r.get("parse") == "none")
    rec["n_errors"] = sum(1 for r in resp.values() if r.get("parse") == "error")
    rec["cost_usd"] = sum(r.get("cost", 0) for r in resp.values())


def retry_errored_cells(results, items, models, chat_url, api_key, pricing, args, rng):
    """Re-score ONLY the items that previously failed with an API error, for the
    named models. Every successful answer is left exactly as it was."""
    by_id = {it["item_id"]: it for it in items}
    grand = 0.0
    for model in models:
        rec = results["models"].get(model)
        if not rec:
            print(f"= {model}: not in results file, skipping", file=sys.stderr); continue
        resp = rec.get("responses") or {}
        errored = [iid for iid, r in resp.items() if r.get("parse") == "error"]
        if not errored:
            print(f"= {model}: no API-errored items, nothing to retry")
            continue
        print(f"> {model}: retrying {len(errored)} errored item(s) ...")
        fixed = 0
        for k, iid in enumerate(errored, 1):
            item = by_id.get(iid)
            if not item:
                continue
            new = score_item(item, model, chat_url, api_key, pricing, args, rng)
            if new["parse"] != "error":
                fixed += 1
            resp[iid] = new
            print(f"    {iid}: {'recovered -> ' + (new['pred'] or 'unparsed') if new['parse'] != 'error' else 'still errored'}",
                  flush=True)
            atomic_write(args.out, results)
            if args.sleep:
                time.sleep(args.sleep)
        recount(rec)
        rec["score"] = round((rec["n_correct_strict"] if args.grading == "strict"
                              else rec["n_correct"]) / rec["n_items"], 4)
        grand += rec["cost_usd"]
        print(f"  {model}: recovered {fixed}/{len(errored)}, "
              f"{rec['n_errors']} still errored, score now "
              f"{rec['n_correct_strict'] if args.grading=='strict' else rec['n_correct']}"
              f"/{rec['n_items']}")
        atomic_write(args.out, results)
    print("\nretry complete.")



def run(args):
    global SSL_CTX
    load_dotenv(args.env_file)
    SSL_CTX, ca_desc = make_ssl_context(args.insecure)
    print(f"TLS: {ca_desc}", file=sys.stderr)
    items = load_bank(args.bank)
    if args.limit:
        items = items[: args.limit]
    cur_hash = bank_hash(items)
    item_ids = [it["item_id"] for it in items]

    # load or init results
    if os.path.exists(args.out):
        results = json.load(open(args.out))
    else:
        results = {"item_bank": {}, "models": {}}
    prev_hash = results.get("item_bank", {}).get("hash")
    if prev_hash and prev_hash != cur_hash:
        print(f"! item bank changed ({prev_hash} -> {cur_hash}). Existing model rows "
              f"run on the old set will be treated as stale until re-run.", file=sys.stderr)
    results["item_bank"] = {"hash": cur_hash, "n": len(items), "item_ids": item_ids}
    results.setdefault("models", {})

    models = [m.strip() for m in (args.models.split(",") if args.models else [])]
    if args.models_file:
        models += [l.strip() for l in open(args.models_file) if l.strip()
                   and not l.strip().startswith("#")]
    models = list(dict.fromkeys(models))  # dedupe, keep order
    if not models:
        sys.exit("no models given (use --models or --models-file)")

    LOCAL_URL = "http://localhost:1234/v1/chat/completions"
    chat_url = LOCAL_URL if args.local else CHAT_URL
    api_key = None if args.local else (args.api_key or os.environ.get("OPENROUTER_API_KEY"))
    if not api_key and not args.dry_run and not args.local:
        sys.exit("set OPENROUTER_API_KEY or pass --api-key (or use --local / --dry-run)")
    if args.local:
        print(f"local mode: {chat_url} (no auth, cost=$0)", file=sys.stderr)

    pricing = {} if (args.dry_run or args.local) else fetch_pricing(api_key)
    rng = random.Random(0)

    if args.retry_errors:
        retry_errored_cells(results, items, models, chat_url, api_key, pricing, args, rng)
        n = write_matrix(results, items, args.csv, args.grading)
        print(f"rewrote {args.csv} ({n} models x {len(items)} items)")
        return

    grand_cost = sum(m.get("cost_usd", 0) for m in results["models"].values()
                     if m.get("item_hash") == cur_hash)

    for model in models:
        existing = results["models"].get(model)
        if args.skip_existing and existing and existing.get("item_hash") == cur_hash:
            print(f"= {model}: already scored ({existing['n_correct']}/{existing['n_items']}), skipping")
            continue
        print(f"> {model}: scoring {len(items)} items ...")
        rec = {"run_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "item_hash": cur_hash,
               "n_items": len(items), "n_correct": 0, "cost_usd": 0.0,
               "n_errors": 0, "responses": {}}
        results["models"][model] = rec  # overwrite immediately

        for k, item in enumerate(items, 1):
            try:
                if args.dry_run:
                    data = dry_run_response(item, model, rng)
                else:
                    data = call_model(chat_url, api_key, model, build_messages(item),
                                      args.max_tokens, args.temperature,
                                      args.timeout, args.max_retries, args.json_mode)
                raw = (data["choices"][0]["message"].get("content") or "")
                correct, pred, tier = grade(item, raw)
                # A model that hit the token ceiling never reached its final
                # "ANSWER:" line, so a blank prediction here is a budget
                # problem, not a wrong answer. Retry once with more room.
                fin = (data["choices"][0].get("finish_reason") or "").lower()
                if (not pred) and fin in ("length", "max_tokens") and not args.dry_run:
                    data = call_model(chat_url, api_key, model, build_messages(item),
                                      args.max_tokens * 3, args.temperature,
                                      args.timeout, args.max_retries, args.json_mode)
                    raw = (data["choices"][0]["message"].get("content") or "")
                    correct, pred, tier = grade(item, raw)
                    fin = (data["choices"][0].get("finish_reason") or "").lower()
                cost, usage = response_cost(data, model, pricing)
                rec["responses"][item["item_id"]] = {
                    "correct": correct,                     # lenient policy
                    "correct_strict": int(correct and tier == "strict"),
                    "parse": tier, "pred": pred, "gold": item["answer"],
                    "raw": keep_raw(raw, args.raw_chars),
                    "cost": round(cost, 6), "err": None, "finish": fin}
                rec["n_strict"] = rec.get("n_strict", 0) + int(tier == "strict")
                rec["n_lenient"] = rec.get("n_lenient", 0) + int(tier == "lenient")
                if tier == "none":
                    rec["n_unparsed"] = rec.get("n_unparsed", 0) + 1
                rec["n_correct"] += correct
                rec["n_correct_strict"] = rec.get("n_correct_strict", 0) + int(
                    correct and tier == "strict")
                rec["cost_usd"] += cost
            except Exception as e:
                rec["responses"][item["item_id"]] = {
                    "correct": 0, "correct_strict": 0, "parse": "error",
                    "pred": "", "gold": item["answer"],
                    "raw": "", "cost": 0.0, "err": str(e)[:180], "finish": ""}
                rec["n_errors"] += 1
                print(f"    ! item {item['item_id']}: {e}", file=sys.stderr)
            if args.progress and (k % args.progress == 0 or k == len(items)):
                print(f"    item {k}/{len(items)}  "
                      f"({rec['n_correct']} right so far)", flush=True)
            if k % args.checkpoint == 0 or k == len(items):
                atomic_write(args.out, results)  # crash-safe checkpoint
            if args.sleep:
                time.sleep(args.sleep)

        shown = (rec.get("n_correct_strict", 0) if args.grading == "strict"
                 else rec["n_correct"])
        rec["score"] = round(shown / rec["n_items"], 4)
        rec["n_scored"] = shown
        print(f"    format: {rec.get('n_strict',0)} strict JSON, "
              f"{rec.get('n_lenient',0)} recovered from prose, "
              f"{rec.get('n_unparsed',0)} unparsable", file=sys.stderr)
        if rec.get("n_unparsed"):
            print(f"    note: {rec['n_unparsed']} reply(ies) had no parseable answer "
                  f"and were scored wrong.", file=sys.stderr)
        grand_cost += rec["cost_usd"]
        atomic_write(args.out, results)
        print(f"  {model}: {shown}/{rec['n_items']} correct [{args.grading}] "
              f"({rec['score']:.0%}), errors={rec['n_errors']}, "
              f"cost=${rec['cost_usd']:.4f} | running total ${grand_cost:.4f}")

    n = write_matrix(results, items, args.csv, args.grading)
    print(f"\nwrote {args.out} and {args.csv} ({n} models x {len(items)} items)")
    print(f"total inference cost this file: ${grand_cost:.4f} "
          f"(excludes OpenRouter's 5.5% credit-purchase fee)")


def main():
    ap = argparse.ArgumentParser(description="Score an item bank on OpenRouter models.")
    ap.add_argument("--bank", default="item_bank_50.json")
    ap.add_argument("--models", default="", help="comma-separated model ids")
    ap.add_argument("--models-file", default=None, help="file with one model id per line")
    ap.add_argument("--out", default="results.json")
    ap.add_argument("--csv", default="matrix.csv")
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--env-file", default=".env",
                    help="path to a .env file with OPENROUTER_API_KEY (default: .env)")
    ap.add_argument("--skip-existing", action="store_true",
                    help="skip models already scored on the current item set")
    ap.add_argument("--limit", type=int, default=0, help="score only first N items (smoke test)")
    ap.add_argument("--max-tokens", type=int, default=3000,
                    help="reply budget; verbose/reasoning models need room to "
                         "finish before emitting the final ANSWER line")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--timeout", type=float, default=90.0)
    ap.add_argument("--max-retries", type=int, default=4)
    ap.add_argument("--sleep", type=float, default=0.0, help="seconds between calls (throttle)")
    ap.add_argument("--checkpoint", type=int, default=5, help="write file every N items")
    ap.add_argument("--progress", type=int, default=10,
                    help="print a progress line every N items (0 to silence)")
    ap.add_argument("--local", action="store_true",
                    help="use a local LM Studio server (http://localhost:1234, no auth, cost=$0). "
                         "Pass the exact model id LM Studio reports via --models/--models-file.")
    ap.add_argument("--grading", choices=["strict", "lenient"], default="strict",
                    help="strict (default): a reply that ignores the JSON output format "
                         "counts as WRONG even where the answer is recoverable from prose - "
                         "following a clear instruction is part of the task. lenient: accept "
                         "answers recovered from prose. Both are always stored in the results "
                         "file, so make_matrix.py can switch policy without re-running.")
    ap.add_argument("--raw-chars", type=int, default=0,
                    help="cap on stored reply length; 0 (default) keeps the full text")
    ap.add_argument("--json-mode", action="store_true",
                    help="also send response_format=json_object (not all models support it)")
    ap.add_argument("--retry-errors", action="store_true",
                    help="re-score ONLY the items that previously failed with an API error, "
                         "for the given models, leaving all successful answers untouched. "
                         "Use this to recover network drops without a full re-run.")
    ap.add_argument("--dry-run", action="store_true", help="no API calls; simulate responses")
    ap.add_argument("--insecure", action="store_true",
                    help="disable TLS certificate verification (last resort; avoid)")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
