#!/usr/bin/env python3
"""
Build a unified, high-range item bank for the "beat the LLMs" human test.

GSM8K and BIG-Bench Hard come from GitHub raw (no key needed). CommonsenseQA
comes from HuggingFace `datasets`, or from a local dev_rand_split.jsonl via
--csqa-local. Answers for all of these are already public on the web, so
embedding them client-side is fine (unlike GPQA / HLE).

NOTE: HellaSwag was the original easy anchor and has been REMOVED. Its contexts
are crowd-written captions of video clips and its distractors are machine
generated and adversarially filtered, so many items are ambiguous or
unanswerable without the video, and the trailing pronoun is lowercased by
construction. It calibrated as the hardest, most erratic content in the bank -
the opposite of an easy anchor. CommonsenseQA replaces it: human-written
throughout, 5 options, genuinely easy.

Ten reasoning flavours, spanning easy -> hard, minimal domain knowledge:

  easy    commonsenseqa                          everyday commonsense (5-opt MC)
  medium  gsm8k                                  arithmetic word problems (numeric)
  medium  bbh object_counting                    counting (numeric)
  medium  bbh date_understanding                 date/temporal arithmetic (MC)
  medium  bbh causal_judgement                   everyday causal reasoning (Yes/No)
  hard    bbh temporal_sequences                 schedule/constraint reasoning (MC)
  hard    bbh reasoning_about_colored_objects    attribute + position tracking (MC)
  hard    bbh logical_deduction_five_objects     ordering deduction (5-opt MC)
  hard    bbh tracking_shuffled_objects_five     swap tracking (5-opt MC)
  hard    bbh web_of_lies                        truth-teller/liar chains (Yes/No)

Usage:
    python build_item_bank.py --seed 7 --out item_bank.json
    # optional: --scale 2   (double every per-source count, ~100 items)
"""
import argparse, json, os, random, re, ssl, sys, urllib.error, urllib.request


def _ssl_contexts():
    """Candidate TLS contexts, tried in order.

    Environments differ: a stock macOS python.org install has no system CA
    certs (certifi fixes it), while a machine behind a TLS-inspecting proxy
    has its root CA in the SYSTEM store but not in certifi. Trying both, in
    that order, works in either case. SSL_CERT_FILE (if set) wins outright,
    which is the escape hatch for a corporate/university CA bundle.
    """
    ctxs = []
    if os.environ.get("SSL_CERT_FILE"):
        ctxs.append(("SSL_CERT_FILE", ssl.create_default_context()))
    try:
        import certifi
        ctxs.append(("certifi", ssl.create_default_context(cafile=certifi.where())))
    except ImportError:
        pass
    ctxs.append(("system CA store", ssl.create_default_context()))
    return ctxs


SSL_CTXS = _ssl_contexts()
_SSL_OK = {}          # remember which context worked, so we only probe once


RAW = "https://raw.githubusercontent.com"
CSQA_HF_IDS = ["tau/commonsense_qa", "commonsense_qa"]   # tried in order
GSM8K = f"{RAW}/openai/grade-school-math/master/grade_school_math/data/test.jsonl"
BBH = lambda t: f"{RAW}/suzgunmirac/BIG-Bench-Hard/main/bbh/{t}.json"


PLAN = [
    ("commonsenseqa",                          "easy",   18),
    ("agieval:lsat-lr",                        "medium", 12),   # argument reasoning
    ("agieval:aqua-rat",                       "medium", 12),   # exam-style quantitative
    ("gsm8k",                                  "medium", 14),
    ("object_counting",                        "medium",  7),
    ("date_understanding",                     "medium",  7),
    ("causal_judgement",                       "medium",  5),
    ("temporal_sequences",                     "hard",    6),
    ("reasoning_about_colored_objects",        "hard",    6),
    ("logical_deduction_five_objects",         "hard",    7),
    ("tracking_shuffled_objects_five_objects", "hard",    3),
    ("web_of_lies",                            "hard",    3),
]

# To add hand-vetted OpenSAT English items, review opensat_candidates.txt, then
# add ("opensat_english", "easy", N) above and pass --opensat / --opensat-ids.
BINARY_CHOICES = {
    "causal_judgement": ["Yes", "No"],
    "web_of_lies":      ["Yes", "No"],
    "boolean_expressions": ["True", "False"],
}
NUMERIC = {"object_counting"}


def fetch(url):
    if _SSL_OK:                      # a context already proved itself
        ctx = _SSL_OK["ctx"]
        with urllib.request.urlopen(url, timeout=60, context=ctx) as r:
            return r.read().decode("utf-8")
    last = None
    for name, ctx in SSL_CTXS:
        try:
            with urllib.request.urlopen(url, timeout=60, context=ctx) as r:
                data = r.read().decode("utf-8")
            _SSL_OK.update(ctx=ctx, name=name)
            print(f"TLS: verified via {name}", file=sys.stderr)
            return data
        except (ssl.SSLCertVerificationError, urllib.error.URLError) as e:
            # urlopen wraps the SSL failure inside URLError, so unwrap and
            # only fall through for genuine certificate problems - a real
            # network outage should surface immediately, not be retried.
            reason = getattr(e, "reason", e)
            if not isinstance(reason, ssl.SSLCertVerificationError) and \
               "CERTIFICATE_VERIFY_FAILED" not in str(e):
                raise
            last = e
            continue                 # certificate problem -> try the next CA source
    raise RuntimeError(
        "HTTPS certificate verification failed with every available CA bundle "
        f"({', '.join(n for n, _ in SSL_CTXS)}). Last error: {last}\n"
        "Fix: pip install certifi   (or, on a network that inspects TLS, set "
        "SSL_CERT_FILE=/path/to/your/ca-bundle.pem)")


def load_commonsenseqa(local_path=None):
    """CommonsenseQA validation split (the one with answer keys). 5-option MC.

    Either read a local dev_rand_split.jsonl, or pull via HuggingFace datasets.
    """
    if local_path:
        recs = []
        for line in open(local_path):
            if not line.strip():
                continue
            r = json.loads(line)
            ch = r["question"]["choices"]
            recs.append((r.get("answerKey"), r["question"]["stem"],
                         [c["text"] for c in ch], [c["label"] for c in ch]))
    else:
        from datasets import load_dataset          # pip install datasets
        ds, last = None, None
        for name in CSQA_HF_IDS:
            try:
                ds = load_dataset(name, split="validation"); break
            except Exception as e:                 # try the next alias
                last = e
        if ds is None:
            raise RuntimeError(f"could not load CommonsenseQA from {CSQA_HF_IDS}: {last}")
        recs = [(x["answerKey"], x["question"], x["choices"]["text"],
                 x["choices"]["label"]) for x in ds]

    out = []
    for key, stem, texts, labels in recs:
        if not key or key not in labels:           # test split has no key
            continue
        idx = labels.index(key)
        stem = stem.strip()
        if not stem or len(texts) < 2:
            continue
        out.append(dict(source="commonsenseqa", format="mc", n_options=len(texts),
                        tier="easy", question=stem,
                        choices=[t.strip() for t in texts],
                        answer=chr(65 + idx), answer_text=texts[idx].strip()))
    return out


OPENSAT_KEEP_DOMAIN = "Standard English Conventions"
OPENSAT_GRAMMAR_STEM = re.compile(
    r"conventions of standard english|punctuation|combines? the sentences|grammatical", re.I)


def load_opensat(path, keep_ids=None):
    """SAT English (OpenSAT / pinesat). Grammar and punctuation only.

    This bank is partly AI-generated and needs aggressive filtering:
      * only Standard English Conventions - the other domains (Information and
        Ideas, Craft and Structure) are reading comprehension.
      * ~27% of items reference an "underlined portion" that does not exist in
        the text, making them unanswerable. Excluded.
      * some have a literal "null" paragraph. Excluded.
      * long passages and long answer choices are excluded to keep items quick.
    Even after this, expect a few mis-keyed or ambiguous items - the Rasch
    misfit diagnostics in rasch_analysis.py are there to catch them.
    """
    raw = fetch(path) if path.startswith("http") else open(path).read()
    data = json.loads(raw)
    out = []
    for x in data:
        if x.get("domain") != OPENSAT_KEEP_DOMAIN:
            continue
        q = x.get("question") or {}
        stem = (q.get("question") or "").strip()
        para = q.get("paragraph")
        para = "" if para in (None, "null") else para.strip()
        choices = q.get("choices") or {}
        key = q.get("correct_answer")
        if not (para and stem and choices and key in choices):
            continue
        if "underlin" in stem.lower():          # underline is not in the text
            continue
        if not OPENSAT_GRAMMAR_STEM.search(stem):
            continue
        if not (60 <= len(para) <= 420):
            continue
        vals = [str(choices[k]).strip() for k in sorted(choices)]
        if not all(4 <= len(c) <= 90 for c in vals):
            continue
        labels = sorted(choices)
        idx = labels.index(key)
        out.append(dict(source="opensat_english", format="mc", n_options=len(vals),
                        tier="easy", question=f"{para}\n\n{stem}",
                        choices=vals, answer=chr(65 + idx), answer_text=vals[idx],
                        src_id=x.get("id")))
    # de-duplicate on the passage
    seen, uniq = set(), []
    for it in out:
        p = it["question"].split("\n\n")[0]
        if p in seen:
            continue
        seen.add(p); uniq.append(it)
    if keep_ids:
        uniq = [it for it in uniq if it.get("src_id") in keep_ids]
    return uniq


AGIEVAL = ("https://raw.githubusercontent.com/ruixiangcui/AGIEval/main/data/v1/{}.jsonl")
_MATH_MARKUP = re.compile(r"\\[a-zA-Z]+|\^\{|_\{")
_FIGURE = re.compile(r"figure above|graph above|table above|shown above|the figure|diagram",
                     re.I)


def load_agieval(task, tier, max_chars=760):
    """AGIEval subsets: real, professionally keyed exam questions.

    Only self-contained subsets are used. sat-en and lsat-rc are excluded
    elsewhere because they are reading comprehension (median ~4400 and ~3100
    characters, with a long passage per question). sat-math is excluded
    because 94% of its items are LaTeX, which will not render as plain text.
    """
    out = []
    for line in fetch(AGIEVAL.format(task)).splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        opts = r.get("options") or []
        label = (r.get("label") or "").strip()
        passage = (r.get("passage") or "").strip()
        question = (r.get("question") or "").strip()
        if not opts or not question or not label:
            continue
        body = (passage + "\n\n" + question).strip() if passage else question
        blob = body + " ".join(str(o) for o in opts)
        if _MATH_MARKUP.search(blob) or _FIGURE.search(blob):
            continue                       # unrenderable markup or a missing figure
        if len(body) > max_chars:
            continue                       # keep items quick to read
        # options arrive as "(A)text" - strip the label prefix
        choices = [re.sub(r"^\(?[A-E]\)\s*", "", str(o)).strip() for o in opts]
        if not all(choices):
            continue
        idx = ord(label[0].upper()) - 65
        if not (0 <= idx < len(choices)):
            continue
        out.append(dict(source=f"agieval_{task.replace('-', '_')}", format="mc",
                        n_options=len(choices), tier=tier, question=body,
                        choices=choices, answer=chr(65 + idx),
                        answer_text=choices[idx]))
    return out


def load_gsm8k():
    rows = [json.loads(l) for l in fetch(GSM8K).splitlines() if l.strip()]
    out = []
    for r in rows:
        final = r["answer"].split("####")[-1].strip().replace(",", "")
        out.append(dict(source="gsm8k", format="numeric", n_options=None,
                        tier="medium", question=r["question"].strip(),
                        choices=None, answer=final, answer_text=final))
    return out


def load_bbh(task, tier):
    data = json.loads(fetch(BBH(task)))["examples"]
    out = []
    for e in data:
        inp, tgt = e["input"].strip(), e["target"].strip()
        if task in NUMERIC:
            out.append(dict(source=f"bbh_{task}", format="numeric", n_options=None,
                            tier=tier, question=inp, choices=None,
                            answer=tgt, answer_text=tgt))
            continue
        if "Options:" in inp:
            stem, opts = inp.split("Options:", 1)
            lettered = re.findall(r"\(([A-Z])\)\s*(.+)", opts)
            if lettered:
                choices = [t.strip() for _, t in lettered]
            else:
                choices = [o.strip("-\u2022 ").strip() for o in opts.splitlines() if o.strip()]
            stem = stem.strip()
        else:
            choices = BINARY_CHOICES.get(task, ["Yes", "No"])
            stem = inp
        if re.fullmatch(r"\([A-Z]\)", tgt):
            letter = tgt.strip("()")
        else:
            low = [c.lower() for c in choices]
            letter = chr(65 + low.index(tgt.lower()))
        out.append(dict(source=f"bbh_{task}", format="mc", n_options=len(choices),
                        tier=tier, question=stem, choices=choices,
                        answer=letter, answer_text=choices[ord(letter) - 65]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--scale", type=int, default=1)
    ap.add_argument("--out", default="item_bank.json")
    ap.add_argument("--opensat", default=None,
                    help="path (or URL) to the OpenSAT question JSON, e.g. pine_sat.json")
    ap.add_argument("--opensat-ids", default=None,
                    help="file of hand-approved OpenSAT ids, one per line "
                         "(use after reviewing opensat_candidates.txt)")
    ap.add_argument("--csqa-local", default=None,
                    help="path to CommonsenseQA dev_rand_split.jsonl "
                         "(skips HuggingFace entirely)")
    args = ap.parse_args()
    random.seed(args.seed)

    bank = []
    for key, tier, n in PLAN:
        if key.startswith("agieval:"):
            pool = load_agieval(key.split(":", 1)[1], tier)
        elif key == "commonsenseqa":
            pool = load_commonsenseqa(args.csqa_local)
        elif key == "opensat_english":
            if not args.opensat:
                print("  !! --opensat not given; skipping SAT English items.",
                      file=sys.stderr)
                continue
            keep = None
            if args.opensat_ids:
                keep = {l.strip() for l in open(args.opensat_ids)
                        if l.strip() and not l.strip().startswith("#")}
            pool = load_opensat(args.opensat, keep)
        elif key == "gsm8k":
            pool = load_gsm8k()
        else:
            pool = load_bbh(key, tier)
        k = min(n * args.scale, len(pool))
        bank += random.sample(pool, k)

    for i, it in enumerate(bank):
        it["item_id"] = f"{it['source']}_{i:03d}"
    random.shuffle(bank)
    json.dump(bank, open(args.out, "w"), ensure_ascii=False, indent=1)

    from collections import Counter
    print(f"wrote {len(bank)} items -> {args.out}")
    print("tier  :", dict(Counter(i["tier"] for i in bank)))
    print("format:", dict(Counter(i["format"] for i in bank)))
    print("source:", dict(Counter(i["source"] for i in bank)))


if __name__ == "__main__":
    main()
