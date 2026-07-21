#!/usr/bin/env python3
"""
Build the static benchmark page.

Merges the item bank (question text, choices, answers) with the Rasch
calibration (item difficulties, model abilities, ability->percent curve)
into ONE self-contained index.html with no external data files and no
backend. Deploy it anywhere that serves static files.

Usage:
    python build_site.py                       # uses the default filenames
    python build_site.py --bank item_bank_50.json \
                         --calibration rasch_calibration.json \
                         --template template.html --out index.html

Items are included only if the calibration gave them a finite difficulty,
so anything you dropped with --drop (or that scored all-right/all-wrong)
is automatically left out of the site.
"""
import argparse, json, statistics, sys

KIND = {
    "commonsenseqa": "Common sense",
    "agieval_lsat_lr": "Argument reasoning",
    "agieval_aqua_rat": "Exam math",
    "opensat_english": "Grammar",
    "hellaswag": "Common sense",
    "gsm8k": "Arithmetic",
    "bbh_object_counting": "Counting",
    "bbh_date_understanding": "Dates",
    "bbh_causal_judgement": "Cause and effect",
    "bbh_temporal_sequences": "Schedules",
    "bbh_reasoning_about_colored_objects": "Objects",
    "bbh_logical_deduction_five_objects": "Deduction",
    "bbh_tracking_shuffled_objects_five_objects": "Keeping track",
    "bbh_web_of_lies": "Truth and lies",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", default="item_bank_50.json")
    ap.add_argument("--calibration", default="rasch_calibration.json")
    ap.add_argument("--template", default="template.html")
    ap.add_argument("--out", default="index.html")
    ap.add_argument("--results", default=None,
                    help="results.json - used to attach a per-item median answer cost "
                         "(for the tongue-in-cheek 'LLM hourly rate'). Optional.")
    ap.add_argument("--variety", type=float, default=0.35,
                    help="0..1 knob for repeat-play variety. 0 = always serve the most "
                         "informative item (precise but repetitive); 1 = draw from a broad "
                         "band near the player's level (varied, slightly less precise). "
                         "Default 0.35.")
    a = ap.parse_args()

    bank = {it["item_id"]: it for it in json.load(open(a.bank))}
    cal = json.load(open(a.calibration))

    # per-item median cost across models that actually answered (cost > 0).
    # Median so one expensive model does not dominate; only paid API models
    # contribute, local/free ones (cost 0) are ignored.
    item_cost, item_right, item_wrong = {}, {}, {}
    if a.results:
        res = json.load(open(a.results))
        per = {}
        for model, rec in res.get("models", {}).items():
            short = model.split("/", 1)[-1].split(":", 1)[0]
            for iid, r in (rec.get("responses") or {}).items():
                c = r.get("cost") or 0
                if c > 0:
                    per.setdefault(iid, []).append(c)
                # strict correctness, matching the shipped grading
                ok = r.get("correct_strict", r.get("correct", 0))
                (item_right if ok else item_wrong).setdefault(iid, []).append(short)
        item_cost = {iid: statistics.median(v) for iid, v in per.items() if v}

    items, skipped = [], []
    for ci in cal["items"]:
        iid = ci["item_id"]
        if ci.get("difficulty") is None:
            skipped.append((iid, "no finite difficulty")); continue
        src = bank.get(iid)
        if not src:
            skipped.append((iid, "not found in bank")); continue
        items.append({
            "item_id": iid,
            "question": src["question"],
            "choices": src.get("choices"),
            "answer": src["answer"],
            "format": src["format"],
            "difficulty": ci["difficulty"],
            "guess_floor": ci.get("guess_floor", 0.0),
            "kind": KIND.get(src.get("source", ""), src.get("source", "")),
            "cost": round(item_cost.get(iid, 0.0), 8),
            "got_right": item_right.get(iid, []),
            "got_wrong": item_wrong.get(iid, []),
            "msg_right": src.get("msg_right"),
            "msg_wrong": src.get("msg_wrong"),
        })

    def split_name(full):
        # "anthropic/claude-haiku-4.5" -> ("anthropic", "claude-haiku-4.5")
        if "/" in full:
            co, short = full.split("/", 1)
        else:
            co, short = "", full
        short = short.split(":", 1)[0]        # drop ":free" / ":paid" suffixes
        return co, short

    models = []
    for m in cal["models"]:
        co, short = split_name(m["model"])
        models.append({
            "model": m["model"], "company": co, "short": short,
            "expected_pct": m["expected_pct"],
            "expected_pct_lo": m.get("expected_pct_lo"),
            "expected_pct_hi": m.get("expected_pct_hi"),
            "perfect_score": bool(m.get("perfect_score")),
        })

    variety = max(0.0, min(1.0, a.variety))
    data = {"items": items, "models": models,
            "ability_to_pct": cal["ability_to_pct"],
            "variety": variety,
            "meta": cal.get("meta", {})}

    tpl = open(a.template).read()
    if "/*__DATA__*/" not in tpl:
        sys.exit("template is missing the /*__DATA__*/ placeholder")
    # </script> inside JSON would close the tag early; escape defensively
    blob = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    html = tpl.replace("/*__DATA__*/ null", blob)

    open(a.out, "w").write(html)
    kb = len(html.encode()) / 1024
    print(f"wrote {a.out}  ({kb:.0f} KB, self-contained)")
    print(f"  {len(items)} items, {len(models)} models "
          f"({sum(1 for m in models if m['perfect_score'])} perfect)")
    if a.results:
        withcost = sum(1 for it in items if it["cost"] > 0)
        print(f"  per-item median cost attached to {withcost}/{len(items)} items")
    print(f"  variety = {variety:.2f}")
    if skipped:
        print(f"  skipped {len(skipped)} item(s):")
        for iid, why in skipped:
            print(f"    {iid}: {why}")


if __name__ == "__main__":
    main()
