#!/usr/bin/env python3
"""
Build matrix.csv (the model x item correct/incorrect matrix) from results.json.

Scoring and matrix construction are deliberately separate jobs. results.json is
the source of truth - it holds every reply, how it parsed, and both gradings.
The matrix is just a derived view of it, so it can be rebuilt at any time, under
either grading policy, with no API calls and no cost.

Grading policies:
  strict  (default) - a reply that ignored the requested JSON output format is
                      marked WRONG even if the answer was recoverable from
                      prose. Following a clear instruction is part of the task.
  lenient           - accept answers recovered from prose.

Usage:
    python make_matrix.py --results results_v3.json --bank item_bank_v2.json
    python make_matrix.py --results results_v3.json --bank item_bank_v2.json \
                          --grading lenient --csv matrix_lenient.csv
    python make_matrix.py --results results_v3.json --bank item_bank_v2.json --compare
"""
import argparse, csv, hashlib, json, sys


def bank_hash(items):
    key = json.dumps([[it["item_id"], it["answer"]] for it in items], sort_keys=True)
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def cell(r, grading):
    if r is None:
        return ""
    if grading == "strict":
        return r.get("correct_strict", r.get("correct", 0))
    return r.get("correct", 0)


def build(results, items, grading):
    cur = bank_hash(items)
    item_ids = [it["item_id"] for it in items]
    rows, stale = [], []
    for model, rec in results["models"].items():
        if rec.get("item_hash") and rec["item_hash"] != cur:
            stale.append(model)
            continue
        resp = rec.get("responses") or {}
        row = {"model": model}
        score = 0
        for iid in item_ids:
            v = cell(resp.get(iid), grading)
            row[iid] = v
            if v == 1:
                score += 1
        row["score"] = score
        row["n"] = len(item_ids)
        rows.append(row)
    return rows, item_ids, stale


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results_v3.json")
    ap.add_argument("--bank", default="item_bank_v2.json")
    ap.add_argument("--csv", default=None,
                    help="output path (default: matrix_<grading>.csv)")
    ap.add_argument("--grading", choices=["strict", "lenient"], default="strict")
    ap.add_argument("--compare", action="store_true",
                    help="show how the two policies differ, and write neither")
    a = ap.parse_args()

    results = json.load(open(a.results))
    items = json.load(open(a.bank))

    if a.compare:
        strict, _, _ = build(results, items, "strict")
        lenient, _, _ = build(results, items, "lenient")
        sc = {r["model"]: r["score"] for r in strict}
        ln = {r["model"]: r["score"] for r in lenient}
        order_s = [m for m, _ in sorted(sc.items(), key=lambda kv: -kv[1])]
        order_l = [m for m, _ in sorted(ln.items(), key=lambda kv: -kv[1])]
        print(f"{'model':44s} {'strict':>7s} {'lenient':>8s} {'diff':>6s} {'rank shift':>11s}")
        for m in order_s:
            shift = order_l.index(m) - order_s.index(m)
            arrow = "" if shift == 0 else (f"{shift:+d}")
            print(f"{m[:44]:44s} {sc[m]:7d} {ln[m]:8d} {ln[m]-sc[m]:6d} {arrow:>11s}")
        moved = sum(1 for m in order_s if order_l.index(m) != order_s.index(m))
        print(f"\n{moved} of {len(order_s)} models change rank between policies.")
        print("'rank shift' is where the model would move under lenient grading.")
        return

    rows, item_ids, stale = build(results, items, a.grading)
    out = a.csv or f"matrix_{a.grading}.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model"] + item_ids + ["score", "n"])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"wrote {out}  ({len(rows)} models x {len(item_ids)} items, {a.grading} grading)")
    if stale:
        print(f"  excluded {len(stale)} model(s) scored against a different item set: "
              f"{', '.join(stale)}", file=sys.stderr)

    # format compliance, since it explains most strict/lenient differences
    fmt = [(m, r.get("n_strict", 0), r.get("n_lenient", 0), r.get("n_unparsed", 0))
           for m, r in results["models"].items()]
    if any(l or u for _, _, l, u in fmt):
        print("\nformat compliance (strict JSON / recovered from prose / unparsable):")
        for m, st, le, un in sorted(fmt, key=lambda x: -(x[2] + x[3])):
            if le or un:
                print(f"  {m[:44]:44s} {st:4d} / {le:4d} / {un:4d}")


if __name__ == "__main__":
    main()
