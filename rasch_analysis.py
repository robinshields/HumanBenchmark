#!/usr/bin/env python3
"""
Rasch (1PL) calibration + item/model diagnostics for the human-vs-LLM bench.

Input : matrix.csv from score_openrouter.py  (rows = models, cols = items, 0/1)
        optionally item_bank_50.json for item text, format and option counts
Output: console report + rasch_calibration.json (compact, for the web client)

Method: Joint Maximum Likelihood Estimation (JMLE), the standard approach for
small matrices like this. Items are centred at mean difficulty 0, so all
numbers are in logits relative to the average item.

Estimates and diagnostics reported:
  * item difficulty (logits) + standard error
  * model ability (logits) + standard error + 95% CI
  * infit / outfit mean-square (flags items behaving oddly)
  * point-measure correlation (flags mis-keyed or broken items)
  * separation & reliability (is the spread real, or noise?)
  * a Wright map putting models and items on the same ruler

No third-party dependencies.

Usage:
    python rasch_analysis.py --matrix matrix.csv --bank item_bank_50.json
    python rasch_analysis.py --matrix matrix.csv --selftest   # verify recovery
"""
import argparse, csv, json, math, random, sys

# --------------------------------------------------------------------------
# core model:  P(correct) = 1 / (1 + exp(-(theta - beta)))
# --------------------------------------------------------------------------
def p_correct(theta, beta):
    z = theta - beta
    if z > 35:  return 1.0 - 1e-15
    if z < -35: return 1e-15
    return 1.0 / (1.0 + math.exp(-z))


def jmle(resp, n_persons, n_items, max_iter=400, tol=1e-6, bias_correct=True):
    """Joint MLE for Rasch. resp[p][i] in {0,1,None}. Returns (theta, beta, info).

    Persons/items with extreme (all-right / all-wrong) scores have no finite
    estimate and are excluded from estimation, then reported separately.
    """
    # observed scores and counts over non-missing responses
    p_score = [0.0] * n_persons; p_n = [0] * n_persons
    i_score = [0.0] * n_items;   i_n = [0] * n_items
    for p in range(n_persons):
        for i in range(n_items):
            v = resp[p][i]
            if v is None: continue
            p_score[p] += v; p_n[p] += 1
            i_score[i] += v; i_n[i] += 1

    p_extreme = [p for p in range(n_persons) if p_n[p] == 0 or p_score[p] == 0 or p_score[p] == p_n[p]]
    i_extreme = [i for i in range(n_items)   if i_n[i] == 0 or i_score[i] == 0 or i_score[i] == i_n[i]]
    p_act = [p for p in range(n_persons) if p not in set(p_extreme)]
    i_act = [i for i in range(n_items)   if i not in set(i_extreme)]

    theta = [0.0] * n_persons
    beta  = [0.0] * n_items
    # sensible starting values (log-odds of observed proportion)
    for p in p_act:
        pr = p_score[p] / p_n[p]; theta[p] = math.log(pr / (1 - pr))
    for i in i_act:
        pr = i_score[i] / i_n[i]; beta[i] = -math.log(pr / (1 - pr))

    for _ in range(max_iter):
        max_delta = 0.0
        # --- update persons ---
        for p in p_act:
            exp_s = 0.0; info = 0.0
            for i in i_act:
                if resp[p][i] is None: continue
                pr = p_correct(theta[p], beta[i]); exp_s += pr; info += pr * (1 - pr)
            if info < 1e-9: continue
            obs = sum(resp[p][i] for i in i_act if resp[p][i] is not None)
            d = (obs - exp_s) / info
            d = max(-1.0, min(1.0, d))          # damped step for stability
            theta[p] += d; max_delta = max(max_delta, abs(d))
        # --- update items ---
        for i in i_act:
            exp_s = 0.0; info = 0.0
            for p in p_act:
                if resp[p][i] is None: continue
                pr = p_correct(theta[p], beta[i]); exp_s += pr; info += pr * (1 - pr)
            if info < 1e-9: continue
            obs = sum(resp[p][i] for p in p_act if resp[p][i] is not None)
            d = -(obs - exp_s) / info
            d = max(-1.0, min(1.0, d))
            beta[i] += d; max_delta = max(max_delta, abs(d))
        # --- centre items at 0 (identification constraint) ---
        if i_act:
            m = sum(beta[i] for i in i_act) / len(i_act)
            for i in i_act: beta[i] -= m
            for p in p_act: theta[p] -= m
        if max_delta < tol: break

    # JMLE is known to over-disperse; the standard (L-1)/L correction shrinks it
    if bias_correct and len(i_act) > 1:
        f_i = (len(i_act) - 1) / len(i_act)
        f_p = (len(p_act) - 1) / len(p_act) if len(p_act) > 1 else 1.0
        for i in i_act: beta[i] *= f_i
        for p in p_act: theta[p] *= f_p

    # --- standard errors from Fisher information ---
    p_se = [float('nan')] * n_persons
    i_se = [float('nan')] * n_items
    for p in p_act:
        info = sum(p_correct(theta[p], beta[i]) * (1 - p_correct(theta[p], beta[i]))
                   for i in i_act if resp[p][i] is not None)
        p_se[p] = 1 / math.sqrt(info) if info > 1e-9 else float('nan')
    for i in i_act:
        info = sum(p_correct(theta[p], beta[i]) * (1 - p_correct(theta[p], beta[i]))
                   for p in p_act if resp[p][i] is not None)
        i_se[i] = 1 / math.sqrt(info) if info > 1e-9 else float('nan')

    return theta, beta, {"p_extreme": p_extreme, "i_extreme": i_extreme,
                         "p_act": p_act, "i_act": i_act,
                         "p_score": p_score, "p_n": p_n,
                         "i_score": i_score, "i_n": i_n}


# --------------------------------------------------------------------------
# fit statistics
# --------------------------------------------------------------------------
def fit_stats(resp, theta, beta, p_act, i_act, axis):
    """Infit/outfit mean-square. axis='item' aggregates down columns."""
    out = {}
    outer = i_act if axis == "item" else p_act
    inner = p_act if axis == "item" else i_act
    for a in outer:
        num_o = den_o = num_i = den_i = 0.0
        for b in inner:
            p, i = (b, a) if axis == "item" else (a, b)
            v = resp[p][i]
            if v is None: continue
            pr = p_correct(theta[p], beta[i])
            w = pr * (1 - pr)
            if w < 1e-9: continue
            z2 = (v - pr) ** 2 / w
            num_o += z2; den_o += 1          # outfit: unweighted mean of z^2
            num_i += z2 * w; den_i += w      # infit: information-weighted
        outfit = num_o / den_o if den_o else float('nan')
        infit  = num_i / den_i if den_i else float('nan')
        # Wilson-Hilferty cube-root transform to a z-score
        def zstd(ms, df):
            if not df or ms <= 0 or math.isnan(ms): return float('nan')
            q = math.sqrt(2.0 / df)
            return (ms ** (1/3) - 1) * (3 / q) + (q / 3)
        out[a] = {"infit": infit, "outfit": outfit,
                  "infit_z": zstd(infit, den_o), "outfit_z": zstd(outfit, den_o)}
    return out


def point_measure_corr(resp, measures, p_act, i_act, axis):
    """Correlation between responses on an item and the person measures.
    Low/negative => item may be mis-keyed or broken (e.g. parser failure)."""
    out = {}
    outer = i_act if axis == "item" else p_act
    inner = p_act if axis == "item" else i_act
    for a in outer:
        xs, ys = [], []
        for b in inner:
            p, i = (b, a) if axis == "item" else (a, b)
            v = resp[p][i]
            if v is None: continue
            xs.append(v); ys.append(measures[b])
        n = len(xs)
        if n < 3: out[a] = float('nan'); continue
        mx = sum(xs)/n; my = sum(ys)/n
        sx = math.sqrt(sum((x-mx)**2 for x in xs)); sy = math.sqrt(sum((y-my)**2 for y in ys))
        out[a] = float('nan') if sx < 1e-12 or sy < 1e-12 else \
                 sum((x-mx)*(y-my) for x, y in zip(xs, ys)) / (sx*sy)
    return out


def separation(measures, ses, active):
    """Separation index & reliability: is the observed spread real signal?
    reliability ~ Cronbach alpha analogue; >0.8 = well separated."""
    if len(active) < 2: return float('nan'), float('nan'), float('nan')
    vals = [measures[a] for a in active]
    m = sum(vals)/len(vals)
    obs_var = sum((v-m)**2 for v in vals)/(len(vals)-1)
    mse = sum(ses[a]**2 for a in active if not math.isnan(ses[a]))/len(active)
    true_var = max(obs_var - mse, 1e-9)
    sep = math.sqrt(true_var/mse) if mse > 0 else float('nan')
    rel = true_var/obs_var if obs_var > 0 else float('nan')
    return sep, rel, math.sqrt(obs_var)


# --------------------------------------------------------------------------
# I/O
# --------------------------------------------------------------------------
def read_matrix(path):
    with open(path, newline='') as f:
        rows = list(csv.DictReader(f))
    if not rows: sys.exit("matrix.csv is empty")
    cols = [c for c in rows[0].keys() if c not in ("model", "score", "n")]
    models = [r["model"] for r in rows]
    resp = []
    for r in rows:
        line = []
        for c in cols:
            v = (r.get(c) or "").strip()
            line.append(None if v == "" else (1 if v in ("1", "1.0", "True") else 0))
        resp.append(line)
    return models, cols, resp


def load_bank(path):
    if not path: return {}
    try:
        return {it["item_id"]: it for it in json.load(open(path))}
    except Exception as e:
        print(f"(could not read bank: {e})", file=sys.stderr); return {}


def bar(x, lo, hi, width=40):
    if math.isnan(x): return " " * width
    f = (x - lo) / (hi - lo) if hi > lo else 0.5
    return "·" * max(0, min(width-1, int(f*width)))


# --------------------------------------------------------------------------
# self-test: can we recover known parameters?
# --------------------------------------------------------------------------
def selftest(n_p=25, n_i=50, seed=1):
    rng = random.Random(seed)
    true_theta = [rng.gauss(0, 2.0) for _ in range(n_p)]
    true_beta  = [rng.gauss(0, 1.5) for _ in range(n_i)]
    true_beta = [b - sum(true_beta)/n_i for b in true_beta]
    resp = [[1 if rng.random() < p_correct(t, b) else 0 for b in true_beta] for t in true_theta]
    theta, beta, info = jmle(resp, n_p, n_i)
    def corr(a, b):
        n=len(a); ma=sum(a)/n; mb=sum(b)/n
        sa=math.sqrt(sum((x-ma)**2 for x in a)); sb=math.sqrt(sum((x-mb)**2 for x in b))
        return sum((x-ma)*(y-mb) for x,y in zip(a,b))/(sa*sb)
    ia, pa = info["i_act"], info["p_act"]
    ci = corr([beta[i] for i in ia], [true_beta[i] for i in ia])
    cp = corr([theta[p] for p in pa], [true_theta[p] for p in pa])
    rmse_i = math.sqrt(sum((beta[i]-true_beta[i])**2 for i in ia)/len(ia))
    print(f"SELF-TEST  ({n_p} persons x {n_i} items, simulated from known truth)")
    print(f"  item difficulty  recovery r = {ci:.3f}   RMSE = {rmse_i:.3f} logits")
    print(f"  person ability   recovery r = {cp:.3f}")
    print(f"  extreme excluded: {len(info['i_extreme'])} items, {len(info['p_extreme'])} persons")
    ok = ci > 0.9 and cp > 0.9
    print("  RESULT:", "PASS - estimates recover truth" if ok else "FAIL")
    return ok


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", default="matrix.csv")
    ap.add_argument("--bank", default=None, help="item_bank_50.json (for text/options)")
    ap.add_argument("--out", default="rasch_calibration.json")
    ap.add_argument("--drop", default=None,
                    help="comma-separated item_ids to exclude (e.g. confirmed mis-keys)")
    ap.add_argument("--drop-file", default=None,
                    help="file with one item_id per line to exclude")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if selftest() else 1)

    models, item_ids, resp = read_matrix(a.matrix)
    bank = load_bank(a.bank)

    # optionally drop items (e.g. confirmed mis-keys) without editing the CSV
    drop = set()
    if a.drop:
        drop |= {s.strip() for s in a.drop.split(",") if s.strip()}
    if a.drop_file:
        drop |= {l.strip() for l in open(a.drop_file)
                 if l.strip() and not l.strip().startswith("#")}
    if drop:
        missing = drop - set(item_ids)
        if missing:
            print(f"! --drop ids not found in matrix (ignored): {', '.join(sorted(missing))}",
                  file=sys.stderr)
        keep = [k for k, iid in enumerate(item_ids) if iid not in drop]
        removed = [iid for iid in item_ids if iid in drop]
        item_ids = [item_ids[k] for k in keep]
        resp = [[row[k] for k in keep] for row in resp]
        print(f"dropped {len(removed)} item(s): {', '.join(removed)}\n")

    nP, nI = len(models), len(item_ids)
    theta, beta, info = jmle(resp, nP, nI)
    i_act, p_act = info["i_act"], info["p_act"]
    ifit = fit_stats(resp, theta, beta, p_act, i_act, "item")
    pfit = fit_stats(resp, theta, beta, p_act, i_act, "person")
    ipm  = point_measure_corr(resp, theta, p_act, i_act, "item")
    i_se = [0.0]*nI; p_se = [0.0]*nP
    # recompute SEs (jmle returned them via closure-free path)
    for i in i_act:
        inf = sum(p_correct(theta[p], beta[i])*(1-p_correct(theta[p], beta[i]))
                  for p in p_act if resp[p][i] is not None)
        i_se[i] = 1/math.sqrt(inf) if inf > 1e-9 else float('nan')
    for p in p_act:
        inf = sum(p_correct(theta[p], beta[i])*(1-p_correct(theta[p], beta[i]))
                  for i in i_act if resp[p][i] is not None)
        p_se[p] = 1/math.sqrt(inf) if inf > 1e-9 else float('nan')

    print("=" * 78)
    print(f"RASCH CALIBRATION   {nP} models x {nI} items")
    print("=" * 78)
    if info["i_extreme"]:
        print(f"\n!! {len(info['i_extreme'])} item(s) with extreme scores (all right / all wrong) "
              f"- no finite difficulty, excluded:")
        for i in info["i_extreme"]:
            print(f"     {item_ids[i]}  ({int(info['i_score'][i])}/{info['i_n'][i]} correct)")
    if info["p_extreme"]:
        print(f"\n!! {len(info['p_extreme'])} model(s) with extreme scores, excluded from fit:")
        for p in info["p_extreme"]:
            print(f"     {models[p]}  ({int(info['p_score'][p])}/{info['p_n'][p]})")

    # ---------------- models ----------------
    print("\n" + "-"*78)
    print("MODEL ABILITY (logits, higher = better).  95% CI = +/- 1.96 SE")
    print("-"*78)
    order = sorted(range(nP), key=lambda p: -(theta[p] if p in p_act else -99))
    los = [theta[p]-1.96*p_se[p] for p in p_act]; his = [theta[p]+1.96*p_se[p] for p in p_act]
    lo, hi = (min(los), max(his)) if p_act else (-1, 1)
    print(f"{'model':38s} {'raw':>7s} {'logit':>7s} {'SE':>5s}  {'95% CI':>15s}  infit")
    for p in order:
        raw = f"{int(info['p_score'][p])}/{info['p_n'][p]}"
        if p in p_act:
            ci = f"[{theta[p]-1.96*p_se[p]:+.2f},{theta[p]+1.96*p_se[p]:+.2f}]"
            fi = pfit.get(p, {}).get("infit", float('nan'))
            print(f"{models[p][:38]:38s} {raw:>7s} {theta[p]:+7.2f} {p_se[p]:5.2f}  {ci:>15s}  {fi:4.2f}")
        else:
            print(f"{models[p][:38]:38s} {raw:>7s} {'extreme':>7s}")

    sep_p, rel_p, sd_p = separation(theta, p_se, p_act)
    print(f"\n  model spread SD = {sd_p:.2f} logits | separation = {sep_p:.2f} | reliability = {rel_p:.3f}")
    print("  (reliability >0.8 means the model ranking is reliably separated, not noise)")

    # ---------------- items ----------------
    print("\n" + "-"*78)
    print("ITEM DIFFICULTY (logits, higher = harder)")
    print("-"*78)
    iorder = sorted(i_act, key=lambda i: -beta[i])
    blo = min(beta[i] for i in i_act); bhi = max(beta[i] for i in i_act)
    print(f"{'item_id':42s} {'p_corr':>6s} {'logit':>7s} {'SE':>5s} {'infit':>5s} {'outfit':>6s} {'pmc':>6s}")
    for i in iorder:
        pc = info['i_score'][i]/info['i_n'][i]
        f = ifit[i]
        print(f"{item_ids[i][:42]:42s} {pc:6.2f} {beta[i]:+7.2f} {i_se[i]:5.2f} "
              f"{f['infit']:5.2f} {f['outfit']:6.2f} {ipm[i]:+6.2f}")

    sep_i, rel_i, sd_i = separation(beta, i_se, i_act)
    print(f"\n  item spread SD = {sd_i:.2f} logits | separation = {sep_i:.2f} | reliability = {rel_i:.3f}")

    # ---------------- flags ----------------
    print("\n" + "-"*78)
    print("MISFITTING ITEMS  (worth eyeballing in results.json)")
    print("-"*78)
    print("Only UNDERFIT is a problem: outfit >2 means erratic responding - strong")
    print("models miss it while weak ones pass. That is the signature of a wrong")
    print("gold answer, an ambiguous item, or a parser break. Severity rises with")
    print("outfit; a negative point-measure correlation all but confirms a mis-key.")
    flagged = False
    for i in iorder:
        why = []
        o = ifit[i]["outfit"]
        if o > 2.0:
            sev = "SEVERE" if o > 4.0 else "moderate"
            why.append(f"outfit {o:.2f} ({sev} underfit)")
        if not math.isnan(ipm[i]) and ipm[i] <= 0.10:
            why.append(f"point-measure {ipm[i]:+.2f} (likely mis-key / broken)")
        if why:
            flagged = True
            print(f"  {item_ids[i]}: " + "; ".join(why))
    if not flagged:
        print("  none - no items show erratic responding.")

    # Low outfit is benign: it means responses were MORE orderly than predicted.
    # It is expected when respondent ability is widely spread, as here. Such
    # items are merely "less productive for measurement", never distorting.
    overfit = [i for i in i_act if ifit[i]["outfit"] < 0.5]
    if overfit:
        print(f"\n  ({len(overfit)} item(s) have outfit <0.5 - unusually predictable. This is")
        print("   benign and expected given the wide model ability spread; no action needed.)")

    # ---------------- Wright map ----------------
    print("\n" + "-"*78)
    print("WRIGHT MAP  (same ruler: models left, items right)")
    print("-"*78)
    allv = [theta[p] for p in p_act] + [beta[i] for i in i_act]
    top, bot = max(allv), min(allv)
    nbin = 16
    for b in range(nbin, -1, -1):
        cut_hi = bot + (top-bot)*(b+1)/(nbin+1)
        cut_lo = bot + (top-bot)*b/(nbin+1)
        mids = [models[p][:16] for p in p_act if cut_lo <= theta[p] < cut_hi]
        its  = [i for i in i_act if cut_lo <= beta[i] < cut_hi]
        lvl = (cut_lo+cut_hi)/2
        left = ",".join(mids)[:34]
        print(f"{lvl:+5.1f} |{left:>35s} | {'#'*len(its)} {len(its) if its else ''}")
    print(f"{'':6s}{'MODELS':>36s} | ITEMS (harder at top)")

    # ---------------- export ----------------
    # Expected percent-correct on the FULL bank at a given ability. This is the
    # display scale: monotonic in ability (never reorders anyone) and directly
    # comparable between a model that answered all items and a human who
    # answered only a handful adaptively.
    scored_items = [i for i in i_act]

    def expected_pct(t):
        if not scored_items: return float('nan')
        return 100.0 * sum(p_correct(t, beta[i]) for i in scored_items) / len(scored_items)

    perfect = [p for p in info["p_extreme"]
               if info["p_n"][p] and info["p_score"][p] == info["p_n"][p]]
    zero = [p for p in info["p_extreme"]
            if info["p_n"][p] and info["p_score"][p] == 0]

    print("\n" + "-"*78)
    print("DISPLAY SCALE: expected % correct on the full bank")
    print("-"*78)
    for p in sorted(p_act, key=lambda p: -theta[p]):
        pct = expected_pct(theta[p])
        lo = expected_pct(theta[p] - 1.96*p_se[p]); hi = expected_pct(theta[p] + 1.96*p_se[p])
        print(f"{models[p][:38]:38s} {pct:5.1f}%   [{lo:4.1f}-{hi:4.1f}]")
    if perfect:
        print(f"\n  PERFECT SCORE ({len(perfect)} models, no finite estimate - display as one bar):")
        for p in perfect: print(f"     {models[p]}")

    export = {
        "meta": {"n_models": nP, "n_items": nI, "n_items_scored": len(scored_items),
                 "item_reliability": None if math.isnan(rel_i) else round(rel_i, 3),
                 "model_reliability": None if math.isnan(rel_p) else round(rel_p, 3),
                 "scale": "expected_pct = mean P(correct) over all scored items, x100",
                 "note": "difficulties centred at mean 0 logits; estimated from LLM responses"},
        # lookup curve so the client can map a human's ability -> % without
        # re-summing over items (interpolate between points)
        "ability_to_pct": [[round(t/10.0, 1), round(expected_pct(t/10.0), 2)]
                           for t in range(-70, 71, 2)],
        "items": [], "models": [], "perfect_score_models": [models[p] for p in perfect]}

    for i in range(nI):
        it = bank.get(item_ids[i], {})
        nopt = it.get("n_options")
        export["items"].append({
            "item_id": item_ids[i],
            "difficulty": None if i in info["i_extreme"] else round(beta[i], 4),
            "se": None if i in info["i_extreme"] else round(i_se[i], 4),
            "p_correct_models": round(info['i_score'][i]/info['i_n'][i], 3) if info['i_n'][i] else None,
            "infit": None if i in info["i_extreme"] else round(ifit[i]["infit"], 3),
            "outfit": None if i in info["i_extreme"] else round(ifit[i]["outfit"], 3),
            "guess_floor": (1.0/nopt) if nopt else 0.0,
            "format": it.get("format"), "tier": it.get("tier"), "source": it.get("source"),
        })
    for p in range(nP):
        is_perfect = p in perfect
        is_zero = p in zero
        export["models"].append({
            "model": models[p],
            "ability": None if p in info["p_extreme"] else round(theta[p], 4),
            "se": None if p in info["p_extreme"] else round(p_se[p], 4),
            "expected_pct": 100.0 if is_perfect else (
                0.0 if is_zero else round(expected_pct(theta[p]), 2)),
            "expected_pct_lo": None if p in info["p_extreme"] else round(expected_pct(theta[p]-1.96*p_se[p]), 2),
            "expected_pct_hi": None if p in info["p_extreme"] else round(expected_pct(theta[p]+1.96*p_se[p]), 2),
            "perfect_score": is_perfect,
            "raw_score": int(info['p_score'][p]), "n_items": info['p_n'][p],
            "pct_raw": round(info['p_score'][p]/info['p_n'][p]*100, 1) if info['p_n'][p] else None,
        })
    json.dump(export, open(a.out, "w"), indent=1)
    print(f"\nwrote {a.out}  (embed this in the web client)")


if __name__ == "__main__":
    main()
