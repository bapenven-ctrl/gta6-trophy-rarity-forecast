"""Validation harness for the gta6-trophy-rarity-forecast-01 entry.

Runs three forecasters -- the candidate (../main.py), the house baseline
(persistence, reimplemented from RULES.txt), and the house curve baseline
(reimplemented from RULES.txt) -- through:

  (i)   a leakage-free time split on the practice cases p00/p01: fit on the
        first half of each case's reads, score against the second half;
  (ii)  a reconstruction of the one real dynamic event the rules document,
        Exophase's launch-night refresh (players 638 -> 3,868): score the
        full practice cases against diluted answers E_i * g / 3868 for
        earners-growth factors g in {1.0, 1.5, 2.0};
  (iii) a scenario ensemble: synthetic final-like cases (checkpoint ~60 h
        after the first read, horizon 6,510 minutes) simulated with varied
        pool growth, dilution and per-trophy saturation, scored exactly.

Scoring is the contest's: mean pinball loss on the logit scale, rates
clamped to 0.01..99.99, lower is better. The kill criterion from the build
plan: the candidate must beat the best house proxy by >= 10% mean pinball
on the scenario ensemble; the 5-95% band should cover ~90% of truths.
"""

import importlib.util
import json
import math
import random
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
KIT_INPUT = Path.home() / "workspace/makefun/gta6-trophy-rarity-forecast-01-kit/kit/input"

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z_HOUSE = [-1.645, -0.674, 0.0, 0.674, 1.645]
Z_EXACT = [-1.6448536269514722, -0.6744897501960817, 0.0, 0.6744897501960817, 1.6448536269514722]
EPOCH = datetime(2026, 11, 19, tzinfo=timezone.utc)


def clamp(x):
    return min(99.99, max(0.01, x))


def logit(pct):
    p = clamp(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(lg):
    return 100.0 / (1.0 + math.exp(-lg)) if lg < 40 else 100.0


def parse_ts(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()


def iso(ts):
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def pinball(qs_by_id, truth_by_id):
    """Mean pinball loss on the logit scale over trophies x levels."""
    total = 0.0
    count = 0
    for tid, y in truth_by_id.items():
        if y is None:
            continue
        ly = logit(y)
        for tau, q in zip(LEVELS, qs_by_id[tid]):
            d = ly - logit(q)
            total += 2.0 * max(tau * d, (tau - 1.0) * d)
            count += 1
    return total / count


def coverage(qs_by_id, truth_by_id):
    inside = 0
    count = 0
    for tid, y in truth_by_id.items():
        if y is None:
            continue
        q = qs_by_id[tid]
        inside += 1 if q[0] <= y <= q[-1] else 0
        count += 1
    return inside / count


# ---------------------------------------------------------------- forecasters

def fc_persistence(case):
    last = case["history"][-1]
    spread = 0.5 * math.sqrt(case["horizon_minutes"] / 1440.0)
    out = {}
    for i, tr in enumerate(case["trophies"]):
        rate = last["rates"][i]
        lg = logit(rate if rate is not None else 0)
        out[tr["id"]] = [inv_logit(lg + z * spread) for z in Z_HOUSE]
    return out


def _linfit(xs, ys):
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0:
        return None
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return sxy / sxx, my - (sxy / sxx) * mx


def fc_house_curve(case):
    """Per RULES.txt: logit line vs ln(1 + hours since 2026-11-19T00:00Z)
    over the last 24 h of reads; spread = random walk combined with the
    fit residual (quadrature); fewer than 3 reads -> persistence."""
    at_ts = parse_ts(case["at"])
    target_ts = parse_ts(case["target"])
    recent = [r for r in case["history"] if parse_ts(r.get("at") or r["slot"]) >= at_ts - 24 * 3600]
    base = fc_persistence(case)
    if len(recent) < 3:
        return base
    rw = 0.5 * math.sqrt(case["horizon_minutes"] / 1440.0)
    out = {}
    for i, tr in enumerate(case["trophies"]):
        pts = []
        for r in recent:
            if r["rates"][i] is not None:
                h = (parse_ts(r.get("at") or r["slot"]) - EPOCH.timestamp()) / 3600.0
                pts.append((math.log1p(h), logit(r["rates"][i])))
        if len(pts) < 3:
            out[tr["id"]] = base[tr["id"]]
            continue
        fit = _linfit([p[0] for p in pts], [p[1] for p in pts])
        if fit is None:
            out[tr["id"]] = base[tr["id"]]
            continue
        slope, intercept = fit
        resid = [y - (intercept + slope * x) for x, y in pts]
        sd = math.sqrt(sum(v * v for v in resid) / len(resid))
        lg = intercept + slope * math.log1p((target_ts - EPOCH.timestamp()) / 3600.0)
        spread = math.sqrt(rw * rw + sd * sd)
        out[tr["id"]] = [inv_logit(lg + z * spread) for z in Z_HOUSE]
    return out


spec = importlib.util.spec_from_file_location("candidate", ROOT / "main.py")
candidate_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(candidate_mod)


def fc_candidate(case):
    fc = candidate_mod.forecast_case(case)
    return {row["id"]: row["q"] for row in fc["trophies"]}


FORECASTERS = {"candidate": fc_candidate, "persistence": fc_persistence, "house_curve": fc_house_curve}


def load_case(cid):
    return json.loads((KIT_INPUT / f"{cid}.in").read_text())


# ---------------------------------------------------------------- (i) time split

def time_split():
    print("== (i) time split on practice history (fit first half, score second half)")
    print("   NOTE: every read in both practice histories is the same page state,")
    print("   so this split can only catch format/anchoring errors, not dynamics.")
    for cid in ("p00", "p01"):
        case = load_case(cid)
        hist = case["history"]
        half = len(hist) // 2
        prefix, suffix = hist[:half], hist[half:]
        sub = dict(case)
        sub["history"] = prefix
        sub["at"] = prefix[-1]["slot"]
        sub["target"] = suffix[-1]["slot"]
        sub["horizon_minutes"] = int((parse_ts(sub["target"]) - parse_ts(sub["at"])) / 60)
        truth = {tr["id"]: suffix[-1]["rates"][i] for i, tr in enumerate(case["trophies"])}
        line = [cid]
        for name, fn in FORECASTERS.items():
            line.append(f"{name}={pinball(fn(sub), truth):.4f}")
        print("   " + "  ".join(line))


# ------------------------------------------------- (ii) refresh reconstruction

def refresh_reconstruction():
    print("== (ii) launch-night refresh reconstruction (players 638 -> 3,868)")
    for cid in ("p00", "p01"):
        case = load_case(cid)
        last = case["history"][-1]
        players0 = last["players_tracked"]
        for g in (1.0, 1.5, 2.0):
            truth = {}
            for i, tr in enumerate(case["trophies"]):
                r = last["rates"][i]
                truth[tr["id"]] = None if r is None else clamp(r * players0 * g / 3868.0)
            line = [f"{cid} g={g}"]
            for name, fn in FORECASTERS.items():
                line.append(f"{name}={pinball(fn(case), truth):.4f}")
            print("   " + "  ".join(line))


# ------------------------------------------------------- (iii) scenario ensemble

def simulate_scenario(rng, base_rates, p_max, tau_p, delta):
    """Simulate a final-like case. Returns (case, truth_by_id)."""
    t0 = datetime(2026, 11, 18, 23, 50, tzinfo=timezone.utc)
    checkpoint_h = 60.0
    horizon_h = 6510.0 / 60.0
    dt = 0.25
    steps = int((checkpoint_h + horizon_h) / dt)
    n = len(base_rates)
    # Per-trophy saturation ceilings and progress rates.
    ceilings = []
    rates_k = []
    for r in base_rates:
        s0 = (r if r is not None else 0.05) / 100.0
        mult = rng.choice([1.5, 3.0, 8.0])
        ceilings.append(min(0.98, max(s0, s0 * mult)))
        rates_k.append(1.0 / rng.choice([18.0, 48.0, 120.0]))
    s = [(r if r is not None else 0.05) / 100.0 for r in base_rates]
    p = 638.0
    # Page state: refreshed every 3-7 h; reads sample it every 30 min.
    page_rates = list(base_rates)
    page_players = 638
    next_refresh = rng.uniform(3.0, 7.0)
    history = []
    truth = None
    for step in range(steps + 1):
        t = step * dt
        # True dynamics.
        p_new = 638.0 + (p_max - 638.0) * (1.0 - math.exp(-t / tau_p))
        dp = p_new - p
        if p_new > 0 and dp > 0:
            for i in range(n):
                s[i] = s[i] * (p + delta * dp) / p_new
        p = p_new
        for i in range(n):
            s[i] += rates_k[i] * (ceilings[i] - s[i]) * dt
            s[i] = min(max(s[i], 0.0), 0.9999)
        if t >= next_refresh:
            page_rates = [round(100.0 * v, 2) for v in s]
            page_players = int(round(p))
            next_refresh = t + rng.uniform(3.0, 7.0)
        if step % 2 == 0 and t <= checkpoint_h:  # a read every 30 min
            slot = t0 + timedelta(hours=t)
            history.append({
                "slot": iso(slot.timestamp()),
                "at": iso(slot.timestamp()),
                "players_tracked": page_players,
                "trophies_earned": int(round(sum(v for v in page_rates if v) * page_players / 100.0)),
                "rates": list(page_rates),
            })
        if abs(t - (checkpoint_h + horizon_h)) < dt / 2:
            lag = rng.uniform(0.0, 3.0)
            truth = {}
            # Recompute the state `lag` hours before the target approximately
            # by re-deriving from s with a small step back is overkill; the
            # answer is the page state at the target, i.e. the last refresh.
            truth = {i + 1: page_rates[i] for i in range(n)}
    at = t0 + timedelta(hours=checkpoint_h)
    target = at + timedelta(minutes=6510)
    case = {
        "schema": "makefun/gta6-trophies-case/v1",
        "id": "sim",
        "at": iso(at.timestamp()),
        "horizon_minutes": 6510,
        "target": iso(target.timestamp()),
        "levels": LEVELS,
        "game": {},
        "trophies": [{"id": i + 1} for i in range(n)],
        "history": history,
        "notes": "simulated",
    }
    return case, truth


def scenario_ensemble():
    print("== (iii) scenario ensemble (24 simulated final-like cases)")
    base = load_case("p01")
    base_rates = base["history"][-1]["rates"]
    trophies = base["trophies"]
    totals = {name: [] for name in FORECASTERS}
    covs = []
    idx = 0
    for p_max in (6000.0, 20000.0, 60000.0, 200000.0):
        for delta in (0.25, 0.5, 0.75):
            for tau_p in (36.0, 96.0):
                rng = random.Random(20261119 + idx)
                idx += 1
                case, truth = simulate_scenario(rng, base_rates, p_max, tau_p, delta)
                # Truth ids in the sim are 1..55 in trophy order.
                truth_by_id = {trophies[i]["id"]: truth[i + 1] for i in range(len(trophies))}
                for name, fn in FORECASTERS.items():
                    totals[name].append(pinball(fn(case), truth_by_id))
                covs.append(coverage(fc_candidate(case), truth_by_id))
    for name, vals in totals.items():
        print(f"   {name}: mean pinball = {sum(vals) / len(vals):.4f}  (n={len(vals)})")
    best_house = min(sum(totals["persistence"]) / 24, sum(totals["house_curve"]) / 24)
    cand = sum(totals["candidate"]) / 24
    margin = (best_house - cand) / best_house * 100.0
    print(f"   candidate margin vs best house proxy: {margin:+.1f}%  (kill criterion: >= +10%)")
    print(f"   candidate 5-95% band coverage: {sum(covs) / len(covs):.3f}  (target ~0.90)")
    return cand, best_house, margin


if __name__ == "__main__":
    time_split()
    refresh_reconstruction()
    scenario_ensemble()
