"""gta6-trophy-rarity-forecast-01 entry: a hybrid quantile forecaster.

For every trophy in a case, forecast its Exophase earn rate (percent of the
players tracked for the game who have earned it) at the case's target, as five
quantiles on the levels the contest scores. Python 3.13, standard library
only, no network, deterministic: two runs on the same case write the same
bytes.

Why a hybrid. A trophy's rate is a ratio, earners / players_tracked, and the
two sides move differently. The tracked pool grows as the game sells and as
Exophase refreshes its counts, and a refresh can multiply the pool at a jump,
diluting every rate at once (launch night: 638 players tracked for hours,
then 3,868 at the next refresh, every rate lower). Earners, in contrast,
accumulate along a saturating curve: story trophies fill in over days, the
platinum over weeks. A forecaster that only extends the rate itself confuses
the two motions, so this program models them separately and divides at the
target:

  1. a parametric decomposition: a growth fit for players_tracked and, per
     trophy, a saturating fit for earners = rate x players / 100, both over
     the case's distinct page states (the page only changes at a refresh, so
     repeated identical reads are one state);
  2. a house-style curve: a least-squares line of the trophy's logit rate
     against ln(1 + hours since launch) over the last 24 hours of reads,
     read at the target;
  3. persistence: the last read's rate, the anchor when the recorded history
     is short or flat.

The three medians are averaged on the logit scale. The spread is built
from the case's own history: the logit moves observed between its
refreshes, scaled by how many refreshes are expected before the target,
plus the random-walk / drift / disagreement terms. When the history is
degenerate (fewer than 3 distinct page states) a refresh prior is
injected instead of silently returning the last pool -- the launch-night
failure, where one flat state hid a refresh that multiplied the pool.
Earners are modelled as integers and divided by the projected pool, so a
refresh's dilution is a division, not a distortion of the earner curve.
"""

import json
import math
import os
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantile at each level.
Z = [-1.6448536269514722, -0.6744897501960817, 0.0, 0.6744897501960817, 1.6448536269514722]
# The house curve baseline measures time from the launch midnight.
EPOCH_ISO = "2026-11-19T00:00:00Z"

# Ensemble weights for the three median components (renormalised over the
# components a case actually has). With a rich history (a day or more of
# distinct page states) the curve and the decomposition have something to
# hold on to and persistence is demoted; with a thin one it anchors.
W_PARAM = 0.50
W_CURVE = 0.30
W_PERS = 0.20
W_PARAM_RICH = 0.32
W_CURVE_RICH = 0.53
W_PERS_RICH = 0.15
# Incoming players between the last read and the target have had less time
# to earn: their earn fraction is modelled as DILUTION_DELTA times the
# progressed fraction of the players already tracked.
DILUTION_DELTA = 0.5
# Spread calibration: the random-walk scale (the house uses 0.5 logit units
# per root day) is multiplied by RW_SCALE; predicted movement and component
# disagreement add in quadrature; the empirical per-refresh jump spread
# (measured on this case's own history) adds in quadrature too, scaled by
# the number of refreshes expected before the target.
RW_SCALE = 0.50
DRIFT_W = 0.25
SIGMA_FLOOR = 0.30
SIGMA_CAP = 4.00
# Overall spread scale, set by the coverage calibration in tools/validate.py
# (target 0.85-0.95 coverage of the 5-95% band on the scenario ensemble).
SIGMA_SCALE = 1.0
# Refresh-process priors, used ONLY when the history is degenerate (fewer
# than 3 distinct page states). Round judgment priors, not fitted values:
# one refresh is assumed before the target, the pool roughly triples at
# it, earners grow a quarter, and the spread floor covers a refresh-sized
# move. With a rich history none of them fires: cadence, jump size and
# jump spread are all estimated from the case itself.
REFRESH_PRIOR_POOL = 3.0
REFRESH_PRIOR_EARNERS = 1.25
REFRESH_PRIOR_SIGMA = 1.0
# Asymmetric fan: TESTED and REJECTED (tools/validate.py ablation,
# 21 Nov): widening the lower tail (1.5x) cost +0.026 mean pinball on
# the 48-case ensemble (0.2252 vs 0.1988 symmetric) -- the median
# already carries the expected dilution, so the fan only adds cost.
# Kept symmetric (1.0) here as a record of the test.
ASYM_LO = 1.0
ASYM_HI = 1.0
# Saturating-curve time constants (hours) tried for earners and players.
TAUS = [6.0, 12.0, 24.0, 48.0, 96.0, 192.0, 384.0, 768.0, 1536.0]


def clamp_rate(x):
    return min(99.99, max(0.01, x))


def logit(pct):
    p = clamp_rate(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(lg):
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg)) if lg < 40 else 100.0
    e = math.exp(lg) if lg > -40 else 0.0
    return 100.0 * e / (1.0 + e)


def round4(x):
    """Round to 4 decimals, half up; whole numbers come back as int."""
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000.0
    return int(value) if float(value).is_integer() else value


def parse_ts(text):
    """Seconds since the epoch for an ISO-8601 UTC timestamp."""
    from datetime import datetime, timezone

    dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


EPOCH_TS = parse_ts(EPOCH_ISO)


def linfit(xs, ys):
    """Least-squares (slope, intercept); None when x has no variance."""
    n = len(xs)
    if n < 2:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0:
        return None
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    slope = sxy / sxx
    return slope, my - slope * mx


def saturating_fit(ts, vs, t_target):
    """Fit v(t) = V - (V - v_first) * exp(-(t - t_first)/tau) over a tau grid.

    Given tau the model is linear in V, so V is solved by least squares and
    clamped to at least the largest observed v. Returns the prediction at
    t_target, or None when there is nothing to fit.
    """
    if not ts:
        return None
    if len(ts) == 1:
        return vs[0]
    t0 = ts[0]
    v0 = vs[0]
    vmax = max(vs)
    best = None
    for tau in TAUS:
        num = 0.0
        den = 0.0
        for t, v in zip(ts, vs):
            u = math.exp(-(t - t0) / tau)
            x = 1.0 - u
            num += x * (v - v0 * u)
            den += x * x
        if den <= 1e-12:
            continue
        v_inf = max(num / den, vmax)
        sse = 0.0
        for t, v in zip(ts, vs):
            u = math.exp(-(t - t0) / tau)
            pred = v_inf - (v_inf - v0) * u
            sse += (v - pred) ** 2
        if best is None or sse < best[0]:
            best = (sse, v_inf, tau)
    if best is None:
        return vs[-1]
    _, v_inf, tau = best
    u = math.exp(-(t_target - t0) / tau)
    return v_inf - (v_inf - v0) * u


def logtime_fit(ts, vs, t_target):
    """Fit v = a + b * ln(1 + (t - t_first)); a falling fit is flattened."""
    if not ts:
        return None
    if len(ts) == 1:
        return vs[0]
    t0 = ts[0]
    xs = [math.log1p(t - t0) for t in ts]
    fit = linfit(xs, vs)
    if fit is None:
        return vs[-1]
    slope, intercept = fit
    if slope < 0:
        return sum(vs) / len(vs)
    return intercept + slope * math.log1p(t_target - t0)


def median3(a, b, c):
    return sorted([a, b, c])[1]


def refresh_stats(states, t_target):
    """Refresh cadence and per-refresh pool jump, from the distinct states.

    Returns (cadence_hours, jump_factor, expected_refreshes) or
    (None, None, 0.0) when the history shows fewer than two pool changes.
    The pool only moves at a refresh, so the gaps between states that
    change the pool are the refresh cadence, and the ratios between
    successive pools are the per-refresh jumps. Both are measured on
    the RECENT states: early jumps are the big ones, and compounding a
    launch-night jump over the whole horizon would explode the pool.
    """
    ts = [s[0] for s in states]
    ps = [float(s[1]) for s in states]
    recent = list(zip(ts, ps))[-6:]
    gaps = []
    jumps = []
    for k in range(1, len(recent)):
        if recent[k][1] != recent[k - 1][1]:
            gaps.append(recent[k][0] - recent[k - 1][0])
            if recent[k - 1][1] > 0:
                jumps.append(recent[k][1] / recent[k - 1][1])
    if not gaps or not jumps:
        return None, None, 0.0
    gaps.sort()
    jumps.sort()
    cadence = gaps[len(gaps) // 2]
    jump = jumps[len(jumps) // 2]
    n_ref = max(0.0, (t_target - ts[-1]) / cadence) if cadence > 0 else 0.0
    return cadence, jump, n_ref


def forecast_players(states, t_target):
    """Forecast players_tracked at t_target from the distinct states.

    Saturating fit and log-log growth fit, combined by median with the
    last pool as the anchor. A forward jump projection (pool multiplied
    by the median recent per-refresh jump for each expected refresh)
    was TESTED and REJECTED (tools/validate.py ablation, 21 Nov): it
    lost on 47 of 48 ensemble cases (0.1988 vs 0.1944) -- compounding
    any per-refresh multiplier over ~21 refreshes explodes the pool,
    and the saturating fit already carries the growth. The refresh
    cadence/jump statistics are still used, for the spread (below).
    """
    ts = [s[0] for s in states]
    ps = [float(s[1]) for s in states]
    p_last = ps[-1]
    if len(set(ps)) < 2:
        return p_last
    sat = saturating_fit(ts, ps, t_target)
    # Log-log growth: ln P against ln(1 + hours since the first state).
    xs = [math.log1p(t - ts[0]) for t in ts]
    fit = linfit(xs, [math.log(p) for p in ps])
    loglog = None
    if fit is not None:
        slope, intercept = fit
        slope = min(slope, 2.0)
        loglog = math.exp(intercept + slope * math.log1p(t_target - ts[0]))
    cands = [v for v in (sat, loglog) if v is not None]
    if not cands:
        return p_last
    pred = median3(cands[0], cands[-1], p_last) if len(cands) == 2 else cands[0]
    return min(max(pred, p_last), 80.0 * p_last)


def forecast_case(case):
    if case["levels"] != LEVELS:
        raise ValueError("unexpected levels")
    history = case["history"]
    if not history:
        raise ValueError("case has no history")
    trophies = case["trophies"]
    n = len(trophies)

    # Read times in hours; the checkpoint and target come from the case.
    t_at = parse_ts(case["at"]) / 3600.0
    t_target = parse_ts(case["target"]) / 3600.0
    horizon_days = max(case["horizon_minutes"], 0) / 1440.0

    # Distinct page states: the page changes only at a refresh, so a run of
    # identical reads is a single observation of it.
    states = []  # (hours, players, rates tuple)
    for read in history:
        t = parse_ts(read.get("at") or read["slot"]) / 3600.0
        rates = tuple(read["rates"])
        key = (read["players_tracked"], rates)
        if states and states[-1][3] == key:
            continue
        states.append((t, read["players_tracked"], rates, key))
    t_first = states[0][0]
    t_last = states[-1][0]
    span = t_last - t_first

    p_target = forecast_players(states, t_target)

    # Degeneracy guard: with fewer than 3 distinct page states the
    # history carries no refresh signal at all, and silently returning
    # the last pool (the launch-night failure) is the worst forecast.
    # Inject a refresh prior instead: one refresh before the target.
    degenerate = len(states) < 3
    if degenerate:
        p_target = max(p_target, float(states[-1][1]) * REFRESH_PRIOR_POOL)
    _, _, n_ref = refresh_stats(states, t_target)

    # Observed inter-refresh jump spread, from this case's own history:
    # the variance of the recent log pool jumps. (A refresh's dilution
    # moves every rate's logit by about -d ln P, so the pool-jump
    # variance IS the jump distribution's common component; the
    # per-trophy dispersion around it is time drift, already covered
    # by the random-walk term -- adding it per refresh double-counted
    # and cost +0.01 pinball in ablation.) Scaled by the number of
    # refreshes expected before the target, and CAPPED at 0.09 variance
    # (sigma_jump <= 0.30): uncapped, the scaling explodes on a
    # saturated pool (the earlier n_eff version reached sigma 1.4 and
    # cost +0.13 pinball). A saturated pool contributes almost none
    # (its recent jumps are near zero).
    ln_jumps = []
    recent = states[-6:] if len(states) > 1 else states
    for k in range(1, len(recent)):
        if recent[k][1] > 0 and recent[k - 1][1] > 0 and recent[k][1] != recent[k - 1][1]:
            ln_jumps.append(math.log(recent[k][1] / recent[k - 1][1]))
    if len(ln_jumps) >= 2:
        mu_lnj = sum(ln_jumps) / len(ln_jumps)
        var_lnj = sum((x - mu_lnj) ** 2 for x in ln_jumps) / len(ln_jumps)
    else:
        var_lnj = 0.0
    jump_var = min(n_ref * var_lnj, 0.09) if n_ref > 0 else 0.0

    # House-style curve points: reads in the last 24 hours before the
    # checkpoint, time measured in hours since the launch midnight.
    curve_states = [s for s in states if s[0] >= t_at - 24.0]

    rows = []
    for i in range(n):
        # This trophy's observations over the distinct states.
        obs = [(s[0], s[1], s[2][i]) for s in states if s[2][i] is not None]
        if not obs:
            rows.append({"id": trophies[i]["id"], "q": [0.05, 0.08, 0.12, 0.2, 0.4]})
            continue
        last_rate = clamp_rate(obs[-1][2])
        lg_pers = logit(last_rate)

        # Component 1: parametric earners / players decomposition. The
        # earned fraction s = rate / 100 is fitted as a saturating curve
        # (concave, unlike the earners series itself, which is the
        # product of two saturating curves and is convex early); earners
        # at the target are then the INTEGER count E = round(s * P), and
        # the rate is E / P_target -- a refresh's dilution is a division,
        # not a distortion of the earner curve. Players who arrive after
        # the last read carry a lower earned fraction (DILUTION_DELTA).
        lg_param = None
        ets = [o[0] for o in obs]
        svs = [o[2] / 100.0 for o in obs]
        s_sat = saturating_fit(ets, svs, t_target)
        s_log = logtime_fit(ets, svs, t_target)
        s_cands = [v for v in (s_sat, s_log) if v is not None]
        e_last = round(obs[-1][2] * obs[-1][1] / 100.0)
        p_last = float(obs[-1][1])
        if s_cands:
            if degenerate:
                # No fit has anything to hold on to: the refresh prior.
                e_target = float(e_last) * REFRESH_PRIOR_EARNERS
            else:
                s_target = median3(s_cands[0], s_cands[-1], svs[-1]) if len(s_cands) == 2 else s_cands[0]
                s_target = min(max(s_target, svs[-1]), 0.9999)
                e_target = float(max(round(s_target * p_target), e_last))
            if p_target > p_last > 0:
                share_old = p_last / p_target
                e_target *= DILUTION_DELTA + (1.0 - DILUTION_DELTA) * share_old
            if p_target > 0:
                s_out = min(max(e_target / p_target, 0.0001), 0.9999)
                lg_param = logit(100.0 * s_out)

        # Component 2: house-style curve on the logit rate.
        lg_curve = None
        cobs = [(s[0], s[2][i]) for s in curve_states if s[2][i] is not None]
        if len(cobs) >= 3:
            xs = [math.log1p(t - EPOCH_TS / 3600.0) for t, _ in cobs]
            ys = [logit(r) for _, r in cobs]
            fit = linfit(xs, ys)
            if fit is not None:
                slope, intercept = fit
                lg_curve = intercept + slope * math.log1p(t_target - EPOCH_TS / 3600.0)

        # Ensemble on the logit scale.
        if span >= 24.0 and len(states) >= 6:
            w_param, w_curve, w_pers = W_PARAM_RICH, W_CURVE_RICH, W_PERS_RICH
        else:
            w_param, w_curve, w_pers = W_PARAM, W_CURVE, W_PERS
        comps = [(lg_pers, w_pers)]
        if lg_param is not None:
            comps.append((lg_param, w_param))
        if lg_curve is not None:
            comps.append((lg_curve, w_curve))
        wsum = sum(w for _, w in comps)
        lg_med = sum(lg * w for lg, w in comps) / wsum
        # Shrink toward persistence when the distinct history is short: the
        # fits have little to hold on to. The degeneracy guard is exempt:
        # the refresh prior replaces the fits there, not persistence.
        if span < 24.0 and not degenerate:
            shrink = 0.5 + 0.5 * (span / 24.0)
            lg_med = lg_pers + shrink * (lg_med - lg_pers)
        var = sum(w * (lg - lg_med) ** 2 for lg, w in comps) / wsum
        disagree = math.sqrt(var)
        drift = abs(lg_med - lg_pers)

        sigma = math.sqrt((RW_SCALE * 0.5 * math.sqrt(horizon_days)) ** 2
                          + (DRIFT_W * drift) ** 2 + disagree ** 2 + jump_var)
        sigma = min(max(sigma, SIGMA_FLOOR), SIGMA_CAP)
        if degenerate:
            sigma = max(sigma, REFRESH_PRIOR_SIGMA)
        sigma *= SIGMA_SCALE
        sigma = min(max(sigma, SIGMA_FLOOR), SIGMA_CAP)
        # Asymmetric fan: refreshes only dilute rates downward, so the
        # lower tail is wider than the upper one.
        qs = [round4(inv_logit(lg_med + z * sigma
                               * (ASYM_LO if z < 0 else ASYM_HI if z > 0 else 1.0)))
              for z in Z]
        rows.append({"id": trophies[i]["id"], "q": qs})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        out = json.dumps(forecast, separators=(",", ":")) + "\n"
        (OUTPUT_DIR / f"{path.stem}.out").write_text(out, encoding="utf-8")


if __name__ == "__main__":
    main()
