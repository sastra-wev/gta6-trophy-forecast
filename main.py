#!/usr/bin/env python3
"""GTA VI trophy rarity forecaster (make.fun gta6-trophy-rarity-forecast-01).

Reads every *.in case in $MAKEFUN_INPUT_DIR (else /input) and writes one
<id>.out per case to $MAKEFUN_OUTPUT_DIR (else /output): for each trophy,
five quantiles (0.05/0.25/0.5/0.75/0.95) of its Exophase earn rate, in
percent, at the case's target instant.

Model, per trophy, on the logit scale the scorer uses:

  * The observed page is a step function: Exophase refreshes its counts on
    its own schedule, so consecutive identical reads are one observation,
    not many.  All fits run on the distinct page states.
  * Pool: log players_tracked is fit against u = ln(1 + hours since
    2026-11-19T00:00Z) (recent window blended with the whole history) and
    extrapolated from the last read.  The pool grows fast, then bends.
  * Earners: earners = rate * players_tracked / 100 is fit the same way,
    per trophy, with the slope shrunk toward the trophy's tier slope
    (hierarchical shrinkage: thin data borrows the tier's trajectory).
  * The median blends two extrapolations on the logit scale: the
    earners/pool decomposition (which prices pool dilution honestly) and
    a direct logit(rate) trend fit.  Degenerate series (never observed,
    stuck at zero earners, pinned at ~100%) fall back to tier priors.
  * The spread is a per-trophy logit sigma: a horizon term wider than the
    house random walk's, plus the fits' residual volatility with an
    extrapolation-leverage factor, plus a regime term that widens when
    the trailing page state has been frozen for hours (a refresh is due
    and the target read will not show the checkpoint's stale page).

Standard library only, fully deterministic, no randomness anywhere.
"""

import glob
import json
import math
import os
import sys

# ---------------------------------------------------------------- time ----

def _days_from_civil(y, m, d):
    """Days since 1970-01-01 for a Gregorian date (Hinnant's algorithm)."""
    y -= 1 if m <= 2 else 0
    era = (y if y >= 0 else y - 399) // 400
    yoe = y - era * 400
    doy = (153 * (m - 3 if m > 2 else m + 9) + 2) // 5 + d - 1
    doe = yoe * 365 + yoe // 4 - yoe // 100 + doy
    return era * 146097 + doe - 719468


def _iso_to_minutes(s):
    """Parse '2026-11-19T03:00:00Z' / '...T23:50:49.360Z' to epoch minutes."""
    y = int(s[0:4]); mo = int(s[5:7]); d = int(s[8:10])
    hh = int(s[11:13]); mi = int(s[14:16])
    rest = s[17:].rstrip('Zz')
    sec = float(rest) if rest else 0.0
    return _days_from_civil(y, mo, d) * 1440.0 + hh * 60.0 + mi + sec / 60.0


ANCHOR_MIN = _iso_to_minutes('2026-11-19T00:00:00Z')


def _u_of(iso):
    """u = ln(1 + hours since 2026-11-19T00:00Z), the house curve's clock."""
    h = (_iso_to_minutes(iso) - ANCHOR_MIN) / 60.0
    return math.log(1.0 + max(h, -0.99))


# --------------------------------------------------------------- maths ----

def _clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


def _logit(pct):
    p = _clamp(pct, 0.01, 99.99) / 100.0
    return math.log(p / (1.0 - p))


def _inv_logit(x):
    if x >= 0:
        return 100.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return 100.0 * e / (1.0 + e)


def _norm_ppf(p):
    """Inverse standard normal CDF (Acklam's rational approximation)."""
    p = _clamp(p, 1e-9, 1.0 - 1e-9)
    a = [-3.969683028665376e+01, 2.209460984245205e+02,
         -2.759285104469687e+02, 1.383577518672690e+02,
         -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02,
         -1.556989798598866e+02, 6.680131188771972e+01,
         -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01,
         -2.400758277161838e+00, -2.549732539343734e+00,
         4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01,
         2.445134137142996e+00, 3.754408661907416e+00]
    plow = 0.02425
    if p < plow:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    if p <= 1.0 - plow:
        q = p - 0.5
        r = q * q
        return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
               (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    q = math.sqrt(-2.0 * math.log(1.0 - p))
    return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)


def _wls(xs, ys, ws):
    """Weighted least squares y ~ x.  Returns (slope, resid_sd, n_eff)."""
    sw = sum(ws)
    if sw <= 0.0 or len(xs) < 2:
        return None
    mx = sum(w * x for x, w in zip(xs, ws)) / sw
    my = sum(w * y for y, w in zip(ys, ws)) / sw
    sxx = sum(w * (x - mx) ** 2 for x, w in zip(xs, ws))
    neff = sw * sw / sum(w * w for w in ws)
    if sxx <= 1e-12:
        return (0.0, 0.0, neff)
    sxy = sum(w * (x - mx) * (y - my) for x, y, w in zip(xs, ys, ws))
    slope = sxy / sxx
    inter = my - slope * mx
    var = sum(w * (y - (inter + slope * x)) ** 2 for x, y, w in zip(xs, ys, ws)) / sw
    if neff > 1.05:
        var *= neff / (neff - 1.0)
    return (slope, math.sqrt(max(var, 0.0)), neff)


def _recency_weights(ts, t_last, half_life_h):
    return [0.5 ** ((t_last - t) / half_life_h) for t in ts]


# ------------------------------------------------------------- priors -----

TIERS = ('platinum', 'gold', 'silver', 'bronze')
EARN_SLOPE_PRIOR = {'platinum': 0.45, 'gold': 0.90, 'silver': 1.20, 'bronze': 1.50}
LOGIT_SLOPE_PRIOR = {'platinum': -0.40, 'gold': 0.05, 'silver': 0.35, 'bronze': 0.65}
POOL_SLOPE_PRIOR = 0.85
EMERGENCE_BASE = {'platinum': 0.03, 'gold': 0.08, 'silver': 0.14, 'bronze': 0.22}


def _emergence(tier, d_days):
    """Median guess (percent) for a trophy with no earners yet by target."""
    base = EMERGENCE_BASE.get(tier, 0.15)
    return base * min(1.0, 0.35 + d_days / 5.0)


# -------------------------------------------------------------- model -----

def _fit_slope(points, t_last, window_h, half_life_h):
    """points: list of (t_hours, u, y).  Fit y ~ u on the recent window
    (falling back to all points); returns (slope, resid_sd, n_distinct_y)
    or (None, None, n_distinct_y)."""
    if not points:
        return (None, None, 0)
    ndist = len({round(p[2], 9) for p in points})
    win = [p for p in points if t_last - p[0] <= window_h]
    use = win if len(win) >= 2 else points
    if len(use) < 2:
        return (0.0, None, ndist)
    ws = _recency_weights([p[0] for p in use], t_last, half_life_h)
    fit = _wls([p[1] for p in use], [p[2] for p in use], ws)
    if fit is None:
        return (0.0, None, ndist)
    return (fit[0], fit[1], ndist)


def forecast_case(case):
    trophies = case['trophies']
    hist = case.get('history') or []
    levels = case.get('levels') or [0.05, 0.25, 0.5, 0.75, 0.95]
    d_days = float(case['horizon_minutes']) / 1440.0
    u_tgt = _u_of(case['target'])
    sqrt_d = math.sqrt(max(d_days, 0.0))

    if hist:
        t_last = (_iso_to_minutes(hist[-1]['at']) - ANCHOR_MIN) / 60.0
        u_last = _u_of(hist[-1]['at'])
        p_last = float(hist[-1]['players_tracked'])
    else:
        t_last = (_iso_to_minutes(case['at']) - ANCHOR_MIN) / 60.0
        u_last = _u_of(case['at'])
        p_last = 0.0
    du = u_tgt - u_last

    # Distinct page states: consecutive reads with identical content are a
    # single observation (the page simply had not refreshed between them).
    state_idx = []
    prev_key = None
    for i, r in enumerate(hist):
        key = (r.get('players_tracked'), r.get('trophies_earned'),
               tuple(r.get('rates') or ()))
        if key != prev_key:
            state_idx.append(i)
            prev_key = key
    states = [hist[i] for i in state_idx]
    state_t = [(_iso_to_minutes(r['at']) - ANCHOR_MIN) / 60.0 for r in states]
    state_u = [_u_of(r['at']) for r in states]
    u_span = (state_u[-1] - state_u[0]) if len(state_u) >= 2 else 0.0

    # Trailing frozen run, in hours: how long the page has shown exactly
    # the last state (a refresh is due; the target read shows a later one).
    stale_h = 0.0
    if len(hist) >= 2:
        lk = (hist[-1].get('players_tracked'), hist[-1].get('trophies_earned'))
        for i in range(len(hist) - 2, -1, -1):
            if (hist[i].get('players_tracked'), hist[i].get('trophies_earned')) == lk:
                stale_h = ((_iso_to_minutes(hist[-1]['at'])
                            - _iso_to_minutes(hist[i]['at'])) / 60.0)
            else:
                break
    regime = min(0.85, 0.16 * math.sqrt(stale_h)) * (1.0 if d_days >= 1.0 else 0.6)

    # ---- pool slope (log players vs u), recent window blended with all --
    b_pool = POOL_SLOPE_PRIOR
    if len(states) >= 2 and p_last > 0:
        pts = [(state_t[k], state_u[k], math.log(float(states[k]['players_tracked'])))
               for k in range(len(states)) if float(states[k]['players_tracked']) > 0]
        b_all, _, _ = _fit_slope(pts, t_last, 1e9, 24.0)
        b_rec, _, _ = _fit_slope(pts, t_last, 36.0, 24.0)
        if b_all is not None and b_rec is not None:
            b_pool = _clamp(0.65 * b_rec + 0.35 * b_all, 0.0, 2.5)
        elif b_all is not None:
            b_pool = _clamp(b_all, 0.0, 2.5)
    ln_p_tgt = math.log(p_last) + b_pool * du if p_last > 0 else None
    p_tgt = math.exp(ln_p_tgt) if ln_p_tgt is not None else None

    # ---- first pass: per-trophy own fits --------------------------------
    n_tr = len(trophies)
    own = []
    for j in range(n_tr):
        # last non-null rate, scanning raw reads backwards
        last_rate = None
        e_last = 0.0
        for i in range(len(hist) - 1, -1, -1):
            rates = hist[i].get('rates') or []
            if j < len(rates) and rates[j] is not None:
                last_rate = float(rates[j])
                e_last = last_rate * float(hist[i]['players_tracked']) / 100.0
                break
        e_pts = []   # (t, u, ln(e + 0.5)) for earners > 0
        l_pts = []   # (t, u, logit(rate)) for every observed rate
        for k, r in enumerate(states):
            rates = r.get('rates') or []
            if j >= len(rates) or rates[j] is None:
                continue
            rate = float(rates[j])
            pool = float(r['players_tracked'])
            e = rate * pool / 100.0
            if e > 0.0:
                e_pts.append((state_t[k], state_u[k], math.log(e + 0.5)))
            l_pts.append((state_t[k], state_u[k], _logit(rate)))
        b_e, s_e, nd_e = _fit_slope(e_pts, t_last, 48.0, 24.0)
        b_l, s_l, nd_l = _fit_slope(l_pts, t_last, 72.0, 30.0)
        own.append({
            'last_rate': last_rate, 'e_last': e_last,
            'b_e': b_e, 's_e': s_e, 'nd_e': nd_e,
            'b_l': b_l, 's_l': s_l, 'nd_l': nd_l,
        })

    # ---- tier slopes (hierarchical shrinkage targets) --------------------
    def tier_mean(key, prior):
        out = {}
        for tier in TIERS:
            num = 0.0
            den = 0.0
            for j in range(n_tr):
                if trophies[j].get('tier') != tier:
                    continue
                o = own[j]
                if o[key] is not None and o['e_last'] >= 5.0:
                    w = math.sqrt(o['e_last'])
                    num += w * o[key]
                    den += w
            out[tier] = (num / den) if den > 0 else None
        # fill missing tiers from the global weighted mean, then the prior
        gnum = sum(math.sqrt(own[j]['e_last']) * own[j][key]
                   for j in range(n_tr)
                   if own[j][key] is not None and own[j]['e_last'] >= 5.0)
        gden = sum(math.sqrt(own[j]['e_last']) for j in range(n_tr)
                   if own[j][key] is not None and own[j]['e_last'] >= 5.0)
        gmean = (gnum / gden) if gden > 0 else None
        for tier in TIERS:
            if out[tier] is None:
                out[tier] = gmean if gmean is not None else prior[tier]
        return out

    tier_be = tier_mean('b_e', EARN_SLOPE_PRIOR)
    tier_bl = tier_mean('b_l', LOGIT_SLOPE_PRIOR)

    # ---- second pass: medians and spreads --------------------------------
    base_sig = 0.42 * sqrt_d + 0.12
    floor_sig = 0.42 * sqrt_d + 0.10
    rows = []
    for j in range(n_tr):
        tr = trophies[j]
        tier = tr.get('tier', 'bronze')
        o = own[j]
        last = o['last_rate']

        if last is None:
            mu = _logit(_emergence(tier, d_days))
            sig = 1.7
        elif last >= 99.5:
            mu = _logit(min(99.97, last + (100.0 - last) * 0.35))
            sig = 0.30 + 0.12 * sqrt_d + 0.3 * regime
        elif o['e_last'] <= 0.0:
            mu = _logit(_emergence(tier, d_days))
            sig = 1.7
        else:
            # shrinkage weights from the amount of real movement observed
            w_e = min(0.85, max(0.0, o['nd_e'] - 1) / 6.0)
            if o['e_last'] < 3.0:
                w_e *= 0.4
            w_l = min(0.85, max(0.0, o['nd_l'] - 1) / 6.0)
            b_e = tier_be[tier] if o['b_e'] is None else \
                w_e * o['b_e'] + (1.0 - w_e) * tier_be[tier]
            b_l = tier_bl[tier] if o['b_l'] is None else \
                w_l * o['b_l'] + (1.0 - w_l) * tier_bl[tier]
            b_e = _clamp(b_e, -0.2, 3.0)
            b_l = _clamp(b_l, -2.5, 2.5)
            # Progression bends: a slope fitted on the recent window is the
            # steep part of an S-curve, and reading it straight out to the
            # target overshoots badly.  Damp both slopes by a factor that
            # shrinks as the extrapolation lengthens.
            damp = 1.7 / (1.7 + du)
            b_e *= damp
            b_l *= damp

            ln_e_tgt = math.log(o['e_last'] + 0.5) + b_e * du
            e_tgt = max(math.exp(ln_e_tgt) - 0.5, 0.0)
            mu_dir = _logit(last) + b_l * du
            # ceilings: a week-one rate cannot run away from its day-2.5
            # level without bound; cap each component's implied rate
            ceil_dec = min(98.0, max(2.2 * last, last + 25.0))
            ceil_dir = min(98.0, max(2.6 * last, last + 30.0))
            if _inv_logit(mu_dir) > ceil_dir:
                mu_dir = _logit(ceil_dir)
            if p_tgt:
                rate_dec = _clamp(100.0 * e_tgt / p_tgt, 0.01, 99.99)
                mu_dec = _logit(min(rate_dec, ceil_dec))
                mu = 0.5 * (mu_dec + mu_dir)
            else:
                mu = mu_dir
            if o['e_last'] < 3.0:
                mu = 0.45 * _logit(_emergence(tier, d_days)) + 0.55 * mu
            # soft-clamp wild extrapolations away from the last rate
            anchor = _logit(last)
            dev = mu - anchor
            if abs(dev) > 3.2:
                dev = math.copysign(3.2 + 0.25 * (abs(dev) - 3.2), dev)
                mu = anchor + _clamp(dev, -4.2, 4.2)

            s_obs_raw = 0.5 * (o['s_l'] if o['s_l'] is not None else 0.35) \
                      + 0.5 * (o['s_e'] if o['s_e'] is not None else 0.40)
            leverage = min(1.8, 1.0 + 0.5 * du / max(u_span, 0.5))
            obs = min(1.15, 0.75 * s_obs_raw * leverage)
            sig = math.sqrt(base_sig ** 2 + obs ** 2 + regime ** 2)
            sig = _clamp(max(sig, floor_sig), 0.05, 2.3)

        if tr.get('secret'):
            sig *= 1.08
        sig = min(sig, 2.45)

        qs = []
        for tau in levels:
            q = _inv_logit(mu + _norm_ppf(float(tau)) * sig)
            qs.append(round(_clamp(q, 0.0, 100.0), 4))
        for k in range(1, len(qs)):
            if qs[k] < qs[k - 1]:
                qs[k] = qs[k - 1]
        rows.append({'id': tr['id'], 'q': qs})

    return {'trophies': rows}


# --------------------------------------------------------------- main -----

def main():
    in_dir = os.environ.get('MAKEFUN_INPUT_DIR') or '/input'
    out_dir = os.environ.get('MAKEFUN_OUTPUT_DIR') or '/output'
    os.makedirs(out_dir, exist_ok=True)
    for path in sorted(glob.glob(os.path.join(in_dir, '*.in'))):
        with open(path, 'r', encoding='utf-8') as f:
            case = json.load(f)
        result = forecast_case(case)
        stem = os.path.splitext(os.path.basename(path))[0]
        with open(os.path.join(out_dir, stem + '.out'), 'w', encoding='utf-8') as f:
            f.write(json.dumps(result, separators=(',', ':')))
    return 0


if __name__ == '__main__':
    sys.exit(main())
