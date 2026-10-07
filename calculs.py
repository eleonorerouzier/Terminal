"""Calculation functions of the terminal (no dependency on Streamlit)."""
import numpy as np
import pandas as pd

DAYS = 252
VOL_WIN = 126          # window (trading days) for the volatility used by the signals
BUY_TH, SELL_TH = 3, -3

STRATEGIES = [
    "Buy & hold",
    "Price above moving average",
    "Golden cross (fast MA > slow MA)",
    "Momentum (N-day return > 0)",
    "RSI mean reversion",
    "Terminal signal (the app's Buy/Hold/Sell rule)",
]


# ------------------------------------------------------------ indicators
def risk_level(vol):
    if vol < 0.15:
        return "🟢 Low"
    if vol < 0.25:
        return "🟡 Moderate"
    if vol < 0.40:
        return "🟠 High"
    return "🔴 Very high"


def rsi(series, n=14):
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    with np.errstate(divide="ignore", invalid="ignore"):
        rs = gain / loss
    out = 100 - 100 / (1 + rs)
    out[:n] = np.nan
    return out


def indicators(series, ref_returns=None):
    """Full statistics of a price series."""
    r = series.pct_change().dropna()
    years = max((series.index[-1] - series.index[0]).days / 365.25, 0.01)
    annual = (series.iloc[-1] / series.iloc[0]) ** (1 / years) - 1
    vol = r.std() * np.sqrt(DAYS)
    down_vol = r[r < 0].std() * np.sqrt(DAYS)

    dd = series / series.cummax() - 1
    under = dd < 0
    longest = int(under.groupby((~under).cumsum()).sum().max())

    var95 = r.quantile(0.05)
    cvar95 = r[r <= var95].mean()
    ma200 = series.rolling(200).mean().iloc[-1]

    beta = corr = np.nan
    if ref_returns is not None:
        df = pd.concat([r, ref_returns], axis=1, join="inner").dropna()
        if len(df) > 60 and df.iloc[:, 1].var() > 0:
            beta = df.iloc[:, 0].cov(df.iloc[:, 1]) / df.iloc[:, 1].var()
            corr = df.iloc[:, 0].corr(df.iloc[:, 1])

    def perf(n):
        return series.iloc[-1] / series.iloc[-n - 1] - 1 if len(series) > n else np.nan

    last_rsi = rsi(series).iloc[-1]
    mr = monthly_returns(series)
    return {
        "Risk level": risk_level(vol),
        "Last price": series.iloc[-1],
        "1M perf": perf(21),
        "6M perf": perf(126),
        "Total perf": series.iloc[-1] / series.iloc[0] - 1,
        "Annual perf": annual,
        "Volatility": vol,
        "Max drawdown": dd.min(),
        "Current drawdown": dd.iloc[-1],
        "Longest underwater (days)": longest,
        "VaR 95% (1d)": var95,
        "CVaR 95% (1d)": cvar95,
        "Worst day": r.min(),
        "Best day": r.max(),
        "% positive days": (r > 0).mean(),
        "Beta": beta,
        "Corr. to benchmark": corr,
        "Sharpe": annual / vol if vol > 0 else np.nan,
        "Sortino": annual / down_vol if down_vol > 0 else np.nan,
        "vs MA200": series.iloc[-1] / ma200 - 1 if not np.isnan(ma200) else np.nan,
        "RSI (14)": last_rsi,
        "Skewness": r.skew(),
        "Excess kurtosis": r.kurt(),
        "Worst month": mr.min() if len(mr) else np.nan,
        "% negative months": (mr < 0).mean() if len(mr) else np.nan,
    }


def quality_score(table):
    """Descriptive score /100: trend (40), 6-month momentum (30), risk control (30)."""
    gap = table["vs MA200"].astype(float).fillna(0)
    mom = table["6M perf"].astype(float).fillna(0)
    vol = table["Volatility"].astype(float)
    trend = 40 * ((gap + 0.2) / 0.4).clip(0, 1)
    momentum = 30 * ((mom + 0.2) / 0.5).clip(0, 1)
    risk = 30 * (1 - ((vol - 0.10) / 0.40).clip(0, 1))
    return (trend + momentum + risk).round(0)


# ------------------------------------------------------------ signals
def signal_components(p):
    """Daily points: trend (+/-2), MA50 vs MA200 (+/-1), 6M momentum (+/-1), RSI>75 (-1)."""
    ma50, ma200 = p.rolling(50).mean(), p.rolling(200).mean()
    mom = p / p.shift(126) - 1
    r = rsi(p)
    comp = pd.DataFrame(index=p.index)
    comp["trend"] = np.where(p > ma200, 2, -2)
    comp["cross"] = np.where(ma50 > ma200, 1, -1)
    comp["momentum"] = np.where(mom > 0, 1, -1)
    comp["rsi_pen"] = np.where(r > 75, -1, 0)
    comp["points"] = comp.sum(axis=1)
    comp["rsi"] = r
    comp.loc[~(ma200.notna() & mom.notna() & r.notna()), ["trend", "cross", "momentum", "rsi_pen", "points"]] = np.nan
    return comp


def latest_signal(p, max_vol):
    comp = signal_components(p)
    last = comp.iloc[-1]
    if pd.isna(last["points"]):
        return {"Signal": "⚪ N/A", "Points": np.nan, "Why": "Not enough history (needs ~200 trading days)"}
    vol = p.pct_change().tail(VOL_WIN).std() * np.sqrt(DAYS)
    pts = int(last["points"])
    why = [
        "Price above its 200-day average (+2)" if last["trend"] > 0 else "Price below its 200-day average (-2)",
        "50-day average above 200-day (+1)" if last["cross"] > 0 else "50-day average below 200-day (-1)",
        "6-month momentum positive (+1)" if last["momentum"] > 0 else "6-month momentum negative (-1)",
    ]
    if last["rsi_pen"] < 0:
        why.append(f"RSI {last['rsi']:.0f}: overbought (-1)")
    if pts >= BUY_TH and vol > max_vol:
        signal = "🟡 HOLD"
        why.append(f"BUY capped: 6M volatility {vol:.0%} is above your {max_vol:.0%} limit")
    elif pts >= BUY_TH:
        signal = "🟢 BUY"
    elif pts <= SELL_TH:
        signal = "🔴 SELL"
    else:
        signal = "🟡 HOLD"
    return {"Signal": signal, "Points": pts, "Why": " · ".join(why)}


# ------------------------------------------------------------ returns / portfolio
def return_stats(r):
    r = r.dropna()
    if len(r) < 20:
        return {}
    curve = (1 + r).cumprod()
    years = max((r.index[-1] - r.index[0]).days / 365.25, 0.01)
    annual = curve.iloc[-1] ** (1 / years) - 1
    vol = r.std() * np.sqrt(DAYS)
    down = r[r < 0].std() * np.sqrt(DAYS)
    dd = curve / curve.cummax() - 1
    mdd = dd.min()
    return {
        "Annual perf": annual,
        "Total perf": curve.iloc[-1] - 1,
        "Volatility": vol,
        "Max drawdown": mdd,
        "VaR 95% (1d)": r.quantile(0.05),
        "Sharpe": annual / vol if vol > 0 else np.nan,
        "Sortino": annual / down if down > 0 else np.nan,
        "Calmar": annual / abs(mdd) if mdd < 0 else np.nan,
    }


def portfolio_returns(prices, weights):
    """Daily returns of a constant-weight portfolio. Each day, weights are renormalised over the
    assets that have a price (so recently listed assets join from their first day)."""
    rets = prices.pct_change(fill_method=None)
    w = weights.reindex(rets.columns).fillna(0)
    total = (rets.notna() * w).sum(axis=1)
    out = (rets.fillna(0) * w).sum(axis=1) / total.replace(0, np.nan)
    return out.dropna()


def risk_contributions(rets, weights):
    """Share of portfolio variance carried by each asset, portfolio volatility, diversification ratio."""
    w = weights.reindex(rets.columns).values
    cov = rets.cov().values * DAYS
    sw = cov @ w
    var = float(w @ sw)
    contrib = pd.Series(w * sw / var, index=rets.columns)
    vol_pf = float(np.sqrt(var))
    vols = np.sqrt(np.diag(cov))
    div = float((w * vols).sum() / vol_pf) if vol_pf > 0 else np.nan
    return contrib, vol_pf, div


# ------------------------------------------------------------ strategies / backtest
def _hysteresis(enter, leave, valid):
    state, out = 0.0, np.full(len(enter), np.nan)
    e, l, v = enter.values, leave.values, valid.values
    for i in range(len(e)):
        if not v[i]:
            continue
        if e[i]:
            state = 1.0
        elif l[i]:
            state = 0.0
        out[i] = state
    return pd.Series(out, index=enter.index)


def strategy_position(p, strategy, **k):
    """Position (1 = invested, 0 = cash) decided at each close. NaN while indicators warm up."""
    if strategy == "Buy & hold":
        return pd.Series(1.0, index=p.index)
    if strategy.startswith("Price above"):
        ma = p.rolling(int(k.get("window", 200))).mean()
        pos = (p > ma).astype(float)
        pos[ma.isna()] = np.nan
        return pos
    if strategy.startswith("Golden"):
        fast = p.rolling(int(k.get("fast", 50))).mean()
        slow = p.rolling(int(k.get("slow", 200))).mean()
        pos = (fast > slow).astype(float)
        pos[slow.isna()] = np.nan
        return pos
    if strategy.startswith("Momentum"):
        m = p / p.shift(int(k.get("lookback", 126))) - 1
        pos = (m > 0).astype(float)
        pos[m.isna()] = np.nan
        return pos
    if strategy.startswith("RSI"):
        r = rsi(p)
        return _hysteresis(r < k.get("rsi_buy", 30), r > k.get("rsi_sell", 55), r.notna())
    if strategy.startswith("Terminal"):
        pts = signal_components(p)["points"]
        vol = p.pct_change().rolling(VOL_WIN).std() * np.sqrt(DAYS)
        enter = (pts >= BUY_TH) & (vol <= k.get("max_vol", 0.35))
        return _hysteresis(enter, pts <= SELL_TH, pts.notna() & vol.notna())
    raise ValueError(strategy)


def run_backtest(p, strategy, cost=0.001, **k):
    """The position decided at a close is applied from the next session (no look-ahead).
    Strategy and buy & hold are compared over exactly the same period."""
    pos = strategy_position(p, strategy, **k)
    first = pos.first_valid_index()
    if first is None:
        return None
    ret = p.pct_change().fillna(0).loc[first:]
    pos = pos.loc[first:].fillna(0).shift(1).fillna(0)
    trades = pos.diff().abs().fillna(0)
    strat = pos * ret - trades * cost
    return {
        "strategy": strat.iloc[1:], "hold": ret.iloc[1:], "position": pos.iloc[1:],
        "entries": int((pos.diff() > 0).sum()), "time_invested": float(pos.mean()),
    }


def param_sweep(p, strategy, values, cost=0.001):
    """Performance of the strategy for a range of parameter values (same period for all)."""
    key = "window" if strategy.startswith("Price above") else "lookback"
    warm = int(max(values)) + 1
    if len(p) < warm + 120:
        return None
    start = p.index[warm]
    ret = p.pct_change().fillna(0).loc[start:]
    rows = []
    for v in values:
        pos = strategy_position(p, strategy, **{key: v}).fillna(0).shift(1).fillna(0).loc[start:]
        strat = pos * ret - pos.diff().abs().fillna(0) * cost
        s = return_stats(strat)
        rows.append({"Parameter": v, "Annual perf": s["Annual perf"],
                     "Max drawdown": s["Max drawdown"], "Sharpe": s["Sharpe"]})
    return pd.DataFrame(rows), return_stats(ret)


# ------------------------------------------------------------ advanced risk
def monthly_returns(s):
    m = s.groupby([s.index.year, s.index.month]).last()
    return m.pct_change().dropna()


def risk_score(table):
    """Descriptive risk score /100 (higher = riskier): volatility 40%, max drawdown 35%, CVaR 25%."""
    v = (table["Volatility"].astype(float) / 0.60).clip(0, 1)
    d = (-table["Max drawdown"].astype(float) / 0.70).clip(0, 1)
    t = (-table["CVaR 95% (1d)"].astype(float) / 0.06).clip(0, 1)
    return (100 * (0.40 * v + 0.35 * d + 0.25 * t)).round(0)


def to_eur(s, rate, scale=1.0):
    """Convert a local-currency price series to EUR. `rate` = units of local currency per 1 EUR."""
    r = rate.dropna()
    if r.empty:
        return None
    aligned = r.reindex(s.index.union(r.index)).ffill().reindex(s.index).bfill()
    return (s / aligned / scale).dropna()


def drawdown_episodes(s, top=5):
    """The `top` worst peak-to-trough episodes of a price/value series."""
    dd = s / s.cummax() - 1
    groups = list(dd.groupby((dd == 0).cumsum()))
    rows = []
    for i, (_, g) in enumerate(groups):
        seg = g.iloc[1:]
        if seg.empty or seg.min() >= 0:
            continue
        trough = seg.idxmin()
        rec = groups[i + 1][1].index[0] if i + 1 < len(groups) else None
        rows.append({"Peak": g.index[0], "Trough": trough, "Depth": seg.min(),
                     "Days peak→trough": (trough - g.index[0]).days,
                     "Recovered on": rec,
                     "Days to recover": (rec - trough).days if rec is not None else np.nan})
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("Depth").head(top).reset_index(drop=True)


def portfolio_risk_report(r, ref_returns=None):
    """Risk statistics of a daily-return series (portfolio)."""
    r = r.dropna()
    curve = (1 + r).cumprod()
    mr = monthly_returns(curve)
    var95, var99 = r.quantile(0.05), r.quantile(0.01)
    beta = corr = np.nan
    if ref_returns is not None:
        df = pd.concat([r, ref_returns], axis=1, join="inner").dropna()
        if len(df) > 60 and df.iloc[:, 1].var() > 0:
            beta = df.iloc[:, 0].cov(df.iloc[:, 1]) / df.iloc[:, 1].var()
            corr = df.iloc[:, 0].corr(df.iloc[:, 1])
    return {
        "Volatility": r.std() * np.sqrt(DAYS), "VaR 95% (1d)": var95, "VaR 99% (1d)": var99,
        "CVaR 95% (1d)": r[r <= var95].mean(), "Max drawdown": (curve / curve.cummax() - 1).min(),
        "Worst month": mr.min() if len(mr) else np.nan,
        "% negative months": (mr < 0).mean() if len(mr) else np.nan,
        "Beta": beta, "Corr. to benchmark": corr, "Skewness": r.skew(), "Excess kurtosis": r.kurt(),
        "Worst day": r.min(), "Best day": r.max(),
    }


def concentration(weights):
    w = weights.sort_values(ascending=False)
    hhi = float((w ** 2).sum())
    return {"top_name": w.index[0], "top_weight": float(w.iloc[0]), "top3": float(w.head(3).sum()),
            "hhi": hhi, "effective_n": 1 / hhi if hhi > 0 else np.nan}


STRESS_WINDOWS = {
    "2008 financial crisis (Oct 2007 → Mar 2009)": ("2007-10-09", "2009-03-09"),
    "2011 euro debt crisis (Apr → Oct 2011)": ("2011-04-29", "2011-10-03"),
    "2020 Covid crash (Feb → Mar 2020)": ("2020-02-19", "2020-03-23"),
    "2022 inflation & rates bear market (Jan → Oct 2022)": ("2022-01-03", "2022-10-12"),
    "2025 tariff shock (Feb → Apr 2025)": ("2025-02-19", "2025-04-08"),
}


def stress_test(series_by_asset, weights, start, end):
    """Return of the portfolio over a historical window. Assets that did not exist yet are left
    out and the other weights are renormalised; `coverage` tells how much of the portfolio is covered."""
    t0, t1 = pd.Timestamp(start), pd.Timestamp(end)
    rets = {}
    for n, s in series_by_asset.items():
        if s.empty or s.index[0] > t0 + pd.Timedelta(days=7):
            continue
        a, b = s.loc[:t0], s.loc[:t1]
        if a.empty or b.empty or b.index[-1] < t1 - pd.Timedelta(days=10):
            continue
        rets[n] = b.iloc[-1] / a.iloc[-1] - 1
    if not rets:
        return None
    r = pd.Series(rets)
    w = weights.reindex(r.index).fillna(0)
    cov = float(w.sum() / weights.sum())
    if cov <= 0:
        return None
    return {"return": float((r * w).sum() / w.sum()), "coverage": cov,
            "worst_asset": r.idxmin(), "worst_ret": float(r.min())}


def monte_carlo(returns, horizon, n_paths=2000, block=10, annual_return=None, seed=42):
    """Block-bootstrap simulation of future portfolio paths from past daily returns.
    If `annual_return` is given, history is re-centred on that average return."""
    r = returns.dropna().values.astype(float)
    if annual_return is not None:
        r = r - r.mean() + ((1 + annual_return) ** (1 / DAYS) - 1)
    n = len(r)
    nb = int(np.ceil(horizon / block))
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n - block + 1, size=(n_paths, nb))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(n_paths, -1)[:, :horizon]
    paths = np.concatenate([np.ones((n_paths, 1)), np.cumprod(1 + r[idx], axis=1)], axis=1)
    max_dd = (paths / np.maximum.accumulate(paths, axis=1) - 1).min(axis=1)
    return paths, paths[:, -1], max_dd


# ------------------------------------------------------------ plain-language layer
def risk_score_from(vol, max_dd, cvar):
    """Same recipe as risk_score() for a single set of numbers (asset or portfolio)."""
    v = min(max(vol / 0.60, 0), 1)
    d = min(max(-max_dd / 0.70, 0), 1)
    t = min(max(-cvar / 0.06, 0), 1)
    return round(100 * (0.40 * v + 0.35 * d + 0.25 * t))


def risk_label(score):
    if pd.isna(score):
        return "n/a"
    if score < 15:
        return "🟢 Very low"
    if score < 30:
        return "🟢 Low"
    if score < 50:
        return "🟡 Moderate"
    if score < 70:
        return "🟠 High"
    return "🔴 Very high"


RISK_LEVEL_GUIDE = {
    "🟢 Very low": "Barely moves: money-market type funds.",
    "🟢 Low": "Small swings: most bond funds.",
    "🟡 Moderate": "Noticeable swings: diversified world equity funds.",
    "🟠 High": "Big swings: most individual shares.",
    "🔴 Very high": "Wild swings, falls of 40-60% are possible: volatile shares, commodities.",
}

HORIZON_NOTE = {
    "🟢 Very low": "Many investors use this kind of asset for money they may need soon.",
    "🟢 Low": "Many investors use this kind of asset for money they may need within a few years.",
    "🟡 Moderate": "Usually considered for money you can leave invested for several years.",
    "🟠 High": "Usually considered only for money you can leave untouched for 5 years or more, "
               "accepting large temporary losses.",
    "🔴 Very high": "Can lose half of its value: usually kept to a small part of a portfolio, with money "
                    "you can leave untouched for a long time.",
}


def _money(x, ccy):
    return f"{x:,.0f} €" if ccy == "EUR" else f"{x:,.0f} {ccy}"


def _duration(days):
    if pd.isna(days):
        return "an unknown time"
    return f"{days / 30.4:.0f} months" if days >= 60 else f"{days:.0f} days"


def explain_asset(row, episodes, local_ccy, shown_ccy, per=1000):
    """Plain-English reading of one asset's risk. Returns a list of sentences (markdown)."""
    mdd, vol, var95 = row["Max drawdown"], row["Volatility"], row["VaR 95% (1d)"]
    cur_dd, neg, worst_m = row["Current drawdown"], row["% negative months"], row["Worst month"]
    label = row["Risk level"]
    out = [f"**Risk level: {label}**: score {row['Risk score /100']:.0f}/100 "
           "(0 = very calm, 100 = very wild). " + RISK_LEVEL_GUIDE.get(label, "")]
    s = f"**Worst fall seen: {mdd:.0%}**"
    if episodes is not None and len(episodes):
        e = episodes.iloc[0]
        s += f" (from {e['Peak']:%b %Y} to {e['Trough']:%b %Y})"
        s += (", and it has **not fully recovered yet**" if pd.isna(e["Recovered on"])
              else f", then it took about {_duration(e['Days to recover'])} to get back to its previous peak")
    s += f". If you had invested {_money(per, shown_ccy)}, that is a drop of about **{_money(abs(per * mdd), shown_ccy)}** at the worst moment."
    out.append(s)
    out.append(f"**Typical yearly move: about ±{vol:.0%}.** On {_money(per, shown_ccy)} that is roughly "
               f"±{_money(per * vol, shown_ccy)} in a normal year, and bigger in an unusual one.")
    out.append(f"**On 1 trading day out of 20 it lost {abs(var95):.1%} or more** "
               f"(about {_money(per * abs(var95), shown_ccy)} on {_money(per, shown_ccy)}).")
    if pd.notna(neg):
        out.append(f"It finished **{neg:.0%} of months in the red**; its worst month was **{worst_m:.0%}**.")
    out.append("**Today** it is at or near its highest level of the period." if cur_dd > -0.02
               else f"**Today** it is **{abs(cur_dd):.0%} below** its highest level of the period.")
    if local_ccy not in ("EUR", "?"):
        out.append(f"It is quoted in {local_ccy}. " + (
            "The figures above include currency moves against the euro." if shown_ccy == "EUR"
            else "The figures above ignore currency moves."))
    out.append(HORIZON_NOTE.get(label, "") + " *(General education, not personal advice.)*")
    return out


PROFILE_LIMITS = {"Cautious": (0.10, 0.15), "Balanced": (0.15, 0.25), "Dynamic": (0.25, 0.40)}


def _lo(value, good, ok):
    return "✅ OK" if value <= good else ("⚠️ Watch" if value <= ok else "❌ Look closer")


def _hi(value, good, ok):
    return "✅ OK" if value >= good else ("⚠️ Watch" if value >= ok else "❌ Look closer")


def portfolio_checks(rep, conc, non_eur, divers, profile):
    vl, dl = PROFILE_LIMITS[profile]
    rows = [
        ("Yearly swings", f"{rep['Volatility']:.0%}", f"up to {vl:.0%}", _lo(rep["Volatility"], vl, vl * 1.3),
         "How much your portfolio value typically moves up and down over a year."),
        ("Worst fall", f"{rep['Max drawdown']:.0%}", f"not worse than -{dl:.0%}",
         _lo(-rep["Max drawdown"], dl, dl * 1.3), "The biggest drop from a peak to a low in the period."),
        ("Biggest single position", f"{conc['top_name']} {conc['top_weight']:.0%}", "up to 20%",
         _lo(conc["top_weight"], 0.20, 0.35), "If one line is too big, its bad day becomes your bad day."),
        ("Number of real positions", f"{conc['effective_n']:.1f}", "6 or more",
         _hi(conc["effective_n"], 6, 3), "Counts lines as if they were equal in size: 20 lines but 90% in one = about 1."),
        ("Do holdings offset each other?", f"{divers:.2f}", "1.3 or more",
         _hi(divers, 1.3, 1.1), "Above 1, some holdings go up when others go down, which smooths the ride."),
        ("Money outside the euro", f"{non_eur:.0%}", "up to 50%", _lo(non_eur, 0.50, 0.80),
         "The part of your money that also depends on exchange rates (dollar, yen...)."),
    ]
    return pd.DataFrame(rows, columns=["Check", "Your portfolio", "Common rule of thumb", "Status", "What it means"])


def explain_portfolio(total, rep, conc, non_eur, years_hist, label, score, checks, profile):
    n_ok = int(checks["Status"].str.startswith("✅").sum())
    n_w = int(checks["Status"].str.startswith("⚠️").sum())
    n_b = int(checks["Status"].str.startswith("❌").sum())
    mdd, vol, var95, cvar = rep["Max drawdown"], rep["Volatility"], rep["VaR 95% (1d)"], rep["CVaR 95% (1d)"]
    out = [
        f"**Health check vs a *{profile}* profile: {n_ok} ✅ OK, {n_w} ⚠️ to watch, {n_b} ❌ to look at.**",
        f"**Worst fall:** over the last {years_hist:.1f} years, a portfolio with your current mix would have "
        f"dropped **{abs(mdd):.0%}** at the worst moment: from {total:,.0f} € to about **{total * (1 + mdd):,.0f} €**.",
        f"**Normal year:** expect it to move by about **±{vol:.0%}** (roughly ±{total * vol:,.0f} €), and more in an unusual year.",
        f"**Bad day:** on 1 day out of 20 you would lose **{abs(var95):.1%} or more** (about {total * abs(var95):,.0f} €). "
        f"On the very worst days the average loss was {abs(cvar):.1%} (about {total * abs(cvar):,.0f} €).",
        f"**Concentration:** your biggest line is {conc['top_name']} ({conc['top_weight']:.0%}) and your 3 biggest lines make "
        f"up {conc['top3']:.0%} of the portfolio.",
        f"**Currency:** {non_eur:.0%} of your money is in assets quoted outside the euro.",
        HORIZON_NOTE.get(label, "") + " *(General education, not personal advice.)*",
    ]
    return out
