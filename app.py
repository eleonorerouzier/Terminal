import time
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh

from calculs import (STRATEGIES, STRESS_WINDOWS, indicators, quality_score, risk_score, latest_signal,
                     signal_components, return_stats, portfolio_returns, risk_contributions, run_backtest,
                     param_sweep, to_eur, drawdown_episodes, portfolio_risk_report, concentration,
                     stress_test, monte_carlo, risk_label, risk_score_from, RISK_LEVEL_GUIDE,
                     explain_asset, portfolio_checks, explain_portfolio)
from universe import UNIVERSE, CORE_CATEGORY, EXAMPLE_AMOUNTS, currency_of, fx_pair

TITLE = "Terminal - Main"
st.set_page_config(page_title=TITLE, page_icon="📊", layout="wide")

# ============================================================ SIDEBAR
st.sidebar.header("⚙️ Settings")
categories = st.sidebar.multiselect("Categories", list(UNIVERSE.keys()), default=list(UNIVERSE.keys()),
                                    help="Fewer categories = faster loading.")
extra = st.sidebar.text_input("Add tickers (comma-separated)", placeholder="e.g. SHEL.AS, VWCE.DE")
period = st.sidebar.selectbox("History", ["1y", "2y", "5y", "10y", "max"], index=2)
profile = st.sidebar.radio("Risk profile", ["Cautious", "Balanced", "Dynamic"], index=1,
                           help="1) Sets the volatility above which a single-asset BUY signal is downgraded to HOLD "
                                "(25% / 35% / 50% a year). 2) Sets the comfort limits of the Portfolio risk "
                                "health check (swings up to 10% / 15% / 25%, worst fall up to 15% / 25% / 40%).")
MAX_VOL = {"Cautious": 0.25, "Balanced": 0.35, "Dynamic": 0.50}[profile]
ref_ticker = st.sidebar.text_input("Benchmark (for beta)", value="IWDA.AS",
                                   help="Default: iShares MSCI World ETF (Amsterdam).").strip().upper()

to_eur_on = st.sidebar.toggle(
    "Show everything in EUR", value=True,
    help="Converts every price to euros with Yahoo exchange rates, so performance and risk include "
         "currency moves (what you really get as a euro investor). Off = each asset in its own currency.")

st.sidebar.subheader("🔄 Refresh")
auto = st.sidebar.toggle("Auto-refresh", value=True)
freq = st.sidebar.selectbox("Frequency", [1, 5, 15, 60], index=1, format_func=lambda m: f"every {m} min")
if auto:
    st_autorefresh(interval=freq * 60 * 1000, key="auto_refresh")
if st.sidebar.button("Refresh now"):
    st.cache_data.clear()
    st.rerun()

st.sidebar.subheader("🔔 Alert thresholds")
alert_month = st.sidebar.slider("Drop over 1 month (%)", 3, 30, 10)
alert_dd = st.sidebar.slider("Drop from peak (%)", 5, 50, 20)
alert_vol = st.sidebar.slider("Yearly volatility (%)", 10, 80, 35)

time_block = int(time.time() // (freq * 60))


# ============================================================ DATA
@st.cache_data(ttl=7200, max_entries=10, show_spinner=False)
def download_prices(tickers, period, time_block):
    """Daily adjusted closes for all tickers (downloaded in chunks, with a per-ticker fallback)."""
    frames = []
    for i in range(0, len(tickers), 25):
        chunk = list(tickers[i:i + 25])
        try:
            data = yf.download(chunk, period=period, auto_adjust=True, progress=False, threads=True)
            close = data["Close"]
            if isinstance(close, pd.Series):
                close = close.to_frame(chunk[0])
            frames.append(close)
        except Exception:
            continue
    prices = pd.concat(frames, axis=1) if frames else pd.DataFrame()
    if len(prices):
        prices.index = pd.DatetimeIndex(prices.index).tz_localize(None).normalize()
        prices = prices.loc[:, ~prices.columns.duplicated()]
    missing = [t for t in tickers if t not in prices.columns or prices[t].dropna().shape[0] < 30]
    for t in missing[:15]:
        try:
            s = yf.Ticker(t).history(period=period, auto_adjust=True)["Close"]
            if len(s) >= 30:
                s.index = s.index.tz_localize(None).normalize()
                s = s[~s.index.duplicated()].rename(t)
                prices = prices.drop(columns=[t], errors="ignore")
                prices = prices.join(s, how="outer") if len(prices) else s.to_frame()
        except Exception:
            pass
    return prices


@st.cache_data(ttl=21600, max_entries=100, show_spinner=False)
def asset_info(ticker):
    """Fundamentals, dividend and news (may be unavailable depending on the asset)."""
    out = {"fund": {}, "dividend_12m": None, "news": []}
    try:
        t = yf.Ticker(ticker)
        try:
            info = t.info or {}
        except Exception:
            info = {}
        out["fund"] = {"Name": info.get("longName") or info.get("shortName"), "Sector": info.get("sector"),
                       "P/E (trailing)": info.get("trailingPE"), "Market cap": info.get("marketCap"),
                       "Currency": info.get("currency")}
        try:
            div = t.dividends
            if div is not None and len(div):
                div.index = div.index.tz_localize(None)
                out["dividend_12m"] = float(div[div.index > pd.Timestamp.now() - pd.Timedelta(days=365)].sum())
        except Exception:
            pass
        try:
            for n in (t.news or [])[:6]:
                c = n.get("content", n)
                title = c.get("title")
                link = (c.get("canonicalUrl") or {}).get("url") or c.get("link")
                source = (c.get("provider") or {}).get("displayName") or c.get("publisher")
                if title and link:
                    out["news"].append((title, link, source))
        except Exception:
            pass
    except Exception:
        pass
    return out


def csv_bytes(df):
    return df.to_csv(sep=";", decimal=",").encode("utf-8-sig")


# ============================================================ LOADING
st.title(f"📊 {TITLE}")

assets, category = {}, {}
for cat in categories:
    for name, tk in UNIVERSE[cat].items():
        assets[name], category[name] = tk, cat
for tk in [x.strip().upper() for x in extra.split(",") if x.strip()]:
    assets[tk], category[tk] = tk, "Other"
if not assets:
    st.info("Pick at least one category in the sidebar.")
    st.stop()

local_ccy = {n: currency_of(tk) for n, tk in assets.items()}
fx_needed = {fx_pair(c) for c in set(local_ccy.values()) if c not in ("EUR", "?")} if to_eur_on else set()
tickers = tuple(dict.fromkeys(list(assets.values()) + ([ref_ticker] if ref_ticker else []) + [p for p, _ in fx_needed]))
with st.spinner(f"Downloading {len(tickers)} tickers (first load can take a minute)..."):
    prices = download_prices(tickers, period, time_block)

cours, not_loaded, no_fx, shown_ccy = {}, [], [], {}
for name, tk in assets.items():
    s = prices[tk].dropna() if (len(prices) and tk in prices.columns) else None
    if s is None or len(s) < 30:
        not_loaded.append(f"{name} ({tk})")
        continue
    ccy = local_ccy[name]
    shown_ccy[name] = ccy
    if to_eur_on and ccy not in ("EUR", "?"):
        pair, scale = fx_pair(ccy)
        s_eur = to_eur(s, prices[pair], scale) if (len(prices) and pair in prices.columns) else None
        if s_eur is not None and len(s_eur) >= 30:
            s, shown_ccy[name] = s_eur, "EUR"
        else:
            no_fx.append(f"{name} ({ccy})")
    cours[name] = s
if not_loaded:
    st.warning("No data for: " + ", ".join(not_loaded) + ". Ignored (check the ticker or retry later).")
if no_fx:
    st.warning("No exchange rate for: " + ", ".join(no_fx) + ". Shown in their local currency.")
if not cours:
    st.error("No data could be downloaded. Try again in a few minutes.")
    st.stop()

ref_ret = None
if ref_ticker and len(prices) and ref_ticker in prices.columns:
    ref_ret = prices[ref_ticker].dropna().pct_change().dropna()
else:
    st.caption("ℹ️ Benchmark unavailable: beta will not be computed.")

table = pd.DataFrame({n: indicators(s, ref_ret) for n, s in cours.items()}).T.infer_objects()
table["Score /100"] = quality_score(table)
table["Category"] = [category[n] for n in table.index]
table["Local ccy"] = [local_ccy[n] for n in table.index]
table["Price ccy"] = [shown_ccy[n] for n in table.index]
table["Risk score /100"] = risk_score(table)
table["Risk level"] = table["Risk score /100"].map(risk_label)
sig = pd.DataFrame({n: latest_signal(s, MAX_VOL) for n, s in cours.items()}).T
sig["Points"] = pd.to_numeric(sig["Points"], errors="coerce")
table = table.join(sig)

wide = pd.concat(cours, axis=1).sort_index().ffill()
cats_present = [c for c in UNIVERSE if c in set(category.values())] + (["Other"] if "Other" in category.values() else [])
last_date = max(s.index[-1] for s in cours.values())
st.caption(f"Latest close available: **{last_date:%d/%m/%Y}** · Page updated {time.strftime('%d/%m/%Y %H:%M')} · "
           f"{len(cours)} assets · Yahoo Finance data · "
           + ("all prices converted to EUR. " if to_eur_on else "performance in each asset's local currency (no FX adjustment). ")
           + "Analysis tool, not investment advice.")

# ============================================================ ALERTS
alerts = []
for name, l in table.iterrows():
    if l["1M perf"] <= -alert_month / 100:
        alerts.append(f"📉 **{name}**: {l['1M perf']:.1%} over 1 month")
    if l["Current drawdown"] <= -alert_dd / 100:
        alerts.append(f"🔻 **{name}**: {l['Current drawdown']:.1%} below its peak")
    if l["Volatility"] >= alert_vol / 100:
        alerts.append(f"⚡ **{name}**: volatility {l['Volatility']:.0%} per year")
with st.expander(f"🔔 Alerts ({len(alerts)})", expanded=False):
    if alerts:
        for a in alerts:
            st.markdown(a)
    else:
        st.success("No alert with your current thresholds.")

# ============================================================ TABS
(t_sig, t_over, t_perf, t_risk, t_corr, t_asset, t_pf, t_prisk, t_core, t_lab) = st.tabs([
    "🎯 Signals", "📋 Overview", "📈 Performance", "⚠️ Risk", "🔗 Correlation",
    "🔍 Asset sheet", "💼 Portfolio", "🛡️ Portfolio risk", "⚖️ Core vs themes", "🧪 Strategy Lab"])

PCT_COLS = ["1M perf", "6M perf", "Annual perf", "Volatility", "Max drawdown", "Current drawdown",
            "VaR 95% (1d)", "vs MA200"]
FMT = {c: "{:.1%}" for c in PCT_COLS}
FMT.update({"VaR 95% (1d)": "{:.2%}", "CVaR 95% (1d)": "{:.2%}", "Skewness": "{:.2f}", "Excess kurtosis": "{:.1f}",
            "Worst month": "{:.1%}", "% negative months": "{:.0%}", "Risk score /100": "{:.0f}",
            "Longest underwater (days)": "{:.0f}", "Corr. to benchmark": "{:.2f}", "Last price": "{:,.2f}", "Beta": "{:.2f}", "Sharpe": "{:.2f}", "Sortino": "{:.2f}",
            "Score /100": "{:.0f}", "Points": "{:.0f}", "RSI (14)": "{:.0f}"})

# ---------------------------------------------------------------- 1. Signals
with t_sig:
    st.subheader("🎯 Buy / Hold / Sell signals")
    st.warning("These are **mechanical, rule-based signals** built only from price trend, momentum and "
               "volatility. They are **not financial advice**: they ignore fundamentals, valuation, "
               "taxes, fees and your personal situation, and past patterns may not repeat. "
               "Test the rule yourself in the 🧪 Strategy Lab.")
    counts = table["Signal"].value_counts()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🟢 BUY", int(counts.get("🟢 BUY", 0)))
    c2.metric("🟡 HOLD", int(counts.get("🟡 HOLD", 0)))
    c3.metric("🔴 SELL", int(counts.get("🔴 SELL", 0)))
    c4.metric("Risk profile", f"{profile} (≤ {MAX_VOL:.0%} vol)")

    f1, f2 = st.columns(2)
    sel_sig = f1.multiselect("Show signals", ["🟢 BUY", "🟡 HOLD", "🔴 SELL", "⚪ N/A"],
                             default=["🟢 BUY", "🟡 HOLD", "🔴 SELL"])
    sel_cat = f2.multiselect("Show categories", cats_present, default=cats_present, key="sig_cats")
    view = table[table["Signal"].isin(sel_sig) & table["Category"].isin(sel_cat)]
    view = view.sort_values("Points", ascending=False)
    cols = ["Category", "Signal", "Points", "Why", "Risk level", "Last price", "Price ccy", "1M perf", "6M perf",
            "Volatility", "vs MA200", "RSI (14)", "Score /100"]
    st.dataframe(view[cols].style.format(FMT, na_rep="-"), use_container_width=True, height=600)
    st.download_button("⬇️ Export signals (CSV, opens in Excel)", csv_bytes(view[cols]), "signals.csv", "text/csv")

    with st.expander("How the signal works"):
        st.markdown(
            "Each asset gets **points**, recomputed on every refresh:\n"
            "- **+2 / -2**: price above / below its 200-day moving average (long-term trend)\n"
            "- **+1 / -1**: 50-day average above / below the 200-day average\n"
            "- **+1 / -1**: 6-month return positive / negative (momentum)\n"
            "- **-1**: RSI above 75 (short-term overbought)\n\n"
            "**🟢 BUY** if points ≥ 3 · **🔴 SELL** if points ≤ -3 · **🟡 HOLD** otherwise.\n\n"
            f"**Risk filter**: if the 6-month volatility is above your limit ({MAX_VOL:.0%} for the "
            f"*{profile}* profile), a BUY is downgraded to HOLD.\n\n"
            "BUY means *the trend is favourable*, SELL means *the trend has turned*. "
            "None of this says an asset is cheap or expensive.")

# ---------------------------------------------------------------- 2. Overview
with t_over:
    cols = ["Category", "Score /100", "Signal", "Risk level", "Last price", "Price ccy", "Local ccy", "1M perf", "6M perf",
            "Annual perf", "Volatility", "Max drawdown", "Current drawdown", "VaR 95% (1d)", "Beta",
            "Sharpe", "Sortino", "vs MA200"]
    over = table[cols].sort_values("Score /100", ascending=False)
    st.dataframe(over.style.format(FMT, na_rep="-"), use_container_width=True, height=600)
    st.caption("**Score /100** = trend (40 pts) + 6-month momentum (30 pts) + risk control (30 pts). "
               "It describes the current state of an asset; it does not tell you to buy or sell.")
    st.download_button("⬇️ Export table (CSV, opens in Excel)", csv_bytes(over), "overview.csv", "text/csv")

# ---------------------------------------------------------------- 3. Performance
WINDOWS = {"Full history": None, "3 years": pd.DateOffset(years=3), "1 year": pd.DateOffset(years=1),
           "6 months": pd.DateOffset(months=6), "3 months": pd.DateOffset(months=3),
           "1 month": pd.DateOffset(months=1)}
with t_perf:
    st.subheader("Comparative performance (base 100 at the start of the window)")
    win = st.radio("Window", list(WINDOWS.keys()), horizontal=True)
    sub = wide if WINDOWS[win] is None else wide.loc[wide.index >= wide.index[-1] - WINDOWS[win]]
    first = sub.apply(lambda c: c.dropna().iloc[0] if c.notna().any() else np.nan)
    base = sub / first * 100
    palette = px.colors.qualitative.Safe
    cat_color = {c: palette[i % len(palette)] for i, c in enumerate(cats_present)}

    st.markdown("**1. All assets** (click a category in the legend to hide it; hover a line to see the asset)")
    fig = go.Figure()
    seen = set()
    for name in base.columns:
        c = category[name]
        fig.add_trace(go.Scatter(
            x=base.index, y=base[name], mode="lines", line=dict(width=1.2, color=cat_color[c]),
            name=c, legendgroup=c, showlegend=c not in seen,
            hovertemplate=name + ": %{y:.1f}<extra></extra>"))
        seen.add(c)
    fig.add_hline(y=100, line_dash="dash", line_color="grey")
    fig.update_layout(height=620, hovermode="closest", yaxis_title="Base 100")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("**2. Category baskets** (equal-weight average of the assets in each category)")
    baskets = pd.DataFrame({c: base[[n for n in base.columns if category[n] == c]].mean(axis=1)
                            for c in cats_present})
    fig = px.line(baskets, labels={"value": "Base 100", "index": "", "variable": ""})
    fig.add_hline(y=100, line_dash="dash", line_color="grey")
    fig.update_layout(height=480, legend_title_text="")
    st.plotly_chart(fig, use_container_width=True)

    l1, l2 = st.columns(2)
    cat_perf = (baskets.iloc[-1] / 100 - 1).sort_values()
    fig = px.bar(cat_perf * 100, orientation="h", labels={"value": f"Performance over {win.lower()} (%)", "index": ""},
                 title="Category ranking")
    fig.update_layout(showlegend=False)
    l1.plotly_chart(fig, use_container_width=True)
    asset_perf = (base.iloc[-1] / 100 - 1).dropna().sort_values(ascending=False)
    top = pd.DataFrame({"Asset": asset_perf.index, "Perf.": asset_perf.values,
                        "Category": [category[n] for n in asset_perf.index]})
    l2.markdown(f"**Top 8 / Bottom 8 over {win.lower()}**")
    l2.dataframe(pd.concat([top.head(8), top.tail(8)]).style.format({"Perf.": "{:.1%}"}),
                 use_container_width=True, hide_index=True, height=420)

    st.markdown("**3. One chart per category**")
    for i in range(0, len(cats_present), 2):
        cols_ = st.columns(2)
        for col, c in zip(cols_, cats_present[i:i + 2]):
            names = [n for n in base.columns if category[n] == c]
            fig = px.line(base[names], labels={"value": "Base 100", "index": "", "variable": ""}, title=c)
            fig.add_hline(y=100, line_dash="dash", line_color="grey")
            fig.update_layout(height=400, legend_title_text="", legend=dict(orientation="h", y=-0.2))
            col.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------- 4. Risk
LEVEL_COLORS = {"🟢 Very low": "#2e9e5b", "🟢 Low": "#7bc47f", "🟡 Moderate": "#f2c94c",
                "🟠 High": "#f2994a", "🔴 Very high": "#eb5757"}
with t_risk:
    st.subheader("⚠️ Risk, explained simply")
    st.info("**What does 'risk' mean here?** How much, and how fast, the value of an investment has fallen in the past. "
            "A higher level means bigger and more frequent falls (and usually bigger gains too). "
            "It describes the past: it is **not** a promise about the future.")
    with st.expander("📏 What do the risk levels look like?", expanded=False):
        st.markdown("\n".join(f"- **{k}**: {v}" for k, v in RISK_LEVEL_GUIDE.items()) +
                    "\n\nThe level comes from a score out of 100 that mixes three things: how much the price swings "
                    "(volatility, 40%), the worst fall seen (35%), and the average loss on the worst days (25%).")

    st.markdown("### 1. The simple table")
    f1, f2, f3, f4 = st.columns(4)
    amount = f1.number_input("Illustrate with an investment of (€)", min_value=100, max_value=1_000_000,
                             value=1000, step=100, key="risk_amount")
    lvl_sel = f2.multiselect("Risk levels", list(RISK_LEVEL_GUIDE.keys()), default=list(RISK_LEVEL_GUIDE.keys()), key="risk_levels")
    cat_sel = f3.multiselect("Categories", cats_present, default=cats_present, key="risk_cats")
    order = f4.radio("Order", ["Riskiest first", "Safest first"], horizontal=True, key="risk_order")
    sub_t = table[table["Risk level"].isin(lvl_sel) & table["Category"].isin(cat_sel)]
    sub_t = sub_t.sort_values("Risk score /100", ascending=(order == "Safest first"))
    col_eur = f"Worst fall in € (on {amount:,.0f})"
    simple = pd.DataFrame({
        "Category": sub_t["Category"], "Risk level": sub_t["Risk level"],
        "Worst fall seen": sub_t["Max drawdown"], col_eur: amount * sub_t["Max drawdown"],
        "Today vs its peak": sub_t["Current drawdown"], "Typical yearly move (±)": sub_t["Volatility"],
        "Loss on 1 bad day in 20": sub_t["VaR 95% (1d)"], "Months in the red": sub_t["% negative months"],
        "Quoted in": sub_t["Price ccy"]})
    st.dataframe(simple.style.format({
        "Worst fall seen": "{:.0%}", col_eur: "{:,.0f}", "Today vs its peak": "{:.0%}",
        "Typical yearly move (±)": "{:.0%}", "Loss on 1 bad day in 20": "{:.1%}", "Months in the red": "{:.0%}"},
        na_rep="-"), use_container_width=True, height=520)
    st.caption("**How to read a line:** 'Worst fall seen -45%' means that at the worst moment the price was 45% below its "
               "previous peak. 'Today vs its peak' tells you where it stands now. 'Typical yearly move ±30%' means that "
               "in a normal year the value can easily go up or down by about 30%. "
               + ("All figures are in euros." if to_eur_on else "Figures are in each asset's own currency."))

    st.markdown("### 2. Read one asset in plain words")
    pick = st.selectbox("Choose an asset", list(sub_t.index) if len(sub_t) else list(table.index), key="risk_pick")
    for line in explain_asset(table.loc[pick], drawdown_episodes(cours[pick]), local_ccy[pick], shown_ccy[pick], per=int(amount)):
        st.markdown("- " + line)

    st.markdown("### 3. Two pictures")
    if len(sub_t):
        bar = pd.DataFrame({"Asset": sub_t.index, "Worst fall (%)": -sub_t["Max drawdown"].values * 100,
                            "Risk level": sub_t["Risk level"].values}).sort_values("Worst fall (%)")
        fig = px.bar(bar, x="Worst fall (%)", y="Asset", color="Risk level", orientation="h",
                     color_discrete_map=LEVEL_COLORS, title="How far each asset fell at its worst moment")
        fig.update_layout(height=max(420, 16 * len(bar)))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Longer bar = deeper fall. Colours show the risk level.")
        cloud = pd.DataFrame({"Asset": sub_t.index, "Typical yearly move (%)": sub_t["Volatility"].values * 100,
                              "Average yearly return (%)": sub_t["Annual perf"].values * 100,
                              "Category": sub_t["Category"].values})
        fig = px.scatter(cloud, x="Typical yearly move (%)", y="Average yearly return (%)", color="Category",
                         hover_name="Asset", title="Return versus risk")
        fig.update_traces(marker_size=9)
        fig.update_layout(height=520)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("**How to read:** the higher a dot, the more it earned per year; the further right, the more it swung. "
                   "Dots at the top left earned a lot with little swinging (what you would like to find). "
                   "Dots at the bottom right swung a lot for little reward.")

    with st.expander("🔬 Technical table for advanced readers"):
        rcols = ["Category", "Risk score /100", "Risk level", "Local ccy", "Volatility", "Max drawdown",
                 "Current drawdown", "VaR 95% (1d)", "CVaR 95% (1d)", "Skewness", "Excess kurtosis",
                 "Worst month", "% negative months", "Longest underwater (days)", "Beta", "Corr. to benchmark"]
        dash = table[rcols].sort_values("Risk score /100", ascending=False)
        st.dataframe(dash.style.format(FMT, na_rep="-"), use_container_width=True, height=500)
        st.download_button("⬇️ Export risk table (CSV, opens in Excel)", csv_bytes(dash), "risk_table.csv", "text/csv")
    with st.expander("📖 Glossary in plain words"):
        st.markdown(
            "- **Volatility (typical yearly move)**: how much the price typically swings over a year. 15% is calm, 40%+ is wild.\n"
            "- **Max drawdown (worst fall)**: the biggest drop from a peak to the next low over the period.\n"
            "- **Current drawdown (today vs its peak)**: how far the price is from its peak today.\n"
            "- **VaR 95% (loss on 1 bad day in 20)**: on 19 days out of 20 the loss was smaller than this.\n"
            "- **CVaR 95%**: the *average* loss on those worst 1-in-20 days. Always worse than VaR.\n"
            "- **Skewness**: negative means big losses are more frequent than big gains.\n"
            "- **Excess kurtosis**: high means extreme days (up or down) are more frequent than usual.\n"
            "- **Worst month / months in the red**: how bad, and how frequent, losing months are.\n"
            "- **Longest underwater**: the longest stretch spent below a previous peak.\n"
            "- **Beta**: 1 = moves like the benchmark, above 1 amplifies it, below 1 dampens it.\n"
            "- **Currency**: with the EUR switch on, a US or Japanese share also carries dollar or yen risk.")

# ---------------------------------------------------------------- 5. Correlation
with t_corr:
    scope_c = st.selectbox("Scope", ["All assets"] + cats_present, key="corr_scope")
    names_c = list(cours.keys()) if scope_c == "All assets" else [n for n in cours if category[n] == scope_c]
    if len(names_c) < 2:
        st.info("At least 2 assets are needed.")
    else:
        correl = wide[names_c].pct_change(fill_method=None).corr()
        fig = px.imshow(correl, text_auto=".2f" if len(names_c) <= 25 else False,
                        color_continuous_scale="RdYlGn_r", zmin=-1, zmax=1)
        fig.update_layout(height=max(500, 22 * len(names_c)))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Close to 1: the assets move together (little diversification). Close to 0: independent. "
                   "Markets have different opening hours (Tokyo, Paris, New York), which slightly "
                   "understates correlations.")

# ---------------------------------------------------------------- 6. Asset sheet
with t_asset:
    name = st.selectbox("Choose an asset", list(cours.keys()), key="sheet_asset")
    s, l = cours[name], table.loc[name]
    r = s.pct_change().dropna()
    st.markdown(f"### {name} · {l['Signal']} · Risk: {l['Risk level']} · Score {l['Score /100']:.0f}/100")
    st.info("**Why this signal:** " + str(l["Why"]))
    eps_a = drawdown_episodes(s)
    st.markdown("**Risk in plain words**")
    for line in explain_asset(l, eps_a, local_ccy[name], shown_ccy[name]):
        st.markdown("- " + line)
    st.caption(f"Ticker: {assets[name]} · Category: {category[name]} · Local currency: {local_ccy[name]} · "
               f"Prices shown in: {shown_ccy[name]} · {len(s)} sessions "
               f"({s.index[0]:%d/%m/%Y} → {s.index[-1]:%d/%m/%Y})")

    st.markdown("**Performance**")
    a, b, c, d = st.columns(4)
    a.metric("Last price", f"{l['Last price']:,.2f}", f"{l['1M perf']:.1%} over 1 month")
    b.metric("Total perf", f"{l['Total perf']:.1%}")
    c.metric("Average yearly perf", f"{l['Annual perf']:.1%}")
    d.metric("vs 200-day average", f"{l['vs MA200']:.1%}" if pd.notna(l["vs MA200"]) else "-",
             help="Positive: price above its long-term trend.")
    st.markdown("**Risk**")
    a, b, c, d = st.columns(4)
    a.metric("Yearly volatility", f"{l['Volatility']:.1%}", help="Typical size of price swings over a year.")
    b.metric("Max drawdown", f"{l['Max drawdown']:.1%}", help="Worst peak-to-trough fall over the period.")
    c.metric("Current drawdown", f"{l['Current drawdown']:.1%}", help="Distance from the all-time high of the period.")
    under = int(l["Longest underwater (days)"])
    d.metric("Longest underwater", f"{under} sessions", help="Longest time spent below a previous peak.")
    a, b, c, d = st.columns(4)
    a.metric("VaR 95% (1 day)", f"{l['VaR 95% (1d)']:.2%}", help="On 95% of days the loss did not exceed this.")
    b.metric("CVaR 95% (1 day)", f"{l['CVaR 95% (1d)']:.2%}", help="Average loss on the worst 5% of days.")
    c.metric("Worst day", f"{l['Worst day']:.1%}")
    d.metric("Best day", f"{l['Best day']:.1%}")
    a, b, c, d = st.columns(4)
    a.metric("Skewness", f"{l['Skewness']:.2f}", help="Negative: big losses more frequent than big gains.")
    b.metric("Excess kurtosis", f"{l['Excess kurtosis']:.1f}", help="High: extreme days happen more often than usual.")
    c.metric("Worst month", f"{l['Worst month']:.1%}" if pd.notna(l["Worst month"]) else "-")
    d.metric("% negative months", f"{l['% negative months']:.0%}" if pd.notna(l["% negative months"]) else "-")
    st.markdown("**Risk-adjusted return and market link**")
    a, b, c, d = st.columns(4)
    a.metric("Sharpe ratio", f"{l['Sharpe']:.2f}", help="Return per unit of risk (simplified, no risk-free rate).")
    b.metric("Sortino ratio", f"{l['Sortino']:.2f}", help="Like Sharpe, but only penalises downside moves.")
    c.metric("Beta vs benchmark", f"{l['Beta']:.2f}" if pd.notna(l["Beta"]) else "-",
             help="1 = follows the market; >1 amplifies it; <1 dampens it.")
    d.metric("RSI (14)", f"{l['RSI (14)']:.0f}", help=">70 often read as overbought, <30 as oversold.")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=s.index, y=s, name="Price"))
    fig.add_trace(go.Scatter(x=s.index, y=s.rolling(50).mean(), name="50-day average"))
    fig.add_trace(go.Scatter(x=s.index, y=s.rolling(200).mean(), name="200-day average"))
    fig.update_layout(height=420, title="Price and moving averages")
    st.plotly_chart(fig, use_container_width=True)

    g1, g2 = st.columns(2)
    dd = (s / s.cummax() - 1) * 100
    fig = px.area(dd, labels={"value": "%", "index": ""}, title="Drop from peak")
    fig.update_layout(showlegend=False, height=320)
    g1.plotly_chart(fig, use_container_width=True)
    pts = signal_components(s)["points"].dropna()
    fig = px.line(pts, labels={"value": "points", "index": ""}, title="Signal points over time (≥3 BUY, ≤-3 SELL)")
    fig.add_hline(y=3, line_dash="dash", line_color="green")
    fig.add_hline(y=-3, line_dash="dash", line_color="red")
    fig.update_layout(showlegend=False, height=320)
    g2.plotly_chart(fig, use_container_width=True)
    g3, g4 = st.columns(2)
    vol_roll = r.rolling(60).std() * np.sqrt(252) * 100
    fig = px.line(vol_roll, labels={"value": "%", "index": ""}, title="Rolling volatility (60 sessions)")
    fig.update_layout(showlegend=False, height=320)
    g3.plotly_chart(fig, use_container_width=True)
    fig = px.histogram(r * 100, nbins=60, labels={"value": "Daily return (%)"}, title="Daily returns distribution")
    fig.add_shape(type="line", x0=l["VaR 95% (1d)"] * 100, x1=l["VaR 95% (1d)"] * 100, y0=0, y1=1,
                  yref="paper", line=dict(color="red", dash="dash"))
    fig.update_layout(showlegend=False, height=320)
    g4.plotly_chart(fig, use_container_width=True)

    with st.expander("📉 Five worst drawdowns of this asset"):
        eps = eps_a
        if eps.empty:
            st.caption("No drawdown episode on this period.")
        else:
            eps_show = eps.copy()
            for col_ in ("Peak", "Trough", "Recovered on"):
                eps_show[col_] = eps_show[col_].apply(lambda x: "ongoing" if pd.isna(x) else f"{x:%d/%m/%Y}")
            st.dataframe(eps_show.style.format({"Depth": "{:.1%}", "Days to recover": "{:.0f}"}, na_rep="-"),
                         use_container_width=True, hide_index=True)

    with st.expander("🏢 Fundamentals and news (when available)"):
        info = asset_info(assets[name])
        fd, lines = info["fund"], []
        if fd.get("Name"):
            lines.append(f"**Name**: {fd['Name']}")
        if fd.get("Sector"):
            lines.append(f"**Sector**: {fd['Sector']}")
        if fd.get("P/E (trailing)"):
            lines.append(f"**P/E (trailing)**: {fd['P/E (trailing)']:.1f}")
        if fd.get("Market cap"):
            lines.append(f"**Market cap**: {fd['Market cap'] / 1e9:,.1f} bn {fd.get('Currency') or ''}")
        if info["dividend_12m"] is not None and l["Last price"] > 0:
            lines.append(f"**Dividend yield (last 12 months)**: {info['dividend_12m'] / l['Last price']:.1%}")
        if lines:
            for line in lines:
                st.markdown(line)
        else:
            st.caption("No fundamentals available for this asset.")
        st.markdown("**Latest news**")
        if info["news"]:
            for title, link, source in info["news"]:
                st.markdown(f"- [{title}]({link})" + (f" — *{source}*" if source else ""))
        else:
            st.caption("No news available for this asset.")

# ---------------------------------------------------------------- 7. Portfolio
if "init_amounts" not in st.session_state:
    st.session_state.init_amounts = {}
    st.session_state.version = 0
version = st.session_state.version

with t_pf:
    st.subheader("My portfolio")
    st.caption("Enter amounts (actual or planned) in the right-hand column; use the table's search icon to "
               "find an asset. The amounts shown are **examples**: replace them. Nothing is saved on the "
               "server: export your portfolio as CSV to find it again. If your app is public, do not "
               "share the link or your screen while real amounts are displayed.")
    base_df = pd.DataFrame({
        "Asset": list(cours.keys()), "Category": [category[n] for n in cours],
        "Amount (€)": [float(st.session_state.init_amounts.get(n, EXAMPLE_AMOUNTS.get(n, 0))) for n in cours]})
    edited = st.data_editor(
        base_df, key=f"editor_{version}", hide_index=True, use_container_width=True, height=380,
        disabled=["Asset", "Category"],
        column_config={"Amount (€)": st.column_config.NumberColumn(min_value=0, step=100, format="%.0f")})
    amounts = edited.set_index("Asset")["Amount (€)"].fillna(0).clip(lower=0)
    total = float(amounts.sum())
    held = amounts[amounts > 0].index.tolist()
    weights = amounts[held] / total if total > 0 else pd.Series(dtype=float)

    cA, cB = st.columns(2)
    cA.download_button("💾 Export my portfolio (CSV)", csv_bytes(amounts.rename("Amount (€)").to_frame()),
                       "my_portfolio.csv", "text/csv")
    up = cB.file_uploader("📂 Import a portfolio (CSV exported here)", type="csv", key=f"upload_{version}")
    if up is not None:
        try:
            imp = pd.read_csv(up, sep=None, engine="python", decimal=",")
            vals = pd.to_numeric(imp.iloc[:, 1], errors="coerce").fillna(0)
            st.session_state.init_amounts = dict(zip(imp.iloc[:, 0].astype(str), vals))
            st.session_state.version += 1
            st.rerun()
        except Exception:
            st.error("Unreadable file: use a CSV exported from this app.")

    total = float(amounts.sum())
    if total <= 0:
        st.info("Enter at least one amount to compute your portfolio risk.")
    else:
        held = amounts[amounts > 0].index.tolist()
        weights = amounts[held] / total
        px_pf = wide[held].dropna()
        if len(px_pf) < 60:
            st.warning("Common history too short to compute portfolio risk.")
        else:
            rets_df = px_pf.pct_change().dropna()
            r_pf = portfolio_returns(px_pf, weights)
            sp = return_stats(r_pf)
            contrib, vol_pf, divers = risk_contributions(rets_df, weights)
            st.caption(f"Common history used: {px_pf.index[0]:%d/%m/%Y} → {px_pf.index[-1]:%d/%m/%Y} "
                       f"({len(px_pf)} sessions). The most recent holding limits the length.")
            a, b, c, d = st.columns(4)
            a.metric("Total value", f"{total:,.0f} €")
            b.metric("Yearly volatility", f"{sp['Volatility']:.1%}")
            c.metric("Simulated max drawdown", f"{sp['Max drawdown']:.1%}", f"≈ {total * sp['Max drawdown']:,.0f} €",
                     delta_color="off", help="Worst fall this portfolio would have suffered on the common history.")
            d.metric("VaR 95% (1 day)", f"{sp['VaR 95% (1d)']:.2%}", f"≈ {total * sp['VaR 95% (1d)']:,.0f} €",
                     delta_color="off", help="On 95% of days the loss did not exceed this.")
            a, b, c, d = st.columns(4)
            a.metric("Simulated yearly perf", f"{sp['Annual perf']:.1%}")
            b.metric("Sharpe", f"{sp['Sharpe']:.2f}")
            c.metric("Diversification ratio", f"{divers:.2f}", help="1.0 = no diversification; higher = assets offset each other.")
            d.metric("Number of holdings", f"{len(held)}")

            cat_s = pd.Series({n: category[n] for n in held})
            df_cat = pd.DataFrame({"Share of capital (%)": weights.groupby(cat_s).sum() * 100,
                                   "Share of risk (%)": contrib.groupby(cat_s).sum() * 100})
            df_cat = df_cat.reset_index(names="Category").melt(id_vars="Category", var_name="Measure", value_name="%")
            fig = px.bar(df_cat, x="Category", y="%", color="Measure", barmode="group",
                         title="Share of capital vs share of risk, by category")
            st.plotly_chart(fig, use_container_width=True)
            st.caption("If the 'share of risk' bar is taller than the 'share of capital' bar, that category "
                       "weighs more in your risk than in your money.")
            detail = pd.DataFrame({
                "Category": cat_s, "Signal": table.loc[held, "Signal"], "Local ccy": table.loc[held, "Local ccy"],
                "Amount (€)": amounts[held],
                "Weight": weights, "Share of risk": contrib, "Asset volatility": table.loc[held, "Volatility"],
                "Asset max drawdown": table.loc[held, "Max drawdown"]}).sort_values("Share of risk", ascending=False)
            st.dataframe(detail.style.format({"Amount (€)": "{:,.0f}", "Weight": "{:.1%}", "Share of risk": "{:.1%}",
                                              "Asset volatility": "{:.1%}", "Asset max drawdown": "{:.1%}"}),
                         use_container_width=True)
            curve = (1 + r_pf).cumprod() * total
            fig = px.line(curve, labels={"value": "€", "index": ""},
                          title="Simulated portfolio value (constant weights, no fees)")
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------- 7b. Portfolio risk
with t_prisk:
    st.subheader("🛡️ Portfolio risk, explained simply")
    if total <= 0:
        st.info("Enter your amounts in the 💼 Portfolio tab first.")
    else:
        px_r = wide[held].dropna()
        if len(px_r) < 120:
            st.warning("Common history too short (about 6 months needed) to analyse portfolio risk.")
        else:
            rets_r = px_r.pct_change().dropna()
            r_p = portfolio_returns(px_r, weights)
            rep_ = portfolio_risk_report(r_p, ref_ret)
            curve_p = (1 + r_p).cumprod()
            conc = concentration(weights)
            ccy_w = weights.groupby(pd.Series({n: local_ccy[n] for n in held})).sum().sort_values(ascending=False) * 100
            non_eur = (100 - ccy_w.get("EUR", 0)) / 100
            divers = risk_contributions(rets_r, weights)[2]
            score_p = risk_score_from(rep_["Volatility"], rep_["Max drawdown"], rep_["CVaR 95% (1d)"])
            label_p = risk_label(score_p)
            checks = portfolio_checks(rep_, conc, non_eur, divers, profile)

            st.caption(f"Based on your current mix over the common history {px_r.index[0]:%d/%m/%Y} → "
                       f"{px_r.index[-1]:%d/%m/%Y} ({len(px_r) / 252:.1f} years), "
                       + ("in euros." if to_eur_on else "in local currencies (switch on 'Show everything in EUR' for the euro view)."))
            st.markdown(f"## Overall risk level: {label_p}  ·  score {score_p:.0f}/100")
            st.caption("0 = very calm, 100 = very wild. " + RISK_LEVEL_GUIDE.get(label_p, ""))
            st.markdown("### In plain words")
            for line in explain_portfolio(total, rep_, conc, non_eur, len(px_r) / 252, label_p, score_p, checks, profile):
                st.markdown("- " + line)

            st.markdown("### Health check")
            st.dataframe(checks.set_index("Check"), use_container_width=True)
            st.caption(f"Compared with a **{profile}** profile (change it in the sidebar). These are common rules of thumb, "
                       "not rules: ❌ means 'look closer at this', not 'this is wrong'. Your own goals and time horizon matter more.")

            with st.expander("📊 Key numbers"):
                a, b, c, d = st.columns(4)
                a.metric("Typical yearly swing", f"±{rep_['Volatility']:.1%}")
                b.metric("Loss on a bad day (1 in 20)", f"{rep_['VaR 95% (1d)']:.2%}", f"≈ {total * rep_['VaR 95% (1d)']:,.0f} €", delta_color="off")
                c.metric("Loss on a very bad day (1 in 100)", f"{rep_['VaR 99% (1d)']:.2%}", f"≈ {total * rep_['VaR 99% (1d)']:,.0f} €", delta_color="off")
                d.metric("Average loss on the worst days", f"{rep_['CVaR 95% (1d)']:.2%}", f"≈ {total * rep_['CVaR 95% (1d)']:,.0f} €", delta_color="off")
                a, b, c, d = st.columns(4)
                a.metric("Worst fall", f"{rep_['Max drawdown']:.1%}", f"≈ {total * rep_['Max drawdown']:,.0f} €", delta_color="off")
                b.metric("Worst month", f"{rep_['Worst month']:.1%}" if pd.notna(rep_["Worst month"]) else "-")
                c.metric("Months in the red", f"{rep_['% negative months']:.0%}" if pd.notna(rep_["% negative months"]) else "-")
                d.metric("Beta (follows the market?)", f"{rep_['Beta']:.2f}" if pd.notna(rep_["Beta"]) else "-",
                         help="1 = moves like the benchmark; above 1 amplifies it; below 1 dampens it.")
                a, b, c, d = st.columns(4)
                a.metric("Skewness", f"{rep_['Skewness']:.2f}", help="Negative: big losses are more frequent than big gains.")
                b.metric("Extreme days (kurtosis)", f"{rep_['Excess kurtosis']:.1f}", help="High: extreme days are more frequent than usual.")
                c.metric("Worst day", f"{rep_['Worst day']:.1%}")
                d.metric("Best day", f"{rep_['Best day']:.1%}")

            with st.expander("📉 When did it hurt? (worst falls)"):
                st.caption("The first chart shows, for each day, how far your portfolio was below its previous high. "
                           "Deep valleys are the painful periods; the table lists the five worst, with how long recovery took.")
                g1, g2 = st.columns(2)
                fig = px.area((curve_p / curve_p.cummax() - 1) * 100, labels={"value": "% below previous high", "index": ""},
                              title="Distance below the previous high")
                fig.update_layout(showlegend=False, height=330)
                g1.plotly_chart(fig, use_container_width=True)
                fig = px.line(r_p.rolling(60).std() * np.sqrt(252) * 100, labels={"value": "% per year", "index": ""},
                              title="How wild the ride was (rolling 60 sessions)")
                fig.update_layout(showlegend=False, height=330)
                g2.plotly_chart(fig, use_container_width=True)
                eps = drawdown_episodes(curve_p)
                if not eps.empty:
                    eps_show = eps.copy()
                    for col_ in ("Peak", "Trough", "Recovered on"):
                        eps_show[col_] = eps_show[col_].apply(lambda x: "not yet" if pd.isna(x) else f"{x:%d/%m/%Y}")
                    st.dataframe(eps_show.style.format({"Depth": "{:.1%}", "Days to recover": "{:.0f}"}, na_rep="-"),
                                 use_container_width=True, hide_index=True)

            with st.expander("🧩 Concentration: is too much in too few lines?"):
                st.caption("Spreading money over many different lines limits the damage if one of them collapses.")
                a, b, c = st.columns(3)
                a.metric("Biggest line", f"{conc['top_name']} · {conc['top_weight']:.0%}")
                b.metric("3 biggest lines", f"{conc['top3']:.0%}")
                c.metric("Number of real positions", f"{conc['effective_n']:.1f}",
                         help="How many equal-sized lines would give the same concentration.")
                wdf = weights.sort_values(ascending=False).head(15) * 100
                fig = px.bar(wdf, labels={"value": "% of portfolio", "index": ""}, title="Your 15 biggest lines")
                fig.update_layout(showlegend=False, height=380)
                st.plotly_chart(fig, use_container_width=True)
                cat_s = pd.Series({n: category[n] for n in held})
                d2 = pd.DataFrame({"Share of capital (%)": weights.groupby(cat_s).sum() * 100,
                                   "Share of risk (%)": risk_contributions(rets_r, weights)[0].groupby(cat_s).sum() * 100})
                d2 = d2.reset_index(names="Category").melt(id_vars="Category", var_name="Measure", value_name="%")
                fig = px.bar(d2, x="Category", y="%", color="Measure", barmode="group",
                             title="Share of your money vs share of your risk, by category")
                st.plotly_chart(fig, use_container_width=True)
                st.caption("If the 'risk' bar is taller than the 'money' bar, that category causes more of your ups and downs "
                           "than its size suggests.")

            with st.expander("💱 Currency: do exchange rates matter for you?"):
                a, b = st.columns([1, 2])
                a.metric("Money outside the euro", f"{non_eur:.0%}")
                fig = px.bar(ccy_w, labels={"value": "% of portfolio", "index": "Currency of the asset"},
                             title="Your portfolio by currency")
                fig.update_layout(showlegend=False, height=320)
                b.plotly_chart(fig, use_container_width=True)
                st.caption("Example: you hold a dollar share that is flat in dollars. If the dollar loses 10% against the euro, "
                           "your holding is worth 10% less in euros. With the EUR switch on, this effect is already inside every "
                           "figure of this app. Euro-hedged ETFs remove most of it.")

            with st.expander("🔥 Stress tests: what if a crisis happened again?"):
                st.caption("We replay past crises on your current mix. 'Portfolio return' is what your portfolio would "
                           "have done between the market peak and low of that crisis. Dates are approximate S&P 500 "
                           "peak-to-low dates. 'Coverage' is the share of your portfolio that already existed then: "
                           "a low coverage means the result says little about you.")
                if st.checkbox("Run historical stress tests (downloads the full history of your holdings)", key="run_stress"):
                    fx_h = {fx_pair(local_ccy[n]) for n in held if local_ccy[n] not in ("EUR", "?")} if to_eur_on else set()
                    tk_long = tuple(dict.fromkeys([assets[n] for n in held] + [p_ for p_, _ in fx_h]))
                    with st.spinner("Downloading full histories..."):
                        long_px = download_prices(tk_long, "max", time_block)
                    long_series = {}
                    for n in held:
                        if len(long_px) and assets[n] in long_px.columns:
                            s_l = long_px[assets[n]].dropna()
                            if to_eur_on and local_ccy[n] not in ("EUR", "?"):
                                pair, scale = fx_pair(local_ccy[n])
                                s_l = to_eur(s_l, long_px[pair], scale) if pair in long_px.columns else None
                            if s_l is not None and len(s_l) > 30:
                                long_series[n] = s_l
                    rows = []
                    for label, (d0, d1) in STRESS_WINDOWS.items():
                        res = stress_test(long_series, weights, d0, d1)
                        if res is None:
                            rows.append({"Crisis": label, "Portfolio return": np.nan, "Impact (€)": np.nan,
                                         "Coverage": 0.0, "Hardest-hit holding": "no data"})
                        else:
                            rows.append({"Crisis": label, "Portfolio return": res["return"],
                                         "Impact (€)": total * res["return"], "Coverage": res["coverage"],
                                         "Hardest-hit holding": f"{res['worst_asset']} ({res['worst_ret']:.0%})"})
                    st.dataframe(pd.DataFrame(rows).set_index("Crisis").style.format(
                        {"Portfolio return": "{:.1%}", "Impact (€)": "{:,.0f}", "Coverage": "{:.0%}"}, na_rep="-"),
                        use_container_width=True)
                st.markdown("**Your own what-if:** an instant fall, without any diversification effect.")
                cats_held = sorted({category[n] for n in held})
                sel_cats = st.multiselect("Categories hit by the shock", cats_held,
                                          default=[c_ for c_ in cats_held if c_ != CORE_CATEGORY], key="whatif_cats")
                shock = st.slider("Price fall on these categories (%)", -60, 0, -20, key="whatif_shock")
                loss = sum(float(amounts[n]) for n in held if category[n] in sel_cats) * shock / 100
                st.metric("Estimated loss", f"{loss:,.0f} €", f"{loss / total:.1%} of your portfolio", delta_color="off")

            with st.expander("🎲 Range of possible futures (Monte Carlo)"):
                st.caption("We shuffle blocks of your portfolio's past days 2,000 times to draw 2,000 possible futures. "
                           "It shows a *range*, not a prediction.")
                m1, m2, m3 = st.columns(3)
                years = m1.selectbox("Horizon", [1, 3, 5], index=0, format_func=lambda y: f"{y} year(s)", key="mc_h")
                mu = m2.slider("Assumed average yearly return (%)", -5, 15, 5, key="mc_mu")
                use_hist = m3.checkbox("Use the historical average return instead", value=False, key="mc_hist",
                                       help="Past averages are often flattered by good periods: use with care.")
                paths, final, mdd = monte_carlo(r_p, int(years * 252), annual_return=None if use_hist else mu / 100)
                pcts = np.percentile(final, [5, 25, 50, 75, 95])
                st.markdown(f"**In plain words:** over {years} year(s), 9 simulated futures out of 10 end between "
                            f"**{pcts[0] * total:,.0f} €** and **{pcts[4] * total:,.0f} €** (middle case: {pcts[2] * total:,.0f} €). "
                            f"About **{(final < 1).mean():.0%}** of them end below today's {total:,.0f} €, and "
                            f"**{(mdd <= -0.20).mean():.0%}** go through a fall of 20% or more along the way.")
                labels_ = ["Bad case (1 in 20 is worse)", "Below average", "Middle case", "Above average", "Good case (1 in 20 is better)"]
                mc_df = pd.DataFrame({"Portfolio value (€)": pcts * total, "Change": pcts - 1}, index=labels_)
                st.dataframe(mc_df.style.format({"Portfolio value (€)": "{:,.0f}", "Change": "{:+.1%}"}), use_container_width=True)
                band = np.percentile(paths * total, [5, 25, 50, 75, 95], axis=0)
                xs = np.arange(paths.shape[1]) / 252
                fig = go.Figure()
                for q, nm in zip(band, ["Bad case (5%)", "25%", "Middle case", "75%", "Good case (95%)"]):
                    fig.add_trace(go.Scatter(x=xs, y=q, mode="lines", name=nm, line=dict(width=2 if nm == "Middle case" else 1)))
                fig.update_layout(title="Simulated portfolio value (€)", xaxis_title="Years from today",
                                  yaxis_title="€", height=420)
                st.plotly_chart(fig, use_container_width=True)
                st.caption("The middle line is the median outcome. Half of the simulated futures stay between the 25% and 75% lines; "
                           "nine out of ten stay between the outer lines. A simulation cannot foresee events unlike those in the past.")

# ---------------------------------------------------------------- 8. Core vs themes
with t_core:
    st.subheader("Simulate a split between safe core and thematic assets")
    core_all = [n for n in cours if category[n] == CORE_CATEGORY]
    theme_all = [n for n in cours if category[n] != CORE_CATEGORY]
    total = float(amounts.sum())
    if total <= 0:
        st.info("Enter your amounts in the 💼 Portfolio tab first.")
    elif not core_all or not theme_all:
        st.info("At least one core ETF AND one thematic asset must be loaded.")
    else:
        def split_weights(group):
            m = amounts.reindex(group).fillna(0)
            m = m[m > 0] if m.sum() > 0 else pd.Series(1.0, index=group)
            return m / m.sum()

        w_core, w_theme = split_weights(core_all), split_weights(theme_all)
        cols_sim = sorted(set(w_core.index) | set(w_theme.index) | set(amounts[amounts > 0].index))
        px_sim = wide[cols_sim]

        def scenario(core_share):
            return pd.concat([w_core * core_share, w_theme * (1 - core_share)]).reindex(cols_sim).fillna(0)

        p_core = st.slider("Share of the safe core in your portfolio (%)", 0, 100, 80, step=5)
        scenarios = {
            "My current portfolio": (amounts / total).reindex(cols_sim).fillna(0),
            f"Custom: {p_core}% core / {100 - p_core}% themes": scenario(p_core / 100),
            "100% core": scenario(1.0), "90% core / 10% themes": scenario(0.9),
            "80% core / 20% themes": scenario(0.8), "60% core / 40% themes": scenario(0.6),
            "100% themes": scenario(0.0)}
        rows, curves = {}, {}
        for nm, w in scenarios.items():
            rr = portfolio_returns(px_sim, w)
            stt = return_stats(rr)
            if not stt:
                continue
            stt["Max loss on your capital (€)"] = total * stt["Max drawdown"]
            rows[nm], curves[nm] = stt, (1 + rr).cumprod() * 100
        if not rows:
            st.warning("Not enough history for the simulation.")
        else:
            comp = pd.DataFrame(rows).T[["Annual perf", "Volatility", "Max drawdown",
                                         "Max loss on your capital (€)", "VaR 95% (1d)", "Sharpe"]]
            st.dataframe(comp.style.format({"Annual perf": "{:.1%}", "Volatility": "{:.1%}", "Max drawdown": "{:.1%}",
                                            "Max loss on your capital (€)": "{:,.0f}", "VaR 95% (1d)": "{:.2%}",
                                            "Sharpe": "{:.2f}"}), use_container_width=True)
            show = [k for k in curves if k in ("My current portfolio", "100% core", "100% themes") or k.startswith("Custom")]
            fig = px.line(pd.DataFrame({k: curves[k] for k in show}),
                          labels={"value": "Base 100", "index": "", "variable": ""}, title="Simulated growth (base 100)")
            fig.update_layout(legend_title_text="")
            st.plotly_chart(fig, use_container_width=True)
            st.caption(f"Core: {', '.join(core_all)}. Themes: {len(theme_all)} thematic assets. Inside each group "
                       "weights follow your amounts (equal weights if none entered); assets join from their "
                       "first available day. Historical simulation, local currencies, no fees or taxes: it does not predict the future.")

# ---------------------------------------------------------------- 9. Strategy Lab
with t_lab:
    st.subheader("🧪 Strategy Lab")
    st.caption("Test a trading rule against simply holding the asset. A signal is decided at the close and "
               "applied from the next session (no look-ahead). Strategy and buy & hold are compared over "
               "exactly the same period. Out of the market = cash earning 0%.")
    c1, c2, c3 = st.columns(3)
    scope = c1.radio("Test on", ["Single asset", "Category basket"], horizontal=True)
    if scope == "Single asset":
        target = c2.selectbox("Asset", list(cours.keys()), key="lab_asset")
        names = [target]
    else:
        lab_cat = c2.selectbox("Category (equal-weight basket)", cats_present, key="lab_cat")
        names = [n for n in cours if category[n] == lab_cat]
    strategy = c3.selectbox("Strategy", STRATEGIES, index=1)

    params = {}
    p1, p2, p3 = st.columns(3)
    if strategy.startswith("Price above"):
        params["window"] = p1.slider("Moving-average window (days)", 20, 250, 200, 10)
    elif strategy.startswith("Golden"):
        params["fast"] = p1.slider("Fast average (days)", 10, 100, 50, 5)
        params["slow"] = p2.slider("Slow average (days)", 100, 300, 200, 10)
    elif strategy.startswith("Momentum"):
        params["lookback"] = p1.slider("Look-back (days)", 21, 252, 126, 21)
    elif strategy.startswith("RSI"):
        params["rsi_buy"] = p1.slider("Buy when RSI falls below", 10, 50, 30)
        params["rsi_sell"] = p2.slider("Sell when RSI rises above", 40, 90, 55)
    elif strategy.startswith("Terminal"):
        params["max_vol"] = MAX_VOL
        p1.info(f"Uses your risk profile: {profile} (≤ {MAX_VOL:.0%} volatility).")
    cost = p3.slider("Cost per position change (%)", 0.0, 1.0, 0.10, 0.05) / 100
    split_pct = st.slider("In-sample share of the history (%)", 30, 90, 60, 5,
                          help="The first part is where you would 'design' the rule, the last part is a "
                               "test it has never seen. A rule that only works in-sample is probably overfitted.")

    runs = {}
    for n in names:
        bt = run_backtest(cours[n], strategy, cost, **params)
        if bt is not None and len(bt["strategy"]) >= 60:
            runs[n] = bt
    if not runs:
        st.info("Not enough history for this strategy: use a longer history or shorter windows.")
    else:
        strat_r = pd.concat({n: b["strategy"] for n, b in runs.items()}, axis=1).mean(axis=1)
        hold_r = pd.concat({n: b["hold"] for n, b in runs.items()}, axis=1).mean(axis=1)
        exposure = pd.concat({n: b["position"] for n, b in runs.items()}, axis=1).mean(axis=1)
        entries = sum(b["entries"] for b in runs.values())
        cut = strat_r.index[int(len(strat_r) * split_pct / 100)]
        periods = {"Full period": (strat_r, hold_r),
                   "In-sample": (strat_r[strat_r.index < cut], hold_r[hold_r.index < cut]),
                   "Out-of-sample": (strat_r[strat_r.index >= cut], hold_r[hold_r.index >= cut])}
        rows = []
        for pname, (a_, b_) in periods.items():
            for label, rr in (("Strategy", a_), ("Buy & hold", b_)):
                stt = return_stats(rr)
                if stt:
                    rows.append({"Period": pname, "Rule": label, **{k: stt[k] for k in
                                 ["Annual perf", "Volatility", "Max drawdown", "Sharpe", "Sortino", "Calmar"]}})
        res = pd.DataFrame(rows).set_index(["Period", "Rule"])
        st.dataframe(res.style.format({"Annual perf": "{:.1%}", "Volatility": "{:.1%}", "Max drawdown": "{:.1%}",
                                       "Sharpe": "{:.2f}", "Sortino": "{:.2f}", "Calmar": "{:.2f}"}),
                     use_container_width=True)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Test period", f"{strat_r.index[0]:%d/%m/%Y} → {strat_r.index[-1]:%d/%m/%Y}")
        m2.metric("Entries into the market", f"{entries}")
        m3.metric("Average time invested", f"{exposure.mean():.0%}")
        m4.metric("Split date", f"{cut:%d/%m/%Y}")

        eq = pd.DataFrame({"Strategy": (1 + strat_r).cumprod() * 100, "Buy & hold": (1 + hold_r).cumprod() * 100})
        fig = px.line(eq, labels={"value": "Base 100", "index": "", "variable": ""}, title="Growth of 100")
        fig.add_shape(type="line", x0=cut, x1=cut, y0=0, y1=1, yref="paper", line=dict(dash="dot", color="grey"))
        fig.add_annotation(x=cut, y=1, yref="paper", text="in-sample | out-of-sample", showarrow=False, yanchor="bottom")
        fig.update_layout(legend_title_text="")
        st.plotly_chart(fig, use_container_width=True)

        d1, d2 = st.columns(2)
        ddf = pd.DataFrame({c: (eq[c] / eq[c].cummax() - 1) * 100 for c in eq.columns})
        fig = px.line(ddf, labels={"value": "%", "index": "", "variable": ""}, title="Drawdown (drop from peak)")
        fig.update_layout(legend_title_text="")
        d1.plotly_chart(fig, use_container_width=True)
        fig = px.area(exposure * 100, labels={"value": "% invested", "index": ""}, title="Market exposure")
        fig.update_layout(showlegend=False)
        d2.plotly_chart(fig, use_container_width=True)

        if scope == "Single asset" and (strategy.startswith("Price above") or strategy.startswith("Momentum")):
            st.markdown("**Robustness: does the rule work for many parameter values, or only one lucky value?**")
            vals = list(range(20, 260, 10)) if strategy.startswith("Price above") else list(range(21, 253, 21))
            sw = param_sweep(cours[target], strategy, vals, cost)
            if sw is None:
                st.caption("Not enough history for the robustness test.")
            else:
                sweep_df, hold_stats = sw
                s1, s2 = st.columns(2)
                for col, key, title in ((s1, "Annual perf", "Yearly performance vs parameter"),
                                        (s2, "Max drawdown", "Max drawdown vs parameter")):
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=sweep_df["Parameter"], y=sweep_df[key] * 100, mode="lines+markers", name="Rule"))
                    fig.add_hline(y=hold_stats[key] * 100, line_dash="dash", line_color="grey",
                                  annotation_text="Buy & hold")
                    fig.update_layout(title=title, yaxis_title="%", xaxis_title="Parameter (days)", height=340)
                    col.plotly_chart(fig, use_container_width=True)
                st.caption("A robust rule gives similar results across a whole range of values. "
                           "A single isolated peak is usually luck (overfitting).")

        if st.checkbox("Run this strategy on every loaded asset (leaderboard)", key="lab_all"):
            lb = {}
            for n, s_ in cours.items():
                bt = run_backtest(s_, strategy, cost, **params)
                if bt is None or len(bt["strategy"]) < 60:
                    continue
                a_, b_ = return_stats(bt["strategy"]), return_stats(bt["hold"])
                if a_ and b_:
                    lb[n] = {"Category": category[n], "Perf (rule)": a_["Annual perf"], "Perf (hold)": b_["Annual perf"],
                             "Max DD (rule)": a_["Max drawdown"], "Max DD (hold)": b_["Max drawdown"],
                             "Sharpe (rule)": a_["Sharpe"], "Sharpe (hold)": b_["Sharpe"]}
            if lb:
                lbd = pd.DataFrame(lb).T.infer_objects()
                lbd["Sharpe gain"] = lbd["Sharpe (rule)"] - lbd["Sharpe (hold)"]
                n_all = len(lbd)
                st.markdown(f"The rule beat buy & hold on Sharpe for **{int((lbd['Sharpe gain'] > 0).sum())} of {n_all}** assets "
                            f"and reduced the worst drop for **{int((lbd['Max DD (rule)'] > lbd['Max DD (hold)']).sum())} of {n_all}**.")
                st.dataframe(lbd.sort_values("Sharpe gain", ascending=False).style.format({
                    "Perf (rule)": "{:.1%}", "Perf (hold)": "{:.1%}", "Max DD (rule)": "{:.1%}", "Max DD (hold)": "{:.1%}",
                    "Sharpe (rule)": "{:.2f}", "Sharpe (hold)": "{:.2f}", "Sharpe gain": "{:+.2f}"}),
                    use_container_width=True, height=500)

        st.warning("Read before concluding: a backtest describes one past period, ignores taxes and bid/ask "
                   "spreads, and a rule that looks great on history can disappoint afterwards. Compare the "
                   "out-of-sample rows and the maximum drawdown: that is where simple trend rules help most, "
                   "usually at the price of lower returns in strong bull markets.")
