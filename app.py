import time
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh

st.set_page_config(page_title="Mon mini terminal", page_icon="📊", layout="wide")

# ------------------------------------------------------------
# TES ACTIFS PAR DÉFAUT (modifie cette liste comme tu veux)
# Tu peux y mettre des actions OU des ETF/fonds cotés (ex : "IWDA.AS")
# ------------------------------------------------------------
ACTIFS_PAR_DEFAUT = {
    "Toyota": "7203.T",
    "Hyundai": "005380.KS",
    "BYD": "1211.HK",
    "Adidas": "ADS.DE",
    "Asics": "7936.T",
    "TotalEnergies": "TTE.PA",
    "Brent (futures)": "BZ=F",
}

# ------------------------------------------------------------
# BARRE LATÉRALE
# ------------------------------------------------------------
st.sidebar.header("⚙️ Réglages")

choix = st.sidebar.multiselect(
    "Actifs suivis",
    options=list(ACTIFS_PAR_DEFAUT.keys()),
    default=list(ACTIFS_PAR_DEFAUT.keys()),
)
supplementaires = st.sidebar.text_input(
    "Ajouter des tickers (séparés par une virgule)", placeholder="Ex : SHEL.AS, IWDA.AS"
)
periode = st.sidebar.selectbox("Période", ["1y", "2y", "5y", "10y", "max"], index=2)
ticker_ref = st.sidebar.text_input(
    "Indice de référence (pour le bêta)", value="IWDA.AS",
    help="Par défaut : ETF actions mondiales (iShares MSCI World, coté à Amsterdam).",
)

st.sidebar.subheader("🔄 Mise à jour")
auto = st.sidebar.toggle("Actualisation automatique", value=True)
freq = st.sidebar.selectbox("Fréquence", [1, 5, 15, 60], index=1,
                            format_func=lambda m: f"toutes les {m} min")
if auto:
    st_autorefresh(interval=freq * 60 * 1000, key="auto_refresh")
if st.sidebar.button("Actualiser maintenant"):
    st.cache_data.clear()
    st.rerun()

st.sidebar.subheader("🔔 Seuils d'alerte")
seuil_mois = st.sidebar.slider("Baisse sur 1 mois (%)", 3, 30, 10)
seuil_baisse = st.sidebar.slider("Baisse depuis le sommet (%)", 5, 50, 20)
seuil_vol = st.sidebar.slider("Volatilité annuelle (%)", 10, 80, 35)

# Change toutes les "freq" minutes : force le rechargement des données à ce rythme
bloc_temps = int(time.time() // (freq * 60))


# ------------------------------------------------------------
# DONNÉES
# ------------------------------------------------------------
@st.cache_data(ttl=7200, max_entries=200, show_spinner=False)
def telecharger(ticker, periode, bloc_temps):
    try:
        serie = yf.Ticker(ticker).history(period=periode, auto_adjust=True)["Close"]
        if serie.empty:
            return None
        serie.index = serie.index.tz_localize(None).normalize()
        serie = serie[~serie.index.duplicated()].dropna()
        return serie if len(serie) >= 30 else None
    except Exception:
        return None


def niveau_risque(vol):
    if vol < 0.15:
        return "🟢 Faible"
    if vol < 0.25:
        return "🟡 Modéré"
    if vol < 0.40:
        return "🟠 Élevé"
    return "🔴 Très élevé"


def indicateurs(serie, rend_ref=None):
    r = serie.pct_change().dropna()
    annees = max((serie.index[-1] - serie.index[0]).days / 365.25, 0.01)
    perf_annuelle = (serie.iloc[-1] / serie.iloc[0]) ** (1 / annees) - 1
    vol = r.std() * np.sqrt(252)
    vol_baisse = r[r < 0].std() * np.sqrt(252)

    dd = serie / serie.cummax() - 1
    sous_eau = dd < 0
    duree_max = int(sous_eau.groupby((~sous_eau).cumsum()).sum().max())

    var95 = r.quantile(0.05)
    cvar95 = r[r <= var95].mean()
    mm200 = serie.rolling(200).mean().iloc[-1]

    beta = corr_ref = np.nan
    if rend_ref is not None:
        df = pd.concat([r, rend_ref], axis=1, join="inner").dropna()
        if len(df) > 60 and df.iloc[:, 1].var() > 0:
            beta = df.iloc[:, 0].cov(df.iloc[:, 1]) / df.iloc[:, 1].var()
            corr_ref = df.iloc[:, 0].corr(df.iloc[:, 1])

    return {
        "Niveau de risque": niveau_risque(vol),
        "Dernier cours": serie.iloc[-1],
        "Perf. 1 mois": serie.iloc[-1] / serie.iloc[-22] - 1 if len(serie) > 22 else np.nan,
        "Perf. totale": serie.iloc[-1] / serie.iloc[0] - 1,
        "Perf. annuelle": perf_annuelle,
        "Volatilité": vol,
        "Pire baisse": dd.min(),
        "Baisse actuelle": dd.iloc[-1],
        "Durée max sous l'eau (j)": duree_max,
        "VaR 95% (jour)": var95,
        "CVaR 95% (jour)": cvar95,
        "Pire jour": r.min(),
        "Meilleur jour": r.max(),
        "% jours positifs": (r > 0).mean(),
        "Bêta": beta,
        "Corrél. référence": corr_ref,
        "Sharpe": perf_annuelle / vol if vol > 0 else np.nan,
        "Sortino": perf_annuelle / vol_baisse if vol_baisse > 0 else np.nan,
        "Écart vs MM200": serie.iloc[-1] / mm200 - 1 if not np.isnan(mm200) else np.nan,
    }


# ------------------------------------------------------------
# CHARGEMENT
# ------------------------------------------------------------
st.title("📊 Mon mini terminal")

actifs = {nom: ACTIFS_PAR_DEFAUT[nom] for nom in choix}
for t in [x.strip() for x in supplementaires.split(",") if x.strip()]:
    actifs[t.upper()] = t.upper()

if not actifs:
    st.info("Choisis au moins un actif dans la barre latérale.")
    st.stop()

cours = {}
with st.spinner("Téléchargement des données..."):
    for nom, ticker in actifs.items():
        serie = telecharger(ticker, periode, bloc_temps)
        if serie is None:
            st.warning(f"⚠️ {nom} ({ticker}) : données indisponibles, actif ignoré.")
        else:
            cours[nom] = serie
    serie_ref = telecharger(ticker_ref.strip().upper(), periode, bloc_temps) if ticker_ref.strip() else None

if not cours:
    st.error("Aucune donnée n'a pu être téléchargée. Réessaie dans quelques minutes.")
    st.stop()

rend_ref = serie_ref.pct_change().dropna() if serie_ref is not None else None
if rend_ref is None:
    st.caption("ℹ️ Indice de référence indisponible : le bêta ne sera pas calculé.")

tableau = pd.DataFrame({nom: indicateurs(s, rend_ref) for nom, s in cours.items()}).T.infer_objects()
derniere_date = max(s.index[-1] for s in cours.values())
st.caption(
    f"Dernière clôture disponible : **{derniere_date:%d/%m/%Y}** · "
    f"Mise à jour de la page : {time.strftime('%d/%m/%Y %H:%M')} · "
    "Données Yahoo Finance, performances en monnaie locale. "
    "Outil d'analyse, pas un conseil en investissement."
)

# ------------------------------------------------------------
# ALERTES
# ------------------------------------------------------------
alertes = []
for nom, l in tableau.iterrows():
    if l["Perf. 1 mois"] <= -seuil_mois / 100:
        alertes.append(f"📉 **{nom}** : {l['Perf. 1 mois']:.1%} sur 1 mois")
    if l["Baisse actuelle"] <= -seuil_baisse / 100:
        alertes.append(f"🔻 **{nom}** : {l['Baisse actuelle']:.1%} sous son plus haut")
    if l["Volatilité"] >= seuil_vol / 100:
        alertes.append(f"⚡ **{nom}** : volatilité de {l['Volatilité']:.0%} par an")
    if pd.notna(l["Écart vs MM200"]) and l["Écart vs MM200"] < 0:
        alertes.append(f"〰️ **{nom}** : sous sa moyenne mobile à 200 jours")

with st.expander(f"🔔 Alertes ({len(alertes)})", expanded=bool(alertes)):
    if alertes:
        for a in alertes:
            st.markdown(a)
    else:
        st.success("Aucune alerte avec tes seuils actuels.")

# ------------------------------------------------------------
# ONGLETS
# ------------------------------------------------------------
tous = pd.concat(cours, axis=1).sort_index().ffill().dropna(how="all")
onglet1, onglet2, onglet3, onglet4 = st.tabs(
    ["📋 Vue d'ensemble", "⚠️ Risque comparé", "🔗 Corrélation", "🔍 Fiche par actif"]
)

# ---- Vue d'ensemble
with onglet1:
    colonnes = ["Niveau de risque", "Dernier cours", "Perf. 1 mois", "Perf. annuelle", "Volatilité",
                "Pire baisse", "Baisse actuelle", "VaR 95% (jour)", "Bêta", "Sharpe", "Sortino",
                "Écart vs MM200"]
    fmt = {c: "{:.1%}" for c in ["Perf. 1 mois", "Perf. annuelle", "Volatilité", "Pire baisse",
                                  "Baisse actuelle", "VaR 95% (jour)", "Écart vs MM200"]}
    fmt.update({"Dernier cours": "{:,.2f}", "Bêta": "{:.2f}", "Sharpe": "{:.2f}", "Sortino": "{:.2f}"})
    st.dataframe(tableau[colonnes].style.format(fmt, na_rep="-"), use_container_width=True)

    base100 = tous / tous.apply(lambda c: c.dropna().iloc[0]) * 100
    fig = px.line(base100, labels={"value": "Base 100", "index": "", "variable": ""},
                  title="Évolution comparée (base 100 au départ)")
    fig.add_hline(y=100, line_dash="dash", line_color="grey")
    fig.update_layout(height=480, legend_title_text="")
    st.plotly_chart(fig, use_container_width=True)

# ---- Risque comparé
with onglet2:
    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar((tableau["Pire baisse"] * 100).sort_values(), orientation="h",
                     labels={"value": "Pire baisse (%)", "index": ""}, title="Pire baisse historique")
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.bar((tableau["Volatilité"] * 100).sort_values(), orientation="h",
                     labels={"value": "Volatilité annuelle (%)", "index": ""}, title="Volatilité annualisée")
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    fig = px.scatter(tableau.reset_index(), x=tableau["Volatilité"].values * 100,
                     y=tableau["Perf. annuelle"].values * 100, text="index",
                     labels={"x": "Volatilité (%)", "y": "Performance annuelle (%)"},
                     title="Rendement vs risque (en haut à gauche = idéal)")
    fig.update_traces(textposition="top center", marker_size=10)
    st.plotly_chart(fig, use_container_width=True)

# ---- Corrélation
with onglet3:
    if len(cours) < 2:
        st.info("Il faut au moins 2 actifs pour calculer une corrélation.")
    else:
        correl = tous.pct_change().dropna(how="all").corr()
        fig = px.imshow(correl, text_auto=".2f", color_continuous_scale="RdYlGn_r", zmin=-1, zmax=1)
        fig.update_layout(height=550)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Proche de 1 : les deux actifs bougent ensemble (peu de diversification). "
                   "Proche de 0 : ils bougent indépendamment. "
                   "Les places ont des horaires différents (Tokyo, Paris…), ce qui sous-estime un peu les corrélations.")

# ---- Fiche par actif
with onglet4:
    nom = st.selectbox("Choisis un actif", list(cours.keys()))
    s = cours[nom]
    l = tableau.loc[nom]
    r = s.pct_change().dropna()

    st.markdown(f"### {nom} · Niveau de risque : {l['Niveau de risque']}")
    st.caption(f"Ticker : {actifs[nom]} · {len(s)} séances analysées "
               f"({s.index[0]:%d/%m/%Y} → {s.index[-1]:%d/%m/%Y})")

    st.markdown("**Performance**")
    a, b, c, d = st.columns(4)
    a.metric("Dernier cours", f"{l['Dernier cours']:,.2f}", f"{l['Perf. 1 mois']:.1%} sur 1 mois")
    b.metric("Perf. totale", f"{l['Perf. totale']:.1%}")
    c.metric("Perf. annuelle moyenne", f"{l['Perf. annuelle']:.1%}")
    d.metric("Écart vs moyenne 200 j", f"{l['Écart vs MM200']:.1%}" if pd.notna(l["Écart vs MM200"]) else "-",
             help="Positif : le cours est au-dessus de sa tendance long terme.")

    st.markdown("**Risque**")
    a, b, c, d = st.columns(4)
    a.metric("Volatilité annuelle", f"{l['Volatilité']:.1%}",
             help="Amplitude typique des variations sur un an. Plus c'est haut, plus ça bouge.")
    b.metric("Pire baisse historique", f"{l['Pire baisse']:.1%}",
             help="La pire chute entre un sommet et le creux suivant, sur la période.")
    c.metric("Baisse actuelle", f"{l['Baisse actuelle']:.1%}",
             help="Distance entre le cours actuel et son plus haut historique.")
    duree = int(l["Durée max sous l'eau (j)"])
    d.metric("Durée max sous l'eau", f"{duree} séances",
             help="Plus longue période passée sous un ancien sommet avant de le retrouver.")

    a, b, c, d = st.columns(4)
    a.metric("VaR 95% (1 jour)", f"{l['VaR 95% (jour)']:.2%}",
             help="Dans 95 % des jours, la perte n'a pas dépassé ce niveau (historique).")
    b.metric("CVaR 95% (1 jour)", f"{l['CVaR 95% (jour)']:.2%}",
             help="Perte moyenne les jours où on dépasse la VaR : les mauvais jours.")
    c.metric("Pire jour", f"{l['Pire jour']:.1%}")
    d.metric("Meilleur jour", f"{l['Meilleur jour']:.1%}")

    st.markdown("**Rendement ajusté du risque et lien avec le marché**")
    a, b, c, d = st.columns(4)
    a.metric("Ratio de Sharpe", f"{l['Sharpe']:.2f}",
             help="Rendement par unité de risque (simplifié, sans taux sans risque). Plus c'est haut, mieux c'est.")
    b.metric("Ratio de Sortino", f"{l['Sortino']:.2f}",
             help="Comme Sharpe, mais ne pénalise que les baisses.")
    c.metric("Bêta vs référence", f"{l['Bêta']:.2f}" if pd.notna(l["Bêta"]) else "-",
             help="1 = suit le marché ; >1 = amplifie ses mouvements ; <1 = les amortit.")
    d.metric("% de jours positifs", f"{l['% jours positifs']:.0%}")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=s.index, y=s, name="Cours"))
    fig.add_trace(go.Scatter(x=s.index, y=s.rolling(50).mean(), name="Moyenne mobile 50 j"))
    fig.add_trace(go.Scatter(x=s.index, y=s.rolling(200).mean(), name="Moyenne mobile 200 j"))
    fig.update_layout(height=420, title="Cours et moyennes mobiles")
    st.plotly_chart(fig, use_container_width=True)

    g1, g2 = st.columns(2)
    with g1:
        dd = (s / s.cummax() - 1) * 100
        fig = px.area(dd, labels={"value": "%", "index": ""}, title="Baisse depuis le sommet")
        fig.update_layout(showlegend=False, height=320)
        st.plotly_chart(fig, use_container_width=True)
    with g2:
        vol_glissante = r.rolling(60).std() * np.sqrt(252) * 100
        fig = px.line(vol_glissante, labels={"value": "%", "index": ""},
                      title="Volatilité glissante (60 séances)")
        fig.update_layout(showlegend=False, height=320)
        st.plotly_chart(fig, use_container_width=True)

    fig = px.histogram(r * 100, nbins=60, labels={"value": "Rendement quotidien (%)"},
                       title="Distribution des rendements quotidiens")
    fig.add_vline(x=l["VaR 95% (jour)"] * 100, line_dash="dash", line_color="red",
                  annotation_text="VaR 95%")
    fig.update_layout(showlegend=False, height=320)
    st.plotly_chart(fig, use_container_width=True)
