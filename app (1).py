import time
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh

from calculs import (indicateurs, score_synthese, stats_rendements, rendements_portefeuille,
                     contributions_risque, backtest_mm)

st.set_page_config(page_title="Mon mini terminal", page_icon="📊", layout="wide")

# ------------------------------------------------------------
# TES ACTIFS (modifie ces listes comme tu veux)
# Tu peux y mettre des actions OU des ETF/fonds cotés. Vérifie toujours les tickers.
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
THEME_PAR_ACTIF = {
    "Toyota": "Auto asiatique", "Hyundai": "Auto asiatique", "BYD": "Auto asiatique",
    "Adidas": "Running", "Asics": "Running",
    "TotalEnergies": "Pétrole", "Brent (futures)": "Pétrole",
}
# Socle prudent : ETF diversifiés cotés en Europe
SOCLE = {
    "Actions monde (MSCI World)": "IWDA.AS",
    "Obligations État euro": "IEGA.AS",
    "Monétaire (€STR)": "XEON.DE",
}
# Montants d'exemple pour l'onglet portefeuille (à modifier dans l'appli)
EXEMPLE_MONTANTS = {
    "Actions monde (MSCI World)": 4000, "Obligations État euro": 2000, "Monétaire (€STR)": 1000,
    "Toyota": 500, "Hyundai": 500, "BYD": 250, "Adidas": 250, "Asics": 250,
    "TotalEnergies": 250, "Brent (futures)": 0,
}

# ------------------------------------------------------------
# BARRE LATÉRALE
# ------------------------------------------------------------
st.sidebar.header("⚙️ Réglages")

choix = st.sidebar.multiselect("Actifs thématiques", list(ACTIFS_PAR_DEFAUT.keys()),
                               default=list(ACTIFS_PAR_DEFAUT.keys()))
choix_socle = st.sidebar.multiselect("Socle prudent (ETF diversifiés)", list(SOCLE.keys()),
                                     default=list(SOCLE.keys()))
supplementaires = st.sidebar.text_input(
    "Ajouter des tickers (séparés par une virgule)", placeholder="Ex : SHEL.AS, VWCE.DE")
periode = st.sidebar.selectbox("Période", ["1y", "2y", "5y", "10y", "max"], index=2)
ticker_ref = st.sidebar.text_input(
    "Indice de référence (pour le bêta)", value="IWDA.AS",
    help="Par défaut : ETF actions mondiales (iShares MSCI World, coté à Amsterdam).")

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


@st.cache_data(ttl=21600, max_entries=100, show_spinner=False)
def infos_actif(ticker):
    """Fondamentaux, dividende et actualités (peuvent être indisponibles selon l'actif)."""
    sortie = {"fondamentaux": {}, "dividende_12m": None, "news": []}
    try:
        t = yf.Ticker(ticker)
        try:
            info = t.info or {}
        except Exception:
            info = {}
        sortie["fondamentaux"] = {
            "Nom": info.get("longName") or info.get("shortName"),
            "Secteur": info.get("sector"),
            "PER (12 derniers mois)": info.get("trailingPE"),
            "Capitalisation": info.get("marketCap"),
            "Devise": info.get("currency"),
        }
        try:
            div = t.dividends
            if div is not None and len(div):
                div.index = div.index.tz_localize(None)
                limite = pd.Timestamp.now() - pd.Timedelta(days=365)
                sortie["dividende_12m"] = float(div[div.index > limite].sum())
        except Exception:
            pass
        try:
            for n in (t.news or [])[:6]:
                c = n.get("content", n)
                titre = c.get("title")
                lien = (c.get("canonicalUrl") or {}).get("url") or c.get("link")
                source = (c.get("provider") or {}).get("displayName") or c.get("publisher")
                if titre and lien:
                    sortie["news"].append((titre, lien, source))
        except Exception:
            pass
    except Exception:
        pass
    return sortie


def csv_bytes(df):
    return df.to_csv(sep=";", decimal=",").encode("utf-8-sig")


# ------------------------------------------------------------
# CHARGEMENT
# ------------------------------------------------------------
st.title("📊 Mon mini terminal")

actifs, themes = {}, {}
for nom in choix:
    actifs[nom], themes[nom] = ACTIFS_PAR_DEFAUT[nom], THEME_PAR_ACTIF[nom]
for nom in choix_socle:
    actifs[nom], themes[nom] = SOCLE[nom], "Socle"
for t in [x.strip().upper() for x in supplementaires.split(",") if x.strip()]:
    actifs[t], themes[t] = t, "Autre"

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
    ref = ticker_ref.strip().upper()
    serie_ref = telecharger(ref, periode, bloc_temps) if ref else None

if not cours:
    st.error("Aucune donnée n'a pu être téléchargée. Réessaie dans quelques minutes.")
    st.stop()

rend_ref = serie_ref.pct_change().dropna() if serie_ref is not None else None
if rend_ref is None:
    st.caption("ℹ️ Indice de référence indisponible : le bêta ne sera pas calculé.")

tableau = pd.DataFrame({nom: indicateurs(s, rend_ref) for nom, s in cours.items()}).T.infer_objects()
tableau["Score /100"] = score_synthese(tableau)
tableau["Thème"] = [themes[n] for n in tableau.index]

derniere_date = max(s.index[-1] for s in cours.values())
st.caption(
    f"Dernière clôture disponible : **{derniere_date:%d/%m/%Y}** · "
    f"Mise à jour de la page : {time.strftime('%d/%m/%Y %H:%M')} · "
    "Données Yahoo Finance, performances en monnaie locale. "
    "Outil d'analyse, pas un conseil en investissement.")

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
(onglet1, onglet2, onglet3, onglet4, onglet5, onglet6, onglet7) = st.tabs([
    "📋 Vue d'ensemble", "⚠️ Risque comparé", "🔗 Corrélation", "🔍 Fiche par actif",
    "💼 Mon portefeuille", "⚖️ Socle vs thèmes", "🧪 Backtest"])

# ---- 1. Vue d'ensemble
with onglet1:
    colonnes = ["Thème", "Score /100", "Niveau de risque", "Dernier cours", "Perf. 1 mois",
                "Perf. 6 mois", "Perf. annuelle", "Volatilité", "Pire baisse", "Baisse actuelle",
                "VaR 95% (jour)", "Bêta", "Sharpe", "Sortino", "Écart vs MM200"]
    fmt = {c: "{:.1%}" for c in ["Perf. 1 mois", "Perf. 6 mois", "Perf. annuelle", "Volatilité",
                                  "Pire baisse", "Baisse actuelle", "VaR 95% (jour)", "Écart vs MM200"]}
    fmt.update({"Dernier cours": "{:,.2f}", "Bêta": "{:.2f}", "Sharpe": "{:.2f}",
                "Sortino": "{:.2f}", "Score /100": "{:.0f}"})
    vue = tableau[colonnes].sort_values("Score /100", ascending=False)
    st.dataframe(vue.style.format(fmt, na_rep="-"), use_container_width=True)
    st.caption("**Score /100** : résume la tendance (40 pts), le momentum à 6 mois (30 pts) et la "
               "maîtrise du risque (30 pts). Il décrit l'état actuel d'un actif, il ne dit pas "
               "d'acheter ou de vendre.")
    st.download_button("⬇️ Exporter le tableau (CSV, s'ouvre dans Excel)", csv_bytes(vue),
                       "tableau_de_bord.csv", "text/csv")

    base100 = tous / tous.apply(lambda c: c.dropna().iloc[0]) * 100
    fig = px.line(base100, labels={"value": "Base 100", "index": "", "variable": ""},
                  title="Évolution comparée (base 100 au départ)")
    fig.add_hline(y=100, line_dash="dash", line_color="grey")
    fig.update_layout(height=480, legend_title_text="")
    st.plotly_chart(fig, use_container_width=True)

# ---- 2. Risque comparé
with onglet2:
    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar((tableau["Pire baisse"] * 100).sort_values(), orientation="h",
                     labels={"value": "Pire baisse (%)", "index": ""}, title="Pire baisse historique")
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.bar((tableau["Volatilité"] * 100).sort_values(), orientation="h",
                     labels={"value": "Volatilité annuelle (%)", "index": ""},
                     title="Volatilité annualisée")
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    nuage = pd.DataFrame({
        "Actif": tableau.index,
        "Volatilité (%)": tableau["Volatilité"].values * 100,
        "Performance annuelle (%)": tableau["Perf. annuelle"].values * 100,
        "Thème": tableau["Thème"].values})
    fig = px.scatter(nuage, x="Volatilité (%)", y="Performance annuelle (%)", text="Actif",
                     color="Thème", title="Rendement vs risque (en haut à gauche = idéal)")
    fig.update_traces(textposition="top center", marker_size=10)
    st.plotly_chart(fig, use_container_width=True)

# ---- 3. Corrélation
with onglet3:
    if len(cours) < 2:
        st.info("Il faut au moins 2 actifs pour calculer une corrélation.")
    else:
        correl = tous.pct_change().dropna(how="all").corr()
        fig = px.imshow(correl, text_auto=".2f", color_continuous_scale="RdYlGn_r", zmin=-1, zmax=1)
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Proche de 1 : les deux actifs bougent ensemble (peu de diversification). "
                   "Proche de 0 : ils bougent indépendamment. Les places ont des horaires "
                   "différents (Tokyo, Paris…), ce qui sous-estime un peu les corrélations.")

# ---- 4. Fiche par actif
with onglet4:
    nom = st.selectbox("Choisis un actif", list(cours.keys()))
    s = cours[nom]
    l = tableau.loc[nom]
    r = s.pct_change().dropna()

    st.markdown(f"### {nom} · Risque : {l['Niveau de risque']} · Score {l['Score /100']:.0f}/100")
    st.caption(f"Ticker : {actifs[nom]} · Thème : {themes[nom]} · {len(s)} séances analysées "
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
             help="Rendement par unité de risque (simplifié, sans taux sans risque).")
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

    with st.expander("🏢 Fondamentaux et actualités (selon disponibilité)"):
        infos = infos_actif(actifs[nom])
        fond = infos["fondamentaux"]
        lignes = []
        if fond.get("Nom"):
            lignes.append(f"**Nom** : {fond['Nom']}")
        if fond.get("Secteur"):
            lignes.append(f"**Secteur** : {fond['Secteur']}")
        if fond.get("PER (12 derniers mois)"):
            lignes.append(f"**PER (12 derniers mois)** : {fond['PER (12 derniers mois)']:.1f}")
        if fond.get("Capitalisation"):
            lignes.append(f"**Capitalisation** : {fond['Capitalisation'] / 1e9:,.1f} Md ({fond.get('Devise', '')})")
        if infos["dividende_12m"] is not None and l["Dernier cours"] > 0:
            rendement_div = infos["dividende_12m"] / l["Dernier cours"]
            lignes.append(f"**Rendement du dividende (12 derniers mois)** : {rendement_div:.1%}")
        if lignes:
            for ligne in lignes:
                st.markdown(ligne)
        else:
            st.caption("Pas de données fondamentales disponibles pour cet actif.")
        st.markdown("**Dernières actualités**")
        if infos["news"]:
            for titre, lien, source in infos["news"]:
                st.markdown(f"- [{titre}]({lien})" + (f" — *{source}*" if source else ""))
        else:
            st.caption("Aucune actualité disponible pour cet actif.")

# ------------------------------------------------------------
# Portefeuille : saisie des montants (partagé par les onglets 5 et 6)
# ------------------------------------------------------------
if "montants_init" not in st.session_state:
    st.session_state.montants_init = {}
    st.session_state.version = 0
version = st.session_state.version

with onglet5:
    st.subheader("Mon portefeuille")
    st.caption("Saisis les montants (réels ou prévus) dans la colonne de droite. Les montants affichés "
               "sont des **exemples** à remplacer. Rien n'est enregistré sur le serveur : exporte ton "
               "portefeuille en CSV pour le retrouver plus tard. Si ton appli est publique, "
               "ne partage pas ton écran ni le lien avec des montants affichés.")

    base = pd.DataFrame({
        "Actif": list(cours.keys()),
        "Thème": [themes[n] for n in cours],
        "Montant (€)": [float(st.session_state.montants_init.get(n, EXEMPLE_MONTANTS.get(n, 0)))
                        for n in cours]})
    edite = st.data_editor(
        base, key=f"editeur_{version}", hide_index=True, use_container_width=True,
        disabled=["Actif", "Thème"],
        column_config={"Montant (€)": st.column_config.NumberColumn(min_value=0, step=100, format="%.0f")})
    montants = edite.set_index("Actif")["Montant (€)"].fillna(0).clip(lower=0)

    cA, cB = st.columns(2)
    cA.download_button("💾 Exporter mon portefeuille (CSV)",
                       csv_bytes(montants.rename("Montant (€)").to_frame()),
                       "mon_portefeuille.csv", "text/csv")
    fichier = cB.file_uploader("📂 Importer un portefeuille (CSV exporté ici)", type="csv",
                               key=f"upload_{version}")
    if fichier is not None:
        try:
            imp = pd.read_csv(fichier, sep=None, engine="python", decimal=",")
            valeurs = pd.to_numeric(imp.iloc[:, 1], errors="coerce").fillna(0)
            st.session_state.montants_init = dict(zip(imp.iloc[:, 0].astype(str), valeurs))
            st.session_state.version += 1
            st.rerun()
        except Exception:
            st.error("Fichier illisible : utilise un CSV exporté depuis cette appli.")

    total = float(montants.sum())
    if total <= 0:
        st.info("Saisis au moins un montant pour calculer le risque de ton portefeuille.")
    else:
        actifs_pf = montants[montants > 0].index.tolist()
        poids = montants[actifs_pf] / total
        prix_pf = tous[actifs_pf].dropna()
        if len(prix_pf) < 60:
            st.warning("Historique commun trop court pour calculer le risque du portefeuille.")
        else:
            rend_df = prix_pf.pct_change().dropna()
            r_pf = rendements_portefeuille(prix_pf, poids)
            sp = stats_rendements(r_pf)
            contrib, vol_pf, divers = contributions_risque(rend_df, poids)

            st.caption(f"Historique commun utilisé : {prix_pf.index[0]:%d/%m/%Y} → "
                       f"{prix_pf.index[-1]:%d/%m/%Y} ({len(prix_pf)} séances). "
                       "L'actif le plus récent limite la durée analysée.")

            a, b, c, d = st.columns(4)
            a.metric("Valeur totale", f"{total:,.0f} €")
            b.metric("Volatilité annuelle", f"{sp['Volatilité']:.1%}")
            c.metric("Pire baisse simulée", f"{sp['Pire baisse']:.1%}",
                     f"≈ {total * sp['Pire baisse']:,.0f} €", delta_color="off",
                     help="Pire chute qu'aurait subie ce portefeuille sur l'historique commun.")
            d.metric("VaR 95% (1 jour)", f"{sp['VaR 95% (jour)']:.2%}",
                     f"≈ {total * sp['VaR 95% (jour)']:,.0f} €", delta_color="off",
                     help="Dans 95 % des jours, la perte n'a pas dépassé ce niveau.")
            a, b, c, d = st.columns(4)
            a.metric("Perf. annuelle simulée", f"{sp['Perf. annuelle']:.1%}")
            b.metric("Sharpe", f"{sp['Sharpe']:.2f}")
            c.metric("Effet de diversification", f"{divers:.2f}",
                     help="1,0 = aucune diversification. Plus c'est haut, plus tes actifs "
                          "s'amortissent entre eux.")
            d.metric("Nombre de lignes", f"{len(actifs_pf)}")

            th = pd.Series({a_: themes[a_] for a_ in actifs_pf})
            df_th = pd.DataFrame({"Poids dans le capital (%)": poids.groupby(th).sum() * 100,
                                  "Part du risque (%)": contrib.groupby(th).sum() * 100})
            df_th = df_th.reset_index(names="Thème").melt(id_vars="Thème", var_name="Mesure",
                                                          value_name="%")
            fig = px.bar(df_th, x="Thème", y="%", color="Mesure", barmode="group",
                         title="Poids dans le capital vs part du risque, par thème")
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Si la barre « part du risque » dépasse celle du poids, ce thème pèse plus "
                       "dans ton risque que dans ton capital.")

            detail = pd.DataFrame({
                "Thème": th, "Montant (€)": montants[actifs_pf], "Poids": poids,
                "Part du risque": contrib, "Volatilité de l'actif": tableau.loc[actifs_pf, "Volatilité"],
                "Pire baisse de l'actif": tableau.loc[actifs_pf, "Pire baisse"]}
            ).sort_values("Part du risque", ascending=False)
            st.dataframe(detail.style.format({
                "Montant (€)": "{:,.0f}", "Poids": "{:.1%}", "Part du risque": "{:.1%}",
                "Volatilité de l'actif": "{:.1%}", "Pire baisse de l'actif": "{:.1%}"}),
                use_container_width=True)

            courbe = (1 + r_pf).cumprod() * total
            fig = px.line(courbe, labels={"value": "€", "index": ""},
                          title="Valeur simulée du portefeuille (poids constants, sans frais)")
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

# ---- 6. Socle vs thèmes
with onglet6:
    st.subheader("Simuler une répartition socle prudent / thèmes")
    total = float(montants.sum())
    socle_all = [a_ for a_ in cours if themes[a_] == "Socle"]
    themes_all = [a_ for a_ in cours if themes[a_] != "Socle"]
    if total <= 0:
        st.info("Saisis d'abord tes montants dans l'onglet « Mon portefeuille ».")
    elif not socle_all or not themes_all:
        st.info("Il faut au moins un actif du socle ET un actif thématique chargés pour comparer.")
    else:
        def repartition(groupe):
            m = montants.reindex(groupe).fillna(0)
            m = m[m > 0] if m.sum() > 0 else pd.Series(1.0, index=groupe)
            return m / m.sum()

        rep_socle, rep_themes = repartition(socle_all), repartition(themes_all)
        colonnes_sim = sorted(set(rep_socle.index) | set(rep_themes.index) | set(montants[montants > 0].index))
        prix_sim = tous[colonnes_sim].dropna()

        if len(prix_sim) < 60:
            st.warning("Historique commun trop court pour la simulation.")
        else:
            def poids_scenario(part_socle):
                p = pd.concat([rep_socle * part_socle, rep_themes * (1 - part_socle)])
                return p.reindex(colonnes_sim).fillna(0)

            p_perso = st.slider("Part du socle prudent dans ton portefeuille (%)", 0, 100, 80, step=5)
            actuel = (montants / total).reindex(colonnes_sim).fillna(0)
            scenarios = {
                "Mon portefeuille actuel": actuel,
                f"Scénario perso : {p_perso} % socle / {100 - p_perso} % thèmes": poids_scenario(p_perso / 100),
                "100 % socle": poids_scenario(1.0),
                "90 % socle / 10 % thèmes": poids_scenario(0.9),
                "80 % socle / 20 % thèmes": poids_scenario(0.8),
                "60 % socle / 40 % thèmes": poids_scenario(0.6),
                "100 % thèmes": poids_scenario(0.0)}

            lignes, courbes = {}, {}
            for nom_sc, p in scenarios.items():
                r_sc = rendements_portefeuille(prix_sim, p)
                st_sc = stats_rendements(r_sc)
                st_sc["Perte max. en € (sur ton capital)"] = total * st_sc["Pire baisse"]
                lignes[nom_sc] = st_sc
                courbes[nom_sc] = (1 + r_sc).cumprod() * 100
            comp = pd.DataFrame(lignes).T[["Perf. annuelle", "Volatilité", "Pire baisse",
                                           "Perte max. en € (sur ton capital)", "VaR 95% (jour)", "Sharpe"]]
            st.dataframe(comp.style.format({
                "Perf. annuelle": "{:.1%}", "Volatilité": "{:.1%}", "Pire baisse": "{:.1%}",
                "Perte max. en € (sur ton capital)": "{:,.0f}", "VaR 95% (jour)": "{:.2%}",
                "Sharpe": "{:.2f}"}), use_container_width=True)

            affiche = [k for k in courbes if k in ("Mon portefeuille actuel", "100 % socle", "100 % thèmes")
                       or k.startswith("Scénario perso")]
            fig = px.line(pd.DataFrame({k: courbes[k] for k in affiche}),
                          labels={"value": "Base 100", "index": "", "variable": ""},
                          title="Évolution simulée (base 100)")
            fig.update_layout(legend_title_text="")
            st.plotly_chart(fig, use_container_width=True)
            st.caption(f"Socle : {', '.join(socle_all)}. Thèmes : {', '.join(themes_all)}. "
                       "Au sein de chaque groupe, la répartition suit tes montants actuels (ou est "
                       "égale si tu n'en as pas saisi). Simulation historique, en monnaie locale, "
                       "sans frais ni impôts : elle ne prédit pas l'avenir.")

# ---- 7. Backtest
with onglet7:
    st.subheader("Backtest d'une règle simple de moyenne mobile")
    st.markdown("**Règle testée** : être investi quand le cours est **au-dessus** de sa moyenne "
                "mobile, sinon rester en liquidités (rendement 0 %). Le signal d'un jour ne "
                "s'applique qu'à la séance suivante. Comparaison avec « acheter et conserver ».")
    c1, c2 = st.columns(2)
    fenetre = c1.slider("Moyenne mobile (jours)", 20, 250, 200, step=10)
    frais = c2.slider("Frais par changement de position (%)", 0.0, 1.0, 0.1, step=0.05) / 100

    resultats, details = {}, {}
    for nom_a, s_a in cours.items():
        bt = backtest_mm(s_a, fenetre, frais)
        if bt is None or len(bt["strategie"]) < 60:
            continue
        s_strat, s_cons = stats_rendements(bt["strategie"]), stats_rendements(bt["conserver"])
        if not s_strat or not s_cons:
            continue
        details[nom_a] = bt
        resultats[nom_a] = {
            "Perf. annuelle (conserver)": s_cons["Perf. annuelle"],
            "Perf. annuelle (règle)": s_strat["Perf. annuelle"],
            "Pire baisse (conserver)": s_cons["Pire baisse"],
            "Pire baisse (règle)": s_strat["Pire baisse"],
            "Volatilité (conserver)": s_cons["Volatilité"],
            "Volatilité (règle)": s_strat["Volatilité"],
            "Changements de position": bt["nb_changements"],
            "% du temps investi": bt["temps_investi"]}

    if not resultats:
        st.info("Pas assez d'historique : choisis une période plus longue ou une moyenne plus courte.")
    else:
        res = pd.DataFrame(resultats).T
        fmt_bt = {c: "{:.1%}" for c in res.columns if c != "Changements de position"}
        fmt_bt["Changements de position"] = "{:.0f}"
        st.dataframe(res.style.format(fmt_bt), use_container_width=True)

        choix_bt = st.selectbox("Voir le détail d'un actif", list(details.keys()), key="choix_bt")
        d_bt = details[choix_bt]
        courbes_bt = pd.DataFrame({
            "Conserver": (1 + d_bt["conserver"]).cumprod() * 100,
            "Règle moyenne mobile": (1 + d_bt["strategie"]).cumprod() * 100})
        fig = px.line(courbes_bt, labels={"value": "Base 100", "index": "", "variable": ""},
                      title=f"{choix_bt} : règle vs conserver (base 100)")
        fig.update_layout(legend_title_text="")
        st.plotly_chart(fig, use_container_width=True)
        st.warning("À lire avant d'en tirer une conclusion : un backtest décrit le passé sur une "
                   "seule période, ne tient compte ni des impôts ni des écarts d'achat/vente, et une "
                   "règle qui marche bien sur l'historique peut très bien décevoir ensuite. "
                   "Compare surtout la **pire baisse** : c'est là qu'une règle de ce type aide le plus, "
                   "souvent au prix d'une performance plus faible.")
