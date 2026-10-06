"""Fonctions de calcul du terminal (aucune dépendance à Streamlit)."""
import numpy as np
import pandas as pd

JOURS = 252


def niveau_risque(vol):
    if vol < 0.15:
        return "🟢 Faible"
    if vol < 0.25:
        return "🟡 Modéré"
    if vol < 0.40:
        return "🟠 Élevé"
    return "🔴 Très élevé"


def indicateurs(serie, rend_ref=None):
    """Statistiques complètes d'une série de cours."""
    r = serie.pct_change().dropna()
    annees = max((serie.index[-1] - serie.index[0]).days / 365.25, 0.01)
    perf_annuelle = (serie.iloc[-1] / serie.iloc[0]) ** (1 / annees) - 1
    vol = r.std() * np.sqrt(JOURS)
    vol_baisse = r[r < 0].std() * np.sqrt(JOURS)

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

    def perf_sur(n):
        return serie.iloc[-1] / serie.iloc[-n - 1] - 1 if len(serie) > n else np.nan

    return {
        "Niveau de risque": niveau_risque(vol),
        "Dernier cours": serie.iloc[-1],
        "Perf. 1 mois": perf_sur(21),
        "Perf. 6 mois": perf_sur(126),
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


def score_synthese(tableau):
    """Score descriptif sur 100 : tendance (40), momentum (30), maîtrise du risque (30).
    Ce n'est PAS un conseil d'achat : il résume l'état actuel de chaque actif."""
    ecart = tableau["Écart vs MM200"].astype(float).fillna(0)
    mom = tableau["Perf. 6 mois"].astype(float).fillna(0)
    vol = tableau["Volatilité"].astype(float)
    tendance = 40 * ((ecart + 0.2) / 0.4).clip(0, 1)
    momentum = 30 * ((mom + 0.2) / 0.5).clip(0, 1)
    risque = 30 * (1 - ((vol - 0.10) / 0.40).clip(0, 1))
    return (tendance + momentum + risque).round(0)


def stats_rendements(r):
    """Statistiques d'une série de rendements quotidiens (portefeuille, stratégie...)."""
    r = r.dropna()
    if len(r) < 20:
        return {}
    courbe = (1 + r).cumprod()
    annees = max((r.index[-1] - r.index[0]).days / 365.25, 0.01)
    perf_annuelle = courbe.iloc[-1] ** (1 / annees) - 1
    vol = r.std() * np.sqrt(JOURS)
    dd = courbe / courbe.cummax() - 1
    return {
        "Perf. annuelle": perf_annuelle,
        "Perf. totale": courbe.iloc[-1] - 1,
        "Volatilité": vol,
        "Pire baisse": dd.min(),
        "VaR 95% (jour)": r.quantile(0.05),
        "Sharpe": perf_annuelle / vol if vol > 0 else np.nan,
    }


def rendements_portefeuille(prix, poids):
    """Rendements quotidiens d'un portefeuille aux poids constants (rééquilibré chaque jour)."""
    rend = prix.pct_change().dropna()
    return (rend * poids.reindex(rend.columns)).sum(axis=1)


def contributions_risque(rend, poids):
    """Part du risque (variance) portée par chaque actif, volatilité du portefeuille,
    et effet de diversification (>1 = la diversification réduit le risque)."""
    w = poids.reindex(rend.columns).values
    cov = rend.cov().values * JOURS
    sigma_w = cov @ w
    variance = float(w @ sigma_w)
    contrib = pd.Series(w * sigma_w / variance, index=rend.columns)
    vol_pf = float(np.sqrt(variance))
    vols = np.sqrt(np.diag(cov))
    diversification = float((w * vols).sum() / vol_pf) if vol_pf > 0 else np.nan
    return contrib, vol_pf, diversification


def backtest_mm(serie, fenetre=200, frais=0.001):
    """Règle : investi si le cours est au-dessus de sa moyenne mobile, sinon en liquidités (0 %).
    Le signal du jour s'applique au rendement du lendemain (pas de triche sur le futur).
    Les deux séries sont comparées sur la même période (après le calcul de la 1re moyenne)."""
    ret = serie.pct_change().fillna(0)
    mm = serie.rolling(fenetre).mean()
    debut = mm.first_valid_index()
    if debut is None:
        return None
    signal = (serie > mm).astype(float).shift(1).fillna(0).loc[debut:]
    ret = ret.loc[debut:]
    echanges = signal.diff().abs().fillna(0)
    strat = signal * ret - echanges * frais
    return {
        "strategie": strat,
        "conserver": ret,
        "signal": signal,
        "nb_changements": int(echanges.sum()),
        "temps_investi": float(signal.mean()),
    }
