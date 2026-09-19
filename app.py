"""
Interface Streamlit du scoring de resiliation.

La presentation est separee du pipeline : les colonnes et les calculs de
prediction restent ceux definis par le modele charge.
"""

import json

import joblib
import pandas as pd
import streamlit as st


st.set_page_config(page_title="AssurAuto | Scoring", page_icon="AA", layout="wide")

SEUIL_RISQUE = 0.55
SEUIL_MODERE = 0.40

st.markdown(
    """
    <style>
    :root {
        --navy: #0F2747;
        --blue: #1677FF;
        --blue-soft: #EAF3FF;
        --green: #16B981;
        --orange: #F59E0B;
        --red: #EF4444;
        --text: #334155;
        --muted: #64748B;
        --line: #E2E8F0;
        --surface: #FFFFFF;
        --background: #F8FAFC;
    }
    html, body, [class*="css"] { font-family: "Trebuchet MS", sans-serif; }
    [data-testid="stAppViewContainer"] { background: var(--background); }
    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stSidebar"] { background: var(--surface); border-right: 1px solid var(--line); }
    [data-testid="stSidebar"] > div:first-child { padding: 1.35rem 1rem 1.5rem; }
    h1, h2, h3 { color: var(--navy); letter-spacing: 0; }
    h1 { font-size: clamp(1.8rem, 3vw, 2.75rem); line-height: 1.1; margin: .2rem 0 .35rem; }
    h2 { font-size: 1.35rem; }
    h3 { font-size: 1.05rem; }
    [data-testid="stMetric"] { background: var(--surface); border: 1px solid var(--line); border-radius: 10px; padding: .95rem 1.05rem; box-shadow: 0 2px 8px rgba(15,39,71,.04); min-height: 112px; }
    [data-testid="stMetricLabel"] { color: var(--muted); font-size: .82rem; }
    [data-testid="stMetricValue"] { color: var(--navy); font-weight: 700; }
    [data-testid="stMetricDelta"] { display: none; }
    div[data-testid="stForm"] { border: 0; padding: 0; background: transparent; }
    div[data-testid="stFormSubmitButton"] button { background: var(--blue); border: 0; color: white; font-weight: 700; min-height: 2.7rem; }
    div[data-testid="stFormSubmitButton"] button:hover { background: #0B63D9; }
    [data-testid="stExpander"] { border: 1px solid var(--line); border-radius: 8px; background: var(--surface); }
    .topbar { display: flex; align-items: center; justify-content: space-between; border-bottom: 1px solid var(--line); padding: .25rem 0 1.05rem; margin-bottom: 1.65rem; }
    .brand { display: flex; align-items: center; gap: .75rem; }
    .brand-mark { display: grid; place-items: center; width: 2.25rem; height: 2.25rem; border-radius: 8px; background: var(--navy); color: white; font-size: .78rem; font-weight: 700; letter-spacing: .03em; }
    .brand-name { color: var(--navy); font-size: 1.08rem; font-weight: 700; line-height: 1.1; }
    .brand-subtitle { color: var(--muted); font-size: .7rem; margin-top: .18rem; }
    .topbar-right { display: flex; align-items: center; gap: 1.1rem; color: var(--muted); font-size: .8rem; }
    .status { display: flex; align-items: center; gap: .4rem; color: #15805e; font-weight: 600; }
    .status-dot { width: .48rem; height: .48rem; background: var(--green); border-radius: 50%; }
    .avatar { display: grid; place-items: center; width: 2rem; height: 2rem; border-radius: 50%; background: var(--blue-soft); color: var(--blue); font-weight: 700; font-size: .75rem; }
    .eyebrow { color: var(--blue); font-size: .72rem; font-weight: 700; letter-spacing: .11em; text-transform: uppercase; }
    .lead { color: var(--muted); max-width: 680px; margin: 0 0 1.35rem; font-size: .98rem; }
    .card { background: var(--surface); border: 1px solid var(--line); border-radius: 10px; padding: 1.2rem 1.3rem; box-shadow: 0 2px 8px rgba(15,39,71,.035); }
    .card-title { color: var(--navy); font-size: 1.05rem; font-weight: 700; margin-bottom: .2rem; }
    .card-subtitle { color: var(--muted); font-size: .82rem; line-height: 1.45; }
    .sidebar-kicker { color: var(--blue); font-size: .7rem; font-weight: 700; letter-spacing: .1em; text-transform: uppercase; }
    .sidebar-title { color: var(--navy); font-size: 1.3rem; font-weight: 700; margin: .2rem 0 .25rem; }
    .sidebar-copy { color: var(--muted); font-size: .78rem; line-height: 1.45; margin-bottom: 1rem; }
    .result-value { color: var(--navy); font-size: clamp(2.3rem, 5vw, 4.2rem); font-weight: 700; line-height: 1; margin: .8rem 0 .35rem; }
    .risk-label { font-size: 1.05rem; font-weight: 700; }
    .risk-low { color: var(--green); }
    .risk-medium { color: var(--orange); }
    .risk-high { color: var(--red); }
    .explanation { border-left: 3px solid var(--blue); background: var(--blue-soft); color: var(--text); border-radius: 0 7px 7px 0; padding: .75rem .85rem; font-size: .84rem; line-height: 1.5; margin-top: 1.1rem; }
    .explanation.low { border-color: var(--green); background: #ECFDF5; }
    .explanation.medium { border-color: var(--orange); background: #FFFBEB; }
    .explanation.high { border-color: var(--red); background: #FEF2F2; }
    .gauge { position: relative; height: 12px; background: #E8EEF5; border-radius: 99px; margin: 1.1rem 0 .35rem; }
    .gauge-fill { height: 100%; border-radius: 99px; background: var(--blue); }
    .gauge-threshold { position: absolute; top: -7px; width: 2px; height: 26px; background: var(--red); }
    .gauge-marker { position: absolute; top: -4px; width: 20px; height: 20px; border: 3px solid white; border-radius: 50%; background: var(--navy); box-shadow: 0 1px 5px rgba(15,39,71,.25); transform: translateX(-50%); }
    .gauge-labels { display: flex; justify-content: space-between; color: var(--muted); font-size: .72rem; }
    .profile-grid { display: grid; grid-template-columns: 1fr 1fr; gap: .72rem .9rem; margin-top: 1rem; }
    .profile-item { border-bottom: 1px solid #EEF2F6; padding-bottom: .55rem; min-width: 0; }
    .profile-label { color: var(--muted); font-size: .73rem; display: block; margin-bottom: .15rem; }
    .profile-value { color: var(--navy); font-size: .83rem; font-weight: 600; overflow-wrap: anywhere; }
    .importance-row { margin: .8rem 0; }
    .importance-header { display: flex; justify-content: space-between; gap: 1rem; color: var(--text); font-size: .82rem; margin-bottom: .3rem; }
    .importance-value { color: var(--muted); font-weight: 700; }
    .importance-track { height: 8px; border-radius: 99px; background: #EAF0F6; overflow: hidden; }
    .importance-fill { height: 100%; border-radius: 99px; background: var(--blue); }
    .takeaway { background: #F8FAFC; border: 1px solid var(--line); border-radius: 8px; padding: 1rem; color: var(--text); font-size: .86rem; line-height: 1.55; }
    .footer { border-top: 1px solid var(--line); color: var(--muted); font-size: .75rem; margin-top: 2.2rem; padding: 1rem 0 1.5rem; display: flex; justify-content: space-between; gap: 1rem; }
    @media (max-width: 700px) { .topbar-right { gap: .5rem; } .topbar-right .status { display: none; } .profile-grid { grid-template-columns: 1fr; } .footer { display: block; } }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def charger_modele():
    pipeline = joblib.load("models/pipeline_resiliation.pkl")
    with open("models/metadata.json", encoding="utf-8") as fichier:
        meta = json.load(fichier)
    return pipeline, meta


pipeline, meta = charger_modele()
num_cols, cat_cols = meta["num_cols"], meta["cat_cols"]
rng, cats = meta["num_ranges"], meta["cat_values"]


def render_header():
    st.markdown(
        """
        <div class="topbar">
            <div class="brand">
                <div class="brand-mark">AA</div>
                <div><div class="brand-name">AssurAuto</div><div class="brand-subtitle">Analytics &amp; Machine Learning</div></div>
            </div>
            <div class="topbar-right">
                <div class="status"><span class="status-dot"></span>Modèle opérationnel</div>
                <div class="avatar">RCF</div><span>Rudy Carlier F. </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def curseur(colonne, step=1.0, formatage=None, label=None):
    bornes = rng[colonne]
    valeur = st.slider(label or colonne, min_value=bornes["min"], max_value=bornes["max"], value=bornes["median"], step=step, format=formatage)
    return int(valeur) if formatage == "%d" else valeur


def render_sidebar():
    st.sidebar.markdown('<div class="sidebar-kicker">Espace conseiller</div>', unsafe_allow_html=True)
    st.sidebar.markdown('<div class="sidebar-title">Profil client</div>', unsafe_allow_html=True)
    st.sidebar.markdown('<div class="sidebar-copy">Renseignez les informations du client pour obtenir une prédiction.</div>', unsafe_allow_html=True)

    with st.sidebar.form("profil_client"):
        client = {}
        with st.expander("Identité & profil", expanded=True):
            client["Âge"] = curseur("Âge", 1.0, "%d")
            client["Catégorie Prof."] = st.selectbox("Catégorie professionnelle", cats["Catégorie Prof."])
            client["Salaire Annuel (€)"] = curseur("Salaire Annuel (€)", 500.0, "%d", "Salaire annuel (€)")

        with st.expander("Contrat", expanded=True):
            client["Type Contrat"] = st.selectbox("Type de contrat", cats["Type Contrat"])
            client["Ancienneté (mois)"] = curseur("Ancienneté (mois)", 1.0, "%d", "Ancienneté (mois)")
            client["Prime Annuelle (€)"] = curseur("Prime Annuelle (€)", 10.0, "%d", "Prime annuelle (€)")

        with st.expander("Véhicule", expanded=False):
            client["Usage Véhicule"] = st.selectbox("Usage du véhicule", cats["Usage Véhicule"])
            st.caption("Les autres caractéristiques véhicule ne font pas partie des variables de ce modèle.")

        with st.expander("Sinistralité", expanded=True):
            client["Coeff. Bonus-Malus"] = curseur("Coeff. Bonus-Malus", 0.01, "%.2f", "Coefficient Bonus-Malus")
            client["Nb Sinistres (3 ans)"] = curseur("Nb Sinistres (3 ans)", 1.0, "%d", "Nombre de sinistres (3 ans)")
            client["Montant Sinistres (€)"] = curseur("Montant Sinistres (€)", 100.0, "%d", "Montant des sinistres (€)")
            client["Dernier Sinistre"] = st.selectbox("Dernier sinistre", cats["Dernier Sinistre"])
            client["Score Risque (0-100)"] = curseur("Score Risque (0-100)", 1.0, "%d", "Score risque (0-100)")

        seuil_perso = st.slider("Seuil d'alerte", min_value=0.30, max_value=0.70, value=SEUIL_RISQUE, step=0.01, format="%.0f%%", help="Un seuil plus bas détecte davantage de clients potentiellement résiliants, au prix de plus de fausses alertes.")
        lancer_scoring = st.form_submit_button("Analyser le client", type="primary", width="stretch")

    if lancer_scoring:
        st.session_state["client"] = client
        st.session_state["seuil"] = seuil_perso
        st.session_state["score_demande"] = True


def render_kpi_cards():
    modele = meta["modele"]
    n_estimators = getattr(pipeline.named_steps.get("model"), "n_estimators", None)
    detail_modele = f"Pipeline Scikit-learn · {n_estimators} arbres" if n_estimators else "Pipeline Scikit-learn"
    cartes = [("Modèle", modele, detail_modele), ("AUC de test", f"{meta['auc_test']:.3f}", "Performance de discrimination"), ("Variables", str(len(num_cols) + len(cat_cols)), "Variables utilisées dans le modèle")]
    colonnes = st.columns(3)
    for colonne, (label, valeur, detail) in zip(colonnes, cartes):
        with colonne:
            st.metric(label, valeur)
            st.caption(detail)


def risk_state(proba, seuil):
    if proba >= seuil:
        return "Client à risque", "high", "action de rétention conseillée"
    if proba >= SEUIL_MODERE:
        return "Risque modéré", "medium", "surveillance recommandée"
    return "Risque faible", "low", "client actuellement fidèle"


def render_gauge(proba, seuil):
    probability = max(0.0, min(1.0, proba))
    probability_pct = probability * 100
    threshold_pct = seuil * 100
    st.markdown(
        f"""
        <div class="gauge">
            <div class="gauge-fill" style="width:{probability_pct:.2f}%"></div>
            <div class="gauge-threshold" style="left:{threshold_pct:.2f}%"></div>
            <div class="gauge-marker" style="left:{probability_pct:.2f}%"></div>
        </div>
        <div class="gauge-labels"><span>0 %</span><span>Seuil de risque : {threshold_pct:.0f} %</span><span>100 %</span></div>
        """,
        unsafe_allow_html=True,
    )


def format_profile(client):
    valeurs = [
        ("Âge", f"{client['Âge']} ans"),
        ("Salaire annuel", f"{client['Salaire Annuel (€)']:,.0f} €".replace(",", " ")),
        ("Catégorie professionnelle", client["Catégorie Prof."]),
        ("Prime annuelle", f"{client['Prime Annuelle (€)']:,.0f} €".replace(",", " ")),
        ("Type de contrat", client["Type Contrat"]),
        ("Bonus-Malus", f"{client['Coeff. Bonus-Malus']:.2f}"),
        ("Ancienneté", f"{client['Ancienneté (mois)']} mois"),
        ("Nombre de sinistres", str(client["Nb Sinistres (3 ans)"])),
        ("Usage véhicule", client["Usage Véhicule"]),
        ("Dernier sinistre", client["Dernier Sinistre"]),
        ("Score risque", f"{client['Score Risque (0-100)']}/100"),
    ]
    items = "".join(f'<div class="profile-item"><span class="profile-label">{label}</span><span class="profile-value">{value}</span></div>' for label, value in valeurs)
    return f'<div class="profile-grid">{items}</div>'


def render_prediction(client, seuil):
    df_client = pd.DataFrame([client])[num_cols + cat_cols]
    proba = float(pipeline.predict_proba(df_client)[0, 1])
    label, tone, action = risk_state(proba, seuil)

    result_col, profile_col = st.columns([1.05, 1], gap="large")
    with result_col:
        st.markdown('<div class="card"><div class="card-title">Résultat de la prédiction</div><div class="card-subtitle">Probabilité estimée par le modèle</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="result-value">{proba:.0%}</div><div class="risk-label risk-{tone}">{label}</div>', unsafe_allow_html=True)
        render_gauge(proba, seuil)
        st.markdown(f'<div class="explanation {tone}">Le client présente un risque de résiliation de <strong>{proba:.0%}</strong>, ce qui le classe actuellement dans la catégorie « {label.lower()} ». {action.capitalize()}.</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with profile_col:
        st.markdown('<div class="card"><div class="card-title">Profil analysé</div><div class="card-subtitle">Résumé des principales caractéristiques du client</div>', unsafe_allow_html=True)
        st.markdown(format_profile(client), unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)


def feature_importance():
    model = pipeline.named_steps.get("model")
    prep = pipeline.named_steps.get("prep")
    if not hasattr(model, "feature_importances_") or prep is None:
        return pd.Series(dtype=float)

    noms = prep.get_feature_names_out()
    aggregated = {colonne: 0.0 for colonne in num_cols + cat_cols}
    for nom, importance in zip(noms, model.feature_importances_):
        nom_sans_prefixe = nom.split("__", 1)[-1]
        colonne = next((col for col in num_cols + cat_cols if nom_sans_prefixe == col or nom_sans_prefixe.startswith(f"{col}_")), None)
        if colonne is not None:
            aggregated[colonne] += float(importance)
    return pd.Series(aggregated).sort_values(ascending=False)


def render_feature_importance():
    importances = feature_importance()
    if importances.empty:
        return
    top = importances.head(8)
    maximum = float(top.max()) or 1.0
    rows = ""
    for label, value in top.items():
        rows += f'<div class="importance-row"><div class="importance-header"><span>{label}</span><span class="importance-value">{value:.1%}</span></div><div class="importance-track"><div class="importance-fill" style="width:{value / maximum * 100:.1f}%"></div></div></div>'

    principales = list(top.head(2).index)
    phrase = f"{principales[0]} et {principales[1]} ressortent comme les variables les plus importantes dans ce modèle." if len(principales) == 2 else f"{principales[0]} ressort comme la variable la plus importante dans ce modèle."
    graphique_col, retenir_col = st.columns([1.65, 1], gap="large")
    with graphique_col:
        st.markdown('<div class="card"><div class="card-title">Variables les plus influentes</div><div class="card-subtitle">Importance relative des variables dans la prédiction du modèle.</div>', unsafe_allow_html=True)
        st.markdown(rows, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with retenir_col:
        st.markdown(f'<div class="card"><div class="card-title">À retenir</div><div class="card-subtitle">Lecture prudente de l’explicabilité</div><div class="takeaway" style="margin-top:1rem">{phrase}<br><br>Il s’agit d’une importance dans le modèle, et non d’une relation de causalité.</div></div>', unsafe_allow_html=True)


def render_batch_scoring():
    st.markdown('<div class="card-title">Importer un portefeuille</div><div class="card-subtitle">Déposez un fichier CSV encodé en UTF-8 pour classer les clients par priorité.</div>', unsafe_allow_html=True)
    with st.expander("Voir les colonnes attendues"):
        st.code(", ".join(num_cols + cat_cols), language="text")
    fichier = st.file_uploader("Fichier CSV de clients", type=["csv"])
    if fichier is None:
        st.info("Déposez un fichier CSV pour obtenir un classement priorisé.")
        return
    try:
        df_lot = pd.read_csv(fichier, encoding="utf-8-sig")
        manquantes = [colonne for colonne in num_cols + cat_cols if colonne not in df_lot.columns]
        if manquantes:
            st.error(f"Colonnes manquantes dans le fichier : {manquantes}")
            return
        df_lot["Probabilité"] = pipeline.predict_proba(df_lot[num_cols + cat_cols])[:, 1]
        df_lot["Probabilité"] = (df_lot["Probabilité"] * 100).round(1)
        df_lot = df_lot.sort_values("Probabilité", ascending=False)
        st.success(f"{len(df_lot)} clients scorés.")
        st.dataframe(df_lot, width="stretch")
        st.download_button("Télécharger le fichier enrichi", df_lot.to_csv(index=False, encoding="utf-8-sig"), file_name="clients_scores.csv", mime="text/csv", width="stretch")
    except Exception as erreur:
        st.error(f"Impossible de lire ce fichier : {erreur}")


def render_footer():
    st.markdown('<div class="footer"><span>Scoring individuel - Outil d’aide à l’analyse — il complète l’analyse du conseiller et ne la remplace pas.</span><span>Random Forest · Scikit-learn · Streamlit</span></div>', unsafe_allow_html=True)
    st.markdown('<div style="height:1.5rem"></div>', unsafe_allow_html=True)
    st.markdown('<div style="text-align:center; font-size:.75rem; color:#64748B">© 2026 AssurAuto. Tous droits réservés.</div>', unsafe_allow_html=True)
    st.markdown('<div style="height:1.5rem"></div>', unsafe_allow_html=True)
    st.markdown('<div style="text-align:center; font-size:.75rem; color:#64748B">Version 2.4.1</div>', unsafe_allow_html=True)
    st.markdown('<div style="height:1.5rem"></div>', unsafe_allow_html=True)
    st.markdown('<div style="text-align:center; font-size:.75rem; color:#64748B">Développé par Rudy Carlier F.</div>', unsafe_allow_html=True)
    st.markdown('<div style="height:1.5rem"></div>', unsafe_allow_html=True)
    st.markdown('<div style="text-align:center; font-size:.75rem; color:#64748B">Pour toute question ou suggestion, contactez <a href="mailto:rudycarlierfba@gmail.com">rudycarlierfba@gmail.com</a></div>', unsafe_allow_html=True)
    st.markdown('<div style="height:1.5rem"></div>', unsafe_allow_html=True)
    st.markdown('<div style="text-align:center; font-size:.75rem; color:#64748B">Dernière mise à jour : 19 Septembre 2026</div>', unsafe_allow_html=True)
    st.markdown('<div style="height:1.5rem"></div>', unsafe_allow_html=True)



render_header()
render_sidebar()

st.markdown('<div class="eyebrow">Pilotage fidélisation · assurance auto</div>', unsafe_allow_html=True)
st.title("Scoring de résiliation")
st.markdown('<p class="lead">Anticipez le départ de vos clients grâce à la puissance du Machine Learning.</p>', unsafe_allow_html=True)
render_kpi_cards()

onglet_client, onglet_lot = st.tabs(["Scoring individuel", "Scoring par lot"])
with onglet_client:
    if st.session_state.get("score_demande") and "client" in st.session_state:
        client_actuel = st.session_state["client"]
        seuil_actuel = st.session_state["seuil"]
        render_prediction(client_actuel, seuil_actuel)
        st.markdown("<div style='height:.9rem'></div>", unsafe_allow_html=True)
        render_feature_importance()
    else:
        st.info("Renseignez le profil dans le panneau de gauche, puis cliquez sur **Analyser le client** se trouvant en dessous dans le panneau de gauche.")

with onglet_lot:
    render_batch_scoring()

render_footer()
