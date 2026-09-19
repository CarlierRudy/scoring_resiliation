"""
train_model.py - Scoring de résiliation client (assurance auto)
=================================================================
Reproduit les parties A, B et C du TP Jour 3 :
  A. Chargement du dataset et choix des variables (sans fuite de données)
  B. Pipeline scikit-learn (prétraitement + modèle), comparaison et évaluation
  C. Sauvegarde du pipeline (.pkl) et des métadonnées (.json) pour l'interface

Usage :
    python train_model.py

Produit :
    data/dataset_assurance_ML.csv
    models/pipeline_resiliation.pkl
    models/metadata.json
"""

import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42
TARGET = "Résiliation"

# Les 8 variables numériques et 4 catégorielles retenues (cf. étape 4 du TP) :
# choisies d'après l'EDA du Jour 2 (les plus liées à la cible) et pour garder
# une interface de saisie raisonnable (12 champs). 'Statut Contrat' est exclue
# car c'est une fuite de données (voir étape 3).
NUM_COLS = [
    "Âge",
    "Salaire Annuel (€)",
    "Prime Annuelle (€)",
    "Ancienneté (mois)",
    "Coeff. Bonus-Malus",
    "Nb Sinistres (3 ans)",
    "Montant Sinistres (€)",
    "Score Risque (0-100)",
]
CAT_COLS = ["Type Contrat", "Catégorie Prof.", "Usage Véhicule", "Dernier Sinistre"]


def log(titre):
    print(f"\n{'=' * 70}\n{titre}\n{'=' * 70}")


# ============================================================
# PARTIE A — Chargement & choix des variables
# ============================================================
def partie_a():
    log("PARTIE A — Chargement & choix des variables")

    # Étape 1 : Excel -> CSV -> reload (le CSV sert de source stable pour la suite)
    df = pd.read_excel("data/dataset_assurance_ML.xlsx")
    df.to_csv("data/dataset_assurance_ML.csv", index=False, encoding="utf-8-sig")
    df = pd.read_csv("data/dataset_assurance_ML.csv", encoding="utf-8-sig")
    print("Shape :", df.shape)
    print("Valeurs manquantes :", int(df.isnull().sum().sum()))
    print("Doublons            :", int(df.duplicated().sum()))

    # Étape 2 : la cible
    print("\nRépartition de la cible :")
    print(df[TARGET].value_counts())
    print((df[TARGET].value_counts(normalize=True) * 100).round(1).astype(str) + " %")

    # Étape 3 : détecter la fuite de données
    print("\nCroisement Statut Contrat x Résiliation (fuite de données) :")
    print(pd.crosstab(df["Statut Contrat"], df[TARGET]))
    print("-> 'Statut Contrat' révèle la cible à 100 % : EXCLUE du modèle.")

    # Étape 4 : X et y
    X = df[NUM_COLS + CAT_COLS]
    y = df[TARGET]
    print(f"\nX : {X.shape}  |  y : {y.shape}")

    # Étape 5 : lien avec la cible
    print("\nCorrélation des variables numériques avec la cible :")
    print(X[NUM_COLS].corrwith(y).round(3).sort_values(ascending=False))
    print("\nTaux de résiliation par 'Dernier Sinistre' :")
    print(df.groupby("Dernier Sinistre")[TARGET].mean().round(2).sort_values())

    return df, X, y


# ============================================================
# PARTIE B — Pipeline, entraînement & évaluation
# ============================================================
def partie_b(X, y):
    log("PARTIE B — Pipeline, entraînement & évaluation")

    # Étape 6 : split stratifié
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Train : {X_train.shape}  |  Test : {X_test.shape}")
    print(f"Taux de résiliation — train : {y_train.mean():.2f} | test : {y_test.mean():.2f}")

    # Étape 7 : prétraitement
    preprocessor = ColumnTransformer(
        [
            ("num", StandardScaler(), NUM_COLS),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS),
        ]
    )

    # Étape 8 : deux candidats
    candidats = {
        "Régression Logistique": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=4,
            min_samples_leaf=10,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
    }
    pipelines = {
        nom: Pipeline([("prep", preprocessor), ("model", algo)])
        for nom, algo in candidats.items()
    }

    # Étape 9 : validation croisée
    print("\nComparaison par validation croisée (5-fold, ROC-AUC, sur le train) :")
    scores_cv = {}
    for nom, pipe in pipelines.items():
        scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring="roc_auc")
        scores_cv[nom] = scores
        print(f"  {nom:22s} AUC = {scores.mean():.3f} ± {scores.std():.3f}")

    # On retient le Random Forest (fournit les importances de variables,
    # utiles pour l'interface, pour un AUC comparable à la régression logistique)
    nom_retenu = "Random Forest"
    pipeline = pipelines[nom_retenu]
    print(f"\nModèle retenu : {nom_retenu}")

    # Étape 10 : entraînement + évaluation sur le test
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    print(f"\nAccuracy : {acc:.3f}")
    print(f"F1       : {f1:.3f}")
    print(f"ROC-AUC  : {auc:.3f}")
    print("\nMatrice de confusion :")
    print(confusion_matrix(y_test, y_pred))
    print("\nRapport de classification :")
    print(classification_report(y_test, y_pred, target_names=["Reste", "Résilie"]))

    # Étape 11 — lecture métier (résumé en commentaire, réponses détaillées dans le notebook)
    log("Étape 11 — Lecture métier de la matrice de confusion")
    n_resil_reelles = int(y_test.sum())
    n_resil_detectees = int(((y_test == 1) & (y_pred == 1)).sum())
    n_fausses_alertes = int(((y_test == 0) & (y_pred == 1)).sum())
    print(f"Q1. Un modèle 'toujours reste' aurait {1 - y_test.mean():.0%} d'accuracy,")
    print(f"    contre {acc:.0%} ici — MAIS il détecterait 0 résiliation. L'accuracy seule trompe.")
    print(f"Q2. Sur {n_resil_reelles} résiliations réelles, le modèle en détecte "
          f"{n_resil_detectees} (recall {n_resil_detectees / n_resil_reelles:.0%}), "
          f"au prix de {n_fausses_alertes} fausses alertes.")
    print("    Pour la Fidélisation, rater un partant (faux négatif) coûte un client perdu ;")
    print("    une fausse alerte (faux positif) ne coûte qu'un appel téléphonique. Le FN coûte plus cher.")
    print("Q3. On privilégie donc le recall : BAISSER le seuil de décision en dessous de 0,5")
    print("    pour détecter plus de résiliants, quitte à multiplier les appels de courtoisie.")

    return pipeline, X_train, X_test, y_train, y_test, y_proba, auc


# ============================================================
# PARTIE C — Sauvegarde & interrogation du modèle
# ============================================================
def partie_c(pipeline, X, X_test, y_test, y_proba, auc):
    log("PARTIE C — Sauvegarde & interrogation du modèle")

    os.makedirs("models", exist_ok=True)

    # Étape 12 : sauvegarde du pipeline
    joblib.dump(pipeline, "models/pipeline_resiliation.pkl")
    taille_ko = os.path.getsize("models/pipeline_resiliation.pkl") / 1024
    print(f"Pipeline sauvegardé : models/pipeline_resiliation.pkl ({taille_ko:.1f} Ko)")

    # Étape 13 : métadonnées pour l'interface
    meta = {
        "modele": "Random Forest",
        "auc_test": round(float(auc), 3),
        "num_cols": NUM_COLS,
        "cat_cols": CAT_COLS,
        "num_ranges": {
            c: {
                "min": float(X[c].min()),
                "max": float(X[c].max()),
                "median": float(X[c].median()),
            }
            for c in NUM_COLS
        },
        "cat_values": {c: sorted(X[c].unique().tolist()) for c in CAT_COLS},
        # Distribution des probabilités du modèle sur tout le portefeuille :
        # permet à l'interface de situer un nouveau client parmi les clients existants
        # ("ce client est plus risqué que 82 % du portefeuille").
        "proba_reference": [round(float(p), 4) for p in pipeline.predict_proba(X)[:, 1]],
    }
    with open("models/metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print("Métadonnées sauvegardées : models/metadata.json")
    print(f"  Âge : {meta['num_ranges']['Âge']['min']:.0f} à "
          f"{meta['num_ranges']['Âge']['max']:.0f} "
          f"(médiane {meta['num_ranges']['Âge']['median']:.0f})")
    print(f"  Type Contrat : {meta['cat_values']['Type Contrat']}")

    # Étape 14 : interroger le modèle sur un nouveau client
    log("Étape 14 — Interrogation sur deux profils types")
    modele = joblib.load("models/pipeline_resiliation.pkl")

    client_risque = pd.DataFrame(
        [
            {
                "Âge": 34,
                "Salaire Annuel (€)": 28000,
                "Prime Annuelle (€)": 950,
                "Ancienneté (mois)": 6,
                "Coeff. Bonus-Malus": 1.25,
                "Nb Sinistres (3 ans)": 3,
                "Montant Sinistres (€)": 4200,
                "Score Risque (0-100)": 72,
                "Type Contrat": "Bronze",
                "Catégorie Prof.": meta["cat_values"]["Catégorie Prof."][0],
                "Usage Véhicule": meta["cat_values"]["Usage Véhicule"][0],
                "Dernier Sinistre": "Vol" if "Vol" in meta["cat_values"]["Dernier Sinistre"]
                else meta["cat_values"]["Dernier Sinistre"][0],
            }
        ]
    )
    proba_risque = modele.predict_proba(client_risque)[0, 1]
    print(f"Profil à risque      -> classe {modele.predict(client_risque)[0]} "
          f"| probabilité {proba_risque:.1%}")

    client_fidele = client_risque.copy()
    client_fidele.loc[0, ["Ancienneté (mois)", "Nb Sinistres (3 ans)", "Montant Sinistres (€)",
                           "Score Risque (0-100)", "Type Contrat", "Dernier Sinistre"]] = [
        200, 0, 0, 10, "Gold", "Aucun"
    ]
    proba_fidele = modele.predict_proba(client_fidele)[0, 1]
    print(f"Profil fidèle         -> classe {modele.predict(client_fidele)[0]} "
          f"| probabilité {proba_fidele:.1%}")

    # Étape 15 : provoquer l'erreur classique (documentée, non bloquante)
    log("Étape 15 — Erreur classique : colonne manquante")
    try:
        modele.predict(client_risque.drop(columns=["Score Risque (0-100)"]))
    except Exception as e:
        print(f"ERREUR (attendue) : {e}")
    print("Règle pour l'interface : fournir EXACTEMENT les 12 colonnes, mêmes noms, même ordre.")


def main():
    df, X, y = partie_a()
    pipeline, X_train, X_test, y_train, y_test, y_proba, auc = partie_b(X, y)
    partie_c(pipeline, X, X_test, y_test, y_proba, auc)
    log("TERMINÉ — models/pipeline_resiliation.pkl et models/metadata.json sont prêts.")


if __name__ == "__main__":
    main()
