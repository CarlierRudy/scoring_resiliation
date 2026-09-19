#  Scoring de résiliation - Assurance Auto

Application de Machine Learning qui estime la probabilité qu'un client d'assurance
automobile résilie son contrat, à partir de son profil (âge, prime, sinistralité,
type de contrat…). Construite dans le cadre de la formation *Machine Learning &
Data Science* (Jour 3 — du dataset au modèle déployé).

**Démo en ligne :** https://scoring-resiliation-rfc.streamlit.app

## Le modèle

- **Algorithme :** Random Forest (`class_weight='balanced'`, `max_depth=4`, `min_samples_leaf=10`)
- **Prétraitement :** `StandardScaler` sur les variables numériques + `OneHotEncoder` sur les
  variables catégorielles, assemblés dans un `ColumnTransformer` puis un `Pipeline` scikit-learn unique
- **Performance (jeu de test, 100 clients) :** ROC-AUC ≈ 0,85 · Accuracy ≈ 0,87 · F1 ≈ 0,52
- **12 variables d'entrée**, choisies pour leur lien avec la cible et en excluant toute
  fuite de données (`Statut Contrat` a été écartée : elle révèle la cible à 100 %)

## Structure du projet

```
scoring_resiliation/
│
├── data/
│   ├── dataset_assurance_ML.xlsx   ← données brutes (fournies)
│   └── dataset_assurance_ML.csv    ← généré par train_model.py
├── models/
│   ├── pipeline_resiliation.pkl    ← pipeline complet (scaler + encodeur + modèle)
│   └── metadata.json               ← colonnes, bornes des curseurs, modalités des menus
├── notebooks/
│   └── tp_final.ipynb              ← exploration pas à pas (parties A à C)
├── .streamlit/
│   └── config.toml                 ← thème visuel de l'application
├── train_model.py                  ← script d'entraînement (reproduit les parties A, B, C)
├── app.py                          ← interface Streamlit (partie D)
├── requirements.txt                ← dépendances figées
└── README.md
```

## Lancer le projet en local

```bash
# 1. Environnement
conda activate formation_ml
pip install -r requirements.txt

# 2. (Ré)entraîner le modèle — génère models/pipeline_resiliation.pkl et metadata.json
python train_model.py

# 3. Lancer l'application
streamlit run app.py
```

L'application s'ouvre sur http://localhost:8501.

## Utilisation

- **Scoring d'un client** : réglez le profil dans la barre latérale (curseurs + menus),
  cliquez sur *Prédire*. La probabilité de résiliation, le niveau de risque et les
  8 variables les plus influentes du modèle s'affichent.
- **Scoring par lot** : déposez un CSV contenant les 12 colonnes attendues, l'application
  score chaque ligne et propose le fichier enrichi d'une colonne `Probabilité` au téléchargement.
- **Seuil d'alerte personnalisé** : un curseur dans la barre latérale permet au conseiller
  d'ajuster la sensibilité de l'alerte rouge.

## Variables du modèle

| Type | Variables |
|---|---|
| Numériques (8) | Âge, Salaire Annuel (€), Prime Annuelle (€), Ancienneté (mois), Coeff. Bonus-Malus, Nb Sinistres (3 ans), Montant Sinistres (€), Score Risque (0-100) |
| Catégorielles (4) | Type Contrat, Catégorie Prof., Usage Véhicule, Dernier Sinistre |

## Déploiement

Application déployée sur [Streamlit Community Cloud](https://share.streamlit.io).
Chaque `git push` sur la branche `main` redéploie automatiquement l'application.

## Limites connues

- Jeu d'entraînement de 500 clients seulement : les performances peuvent varier
  sensiblement selon le découpage train/test.
- Le modèle ne connaît pas la `Ville` du client (non retenue en entrée) : deux clients
  identiques en tout sauf la ville reçoivent le même score.
- Ce scoring est indicatif ; il vient en appui de l'analyse d'un conseiller, pas en remplacement.
