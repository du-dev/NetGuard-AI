"""
Configuration centrale de NetGuard AI.

Ce module centralise tous les paramètres modifiables du projet :
- Les chemins des fichiers et dossiers
- Les paramètres du modèle de Machine Learning
- Les paramètres de prétraitement des données
- Les paramètres d'évaluation
"""

from pathlib import Path


# ─── Chemins du projet ───────────────────────────────────────────────────────

# Racine du projet
PROJET_RACINE = Path(__file__).resolve().parent.parent

# Données
DOSSIER_DATA = PROJET_RACINE / "data"
DOSSIER_DATA_BRUT = DOSSIER_DATA / "raw"
DOSSIER_DATA_TRANSFORME = DOSSIER_DATA / "processed"

# ─── Paramètres du jeu de données ────────────────────────────────────────────

# Nom du fichier CSV du dataset (à placer dans data/raw/)
FICHIER_DATASET = "network_traffic.csv"

# Séparateur CSV
SEPARATEUR_CSV = ","

# Fraction du dataset utilisée pour l'entraînement (70%)
FRACTION_ENTRAINEMENT = 0.7

# Fraction du dataset utilisée pour la validation (15%)
FRACTION_VALIDATION = 0.15
# Le reste (15%) est utilisé pour le test

# Valeur seed pour la reproductibilité
SEED_ALEATOIRE = 42

# ─── Paramètres du modèle ────────────────────────────────────────────────────

# Type de modèle : "random_forest", "svm", "knn", "gradient_boosting"
TYPE_MODELE = "random_forest"

# Paramètres du Random Forest
PARAMS_RANDOM_FOREST = {
    "n_estimators": 100,         # Nombre d'arbres
    "max_depth": 20,             # Profondeur maximale
    "min_samples_split": 5,      # Échantillons minimum pour diviser
    "min_samples_leaf": 2,       # Échantillons minimum par feuille
    "random_state": SEED_ALEATOIRE,
    "n_jobs": -1,                # Utiliser tous les CPU disponibles
}

# Paramètres du SVM
PARAMS_SVM = {
    "kernel": "rbf",
    "C": 1.0,
    "gamma": "scale",
    "random_state": SEED_ALEATOIRE,
}

# Paramètres du KNN
PARAMS_KNN = {
    "n_neighbors": 5,
    "weights": "distance",
    "n_jobs": -1,
}

# Paramètres du Gradient Boosting
PARAMS_GRADIENT_BOOSTING = {
    "n_estimators": 100,
    "max_depth": 5,
    "learning_rate": 0.1,
    "random_state": SEED_ALEATOIRE,
}

# ─── Colonnes du dataset ─────────────────────────────────────────────────────

# Colonne cible (label : 0 = normal, 1 = attaque)
COLONNE_CIBLE = "label"

# Colonnes à ignorer (identifiants, timestamps, etc.)
COLONNES_A_IGNORER = ["id", "timestamp"]

# ─── Paramètres d'évaluation ─────────────────────────────────────────────────

# Métriques affichées après l'évaluation
METRIQUES_AFFICHAGE = [
    "accuracy",
    "precision",
    "recall",
    "f1_score",
    "confusion_matrix",
]
