"""
Configuration centrale de NetGuard AI.

Ce module centralise tous les paramètres modifiables du projet :
- Les chemins des fichiers et dossiers
- Les paramètres du modèle de Machine Learning
- Les paramètres de prétraitement des données
- Les paramètres d'évaluation
- Les configurations des datasets supportés
"""

from pathlib import Path


# ─── Chemins du projet ───────────────────────────────────────────────────────

# Racine du projet
PROJET_RACINE = Path(__file__).resolve().parent.parent

# Données
DOSSIER_DATA = PROJET_RACINE / "data"
DOSSIER_DATA_BRUT = DOSSIER_DATA / "raw"
DOSSIER_DATA_TRANSFORME = DOSSIER_DATA / "processed"

# Dossier des datasets (téléchargement automatique)
DOSSIER_DATASETS = PROJET_RACINE / "datasets"

# ─── Paramètres généraux du jeu de données ───────────────────────────────────

# Nom du fichier CSV du dataset synthétique (par défaut)
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

# ─── Configuration du dataset CICIDS2017 ─────────────────────────────────────

# Documentation : https://www.unb.ca/cic/datasets/ids-2017.html
# Cet dataset contient du trafic réseau réel avec 80+ caractéristiques
# extraites par CICFlowMeter, incluant des attaques variées.

CICIDS2017_CONFIG = {
    # Nom du fichier consolidé après téléchargement et nettoyage
    "fichier_sortie": "cicids2017_consolide.csv",

    # Colonne contenant le label dans les fichiers sources
    "colonne_label": "Label",

    # Colonnes à ignorer (identifiants réseau, timestamps, etc.)
    "colonnes_a_ignorer": [
        "Flow ID",
        "Source IP",
        "Source Port",
        "Destination IP",
        "Destination Port",
        "Protocol",
        "Timestamp",
        "Fwd Header Length.1",  # Colonne dupliquée parfois présente
    ],

    # Liste des 8 fichiers CSV dans l'archive MachineLearningCSV.zip
    "fichiers_csv": [
        "Monday-WorkingHours.pcap_ISCX.csv",
        "Tuesday-WorkingHours.pcap_ISCX.csv",
        "Wednesday-workingHours.pcap_ISCX.csv",
        "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
        "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
        "Friday-WorkingHours-Morning.pcap_ISCX.csv",
        "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
        "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
    ],
}

# Chemin complet vers le fichier consolidé
CICIDS2017_CONFIG["chemin_sortie"] = (
    DOSSIER_DATASETS / CICIDS2017_CONFIG["fichier_sortie"]
)

# ─── Datasets disponibles ────────────────────────────────────────────────────

# Liste des datasets supportés par le projet
DATASETS_DISPONIBLES = {
    "synthetique": {
        "nom": "Données synthétiques (démo)",
        "description": "Généré automatiquement par demo.py",
        "chemin": DOSSIER_DATA_BRUT / FICHIER_DATASET,
        "colonne_label": "label",
        "colonnes_a_ignorer": ["id", "timestamp"],
    },
    "cicids2017": {
        "nom": "CICIDS2017",
        "description": "Dataset réel de l'Université du Nouveau-Brunswick",
        "chemin": CICIDS2017_CONFIG["chemin_sortie"],
        "colonne_label": "label",
        "colonnes_a_ignorer": [],  # Déjà nettoyé par le script
    },
}

# Dataset par défaut
DATASET_PAR_DEFAUT = "synthetique"

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

# ─── Colonnes du dataset (générique) ─────────────────────────────────────────

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
