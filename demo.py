"""
Script de démonstration de NetGuard AI.

Ce script propose plusieurs modes de démonstration :
1. Données synthétiques (par défaut) : génère et teste un dataset aléatoire
2. Dataset CICIDS2017 : utilise le dataset réel si disponible

Utilisation :
    python demo.py                          # Démo avec données synthétiques
    python demo.py --dataset cicids2017     # Démo avec CICIDS2017
    python demo.py --dataset cicids2017 --download  # Télécharge puis démo
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    DOSSIER_DATA_BRUT,
    DOSSIER_DATA_TRANSFORME,
    DOSSIER_DATASETS,
    SEED_ALEATOIRE,
    FICHIER_DATASET,
    DATASETS_DISPONIBLES,
    DATASET_PAR_DEFAUT,
)
from src.preprocessing.data_loader import ChargeurDonnees
from src.features.feature_extractor import ExtracteurCaracteristiques
from src.models.detector import DetecteurIntrusions
from src.evaluation.metrics import Evaluateur
from src.utils.helpers import configurer_logging, Chronometre


def definir_arguments() -> argparse.ArgumentParser:
    """Configure les arguments en ligne de commande."""
    parser = argparse.ArgumentParser(
        description="NetGuard AI - Démonstration de détection d'intrusions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default=DATASET_PAR_DEFAUT,
        choices=list(DATASETS_DISPONIBLES.keys()),
        help=f"Dataset à utiliser (defaut: {DATASET_PAR_DEFAUT})",
    )

    parser.add_argument(
        "--download",
        action="store_true",
        help="Télécharger automatiquement CICIDS2017 si absent",
    )

    return parser


def generer_donnees_synthetiques(
    n_echantillons: int = 2000,
    n_caracteristiques: int = 30,
    ratio_attaques: float = 0.3,
) -> Path:
    """
    Génère un jeu de données synthétique pour la démonstration.

    Simule du trafic réseau avec des caractéristiques variées
    et un mélange de trafic normal et d'attaques.

    Args:
        n_echantillons: Nombre d'échantillons à générer.
        n_caracteristiques: Nombre de caractéristiques réseau.
        ratio_attaques: Proportion d'attaques dans les données.

    Returns:
        Chemin vers le fichier CSV généré.
    """
    np.random.seed(SEED_ALEATOIRE)

    logging.info(
        "Génération de %d échantillons synthétiques...", n_echantillons
    )

    # Création des données normales (trafic légitime)
    n_normal = int(n_echantillons * (1 - ratio_attaques))
    donnees_normales = np.random.normal(loc=0.0, scale=1.0, size=(n_normal, n_caracteristiques))

    # Création des données d'attaques (trafic malveillant)
    n_attaque = n_echantillons - n_normal
    donnees_attaques = np.random.normal(
        loc=2.0, scale=1.5, size=(n_attaque, n_caracteristiques)
    )

    # Assemblage
    X = np.vstack([donnees_normales, donnees_attaques])
    y = np.array([0] * n_normal + [1] * n_attaque)

    # Mélange aléatoire
    indices = np.random.permutation(n_echantillons)
    X = X[indices]
    y = y[indices]

    # Création du DataFrame
    noms_colonnes = [f"feature_{i+1:02d}" for i in range(n_caracteristiques)]
    df = pd.DataFrame(X, columns=noms_colonnes)
    df["label"] = y

    # Sauvegarde
    DOSSIER_DATA_BRUT.mkdir(parents=True, exist_ok=True)
    chemin_fichier = DOSSIER_DATA_BRUT / FICHIER_DATASET
    df.to_csv(chemin_fichier, index=False)

    logging.info(
        "Données générées : %d normaux, %d attaques → %s",
        n_normal,
        n_attaque,
        chemin_fichier,
    )

    return chemin_fichier


def executer_demo(dataset: str = "synthetique", download: bool = False) -> None:
    """
    Exécute la démonstration complète de NetGuard AI.

    Args:
        dataset: Nom du dataset à utiliser.
        download: Si True, télécharge le dataset si absent.
    """
    config_dataset = DATASETS_DISPONIBLES[dataset]

    print()
    print("+" + "-" * 58 + "+")
    print("|" + " " * 18 + "NETGUARD AI" + " " * 28 + "|")
    print("|" + " " * 8 + "Detection d'intrusions reseau" + " " * 16 + "|")
    print("|" + " " * 12 + "par Machine Learning" + " " * 25 + "|")
    print("+" + "-" * 58 + "+")
    print()
    print(f" Dataset : {config_dataset['nom']}")
    print()

    # ─── Étape 0 : Téléchargement si nécessaire ──────────────────────────────

    if download and dataset == "cicids2017":
        if not config_dataset["chemin"].exists():
            with Chronometre("Téléchargement du dataset CICIDS2017"):
                from datasets.download_cicids2017 import executer_pipeline_complet
                succes = executer_pipeline_complet()
                if not succes:
                    logging.error(
                        "Téléchargement impossible."
                    )
                    return
        else:
            logging.info("Dataset CICIDS2017 déjà présent.")

    # ─── Étape 1 : Génération ou chargement des données ──────────────────────

    if dataset == "synthetique":
        with Chronometre("Génération des données synthétiques"):
            generer_donnees_synthetiques(n_echantillons=2000)

    # ─── Étape 2 : Pipeline complet ──────────────────────────────────────────

    with Chronometre("Pipeline complet (prétraitement + entraînement + évaluation)"):
        # Chargement et prétraitement
        chargeur = ChargeurDonnees(
            chemin_dataset=config_dataset["chemin"],
            colonne_label=config_dataset["colonne_label"],
            colonnes_a_ignorer=config_dataset["colonnes_a_ignorer"],
        )
        X_train, X_val, X_test, y_train, y_val, y_test = (
            chargeur.preparer_donnees()
        )

        # Sélection des caractéristiques
        n_caracs = min(15, X_train.shape[1])
        extracteur = ExtracteurCaracteristiques(methode="k_best", n_caracteristiques=n_caracs)
        X_train, X_val, X_test = extracteur.transformer(
            X_train, X_val, X_test, y_train
        )

        # Entraînement
        detecteur = DetecteurIntrusions(type_modele="random_forest")
        detecteur.entrainer(X_train, y_train, X_val, y_val)

        # Prédiction et évaluation
        y_pred = detecteur.predire(X_test)
        evaluateur = Evaluateur()
        metriques, matrice, rapport = evaluateur.evaluer_complet(y_test, y_pred)
        taux = evaluateur.calculer_taux_detection(matrice)

    # ─── Étape 3 : Sauvegarde ────────────────────────────────────────────────

    DOSSIER_DATA_TRANSFORME.mkdir(parents=True, exist_ok=True)
    detecteur.sauvegarder(DOSSIER_DATA_TRANSFORME / "modele_demo.joblib")

    # ─── Affichage des résultats ─────────────────────────────────────────────

    print()
    print("-" * 60)
    print("  RESULTATS DE LA DEMONSTRATION")
    print("-" * 60)
    print(f"  Dataset     : {config_dataset['nom']}")
    print(f"  Modèle      : Random Forest")
    print(f"  Échantillons de test : {len(y_test)}")
    print()
    print(f"  Accuracy  : {metriques['accuracy']:.4f}")
    print(f"  Précision : {metriques['precision']:.4f}")
    print(f"  Rappel    : {metriques['recall']:.4f}")
    print(f"  F1-Score  : {metriques['f1_score']:.4f}")
    print()
    print(f"  Taux de detection des attaques  : {taux['taux_vrais_positifs']:.2%}")
    print(f"  Taux de fausses alarmes         : {taux['taux_faux_positifs']:.2%}")
    print()

    # Matrice de confusion simplifiée
    vn, fp, fn, vp = matrice.ravel()
    print("  Matrice de confusion :")
    print(f"                    Predite")
    print(f"                   Normal  Attaque")
    print(f"  Reel    Normal    {vn:4d}    {fp:4d}")
    print(f"          Attaque   {fn:4d}    {vp:4d}")
    print()
    print("-" * 60)
    print("  Rapport de classification :")
    print()
    print(rapport)
    print("-" * 60)
    print("  Demonstration terminee avec succes !")
    print("-" * 60)
    print()


if __name__ == "__main__":
    parser = definir_arguments()
    args = parser.parse_args()
    configurer_logging(niveau=logging.INFO)
    executer_demo(dataset=args.dataset, download=args.download)
