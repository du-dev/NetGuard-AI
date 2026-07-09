"""
Script de démonstration de NetGuard AI.

Génère un jeu de données synthétique pour tester le pipeline
sans avoir besoin d'un véritable fichier CSV d'entrée.

Ce script permet une démonstration complète du projet :
1. Génération de données réseau synthétiques
2. Pipeline complet d'entraînement et d'évaluation
3. Affichage des résultats
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    DOSSIER_DATA_BRUT,
    DOSSIER_DATA_TRANSFORME,
    SEED_ALEATOIRE,
    FICHIER_DATASET,
)
from src.preprocessing.data_loader import ChargeurDonnees
from src.features.feature_extractor import ExtracteurCaracteristiques
from src.models.detector import DetecteurIntrusions
from src.evaluation.metrics import Evaluateur
from src.utils.helpers import configurer_logging, Chronometre


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


def executer_demo() -> None:
    """Exécute la démonstration complète de NetGuard AI."""
    print()
    print("+" + "-" * 58 + "+")
    print("|" + " " * 18 + "NETGUARD AI" + " " * 28 + "|")
    print("|" + " " * 8 + "Detection d'intrusions reseau" + " " * 16 + "|")
    print("|" + " " * 12 + "par Machine Learning" + " " * 25 + "|")
    print("+" + "-" * 58 + "+")
    print()

    # ─── Étape 1 : Génération des données ────────────────────────────────────

    with Chronometre("Génération des données synthétiques"):
        generer_donnees_synthetiques(n_echantillons=2000)

    # ─── Étape 2 : Pipeline complet ──────────────────────────────────────────

    with Chronometre("Pipeline complet (prétraitement + entraînement + évaluation)"):
        # Chargement et prétraitement
        chargeur = ChargeurDonnees()
        X_train, X_val, X_test, y_train, y_val, y_test = (
            chargeur.preparer_donnees()
        )

        # Sélection des caractéristiques
        extracteur = ExtracteurCaracteristiques(methode="k_best", n_caracteristiques=15)
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
    print(f"  Modèle utilisé : Random Forest")
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

    # Matrice de confusion simplifiee
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
    configurer_logging(niveau=logging.INFO)
    executer_demo()
