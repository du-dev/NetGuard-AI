"""
Comparaison des modèles de Machine Learning pour NetGuard AI.

Ce script entraîne et évalue les 6 modèles sur le même jeu de données,
puis affiche un tableau comparatif de leurs performances.

Utilisation :
    python -m src.models.compare
    python -m src.models.compare --dataset cicids2017
    python -m src.models.compare --sauvegarder
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# Ajout de la racine du projet pour les imports
PROJET_RACINE = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJET_RACINE))

from src.config import (
    LISTE_MODELES,
    DESCRIPTION_MODELES,
    DATASETS_DISPONIBLES,
    DATASET_PAR_DEFAUT,
    DOSSIER_DATA_TRANSFORME,
)
from src.preprocessing.pipeline import PipelinePretraitement
from src.models.detector import DetecteurIntrusions
from src.evaluation.metrics import Evaluateur
from src.utils.helpers import configurer_logging, Chronometre


def definir_arguments() -> argparse.ArgumentParser:
    """Configure les arguments de la ligne de commande."""
    parser = argparse.ArgumentParser(
        description="NetGuard AI - Comparaison des modèles de ML",
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
        help="Télécharger le dataset si absent",
    )

    parser.add_argument(
        "--modeles",
        type=str,
        nargs="+",
        default=LISTE_MODELES,
        choices=LISTE_MODELES,
        help="Modèles à comparer (defaut: tous)",
    )

    parser.add_argument(
        "--sauvegarder",
        action="store_true",
        help="Sauvegarder les résultats dans un fichier CSV",
    )

    return parser


def preparer_donnees(
    nom_dataset: str, download: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Charge et prétraite les données via le PipelinePretraitement.

    Args:
        nom_dataset: Nom du dataset à utiliser.
        download: Si True, télécharge le dataset si absent.

    Returns:
        Tuple (X_train, X_test, y_train, y_test).
    """
    config_dataset = DATASETS_DISPONIBLES[nom_dataset]

    # Téléchargement si nécessaire
    if download and nom_dataset == "cicids2017":
        if not config_dataset["chemin"].exists():
            from datasets.download_cicids2017 import executer_pipeline_complet
            succes = executer_pipeline_complet()
            if not succes:
                logging.error("Téléchargement impossible.")
                sys.exit(1)

    # Pipeline de prétraitement
    pipeline = PipelinePretraitement(
        colonne_label=config_dataset["colonne_label"],
        colonnes_a_ignorer=config_dataset["colonnes_a_ignorer"],
        taille_test=0.3,
    )

    # Exécution du pipeline complet
    X_train, X_test, y_train, y_test = pipeline.executer_pipeline_complet(
        chemin_dataset=config_dataset["chemin"],
        sauvegarder=False,
    )

    return X_train, X_test, y_train, y_test


def compare_modeles(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    modeles_a_comparer: List[str],
) -> pd.DataFrame:
    """
    Entraîne et évalue chaque modèle, retourne un DataFrame comparatif.

    Pour chaque modèle, calcule :
    - Temps d'entraînement
    - Accuracy, Précision, Rappel, F1-Score
    - Taux de détection et taux de faux positifs

    Args:
        X_train: Caractéristiques d'entraînement.
        X_test: Caractéristiques de test.
        y_train: Labels d'entraînement.
        y_test: Labels de test.
        modeles_a_comparer: Liste des modèles à évaluer.

    Returns:
        DataFrame avec les résultats de chaque modèle.
    """
    resultats = []

    logging.info("")
    logging.info("=" * 60)
    logging.info("COMPARAISON DES MODÈLES DE MACHINE LEARNING")
    logging.info("=" * 60)
    logging.info("Échantillons d'entraînement : %d", len(X_train))
    logging.info("Échantillons de test       : %d", len(X_test))
    logging.info("Caractéristiques           : %d", X_train.shape[1])
    logging.info("=" * 60)
    logging.info("")

    for nom_modele in modeles_a_comparer:
        description = DESCRIPTION_MODELES.get(nom_modele, nom_modele)
        logging.info("-" * 60)
        logging.info("Modèle : %s", description)

        # Initialisation
        detecteur = DetecteurIntrusions(type_modele=nom_modele)

        # Entraînement chronométré
        with Chronometre(f"  Entraînement {nom_modele}"):
            detecteur.entrainer(X_train, y_train)

        # Évaluation
        y_pred = detecteur.predire(X_test)

        # Métriques
        evaluateur = Evaluateur()
        metriques = evaluateur.calculer_metriques(y_test, y_pred)
        matrice = evaluateur.calculer_matrice_confusion(y_test, y_pred)
        taux = evaluateur.calculer_taux_detection(matrice)

        # Agrégation des résultats
        resultats.append({
            "modèle": nom_modele,
            "description": description,
            "accuracy": round(metriques["accuracy"], 4),
            "precision": round(metriques["precision"], 4),
            "rappel": round(metriques["recall"], 4),
            "f1_score": round(metriques["f1_score"], 4),
            "taux_detection": round(taux["taux_vrais_positifs"], 4),
            "taux_faux_positifs": round(taux["taux_faux_positifs"], 4),
        })

        logging.info(
            "  Accuracy: %.4f | Précision: %.4f | Rappel: %.4f | F1: %.4f",
            metriques["accuracy"],
            metriques["precision"],
            metriques["recall"],
            metriques["f1_score"],
        )

    # Création du DataFrame comparatif
    df_resultats = pd.DataFrame(resultats)

    # Tri par F1-Score décroissant
    df_resultats.sort_values(by="f1_score", ascending=False, inplace=True)
    df_resultats.reset_index(drop=True, inplace=True)
    df_resultats.index += 1  # Commencer l'index à 1

    return df_resultats


def afficher_tableau(df: pd.DataFrame) -> None:
    """
    Affiche un tableau formaté des résultats.

    Args:
        df: DataFrame avec les résultats.
    """
    print()
    print("=" * 80)
    print("  TABLEAU COMPARATIF DES MODÈLES")
    print("=" * 80)

    # En-tête
    print(f"  {'#':<3} {'Modèle':<22} {'Accuracy':<10} {'Précision':<10} "
          f"{'Rappel':<10} {'F1-Score':<10} {'Détection':<10} {'Faux+':<10}")
    print("  " + "-" * 77)

    # Lignes
    for idx, row in df.iterrows():
        print(f"  {idx:<3} {row['description']:<22} "
              f"{row['accuracy']:<10.4f} {row['precision']:<10.4f} "
              f"{row['rappel']:<10.4f} {row['f1_score']:<10.4f} "
              f"{row['taux_detection']:<10.4f} {row['taux_faux_positifs']:<10.4f}")

    print("=" * 80)
    print()

    # Meilleur modèle
    meilleur = df.iloc[0]
    print(f"  >>> Meilleur modèle : {meilleur['description']}")
    print(f"      F1-Score : {meilleur['f1_score']:.4f}")
    print(f"      Accuracy : {meilleur['accuracy']:.4f}")
    print()


def sauvegarder_resultats(df: pd.DataFrame, nom_dataset: str) -> None:
    """
    Sauvegarde le tableau comparatif en CSV.

    Args:
        df: DataFrame avec les résultats.
        nom_dataset: Nom du dataset utilisé.
    """
    DOSSIER_DATA_TRANSFORME.mkdir(parents=True, exist_ok=True)
    chemin = DOSSIER_DATA_TRANSFORME / f"comparaison_modeles_{nom_dataset}.csv"
    df.to_csv(chemin, index_label="rang")
    logging.info("Résultats sauvegardés : %s", chemin)

    # Version texte lisible
    chemin_txt = DOSSIER_DATA_TRANSFORME / f"comparaison_modeles_{nom_dataset}.txt"
    with open(chemin_txt, "w", encoding="utf-8") as f:
        f.write("=== Comparaison des modèles NetGuard AI ===\n\n")
        f.write(df.to_string(index=True))
    logging.info("Résultats texte sauvegardés : %s", chemin_txt)


def main() -> None:
    """Point d'entrée du script de comparaison."""
    parser = definir_arguments()
    args = parser.parse_args()

    configurer_logging()

    print()
    print("+" + "-" * 58 + "+")
    print("|" + " " * 10 + "COMPARAISON DES MODELES ML - NETGUARD AI" + " " * 9 + "|")
    print("+" + "-" * 58 + "+")
    print()

    # Préparation des données
    with Chronometre("Préparation des données"):
        X_train, X_test, y_train, y_test = preparer_donnees(
            args.dataset, download=args.download
        )

    # Comparaison des modèles
    df_resultats = compare_modeles(
        X_train, X_test, y_train, y_test,
        modeles_a_comparer=args.modeles,
    )

    # Affichage du tableau
    afficher_tableau(df_resultats)

    # Sauvegarde
    if args.sauvegarder:
        sauvegarder_resultats(df_resultats, args.dataset)

    logging.info("Comparaison terminée avec succès.")


if __name__ == "__main__":
    main()
