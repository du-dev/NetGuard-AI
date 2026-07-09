"""
Pipeline complet d'entraînement des modèles NetGuard AI.

Ce script orchestre l'intégralité du processus d'entraînement :
1. Chargement et prétraitement des données
2. Entraînement des 6 modèles de Machine Learning
3. Évaluation complète (Accuracy, Precision, Recall, F1-Score)
4. Matrice de confusion et rapport de classification
5. Comparaison automatique des modèles
6. Identification du meilleur modèle (selon le F1-Score)
7. Sauvegarde du meilleur modèle entraîné
8. Génération d'un rapport complet

Utilisation :
    python train.py                          # Pipeline complet (données synthétiques)
    python train.py --dataset cicids2017     # Avec CICIDS2017
    python train.py --dataset cicids2017 --download  # Téléchargement automatique
    python train.py --modele random_forest   # Entraîner un seul modèle
    python train.py --compare                # Comparer les 6 modèles (par défaut)
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.config import (
    LISTE_MODELES,
    DESCRIPTION_MODELES,
    DATASETS_DISPONIBLES,
    DATASET_PAR_DEFAUT,
    DOSSIER_DATA_TRANSFORME,
    SEED_ALEATOIRE,
)
from src.preprocessing.pipeline import PipelinePretraitement
from src.models.detector import DetecteurIntrusions
from src.evaluation.metrics import Evaluateur
from src.utils.helpers import (
    configurer_logging,
    Chronometre,
    sauvegarder_modele,
)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  CONFIGURATION DES ARGUMENTS                                               ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def definir_arguments() -> argparse.ArgumentParser:
    """
    Configure les arguments passés en ligne de commande.

    Permet de choisir le dataset, le(s) modèle(s), et le mode d'exécution.
    """
    parser = argparse.ArgumentParser(
        description=(
            "NetGuard AI - Pipeline complet d'entraînement des modèles "
            "de détection d'intrusions réseau"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples :
  python train.py                                    # Pipeline complet (6 modèles)
  python train.py --modele random_forest             # Un seul modèle
  python train.py --dataset cicids2017 --download    # Avec CICIDS2017
  python train.py --sauvegarder                      # Sauvegarder les résultats
  python train.py --verbose                          # Logs détaillés
        """,
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default=DATASET_PAR_DEFAUT,
        choices=list(DATASETS_DISPONIBLES.keys()),
        help=f"Dataset à utiliser (défaut: {DATASET_PAR_DEFAUT})",
    )

    parser.add_argument(
        "--download",
        action="store_true",
        help="Télécharger automatiquement le dataset si absent (CICIDS2017)",
    )

    parser.add_argument(
        "--modele",
        type=str,
        default=None,
        choices=LISTE_MODELES,
        help=(
            "Modèle à entraîner (défaut: tous les modèles). "
            "Si spécifié, le --compare est ignoré."
        ),
    )

    parser.add_argument(
        "--no-compare",
        action="store_true",
        help="Désactiver la comparaison des modèles (entraînement seul)",
    )

    parser.add_argument(
        "--taille-test",
        type=float,
        default=0.3,
        help="Proportion des données réservée au test (défaut: 0.3)",
    )

    parser.add_argument(
        "--sauvegarder",
        action="store_true",
        help="Sauvegarder les résultats et le meilleur modèle",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Afficher les logs détaillés (niveau DEBUG)",
    )

    return parser


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  ÉTAPE 1 : PRÉPARATION DES DONNÉES                                        ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def preparer_donnees(
    nom_dataset: str,
    taille_test: float = 0.3,
    download: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    ÉTAPE 1 : Charge, nettoie et prépare les données pour l'entraînement.

    Utilise le PipelinePretraitement pour exécuter toutes les étapes
    de transformation des données brutes en caractéristiques exploitables.

    Args:
        nom_dataset: Nom du dataset à utiliser.
        taille_test: Proportion des données pour le test.
        download: Si True, télécharge le dataset si absent.

    Returns:
        Tuple (X_train, X_test, y_train, y_test, infos_dataset)
        où infos_dataset contient les métadonnées du dataset.
    """
    logging.info("")
    logging.info("╔══════════════════════════════════════════════════╗")
    logging.info("║   ÉTAPE 1 : PRÉPARATION DES DONNÉES             ║")
    logging.info("╚══════════════════════════════════════════════════╝")
    logging.info("")

    # Récupération de la configuration du dataset
    config_dataset = DATASETS_DISPONIBLES[nom_dataset]
    logging.info("Dataset sélectionné : %s", config_dataset["nom"])
    logging.info("Description         : %s", config_dataset["description"])
    logging.info("Fichier source      : %s", config_dataset["chemin"])

    # Téléchargement automatique si demandé et nécessaire
    if download and nom_dataset == "cicids2017":
        if not config_dataset["chemin"].exists():
            logging.info("Téléchargement du dataset CICIDS2017...")
            from datasets.download_cicids2017 import executer_pipeline_complet
            succes = executer_pipeline_complet()
            if not succes:
                logging.error(
                    "Téléchargement impossible. "
                    "Placez manuellement le fichier dans datasets/"
                )
                sys.exit(1)
            logging.info("Téléchargement terminé.")
        else:
            logging.info("Dataset CICIDS2017 déjà présent, téléchargement ignoré.")

    # Création du pipeline de prétraitement
    pipeline = PipelinePretraitement(
        colonne_label=config_dataset["colonne_label"],
        colonnes_a_ignorer=config_dataset["colonnes_a_ignorer"],
        taille_test=taille_test,
    )

    # Exécution du pipeline complet (8 étapes)
    logging.info("Exécution du pipeline de prétraitement...")
    X_train, X_test, y_train, y_test = pipeline.executer_pipeline_complet(
        chemin_dataset=config_dataset["chemin"],
        sauvegarder=False,
    )

    # Collecte des informations sur le dataset traité
    infos_dataset = {
        "nom": nom_dataset,
        "nom_complet": config_dataset["nom"],
        "source": str(config_dataset["chemin"]),
        "nb_echantillons_train": len(X_train),
        "nb_echantillons_test": len(X_test),
        "nb_caracteristiques": X_train.shape[1],
        "ratio_test": taille_test,
    }

    logging.info(
        "Données prêtes : %d train, %d test, %d caractéristiques",
        len(X_train), len(X_test), X_train.shape[1],
    )

    return X_train, X_test, y_train, y_test, infos_dataset


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  ÉTAPE 2 : ENTRAÎNEMENT D'UN MODÈLE                                       ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def entrainer_et_evaluer_modele(
    nom_modele: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict:
    """
    ÉTAPE 2 : Entraîne un modèle et calcule toutes les métriques d'évaluation.

    Pour un modèle donné, cette fonction :
    1. Crée une instance du modèle via la fabrique
    2. Entraîne le modèle sur les données d'entraînement
    3. Prédit sur les données de test
    4. Calcule Accuracy, Precision, Recall, F1-Score
    5. Calcule la matrice de confusion
    6. Génère le rapport de classification
    7. Calcule les taux de détection (cybersécurité)

    Args:
        nom_modele: Nom du modèle à entraîner.
        X_train: Caractéristiques d'entraînement.
        y_train: Labels d'entraînement.
        X_test: Caractéristiques de test.
        y_test: Labels de test.

    Returns:
        Dictionnaire contenant toutes les métriques et le modèle entraîné.
    """
    description = DESCRIPTION_MODELES.get(nom_modele, nom_modele)
    logging.info("")
    logging.info("─" * 60)
    logging.info("Modèle : %s", description)
    logging.info("─" * 60)

    # ─── 2a : Initialisation du modèle ───────────────────────────────────────
    logging.info("  [Initialisation] Création du modèle...")
    detecteur = DetecteurIntrusions(type_modele=nom_modele)
    logging.info("  Paramètres : %s", detecteur.obtenir_parametres())

    # ─── 2b : Entraînement chronométré ──────────────────────────────────────
    logging.info("  [Entraînement] Début de l'apprentissage sur %d échantillons...",
                  len(X_train))
    debut_entrainement = time.time()
    resultats_entrainement = detecteur.entrainer(X_train, y_train)
    duree_entrainement = time.time() - debut_entrainement

    score_train = resultats_entrainement.get("score_entrainement", 0)
    logging.info(
        "  [Entraînement] Terminé en %.2f secondes. Score: %.4f",
        duree_entrainement, score_train,
    )

    # ─── 2c : Prédiction sur l'ensemble de test ─────────────────────────────
    logging.info("  [Prédiction] Évaluation sur %d échantillons de test...",
                  len(X_test))
    y_pred = detecteur.predire(X_test)
    logging.info("  [Prédiction] Terminée.")

    # ─── 2d : Calcul des métriques de classification ─────────────────────────
    logging.info("  [Évaluation] Calcul des métriques...")
    evaluateur = Evaluateur()
    metriques = evaluateur.calculer_metriques(y_test, y_pred)

    # ─── 2e : Matrice de confusion ──────────────────────────────────────────
    logging.info("  [Évaluation] Calcul de la matrice de confusion...")
    matrice = evaluateur.calculer_matrice_confusion(y_test, y_pred)

    # Affichage lisible de la matrice de confusion
    vn, fp, fn, vp = matrice.ravel()
    logging.info(
        "  Matrice de confusion :\n"
        "                    Prédit\n"
        "                   Normal  Attaque\n"
        "  Réel    Normal    %4d    %4d\n"
        "          Attaque   %4d    %4d",
        vn, fp, fn, vp,
    )

    # ─── 2f : Rapport de classification détaillé ────────────────────────────
    rapport_texte = evaluateur.generer_rapport(y_test, y_pred)
    logging.info("  [Évaluation] Rapport de classification :")
    for ligne in rapport_texte.strip().split("\n"):
        logging.info("   %s", ligne)

    # ─── 2g : Taux de détection spécifiques à la cybersécurité ─────────────
    taux = evaluateur.calculer_taux_detection(matrice)
    logging.info(
        "  [Cybersécurité] Taux de détection des attaques : %.2f%%",
        taux["taux_vrais_positifs"] * 100,
    )
    logging.info(
        "  [Cybersécurité] Taux de fausses alarmes : %.2f%%",
        taux["taux_faux_positifs"] * 100,
    )

    # ─── 2h : Agrégation de tous les résultats ──────────────────────────────
    resultats = {
        "nom_modele": nom_modele,
        "description": description,
        "modele_entraine": detecteur,
        "duree_entrainement": duree_entrainement,
        "accuracy": metriques["accuracy"],
        "precision": metriques["precision"],
        "rappel": metriques["recall"],
        "f1_score": metriques["f1_score"],
        "matrice_confusion": matrice,
        "rapport_classification": rapport_texte,
        "vp": int(vp),   # Vrais positifs (attaques détectées)
        "vn": int(vn),   # Vrais négatifs (trafic normal correct)
        "fp": int(fp),   # Faux positifs (fausses alarmes)
        "fn": int(fn),   # Faux négatifs (attaques non détectées)
        "taux_detection": taux["taux_vrais_positifs"],
        "taux_faux_positifs": taux["taux_faux_positifs"],
    }

    return resultats


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  ÉTAPE 3 : COMPARAISON ET SÉLECTION DU MEILLEUR MODÈLE                    ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def comparer_et_selectionner_meilleur(
    resultats_modeles: List[dict],
) -> Tuple[pd.DataFrame, dict]:
    """
    ÉTAPE 3 : Compare tous les modèles et sélectionne le meilleur.

    Le classement est basé sur le F1-Score (moyenne harmonique de
    la précision et du rappel). En cas d'égalité, le modèle le plus
    rapide à l'entraînement est privilégié.

    Le meilleur modèle est celui qui maximise le F1-Score, car :
    - L'Accuracy peut être trompeuse si les classes sont déséquilibrées
    - Le F1-Score équilibre la précision et le rappel
    - En détection d'intrusions, un bon F1-Score signifie qu'on détecte
      les attaques (rappel) sans trop de fausses alarmes (précision)

    Args:
        resultats_modeles: Liste des dictionnaires de résultats.

    Returns:
        Tuple (dataframe_comparatif, meilleur_resultat).
    """
    logging.info("")
    logging.info("╔══════════════════════════════════════════════════╗")
    logging.info("║   ÉTAPE 3 : COMPARAISON DES MODÈLES             ║")
    logging.info("╚══════════════════════════════════════════════════╝")
    logging.info("")

    # Création du DataFrame comparatif à partir des résultats
    lignes = []
    for r in resultats_modeles:
        lignes.append({
            "modèle": r["nom_modele"],
            "description": r["description"],
            "accuracy": round(r["accuracy"], 4),
            "precision": round(r["precision"], 4),
            "rappel": round(r["rappel"], 4),
            "f1_score": round(r["f1_score"], 4),
            "taux_detection": round(r["taux_detection"], 4),
            "taux_faux_positifs": round(r["taux_faux_positifs"], 4),
            "temps": round(r["duree_entrainement"], 3),
            "vp": r["vp"],
            "vn": r["vn"],
            "fp": r["fp"],
            "fn": r["fn"],
        })

    df = pd.DataFrame(lignes)

    # Classement par F1-Score décroissant (meilleur en premier)
    df.sort_values(by="f1_score", ascending=False, inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.index += 1  # Index commence à 1 pour le classement

    # ─── Identification automatique du meilleur modèle ─────────────────────
    meilleur_index = df.index[0]  # Première ligne = meilleur F1-Score
    meilleur_nom = df.loc[meilleur_index, "modèle"]

    # Recherche du dictionnaire de résultats correspondant
    meilleur_resultat = None
    for r in resultats_modeles:
        if r["nom_modele"] == meilleur_nom:
            meilleur_resultat = r
            break

    logging.info(
        ">>> Meilleur modèle identifié : %s (F1-Score = %.4f)",
        meilleur_resultat["description"],
        meilleur_resultat["f1_score"],
    )

    return df, meilleur_resultat


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  ÉTAPE 4 : SAUVEGARDE DU MEILLEUR MODÈLE                                 ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def sauvegarder_meilleur_modele(
    meilleur_resultat: dict,
    infos_dataset: dict,
    dossier_sortie: Optional[Path] = None,
) -> Path:
    """
    ÉTAPE 4 : Sauvegarde le meilleur modèle entraîné et un rapport complet.

    Fichiers sauvegardés dans data/processed/ :
    - meilleur_modele.joblib        : Modèle entraîné (format joblib)
    - meilleur_modele_metriques.txt : Métriques et rapport détaillé
    - rapport_entrainement.txt      : Rapport complet du pipeline

    Args:
        meilleur_resultat: Dictionnaire du meilleur modèle.
        infos_dataset: Métadonnées du dataset utilisé.
        dossier_sortie: Dossier de destination.

    Returns:
        Chemin du fichier du modèle sauvegardé.
    """
    dossier_sortie = dossier_sortie or DOSSIER_DATA_TRANSFORME
    dossier_sortie.mkdir(parents=True, exist_ok=True)

    logging.info("")
    logging.info("╔══════════════════════════════════════════════════╗")
    logging.info("║   ÉTAPE 4 : SAUVEGARDE DU MEILLEUR MODÈLE      ║")
    logging.info("╚══════════════════════════════════════════════════╝")
    logging.info("")

    # ─── 4a : Sauvegarde du modèle entraîné ─────────────────────────────────
    chemin_modele = dossier_sortie / "meilleur_modele.joblib"
    detecteur = meilleur_resultat["modele_entraine"]
    sauvegarder_modele(detecteur.modele, chemin_modele)
    logging.info("Modèle sauvegardé : %s", chemin_modele)
    logging.info("  Type    : %s", meilleur_resultat["description"])
    logging.info("  F1-Score: %.4f", meilleur_resultat["f1_score"])

    # ─── 4b : Sauvegarde des métriques du meilleur modèle ───────────────────
    chemin_metriques = dossier_sortie / "meilleur_modele_metriques.txt"
    with open(chemin_metriques, "w", encoding="utf-8") as f:
        f.write("╔══════════════════════════════════════════════════╗\n")
        f.write("║   NETGUARD AI - Meilleur modèle entraîné        ║\n")
        f.write("╚══════════════════════════════════════════════════╝\n\n")

        f.write(f"Date d'entraînement : {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Dataset             : {infos_dataset['nom_complet']}\n")
        f.write(f"Source              : {infos_dataset['source']}\n\n")

        f.write("─" * 50 + "\n")
        f.write("  INFORMATIONS DU MODÈLE\n")
        f.write("─" * 50 + "\n")
        f.write(f"  Modèle    : {meilleur_resultat['description']}\n")
        f.write(f"  F1-Score  : {meilleur_resultat['f1_score']:.4f}\n")
        f.write(f"  Accuracy  : {meilleur_resultat['accuracy']:.4f}\n")
        f.write(f"  Précision : {meilleur_resultat['precision']:.4f}\n")
        f.write(f"  Rappel    : {meilleur_resultat['rappel']:.4f}\n")
        f.write(f"  Temps     : {meilleur_resultat['duree_entrainement']:.2f}s\n\n")

        f.write("─" * 50 + "\n")
        f.write("  MATRICE DE CONFUSION\n")
        f.write("─" * 50 + "\n")
        f.write(f"                  Prédit\n")
        f.write(f"                 Normal  Attaque\n")
        f.write(f"  Réel  Normal    {meilleur_resultat['vn']:4d}    {meilleur_resultat['fp']:4d}\n")
        f.write(f"        Attaque   {meilleur_resultat['fn']:4d}    {meilleur_resultat['vp']:4d}\n\n")

        f.write("─" * 50 + "\n")
        f.write("  TAUX DE DÉTECTION\n")
        f.write("─" * 50 + "\n")
        f.write(f"  Taux de détection des attaques : {meilleur_resultat['taux_detection']:.2%}\n")
        f.write(f"  Taux de fausses alarmes        : {meilleur_resultat['taux_faux_positifs']:.2%}\n")
        f.write(f"  Taux de vrais négatifs         : {(1-meilleur_resultat['taux_faux_positifs']):.2%}\n")
        f.write(f"  Taux de faux négatifs          : {(1-meilleur_resultat['taux_detection']):.2%}\n\n")

        f.write("─" * 50 + "\n")
        f.write("  RAPPORT DE CLASSIFICATION\n")
        f.write("─" * 50 + "\n")
        f.write(meilleur_resultat["rapport_classification"])

    logging.info("Métriques sauvegardées : %s", chemin_metriques)

    # ─── 4c : Génération du rapport complet du pipeline ─────────────────────
    return chemin_modele


def generer_rapport_complet(
    dataframe_comparatif: pd.DataFrame,
    meilleur_resultat: dict,
    infos_dataset: dict,
    duree_totale: float,
    dossier_sortie: Optional[Path] = None,
) -> Path:
    """
    Génère un rapport texte complet du pipeline d'entraînement.

    Inclut :
    - Résumé du dataset
    - Tableau comparatif de tous les modèles
    - Détails du meilleur modèle
    - Matrice de confusion et métriques

    Args:
        dataframe_comparatif: DataFrame classé des résultats.
        meilleur_resultat: Dictionnaire du meilleur modèle.
        infos_dataset: Métadonnées du dataset.
        duree_totale: Durée totale du pipeline.
        dossier_sortie: Dossier de destination.

    Returns:
        Chemin du rapport généré.
    """
    dossier_sortie = dossier_sortie or DOSSIER_DATA_TRANSFORME
    dossier_sortie.mkdir(parents=True, exist_ok=True)

    chemin_rapport = dossier_sortie / "rapport_entrainement.txt"

    with open(chemin_rapport, "w", encoding="utf-8") as f:
        f.write("╔══════════════════════════════════════════════════╗\n")
        f.write("║   RAPPORT D'ENTRAÎNEMENT NETGUARD AI            ║\n")
        f.write("╚══════════════════════════════════════════════════╝\n\n")

        f.write(f"Date          : {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Durée totale  : {duree_totale:.2f} secondes\n")
        f.write(f"Seed aléatoire: {SEED_ALEATOIRE}\n\n")

        # ─── Section 1 : Dataset ────────────────────────────────────────────
        f.write("═" * 60 + "\n")
        f.write("  1. DATASET\n")
        f.write("═" * 60 + "\n\n")
        f.write(f"  Nom               : {infos_dataset['nom_complet']}\n")
        f.write(f"  Source            : {infos_dataset['source']}\n")
        f.write(f"  Échantillons train: {infos_dataset['nb_echantillons_train']}\n")
        f.write(f"  Échantillons test : {infos_dataset['nb_echantillons_test']}\n")
        f.write(f"  Caractéristiques  : {infos_dataset['nb_caracteristiques']}\n")
        f.write(f"  Ratio test        : {infos_dataset['ratio_test']:.0%}\n\n")

        # ─── Section 2 : Modèles entraînés ──────────────────────────────────
        f.write("═" * 60 + "\n")
        f.write("  2. MODÈLES ENTRAÎNÉS\n")
        f.write("═" * 60 + "\n\n")

        for idx, row in dataframe_comparatif.iterrows():
            f.write(f"  Rang {idx} : {row['description']}\n")
            f.write(f"    Accuracy        : {row['accuracy']:.4f}\n")
            f.write(f"    Précision       : {row['precision']:.4f}\n")
            f.write(f"    Rappel          : {row['rappel']:.4f}\n")
            f.write(f"    F1-Score        : {row['f1_score']:.4f}\n")
            f.write(f"    Taux détection  : {row['taux_detection']:.2%}\n")
            f.write(f"    Taux faux +     : {row['taux_faux_positifs']:.2%}\n")
            f.write(f"    Temps           : {row['temps']:.2f}s\n")
            f.write(f"    VP/VN/FP/FN     : {int(row['vp'])}/{int(row['vn'])}/{int(row['fp'])}/{int(row['fn'])}\n\n")

        # ─── Section 3 : Meilleur modèle ────────────────────────────────────
        f.write("═" * 60 + "\n")
        f.write("  3. MEILLEUR MODÈLE\n")
        f.write("═" * 60 + "\n\n")
        f.write(f"  {meilleur_resultat['description']}\n\n")
        f.write(f"  Accuracy   : {meilleur_resultat['accuracy']:.4f}\n")
        f.write(f"  Précision  : {meilleur_resultat['precision']:.4f}\n")
        f.write(f"  Rappel     : {meilleur_resultat['rappel']:.4f}\n")
        f.write(f"  F1-Score   : {meilleur_resultat['f1_score']:.4f}\n")
        f.write(f"  Temps      : {meilleur_resultat['duree_entrainement']:.2f}s\n\n")

        f.write("  Matrice de confusion :\n")
        f.write(f"                    Prédit\n")
        f.write(f"                   Normal  Attaque\n")
        f.write(f"  Réel    Normal    {meilleur_resultat['vn']:4d}    {meilleur_resultat['fp']:4d}\n")
        f.write(f"          Attaque   {meilleur_resultat['fn']:4d}    {meilleur_resultat['vp']:4d}\n\n")

        f.write(f"  Taux de détection des attaques : {meilleur_resultat['taux_detection']:.2%}\n")
        f.write(f"  Taux de fausses alarmes        : {meilleur_resultat['taux_faux_positifs']:.2%}\n\n")

        f.write("═" * 60 + "\n")
        f.write("  RAPPORT D'ENTRAÎNEMENT TERMINÉ\n")
        f.write("═" * 60 + "\n")

    logging.info("Rapport complet généré : %s", chemin_rapport)
    return chemin_rapport


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  AFFICHAGE DES RÉSULTATS                                                  ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def afficher_en_tete() -> None:
    """Affiche l'en-tête du pipeline."""
    print()
    print("+" + "=" * 58 + "+")
    print("|" + " " * 17 + "NETGUARD AI" + " " * 28 + "|")
    print("|" + " " * 6 + "Pipeline d'entraînement complet" + " " * 18 + "|")
    print("|" + " " * 10 + "Détection d'intrusions" + " " * 23 + "|")
    print("+" + "=" * 58 + "+")
    print()


def afficher_tableau_comparatif(df: pd.DataFrame) -> None:
    """
    Affiche un tableau formaté comparant tous les modèles.

    Le tableau est trié par F1-Score décroissant.
    Le meilleur modèle est mis en évidence.

    Args:
        df: DataFrame contenant les résultats classés.
    """
    print()
    print("=" * 95)
    print("  TABLEAU COMPARATIF DES MODÈLES (classé par F1-Score)")
    print("=" * 95)

    # En-tête du tableau
    print(f"  {'#':<3} {'Modèle':<24} {'Accuracy':<10} {'Précision':<10} "
          f"{'Rappel':<10} {'F1-Score':<10} {'Détection':<10} {'Faux+':<10}")
    print("  " + "-" * 92)

    # Lignes du tableau
    for idx, row in df.iterrows():
        # Le meilleur modèle (rang 1) est mis en évidence avec ">>>"
        prefixe = ">>>" if idx == 1 else "   "
        print(f"  {prefixe} {idx:<3} {row['description']:<24} "
              f"{row['accuracy']:<10.4f} {row['precision']:<10.4f} "
              f"{row['rappel']:<10.4f} {row['f1_score']:<10.4f} "
              f"{row['taux_detection']:<10.2%} {row['taux_faux_positifs']:<10.2%}")

    print("=" * 95)
    print()


def afficher_resume_meilleur_modele(meilleur_resultat: dict) -> None:
    """
    Affiche un résumé détaillé du meilleur modèle.

    Args:
        meilleur_resultat: Dictionnaire du meilleur modèle.
    """
    print()
    print("=" * 60)
    print("  >>> MEILLEUR MODÈLE <<<")
    print("=" * 60)
    print(f"  Modèle       : {meilleur_resultat['description']}")
    print(f"  F1-Score     : {meilleur_resultat['f1_score']:.4f}")
    print(f"  Accuracy     : {meilleur_resultat['accuracy']:.4f}")
    print(f"  Précision    : {meilleur_resultat['precision']:.4f}")
    print(f"  Rappel       : {meilleur_resultat['rappel']:.4f}")
    print(f"  Temps        : {meilleur_resultat['duree_entrainement']:.2f}s")
    print()
    print(f"  Taux de détection des attaques : {meilleur_resultat['taux_detection']:.2%}")
    print(f"  Taux de fausses alarmes        : {meilleur_resultat['taux_faux_positifs']:.2%}")
    print()

    # Matrice de confusion formatée
    vn, fp, fn, vp = (
        meilleur_resultat["vn"],
        meilleur_resultat["fp"],
        meilleur_resultat["fn"],
        meilleur_resultat["vp"],
    )
    print(f"  Matrice de confusion :")
    print(f"                    Prédit")
    print(f"                   Normal  Attaque")
    print(f"  Réel    Normal    {vn:4d}    {fp:4d}")
    print(f"          Attaque   {fn:4d}    {vp:4d}")
    print()
    print("=" * 60)
    print()


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  PIPELINE COMPLET                                                         ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def executer_pipeline_complet(args: argparse.Namespace) -> None:
    """
    Exécute le pipeline complet d'entraînement.

    Orchestre les 4 étapes principales :
    1. Préparation des données
    2. Entraînement des modèles (1 ou 6 selon les arguments)
    3. Comparaison et sélection du meilleur modèle
    4. Sauvegarde du meilleur modèle et génération du rapport

    Args:
        args: Arguments de la ligne de commande.
    """
    duree_debut = time.time()

    # ─── En-tête ────────────────────────────────────────────────────────────
    afficher_en_tete()
    logging.info("Seed aléatoire : %d", SEED_ALEATOIRE)

    # ═════════════════════════════════════════════════════════════════════════
    #  ÉTAPE 1 : PRÉPARATION DES DONNÉES
    # ═════════════════════════════════════════════════════════════════════════

    with Chronometre("Préparation des données"):
        X_train, X_test, y_train, y_test, infos_dataset = preparer_donnees(
            nom_dataset=args.dataset,
            taille_test=args.taille_test,
            download=args.download,
        )

    # ═════════════════════════════════════════════════════════════════════════
    #  ÉTAPE 2 : ENTRAÎNEMENT DES MODÈLES
    # ═════════════════════════════════════════════════════════════════════════

    if args.modele:
        # Mode mono-modèle : entraîner uniquement celui demandé
        logging.info("")
        logging.info("╔══════════════════════════════════════════════════╗")
        logging.info("║   ÉTAPE 2 : ENTRAÎNEMENT DU MODÈLE              ║")
        logging.info("╚══════════════════════════════════════════════════╝")

        resultat = entrainer_et_evaluer_modele(
            args.modele, X_train, y_train, X_test, y_test,
        )
        resultats_modeles = [resultat]
        dataframe_comparatif = None

    else:
        # Mode multi-modèles : entraîner les 6 modèles
        logging.info("")
        logging.info("╔══════════════════════════════════════════════════╗")
        logging.info("║   ÉTAPE 2 : ENTRAÎNEMENT DES 6 MODÈLES         ║")
        logging.info("╚══════════════════════════════════════════════════╝")
        logging.info("Modèles à entraîner : %s", ", ".join(
            [f"{m} ({DESCRIPTION_MODELES[m]})" for m in LISTE_MODELES]
        ))

        resultats_modeles = []
        for nom_modele in LISTE_MODELES:
            with Chronometre(f"Entraînement de {nom_modele}"):
                resultat = entrainer_et_evaluer_modele(
                    nom_modele, X_train, y_train, X_test, y_test,
                )
                resultats_modeles.append(resultat)

    # ═════════════════════════════════════════════════════════════════════════
    #  ÉTAPE 3 : COMPARAISON ET SÉLECTION
    # ═════════════════════════════════════════════════════════════════════════

    if len(resultats_modeles) > 1 and not args.no_compare:
        with Chronometre("Comparaison des modèles"):
            dataframe_comparatif, meilleur_resultat = (
                comparer_et_selectionner_meilleur(resultats_modeles)
            )

        # Affichage du tableau comparatif
        afficher_tableau_comparatif(dataframe_comparatif)
    else:
        # Un seul modèle : pas de comparaison
        meilleur_resultat = resultats_modeles[0]
        dataframe_comparatif = None
        logging.info(
            "Modèle unique : %s", meilleur_resultat["description"]
        )

    # Affichage du résumé du meilleur modèle
    afficher_resume_meilleur_modele(meilleur_resultat)

    # ═════════════════════════════════════════════════════════════════════════
    #  ÉTAPE 4 : SAUVEGARDE
    # ═════════════════════════════════════════════════════════════════════════

    if args.sauvegarder:
        with Chronometre("Sauvegarde des résultats"):
            # Sauvegarde du meilleur modèle
            chemin_modele = sauvegarder_meilleur_modele(
                meilleur_resultat, infos_dataset,
            )

            # Génération du rapport complet
            if dataframe_comparatif is not None:
                duree_totale = time.time() - duree_debut
                generer_rapport_complet(
                    dataframe_comparatif,
                    meilleur_resultat,
                    infos_dataset,
                    duree_totale,
                )

    # ═════════════════════════════════════════════════════════════════════════
    #  RÉSUMÉ FINAL
    # ═════════════════════════════════════════════════════════════════════════

    duree_totale = time.time() - duree_debut
    print()
    print("=" * 60)
    print("  PIPELINE D'ENTRAÎNEMENT TERMINÉ AVEC SUCCÈS")
    print("=" * 60)
    print(f"  Dataset     : {infos_dataset['nom_complet']}")
    print(f"  Modèles     : {len(resultats_modeles)} entraîné(s)")
    print(f"  Meilleur    : {meilleur_resultat['description']}")
    print(f"  F1-Score    : {meilleur_resultat['f1_score']:.4f}")
    print(f"  Durée totale: {duree_totale:.2f}s")
    if args.sauvegarder:
        print(f"  Modèle      : {DOSSIER_DATA_TRANSFORME / 'meilleur_modele.joblib'}")
        print(f"  Rapport     : {DOSSIER_DATA_TRANSFORME / 'rapport_entrainement.txt'}")
    print("=" * 60)
    print()


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  POINT D'ENTRÉE                                                           ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def main() -> None:
    """Point d'entrée principal du pipeline d'entraînement."""
    parser = definir_arguments()
    args = parser.parse_args()

    # Configuration du niveau de logging
    niveau_log = logging.DEBUG if args.verbose else logging.INFO
    configurer_logging(niveau=niveau_log)

    # Exécution du pipeline complet
    executer_pipeline_complet(args)


if __name__ == "__main__":
    main()
