"""
Script de téléchargement automatique du dataset CICIDS2017.

Ce script :
1. Télécharge le dataset CICIDS2017 depuis une source fiable
2. Extrait les fichiers CSV de l'archive ZIP
3. Combine les 8 fichiers CSV en un seul fichier consolidé
4. Nettoie les noms de colonnes et les valeurs aberrantes

Le dataset CICIDS2017 contient du trafic réseau étiqueté
avec des attaques variées (DoS, DDoS, Brute Force, etc.).

Sources de téléchargement (par ordre de priorité) :
  1. Dépôt officiel UNB (nécessite un formulaire)
  2. Mirrors académiques
  3. Archive.org

Utilisation :
    python datasets/download_cicids2017.py
"""

import logging
import sys
from pathlib import Path
from typing import Optional

# Ajout de la racine du projet pour les imports
PROJET_RACINE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJET_RACINE))

# ─── Imports ─────────────────────────────────────────────────────────────────

import zipfile
import shutil
import pandas as pd
import numpy as np

from src.config import (
    DOSSIER_DATASETS,
    CICIDS2017_CONFIG,
)
from src.utils.helpers import configurer_logging, Chronometre


# ─── URLs de téléchargement ──────────────────────────────────────────────────

# Liste des URLs candidates pour télécharger MachineLearningCSV.zip
# Tentées dans l'ordre jusqu'à ce que l'une fonctionne
URLS_CICIDS2017 = [
    # Mirror 1 : Archive.org (hébergement stable)
    "https://archive.org/download/cic-ids-2017-dataset/MachineLearningCSV.zip",
    # Mirror 2 : Dépôt universitaire alternatif
    "https://cloud.cs.uni-bonn.de/s/cicids2017/download/MachineLearningCSV.zip",
]

# Nom du fichier ZIP attendu
NOM_FICHIER_ZIP = "MachineLearningCSV.zip"

# Liste des 8 fichiers CSV attendus dans l'archive
FICHIERS_CSV_ATTENDUS = [
    "Monday-WorkingHours.pcap_ISCX.csv",
    "Tuesday-WorkingHours.pcap_ISCX.csv",
    "Wednesday-workingHours.pcap_ISCX.csv",
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
    "Friday-WorkingHours-Morning.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
]


def verifier_si_dataset_existe() -> bool:
    """
    Vérifie si le dataset consolidé existe déjà.

    Returns:
        True si le fichier consolidé est présent.
    """
    chemin_sortie = CICIDS2017_CONFIG["chemin_sortie"]
    if chemin_sortie.exists():
        taille_mo = chemin_sortie.stat().st_size / (1024 * 1024)
        logging.info(
            "Dataset déjà présent : %s (%.1f Mo)",
            chemin_sortie,
            taille_mo,
        )
        return True
    return False


def verifier_si_zip_existe() -> Optional[Path]:
    """
    Vérifie si l'archive ZIP est déjà présente.

    Returns:
        Chemin du ZIP s'il existe, None sinon.
    """
    chemin_zip = DOSSIER_DATASETS / NOM_FICHIER_ZIP
    if chemin_zip.exists():
        logging.info("Archive ZIP déjà présente : %s", chemin_zip)
        return chemin_zip
    return None


def telecharger_fichier(url: str, destination: Path) -> bool:
    """
    Télécharge un fichier depuis une URL.

    Args:
        url: URL du fichier à télécharger.
        destination: Chemin de destination.

    Returns:
        True si le téléchargement a réussi, False sinon.
    """
    import requests

    try:
        logging.info("Tentative de téléchargement depuis : %s", url)
        logging.info("Destination : %s", destination)

        # Téléchargement avec barre de progression
        reponse = requests.get(url, stream=True, timeout=30, allow_redirects=True)

        if reponse.status_code != 200:
            logging.warning(
                "Échec du téléchargement (HTTP %d)", reponse.status_code
            )
            return False

        # Taille totale du fichier
        taille_totale = int(reponse.headers.get("content-length", 0))
        taille_mo = taille_totale / (1024 * 1024)
        logging.info("Taille du fichier : %.1f Mo", taille_mo)

        # Écriture par morceaux
        with open(destination, "wb") as f:
            telecharge = 0
            for morceau in reponse.iter_content(chunk_size=1024 * 1024):
                if morceau:
                    f.write(morceau)
                    telecharge += len(morceau)
                    # Affichage de la progression
                    if taille_totale > 0:
                        pourcentage = telecharge / taille_totale * 100
                        if telecharge % (10 * 1024 * 1024) < 1024 * 1024:
                            logging.info(
                                "Progression : %.0f%% (%.0f/%.0f Mo)",
                                pourcentage,
                                telecharge / (1024 * 1024),
                                taille_mo,
                            )

        logging.info("Téléchargement terminé avec succès !")
        return True

    except requests.exceptions.Timeout:
        logging.warning("Délai d'attente dépassé pour : %s", url)
    except requests.exceptions.ConnectionError:
        logging.warning("Connexion impossible pour : %s", url)
    except Exception as e:
        logging.warning("Erreur inattendue : %s", e)

    return False


def telecharger_dataset() -> Optional[Path]:
    """
    Télécharge le dataset CICIDS2017 depuis les sources disponibles.

    Tente chaque URL de la liste jusqu'à ce que l'une fonctionne.

    Returns:
        Chemin du fichier ZIP téléchargé, None si échec.
    """
    chemin_zip = DOSSIER_DATASETS / NOM_FICHIER_ZIP

    # Vérifier si déjà présent
    zip_existant = verifier_si_zip_existe()
    if zip_existant:
        return zip_existant

    logging.info(
        "Téléchargement du dataset CICIDS2017 (~550 Mo)..."
    )

    # Tentative avec chaque URL
    for url in URLS_CICIDS2017:
        if telecharger_fichier(url, chemin_zip):
            return chemin_zip

    # Aucune URL n'a fonctionné
    logging.error("=" * 60)
    logging.error("TÉLÉCHARGEMENT AUTOMATIQUE IMPOSSIBLE")
    logging.error("=" * 60)
    logging.error(
        "Le dataset CICIDS2017 n'a pas pu être téléchargé "
        "automatiquement."
    )
    logging.error("")
    logging.error("Solution manuelle :")
    logging.error(
        "1. Rendez-vous sur : "
        "https://www.unb.ca/cic/datasets/ids-2017.html"
    )
    logging.error(
        "2. Remplissez le formulaire de téléchargement"
    )
    logging.error(
        "3. Téléchargez 'MachineLearningCSV.zip' (~550 Mo)"
    )
    logging.error(
        "4. Placez le fichier dans : %s", DOSSIER_DATASETS
    )
    logging.error(
        "5. Réexécutez ce script pour l'extraction"
    )
    logging.error("=" * 60)

    return None


def extraire_zip(chemin_zip: Path) -> Path:
    """
    Extrait l'archive ZIP dans un dossier temporaire.

    Args:
        chemin_zip: Chemin vers le fichier ZIP.

    Returns:
        Chemin du dossier d'extraction.
    """
    dossier_extraction = DOSSIER_DATASETS / "extraction_cicids2017"

    if dossier_extraction.exists():
        logging.info("Nettoyage de l'extraction précédente...")
        shutil.rmtree(dossier_extraction)

    logging.info("Extraction de l'archive ZIP...")
    with zipfile.ZipFile(chemin_zip, "r") as zip_ref:
        zip_ref.extractall(dossier_extraction)

    # Afficher la liste des fichiers extraits
    fichiers_extraits = list(dossier_extraction.rglob("*.csv"))
    logging.info(
        "Extraction terminée : %d fichiers CSV trouvés",
        len(fichiers_extraits),
    )

    return dossier_extraction


def consolider_fichiers_csv(dossier_extraction: Path) -> pd.DataFrame:
    """
    Combine les 8 fichiers CSV en un seul DataFrame.

    Pendant la consolidation :
    - Les noms de colonnes sont nettoyés (espaces supprimés)
    - Les lignes vides sont ignorées
    - Les colonnes inutiles sont supprimées

    Args:
        dossier_extraction: Dossier contenant les CSV extraits.

    Returns:
        DataFrame consolidé contenant toutes les données.
    """
    logging.info("Consolidation des fichiers CSV...")
    chunks = []

    for fichier in FICHIERS_CSV_ATTENDUS:
        chemin_fichier = dossier_extraction / fichier

        if not chemin_fichier.exists():
            logging.warning(
                "Fichier non trouvé : %s", fichier
            )
            continue

        logging.info(
            "Chargement : %s", fichier
        )

        # Chargement du CSV
        df = pd.read_csv(chemin_fichier)

        # Nettoyage des noms de colonnes
        df.columns = df.columns.str.strip()

        # Suppression des colonnes entièrement vides
        df.dropna(axis=1, how="all", inplace=True)

        # Ajout des informations de provenance
        df["_fichier_source"] = fichier

        chunks.append(df)

    if not chunks:
        raise ValueError(
            "Aucun fichier CSV valide trouvé dans l'archive."
        )

    # Fusion de tous les DataFrames
    df_consolide = pd.concat(chunks, ignore_index=True)

    logging.info(
        "Consolidation terminée : %d lignes, %d colonnes",
        df_consolide.shape[0],
        df_consolide.shape[1],
    )

    return df_consolide


def nettoyer_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Nettoie le dataset consolidé.

    Étapes de nettoyage :
    - Suppression des colonnes d'identification (IP, ports, etc.)
    - Suppression des valeurs Infinity et NaN
    - Suppression des doublons
    - Conversion de la colonne Label en binaire (BENIGN=0, attaque=1)

    Args:
        df: DataFrame brut consolidé.

    Returns:
        DataFrame nettoyé et prêt pour le ML.
    """
    logging.info("Nettoyage du dataset...")

    # ─── Suppression des colonnes d'identification ───────────────────────────
    colonnes_a_supprimer = [
        col for col in CICIDS2017_CONFIG["colonnes_a_ignorer"]
        if col in df.columns
    ]
    if colonnes_a_supprimer:
        df.drop(columns=colonnes_a_supprimer, inplace=True)
        logging.info(
            "Colonnes d'identification supprimées : %d",
            len(colonnes_a_supprimer),
        )

    # ─── Suppression des colonnes de type objet (non numériques) ─────────────
    colonnes_non_numeriques = df.select_dtypes(include=["object"]).columns.tolist()

    # Garder la colonne Label et _fichier_source si présentes
    colonnes_a_garder = [CICIDS2017_CONFIG["colonne_label"], "_fichier_source"]
    colonnes_suppr = [
        c for c in colonnes_non_numeriques
        if c not in colonnes_a_garder and c in df.columns
    ]
    if colonnes_suppr:
        df.drop(columns=colonnes_suppr, inplace=True)
        logging.info(
            "Colonnes non numériques supprimées : %d",
            len(colonnes_suppr),
        )

    # ─── Gestion des valeurs Infinity et NaN ─────────────────────────────────
    nb_avant = len(df)

    # Remplacer les valeurs infinies par NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Supprimer les lignes avec des NaN
    colonnes_ml = [
        c for c in df.columns
        if c not in [CICIDS2017_CONFIG["colonne_label"], "_fichier_source"]
    ]
    df.dropna(subset=colonnes_ml, inplace=True)

    nb_supprimes = nb_avant - len(df)
    if nb_supprimes > 0:
        logging.info(
            "Lignes avec valeurs invalides supprimées : %d",
            nb_supprimes,
        )

    # ─── Suppression des doublons ────────────────────────────────────────────
    nb_avant = len(df)
    df.drop_duplicates(subset=colonnes_ml, inplace=True)
    nb_doublons = nb_avant - len(df)
    if nb_doublons > 0:
        logging.info("Doublons supprimés : %d", nb_doublons)

    # ─── Nettoyage de la colonne Label ───────────────────────────────────────
    if CICIDS2017_CONFIG["colonne_label"] in df.columns:
        # Nettoyer les espaces
        df[CICIDS2017_CONFIG["colonne_label"]] = (
            df[CICIDS2017_CONFIG["colonne_label"]]
            .astype(str)
            .str.strip()
            .str.upper()
        )

        # Statistiques des labels
        distribution = (
            df[CICIDS2017_CONFIG["colonne_label"]].value_counts()
        )
        logging.info(
            "Distribution des labels avant conversion binaire :"
        )
        for label, count in distribution.items():
            logging.info("  - %s : %d", label, count)

    # ─── Suppression de la colonne _fichier_source ───────────────────────────
    if "_fichier_source" in df.columns:
        df.drop(columns=["_fichier_source"], inplace=True)

    logging.info(
        "Nettoyage terminé : %d lignes restantes",
        len(df),
    )

    return df


def convertir_label_binaire(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convertit la colonne Label en binaire (0 = BENIGN, 1 = attaque).

    Args:
        df: DataFrame avec la colonne Label.

    Returns:
        DataFrame avec la colonne Label en binaire.
    """
    colonne_label = CICIDS2017_CONFIG["colonne_label"]

    if colonne_label not in df.columns:
        logging.warning("Colonne '%s' introuvable", colonne_label)
        return df

    # Conversion : "BENIGN" → 0, tout le reste → 1
    df[colonne_label] = df[colonne_label].apply(
        lambda x: 0 if x == "BENIGN" else 1
    )

    # Statistiques après conversion
    nb_benign = (df[colonne_label] == 0).sum()
    nb_attaque = (df[colonne_label] == 1).sum()
    logging.info(
        "Conversion binaire : %d BENIGN, %d ATTAQUES",
        nb_benign,
        nb_attaque,
    )

    # Renommer la colonne 'Label' → 'label' (convention du projet)
    df.rename(columns={colonne_label: "label"}, inplace=True)

    return df


def sauvegarder_dataset(df: pd.DataFrame) -> Path:
    """
    Sauvegarde le dataset consolidé et nettoyé.

    Args:
        df: DataFrame à sauvegarder.

    Returns:
        Chemin du fichier sauvegardé.
    """
    chemin_sortie = CICIDS2017_CONFIG["chemin_sortie"]
    chemin_sortie.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(chemin_sortie, index=False)

    taille_mo = chemin_sortie.stat().st_size / (1024 * 1024)
    logging.info(
        "Dataset sauvegardé : %s (%.1f Mo)",
        chemin_sortie,
        taille_mo,
    )

    return chemin_sortie


def nettoyer_dossier_extraction(dossier: Path) -> None:
    """
    Supprime le dossier d'extraction temporaire.

    Args:
        dossier: Dossier à supprimer.
    """
    if dossier.exists():
        shutil.rmtree(dossier)
        logging.info("Dossier d'extraction temporaire supprimé.")


def executer_pipeline_complet() -> bool:
    """
    Exécute le pipeline complet de téléchargement et préparation.

    Returns:
        True si le dataset est prêt, False sinon.
    """
    # Étape 1 : Vérifier si déjà présent
    if verifier_si_dataset_existe():
        return True

    # Création du dossier datasets
    DOSSIER_DATASETS.mkdir(parents=True, exist_ok=True)

    # Étape 2 : Téléchargement
    chemin_zip = telecharger_dataset()
    if chemin_zip is None:
        return False

    # Étape 3 : Extraction
    dossier_extraction = None
    try:
        dossier_extraction = extraire_zip(chemin_zip)

        # Étape 4 : Consolidation
        df = consolider_fichiers_csv(dossier_extraction)

        # Étape 5 : Nettoyage
        df = nettoyer_dataset(df)

        # Étape 6 : Conversion binaire
        df = convertir_label_binaire(df)

        # Étape 7 : Sauvegarde
        sauvegarder_dataset(df)

        # Étape 8 : Nettoyage
        nettoyer_dossier_extraction(dossier_extraction)
        # Optionnel : supprimer le ZIP pour économiser de l'espace
        # chemin_zip.unlink()

        return True

    except Exception as e:
        logging.error("Erreur lors de la préparation du dataset : %s", e)
        if dossier_extraction:
            nettoyer_dossier_extraction(dossier_extraction)
        return False


def afficher_resume() -> None:
    """Affiche un résumé des informations sur le dataset préparé."""
    chemin_sortie = CICIDS2017_CONFIG["chemin_sortie"]

    if not chemin_sortie.exists():
        logging.warning("Dataset non disponible.")
        return

    df = pd.read_csv(chemin_sortie)
    print()
    print("=" * 60)
    print("  DATASET CICIDS2017 - Resume")
    print("=" * 60)
    print(f"  Fichier       : {chemin_sortie}")
    print(f"  Echantillons  : {df.shape[0]:,}")
    print(f"  Caracteristiques : {df.shape[1] - 1}")
    print(f"  Taille        : {chemin_sortie.stat().st_size / (1024*1024):.1f} Mo")
    print()

    if "label" in df.columns:
        nb_benign = (df["label"] == 0).sum()
        nb_attaque = (df["label"] == 1).sum()
        ratio_attaque = nb_attaque / len(df) * 100
        print(f"  Trafic normal (BENIGN) : {nb_benign:,}")
        print(f"  Attaques               : {nb_attaque:,}")
        print(f"  Ratio d'attaques       : {ratio_attaque:.1f}%")
        print()

    print("  Variables numeriques :")
    colonnes_num = df.select_dtypes(include=[np.number]).columns.tolist()
    colonnes_num = [c for c in colonnes_num if c != "label"]
    for col in colonnes_num[:10]:
        print(f"    - {col}")
    if len(colonnes_num) > 10:
        print(f"    ... et {len(colonnes_num) - 10} autres")
    print("=" * 60)
    print()


def main() -> None:
    """Point d'entrée du script de téléchargement."""
    configurer_logging()

    print()
    print("+" + "-" * 58 + "+")
    print("|" + " " * 12 + "Telechargement du dataset CICIDS2017" + " " * 13 + "|")
    print("+" + "-" * 58 + "+")
    print()

    with Chronometre("Preparation du dataset CICIDS2017"):
        succes = executer_pipeline_complet()

    if succes:
        afficher_resume()
        logging.info("Dataset CICIDS2017 pret a l'emploi !")
    else:
        logging.error(
            "Le dataset CICIDS2017 n'a pas pu etre prepare."
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
