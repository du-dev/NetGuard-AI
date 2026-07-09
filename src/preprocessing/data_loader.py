"""
Chargement et prétraitement des données réseau.

Ce module gère :
- Le chargement du dataset depuis un fichier CSV
- Le nettoyage des données (valeurs manquantes, doublons)
- La séparation en ensembles d'entraînement, validation et test
- La normalisation des caractéristiques numériques
- L'encodage des variables catégorielles
"""

import logging
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

from src.config import (
    DOSSIER_DATA_BRUT,
    FICHIER_DATASET,
    SEPARATEUR_CSV,
    FRACTION_ENTRAINEMENT,
    FRACTION_VALIDATION,
    COLONNE_CIBLE,
    COLONNES_A_IGNORER,
    SEED_ALEATOIRE,
)


class ChargeurDonnees:
    """Charge et prétraite les données réseau pour la détection d'intrusions."""

    def __init__(
        self,
        chemin_dataset: Optional[Path] = None,
    ):
        """
        Args:
            chemin_dataset: Chemin vers le fichier CSV.
                            Par défaut : data/raw/network_traffic.csv
        """
        self.chemin_dataset = chemin_dataset or (
            DOSSIER_DATA_BRUT / FICHIER_DATASET
        )
        self.donnees: Optional[pd.DataFrame] = None
        self.encodage_label: Optional[LabelEncoder] = None
        self.standardisation: Optional[StandardScaler] = None

    def charger_csv(self) -> pd.DataFrame:
        """
        Charge le dataset depuis un fichier CSV.

        Returns:
            DataFrame contenant les données brutes.

        Lève:
            FileNotFoundError: Si le fichier est introuvable.
        """
        if not self.chemin_dataset.exists():
            raise FileNotFoundError(
                f"Dataset introuvable : {self.chemin_dataset}\n"
                "Veuillez placer votre fichier CSV dans le dossier data/raw/"
            )

        logging.info("Chargement du dataset : %s", self.chemin_dataset)
        self.donnees = pd.read_csv(self.chemin_dataset, sep=SEPARATEUR_CSV)
        logging.info(
            "Dataset chargé : %d lignes, %d colonnes",
            self.donnees.shape[0],
            self.donnees.shape[1],
        )
        return self.donnees

    def nettoyer(self) -> pd.DataFrame:
        """
        Nettoie les données : suppression des colonnes inutiles,
        valeurs manquantes et doublons.

        Returns:
            DataFrame nettoyé.
        """
        if self.donnees is None:
            raise ValueError("Aucune donnée chargée. Appelez charger_csv() d'abord.")

        logging.info("Nettoyage des données...")

        # Suppression des colonnes à ignorer (si elles existent)
        colonnes_a_supprimer = [
            col for col in COLONNES_A_IGNORER if col in self.donnees.columns
        ]
        if colonnes_a_supprimer:
            self.donnees.drop(columns=colonnes_a_supprimer, inplace=True)
            logging.info("Colonnes ignorées : %s", colonnes_a_supprimer)

        # Suppression des doublons
        nb_avant = len(self.donnees)
        self.donnees.drop_duplicates(inplace=True)
        nb_supprimes = nb_avant - len(self.donnees)
        if nb_supprimes > 0:
            logging.info("Doublons supprimés : %d", nb_supprimes)

        # Gestion des valeurs manquantes
        valeurs_manquantes = self.donnees.isnull().sum()
        colonnes_avec_nan = valeurs_manquantes[valeurs_manquantes > 0]
        if not colonnes_avec_nan.empty:
            logging.info("Valeurs manquantes détectées :")
            for col, count in colonnes_avec_nan.items():
                logging.info("  - %s : %d", col, count)
            # Suppression des lignes avec des valeurs manquantes
            self.donnees.dropna(inplace=True)
            logging.info("Lignes avec valeurs manquantes supprimées.")

        logging.info("Données nettoyées : %d lignes restantes", len(self.donnees))
        return self.donnees

    def separer_jeux(
        self,
    ) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
    ]:
        """
        Sépare les données en ensembles d'entraînement, validation et test.

        Returns:
            Tuple (X_train, X_val, X_test, y_train, y_val, y_test).
        """
        if self.donnees is None:
            raise ValueError("Aucune donnée chargée. Appelez charger_csv() d'abord.")

        logging.info("Séparation des jeux de données...")

        # Séparation des caractéristiques (X) et de la cible (y)
        X = self.donnees.drop(columns=[COLONNE_CIBLE])
        y = self.donnees[COLONNE_CIBLE]

        # Premier split : entraînement (70%) et reste (30%)
        X_train, X_temp, y_train, y_temp = train_test_split(
            X,
            y,
            test_size=1.0 - FRACTION_ENTRAINEMENT,
            random_state=SEED_ALEATOIRE,
            stratify=y,
        )

        # Deuxième split : validation (15%) et test (15%)
        ratio_val = FRACTION_VALIDATION / (1.0 - FRACTION_ENTRAINEMENT)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp,
            y_temp,
            test_size=1.0 - ratio_val,
            random_state=SEED_ALEATOIRE,
            stratify=y_temp,
        )

        logging.info(
            "Répartition : Entraînement=%d, Validation=%d, Test=%d",
            len(X_train),
            len(X_val),
            len(X_test),
        )

        return X_train.values, X_val.values, X_test.values, y_train.values, y_val.values, y_test.values

    def encoder_et_normaliser(
        self,
        X_train: np.ndarray,
        X_val: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_val: np.ndarray,
        y_test: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Encode les cibles (labels) et normalise les caractéristiques.

        Args:
            X_train: Caractéristiques d'entraînement.
            X_val: Caractéristiques de validation.
            X_test: Caractéristiques de test.
            y_train: Cible d'entraînement.
            y_val: Cible de validation.
            y_test: Cible de test.

        Returns:
            Tuple (X_train_norm, X_val_norm, X_test_norm,
                   y_train_enc, y_val_enc, y_test_enc).
        """
        # Encodage des labels (normal=0, attaque=1)
        self.encodage_label = LabelEncoder()
        y_train_enc = self.encodage_label.fit_transform(y_train)
        y_val_enc = self.encodage_label.transform(y_val)
        y_test_enc = self.encodage_label.transform(y_test)

        # Normalisation des caractéristiques (moyenne=0, écart-type=1)
        self.standardisation = StandardScaler()
        X_train_norm = self.standardisation.fit_transform(X_train)
        X_val_norm = self.standardisation.transform(X_val)
        X_test_norm = self.standardisation.transform(X_test)

        logging.info(
            "Normalisation terminée. %d caractéristiques normalisées.",
            X_train_norm.shape[1],
        )

        return X_train_norm, X_val_norm, X_test_norm, y_train_enc, y_val_enc, y_test_enc

    def preparer_donnees(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Pipeline complet : charge, nettoie, sépare, encode et normalise.

        Returns:
            Tuple (X_train, X_val, X_test, y_train, y_val, y_test).
            Tous les tableaux sont normalisés/encodés.
        """
        self.charger_csv()
        self.nettoyer()
        X_train, X_val, X_test, y_train, y_val, y_test = self.separer_jeux()
        X_train, X_val, X_test, y_train, y_val, y_test = self.encoder_et_normaliser(
            X_train, X_val, X_test, y_train, y_val, y_test
        )
        return X_train, X_val, X_test, y_train, y_val, y_test
