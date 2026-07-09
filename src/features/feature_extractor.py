"""
Extraction et sélection des caractéristiques réseau.

Ce module permet :
- La sélection des caractéristiques les plus importantes
- La réduction de dimensionnalité (PCA)
- L'extraction de caractéristiques statistiques
"""

import logging
from typing import Tuple

import numpy as np
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif

from src.config import SEED_ALEATOIRE


class ExtracteurCaracteristiques:
    """
    Extrait et sélectionne les caractéristiques pertinentes
    pour la détection d'intrusions.
    """

    def __init__(self, methode: str = "k_best", n_caracteristiques: int = 20):
        """
        Args:
            methode: Méthode de sélection ('k_best', 'pca', 'aucun').
            n_caracteristiques: Nombre de caractéristiques à garder.
        """
        self.methode = methode
        self.n_caracteristiques = n_caracteristiques
        self.selecteur = None
        self.indices_selectionnes: np.ndarray = None

    def selectionner_k_best(
        self, X_train: np.ndarray, y_train: np.ndarray
    ) -> np.ndarray:
        """
        Sélectionne les K meilleures caractéristiques avec ANOVA F-test.

        Args:
            X_train: Caractéristiques d'entraînement.
            y_train: Cible d'entraînement.

        Returns:
            Caractéristiques sélectionnées.
        """
        n_caracs = min(self.n_caracteristiques, X_train.shape[1])
        self.selecteur = SelectKBest(score_func=f_classif, k=n_caracs)
        X_selected = self.selecteur.fit_transform(X_train, y_train)
        self.indices_selectionnes = self.selecteur.get_support(indices=True)

        logging.info(
            "Sélection des %d meilleures caractéristiques (méthode: ANOVA F-test)",
            n_caracs,
        )
        return X_selected

    def appliquer_pca(
        self, X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Réduit la dimensionnalité avec PCA.

        Args:
            X_train: Caractéristiques d'entraînement.
            X_val: Caractéristiques de validation.
            X_test: Caractéristiques de test.

        Returns:
            Tuple (X_train_pca, X_val_pca, X_test_pca).
        """
        n_composantes = min(self.n_caracteristiques, X_train.shape[1])
        pca = PCA(
            n_components=n_composantes,
            random_state=SEED_ALEATOIRE,
        )
        X_train_pca = pca.fit_transform(X_train)
        X_val_pca = pca.transform(X_val)
        X_test_pca = pca.transform(X_test)

        variance_expliquee = sum(pca.explained_variance_ratio_) * 100
        logging.info(
            "PCA : %d composantes, %.1f%% de variance expliquée",
            n_composantes,
            variance_expliquee,
        )

        return X_train_pca, X_val_pca, X_test_pca

    def transformer(
        self, X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray, y_train: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Applique la méthode de sélection/config des caractéristiques.

        Args:
            X_train: Caractéristiques d'entraînement.
            X_val: Caractéristiques de validation.
            X_test: Caractéristiques de test.
            y_train: Cible d'entraînement.

        Returns:
            Tuple (X_train_t, X_val_t, X_test_t).
        """
        if self.methode == "k_best":
            X_train_t = self.selectionner_k_best(X_train, y_train)
            # Appliquer la même sélection à X_val et X_test
            X_val_t = X_val[:, self.indices_selectionnes]
            X_test_t = X_test[:, self.indices_selectionnes]

        elif self.methode == "pca":
            X_train_t, X_val_t, X_test_t = self.appliquer_pca(X_train, X_val, X_test)

        else:  # 'aucun' : pas de sélection
            logging.info("Aucune sélection de caractéristiques appliquée.")
            X_train_t, X_val_t, X_test_t = X_train, X_val, X_test

        return X_train_t, X_val_t, X_test_t
