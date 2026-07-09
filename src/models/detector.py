"""
Modèle de détection d'intrusions réseau.

Ce module implémente le cœur du système NetGuard AI :
un classifieur capable de distinguer le trafic normal des attaques.

Modèles disponibles :
- Random Forest (par défaut, bon compromis performance/interprétabilité)
- SVM (précis mais plus lent)
- KNN (simple et efficace)
- Gradient Boosting (puissant mais peut surapprendre)
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from src.config import (
    TYPE_MODELE,
    PARAMS_RANDOM_FOREST,
    PARAMS_SVM,
    PARAMS_KNN,
    PARAMS_GRADIENT_BOOSTING,
    SEED_ALEATOIRE,
)
from src.utils.helpers import sauvegarder_modele, charger_modele


class DetecteurIntrusions:
    """
    Classifieur supervisé pour la détection d'intrusions réseau.
    """

    def __init__(self, type_modele: Optional[str] = None):
        """
        Args:
            type_modele: Type de modèle ('random_forest', 'svm', 'knn',
                         'gradient_boosting'). Par défaut depuis config.
        """
        self.type_modele = type_modele or TYPE_MODELE
        self.modele: Any = None
        self.modele_entraine: bool = False
        self._initialiser_modele()

    def _initialiser_modele(self) -> None:
        """Crée l'instance du modèle selon le type choisi."""
        if self.type_modele == "random_forest":
            self.modele = RandomForestClassifier(**PARAMS_RANDOM_FOREST)
        elif self.type_modele == "svm":
            self.modele = SVC(**PARAMS_SVM)
        elif self.type_modele == "knn":
            self.modele = KNeighborsClassifier(**PARAMS_KNN)
        elif self.type_modele == "gradient_boosting":
            self.modele = GradientBoostingClassifier(**PARAMS_GRADIENT_BOOSTING)
        else:
            logging.warning(
                "Type de modèle '%s' inconnu. Utilisation du Random Forest.",
                self.type_modele,
            )
            self.type_modele = "random_forest"
            self.modele = RandomForestClassifier(**PARAMS_RANDOM_FOREST)

        logging.info("Modèle initialisé : %s", self.type_modele)

    def entrainer(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """
        Entraîne le modèle sur les données d'entraînement.

        Args:
            X_train: Caractéristiques d'entraînement.
            y_train: Cibles d'entraînement.
            X_val: Caractéristiques de validation (optionnel).
            y_val: Cibles de validation (optionnel).

        Returns:
            Dictionnaire avec la performance sur l'entraînement
            (et la validation si fournie).
        """
        logging.info("Entraînement du modèle %s...", self.type_modele)
        self.modele.fit(X_train, y_train)
        self.modele_entraine = True

        # Performance sur l'entraînement
        score_train = self.modele.score(X_train, y_train)
        resultats = {"score_entrainement": score_train}

        # Performance sur la validation (si fournie)
        if X_val is not None and y_val is not None:
            score_val = self.modele.score(X_val, y_val)
            resultats["score_validation"] = score_val
            logging.info(
                "Score - Entraînement: %.4f, Validation: %.4f",
                score_train,
                score_val,
            )
        else:
            logging.info("Score d'entraînement: %.4f", score_train)

        return resultats

    def predire(self, X: np.ndarray) -> np.ndarray:
        """
        Prédit la classe pour de nouveaux échantillons.

        Args:
            X: Caractéristiques à classifier.

        Returns:
            Prédictions (0 = normal, 1 = attaque).
        """
        if not self.modele_entraine:
            raise ValueError("Le modèle n'est pas encore entraîné.")
        return self.modele.predict(X)

    def predire_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Retourne les probabilités d'appartenance à chaque classe.

        Args:
            X: Caractéristiques à classifier.

        Returns:
            Probabilités pour chaque classe.
        """
        if not self.modele_entraine:
            raise ValueError("Le modèle n'est pas encore entraîné.")

        # Vérifier si le modèle supporte predict_proba
        if hasattr(self.modele, "predict_proba"):
            return self.modele.predict_proba(X)
        else:
            logging.warning(
                "Le modèle %s ne supporte pas predict_proba. "
                "Retour de prédictions binaires.",
                self.type_modele,
            )
            # Fallback : retourner des probabilités simulées
            predictions = self.predire(X)
            probabilites = np.zeros((len(X), 2))
            probabilites[np.arange(len(X)), predictions] = 1.0
            return probabilites

    def obtenir_parametres(self) -> Dict[str, Any]:
        """Retourne les paramètres du modèle."""
        if self.modele is None:
            return {}
        return self.modele.get_params()

    def evaluer(self, X_test: np.ndarray, y_test: np.ndarray) -> float:
        """
        Évalue la précision du modèle sur un ensemble de test.

        Args:
            X_test: Caractéristiques de test.
            y_test: Cibles de test.

        Returns:
            Précision (accuracy) du modèle.
        """
        if not self.modele_entraine:
            raise ValueError("Le modèle n'est pas encore entraîné.")
        return self.modele.score(X_test, y_test)

    def sauvegarder(self, chemin: Path) -> None:
        """Sauvegarde le modèle entraîné."""
        sauvegarder_modele(self.modele, chemin)

    def charger(self, chemin: Path) -> None:
        """Charge un modèle pré-entraîné."""
        self.modele = charger_modele(chemin)
        self.modele_entraine = True
        self.type_modele = type(self.modele).__name__
        logging.info("Modèle chargé : %s", self.type_modele)
