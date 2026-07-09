"""
Module de détection d'intrusions réseau.

Ce module implémente 6 modèles de classification supervisée
pour distinguer le trafic réseau normal des attaques.

Modèles disponibles :
- decision_tree      : Arbre de décision (rapide, interprétable)
- random_forest      : Forêt aléatoire (robuste, fiable) [DÉFAUT]
- logistic_regression: Régression logistique (simple, rapide)
- svm                : SVM (performant en haute dimension)
- knn                : KNN (basé sur la proximité)
- naive_bayes        : Naive Bayes (très rapide, probabiliste)
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

from src.config import (
    TYPE_MODELE,
    LISTE_MODELES,
    DESCRIPTION_MODELES,
    PARAMS_DECISION_TREE,
    PARAMS_RANDOM_FOREST,
    PARAMS_LOGISTIC_REGRESSION,
    PARAMS_SVM,
    PARAMS_KNN,
    PARAMS_NAIVE_BAYES,
)
from src.utils.helpers import sauvegarder_modele, charger_modele


# ─── Dictionnaire associant chaque modèle à sa classe et ses paramètres ─────

# Structure : { nom_modele: (classe sklearn, dict_parametres) }
FABRIQUE_MODELES = {
    "decision_tree": (
        DecisionTreeClassifier,
        PARAMS_DECISION_TREE,
    ),
    "random_forest": (
        RandomForestClassifier,
        PARAMS_RANDOM_FOREST,
    ),
    "logistic_regression": (
        LogisticRegression,
        PARAMS_LOGISTIC_REGRESSION,
    ),
    "svm": (
        SVC,
        PARAMS_SVM,
    ),
    "knn": (
        KNeighborsClassifier,
        PARAMS_KNN,
    ),
    "naive_bayes": (
        GaussianNB,
        PARAMS_NAIVE_BAYES,
    ),
}

# Modèles qui supportent predict_proba (pour les probabilités)
MODELES_AVEC_PROBA = {"decision_tree", "random_forest",
                       "logistic_regression", "knn", "naive_bayes"}


def creer_modele(type_modele: str, **params_supplementaires) -> Any:
    """
    Fabrique un modèle scikit-learn à partir de son nom.

    Utilise le dictionnaire FABRIQUE_MODELES pour instancier
    le modèle avec ses paramètres par défaut, fusionnés avec
    d'éventuels paramètres supplémentaires.

    Args:
        type_modele: Nom du modèle ('decision_tree', 'random_forest', ...).
        **params_supplementaires: Paramètres supplémentaires à fusionner.

    Returns:
        Instance du modèle scikit-learn.

    Lève:
        ValueError: Si le type de modèle est inconnu.
    """
    if type_modele not in FABRIQUE_MODELES:
        raise ValueError(
            f"Modèle inconnu : '{type_modele}'.\n"
            f"Modèles disponibles : {list(FABRIQUE_MODELES.keys())}"
        )

    classe_modele, params_default = FABRIQUE_MODELES[type_modele]

    # Fusion des paramètres par défaut avec les suppléments
    params_finaux = {**params_default, **params_supplementaires}

    logging.debug("Création du modèle %s avec params : %s",
                   type_modele, params_finaux)

    return classe_modele(**params_finaux)


# ─── Classe principale de détection ─────────────────────────────────────────

class DetecteurIntrusions:
    """
    Classifieur supervisé pour la détection d'intrusions réseau.

    Supporte 6 modèles de Machine Learning. Tous partagent
    la même interface : entrainer(), predire(), evaluer().

    Utilisation :
        detecteur = DetecteurIntrusions(type_modele="random_forest")
        detecteur.entrainer(X_train, y_train)
        predictions = detecteur.predire(X_test)
    """

    def __init__(self, type_modele: Optional[str] = None):
        """
        Initialise le détecteur avec le modèle choisi.

        Args:
            type_modele: Nom du modèle. Par défaut : random_forest.
        """
        self.type_modele = type_modele or TYPE_MODELE
        self.modele: Any = None
        self.modele_entraine: bool = False
        self._initialiser_modele()

    def _initialiser_modele(self) -> None:
        """Crée l'instance du modèle via la fabrique."""
        try:
            self.modele = creer_modele(self.type_modele)
            logging.info("Modèle initialisé : %s - %s",
                          self.type_modele,
                          DESCRIPTION_MODELES.get(
                              self.type_modele, ""
                          ))
        except ValueError as e:
            logging.error(
                "Type de modèle '%s' inconnu. "
                "Utilisation du modèle par défaut '%s'.",
                self.type_modele, TYPE_MODELE,
            )
            self.type_modele = TYPE_MODELE
            self.modele = creer_modele(self.type_modele)
            logging.info("Modèle de remplacement : %s", self.type_modele)

    # ─── Entraînement ───────────────────────────────────────────────────────

    def entrainer(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """
        Entraîne le modèle sur les données fournies.

        Args:
            X_train: Caractéristiques d'entraînement.
            y_train: Cibles d'entraînement.
            X_val: Caractéristiques de validation (optionnel).
            y_val: Cibles de validation (optionnel).

        Returns:
            Dictionnaire avec les scores d'entraînement
            et de validation (si fournie).
        """
        logging.info(
            "Entraînement : %s sur %d échantillons...",
            self.type_modele, len(X_train),
        )
        self.modele.fit(X_train, y_train)
        self.modele_entraine = True

        # Score sur l'entraînement
        score_train = self.modele.score(X_train, y_train)
        resultats = {"score_entrainement": score_train}

        # Score sur la validation (optionnel)
        if X_val is not None and y_val is not None:
            score_val = self.modele.score(X_val, y_val)
            resultats["score_validation"] = score_val
            logging.info(
                "  Entraînement: %.4f | Validation: %.4f",
                score_train, score_val,
            )
        else:
            logging.info("  Score entraînement: %.4f", score_train)

        return resultats

    # ─── Prédiction ─────────────────────────────────────────────────────────

    def predire(self, X: np.ndarray) -> np.ndarray:
        """
        Prédit la classe de nouveaux échantillons.

        Args:
            X: Caractéristiques à classifier.

        Returns:
            Prédictions (0 = trafic normal, 1 = attaque).
        """
        if not self.modele_entraine:
            raise ValueError(
                "Le modèle '%s' n'est pas encore entraîné. "
                "Appelez entrainer() d'abord." % self.type_modele
            )
        return self.modele.predict(X)

    def predire_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Retourne les probabilités pour chaque classe.

        Args:
            X: Caractéristiques à classifier.

        Returns:
            Probabilités (classe 0, classe 1).
        """
        if not self.modele_entraine:
            raise ValueError("Le modèle n'est pas encore entraîné.")

        if self.type_modele in MODELES_AVEC_PROBA:
            return self.modele.predict_proba(X)
        else:
            # Fallback pour les modèles sans predict_proba (ex: SVM)
            logging.warning(
                "%s ne supporte pas predict_proba. "
                "Retour de prédictions binaires.",
                self.type_modele,
            )
            predictions = self.predire(X)
            probabilites = np.zeros((len(X), 2))
            probabilites[np.arange(len(X)), predictions] = 1.0
            return probabilites

    # ─── Évaluation ─────────────────────────────────────────────────────────

    def evaluer(self, X_test: np.ndarray, y_test: np.ndarray) -> float:
        """
        Évalue la précision (accuracy) sur un ensemble de test.

        Args:
            X_test: Caractéristiques de test.
            y_test: Cibles de test.

        Returns:
            Accuracy (entre 0 et 1).
        """
        return self.modele.score(X_test, y_test)

    # ─── Paramètres ─────────────────────────────────────────────────────────

    def obtenir_parametres(self) -> Dict[str, Any]:
        """
        Retourne les hyperparamètres du modèle.

        Note : Certains modèles (RandomForest sur Windows) peuvent
        lever une AttributeError si get_params() est appelé avant
        l'entraînement. La méthode capture cette exception et
        retourne les paramètres disponibles.
        """
        if self.modele is None:
            return {}
        try:
            return self.modele.get_params()
        except AttributeError:
            # Fallback : retourne les infos de base
            return {
                "type_modele": self.type_modele,
                "description": self.obtenir_description(),
            }

    def obtenir_description(self) -> str:
        """Retourne une description lisible du modèle."""
        return DESCRIPTION_MODELES.get(self.type_modele, self.type_modele)

    # ─── Sauvegarde et chargement ───────────────────────────────────────────

    def sauvegarder(self, chemin: Path) -> None:
        """Sauvegarde le modèle entraîné sur le disque."""
        sauvegarder_modele(self.modele, chemin)

    def charger(self, chemin: Path) -> None:
        """Charge un modèle pré-entraîné depuis le disque."""
        self.modele = charger_modele(chemin)
        self.modele_entraine = True
        # Déterminer le type à partir de la classe chargée
        for nom, (classe, _) in FABRIQUE_MODELES.items():
            if isinstance(self.modele, classe):
                self.type_modele = nom
                break
        else:
            self.type_modele = type(self.modele).__name__
        logging.info("Modèle chargé : %s", self.type_modele)
