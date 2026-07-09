"""
Évaluation des performances du modèle de détection.

Ce module fournit :
- Le calcul des métriques (accuracy, précision, rappel, F1)
- La matrice de confusion
- Le rapport de classification complet
- La visualisation des résultats
"""

import logging
from typing import Dict, Tuple

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


class Evaluateur:
    """
    Évalue les performances du modèle de détection d'intrusions.
    """

    def __init__(self):
        self.metriques: Dict[str, float] = {}

    def calculer_metriques(
        self, y_reel: np.ndarray, y_pred: np.ndarray
    ) -> Dict[str, float]:
        """
        Calcule toutes les métriques de classification.

        Args:
            y_reel: Valeurs réelles.
            y_pred: Valeurs prédites.

        Returns:
            Dictionnaire contenant les métriques calculées.
        """
        self.metriques = {
            "accuracy": accuracy_score(y_reel, y_pred),
            "precision": precision_score(y_reel, y_pred, average="binary", zero_division=0),
            "recall": recall_score(y_reel, y_pred, average="binary", zero_division=0),
            "f1_score": f1_score(y_reel, y_pred, average="binary", zero_division=0),
        }

        logging.info(
            "Métriques - Accuracy: %.4f, Précision: %.4f, "
            "Rappel: %.4f, F1-Score: %.4f",
            self.metriques["accuracy"],
            self.metriques["precision"],
            self.metriques["recall"],
            self.metriques["f1_score"],
        )

        return self.metriques

    def calculer_matrice_confusion(
        self, y_reel: np.ndarray, y_pred: np.ndarray
    ) -> np.ndarray:
        """
        Calcule la matrice de confusion.

        Args:
            y_reel: Valeurs réelles.
            y_pred: Valeurs prédites.

        Returns:
            Matrice de confusion 2x2.
        """
        matrice = confusion_matrix(y_reel, y_pred)
        logging.info("Matrice de confusion :\n%s", matrice)
        return matrice

    def generer_rapport(self, y_reel: np.ndarray, y_pred: np.ndarray) -> str:
        """
        Génère un rapport de classification complet.

        Args:
            y_reel: Valeurs réelles.
            y_pred: Valeurs prédites.

        Returns:
            Rapport textuel formaté.
        """
        rapport = classification_report(
            y_reel,
            y_pred,
            target_names=["Normal (0)", "Attaque (1)"],
            zero_division=0,
        )
        return rapport

    def afficher_matrice_confusion(
        self,
        matrice: np.ndarray,
        titre: str = "Matrice de confusion - NetGuard AI",
    ) -> None:
        """
        Affiche la matrice de confusion avec seaborn.

        Args:
            matrice: Matrice de confusion 2x2.
            titre: Titre du graphique.
        """
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            matrice,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Normal", "Attaque"],
            yticklabels=["Normal", "Attaque"],
        )
        plt.xlabel("Prédictions")
        plt.ylabel("Valeurs réelles")
        plt.title(titre)
        plt.tight_layout()
        plt.show()

    def evaluer_complet(
        self, y_reel: np.ndarray, y_pred: np.ndarray
    ) -> Tuple[Dict[str, float], np.ndarray, str]:
        """
        Évaluation complète : métriques + matrice + rapport.

        Args:
            y_reel: Valeurs réelles.
            y_pred: Valeurs prédites.

        Returns:
            Tuple (metriques, matrice_confusion, rapport_texte).
        """
        metriques = self.calculer_metriques(y_reel, y_pred)
        matrice = self.calculer_matrice_confusion(y_reel, y_pred)
        rapport = self.generer_rapport(y_reel, y_pred)

        logging.info("Rapport de classification :\n%s", rapport)

        return metriques, matrice, rapport

    def calculer_taux_detection(self, matrice: np.ndarray) -> Dict[str, float]:
        """
        Calcule les taux de détection spécifiques à la cybersécurité.

        Args:
            matrice: Matrice de confusion 2x2.

        Returns:
            Dictionnaire avec les taux calculés.
        """
        vn, fp, fn, vp = matrice.ravel()

        taux_detection = {
            "taux_vrais_positifs": vp / (vp + fn) if (vp + fn) > 0 else 0.0,
            "taux_faux_positifs": fp / (fp + vn) if (fp + vn) > 0 else 0.0,
            "taux_vrais_negatifs": vn / (vn + fp) if (vn + fp) > 0 else 0.0,
            "taux_faux_negatifs": fn / (fn + vp) if (fn + vp) > 0 else 0.0,
        }

        return taux_detection
