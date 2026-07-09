"""
Fonctions utilitaires pour NetGuard AI.

Fournit des helpers génériques utilisés dans tout le projet :
- Gestion des chemins
- Journalisation (logging)
- Chronométrage
- Sauvegarde et chargement de modèles
"""

import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional

import joblib


# ─── Configuration du logging ────────────────────────────────────────────────

def configurer_logging(
    niveau: int = logging.INFO,
    fichier_log: Optional[Path] = None,
) -> None:
    """
    Configure le système de logging.

    Args:
        niveau: Niveau de logging (defaut: INFO).
        fichier_log: Chemin optionnel vers un fichier de log.
    """
    handlers = [logging.StreamHandler()]
    if fichier_log:
        handlers.append(logging.FileHandler(fichier_log))

    logging.basicConfig(
        level=niveau,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers,
    )


# ─── Chronomètre simple ──────────────────────────────────────────────────────

class Chronometre:
    """Chronomètre simple pour mesurer le temps d'exécution."""

    def __init__(self, nom: str = "Execution"):
        """
        Args:
            nom: Nom de l'opération chronométrée.
        """
        self.nom = nom
        self.debut: float = 0.0

    def __enter__(self):
        self.debut = time.time()
        return self

    def __exit__(self, *args):
        duree = time.time() - self.debut
        logging.info("%s terminé en %.2f secondes", self.nom, duree)


# ─── Sauvegarde et chargement ───────────────────────────────────────────────

def sauvegarder_modele(modele: Any, chemin: Path) -> None:
    """
    Sauvegarde un modèle entraîné sur le disque.

    Args:
        modele: Modèle à sauvegarder.
        chemin: Chemin de destination.
    """
    chemin.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(modele, chemin)
    logging.info("Modèle sauvegardé dans %s", chemin)


def charger_modele(chemin: Path) -> Any:
    """
    Charge un modèle depuis le disque.

    Args:
        chemin: Chemin du modèle à charger.

    Returns:
        Le modèle chargé.
    """
    if not chemin.exists():
        raise FileNotFoundError(f"Modèle introuvable : {chemin}")
    modele = joblib.load(chemin)
    logging.info("Modèle chargé depuis %s", chemin)
    return modele


def sauvegarder_metriques(metriques: Dict[str, Any], chemin: Path) -> None:
    """
    Sauvegarde les métriques d'évaluation dans un fichier texte.

    Args:
        metriques: Dictionnaire des métriques.
        chemin: Chemin de sortie.
    """
    chemin.parent.mkdir(parents=True, exist_ok=True)
    with open(chemin, "w", encoding="utf-8") as f:
        f.write("=== Métriques d'évaluation NetGuard AI ===\n\n")
        for nom, valeur in metriques.items():
            if isinstance(valeur, float):
                f.write(f"{nom} : {valeur:.4f}\n")
            else:
                f.write(f"{nom} : {valeur}\n")
    logging.info("Métriques sauvegardées dans %s", chemin)
