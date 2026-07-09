"""
Point d'entrée principal de NetGuard AI.

Utilisation :
    python main.py              # Entraînement et évaluation complète
    python main.py --predict    # Mode prédiction uniquement
    python main.py --help       # Aide détaillée

Ce script orchestre le pipeline complet :
1. Chargement et prétraitement des données
2. Extraction des caractéristiques
3. Entraînement du modèle
4. Évaluation des performances
5. Sauvegarde du modèle entraîné
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
from joblib import dump, load

from src.config import (
    TYPE_MODELE,
    DOSSIER_DATA_TRANSFORME,
    SEED_ALEATOIRE,
)
from src.preprocessing.data_loader import ChargeurDonnees
from src.features.feature_extractor import ExtracteurCaracteristiques
from src.models.detector import DetecteurIntrusions
from src.evaluation.metrics import Evaluateur
from src.utils.helpers import (
    configurer_logging,
    Chronometre,
    sauvegarder_metriques,
)


def definir_arguments() -> argparse.ArgumentParser:
    """Configure les arguments en ligne de commande."""
    parser = argparse.ArgumentParser(
        description="NetGuard AI - Détection d'intrusions réseau par Machine Learning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples :
  python main.py                          # Pipeline complet
  python main.py --modele svm            # Utiliser SVM
  python main.py --aucune-selection      # Sans sélection de caractéristiques
  python main.py --predict --fichier donnees.csv  # Prédire sur de nouvelles données
        """,
    )

    parser.add_argument(
        "--modele",
        type=str,
        default=TYPE_MODELE,
        choices=["random_forest", "svm", "knn", "gradient_boosting"],
        help="Type de modèle à utiliser (defaut: random_forest)",
    )

    parser.add_argument(
        "--selection-caracs",
        type=str,
        default="k_best",
        choices=["k_best", "pca", "aucun"],
        help="Méthode de sélection des caractéristiques (defaut: k_best)",
    )

    parser.add_argument(
        "--n-caracteristiques",
        type=int,
        default=20,
        help="Nombre de caractéristiques à garder (defaut: 20)",
    )

    parser.add_argument(
        "--predict",
        action="store_true",
        help="Mode prédiction : charger le modèle et prédire sur un fichier",
    )

    parser.add_argument(
        "--fichier",
        type=str,
        default="",
        help="Fichier CSV à analyser (mode predict)",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Afficher les logs détaillés (DEBUG)",
    )

    return parser


def executer_pipeline_entrainement(args: argparse.Namespace) -> None:
    """
    Exécute le pipeline complet : données → modèle → évaluation.

    Args:
        args: Arguments de la ligne de commande.
    """
    logging.info("=" * 50)
    logging.info("NETGUARD AI - Pipeline de détection d'intrusions")
    logging.info("=" * 50)
    logging.info("Modèle : %s", args.modele)
    logging.info("Sélection des caractéristiques : %s", args.selection_caracs)
    logging.info("Seed aléatoire : %d", SEED_ALEATOIRE)

    # ─── Étape 1 : Chargement et prétraitement ───────────────────────────────

    with Chronometre("Chargement et prétraitement"):
        chargeur = ChargeurDonnees()
        X_train, X_val, X_test, y_train, y_val, y_test = (
            chargeur.preparer_donnees()
        )

    # ─── Étape 2 : Sélection des caractéristiques ────────────────────────────

    with Chronometre("Extraction des caractéristiques"):
        extracteur = ExtracteurCaracteristiques(
            methode=args.selection_caracs,
            n_caracteristiques=args.n_caracteristiques,
        )
        X_train, X_val, X_test = extracteur.transformer(
            X_train, X_val, X_test, y_train
        )

    # ─── Étape 3 : Entraînement du modèle ────────────────────────────────────

    with Chronometre("Entraînement du modèle"):
        detecteur = DetecteurIntrusions(type_modele=args.modele)
        resultats_entrainement = detecteur.entrainer(
            X_train, y_train, X_val, y_val
        )

    # ─── Étape 4 : Évaluation ────────────────────────────────────────────────

    with Chronometre("Évaluation du modèle"):
        y_pred = detecteur.predire(X_test)
        evaluateur = Evaluateur()
        metriques, matrice, rapport = evaluateur.evaluer_complet(y_test, y_pred)

        # Calcul des taux spécifiques à la cybersécurité
        taux = evaluateur.calculer_taux_detection(matrice)
    # ─── Étape 5 : Sauvegarde ────────────────────────────────────────────────

    # Sauvegarde du modèle
    chemin_modele = DOSSIER_DATA_TRANSFORME / "modele_entraine.joblib"
    detecteur.sauvegarder(chemin_modele)

    # Sauvegarde du StandardScaler (nécessaire pour le mode predict)
    chemin_scaler = DOSSIER_DATA_TRANSFORME / "standard_scaler.joblib"
    if chargeur.standardisation is not None:
        dump(chargeur.standardisation, chemin_scaler)
        logging.info("StandardScaler sauvegardé : %s", chemin_scaler)

    # Sauvegarde des métriques
    chemin_metriques = DOSSIER_DATA_TRANSFORME / "metriques.txt"
    toutes_metriques = {**metriques, **taux}
    sauvegarder_metriques(toutes_metriques, chemin_metriques)

    # ─── Résumé final ────────────────────────────────────────────────────────

    logging.info("=" * 50)
    logging.info("RÉSULTATS FINAUX")
    logging.info("=" * 50)
    logging.info("Accuracy  : %.4f", metriques["accuracy"])
    logging.info("Précision : %.4f", metriques["precision"])
    logging.info("Rappel    : %.4f", metriques["recall"])
    logging.info("F1-Score  : %.4f", metriques["f1_score"])
    logging.info("-" * 50)
    logging.info("Taux de vrais positifs  : %.4f", taux["taux_vrais_positifs"])
    logging.info("Taux de faux positifs   : %.4f", taux["taux_faux_positifs"])
    logging.info("Modèle sauvegardé : %s", chemin_modele)
    logging.info("=" * 50)


def executer_prediction(args: argparse.Namespace) -> None:
    """
    Utilise un modèle entraîné pour prédire sur de nouvelles données.

    Args:
        args: Arguments de la ligne de commande.
    """
    # Chargement du modèle
    chemin_modele = DOSSIER_DATA_TRANSFORME / "modele_entraine.joblib"
    if not chemin_modele.exists():
        logging.error(
            "Aucun modèle entraîné trouvé. Exécutez d'abord 'python main.py' "
            "sans l'option --predict."
        )
        sys.exit(1)

    detecteur = DetecteurIntrusions()
    detecteur.charger(chemin_modele)

    # Chargement des données à prédire
    chemin_fichier = Path(args.fichier)
    if not chemin_fichier.exists():
        logging.error("Fichier introuvable : %s", chemin_fichier)
        sys.exit(1)

    logging.info("Prédiction sur le fichier : %s", chemin_fichier)

    donnees = pd.read_csv(chemin_fichier)

    # Chargement du StandardScaler pour normaliser les données
    chemin_scaler = DOSSIER_DATA_TRANSFORME / "standard_scaler.joblib"
    if chemin_scaler.exists():
        scaler = load(chemin_scaler)
        donnees_normalisees = scaler.transform(donnees.values)
    else:
        logging.warning(
            "StandardScaler non trouvé. Utilisation des données brutes."
        )
        donnees_normalisees = donnees.values

    # Prédiction
    predictions = detecteur.predire(donnees_normalisees)
    logging.info("Prédictions terminées pour %d échantillons.", len(predictions))

    # Affichage des résultats
    nb_attaques = int(sum(predictions))
    nb_normal = len(predictions) - nb_attaques
    logging.info(
        "Résultats : %d normaux, %d attaques détectées",
        nb_normal,
        nb_attaques,
    )


def main() -> None:
    """Point d'entrée principal."""
    parser = definir_arguments()
    args = parser.parse_args()

    # Configuration du logging
    niveau_log = logging.DEBUG if args.verbose else logging.INFO
    configurer_logging(niveau=niveau_log)

    # Exécution
    if args.predict:
        executer_prediction(args)
    else:
        executer_pipeline_entrainement(args)


if __name__ == "__main__":
    main()
