"""
Script CLI pour exécuter le pipeline de prétraitement NetGuard AI.

Utilisation :
    python -m src.preprocessing.run_pipeline
    python -m src.preprocessing.run_pipeline --fichier data/raw/mon_dataset.csv
    python -m src.preprocessing.run_pipeline --sauvegarder
    python -m src.preprocessing.run_pipeline --afficher-resume
"""

import argparse
import logging
import sys
from pathlib import Path

# Ajout de la racine du projet au chemin
PROJET_RACINE = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJET_RACINE))

from src.preprocessing.pipeline import PipelinePretraitement
from src.utils.helpers import configurer_logging, Chronometre


def definir_arguments() -> argparse.ArgumentParser:
    """Configure les arguments de la ligne de commande."""
    parser = argparse.ArgumentParser(
        description="NetGuard AI - Pipeline de prétraitement des données",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples :
  python -m src.preprocessing.run_pipeline
  python -m src.preprocessing.run_pipeline --fichier datasets/cicids2017_consolide.csv
  python -m src.preprocessing.run_pipeline --afficher-resume
        """,
    )

    parser.add_argument(
        "--fichier",
        type=str,
        default=None,
        help="Chemin vers le fichier CSV à prétraiter",
    )

    parser.add_argument(
        "--colonne-label",
        type=str,
        default="label",
        help="Nom de la colonne cible (defaut: label)",
    )

    parser.add_argument(
        "--taille-test",
        type=float,
        default=0.3,
        help="Proportion des données pour le test (defaut: 0.3)",
    )

    parser.add_argument(
        "--sauvegarder",
        action="store_true",
        default=True,
        help="Sauvegarder les données nettoyées (defaut: True)",
    )

    parser.add_argument(
        "--afficher-resume",
        action="store_true",
        help="Afficher un résumé du dernier pipeline exécuté",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Afficher les logs détaillés",
    )

    return parser


def afficher_resume_reussi(
    X_train, X_test, y_train, y_test, pipeline
) -> None:
    """Affiche un résumé formaté du pipeline."""
    print()
    print("=" * 60)
    print("  PIPELINE DE PRETRAITEMENT - RESUME")
    print("=" * 60)
    print(f"  Dataset        : {pipeline.chemin_dataset}")
    print(f"  Echantillons   : {pipeline.metadonnees['nb_lignes_initial']:,}")
    print(f"  Caracteristiques initiales : {pipeline.metadonnees['nb_colonnes_initial']}")
    print(f"  Caracteristiques finales   : {pipeline.metadonnees['nb_caracteristiques_final']}")
    print(f"  Doublons supprimes         : {pipeline.metadonnees['nb_doublons_supprimes']}")
    print(f"  Valeurs manquantes traitees : {pipeline.metadonnees['nb_valeurs_manquantes_supprimees']}")
    print()
    print(f"  Entrainement : {len(X_train):,} echantillons")
    print(f"  Test         : {len(X_test):,} echantillons")
    print()
    print(f"  Donnees sauvegardees dans : data/processed/")
    print("=" * 60)
    print()


def executer_pipeline(args: argparse.Namespace) -> None:
    """Exécute le pipeline avec les arguments fournis."""
    # Création du pipeline
    pipeline = PipelinePretraitement(
        colonne_label=args.colonne_label,
        taille_test=args.taille_test,
    )

    # Détermination du chemin du dataset
    chemin = Path(args.fichier) if args.fichier else None

    # Exécution complète
    with Chronometre("Pipeline de prétraitement") as chrono:
        X_train, X_test, y_train, y_test = (
            pipeline.executer_pipeline_complet(
                chemin_dataset=chemin,
                sauvegarder=args.sauvegarder,
            )
        )

    afficher_resume_reussi(
        X_train, X_test, y_train, y_test, pipeline
    )


def main() -> None:
    """Point d'entrée principal."""
    parser = definir_arguments()
    args = parser.parse_args()

    # Configuration du logging
    niveau = logging.DEBUG if args.verbose else logging.INFO
    configurer_logging(niveau=niveau)

    if args.afficher_resume:
        # Vérifier si un résumé existe
        chemin_resume = (
            PROJET_RACINE / "data" / "processed" / "netguard_clean_resume.txt"
        )
        if chemin_resume.exists():
            print(chemin_resume.read_text(encoding="utf-8"))
        else:
            logging.error(
                "Aucun résumé trouvé. Exécutez d'abord le pipeline."
            )
            sys.exit(1)
    else:
        executer_pipeline(args)


if __name__ == "__main__":
    main()
