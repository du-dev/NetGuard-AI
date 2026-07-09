"""
Script de generation de graphiques pour le memoire NetGuard AI.

Genere tous les graphiques necessaires au memoire dans le dossier data/processed/graphiques/ :
1. Distribution des classes du dataset
2. Boxplots des caracteristiques par classe
3. Matrice de correlation
4. Courbes ROC avec AUC (6 modeles)
5. Matrice de confusion du meilleur modele
6. Comparaison des metriques (barres groupees)
7. Importance des caracteristiques
8. Temps d'entrainement par modele
9. Taux de detection vs faux positifs
10. Tableau comparatif final

Utilisation :
    python generate_graphiques.py
    python generate_graphiques.py --dataset cicids2017
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

PROJET_RACINE = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJET_RACINE))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Mode non-interactif (pas besoin d'affichage)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc

from src.config import (
    LISTE_MODELES,
    DESCRIPTION_MODELES,
    DATASETS_DISPONIBLES,
    DATASET_PAR_DEFAUT,
    DOSSIER_DATA_TRANSFORME,
)
from src.preprocessing.pipeline import PipelinePretraitement
from src.models.detector import DetecteurIntrusions
from src.evaluation.metrics import Evaluateur
from src.utils.helpers import configurer_logging, Chronometre

# --- Configuration ---------------------------------------------------------
DOSSIER_GRAPH = DOSSIER_DATA_TRANSFORME / "graphiques"
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
})


def preparer_donnees(nom_dataset: str, download: bool = False):
    """Charge et pretratie les donnees."""
    config_dataset = DATASETS_DISPONIBLES[nom_dataset]

    if download and nom_dataset == "cicids2017":
        if not config_dataset["chemin"].exists():
            from datasets.download_cicids2017 import executer_pipeline_complet
            executer_pipeline_complet()

    pipeline = PipelinePretraitement(
        colonne_label=config_dataset["colonne_label"],
        colonnes_a_ignorer=config_dataset["colonnes_a_ignorer"],
        taille_test=0.3,
    )
    X_train, X_test, y_train, y_test = pipeline.executer_pipeline_complet(
        chemin_dataset=config_dataset["chemin"], sauvegarder=False,
    )
    return X_train, X_test, y_train, y_test, pipeline


def entrainer_tous_les_modeles(X_train, y_train, X_test, y_test):
    """Entraine les 6 modeles et retourne les resultats."""
    resultats = []
    for nom in LISTE_MODELES:
        det = DetecteurIntrusions(type_modele=nom)
        import time
        t0 = time.time()
        det.entrainer(X_train, y_train)
        duree = time.time() - t0
        y_pred = det.predire(X_test)
        y_proba = None
        if hasattr(det.modele, "predict_proba"):
            try:
                y_proba = det.modele.predict_proba(X_test)[:, 1]
            except Exception:
                pass
        ev = Evaluateur()
        met = ev.calculer_metriques(y_test, y_pred)
        matrice = ev.calculer_matrice_confusion(y_test, y_pred)
        resultats.append({
            "nom": nom,
            "desc": DESCRIPTION_MODELES[nom],
            "modele": det.modele,
            "y_pred": y_pred,
            "y_proba": y_proba,
            "metriques": met,
            "matrice": matrice,
            "duree": duree,
        })
    return resultats


# ===========================================================================
#  1. Distribution des classes
# ===========================================================================
def graphique_distribution_classes(y_train, y_test, chemin):
    """Graphique 1 : Repartition normal vs attaque dans train et test."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    for ax, y, titre in [(ax1, y_train, "Entrainement"), (ax2, y_test, "Test")]:
        labels = ["Normal (0)", "Attaque (1)"]
        counts = [np.sum(y == 0), np.sum(y == 1)]
        couleurs = ["#2ecc71", "#e74c3c"]
        bars = ax.bar(labels, counts, color=couleurs, edgecolor="white", linewidth=1.5)
        for bar, v in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts)*0.02,
                    str(v), ha="center", fontsize=12, fontweight="bold")
        ax.set_title("Distribution - " + titre, fontweight="bold")
        ax.set_ylabel("Nombre d'echantillons")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.suptitle("Distribution des classes (Normal vs Attaque)", fontweight="bold", fontsize=14)
    plt.tight_layout()
    plt.savefig(chemin, bbox_inches="tight")
    plt.close()
    print(f"  [OK] {chemin.name}")


# ===========================================================================
#  2. Boxplots des caracteristiques
# ===========================================================================
def graphique_boxplots(pipeline, chemin, n_caracs=8):
    """Graphique 2 : Boxplots des N premieres caracteristiques par classe."""
    df = pipeline.donnees.copy()
    label_col = pipeline.colonne_label
    if label_col not in df.columns:
        print("  [!] Colonne label absente, boxplots ignores")
        return

    caracs = [c for c in df.columns if c != label_col][:n_caracs]
    n_cols = 4
    n_rows = (len(caracs) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 3 * n_rows))
    axes = axes.flatten()

    for i, carac in enumerate(caracs):
        for classe, couleur, label in [(0, "#2ecc71", "Normal"), (1, "#e74c3c", "Attaque")]:
            data = df[df[label_col] == classe][carac].dropna()
            if len(data) > 0:
                axes[i].boxplot(data, positions=[classe + 1], widths=0.5,
                                patch_artist=True,
                                boxprops=dict(facecolor=couleur, alpha=0.7),
                                medianprops=dict(color="white", linewidth=2))
        axes[i].set_title(carac, fontsize=9)
        axes[i].set_xticks([1, 2])
        axes[i].set_xticklabels(["Normal", "Attaque"], fontsize=7)
        axes[i].spines["top"].set_visible(False)
        axes[i].spines["right"].set_visible(False)

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.suptitle("Distribution des caracteristiques par classe (Boxplots)", fontweight="bold", fontsize=14)
    plt.tight_layout()
    plt.savefig(chemin, bbox_inches="tight")
    plt.close()
    print(f"  [OK] {chemin.name}")


# ===========================================================================
#  3. Matrice de correlation
# ===========================================================================
def graphique_correlation(pipeline, chemin):
    """Graphique 3 : Heatmap de correlation entre caracteristiques."""
    df = pipeline.donnees.select_dtypes(include=[np.number])
    corr = df.corr()

    fig, ax = plt.subplots(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, mask=mask, cmap="RdBu_r", center=0, annot=False,
                square=True, linewidths=0.5, ax=ax, cbar_kws={"shrink": 0.8})
    ax.set_title("Matrice de correlation des caracteristiques", fontweight="bold", fontsize=14)
    plt.tight_layout()
    plt.savefig(chemin, bbox_inches="tight")
    plt.close()
    print(f"  [OK] {chemin.name}")


# ===========================================================================
#  4. Courbes ROC avec AUC
# ===========================================================================
def graphique_courbes_roc(resultats, y_test, chemin):
    """Graphique 4 : Courbes ROC des 6 modeles avec AUC."""
    fig, ax = plt.subplots(figsize=(10, 8))

    couleurs = ["#2ecc71", "#3498db", "#e74c3c", "#f39c12", "#9b59b6", "#1abc9c"]

    for i, r in enumerate(resultats):
        if r["y_proba"] is not None:
            fpr, tpr, _ = roc_curve(y_test, r["y_proba"])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=couleurs[i % len(couleurs)], lw=2,
                    label=f"{r['desc']} (AUC = {roc_auc:.4f})")

    # Diagonale (classifieur aleatoire)
    ax.plot([0, 1], [0, 1], "k--", lw=1.5, alpha=0.5, label="Aleatoire (AUC = 0.5000)")

    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel("Taux de faux positifs (FPR)", fontsize=12)
    ax.set_ylabel("Taux de vrais positifs (TPR)", fontsize=12)
    ax.set_title("Courbes ROC - Comparaison des 6 modeles", fontweight="bold", fontsize=14)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(chemin, bbox_inches="tight")
    plt.close()
    print(f"  [OK] {chemin.name}")


# ===========================================================================
#  5. Matrice de confusion du meilleur modele
# ===========================================================================
def graphique_matrice_confusion(resultats, chemin):
    """Graphique 5 : Matrice de confusion du meilleur modele (F1-Score)."""
    meilleur = max(resultats, key=lambda r: r["metriques"]["f1_score"])
    matrice = meilleur["matrice"]

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(matrice, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Normal", "Attaque"],
                yticklabels=["Normal", "Attaque"],
                ax=ax, cbar=True,
                annot_kws={"fontsize": 16, "fontweight": "bold"})
    ax.set_xlabel("Predictions", fontsize=12)
    ax.set_ylabel("Valeurs reelles", fontsize=12)
    ax.set_title(f"Matrice de confusion - {meilleur['desc']}\n(F1-Score = {meilleur['metriques']['f1_score']:.4f})",
                 fontweight="bold", fontsize=13)
    plt.tight_layout()
    plt.savefig(chemin, bbox_inches="tight")
    plt.close()
    print(f"  [OK] {chemin.name} (meilleur: {meilleur['desc']})")


# ===========================================================================
#  6. Comparaison des metriques (barres groupees)
# ===========================================================================
def graphique_comparaison_metriques(resultats, chemin):
    """Graphique 6 : Barres groupees Accuracy/Precision/Recall/F1 par modele."""
    fig, ax = plt.subplots(figsize=(12, 6))

    modeles = [r["desc"].split(" - ")[0] for r in resultats]
    metriques = ["accuracy", "precision", "recall", "f1_score"]
    couleurs = ["#3498db", "#2ecc71", "#f39c12", "#e74c3c"]
    x = np.arange(len(modeles))
    largeur = 0.2

    for i, (met, couleur) in enumerate(zip(metriques, couleurs)):
        valeurs = [r["metriques"][met] for r in resultats]
        bars = ax.bar(x + i * largeur, valeurs, largeur, label=met, color=couleur,
                      edgecolor="white", linewidth=1)
        for bar, v in zip(bars, valeurs):
            if v < 0.95:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f"{v:.3f}", ha="center", fontsize=7, rotation=90)

    ax.set_xticks(x + largeur * 1.5)
    ax.set_xticklabels(modeles, fontsize=10, rotation=15)
    ax.set_ylabel("Score")
    ax.set_title("Comparaison des metriques par modele", fontweight="bold", fontsize=14)
    ax.legend(loc="lower right", fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.3)

    plt.tight_layout()
    plt.savefig(chemin, bbox_inches="tight")
    plt.close()
    print(f"  [OK] {chemin.name}")


# ===========================================================================
#  7. Importance des caracteristiques
# ===========================================================================
def graphique_importance(resultats, chemin, n_top=15):
    """Graphique 7 : Top N caracteristiques les plus importantes."""
    # Prendre le meilleur modele qui a feature_importances_
    meilleur = None
    for r in resultats:
        if hasattr(r["modele"], "feature_importances_"):
            meilleur = r
            break

    if meilleur is None:
        print("  [!] Aucun modele avec feature_importances_, graphique ignore")
        return

    importances = meilleur["modele"].feature_importances_
    indices = np.argsort(importances)[-n_top:]

    fig, ax = plt.subplots(figsize=(10, 6))
    couleurs = plt.cm.Blues(np.linspace(0.3, 0.9, n_top))
    noms = [f"Feature #{i+1}" for i in indices]

    ax.barh(range(n_top), importances[indices], color=couleurs, edgecolor="white")
    ax.set_yticks(range(n_top))
    ax.set_yticklabels(noms, fontsize=9)
    ax.set_xlabel("Importance relative")
    ax.set_title(f"Top {n_top} caracteristiques les plus importantes\n({meilleur['desc']})",
                 fontweight="bold", fontsize=14)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(chemin, bbox_inches="tight")
    plt.close()
    print(f"  [OK] {chemin.name}")


# ===========================================================================
#  8. Temps d'entrainement
# ===========================================================================
def graphique_temps_entrainement(resultats, chemin):
    """Graphique 8 : Temps d'entrainement par modele."""
    fig, ax = plt.subplots(figsize=(10, 5))

    modeles = [r["desc"].split(" - ")[0] for r in resultats]
    temps = [r["duree"] for r in resultats]
    couleurs = plt.cm.viridis(np.linspace(0.2, 0.8, len(resultats)))

    bars = ax.barh(modeles, temps, color=couleurs, edgecolor="white")
    for bar, v in zip(bars, temps):
        ax.text(bar.get_width() + max(temps)*0.01, bar.get_y() + bar.get_height()/2,
                f"{v:.3f}s", ha="left", va="center", fontsize=10)

    ax.set_xlabel("Temps (secondes)")
    ax.set_title("Temps d'entrainement par modele", fontweight="bold", fontsize=14)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(chemin, bbox_inches="tight")
    plt.close()
    print(f"  [OK] {chemin.name}")


# ===========================================================================
#  9. Taux de detection vs Faux positifs
# ===========================================================================
def graphique_detection_vs_fauxpositifs(resultats, chemin):
    """Graphique 9 : Scatter plot taux detection vs taux faux positifs."""
    fig, ax = plt.subplots(figsize=(9, 7))

    couleurs = ["#2ecc71", "#3498db", "#e74c3c", "#f39c12", "#9b59b6", "#1abc9c"]

    for i, r in enumerate(resultats):
        matrice = r["matrice"]
        vn, fp, fn, vp = matrice.ravel()
        tpr = vp / (vp + fn) if (vp + fn) > 0 else 0  # Taux detection
        fpr = fp / (fp + vn) if (fp + vn) > 0 else 0  # Taux faux +

        ax.scatter(fpr, tpr, s=200, color=couleurs[i % len(couleurs)],
                   edgecolor="white", linewidth=2, zorder=5)
        ax.annotate(r["desc"].split(" - ")[0],
                    (fpr, tpr), fontsize=9, ha="center",
                    xytext=(0, -20), textcoords="offset points")

    ax.set_xlabel("Taux de faux positifs (FPR)", fontsize=12)
    ax.set_ylabel("Taux de detection (TPR)", fontsize=12)
    ax.set_title("Taux de detection vs Faux positifs", fontweight="bold", fontsize=14)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.2)
    # Zone ideale en haut a gauche
    ax.axhline(y=1.0, color="green", linestyle="--", alpha=0.2)
    ax.axvline(x=0.0, color="green", linestyle="--", alpha=0.2)

    plt.tight_layout()
    plt.savefig(chemin, bbox_inches="tight")
    plt.close()
    print(f"  [OK] {chemin.name}")


# ===========================================================================
#  10. Tableau comparatif (format image)
# ===========================================================================
def graphique_tableau_comparatif(resultats, chemin):
    """Graphique 10 : Tableau comparatif des 6 modeles."""
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis("off")

    # Trier par F1-Score
    resultats_tries = sorted(resultats, key=lambda r: r["metriques"]["f1_score"], reverse=True)

    colonnes = ["Modele", "Accuracy", "Precision", "Rappel", "F1-Score", "Detection", "Faux +", "Temps (s)"]
    donnees = []
    for r in resultats_tries:
        m = r["matrice"]
        vn, fp, fn, vp = m.ravel()
        tpr = vp / (vp + fn) if (vp + fn) > 0 else 0
        fpr = fp / (fp + vn) if (fp + vn) > 0 else 0
        donnees.append([
            r["desc"].split(" - ")[0],
            f"{r['metriques']['accuracy']:.4f}",
            f"{r['metriques']['precision']:.4f}",
            f"{r['metriques']['recall']:.4f}",
            f"{r['metriques']['f1_score']:.4f}",
            f"{tpr:.2%}",
            f"{fpr:.2%}",
            f"{r['duree']:.3f}",
        ])

    table = ax.table(cellText=donnees, colLabels=colonnes,
                     cellLoc="center", loc="center",
                     colColours=["#1E88E5"] * len(colonnes))
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.6)

    # Colorer la premiere ligne (meilleur modele)
    for j in range(len(colonnes)):
        cellule = table[1, j]
        cellule.set_facecolor("#e8f5e9")
        cellule.get_text().set_fontsize(10)
        cellule.get_text().set_fontweight("bold")

    ax.set_title("Comparaison des 6 modeles - Classement par F1-Score", fontweight="bold", fontsize=14, pad=20)

    plt.tight_layout()
    plt.savefig(chemin, bbox_inches="tight")
    plt.close()
    print(f"  [OK] {chemin.name}")


# ===========================================================================
#  MAIN
# ===========================================================================
def main():
    parser = argparse.ArgumentParser(description="Generation des graphiques pour le memoire NetGuard AI")
    parser.add_argument("--dataset", default=DATASET_PAR_DEFAUT, choices=list(DATASETS_DISPONIBLES.keys()))
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()

    configurer_logging()
    DOSSIER_GRAPH.mkdir(parents=True, exist_ok=True)
    print()

    # --- 1. Preparation des donnees -----------------------------------------
    print("=" * 60)
    print("  GENERATION DES GRAPHIQUES POUR LE MEMOIRE")
    print("=" * 60)
    print(f"  Dataset : {args.dataset}")
    print(f"  Dossier  : {DOSSIER_GRAPH}/")
    print()

    with Chronometre("Preparation des donnees"):
        X_train, X_test, y_train, y_test, pipeline = preparer_donnees(
            args.dataset, download=args.download)
    print()

    # --- 2. Entrainement des 6 modeles --------------------------------------
    with Chronometre("Entrainement des 6 modeles"):
        resultats = entrainer_tous_les_modeles(X_train, y_train, X_test, y_test)
    print()

    # --- 3. Generation des graphiques ---------------------------------------
    print("Generation des graphiques...")
    print()

    graphique_distribution_classes(y_train, y_test, DOSSIER_GRAPH / "01_distribution_classes.png")
    graphique_boxplots(pipeline, DOSSIER_GRAPH / "02_boxplots_caracteristiques.png")
    graphique_correlation(pipeline, DOSSIER_GRAPH / "03_matrice_correlation.png")
    graphique_courbes_roc(resultats, y_test, DOSSIER_GRAPH / "04_courbes_roc.png")
    graphique_matrice_confusion(resultats, DOSSIER_GRAPH / "05_matrice_confusion.png")
    graphique_comparaison_metriques(resultats, DOSSIER_GRAPH / "06_comparaison_metriques.png")
    graphique_importance(resultats, DOSSIER_GRAPH / "07_importance_caracteristiques.png")
    graphique_temps_entrainement(resultats, DOSSIER_GRAPH / "08_temps_entrainement.png")
    graphique_detection_vs_fauxpositifs(resultats, DOSSIER_GRAPH / "09_detection_vs_faux_positifs.png")
    graphique_tableau_comparatif(resultats, DOSSIER_GRAPH / "10_tableau_comparatif.png")

    # --- 4. Resume ----------------------------------------------------------
    meilleur = max(resultats, key=lambda r: r["metriques"]["f1_score"])
    print()
    print("=" * 60)
    print("  10 GRAPHIQUES GENERES AVEC SUCCES")
    print("=" * 60)
    print(f"  Dossier : {DOSSIER_GRAPH}/")
    print(f"  Meilleur modele : {meilleur['desc']}")
    print(f"  F1-Score : {meilleur['metriques']['f1_score']:.4f}")
    print(f"  Accuracy : {meilleur['metriques']['accuracy']:.4f}")
    print("=" * 60)
    print()
    print("  Liste des fichiers :")
    for f in sorted(DOSSIER_GRAPH.glob("*.png")):
        taille = f.stat().st_size / 1024
        print(f"    [IMG] {f.name} ({taille:.0f} Ko)")
    print()


if __name__ == "__main__":
    main()
