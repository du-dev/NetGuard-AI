"""
Script de generation des diagrammes supplementaires pour le memoire :
11 - Diagramme d'architecture du pipeline
12 - Mockup interface Streamlit

Utilisation :
    python generate_diagrammes_memoire.py
"""

import sys
from pathlib import Path

PROJET_RACINE = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJET_RACINE))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from src.config import DOSSIER_DATA_TRANSFORME
from src.utils.helpers import configurer_logging

DOSSIER_GRAPH = DOSSIER_DATA_TRANSFORME / "graphiques"
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "font.size": 11,
})


# ===========================================================================
#  11 - Diagramme d'architecture du pipeline
# ===========================================================================
def diagramme_architecture(chemin):
    """Graphique 11 : Diagramme d'architecture du systeme."""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis("off")

    couleurs = {
        "dataset": "#3498db",
        "preprocessing": "#2ecc71",
        "features": "#9b59b6",
        "models": "#e74c3c",
        "evaluation": "#f39c12",
        "interface": "#1abc9c",
    }

    # ─── Niveau 1 : Entree ───────────────────────────────────────────────
    boite_dataset = mpatches.FancyBboxPatch(
        (1, 6.5), 4, 1.2, boxstyle="round,pad=0.15",
        facecolor=couleurs["dataset"], edgecolor="white", linewidth=2)
    ax.add_patch(boite_dataset)
    ax.text(3, 7.1, "DATASET", ha="center", va="center",
            fontsize=13, fontweight="bold", color="white")
    ax.text(3, 6.7, "CSV : synthetique / CICIDS2017", ha="center",
            va="center", fontsize=9, color="white", alpha=0.9)

    # ─── Niveau 2 : Pretraitement ─────────────────────────────────────────
    boite_preproc = mpatches.FancyBboxPatch(
        (1, 4.5), 4, 1.6, boxstyle="round,pad=0.15",
        facecolor=couleurs["preprocessing"], edgecolor="white", linewidth=2)
    ax.add_patch(boite_preproc)
    ax.text(3, 5.7, "PRETRAITEMENT", ha="center", va="center",
            fontsize=12, fontweight="bold", color="white")
    etapes = ["Chargement", "Nettoyage", "Normalisation", "Train/Test"]
    for i, e in enumerate(etapes):
        ax.text(3, 5.2 - i * 0.3, f"  {i+1}. {e}", ha="center",
                va="center", fontsize=8, color="white", alpha=0.9)

    # Fleche 1 -> 2
    ax.annotate("", xy=(3, 6.5), xytext=(3, 5.8),
                arrowprops=dict(arrowstyle="->", color="#555", lw=2.5))

    # ─── Niveau 3 : Extraction caracteristiques ───────────────────────────
    boite_features = mpatches.FancyBboxPatch(
        (7, 5.5), 4, 1, boxstyle="round,pad=0.15",
        facecolor=couleurs["features"], edgecolor="white", linewidth=2)
    ax.add_patch(boite_features)
    ax.text(9, 6.0, "CARACTERISTIQUES", ha="center", va="center",
            fontsize=11, fontweight="bold", color="white")
    ax.text(9, 5.7, "PCA / SelectKBest / ANOVA", ha="center",
            va="center", fontsize=8, color="white", alpha=0.9)

    # Fleche 2 -> 3
    ax.annotate("", xy=(5, 5.3), xytext=(7, 5.8),
                arrowprops=dict(arrowstyle="->", color="#555", lw=2.5))

    # ─── Niveau 4 : Modeles ──────────────────────────────────────────────
    boite_modeles = mpatches.FancyBboxPatch(
        (1, 2.5), 4, 2, boxstyle="round,pad=0.15",
        facecolor=couleurs["models"], edgecolor="white", linewidth=2)
    ax.add_patch(boite_modeles)
    ax.text(3, 4.1, "6 MODELES DE ML", ha="center", va="center",
            fontsize=11, fontweight="bold", color="white")
    modeles = ["Decision Tree", "Random Forest", "Logistic Regression",
               "SVM", "KNN", "Naive Bayes"]
    for i, m in enumerate(modeles):
        col = "lightgreen" if i == 1 else "white"
        ax.text(3, 3.8 - i * 0.25, f"{'★' if i==1 else ' '} {m}",
                ha="center", va="center", fontsize=7.5,
                color=col, fontweight="bold" if i == 1 else "normal")

    # Fleche 3 -> 4
    ax.annotate("", xy=(9, 5.5), xytext=(5, 3.5),
                arrowprops=dict(arrowstyle="->", color="#555", lw=2.5))

    # ─── Niveau 5 : Evaluation ───────────────────────────────────────────
    boite_eval = mpatches.FancyBboxPatch(
        (7, 3.5), 4, 1.2, boxstyle="round,pad=0.15",
        facecolor=couleurs["evaluation"], edgecolor="white", linewidth=2)
    ax.add_patch(boite_eval)
    ax.text(9, 4.3, "EVALUATION", ha="center", va="center",
            fontsize=11, fontweight="bold", color="white")
    ax.text(9, 3.9, "Accuracy / Precision / Recall / F1\nMatrice / ROC / AUC",
            ha="center", va="center", fontsize=7.5, color="white", alpha=0.9)

    # Fleche 4 -> 5
    ax.annotate("", xy=(5, 2.5), xytext=(7, 3.5),
                arrowprops=dict(arrowstyle="->", color="#555", lw=2.5))

    # ─── Niveau 6 : Interface ────────────────────────────────────────────
    boite_interface = mpatches.FancyBboxPatch(
        (1, 0.5), 4, 1.2, boxstyle="round,pad=0.15",
        facecolor=couleurs["interface"], edgecolor="white", linewidth=2)
    ax.add_patch(boite_interface)
    ax.text(3, 1.1, "INTERFACE UTILISATEUR", ha="center", va="center",
            fontsize=11, fontweight="bold", color="white")
    ax.text(3, 0.7, "CLI (main.py) / Web (Streamlit) / Demo (demo.py)",
            ha="center", va="center", fontsize=7.5, color="white", alpha=0.9)

    # Fleche 5 -> 6
    ax.annotate("", xy=(9, 3.5), xytext=(5, 1.1),
                arrowprops=dict(arrowstyle="->", color="#555", lw=2.5))

    # ─── Boite sauvegarde ─────────────────────────────────────────────────
    boite_save = mpatches.FancyBboxPatch(
        (7, 0.5), 4, 1.2, boxstyle="round,pad=0.15",
        facecolor="#95a5a6", edgecolor="white", linewidth=2)
    ax.add_patch(boite_save)
    ax.text(9, 1.1, "SAUVEGARDE", ha="center", va="center",
            fontsize=11, fontweight="bold", color="white")
    ax.text(9, 0.7, "Modele (.joblib) / Metriques / Graphes",
            ha="center", va="center", fontsize=7.5, color="white", alpha=0.9)

    ax.annotate("", xy=(5, 0.5), xytext=(7, 0.5),
                arrowprops=dict(arrowstyle="->", color="#555", lw=2))

    # Titre
    ax.text(7, 7.8, "Architecture du systeme NetGuard AI",
            ha="center", va="center", fontsize=16, fontweight="bold",
            color="#2c3e50")

    # Legende
    legendes = [
        mpatches.Patch(color=couleurs["dataset"], label="Entree"),
        mpatches.Patch(color=couleurs["preprocessing"], label="Pretraitement"),
        mpatches.Patch(color=couleurs["features"], label="Caracteristiques"),
        mpatches.Patch(color=couleurs["models"], label="Modeles ML"),
        mpatches.Patch(color=couleurs["evaluation"], label="Evaluation"),
        mpatches.Patch(color=couleurs["interface"], label="Interface"),
    ]
    ax.legend(handles=legendes, loc="upper right", fontsize=9,
              framealpha=0.9, ncol=2)

    plt.tight_layout()
    plt.savefig(chemin, bbox_inches="tight")
    plt.close()
    print(f"  [OK] {chemin.name}")


# ===========================================================================
#  12 - Mockup interface Streamlit
# ===========================================================================
def mockup_streamlit(chemin):
    """Graphique 12 : Mockup visuel de l'interface Streamlit."""
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.axis("off")

    # Fond de l'interface
    fond = mpatches.FancyBboxPatch(
        (0.2, 0.2), 9.6, 6.6, boxstyle="round,pad=0.1",
        facecolor="#f0f2f6", edgecolor="#ddd", linewidth=2)
    ax.add_patch(fond)

    # ─── Barre laterale (sidebar) ─────────────────────────────────────────
    sidebar = mpatches.FancyBboxPatch(
        (0.4, 0.4), 2.5, 6.2, boxstyle="round,pad=0.08",
        facecolor="#1E1E2E", edgecolor="none")
    ax.add_patch(sidebar)

    # Titre sidebar
    ax.text(1.65, 6.3, "NetGuard AI", ha="center", va="center",
            fontsize=12, fontweight="bold", color="#1E88E5")
    ax.text(1.65, 6.0, "Detection d'intrusions", ha="center", va="center",
            fontsize=8, color="#aaa")

    # Separateur
    ax.plot([0.7, 2.6], [5.7, 5.7], color="#444", lw=0.5)

    # Menu sidebar
    items = ["Charger modele", "Upload fichier", "Lancer prediction"]
    icones = ["[H]", "[C]", "[R]"]
    for i, (item, icone) in enumerate(zip(items, icones)):
        y = 5.3 - i * 0.6
        btn = mpatches.FancyBboxPatch(
            (0.7, y - 0.2), 2.0, 0.4, boxstyle="round,pad=0.05",
            facecolor="#2d2d44" if i == 0 else "none",
            edgecolor="#444" if i == 0 else "none")
        ax.add_patch(btn)
        ax.text(1.7, y, f"{icone}  {item}", ha="center", va="center",
                fontsize=7.5, color="#ccc" if i > 0 else "white")

    # ─── Zone principale ──────────────────────────────────────────────────
    # Titre principal
    ax.text(6, 6.3, "Prediction de trafic reseau",
            ha="center", va="center", fontsize=15, fontweight="bold",
            color="#2c3e50")

    # Carte resultat
    carte_resultat = mpatches.FancyBboxPatch(
        (3.5, 4.8), 5, 1, boxstyle="round,pad=0.1",
        facecolor="#27ae60", edgecolor="none")
    ax.add_patch(carte_resultat)
    ax.text(6, 5.5, "TRAFIC NORMAL", ha="center", va="center",
            fontsize=18, fontweight="bold", color="white")
    ax.text(6, 5.1, "Aucune menace detectee", ha="center", va="center",
            fontsize=10, color="white", alpha=0.9)

    # Cartes metriques
    metriques = [("Accuracy", "1.0000", "#2ecc71"),
                 ("Precision", "1.0000", "#3498db"),
                 ("Rappel", "1.0000", "#f39c12"),
                 ("F1-Score", "1.0000", "#e74c3c")]
    for i, (nom, val, couleur) in enumerate(metriques):
        x = 3.5 + i * 1.35
        carte = mpatches.FancyBboxPatch(
            (x, 3.3), 1.2, 0.7, boxstyle="round,pad=0.08",
            facecolor=couleur, edgecolor="white", linewidth=1)
        ax.add_patch(carte)
        ax.text(x + 0.6, 3.8, val, ha="center", va="center",
                fontsize=14, fontweight="bold", color="white")
        ax.text(x + 0.6, 3.45, nom, ha="center", va="center",
                fontsize=6.5, color="white", alpha=0.9)

    # Tableau de resultats
    tableau = mpatches.FancyBboxPatch(
        (3.5, 0.8), 5.5, 2, boxstyle="round,pad=0.08",
        facecolor="white", edgecolor="#ddd", linewidth=1)
    ax.add_patch(tableau)
    ax.text(4, 0.95, "  Resultats de la prediction", ha="left", va="center",
            fontsize=9, fontweight="bold", color="#555")

    # Lignes du tableau
    cols = ["#", "Feature", "Valeur", "Prediction"]
    headers = ["1", "feature_01", "0.234", "Normal",
               "2", "feature_02", "-1.567", "Normal",
               "3", "feature_03", "3.456", "Attaque"]
    for i, h in enumerate(cols):
        ax.text(3.7 + i * 1.3, 2.5, h, ha="center", va="center",
                fontsize=7, fontweight="bold", color="#888")
    for row in range(3):
        for col in range(4):
            val = headers[col + row * 4]
            couleur_t = "#e74c3c" if val == "Attaque" else "#2ecc71" if val == "Normal" else "#555"
            ax.text(3.7 + col * 1.3, 2.05 - row * 0.3, val,
                    ha="center", va="center", fontsize=7,
                    fontweight="bold" if col == 3 else "normal",
                    color=couleur_t if col == 3 else "#555")

    # Pied de page
    ax.text(6, 0.45, "NetGuard AI v1.0  |  Modele: Random Forest  |  93 tests OK",
            ha="center", va="center", fontsize=7.5, color="#999")

    # Titre
    ax.text(6, 6.8, "Interface Streamlit - Visualisation des predictions",
            ha="center", va="center", fontsize=13, fontweight="bold",
            color="#2c3e50")

    plt.tight_layout()
    plt.savefig(chemin, bbox_inches="tight")
    plt.close()
    print(f"  [OK] {chemin.name}")


# ===========================================================================
#  MAIN
# ===========================================================================
def main():
    configurer_logging()
    DOSSIER_GRAPH.mkdir(parents=True, exist_ok=True)
    print()
    print("=" * 60)
    print("  GENERATION DES DIAGRAMMES SUPPLEMENTAIRES")
    print("=" * 60)
    print()

    diagramme_architecture(DOSSIER_GRAPH / "11_architecture_pipeline.png")
    mockup_streamlit(DOSSIER_GRAPH / "12_interface_streamlit.png")

    print()
    print("=" * 60)
    print("  2 DIAGRAMMES GENERES AVEC SUCCES")
    print("=" * 60)
    print(f"  Dossier : {DOSSIER_GRAPH}/")
    for f in sorted(DOSSIER_GRAPH.glob("11_*.png")):
        print(f"    [DIAG] {f.name}")
    for f in sorted(DOSSIER_GRAPH.glob("12_*.png")):
        print(f"    [UI] {f.name}")
    print()


if __name__ == "__main__":
    main()
