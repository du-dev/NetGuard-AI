"""
Application Streamlit pour NetGuard AI.

Interface web de détection d'intrusions réseau permettant :
- Charger un modèle entraîné
- Charger un fichier de données (CSV)
- Effectuer des prédictions
- Visualiser si le trafic est normal ou malveillant
- Afficher les performances du modèle
- Afficher des graphiques (matrice de confusion, importance)

Utilisation :
    streamlit run app.py
"""

import sys
from pathlib import Path
from typing import Any, Optional

# Ajout de la racine du projet pour les imports
PROJET_RACINE = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJET_RACINE))

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from src.config import (
    DESCRIPTION_MODELES,
    DOSSIER_DATA_TRANSFORME,
    DOSSIER_DATA_BRUT,
)
from src.models.detector import DetecteurIntrusions, FABRIQUE_MODELES
from src.evaluation.metrics import Evaluateur
from src.utils.helpers import charger_modele


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  CONFIGURATION DE LA PAGE                                              ║
# ╚══════════════════════════════════════════════════════════════════════════╝

st.set_page_config(
    page_title="NetGuard AI - Détection d'intrusions",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Style CSS personnalisé ───────────────────────────────────────────────
st.markdown("""
    <style>
        /* Titre principal */
        .main-title {
            font-size: 2.5rem;
            font-weight: 700;
            color: #1E88E5;
            margin-bottom: 0.2rem;
        }
        .main-subtitle {
            font-size: 1.1rem;
            color: #666;
            margin-bottom: 2rem;
        }
        /* Cartes de métriques */
        .metric-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 1.2rem;
            border-radius: 12px;
            color: white;
            text-align: center;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }
        .metric-card.green {
            background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        }
        .metric-card.orange {
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        }
        .metric-card.blue {
            background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        }
        .metric-card.purple {
            background: linear-gradient(135deg, #a18cd1 0%, #fbc2eb 100%);
        }
        .metric-value {
            font-size: 2rem;
            font-weight: 700;
        }
        .metric-label {
            font-size: 0.85rem;
            opacity: 0.9;
        }
        /* Résultat de prédiction */
        .prediction-normal {
            background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
            padding: 1.5rem;
            border-radius: 12px;
            color: white;
            text-align: center;
            font-size: 1.5rem;
            font-weight: 700;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }
        .prediction-attack {
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            padding: 1.5rem;
            border-radius: 12px;
            color: white;
            text-align: center;
            font-size: 1.5rem;
            font-weight: 700;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }
        /* En-tête de section */
        .section-header {
            font-size: 1.3rem;
            font-weight: 600;
            color: #1E88E5;
            margin-top: 1.5rem;
            margin-bottom: 1rem;
            padding-bottom: 0.5rem;
            border-bottom: 2px solid #1E88E5;
        }
        /* Pied de page */
        .footer {
            text-align: center;
            color: #999;
            font-size: 0.85rem;
            margin-top: 3rem;
            padding-top: 1rem;
            border-top: 1px solid #eee;
        }
        /* Info box */
        .info-box {
            background: #f0f2f6;
            padding: 1rem;
            border-radius: 10px;
            border-left: 4px solid #1E88E5;
        }
    </style>
""", unsafe_allow_html=True)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  FONCTIONS UTILITAIRES (AVEC CACHE)                                    ║
# ╚══════════════════════════════════════════════════════════════════════════╝

@st.cache_resource
def charger_modele_cache(chemin: str) -> Optional[Any]:
    """
    Charge un modèle depuis le disque avec mise en cache Streamlit.

    Le décorateur @st.cache_resource garantit que le modèle
    n'est chargé qu'une seule fois en mémoire, même si
    l'utilisateur interagit avec l'application.

    Args:
        chemin: Chemin vers le fichier .joblib du modèle.

    Returns:
        Le modèle scikit-learn chargé, ou None si erreur.
    """
    chemin_modele = Path(chemin)
    if not chemin_modele.exists():
        return None
    try:
        return charger_modele(chemin_modele)
    except Exception as e:
        st.error(f"Erreur lors du chargement : {e}")
        return None


def chercher_modeles_disponibles() -> list:
    """
    Cherche tous les modèles .joblib disponibles dans data/processed/.

    Returns:
        Liste des chemins des modèles trouvés.
    """
    dossier = DOSSIER_DATA_TRANSFORME
    if not dossier.exists():
        return []
    modeles = sorted(dossier.glob("*.joblib"))
    return modeles


def analyser_fichier(uploaded_file) -> Optional[pd.DataFrame]:
    """
    Analyse un fichier CSV uploadé et retourne un DataFrame.

    Gère les erreurs de format et les colonnes manquantes.

    Args:
        uploaded_file: Fichier uploadé via Streamlit.

    Returns:
        DataFrame des données, ou None si erreur.
    """
    try:
        df = pd.read_csv(uploaded_file)
        return df
    except Exception as e:
        st.error(f"Erreur de lecture du fichier : {e}")
        return None


def generer_graphique_matrice_confusion(matrice: np.ndarray) -> plt.Figure:
    """
    Génère un graphique de matrice de confusion avec seaborn.

    Args:
        matrice: Matrice de confusion 2x2.

    Returns:
        Figure matplotlib prête à afficher.
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        matrice,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Normal", "Attaque"],
        yticklabels=["Normal", "Attaque"],
        ax=ax,
        cbar_kws={"shrink": 0.8},
    )
    ax.set_xlabel("Prédictions", fontsize=12)
    ax.set_ylabel("Valeurs réelles", fontsize=12)
    ax.set_title("Matrice de confusion", fontsize=14, fontweight="bold")
    plt.tight_layout()
    return fig


def generer_graphique_importance(
    modele: Any, noms_colonnes: Optional[list] = None
) -> plt.Figure:
    """
    Génère un graphique d'importance des caractéristiques.

    Fonctionne avec les modèles qui exposent feature_importances_
    (Random Forest, Decision Tree) ou coef_ (Logistic Regression, SVM linéaire).

    Args:
        modele: Modèle entraîné scikit-learn.
        noms_colonnes: Noms des caractéristiques (optionnel).

    Returns:
        Figure matplotlib prête à afficher, ou None si non disponible.
    """
    # Vérification de la disponibilité des importances
    importances = None
    if hasattr(modele, "feature_importances_"):
        importances = modele.feature_importances_
        titre = "Importance des caractéristiques (feature_importances_)"
    elif hasattr(modele, "coef_") and modele.coef_.shape[0] <= 1:
        # Pour les modèles linéaires
        importances = np.abs(modele.coef_[0]) if modele.coef_.ndim > 1 else np.abs(modele.coef_)
        # Normalisation
        importances = importances / importances.sum() if importances.sum() > 0 else importances
        titre = "Importance des caractéristiques (coefficients)"

    if importances is None:
        return None

    # Création du graphique
    fig, ax = plt.subplots(figsize=(10, 6))

    n_features = min(len(importances), 20)  # Top 20 max
    indices = np.argsort(importances)[-n_features:]

    if noms_colonnes is not None and len(noms_colonnes) >= len(importances):
        noms = [noms_colonnes[i] for i in indices]
    else:
        noms = [f"Caractéristique #{i+1}" for i in indices]

    colors = plt.cm.Blues(np.linspace(0.3, 0.9, n_features))

    ax.barh(range(n_features), importances[indices], color=colors)
    ax.set_yticks(range(n_features))
    ax.set_yticklabels(noms, fontsize=10)
    ax.set_xlabel("Importance relative", fontsize=12)
    ax.set_title(titre, fontsize=14, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    return fig


def generer_graphique_distribution(y_pred: np.ndarray, y_proba: Optional[np.ndarray] = None) -> plt.Figure:
    """
    Génère un graphique de distribution des prédictions.

    Affiche un camembert de la répartition normal/attaque.

    Args:
        y_pred: Prédictions (0 = normal, 1 = attaque).
        y_proba: Probabilités (optionnel).

    Returns:
        Figure matplotlib prête à afficher.
    """
    nb_normal = int(np.sum(y_pred == 0))
    nb_attaque = int(np.sum(y_pred == 1))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Camembert
    labels = ["Normal", "Attaque"]
    sizes = [nb_normal, nb_attaque]
    colors = ["#38ef7d", "#f5576c"]
    explode = (0, 0.1) if nb_attaque > 0 else (0, 0)

    ax1.pie(
        sizes, labels=labels, colors=colors, explode=explode,
        autopct="%1.1f%%", startangle=90, shadow=True,
        textprops={"fontsize": 12, "fontweight": "bold"},
    )
    ax1.set_title("Répartition des prédictions", fontsize=14, fontweight="bold")

    # Barres
    barres = ax2.bar(labels, sizes, color=colors, width=0.5, edgecolor="white", linewidth=2)
    for barre, valeur in zip(barres, sizes):
        ax2.text(
            barre.get_x() + barre.get_width() / 2,
            barre.get_height() + max(sizes) * 0.02,
            str(valeur),
            ha="center", va="bottom", fontsize=14, fontweight="bold",
        )
    ax2.set_ylabel("Nombre d'échantillons", fontsize=12)
    ax2.set_title("Résumé des prédictions", fontsize=14, fontweight="bold")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.set_ylim(0, max(sizes) * 1.15)

    plt.tight_layout()
    return fig


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  INTERFACE UTILISATEUR                                                ║
# ╚══════════════════════════════════════════════════════════════════════════╝

# ─── Barre latérale ───────────────────────────────────────────────────────
with st.sidebar:
    # Logo et titre
    st.markdown("""
        <div style="text-align: center; margin-bottom: 1.5rem;">
            <span style="font-size: 3rem;">🛡️</span>
            <h2 style="margin: 0; color: #1E88E5;">NetGuard AI</h2>
            <p style="color: #666; font-size: 0.9rem;">Détection d'intrusions</p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # ─── Section : Chargement du modèle ───────────────────────────────────
    st.markdown("### 📦 Modèle")

    # Chercher les modèles disponibles
    modeles_disponibles = chercher_modeles_disponibles()

    if modeles_disponibles:
        # Créer une liste de noms pour la sélection
        noms_modeles = [m.name for m in modeles_disponibles]
        modele_choisi = st.selectbox(
            "Sélectionnez un modèle entraîné :",
            noms_modeles,
            index=0,
            help="Modèles disponibles dans data/processed/",
        )
        chemin_modele = DOSSIER_DATA_TRANSFORME / modele_choisi

        # Bouton pour charger le modèle
        if st.button("🔃 Charger le modèle", use_container_width=True, type="primary"):
            with st.spinner("Chargement du modèle..."):
                modele_sklearn = charger_modele_cache(str(chemin_modele))
                if modele_sklearn is not None:
                    st.session_state["modele"] = modele_sklearn
                    st.session_state["chemin_modele"] = str(chemin_modele)
                    st.success(f"✅ Modèle chargé : {modele_choisi}")
                else:
                    st.error("❌ Modèle introuvable ou invalide.")
                    st.session_state["modele"] = None
    else:
        st.warning("""
        ⚠️ Aucun modèle trouvé.
        Entraînez d'abord un modèle :
        ```
        python train.py --sauvegarder
        ```
        """)
        if st.button("🎲 Entraîner un modèle de démo", use_container_width=True):
            with st.spinner("Entraînement en cours..."):
                import subprocess
                resultat = subprocess.run(
                    [sys.executable, "demo.py"],
                    capture_output=True, text=True, timeout=60,
                )
                if resultat.returncode == 0:
                    st.success("✅ Modèle de démonstration créé ! Rechargez la page.")
                    st.rerun()
                else:
                    st.error(f"❌ Erreur : {resultat.stderr[:200]}")

    # Afficher les infos du modèle chargé
    if "modele" in st.session_state and st.session_state["modele"] is not None:
        modele = st.session_state["modele"]
        type_modele = type(modele).__name__

        # Trouver la description
        description = type_modele
        for nom, (classe, _) in FABRIQUE_MODELES.items():
            if isinstance(modele, classe):
                description = DESCRIPTION_MODELES.get(nom, type_modele)
                break

        st.markdown("""
        <div class="info-box">
            <b>Modèle chargé :</b><br>
        """ + f"{description}" + """
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # ─── Section : Chargement des données ─────────────────────────────────
    st.markdown("### 📁 Données")

    fichier_upload = st.file_uploader(
        "Chargez un fichier CSV :",
        type=["csv"],
        help="Fichier CSV avec les caractéristiques réseau (sans la colonne label)",
    )

    if fichier_upload is not None:
        with st.spinner("Analyse du fichier..."):
            df = analyser_fichier(fichier_upload)
            if df is not None:
                st.session_state["donnees"] = df
                st.success(f"✅ {len(df)} échantillons chargés")

    st.markdown("---")

    # ─── Section : Exemple de fichier ─────────────────────────────────────
    with st.expander("📖 Format attendu"):
        st.markdown("""
        **Le fichier CSV doit contenir :**
        - Des colonnes de caractéristiques numériques
        - Pas de colonne 'label' (optionnelle pour comparaison)
        
        **Exemple de format :**
        ```
        feature_01,feature_02,feature_03,...
        0.5,1.2,30,...
        -0.3,0.8,45,...
        ```
        
        **Nom des colonnes :** Ceux utilisés lors de l'entraînement.
        """)

    # Informations système
    st.markdown("---")
    st.markdown(
        "<p style='text-align: center; color: #999; font-size: 0.8rem;'>"
        "NetGuard AI v1.0 | Projet Master</p>",
        unsafe_allow_html=True,
    )


# ─── Page principale ──────────────────────────────────────────────────────

# Titre
st.markdown('<div class="main-title">🛡️ NetGuard AI</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="main-subtitle">'
    'Détection d\'intrusions réseau par Machine Learning</div>',
    unsafe_allow_html=True,
)

# ─── Vérifications préalables ─────────────────────────────────────────────
modele_charge = (
    "modele" in st.session_state
    and st.session_state["modele"] is not None
)
donnees_chargees = "donnees" in st.session_state

if not modele_charge and not donnees_chargees:
    # État initial : instructions
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div style="text-align: center; padding: 2rem;">
            <span style="font-size: 3rem;">1️⃣</span>
            <h4>Charger un modèle</h4>
            <p style="color: #666;">
                Sélectionnez un modèle entraîné dans la barre latérale.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div style="text-align: center; padding: 2rem;">
            <span style="font-size: 3rem;">2️⃣</span>
            <h4>Charger des données</h4>
            <p style="color: #666;">
                Uploadez un fichier CSV de trafic réseau.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div style="text-align: center; padding: 2rem;">
            <span style="font-size: 3rem;">3️⃣</span>
            <h4>Analyser</h4>
            <p style="color: #666;">
                Lancez la prédiction et visualisez les résultats.
            </p>
        </div>
        """, unsafe_allow_html=True)

elif modele_charge and donnees_chargees:
    # Mode prédiction : modèle + données chargés
    modele = st.session_state["modele"]
    df = st.session_state["donnees"]

    # ─── Aperçu des données ──────────────────────────────────────────────
    st.markdown('<div class="section-header">📊 Aperçu des données</div>',
                unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("Échantillons", len(df))
    col2.metric("Caractéristiques", df.shape[1])
    col3.metric("Type", "CSV importé")

    with st.expander("Voir les données", expanded=False):
        st.dataframe(df.head(10), use_container_width=True)
        st.caption(f"Affichage des 10 premières lignes sur {len(df)}")

    # ─── Prédiction ──────────────────────────────────────────────────────
    st.markdown('<div class="section-header">🔍 Prédiction</div>',
                unsafe_allow_html=True)

    col1, col2 = st.columns([1, 2])

    with col1:
        bouton_predire = st.button(
            "🚀 Lancer la prédiction",
            type="primary",
            use_container_width=True,
        )

    with col2:
        # Option pour afficher les probabilités
        afficher_probas = st.checkbox(
            "Afficher les probabilités", value=True,
            help="Montre le niveau de confiance pour chaque prédiction",
        )

    if bouton_predire:
        with st.spinner("Prédiction en cours..."):
            try:
                # Conversion en numpy array
                X = df.values.astype(np.float64)

                # Prédiction
                y_pred = modele.predict(X)

                # Probabilités (si supportées)
                y_proba = None
                if afficher_probas and hasattr(modele, "predict_proba"):
                    try:
                        y_proba = modele.predict_proba(X)
                    except Exception:
                        pass

                # Stockage dans la session
                st.session_state["y_pred"] = y_pred
                st.session_state["y_proba"] = y_proba

                # Affichage du résultat
                nb_normal = int(np.sum(y_pred == 0))
                nb_attaque = int(np.sum(y_pred == 1))

                # ─── Section Performance si label présent ─────────────────
                colonne_label = None
                for col_name in ["label", "Label", "class", "Class"]:
                    if col_name in df.columns:
                        colonne_label = col_name
                        break

                if colonne_label is not None:
                    st.markdown(
                        '<div class="section-header">📈 Performances du modèle</div>',
                        unsafe_allow_html=True,
                    )

                    y_reel = df[colonne_label].values

                    # Conversion du label si nécessaire (texte → numérique)
                    if y_reel.dtype == "object" or y_reel.dtype.kind in ("U", "S"):
                        from sklearn.preprocessing import LabelEncoder
                        le = LabelEncoder()
                        y_reel = le.fit_transform(y_reel)

                    # Calcul des métriques
                    evaluateur = Evaluateur()
                    metriques = evaluateur.calculer_metriques(y_reel, y_pred)
                    matrice = evaluateur.calculer_matrice_confusion(y_reel, y_pred)
                    rapport = evaluateur.generer_rapport(y_reel, y_pred)
                    taux = evaluateur.calculer_taux_detection(matrice)

                    # Affichage des métriques en cartes
                    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                    with col_m1:
                        st.markdown(
                            f'<div class="metric-card green">'
                            f'<div class="metric-value">{metriques["accuracy"]:.4f}</div>'
                            f'<div class="metric-label">Accuracy</div></div>',
                            unsafe_allow_html=True,
                        )
                    with col_m2:
                        st.markdown(
                            f'<div class="metric-card blue">'
                            f'<div class="metric-value">{metriques["precision"]:.4f}</div>'
                            f'<div class="metric-label">Précision</div></div>',
                            unsafe_allow_html=True,
                        )
                    with col_m3:
                        st.markdown(
                            f'<div class="metric-card orange">'
                            f'<div class="metric-value">{metriques["recall"]:.4f}</div>'
                            f'<div class="metric-label">Rappel</div></div>',
                            unsafe_allow_html=True,
                        )
                    with col_m4:
                        st.markdown(
                            f'<div class="metric-card purple">'
                            f'<div class="metric-value">{metriques["f1_score"]:.4f}</div>'
                            f'<div class="metric-label">F1-Score</div></div>',
                            unsafe_allow_html=True,
                        )

                    # Matrice de confusion
                    col_mc1, col_mc2 = st.columns([1, 1])
                    with col_mc1:
                        fig_cm = generer_graphique_matrice_confusion(matrice)
                        st.pyplot(fig_cm)
                        plt.close(fig_cm)

                    with col_mc2:
                        st.markdown(
                            f'<div class="info-box">'
                            f'<b>🔒 Taux de détection</b><br>'
                            f'✅ Vrais positifs (attaques détectées) : {int(matrice[1,1])}<br>'
                            f'❌ Faux négatifs (attaques manquées) : {int(matrice[1,0])}<br>'
                            f'✅ Vrais négatifs (normal correct) : {int(matrice[0,0])}<br>'
                            f'⚠️ Faux positifs (fausses alarmes) : {int(matrice[0,1])}<br><br>'
                            f'<b>Taux détection :</b> {taux["taux_vrais_positifs"]:.2%}<br>'
                            f'<b>Taux faux + :</b> {taux["taux_faux_positifs"]:.2%}'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

                    # Rapport de classification (détaillé)
                    with st.expander("📄 Voir le rapport de classification complet"):
                        st.text(rapport)

                # Résumé en haut
                col_r1, col_r2 = st.columns(2)

                with col_r1:
                    if nb_attaque > 0:
                        st.markdown(
                            f'<div class="prediction-attack">'
                            f'⚠️ {nb_attaque} intrusion(s) détectée(s) !</div>',
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(
                            f'<div class="prediction-normal">'
                            f'✅ Trafic normal - {nb_normal} échantillons</div>',
                            unsafe_allow_html=True,
                        )

                with col_r2:
                    st.markdown(f"""
                    <div class="info-box" style="border-left-color: #f5576c;">
                        <b>Résumé de l'analyse :</b><br>
                        ✅ Normal : {nb_normal} ({nb_normal/len(y_pred)*100:.1f}%)<br>
                        ⚠️ Attaques : {nb_attaque} ({nb_attaque/len(y_pred)*100:.1f}%)
                    </div>
                    """, unsafe_allow_html=True)

                # ─── Graphiques ───────────────────────────────────────────
                st.markdown(
                    '<div class="section-header">📈 Visualisations</div>',
                    unsafe_allow_html=True,
                )

                col_g1, col_g2 = st.columns(2)

                with col_g1:
                    # Graphique de distribution
                    fig_dist = generer_graphique_distribution(y_pred, y_proba)
                    st.pyplot(fig_dist)
                    plt.close(fig_dist)

                with col_g2:
                    # Graphique d'importance si disponible
                    fig_importance = generer_graphique_importance(
                        modele,
                        noms_colonnes=list(df.columns),
                    )
                    if fig_importance is not None:
                        st.pyplot(fig_importance)
                        plt.close(fig_importance)
                    else:
                        st.info(
                            "ℹ️ L'importance des caractéristiques n'est pas "
                            "disponible pour ce type de modèle."
                        )

                # ─── Résultats détaillés ──────────────────────────────────
                st.markdown(
                    '<div class="section-header">📋 Résultats détaillés</div>',
                    unsafe_allow_html=True,
                )

                # DataFrame avec les prédictions
                df_resultats = df.copy()
                df_resultats["Prédiction"] = [
                    "🔴 Attaque" if p == 1 else "🟢 Normal"
                    for p in y_pred
                ]

                # Ajout des probabilités si disponibles
                if y_proba is not None:
                    df_resultats["Confiance Normal"] = [
                        f"{p[0]:.2%}" for p in y_proba
                    ]
                    df_resultats["Confiance Attaque"] = [
                        f"{p[1]:.2%}" for p in y_proba
                    ]

                st.dataframe(df_resultats, use_container_width=True)
                st.caption(
                    f"Tableau complet des prédictions ({len(df_resultats)} lignes)"
                )

                # Bouton de téléchargement
                csv_resultats = df_resultats.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 Télécharger les résultats (CSV)",
                    data=csv_resultats,
                    file_name="predictions_netguard.csv",
                    mime="text/csv",
                )

            except Exception as e:
                st.error(f"❌ Erreur lors de la prédiction : {e}")
                st.info(
                    "💡 Vérifiez que les colonnes du fichier correspondent "
                    "à celles attendues par le modèle."
                )

elif modele_charge and not donnees_chargees:
    # Modèle chargé mais pas de données
    st.info("📁 Chargez un fichier CSV dans la barre latérale pour commencer l'analyse.")

elif not modele_charge and donnees_chargees:
    # Données chargées mais pas de modèle
    st.info("📦 Chargez un modèle entraîné dans la barre latérale pour commencer l'analyse.")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  PIED DE PAGE                                                         ║
# ╚══════════════════════════════════════════════════════════════════════════╝

st.markdown("""
<div class="footer">
    <b>NetGuard AI</b> — Projet universitaire de Master<br>
    Détection d'intrusions réseau par Machine Learning<br>
    <span style="font-size: 0.8rem;">© 2026 du-dev</span>
</div>
""", unsafe_allow_html=True)
