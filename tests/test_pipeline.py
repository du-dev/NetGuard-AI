"""
Tests unitaires pour le pipeline de prétraitement NetGuard AI.

Teste chaque étape du pipeline :
- Chargement des données
- Suppression des doublons
- Gestion des valeurs manquantes
- Encodage des variables catégorielles
- Normalisation
- Séparation entraînement/test
- Pipeline complet
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ajout de la racine du projet au chemin pour les imports
PROJET_RACINE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJET_RACINE))

from src.preprocessing.pipeline import PipelinePretraitement


@pytest.fixture
def pipeline():
    """Fixture : pipeline de prétraitement."""
    return PipelinePretraitement(
        colonne_label="label",
        colonnes_a_ignorer=["id", "timestamp"],
    )


@pytest.fixture
def petit_dataset(tmp_path):
    """
    Crée un fichier CSV de test avec assez d'échantillons
    pour permettre la stratification (min 2 par classe).

    Contient :
    - 20 lignes de données numériques
    - Une colonne label binaire (10 normal, 10 attaque)
    - Un doublon (ligne 1 et ligne 10 identiques)
    - Une valeur manquante (NaN en position 5)
    - Une colonne catégorielle
    """
    np.random.seed(42)
    n = 20
    data = {
        "feature_01": np.random.randn(n).tolist(),
        "feature_02": np.random.randn(n).tolist(),
        "feature_03": np.random.randint(0, 100, n).tolist(),
        "categorie": ["A", "B", "A", "C", "B"] * 4,
        "label": ["normal"] * 10 + ["attaque"] * 10,
    }
    # Ajout d'un doublon : ligne 1 = ligne 10
    data["feature_01"][10] = data["feature_01"][1]
    data["feature_02"][10] = data["feature_02"][1]
    data["feature_03"][10] = data["feature_03"][1]
    data["categorie"][10] = data["categorie"][1]
    data["label"][10] = data["label"][1]

    # Ajout d'une valeur manquante
    data["feature_02"][5] = np.nan

    df = pd.DataFrame(data)
    chemin = tmp_path / "test_dataset.csv"
    df.to_csv(chemin, index=False)
    return chemin


class TestChargement:
    """Tests pour l'étape de chargement."""

    def test_charger_fichier_valide(self, pipeline, petit_dataset):
        """Vérifie le chargement d'un fichier CSV valide."""
        df = pipeline.charger(petit_dataset)
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (20, 5)
        assert pipeline.metadonnees["nb_lignes_initial"] == 20
        assert pipeline.metadonnees["nb_colonnes_initial"] == 5

    def test_charger_fichier_inexistant(self, pipeline):
        """Vérifie l'erreur pour un fichier inexistant."""
        with pytest.raises(FileNotFoundError):
            pipeline.charger(Path("fichier_inexistant.csv"))


class TestSuppressionColonnes:
    """Tests pour la suppression des colonnes inutiles."""

    def test_supprimer_colonnes_presentes(self, pipeline, petit_dataset):
        """Vérifie la suppression des colonnes configurées."""
        pipeline.charger(petit_dataset)
        df = pipeline.supprimer_colonnes_inutiles()
        assert "id" not in df.columns
        assert "timestamp" not in df.columns

    def test_supprimer_colonnes_aucune(self):
        """Vérifie qu'aucune colonne n'est supprimée si la liste est vide."""
        p = PipelinePretraitement(colonnes_a_ignorer=[])
        # Création d'un petit dataset sans les colonnes à ignorer
        data = {"a": [1, 2], "label": [0, 1]}
        df = pd.DataFrame(data)
        tmp = Path("data/raw")
        tmp.mkdir(parents=True, exist_ok=True)
        chemin = tmp / "_test_noignore.csv"
        df.to_csv(chemin, index=False)
        try:
            p.charger(chemin)
            df_clean = p.supprimer_colonnes_inutiles()
            assert list(df_clean.columns) == ["a", "label"]
        finally:
            chemin.unlink(missing_ok=True)


class TestDoublons:
    """Tests pour la suppression des doublons."""

    def test_doublons_supprimes(self, pipeline, petit_dataset):
        """Vérifie que les doublons sont correctement supprimés."""
        pipeline.charger(petit_dataset)
        pipeline.supprimer_colonnes_inutiles()
        df = pipeline.supprimer_doublons()

        # La ligne 10 est un doublon de la ligne 1
        assert pipeline.metadonnees["nb_doublons_supprimes"] == 1
        assert len(df) == 19  # 20 - 1 doublon

    def test_aucun_doublon(self):
        """Vérifie qu'aucun doublon n'est détecté."""
        p = PipelinePretraitement(colonnes_a_ignorer=[])
        data = {"a": [1, 2, 3], "label": [0, 1, 0]}
        df = pd.DataFrame(data)
        tmp = Path("data/raw")
        tmp.mkdir(parents=True, exist_ok=True)
        chemin = tmp / "_test_nodup.csv"
        df.to_csv(chemin, index=False)
        try:
            p.charger(chemin)
            p.supprimer_doublons()
            assert p.metadonnees["nb_doublons_supprimes"] == 0
        finally:
            chemin.unlink(missing_ok=True)


class TestValeursManquantes:
    """Tests pour la gestion des valeurs manquantes."""

    def test_valeurs_manquantes_supprimees(self, pipeline, petit_dataset):
        """Vérifie que les lignes avec NaN sont supprimées."""
        pipeline.charger(petit_dataset)
        pipeline.supprimer_colonnes_inutiles()
        pipeline.supprimer_doublons()
        df = pipeline.gerer_valeurs_manquantes()

        # La 3e ligne a feature_02=NaN → doit être supprimée
        assert pipeline.metadonnees["nb_valeurs_manquantes_supprimees"] > 0
        assert not df.isnull().any().any()


class TestEncodage:
    """Tests pour l'encodage des variables catégorielles."""

    def test_encodage_label(self, pipeline, petit_dataset):
        """Vérifie l'encodage de la colonne cible."""
        pipeline.charger(petit_dataset)
        pipeline.supprimer_colonnes_inutiles()
        pipeline.supprimer_doublons()
        pipeline.gerer_valeurs_manquantes()
        X, y = pipeline.encoder_variables_categorielles()

        # Vérification du type
        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)

        # Vérification des classes encodées
        assert len(pipeline.encodage_label.classes_) == 2
        assert "normal" in pipeline.encodage_label.classes_
        assert "attaque" in pipeline.encodage_label.classes_

        # Vérification que y ne contient que 0 et 1
        assert set(y) == {0, 1}

    def test_encodage_categorielles(self, pipeline, petit_dataset):
        """Vérifie l'encodage One-Hot des variables catégorielles."""
        pipeline.charger(petit_dataset)
        pipeline.supprimer_colonnes_inutiles()
        pipeline.supprimer_doublons()
        pipeline.gerer_valeurs_manquantes()
        X, y = pipeline.encoder_variables_categorielles()

        # Vérifie que 'categorie' a été encodée (drop_first=True)
        assert "categorie_B" in pipeline.X.columns or "categorie" not in pipeline.X.columns
        # Les colonnes catégorielles sont encodées
        assert len(pipeline.metadonnees["colonnes_categorielles_encodes"]) > 0


class TestNormalisation:
    """Tests pour la normalisation des données."""

    def test_normalisation_zscore(self, pipeline, petit_dataset):
        """Vérifie que les données sont centrées-réduites."""
        pipeline.charger(petit_dataset)
        pipeline.supprimer_colonnes_inutiles()
        pipeline.supprimer_doublons()
        pipeline.gerer_valeurs_manquantes()
        pipeline.encoder_variables_categorielles()
        X_norm = pipeline.normaliser_donnees()

        # Vérification des propriétés Z-score
        assert abs(X_norm.mean(axis=0).sum()) < 1e-10  # Moyenne ≈ 0
        assert abs(X_norm.std(axis=0).mean() - 1.0) < 0.1  # Écart-type ≈ 1


class TestSeparation:
    """Tests pour la séparation entraînement/test."""

    def test_taille_ensembles(self, pipeline, petit_dataset):
        """Vérifie les proportions train/test."""
        pipeline.charger(petit_dataset)
        pipeline.supprimer_colonnes_inutiles()
        pipeline.supprimer_doublons()
        pipeline.gerer_valeurs_manquantes()
        X, y = pipeline.encoder_variables_categorielles()
        X_norm = pipeline.normaliser_donnees()

        X_train, X_test, y_train, y_test = pipeline.separer_train_test(X_norm, y)

        # Vérification des dimensions
        assert len(X_train) > 0
        assert len(X_test) > 0
        assert len(X_train) + len(X_test) == len(X_norm)

        # Vérification de la stratification
        _, counts_train = np.unique(y_train, return_counts=True)
        _, counts_test = np.unique(y_test, return_counts=True)
        assert len(counts_train) == len(counts_test)

    def test_taille_test_personnalisee(self, pipeline, petit_dataset):
        """Vérifie la taille de test personnalisée."""
        pipeline.taille_test = 0.5
        pipeline.charger(petit_dataset)
        pipeline.supprimer_colonnes_inutiles()
        pipeline.supprimer_doublons()
        pipeline.gerer_valeurs_manquantes()
        X, y = pipeline.encoder_variables_categorielles()
        X_norm = pipeline.normaliser_donnees()
        X_train, X_test, _, _ = pipeline.separer_train_test(X_norm, y)

        total = len(X_train) + len(X_test)
        assert abs(len(X_test) / total - 0.5) < 0.3  # Approximation


class TestPipelineComplet:
    """Tests pour le pipeline complet."""

    def test_pipeline_complet_sans_sauvegarde(self, pipeline, petit_dataset):
        """Vérifie que le pipeline complet s'exécute sans erreur."""
        X_train, X_test, y_train, y_test = pipeline.executer_pipeline_complet(
            chemin_dataset=petit_dataset,
            sauvegarder=False,
        )

        assert isinstance(X_train, np.ndarray)
        assert isinstance(X_test, np.ndarray)
        assert isinstance(y_train, np.ndarray)
        assert isinstance(y_test, np.ndarray)

        assert len(X_train) > 0
        assert len(X_test) > 0
        assert len(X_train) + len(X_test) > 0

    def test_pipeline_complet_avec_sauvegarde(self, pipeline, petit_dataset, tmp_path):
        """Vérifie la sauvegarde des fichiers."""
        X_train, X_test, y_train, y_test = pipeline.executer_pipeline_complet(
            chemin_dataset=petit_dataset,
            sauvegarder=True,
        )

        # Vérifier l'existence des fichiers (chemin par défaut)
        dossier = Path("data/processed")
        if dossier.exists():
            fichiers = list(dossier.glob("netguard_clean_*"))
            assert len(fichiers) >= 4

    def test_metadonnees_remplies(self, pipeline, petit_dataset):
        """Vérifie que les métadonnées sont complètes après le pipeline."""
        pipeline.executer_pipeline_complet(
            chemin_dataset=petit_dataset,
            sauvegarder=False,
        )

        assert pipeline.metadonnees["nb_lignes_initial"] > 0
        assert pipeline.metadonnees["nb_colonnes_initial"] > 0
        assert pipeline.metadonnees["nb_caracteristiques_final"] > 0

    def test_transformations_conservatives(self, pipeline, petit_dataset):
        """Vérifie que les transformateurs sont accessibles après le pipeline."""
        pipeline.executer_pipeline_complet(
            chemin_dataset=petit_dataset,
            sauvegarder=False,
        )

        assert pipeline.encodage_label is not None
        assert pipeline.standardisation is not None

    def test_pipeline_sans_colonnes_a_ignorer(self, petit_dataset):
        """Vérifie le pipeline avec des colonnes à ignorer vides."""
        p = PipelinePretraitement(
            colonne_label="label",
            colonnes_a_ignorer=[],
        )
        X_train, X_test, y_train, y_test = p.executer_pipeline_complet(
            chemin_dataset=petit_dataset,
            sauvegarder=False,
        )
        assert len(X_train) > 0
