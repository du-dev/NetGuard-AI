"""
Tests unitaires pour NetGuard AI.

Teste les composants principaux du projet :
- Le chargement et prétraitement des données
- Le modèle de détection
- L'évaluation des performances
"""

import numpy as np
import pytest

from src.models.detector import DetecteurIntrusions
from src.evaluation.metrics import Evaluateur


class TestDetecteurIntrusions:
    """Tests pour le modèle de détection."""

    @pytest.fixture
    def detecteur(self):
        """Fixture : détecteur Random Forest."""
        return DetecteurIntrusions(type_modele="random_forest")

    def test_initialisation_random_forest(self, detecteur):
        """Vérifie l'initialisation du Random Forest."""
        assert detecteur.modele is not None
        assert detecteur.type_modele == "random_forest"
        assert not detecteur.modele_entraine

    def test_initialisation_svm(self):
        """Vérifie l'initialisation du SVM."""
        detecteur = DetecteurIntrusions(type_modele="svm")
        assert detecteur.modele is not None
        assert detecteur.type_modele == "svm"

    def test_initialisation_knn(self):
        """Vérifie l'initialisation du KNN."""
        detecteur = DetecteurIntrusions(type_modele="knn")
        assert detecteur.modele is not None
        assert detecteur.type_modele == "knn"

    def test_entrainement_et_prediction(self, detecteur):
        """Teste l'entraînement et la prédiction."""
        # Données synthétiques simples
        n_echantillons = 100
        np.random.seed(42)
        X_train = np.random.randn(n_echantillons, 5)
        y_train = np.random.randint(0, 2, n_echantillons)

        # Entraînement
        resultats = detecteur.entrainer(X_train, y_train)
        assert detecteur.modele_entraine
        assert "score_entrainement" in resultats
        assert resultats["score_entrainement"] > 0

        # Prédiction
        X_test = np.random.randn(20, 5)
        predictions = detecteur.predire(X_test)
        assert len(predictions) == 20
        assert all(p in [0, 1] for p in predictions)

    def test_predire_sans_entrainement(self, detecteur):
        """Vérifie qu'une erreur est levée si on prédit sans entraînement."""
        X_test = np.random.randn(10, 5)
        with pytest.raises(ValueError, match="pas encore entraîné"):
            detecteur.predire(X_test)

    def test_predire_proba(self, detecteur):
        """Teste la prédiction de probabilités."""
        n_echantillons = 50
        np.random.seed(42)
        X_train = np.random.randn(n_echantillons, 5)
        y_train = np.random.randint(0, 2, n_echantillons)

        detecteur.entrainer(X_train, y_train)
        X_test = np.random.randn(10, 5)
        probabilites = detecteur.predire_proba(X_test)

        assert probabilites.shape == (10, 2)
        # Les probabilités doivent être entre 0 et 1
        assert np.all(probabilites >= 0) and np.all(probabilites <= 1)

    def test_obtenir_parametres(self, detecteur):
        """Vérifie la récupération des paramètres."""
        parametres = detecteur.obtenir_parametres()
        assert isinstance(parametres, dict)
        assert "n_estimators" in parametres

    def test_evaluation(self, detecteur):
        """Teste l'évaluation du modèle."""
        n_echantillons = 100
        np.random.seed(42)
        X_train = np.random.randn(n_echantillons, 5)
        y_train = np.random.randint(0, 2, n_echantillons)

        detecteur.entrainer(X_train, y_train)

        X_test = np.random.randn(30, 5)
        y_test = np.random.randint(0, 2, 30)

        score = detecteur.evaluer(X_test, y_test)
        assert 0 <= score <= 1


class TestEvaluateur:
    """Tests pour l'évaluation."""

    @pytest.fixture
    def evaluateur(self):
        """Fixture : évaluateur."""
        return Evaluateur()

    def test_calculer_metriques_parfait(self, evaluateur):
        """Teste les métriques avec prédictions parfaites."""
        y_reel = np.array([0, 0, 1, 1, 0, 1])
        y_pred = np.array([0, 0, 1, 1, 0, 1])

        metriques = evaluateur.calculer_metriques(y_reel, y_pred)
        assert metriques["accuracy"] == 1.0
        assert metriques["precision"] == 1.0
        assert metriques["recall"] == 1.0
        assert metriques["f1_score"] == 1.0

    def test_calculer_metriques_aleatoires(self, evaluateur):
        """Teste les métriques avec des prédictions aléatoires."""
        np.random.seed(42)
        y_reel = np.random.randint(0, 2, 100)
        y_pred = np.random.randint(0, 2, 100)

        metriques = evaluateur.calculer_metriques(y_reel, y_pred)
        assert 0 <= metriques["accuracy"] <= 1
        assert 0 <= metriques["precision"] <= 1
        assert 0 <= metriques["recall"] <= 1
        assert 0 <= metriques["f1_score"] <= 1

    def test_matrice_confusion(self, evaluateur):
        """Teste la matrice de confusion."""
        y_reel = np.array([0, 0, 0, 1, 1, 1])
        y_pred = np.array([0, 0, 1, 1, 1, 0])

        matrice = evaluateur.calculer_matrice_confusion(y_reel, y_pred)
        assert matrice.shape == (2, 2)
        assert matrice[0, 0] == 2  # Vrais négatifs
        assert matrice[1, 1] == 2  # Vrais positifs

    def test_generer_rapport(self, evaluateur):
        """Teste la génération du rapport."""
        y_reel = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 1, 1])

        rapport = evaluateur.generer_rapport(y_reel, y_pred)
        assert isinstance(rapport, str)
        assert "Normal" in rapport
        assert "Attaque" in rapport

    def test_evaluation_complete(self, evaluateur):
        """Teste l'évaluation complète."""
        np.random.seed(42)
        y_reel = np.random.randint(0, 2, 50)
        y_pred = np.random.randint(0, 2, 50)

        metriques, matrice, rapport = evaluateur.evaluer_complet(y_reel, y_pred)
        assert isinstance(metriques, dict)
        assert isinstance(matrice, np.ndarray)
        assert isinstance(rapport, str)

    def test_taux_detection(self, evaluateur):
        """Teste les taux de détection."""
        matrice = np.array([[100, 10], [5, 150]])
        taux = evaluateur.calculer_taux_detection(matrice)

        # Taux de vrais positifs : 150/(150+5) = 0.9677
        assert abs(taux["taux_vrais_positifs"] - 0.9677) < 0.01
        # Taux de faux positifs : 10/(10+100) = 0.0909
        assert abs(taux["taux_faux_positifs"] - 0.0909) < 0.01
        # Taux de vrais négatifs : 100/(100+10) = 0.9091
        assert abs(taux["taux_vrais_negatifs"] - 0.9091) < 0.01
        # Taux de faux négatifs : 5/(5+150) = 0.0323
        assert abs(taux["taux_faux_negatifs"] - 0.0323) < 0.01
