"""
Tests unitaires pour NetGuard AI.

Teste les 6 modèles de Machine Learning :
- Decision Tree
- Random Forest
- Logistic Regression
- SVM
- KNN
- Naive Bayes

Ainsi que la fabrique de modèles, la prédiction et l'évaluation.
"""

import numpy as np
import pytest

from src.models.detector import (
    DetecteurIntrusions,
    creer_modele,
    FABRIQUE_MODELES,
    MODELES_AVEC_PROBA,
)
from src.evaluation.metrics import Evaluateur
from src.config import LISTE_MODELES


# ─── Tests de la fabrique de modèles ────────────────────────────────────────

class TestFabriqueModeles:
    """Tests pour la fonction creer_modele()."""

    def test_tous_les_modeles_disponibles(self):
        """Vérifie que les 6 modèles sont dans la fabrique."""
        assert set(LISTE_MODELES) == set(FABRIQUE_MODELES.keys())
        assert len(FABRIQUE_MODELES) == 6

    def test_creer_decision_tree(self):
        """Teste la création d'un Decision Tree."""
        from sklearn.tree import DecisionTreeClassifier
        modele = creer_modele("decision_tree")
        assert isinstance(modele, DecisionTreeClassifier)
        assert modele.max_depth == 20

    def test_creer_random_forest(self):
        """Teste la création d'un Random Forest."""
        from sklearn.ensemble import RandomForestClassifier
        modele = creer_modele("random_forest")
        assert isinstance(modele, RandomForestClassifier)
        assert modele.n_estimators == 100

    def test_creer_logistic_regression(self):
        """Teste la création d'une Logistic Regression."""
        from sklearn.linear_model import LogisticRegression
        modele = creer_modele("logistic_regression")
        assert isinstance(modele, LogisticRegression)
        assert modele.C == 1.0

    def test_creer_svm(self):
        """Teste la création d'un SVM."""
        from sklearn.svm import SVC
        modele = creer_modele("svm")
        assert isinstance(modele, SVC)
        assert modele.kernel == "rbf"
        assert modele.probability == True  # noqa: E712

    def test_creer_knn(self):
        """Teste la création d'un KNN."""
        from sklearn.neighbors import KNeighborsClassifier
        modele = creer_modele("knn")
        assert isinstance(modele, KNeighborsClassifier)
        assert modele.n_neighbors == 5

    def test_creer_naive_bayes(self):
        """Teste la création d'un Naive Bayes."""
        from sklearn.naive_bayes import GaussianNB
        modele = creer_modele("naive_bayes")
        assert isinstance(modele, GaussianNB)

    def test_modele_inconnu(self):
        """Vérifie qu'une erreur est levée pour un modèle inconnu."""
        with pytest.raises(ValueError, match="Modèle inconnu"):
            creer_modele("modele_inexistant")

    def test_params_supplementaires(self):
        """Vérifie la fusion des paramètres supplémentaires."""
        from sklearn.ensemble import RandomForestClassifier
        modele = creer_modele("random_forest", n_estimators=50, max_depth=10)
        assert isinstance(modele, RandomForestClassifier)
        assert modele.n_estimators == 50
        assert modele.max_depth == 10


# ─── Tests du DetecteurIntrusions ───────────────────────────────────────────

class TestDetecteurIntrusions:
    """Tests pour la classe DetecteurIntrusions avec tous les modèles."""

    @pytest.fixture(params=LISTE_MODELES)
    def detecteur(self, request):
        """Fixture paramétrée : un détecteur pour chaque modèle."""
        return DetecteurIntrusions(type_modele=request.param)

    def test_initialisation_modele(self, detecteur):
        """Vérifie que chaque modèle s'initialise correctement."""
        assert detecteur.modele is not None
        assert detecteur.type_modele in LISTE_MODELES
        assert not detecteur.modele_entraine

    def test_description_modele(self, detecteur):
        """Vérifie qu'une description existe pour chaque modèle."""
        description = detecteur.obtenir_description()
        assert isinstance(description, str)
        assert len(description) > 0

    def test_entrainement_et_prediction(self, detecteur):
        """Teste l'entraînement et la prédiction pour chaque modèle."""
        n_echantillons = 200
        np.random.seed(42)
        X_train = np.random.randn(n_echantillons, 5)
        y_train = np.random.randint(0, 2, n_echantillons)

        # Entraînement
        resultats = detecteur.entrainer(X_train, y_train)
        assert detecteur.modele_entraine
        assert "score_entrainement" in resultats

        # Prédiction
        X_test = np.random.randn(30, 5)
        predictions = detecteur.predire(X_test)
        assert len(predictions) == 30
        assert all(p in [0, 1] for p in predictions)

    def test_predict_proba(self, detecteur):
        """Teste predict_proba pour les modèles qui le supportent."""
        n_echantillons = 100
        np.random.seed(42)
        X_train = np.random.randn(n_echantillons, 5)
        y_train = np.random.randint(0, 2, n_echantillons)

        detecteur.entrainer(X_train, y_train)
        X_test = np.random.randn(10, 5)
        probabilites = detecteur.predire_proba(X_test)

        assert probabilites.shape == (10, 2)
        assert np.all(probabilites >= 0) and np.all(probabilites <= 1)

    def test_predict_proba_modele_sans_proba(self):
        """Teste le fallback predict_proba pour SVM."""
        detecteur = DetecteurIntrusions(type_modele="svm")
        n_echantillons = 100
        np.random.seed(42)
        X_train = np.random.randn(n_echantillons, 5)
        y_train = np.random.randint(0, 2, n_echantillons)
        detecteur.entrainer(X_train, y_train)

        X_test = np.random.randn(10, 5)
        probabilites = detecteur.predire_proba(X_test)
        assert probabilites.shape == (10, 2)

    def test_predire_sans_entrainement(self, detecteur):
        """Vérifie l'erreur si prédiction sans entraînement."""
        X_test = np.random.randn(10, 5)
        with pytest.raises(ValueError, match="pas encore entraîné"):
            detecteur.predire(X_test)

    def test_obtenir_parametres(self, detecteur):
        """Vérifie le retour des hyperparamètres."""
        parametres = detecteur.obtenir_parametres()
        assert isinstance(parametres, dict)
        # Tout modèle sklearn retourne au moins un paramètre
        assert len(parametres) >= 1

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

    def test_entrainement_avec_validation(self, detecteur):
        """Teste l'entraînement avec validation."""
        n_echantillons = 150
        np.random.seed(42)
        X_train = np.random.randn(100, 5)
        y_train = np.random.randint(0, 2, 100)
        X_val = np.random.randn(50, 5)
        y_val = np.random.randint(0, 2, 50)

        resultats = detecteur.entrainer(X_train, y_train, X_val, y_val)
        assert "score_entrainement" in resultats
        assert "score_validation" in resultats


# ─── Tests du modeles avec probabilites supportees ─────────────────────────

class TestModelesAvecProba:
    """Vérifie que seuls les bons modèles sont dans MODELES_AVEC_PROBA."""

    def test_svm_sans_proba(self):
        """SVM ne devrait PAS être dans MODELES_AVEC_PROBA."""
        assert "svm" not in MODELES_AVEC_PROBA

    def test_random_forest_avec_proba(self):
        """Random Forest devrait être dans MODELES_AVEC_PROBA."""
        assert "random_forest" in MODELES_AVEC_PROBA

    def test_decision_tree_avec_proba(self):
        """Decision Tree devrait être dans MODELES_AVEC_PROBA."""
        assert "decision_tree" in MODELES_AVEC_PROBA

    def test_logistic_regression_avec_proba(self):
        """Logistic Regression devrait être dans MODELES_AVEC_PROBA."""
        assert "logistic_regression" in MODELES_AVEC_PROBA

    def test_knn_avec_proba(self):
        """KNN devrait être dans MODELES_AVEC_PROBA."""
        assert "knn" in MODELES_AVEC_PROBA

    def test_naive_bayes_avec_proba(self):
        """Naive Bayes devrait être dans MODELES_AVEC_PROBA."""
        assert "naive_bayes" in MODELES_AVEC_PROBA


# ─── Tests d'intégration : fabrique + détecteur ────────────────────────────

class TestIntegrationModeleDetecteur:
    """Vérifie que DetecteurIntrusions utilise correctement la fabrique."""

    @pytest.mark.parametrize("nom_modele", LISTE_MODELES)
    def test_detecteur_utilise_fabrique(self, nom_modele):
        """Vérifie que le détecteur initialise via creer_modele()."""
        detecteur = DetecteurIntrusions(type_modele=nom_modele)
        # Le modèle doit être de la bonne classe
        modele_test = creer_modele(nom_modele)
        assert type(detecteur.modele) == type(modele_test)


# ─── Tests de l'évaluateur (inchangés) ──────────────────────────────────────

class TestEvaluateur:
    """Tests pour l'évaluation des métriques."""

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
        assert matrice[0, 0] == 2
        assert matrice[1, 1] == 2

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

        assert abs(taux["taux_vrais_positifs"] - 0.9677) < 0.01
        assert abs(taux["taux_faux_positifs"] - 0.0909) < 0.01
        assert abs(taux["taux_vrais_negatifs"] - 0.9091) < 0.01
        assert abs(taux["taux_faux_negatifs"] - 0.0323) < 0.01
