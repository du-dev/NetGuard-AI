# 🔧 Guide développeur — NetGuard AI

Architecture, conception et extension du système de détection d'intrusions réseau.

---

## 🏗️ Architecture générale

```
                    ┌─────────────────────────┐
                    │   Interface utilisateur    │
                    │ CLI │ Streamlit │ Démo     │
                    └──────────┬────────────────┘
                               │
         ┌────────────────────┼────────────────────┐
         ▼                    ▼                    ▼
   ┌──────────┐        ┌──────────┐        ┌──────────────┐
   │Prétrai-  │        │ Modèles  │        │ Évaluation   │
   │tement    │───────►│ 6 algo.  │───────►│ Métriques    │
   │8 étapes  │        │ Fabrique │        │ + Graphiques │
   └──────────┘        └──────────┘        └──────────────┘
```

### Flux des données

1. **Données brutes** (CSV) → `src/preprocessing/`
2. **Données nettoyées** → `src/features/`
3. **Caractéristiques sélectionnées** → `src/models/`
4. **Modèle entraîné** → `src/evaluation/`
5. **Résultats** → `data/processed/`

---

## 📁 Structure du code source

```
src/
├── __init__.py
├── config.py                    # Configuration centralisée
│
├── preprocessing/               # Étape 1 : Préparation des données
│   ├── __init__.py
│   ├── data_loader.py           #   ChargeurDonnees
│   ├── pipeline.py              #   PipelinePretraitement (8 étapes)
│   └── run_pipeline.py          #   Script CLI du pipeline
│
├── features/                    # Étape 2 : Caractéristiques
│   ├── __init__.py
│   └── feature_extractor.py     #   ANOVA F-test, PCA
│
├── models/                      # Étape 3 : Modèles ML
│   ├── __init__.py
│   ├── detector.py              #   6 modèles + DetecteurIntrusions
│   └── compare.py               #   Comparaison des modèles
│
├── evaluation/                  # Étape 4 : Évaluation
│   ├── __init__.py
│   └── metrics.py               #   Evaluateur (métriques + matrice)
│
└── utils/                       # Utilitaires
    ├── __init__.py
    └── helpers.py               #   Logging, chrono, sauvegarde
```

---

## 🧩 Modules détaillés

### 1. `src/config.py` — Configuration centralisée

Contient tous les paramètres modifiables du projet :

| Variable | Description | Valeur par défaut |
|----------|-------------|------------------|
| `SEED_ALEATOIRE` | Seed pour la reproductibilité | `42` |
| `TYPE_MODELE` | Modèle par défaut | `random_forest` |
| `LISTE_MODELES` | Tous les modèles disponibles | 6 modèles |
| `FABRIQUE_MODELES` | Mapping nom → (classe, params) | Dictionnaire |
| `TAILLE_TEST` | Ratio train/test | `0.3` |
| `DOSSIER_DATA_BRUT` | Dossier données brutes | `data/raw/` |
| `DOSSIER_DATA_TRANSFORME` | Dossier données traitées | `data/processed/` |

**Exemple :** Pour ajouter un paramètre global, modifiez `config.py`.

### 2. `src/preprocessing/pipeline.py` — Pipeline de prétraitement

**Classe : `PipelinePretraitement`**

8 étapes automatisées :

| Étape | Méthode | Description |
|:-----:|---------|-------------|
| 1 | `charger_donnees()` | Chargement du CSV |
| 2 | `separer_caracteristiques_label()` | Séparation X/y |
| 3 | `nettoyer_donnees()` | Suppression des valeurs manquantes |
| 4 | `encoder_variables_categorielles()` | Encodage One-Hot |
| 5 | `standardiser_caracteristiques()` | StandardScaler (μ=0, σ=1) |
| 6 | `reduire_dimension()` | Sélection des caractéristiques |
| 7 | `diviser_train_test()` | Split 70/30 |
| 8 | `executer_pipeline_complet()` | Orchestrateur |

### 3. `src/models/detector.py` — Détecteur d'intrusions

**Classe : `DetecteurIntrusions`**

Fabrique de 6 modèles :

```python
FABRIQUE_MODELES = {
    "decision_tree":     (DecisionTreeClassifier, {"random_state": 42}),
    "random_forest":     (RandomForestClassifier, {"random_state": 42, "n_estimators": 100}),
    "logistic_regression": (LogisticRegression, {"random_state": 42, "max_iter": 1000}),
    "svm":               (SVC, {"random_state": 42, "probability": True}),
    "knn":               (KNeighborsClassifier, {"n_neighbors": 5}),
    "naive_bayes":       (GaussianNB, {}),
}
```

**Méthodes principales :**

| Méthode | Description |
|---------|-------------|
| `entrainer(X_train, y_train)` | Entraîne le modèle |
| `predire(X)` | Prédit sur de nouvelles données |
| `obtenir_parametres()` | Retourne les hyperparamètres |
| `sauvegarder(chemin)` | Sauvegarde au format joblib |
| `charger(chemin)` | Charge un modèle joblib |

### 4. `src/evaluation/metrics.py` — Évaluateur

**Classe : `Evaluateur`**

| Méthode | Description |
|---------|-------------|
| `calculer_metriques(y_test, y_pred)` | Accuracy, Precision, Recall, F1 |
| `calculer_matrice_confusion(y_test, y_pred)` | Matrice 2x2 |
| `generer_rapport(y_test, y_pred)` | Rapport classification complet |
| `calculer_taux_detection(matrice)` | Taux VP, FP (cybersécurité) |

### 5. `src/features/feature_extractor.py` — Extraction

**Classe : `ExtracteurCaracteristiques`**

| Méthode | Description |
|---------|-------------|
| `k_best` | Sélection ANOVA F-test (top K caractéristiques) |
| `pca` | Analyse en Composantes Principales |
| `aucun` | Pas de sélection |

---

## ➕ Ajouter un nouveau modèle

### Étapes

1. **Ajouter le modèle dans `src/config.py`**
   ```python
   LISTE_MODELES = [
       "decision_tree",
       "random_forest",
       "logistic_regression",
       "svm",
       "knn",
       "naive_bayes",
       "gradient_boosting",       # 👈 Nouveau
   ]

   FABRIQUE_MODELES["gradient_boosting"] = (
       GradientBoostingClassifier,
       {"random_state": 42, "n_estimators": 100},
   )
   DESCRIPTION_MODELES["gradient_boosting"] = "Gradient Boosting"
   ```

2. **Ajouter une description dans le README** (tableau des modèles)

3. **Écrire un test** dans `tests/test_detector.py`

### Test du nouveau modèle

```python
def test_gradient_boosting():
    """Teste le modèle Gradient Boosting."""
    detecteur = DetecteurIntrusions(type_modele="gradient_boosting")
    X_train, y_train = generer_donnees_test()
    resultats = detecteur.entrainer(X_train, y_train)
    assert resultats["succes"] is True
    assert detecteur.modele is not None
```

---

## 🧪 Tests

```bash
# Lancer tous les tests
python -m pytest tests/ -v

# Avec couverture
python -m pytest tests/ --cov=src -v
```

### Structure des tests

```
tests/
├── __init__.py
├── test_detector.py      # 6 modèles × initialisation, entraînement, prédiction
└── test_pipeline.py      # Pipeline prétraitement : 8 étapes, cas limites
```

---

## 📦 Ajouter un dataset

1. **Télécharger ou placer** le fichier CSV dans `data/raw/`
2. **Configurer** dans `src/config.py` :
   ```python
   DATASETS_DISPONIBLES["mon_dataset"] = {
       "nom": "Mon Dataset",
       "description": "Description du dataset",
       "chemin": DOSSIER_DATA_BRUT / "mon_fichier.csv",
       "colonne_label": "label",
       "colonnes_a_ignorer": ["id", "timestamp"],
   }
   ```
3. **Tester** : `python train.py --dataset mon_dataset`

---

## 🐳 Déploiement

### Créer un exécutable (PyInstaller)

```bash
pip install pyinstaller
pyinstaller --onefile app.py
```

### API REST (Flask / FastAPI)

Le module `DetecteurIntrusions` peut être intégré dans une API REST :

```python
from flask import Flask, request, jsonify
from src.models.detector import DetecteurIntrusions

app = Flask(__name__)
detecteur = DetecteurIntrusions(type_modele="random_forest")

@app.route("/predict", methods=["POST"])
def predict():
    donnees = request.json
    prediction = detecteur.predire([donnees])
    return jsonify({"prediction": int(prediction[0])})
```

---

## 📈 Améliorations possibles

| Amélioration | Fichier concerné | Difficulté |
|-------------|-----------------|:----------:|
| Deep Learning (MLP, LSTM) | `src/models/detector.py` | 🔴 Difficile |
| Optimisation des hyperparamètres (GridSearch) | `src/models/` | 🟡 Moyenne |
| Nouveau dataset réel | `src/config.py` | 🟢 Facile |
| Interface web enrichie | `app.py` | 🟡 Moyenne |
| API REST | Nouveau fichier | 🟡 Moyenne |
| Docker | Nouveau fichier | 🟡 Moyenne |
| CI/CD (GitHub Actions) | Nouveau fichier | 🟡 Moyenne |

---

## 🔗 Dépendances

| Paquet | Version mini | Utilisation |
|--------|:-----------:|-------------|
| Python | 3.9 | Langage |
| numpy | 1.24 | Calcul numérique |
| pandas | 1.5 | Manipulation données |
| scikit-learn | 1.2 | ML (modèles, preprocessing, métriques) |
| streamlit | 1.28 | Interface web |
| matplotlib | 3.7 | Graphiques |
| seaborn | 0.12 | Graphiques avancés |
| joblib | 1.2 | Sauvegarde modèles |
| pytest | 7.4 | Tests unitaires |

---

## 📄 License

Projet sous licence MIT — voir le fichier `LICENSE`.

---

*NetGuard AI — Détection d'intrusions réseau par Machine Learning*
