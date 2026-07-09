# Guide développeur NetGuard AI

Ce guide détaille l'architecture technique, la structure du code et les conventions du projet **NetGuard AI**.

---

## 1. Architecture générale

```
┌──────────────────────────────────────────────────────────┐
│                   Scripts utilisateur                     │
│  main.py │ demo.py │ train.py │ app.py (Streamlit)      │
├──────────────────────────────────────────────────────────┤
│                    Packages src/                          │
│  config ── preprocessing ── features ── models ── eval   │
│                                   │                      │
│                              utils/helpers                │
├──────────────────────────────────────────────────────────┤
│                    Datasets                               │
│  datasets/download_cicids2017.py                          │
├──────────────────────────────────────────────────────────┤
│                    Tests                                  │
│  tests/test_detector.py │ tests/test_pipeline.py          │
└──────────────────────────────────────────────────────────┘
```

### Flux de données

```
CSV → PipelinePretraitement → X_train, X_test, y_train, y_test
                                     ↓
                            DetecteurIntrusions.entrainer()
                                     ↓
                                     ↓
                            Evaluateur (métriques)
                                     ↓
                            Meilleur modèle sauvegardé
```

---

## 2. Structure du projet

```
NetGuard-AI/
│
├── app.py                          # Interface Streamlit
├── main.py                         # CLI principal
├── demo.py                         # Démonstration
├── train.py                        # Pipeline d'entraînement complet
│
├── src/                            # Code source principal
│   ├── config.py                   # Configuration centralisée
│   │
│   ├── preprocessing/
│   │   ├── data_loader.py          # Chargement CSV (ChargeurDonnees)
│   │   ├── pipeline.py             # Pipeline 8 étapes (PipelinePretraitement)
│   │   └── run_pipeline.py         # Script CLI du pipeline
│   │
│   ├── features/
│   │   └── feature_extractor.py    # Sélection (ANOVA, PCA)
│   │
│   ├── models/
│   │   ├── detector.py             # 6 modèles + fabrique
│   │   └── compare.py              # Comparaison des modèles
│   │
│   ├── evaluation/
│   │   └── metrics.py              # Métriques + matrice de confusion
│   │
│   └── utils/
│       └── helpers.py              # Logging, chrono, sauvegarde
│
├── datasets/
│   ├── download_cicids2017.py      # Téléchargement CICIDS2017
│   └── cicids2017_consolide.csv    # Dataset préparé
│
├── data/
│   ├── raw/                        # Données brutes
│   └── processed/                  # Modèles + métriques sauvegardés
│
├── docs/
│   ├── GUIDE_UTILISATEUR.md
│   └── GUIDE_DEVELOPPEUR.md
│
├── tests/
│   ├── test_detector.py            # 93 tests (6 modèles × plusieurs cas)
│   └── test_pipeline.py            # Tests du pipeline de prétraitement
│
├── notebooks/                      # Notebooks Jupyter
├── requirements.txt
├── RAPPORT_FINAL.md
└── README.md
```

---

## 3. Modules détaillés

### 3.1 Configuration (`src/config.py`)

Centralise tous les paramètres :
- Chemins des dossiers
- Hyperparamètres des 6 modèles
- Configuration des datasets supportés
- Seed aléatoire pour la reproductibilité

```python
from src.config import TYPE_MODELE, LISTE_MODELES, DATASETS_DISPONIBLES
```

### 3.2 Pipeline de prétraitement (`src/preprocessing/pipeline.py`)

Classe `PipelinePretraitement` avec 8 étapes :
1. Chargement CSV
2. Suppression colonnes inutiles
3. Suppression doublons
4. Gestion valeurs manquantes (NaN, Infinity)
5. Encodage variables catégorielles (LabelEncoder + One-Hot)
6. Normalisation Z-score (StandardScaler)
7. Séparation train/test stratifiée
8. Sauvegarde des données nettoyées

```python
pipeline = PipelinePretraitement(colonne_label="label")
X_train, X_test, y_train, y_test = pipeline.executer_pipeline_complet()
```

### 3.3 Détection (`src/models/detector.py`)

Fabrique de modèles avec `FABRIQUE_MODELES` supportant 6 modèles :
- Decision Tree, Random Forest, Logistic Regression
- SVM, KNN, Naive Bayes

```python
detecteur = DetecteurIntrusions(type_modele="random_forest")
detecteur.entrainer(X_train, y_train)
predictions = detecteur.predire(X_test)
```

### 3.4 Évaluation (`src/evaluation/metrics.py`)

Calcule automatiquement :
- Accuracy, Precision, Recall, F1-Score
- Matrice de confusion
- Rapport de classification
- Taux de détection (cybersécurité)

### 3.5 Utilitaires (`src/utils/helpers.py`)

- `Chronometre` : Mesure le temps d'exécution
- `sauvegarder_modele` / `charger_modele` : Persistance des modèles
- `configurer_logging` : Configuration des logs

---

## 4. Modèles disponibles

| Modèle | Classe sklearn | Hyperparamètres clés |
|--------|---------------|---------------------|
| Decision Tree | `DecisionTreeClassifier` | max_depth=20, criterion="gini" |
| Random Forest | `RandomForestClassifier` | n_estimators=100, max_depth=20 |
| Logistic Regression | `LogisticRegression` | C=1.0, solver="lbfgs" |
| SVM | `SVC` | kernel="rbf", probability=True |
| KNN | `KNeighborsClassifier` | n_neighbors=5, weights="distance" |
| Naive Bayes | `GaussianNB` | var_smoothing=1e-9 |

---

## 5. Tests

93 tests unitaires couvrent :
- Les 6 modèles (initialisation, entraînement, prédiction, probas)
- L'évaluateur (métriques, matrice, rapport, taux)
- Le pipeline (8 étapes, cas aux limites)

```bash
python -m pytest tests/ -v
```

---

## 6. Bonnes pratiques

- **Commentaires** : Toujours en français
- **Docstrings** : En français, format Google
- **Imports** : Standards Python, puis librairies, puis modules projet
- **Logging** : Utiliser `logging.info()` plutôt que `print()`
- **Modèles** : Passer par la fabrique `creer_modele()` pour toute instanciation

---

## 7. Extension du projet

### Ajouter un nouveau dataset
1. Créer un script de téléchargement dans `datasets/`
2. Ajouter la configuration dans `src/config.py` (`DATASETS_DISPONIBLES`)
3. Le reste du pipeline s'adapte automatiquement

### Ajouter un nouveau modèle
1. Ajouter les paramètres dans `src/config.py`
2. Ajouter le modèle dans `FABRIQUE_MODELES` dans `src/models/detector.py`
3. Ajouter la description dans `DESCRIPTION_MODELES`
4. Ajouter le modèle dans `LISTE_MODELES`

---

*NetGuard AI - Architecture technique*
