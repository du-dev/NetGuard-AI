# 📖 Guide utilisateur — NetGuard AI

Système intelligent de détection d'intrusions réseau basé sur le Machine Learning.

---

## 📦 Installation

### Prérequis
- **Python 3.9** ou supérieur
- **pip** (gestionnaire de paquets Python)
- **Git** (pour cloner le dépôt)

### Procédure pas à pas

```bash
# 1. Cloner le dépôt
git clone https://github.com/du-dev/NetGuard-AI.git
cd NetGuard-AI

# 2. Créer un environnement virtuel
python -m venv venv

# 3. Activer l'environnement virtuel
# Sur Windows :
venv\Scripts\activate
# Sur macOS / Linux :
source venv/bin/activate

# 4. Installer les dépendances
pip install -r requirements.txt
```

> ⚠️ **Important** : Toujours activer l'environnement virtuel avant d'utiliser NetGuard AI.

---

## 🚀 Utilisation rapide

La manière la plus simple de tester NetGuard AI est de lancer la démonstration automatique :

```bash
python demo.py
```

Cette commande va :
1. Générer un dataset synthétique de 2000 échantillons
2. Prétraiter les données (8 étapes automatiques)
3. Entraîner un modèle Random Forest
4. Afficher les résultats (Accuracy, F1-Score, matrice de confusion)

---

## 🎯 Pipelines disponibles

### Pipeline d'entraînement complet

Compare les 6 modèles et sélectionne automatiquement le meilleur :

```bash
python train.py --sauvegarder
```

**Options principales :**

| Option | Description | Défaut |
|--------|-------------|--------|
| `--dataset` | Dataset à utiliser (`synthetique`, `cicids2017`) | `synthetique` |
| `--modele` | Modèle spécifique à entraîner | Tous les modèles |
| `--sauvegarder` | Sauvegarder le meilleur modèle | Non |
| `--taille-test` | Proportion des données pour le test | `0.3` |
| `--verbose` | Logs détaillés | Non |
| `--download` | Téléchargement du dataset si absent | Non |
| `--no-compare` | Désactiver la comparaison | Non |

**Exemples :**

```bash
# Un seul modèle avec logs détaillés
python train.py --modele random_forest --verbose

# Avec le dataset CICIDS2017
python train.py --dataset cicids2017 --sauvegarder

# Téléchargement automatique + entraînement
python train.py --dataset cicids2017 --download --sauvegarder
```

### Pipeline simple (CLI)

```bash
# Pipeline complet avec les réglages par défaut
python main.py

# Avec un modèle spécifique
python main.py --modele svm

# Avec sélection de caractéristiques PCA
python main.py --selection-caracs pca --n-caracteristiques 10

# Comparer tous les modèles
python main.py --compare
```

**Options principales :**

| Option | Description | Défaut |
|--------|-------------|--------|
| `--modele` | Modèle à utiliser | `random_forest` |
| `--selection-caracs` | Méthode de sélection (`k_best`, `pca`, `aucun`) | `k_best` |
| `--predict` | Mode prédiction sur un fichier | Non |
| `--fichier` | Fichier CSV à analyser (mode predict) | — |

### Prédiction sur de nouvelles données

```bash
# 1. D'abord, entraîner un modèle
python train.py --sauvegarder

# 2. Prédire sur un fichier CSV
python main.py --predict --fichier mes_donnees.csv
```

---

## 🖥️ Interface web Streamlit

L'interface graphique permet d'utiliser NetGuard AI sans ligne de commande.

### Lancement

```bash
streamlit run app.py
```

### Fonctionnalités

1. **Barre latérale — Chargement du modèle**
   - Sélectionnez un modèle entraîné parmi ceux disponibles
   - Cliquez sur "Charger le modèle"
   - Si aucun modèle n'existe, cliquez sur "Entraîner un modèle de démo"

2. **Barre latérale — Chargement des données**
   - Uploadez un fichier CSV (format : colonnes numériques)
   - Le fichier peut optionnellement contenir une colonne `label` pour comparer les performances

3. **Page principale — Analyse**
   - Visualisez l'aperçu des données chargées
   - Lancez la prédiction
   - Consultez les résultats :

| Élément | Description |
|---------|-------------|
| 🟢 / 🔴 | Statut de chaque échantillon (Normal / Attaque) |
| 📈 Métriques | Accuracy, Précision, Rappel, F1-Score |
| 🔒 Taux de détection | Pourcentage d'attaques détectées |
| 📊 Graphiques | Matrice de confusion, distribution, importance |
| 📋 Tableau | Résultats complets téléchargeables en CSV |

---

## 🧪 Tests unitaires

```bash
# Lancer tous les tests
python -m pytest tests/ -v

# Tester un module spécifique
python -m pytest tests/test_detector.py -v
python -m pytest tests/test_pipeline.py -v
```

93 tests unitaires valident le bon fonctionnement du projet.

---

## 📁 Structure des données

```
data/
├── raw/                    # Données brutes (CSV)
│   ├── network_traffic.csv # Dataset synthétique
│   ├── sample_evaluate.csv # Exemple pour évaluation
│   └── sample_predict.csv  # Exemple pour prédiction
│
└── processed/              # Données traitées et modèles
    ├── meilleur_modele.joblib      # Meilleur modèle (joblib)
    ├── meilleur_modele_metriques.txt # Métriques du meilleur modèle
    ├── modele_demo.joblib          # Modèle de démonstration
    ├── standard_scaler.joblib      # Normalisateur sauvegardé
    ├── rapport_entrainement.txt    # Rapport complet d'entraînement
    ├── netguard_clean_X_train.csv  # Caractéristiques d'entraînement
    ├── netguard_clean_X_test.csv   # Caractéristiques de test
    ├── netguard_clean_y_train.csv  # Labels d'entraînement
    ├── netguard_clean_y_test.csv   # Labels de test
    └── graphiques/                 # Graphiques générés
        ├── 01_distribution_classes.png
        ├── 02_boxplots_caracteristiques.png
        ├── 03_matrice_correlation.png
        ├── 04_courbes_roc.png
        ├── ...
```

---

## 📊 Résultats attendus (dataset synthétique)

| Modèle | F1-Score | Détection | Fausses alarmes |
|--------|:--------:|:---------:|:---------------:|
| **Random Forest** 🥇 | **1.0000** | **100%** | **0%** |
| Logistic Regression 🥈 | 1.0000 | 100% | 0% |
| SVM 🥉 | 1.0000 | 100% | 0% |
| Naive Bayes | 1.0000 | 100% | 0% |
| KNN | 0.9972 | 99.44% | 0% |
| Decision Tree | 0.9060 | 88.33% | 6% |

---

## ❓ FAQ / Dépannage

### "Module introuvable" ou "ImportError"
```bash
# Réinstaller les dépendances
pip install -r requirements.txt
```

### "Fichier non trouvé"
```bash
# Vérifier que vous êtes dans le bon dossier
cd chemin/vers/NetGuard-AI

# Générer les données de test
python demo.py
```

### L'interface Streamlit ne se lance pas
```bash
# Vérifier que Streamlit est installé
pip install streamlit

# Lancer avec le bon interpréteur
python -m streamlit run app.py
```

### Les résultats semblent étranges
- Vérifiez que votre fichier CSV a le bon format (colonnes numériques)
- Assurez-vous que le modèle a été entraîné sur des données similaires
- Ré-entraînez le modèle si nécessaire

---

## 📝 Contact

Projet universitaire réalisé par **du-dev**.

- Dépôt GitHub : [https://github.com/du-dev/NetGuard-AI](https://github.com/du-dev/NetGuard-AI)
- Signalez un bug : Ouvrir une *issue* sur GitHub

---

*NetGuard AI — Détection d'intrusions réseau par Machine Learning*
