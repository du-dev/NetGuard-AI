# 🛡️ NetGuard AI

**Système intelligent de détection d'intrusions réseau basé sur le Machine Learning**

NetGuard AI est un projet universitaire de Master qui utilise l'apprentissage supervisé pour distinguer le trafic réseau normal des attaques informatiques.

---

## 📋 Table des matières

- [Architecture](#-architecture)
- [Fonctionnalités](#-fonctionnalités)
- [Installation](#-installation)
- [Utilisation](#-utilisation)
- [Structure du projet](#-structure-du-projet)
- [Modèles disponibles](#-modèles-disponibles)
- [Jeu de données](#-jeu-de-données)
- [Métriques](#-métriques-dévaluation)
- [Démonstration](#-démonstration)
- [Technologies](#-technologies)
- [License](#-license)

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────┐
│                   main.py / demo.py                   │
├─────────────────────────────────────────────────────┤
│                    src/                               │
├──────────┬──────────┬──────────┬────────────────────┤
│ config   │ preproc  │ features │   models           │
│          │ .essing  │          │                     │
│ Paramètres│ Charger │ Extraire │ Entraîner &         │
│ globaux  │ & nett. │ & sélect.│ Prédire             │
├──────────┴──────────┴──────────┴────────────────────┤
│                  evaluation/                         │
│           Métriques, matrice de confusion            │
├─────────────────────────────────────────────────────┤
│                   utils/                             │
│           Logging, chronométrage, sauvegarde         │
└─────────────────────────────────────────────────────┘
```

Le pipeline suit 4 étapes principales :

1. **Prétraitement** : Chargement, nettoyage, normalisation
2. **Caractéristiques** : Sélection ou réduction de dimensionnalité
3. **Modèle** : Entraînement supervisé (Random Forest, SVM, KNN, Gradient Boosting)
4. **Évaluation** : Métriques, matrice de confusion, rapport détaillé

---

## ✨ Fonctionnalités

- 🔍 **Détection binaire** : trafic normal (0) vs attaque (1)
- 🤖 **4 modèles de ML** : Random Forest, SVM, KNN, Gradient Boosting
- 📊 **Sélection de caractéristiques** : ANOVA F-test, PCA
- 📈 **Évaluation complète** : Accuracy, Précision, Rappel, F1-Score
- 📉 **Matrice de confusion** avec visualisation
- 🎯 **Taux spécifiques** : détection, faux positifs, etc.
- 💾 **Sauvegarde et chargement** des modèles entraînés
- 🧪 **Données synthétiques** : démonstration sans dataset réel
- 📝 **Interface CLI** : arguments en ligne de commande

---

## 📦 Installation

### Prérequis

- Python 3.9 ou supérieur
- pip

### Procédure

```bash
# 1. Cloner le dépôt
git clone https://github.com/du-dev/NetGuard-AI.git
cd NetGuard-AI

# 2. Créer un environnement virtuel
python -m venv venv

# 3. Activer l'environnement virtuel
# Windows :
venv\Scripts\activate
# macOS / Linux :
source venv/bin/activate

# 4. Installer les dépendances
pip install -r requirements.txt
```

---

## 🚀 Utilisation

### Pipeline complet

```bash
python main.py
```

### Avec un modèle spécifique

```bash
python main.py --modele svm
python main.py --modele gradient_boosting
```

### Avec sélection de caractéristiques

```bash
# Sélection des 20 meilleures caractéristiques (défaut)
python main.py --selection-caracs k_best --n-caracteristiques 20

# Réduction PCA
python main.py --selection-caracs pca --n-caracteristiques 10

# Pas de sélection
python main.py --selection-caracs aucun
```

### Mode verbose

```bash
python main.py --verbose
```

### Démonstration avec données synthétiques

```bash
python demo.py
```

### Prédiction sur de nouvelles données

```bash
python main.py --predict --fichier mes_donnees.csv
```

---

## 📁 Structure du projet

```
NetGuard-AI/
│
├── data/
│   ├── raw/              # Jeux de données bruts (CSV)
│   └── processed/        # Modèles et métriques sauvegardés
│
├── notebooks/            # Notebooks Jupyter (exploration)
│
├── src/                  # Code source principal
│   ├── __init__.py
│   ├── config.py         # Configuration centralisée
│   ├── preprocessing/    # Chargement et nettoyage
│   │   └── data_loader.py
│   ├── features/         # Extraction de caractéristiques
│   │   └── feature_extractor.py
│   ├── models/           # Modèles ML
│   │   └── detector.py
│   ├── evaluation/       # Métriques et évaluation
│   │   └── metrics.py
│   └── utils/            # Utilitaires (logging, sauvegarde)
│       └── helpers.py
│
├── tests/                # Tests unitaires
│   └── test_detector.py
│
├── main.py               # Point d'entrée CLI
├── demo.py               # Démonstration avec données synthétiques
│
├── requirements.txt      # Dépendances Python
├── .gitignore
├── LICENSE               # Licence MIT
└── README.md
```

---

## 🤖 Modèles disponibles

| Modèle | Avantages | Inconvénients | Usage |
|--------|-----------|---------------|-------|
| **Random Forest** | Robuste, interprétable, parallélisable | Peut être lent avec beaucoup d'arbres | Défaut |
| **SVM** | Efficace en haute dimension | Long à entraîner, pas de probas | Petits datasets |
| **KNN** | Simple, non paramétrique | Lent en prédiction, sensible aux échelles | Exploration |
| **Gradient Boosting** | Très précis, gestion des non-linéarités | Peut surapprendre, plus d'hyperparamètres | Performance max |

---

## 📊 Jeu de données

Le projet attend un fichier CSV dans `data/raw/network_traffic.csv` avec :

- Des caractéristiques numériques nommées `feature_01`, `feature_02`, ...
- Une colonne `label` (0 = trafic normal, 1 = attaque)

Si vous ne disposez pas de dataset réel, utilisez `demo.py` pour générer des données synthétiques.

---

## 📈 Métriques d'évaluation

- **Accuracy** : Proportion de prédictions correctes
- **Précision** : Parmi les attaques prédites, combien sont réelles
- **Rappel (Recall)** : Parmi les attaques réelles, combien sont détectées
- **F1-Score** : Moyenne harmonique précision/rappel
- **Matrice de confusion** : VN / FP / FN / VP
- **Taux de détection** : VP / (VP + FN)
- **Taux de faux positifs** : FP / (FP + VN)

---

## 🧪 Démonstration

```bash
python demo.py
```

Ce script génère 2000 échantillons synthétiques et exécute le pipeline complet :

1. Génération aléatoire de trafic normal et d'attaques
2. Prétraitement et normalisation
3. Sélection des 15 meilleures caractéristiques
4. Entraînement d'un Random Forest
5. Évaluation complète avec métriques
6. Affichage détaillé des résultats

---

## 🛠️ Technologies

- **Python 3.9+** - Langage principal
- **NumPy / Pandas** - Manipulation des données
- **Scikit-learn** - Machine Learning (modèles, preprocessing, métriques)
- **Matplotlib / Seaborn** - Visualisation
- **Pytest** - Tests unitaires
- **Joblib** - Sauvegarde des modèles

---

## 📄 License

Ce projet est sous licence MIT. Voir le fichier [LICENSE](LICENSE).

---

## 👨‍💻 Auteur

Projet universitaire réalisé par **du-dev**.

---

*NetGuard AI — Détection d'intrusions réseau par Machine Learning*
