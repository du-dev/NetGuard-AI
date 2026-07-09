# 🛡️ NetGuard AI

**Système intelligent de détection d'intrusions réseau basé sur le Machine Learning**

NetGuard AI est un projet universitaire de Master qui utilise l'apprentissage supervisé pour distinguer le trafic réseau normal des attaques informatiques. Six modèles de Machine Learning sont implémentés et comparés : Decision Tree, Random Forest, Logistic Regression, SVM, KNN et Naive Bayes.

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2%2B-orange)](https://scikit-learn.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red)](https://streamlit.io)
[![Tests](https://img.shields.io/badge/Tests-93%20passed-green)]()
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## 📋 Table des matières

- [Présentation](#-présentation)
- [Fonctionnalités](#-fonctionnalités)
- [Architecture](#-architecture)
- [Installation](#-installation)
- [Utilisation](#-utilisation)
- [Interface web](#-interface-web-streamlit)
- [Structure du projet](#-structure-du-projet)
- [Modèles disponibles](#-modèles-disponibles)
- [Résultats](#-résultats)
- [Datasets](#-datasets)
- [Tests](#-tests)
- [Technologies](#-technologies)
- [Documentation](#-documentation)
- [License](#-license)

---

## 🎯 Présentation

NetGuard AI permet de :

1. **Charger et prétraiter** des données de trafic réseau (CSV)
2. **Entraîner 6 modèles** de Machine Learning pour la détection
3. **Comparer automatiquement** les performances des modèles
4. **Sélectionner et sauvegarder** le meilleur modèle
5. **Prédire** sur de nouvelles données via CLI ou interface web

---

## ✨ Fonctionnalités

- 🔍 **6 modèles de ML** : Decision Tree, Random Forest, Logistic Regression, SVM, KNN, Naive Bayes
- 📊 **Pipeline prétraitement** : 8 étapes automatisées (chargement → normalisation → split)
- 📈 **Évaluation complète** : Accuracy, Précision, Rappel, F1-Score, matrice de confusion
- 🤖 **Comparaison automatique** : Classement des modèles par F1-Score
- 💾 **Sauvegarde du meilleur modèle** : Joblib + rapport texte
- 🖥️ **Interface web** : Application Streamlit pour les prédictions visuelles
- 📝 **CLI complète** : Arguments en ligne de commande
- 🧪 **Données synthétiques** : Démonstration sans dataset réel
- 🌐 **CICIDS2017** : Support du dataset réel de référence

---

## 🏗️ Architecture

```
                    ┌─────────────────────┐
                    │   Interface utilisateur  │
                    │ CLI │ Streamlit │ Démo   │
                    └─────────┬───────────┘
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
        ┌──────────┐  ┌──────────┐  ┌──────────────┐
        │Prétrai-  │  │ Modèles  │  │ Évaluation   │
        │tement    │→ │ 6 algo.  │→ │ Métriques    │
        │8 étapes  │  │ Fabrique │  │ + Graphiques │
        └──────────┘  └──────────┘  └──────────────┘
```

Le flux de données suit 4 étapes principales :

1. **Prétraitement** → Chargement, nettoyage, normalisation, split
2. **Caractéristiques** → Sélection ANOVA F-test ou PCA
3. **Modèle** → Entraînement supervisé (1 ou 6 modèles)
4. **Évaluation** → Métriques, matrice de confusion, classement

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

### Démonstration rapide
```bash
python demo.py
```

### Pipeline d'entraînement complet
```bash
# Comparer les 6 modèles et sauvegarder le meilleur
python train.py --sauvegarder
```

### Pipeline avec un modèle spécifique
```bash
# Un seul modèle
python main.py

# Modèle spécifique
python main.py --modele svm

# Avec sélection de caractéristiques
python main.py --selection-caracs pca --n-caracteristiques 10
```

### Avec le dataset CICIDS2017
```bash
# Télécharger et utiliser CICIDS2017
python main.py --dataset cicids2017 --download

# Pipeline d'entraînement avec CICIDS2017
python train.py --dataset cicids2017 --sauvegarder
```

### Prédiction sur de nouvelles données
```bash
python main.py --predict --fichier mes_donnees.csv
```

### Lancer les tests
```bash
python -m pytest tests/ -v
```

---

## 🖥️ Interface web Streamlit

L'application web permet une utilisation interactive sans ligne de commande.

```bash
streamlit run app.py
```

**Fonctionnalités :**
- Chargement d'un modèle entraîné depuis l'interface
- Upload de fichier CSV
- Prédiction avec affichage Normal 🟢 / Attaque 🔴
- Métriques de performance (avec colonne label)
- Graphiques : distribution, importance des caractéristiques, matrice de confusion
- Téléchargement des résultats en CSV

---

## 📁 Structure du projet

```
NetGuard-AI/
│
├── app.py                       # Interface web Streamlit
├── main.py                      # Point d'entrée CLI
├── demo.py                      # Démonstration rapide
├── train.py                     # Pipeline d'entraînement complet
│
├── src/                         # Code source principal
│   ├── config.py                # Configuration centralisée
│   ├── preprocessing/           # Chargement, nettoyage, normalisation
│   │   ├── data_loader.py       #   ChargeurDonnees
│   │   ├── pipeline.py          #   PipelinePretraitement (8 étapes)
│   │   └── run_pipeline.py      #   Script CLI
│   ├── features/                # Extraction de caractéristiques
│   │   └── feature_extractor.py #   ANOVA F-test, PCA
│   ├── models/                  # Modèles ML
│   │   ├── detector.py          #   6 modèles + fabrique
│   │   └── compare.py           #   Comparaison des modèles
│   ├── evaluation/              # Métriques
│   │   └── metrics.py           #   Évaluation complète
│   └── utils/                   # Utilitaires
│       └── helpers.py           #   Logging, chrono, sauvegarde
│
├── datasets/                    # Scripts de téléchargement
│   └── download_cicids2017.py   #   Téléchargement CICIDS2017
│
├── data/
│   ├── raw/                     # Données brutes
│   └── processed/               # Modèles et métriques
│
├── tests/                       # Tests unitaires
│   ├── test_detector.py         #   6 modèles × plusieurs cas
│   └── test_pipeline.py         #   Pipeline de prétraitement
│
├── docs/                        # Documentation
│   ├── GUIDE_UTILISATEUR.md     # Guide utilisateur
│   └── GUIDE_DEVELOPPEUR.md     # Guide développeur
│
├── RAPPORT_FINAL.md             # Rapport complet du projet
│
├── requirements.txt             # Dépendances Python
├── .gitignore
├── LICENSE                      # Licence MIT
└── README.md
```

---

## 🤖 Modèles disponibles

| Modèle | Description | Avantages | Inconvénients |
|--------|-------------|-----------|---------------|
| **Decision Tree** | Arbre de décision | Rapide, interprétable | Surapprentissage |
| **Random Forest** | Forêt aléatoire | **Robuste, fiable** | Plus lent |
| **Logistic Regression** | Régression logistique | Très rapide, probas | Linéaire |
| **SVM** | Machine à vecteurs de support | Performant en HD | Long à entraîner |
| **KNN** | K plus proches voisins | Simple, non paramétrique | Lent en prédiction |
| **Naive Bayes** | Classifieur bayésien | Extrêmement rapide | Indépendance naïve |

---

## 📊 Résultats

Classement des 6 modèles sur données synthétiques (test : 600 échantillons) :

| Rang | Modèle | F1-Score | Détection | Fausses alarmes |
|:----:|--------|:--------:|:---------:|:---------------:|
| 🥇 | **Random Forest** | **1.0000** | **100%** | **0%** |
| 🥈 | Logistic Regression | 1.0000 | 100% | 0% |
| 🥉 | SVM | 1.0000 | 100% | 0% |
| 4 | Naive Bayes | 1.0000 | 100% | 0% |
| 5 | KNN | 0.9972 | 99.44% | 0% |
| 6 | Decision Tree | 0.9060 | 88.33% | 6% |

---

## 📦 Datasets

### Synthétique
- 2000 échantillons, 30 caractéristiques
- Généré automatiquement par `demo.py`
- Distribution gaussienne (normal μ=0, attaque μ=2)

### CICIDS2017
- Dataset réel de l'Université du Nouveau-Brunswick
- 80+ caractéristiques, 5 jours de trafic
- Attaques : DoS, DDoS, Brute Force, Web, Botnet, Port Scan
- Téléchargement : `python datasets/download_cicids2017.py`

---

## 📚 Documentation

- 📖 [Guide utilisateur](docs/GUIDE_UTILISATEUR.md) — Installation et utilisation
- 🔧 [Guide développeur](docs/GUIDE_DEVELOPPEUR.md) — Architecture et extension
- 📄 [Rapport final](RAPPORT_FINAL.md) — Rapport complet du projet

---

## 🧪 Tests

**93 tests unitaires** (Pytest) :
- 6 modèles × initialisation, entraînement, prédiction
- Évaluateur : métriques, matrice, rapport
- Pipeline : 8 étapes, cas aux limites

```bash
python -m pytest tests/ -v
```

---

## 🛠️ Technologies

- **Python 3.9+** — Langage principal
- **NumPy / Pandas** — Manipulation des données
- **Scikit-learn** — Machine Learning (6 modèles, preprocessing, métriques)
- **Streamlit** — Interface web interactive
- **Matplotlib / Seaborn** — Visualisation
- **Joblib** — Sauvegarde des modèles
- **Pytest** — Tests unitaires

---

## 📄 License

Ce projet est sous licence MIT. Voir le fichier [LICENSE](LICENSE).

---

## 👨‍💻 Auteur

Projet universitaire réalisé par **du-dev**.

- Dépôt : [https://github.com/du-dev/NetGuard-AI](https://github.com/du-dev/NetGuard-AI)

---

*NetGuard AI — Détection d'intrusions réseau par Machine Learning*
