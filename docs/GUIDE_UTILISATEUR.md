# Guide utilisateur NetGuard AI

Bienvenue dans **NetGuard AI**, un système intelligent de détection d'intrusions réseau basé sur le Machine Learning.

Ce guide vous explique comment installer, lancer et utiliser l'application.

---

## 1. Installation

### Prérequis
- Python 3.9 ou supérieur
- Git

### Procédure d'installation

```bash
# 1. Cloner le dépôt
git clone https://github.com/du-dev/NetGuard-AI.git
cd NetGuard-AI

# 2. Créer l'environnement virtuel
python -m venv venv

# 3. Activer l'environnement virtuel
# Sur Windows :
venv\Scripts\activate
# Sur macOS / Linux :
source venv/bin/activate

# 4. Installer les dépendances
pip install -r requirements.txt
```

---

## 2. Utilisation rapide

### Démonstration automatique (recommandé)

```bash
python demo.py
```

Cette commande génère 2000 échantillons synthétiques, entraîne un modèle Random Forest et affiche les résultats.

### Pipeline d'entraînement complet

```bash
# Entraîner et comparer les 6 modèles
python train.py

# Avec sauvegarde du meilleur modèle
python train.py --sauvegarder
```

### Interface graphique Streamlit

```bash
# Lancer l'application web
streamlit run app.py
```

Puis ouvrez votre navigateur à l'adresse : **http://localhost:8501**

---

## 3. Utilisation de l'interface Streamlit

L'application se compose de deux zones :

### Barre latérale (à gauche)
1. **Modèle** : Sélectionnez un modèle entraîné parmi ceux disponibles
2. **Données** : Uploadez un fichier CSV de trafic réseau

### Page principale
1. **Résultat** : Voir si le trafic est normal 🟢 ou malveillant 🔴
2. **Performances** : Accuracy, Précision, Rappel, F1-Score (si label présent)
3. **Graphiques** : Distribution, importance des caractéristiques
4. **Résultats détaillés** : Tableau complet avec possibilité de téléchargement

---

## 4. Commandes principales

| Commande | Description |
|----------|-------------|
| `python demo.py` | Démonstration rapide avec données synthétiques |
| `python main.py` | Pipeline complet avec un modèle |
| `python main.py --modele svm` | Utiliser un modèle spécifique |
| `python train.py` | Entraîner et comparer les 6 modèles |
| `python train.py --sauvegarder` | Sauvegarder le meilleur modèle |
| `streamlit run app.py` | Lancer l'interface web |
| `python main.py --predict --fichier data.csv` | Prédire sur un fichier |
| `python -m pytest tests/` | Lancer les tests unitaires |

---

## 5. Datasets supportés

### Données synthétiques (par défaut)
Générées automatiquement par `demo.py`. 30 caractéristiques, 2000 échantillons.

### CICIDS2017 (dataset réel)
```bash
# Téléchargement automatique
python datasets/download_cicids2017.py

# Ou via le pipeline
python main.py --dataset cicids2017 --download
```

---

## 6. Format des fichiers CSV

Le fichier doit contenir :
- Des colonnes de caractéristiques numériques
- Optionnellement une colonne `label` (0 = normal, 1 = attaque)

Exemple :
```csv
feature_01,feature_02,feature_03,label
0.5,1.2,30,0
-0.3,0.8,45,1
```

---

## 7. Dépannage

**Problème : "Module introuvable"**
```bash
source venv/bin/activate  # ou venv\Scripts\activate
pip install -r requirements.txt
```

**Problème : "Aucun modèle trouvé"**
```bash
python train.py --sauvegarder
```

**Problème : L'interface Streamlit ne se lance pas**
```bash
pip install streamlit
streamlit run app.py
```

---

*NetGuard AI - Détection d'intrusions réseau par Machine Learning*
