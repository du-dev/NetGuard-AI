# NetGuard AI

## Système intelligent de détection d'intrusions réseau basé sur le Machine Learning

---

**Mémoire présenté dans le cadre de l'évaluation de fin de module**

**Master en Informatique**

**Année universitaire 2025-2026**

---

---

**Master en Informatique — Année 2025-2026**
**Module : Sécurité des Réseaux**
**Professeur : [Nom du professeur]**
**Étudiant : [Votre nom]**

---

# Table des matières

1. [Introduction](#chapitre-1--introduction)
   - 1.1 Contexte de la cybersécurité
   - 1.2 Problématique
   - 1.3 Objectifs
   - 1.4 Méthodologie
   - 1.5 Organisation du mémoire
2. [Revue de littérature](#chapitre-2--revue-de-littérature)
   - 2.1 Réseaux informatiques
   - 2.2 Menaces réseau
   - 2.3 Systèmes de détection d'intrusions (IDS)
3. [Machine Learning](#chapitre-3--machine-learning)
   - 3.1 Définition et types d'apprentissage
   - 3.2 Algorithmes de classification supervisée
4. [Méthodologie](#chapitre-4--méthodologie)
   - 4.1 Dataset
   - 4.2 Prétraitement des données
   - 4.3 Entraînement des modèles
   - 4.4 Évaluation des performances
5. [Réalisation](#chapitre-5--réalisation)
   - 5.1 Outils et technologies
   - 5.2 Architecture du système
   - 5.3 Pipeline de traitement
   - 5.4 Interface utilisateur
6. [Résultats](#chapitre-6--résultats)
   - 6.1 Performances des modèles
   - 6.2 Analyse comparative
   - 6.3 Discussion et limites
7. [Conclusion](#conclusion)
   - 7.1 Résumé du travail
   - 7.2 Réponse à la problématique
   - 7.3 Perspectives
8. [Bibliographie](#bibliographie)

---

# Chapitre 1 : Introduction

## 1.1 Contexte de la cybersécurité

### Importance d'Internet

Internet est devenu un pilier fondamental de la société moderne. En 2025, plus de 5,5 milliards de personnes sont connectées à Internet, représentant environ 67 % de la population mondiale. Les infrastructures critiques — systèmes bancaires, réseaux électriques, services de santé, communications gouvernementales — dépendent toutes de réseaux informatiques interconnectés. Cette dépendance croissante rend la sécurité des réseaux plus cruciale que jamais.

Les entreprises, les institutions et les particuliers stockent et échangent des quantités massives de données sensibles via les réseaux : informations financières, données médicales, propriété intellectuelle, données personnelles. La protection de ces actifs numériques est devenue un enjeu stratégique et économique majeur.

### Augmentation des cyberattaques

Parallèlement à la croissance d'Internet, les cyberattaques ont connu une augmentation exponentielle. Selon les rapports de cybersécurité, le nombre d'attaques a plus que doublé entre 2020 et 2025. Les attaquants sont de plus en plus sophistiqués, utilisant des techniques avancées pour contourner les défenses traditionnelles.

Quelques chiffres clés :
- Un ransomware attaque une entreprise toutes les 11 secondes dans le monde
- Le coût moyen d'une cyberattaque pour une entreprise dépasse 4 millions de dollars
- 43 % des cyberattaques ciblent les petites et moyennes entreprises
- Le trafic malveillant représente environ 25 % du trafic Internet mondial

Les types d'attaques les plus courants incluent les dénis de service (DoS/DDoS), les attaques par force brute, les injections SQL, les logiciels malveillants (malwares), le hameçonnage (phishing), et les botnets. Chacune de ces menaces exploite des vulnérabilités spécifiques des réseaux et des systèmes informatiques.

### Limites des méthodes traditionnelles

Les systèmes de sécurité traditionnels, tels que les pare-feux (firewalls) et les systèmes de détection d'intrusions basés sur des signatures (Signature-based IDS), présentent des limitations importantes :

1. **Dépendance aux signatures connues** : Ces systèmes ne peuvent détecter que les attaques dont la signature est déjà répertoriée dans leur base de données. Les attaques inconnues, dites « Zero-Day », passent inaperçues.

2. **Faux positifs élevés** : Les règles strictes génèrent souvent de nombreuses fausses alarmes, submergeant les analystes de sécurité et causant une fatigue des alertes.

3. **Mise à jour manuelle** : Les bases de signatures doivent être mises à jour régulièrement, un processus qui prend du temps et peut laisser des fenêtres de vulnérabilité.

4. **Incapacité à détecter les variantes** : Une légère modification d'une attaque connue peut suffire à contourner la détection par signatures.

5. **Volume de données** : Les réseaux modernes génèrent des téraoctets de données par jour, un volume que les humains seuls ne peuvent pas analyser efficacement.

### Apport du Machine Learning

Le Machine Learning (apprentissage automatique) offre une alternative prometteuse aux méthodes traditionnelles. Contrairement aux systèmes basés sur des règles fixes, les modèles de Machine Learning apprennent à distinguer le trafic normal du trafic malveillant à partir des données elles-mêmes, sans nécessiter de signatures pré-définies.

Les avantages du Machine Learning pour la détection d'intrusions sont nombreux :

- **Détection d'attaques inconnues** : Un modèle bien entraîné peut identifier des comportements anormaux même s'ils ne correspondent à aucune attaque connue.
- **Adaptabilité** : Les modèles peuvent être ré-entraînés périodiquement pour s'adapter à l'évolution des menaces.
- **Automatisation** : Une fois entraîné, un modèle peut analyser des millions de flux réseau en temps réel.
- **Détection de patterns complexes** : Les algorithmes de Machine Learning peuvent identifier des corrélations subtiles entre des centaines de caractéristiques réseau, invisibles à l'œil humain.

C'est dans ce contexte que s'inscrit le projet **NetGuard AI**, qui vise à développer un système de détection d'intrusions réseau intelligent basé sur des techniques de Machine Learning.

## 1.2 Problématique

Les systèmes de sécurité traditionnels reposent principalement sur des signatures d'attaques connues. Ils détectent difficilement les nouvelles attaques (Zero-Day) et génèrent souvent de faux positifs. Face à l'évolution rapide des cybermenaces et à l'augmentation du volume de trafic réseau, les approches conventionnelles montrent leurs limites.

**Comment concevoir un système intelligent capable de détecter efficacement les intrusions réseau grâce aux techniques de Machine Learning ?**

Cette problématique centrale soulève plusieurs questions sous-jacentes :

1. Quels algorithmes de Machine Learning sont les plus adaptés à la détection d'intrusions réseau ?
2. Comment prétraiter efficacement les données réseau pour optimiser les performances des modèles ?
3. Quelles métriques utiliser pour évaluer et comparer les modèles de détection ?
4. Comment concevoir une architecture logicielle modulaire et évolutive pour un tel système ?

## 1.3 Objectifs

### Objectif général

Développer un système intelligent de détection d'intrusions réseau basé sur l'apprentissage automatique, capable de classifier le trafic réseau en trafic normal ou malveillant avec une haute fiabilité.

### Objectifs spécifiques

1. **Étudier les systèmes de détection d'intrusions (IDS)** : Analyser les différentes approches existantes, leurs avantages et leurs limites.

2. **Étudier plusieurs algorithmes de Machine Learning** : Explorer et comparer différents algorithmes de classification supervisée adaptés à la détection d'intrusions.

3. **Prétraiter un dataset réseau** : Mettre en œuvre un pipeline complet de nettoyage, normalisation et préparation des données.

4. **Implémenter et comparer plusieurs modèles** : Développer, entraîner et évaluer au moins six modèles de classification sur un même jeu de données.

5. **Identifier le meilleur modèle** : Déterminer, sur la base de métriques objectives (Accuracy, Précision, Rappel, F1-Score), le modèle le plus performant.

6. **Développer une interface utilisateur** : Créer une application interactive permettant de charger un modèle, importer des données et visualiser les prédictions.

## 1.4 Méthodologie

La démarche adoptée pour ce projet suit une méthodologie en six étapes :

1. **Recherche bibliographique** : Étude des concepts fondamentaux (réseaux, cybersécurité, IDS, Machine Learning) et des travaux existants dans le domaine.

2. **Choix d'un dataset** : Sélection d'un ensemble de données représentatif du trafic réseau, à la fois synthétique (pour le développement) et réel (CICIDS2017).

3. **Nettoyage des données** : Application d'un pipeline de prétraitement incluant la suppression des doublons, la gestion des valeurs manquantes, la normalisation et l'encodage.

4. **Entraînement** : Mise en œuvre et entraînement de six modèles de classification supervisée sur les données prétraitées.

5. **Évaluation** : Calcul systématique des métriques de performance (Accuracy, Précision, Rappel, F1-Score, matrice de confusion, courbe ROC) pour chaque modèle.

6. **Comparaison des modèles** : Analyse comparative des résultats pour identifier le modèle le plus performant et comprendre les forces et faiblesses de chaque approche.

## 1.5 Organisation du mémoire

Ce mémoire est structuré en six chapitres :

- **Chapitre 1 (Introduction)** : Présente le contexte, la problématique, les objectifs et la méthodologie du projet.
- **Chapitre 2 (Revue de littérature)** : Explore les concepts fondamentaux des réseaux informatiques, des menaces réseau et des systèmes de détection d'intrusions.
- **Chapitre 3 (Machine Learning)** : Définit l'apprentissage automatique et décrit en détail les algorithmes utilisés dans ce projet.
- **Chapitre 4 (Méthodologie)** : Détaille le processus suivi : choix du dataset, étapes de prétraitement, protocole d'entraînement et d'évaluation.
- **Chapitre 5 (Réalisation)** : Présente l'implémentation technique du système, les outils utilisés et l'architecture logicielle.
- **Chapitre 6 (Résultats)** : Expose et analyse les résultats obtenus, compare les performances des modèles et discute des limites de l'étude.
- **Conclusion** : Résume le travail accompli, répond à la problématique et propose des perspectives d'amélioration.

---

# Chapitre 2 : Revue de littérature

## 2.1 Réseaux informatiques

### Définition

Un réseau informatique est un ensemble d'équipements interconnectés (ordinateurs, serveurs, routeurs, commutateurs) qui communiquent entre eux pour échanger des informations et partager des ressources. Les réseaux sont classés selon leur échelle géographique :

- **PAN (Personal Area Network)** : Réseau personnel, portée de quelques mètres
- **LAN (Local Area Network)** : Réseau local, généralement dans un bâtiment
- **MAN (Metropolitan Area Network)** : Réseau métropolitain, couvre une ville
- **WAN (Wide Area Network)** : Réseau étendu, couvre un pays ou le monde (Internet)

### Architecture TCP/IP

Le modèle TCP/IP est le fondement d'Internet. Il organise la communication réseau en quatre couches :

1. **Couche Application** (HTTP, FTP, SMTP, DNS) : Fournit les services aux applications utilisateur.
2. **Couche Transport** (TCP, UDP) : Assure le transport fiable (TCP) ou non fiable (UDP) des données entre les hôtes.
3. **Couche Internet** (IP, ICMP) : Gère l'adressage et le routage des paquets à travers le réseau.
4. **Couche Accès réseau** (Ethernet, WiFi) : Gère la transmission physique des données sur le support réseau.

### Protocoles

Un protocole réseau est un ensemble de règles qui régissent la communication entre les équipements. Les protocoles les plus importants sont :

| Protocole | Couche | Fonction | Port |
|-----------|--------|----------|------|
| HTTP/HTTPS | Application | Transfert de pages web | 80/443 |
| FTP | Application | Transfert de fichiers | 21 |
| SMTP | Application | Envoi d'emails | 25 |
| DNS | Application | Résolution de noms de domaine | 53 |
| TCP | Transport | Transport fiable orienté connexion | - |
| UDP | Transport | Transport non fiable sans connexion | - |
| IP | Internet | Routage des paquets | - |
| ICMP | Internet | Diagnostic et contrôle d'erreurs | - |

### Adresse IP

Une adresse IP (Internet Protocol) est un identifiant unique attribué à chaque appareil connecté à un réseau. Il existe deux versions :

- **IPv4** : 32 bits, format xxx.xxx.xxx.xxx (ex: 192.168.1.1), environ 4,3 milliards d'adresses
- **IPv6** : 128 bits, format hexadécimal (ex: 2001:0db8::1), nombre quasi illimité d'adresses

Les adresses IP sont essentielles pour la détection d'intrusions, car elles permettent d'identifier la source et la destination du trafic réseau.

### Ports

Un port est un numéro (de 0 à 65535) qui identifie une application ou un service spécifique sur un hôte. Les ports sont divisés en trois catégories :

- **Ports bien connus (0-1023)** : Associés à des services standard (HTTP=80, HTTPS=443, SSH=22)
- **Ports enregistrés (1024-49151)** : Attribués à des applications spécifiques
- **Ports dynamiques/privés (49152-65535)** : Utilisés temporairement par les clients

L'analyse des ports est cruciale en détection d'intrusions, car de nombreuses attaques ciblent des ports spécifiques ou effectuent des scans de ports.

## 2.2 Menaces réseau

Les menaces réseau sont nombreuses et variées. Voici les principales catégories :

### Déni de service (DoS)

Une attaque par déni de service (Denial of Service) vise à rendre un service, un serveur ou une infrastructure indisponible pour ses utilisateurs légitimes. L'attaquant submerge la cible de requêtes jusqu'à l'épuisement de ses ressources (bande passante, mémoire, CPU).

**Caractéristiques** :
- Source unique (DoS) ou multiple (DDoS)
- Volumétrique (saturation de bande passante) ou applicative (exploitation de vulnérabilités)
- Difficile à bloquer car le trafic peut ressembler à du trafic légitime

### Déni de service distribué (DDoS)

Le DDoS est une version amplifiée du DoS où l'attaque provient de multiples sources simultanément, généralement un botnet (réseau d'ordinateurs infectés). L'attaque DDoS est particulièrement difficile à contrer car :

- Des milliers d'adresses IP différentes sont impliquées
- Le volume de trafic peut atteindre plusieurs téraoctets par seconde
- La source réelle est masquée par le réseau d'attaquants intermédiaires

### Attaque par force brute (Brute Force)

Une attaque par force brute consiste à essayer systématiquement toutes les combinaisons possibles de mots de passe ou de clés jusqu'à trouver la bonne. Bien que simple dans son principe, cette attaque reste efficace contre des mots de passe faibles.

**Variantes** :
- Attaque par dictionnaire : utilise une liste de mots de passe courants
- Attaque par force brute classique : essaie toutes les combinaisons possibles
- Credential stuffing : utilise des identifiants volés sur d'autres services

### Logiciel malveillant (Malware)

Un malware est un programme conçu pour infecter un système informatique et y effectuer des actions malveillantes. Les principaux types sont :

- **Virus** : Se propage en infectant d'autres fichiers
- **Ver (Worm)** : Se propage automatiquement via le réseau
- **Cheval de Troie (Trojan)** : Se fait passer pour un logiciel légitime
- **Ransomware** : Chiffre les données et demande une rançon
- **Spyware** : Espionne l'activité de l'utilisateur

### Hameçonnage (Phishing)

Le phishing est une technique d'ingénierie sociale qui vise à tromper l'utilisateur pour lui soutirer des informations sensibles (identifiants, numéros de carte bancaire). L'attaque se fait généralement par email ou site web frauduleux imitant un service légitime.

### Injection SQL

L'injection SQL exploite les vulnérabilités des applications web qui construisent des requêtes SQL à partir de données utilisateur non nettoyées. L'attaquant peut :

- Contourner l'authentification
- Lire, modifier ou supprimer des données de la base
- Exécuter des commandes sur le serveur

### Scan de ports (Port Scanning)

Le scan de ports est une technique de reconnaissance utilisée par les attaquants pour identifier les services ouverts sur une cible. Bien que non malveillant en soi, le scan de ports est souvent le prélude à une attaque plus ciblée.

**Techniques courantes** :
- SYN scan (demi-ouverture de connexion)
- Connect scan (connexion complète)
- UDP scan
- Stealth scan (furtif)

### Botnet

Un botnet est un réseau d'ordinateurs infectés (bots) contrôlés à distance par un attaquant (botmaster). Les botnets sont utilisés pour :

- Lancer des attaques DDoS massives
- Envoyer des spams
- Miner des cryptomonnaies
- Voler des données
- Propager des malwares

## 2.3 Systèmes de détection d'intrusions (IDS)

Un système de détection d'intrusions (Intrusion Detection System - IDS) est un dispositif logiciel ou matériel qui surveille le trafic réseau ou les activités d'un système pour détecter des comportements malveillants ou des violations de politique de sécurité.

### IDS réseau (NIDS)

Le NIDS (Network-based IDS) surveille le trafic circulant sur le réseau en analysant les paquets. Il est placé à des points stratégiques du réseau (généralement en sortie de réseau) et analyse tout le trafic qui passe.

**Avantages** :
- Surveille l'ensemble du réseau
- Aucun impact sur les systèmes surveillés
- Détection en temps réel
- Impossible à contourner pour les attaquants internes

**Inconvénients** :
- Ne peut pas analyser le trafic chiffré
- Volume de données à analyser très élevé
- Sensible aux faux positifs
- Coûteux en bande passante et en ressources

### IDS hôte (HIDS)

Le HIDS (Host-based IDS) s'exécute sur un hôte spécifique (serveur, poste de travail) et surveille les activités locales : journaux système, fichiers, processus, connexions.

**Avantages** :
- Analyse détaillée de l'activité de l'hôte
- Peut détecter des attaques cryptées (après déchiffrement)
- Détecte les modifications de fichiers
- Efficace contre les attaques internes

**Inconvénients** :
- Consomme des ressources sur l'hôte
- Doit être installé sur chaque machine
- Ne voit pas le trafic réseau global
- Peut être désactivé par un attaquant ayant compromis l'hôte

### IDS par signatures

L'IDS par signatures (Signature-based IDS) fonctionne en comparant le trafic réseau à une base de données de signatures d'attaques connues. Chaque signature est un motif spécifique (séquence d'octets, pattern de paquets) qui caractérise une attaque particulière.

**Avantages** :
- Très faible taux de faux positifs pour les attaques connues
- Rapide et efficace
- Facile à implémenter
- Documentation des attaques disponible

**Inconvénients** :
- Ne détecte pas les attaques Zero-Day
- Nécessite des mises à jour régulières de la base de signatures
- Une légère variation d'une attaque peut contourner la détection
- Base de signatures volumineuse

### IDS par anomalies

L'IDS par anomalies (Anomaly-based IDS) établit un modèle de comportement normal du réseau (baseline) et signale tout écart significatif comme potentiellement malveillant. C'est dans cette catégorie que s'inscrivent les approches par Machine Learning.

**Avantages** :
- Détecte les attaques inconnues (Zero-Day)
- Détecte les variantes d'attaques connues
- S'adapte automatiquement au profil du réseau
- Moins de maintenance que les IDS par signatures

**Inconvénients** :
- Taux de faux positifs plus élevé
- Plus complexe à mettre en œuvre
- Nécessite une phase d'apprentissage
- Sensible aux variations légitimes du trafic

---

# Chapitre 3 : Machine Learning

## 3.1 Machine Learning

### Définition

Le Machine Learning (apprentissage automatique) est une branche de l'intelligence artificielle qui permet à un système d'apprendre et de s'améliorer à partir de données, sans être explicitement programmé pour chaque tâche. Au lieu de suivre des règles fixes, un algorithme de Machine Learning analyse des données d'entraînement, identifie des patterns et construit un modèle capable de faire des prédictions sur de nouvelles données.

Formellement, on définit l'apprentissage comme suit : un programme informatique apprend à partir d'une expérience E (les données), par rapport à une tâche T (la classification), et une mesure de performance P (la précision), si sa performance sur T s'améliore avec l'expérience E.

### Apprentissage supervisé

L'apprentissage supervisé est la forme la plus courante de Machine Learning. L'algorithme reçoit un ensemble de données d'entraînement étiquetées, c'est-à-dire que chaque exemple est associé à sa sortie désirée (la « vérité terrain »). Le modèle apprend alors à mapper les entrées (caractéristiques) vers les sorties (étiquettes).

**Types de problèmes supervisés** :
- **Classification** : Prédire une catégorie discrète (normal vs attaque, spam vs non-spam)
- **Régression** : Prédire une valeur numérique continue (prix, température)

**Exemples d'algorithmes supervisés** : Random Forest, SVM, KNN, Régression logistique, Réseaux de neurones

Dans le contexte de ce projet, la détection d'intrusions réseau est formulée comme un problème de **classification binaire supervisée** : à partir de caractéristiques extraites du trafic réseau, le modèle doit prédire si le trafic est normal (classe 0) ou malveillant (classe 1).

### Apprentissage non supervisé

L'apprentissage non supervisé travaille avec des données non étiquetées. L'algorithme doit découvrir par lui-même la structure cachée des données : regroupements naturels (clustering), réductions de dimensionnalité, ou détection d'anomalies.

**Types de problèmes non supervisés** :
- **Clustering** : Regrouper des données similaires (K-Means, DBSCAN)
- **Réduction de dimensionnalité** : Réduire le nombre de variables (PCA, t-SNE)
- **Détection d'anomalies** : Identifier des points aberrants

**Avantage pour la détection d'intrusions** : L'apprentissage non supervisé peut détecter des attaques inconnues sans nécessiter de données étiquetées, ce qui est utile quand les données d'attaque sont rares.

## 3.2 Algorithmes de classification supervisée

Cette section décrit les algorithmes utilisés dans ce projet ainsi que deux algorithmes avancés (XGBoost, LightGBM) qui sont discutés dans la littérature.

### Arbre de décision (Decision Tree)

**Fonctionnement** : L'arbre de décision est un modèle hiérarchique qui prend des décisions en suivant une structure d'arbre. Chaque nœud interne représente un test sur une caractéristique (ex: « est-ce que le nombre de paquets > 1000 ? »), chaque branche représente le résultat du test, et chaque feuille représente une classe (normal ou attaque).

L'arbre est construit récursivement en choisissant à chaque nœud la caractéristique qui sépare le mieux les classes, selon des critères comme le gain d'information (entropie) ou l'indice de Gini.

**Avantages** :
- Très interprétable et visualisable
- Rapide à entraîner et à exécuter
- Gère à la fois les données numériques et catégorielles
- Ne nécessite pas de normalisation des données

**Inconvénients** :
- Tendance au surapprentissage (overfitting), surtout avec des arbres profonds
- Sensible aux petites variations des données
- Peut être instable (un petit changement peut produire un arbre très différent)
- Précision généralement inférieure aux méthodes d'ensemble

**Paramètres clés** : profondeur max (`max_depth`), échantillons min pour diviser (`min_samples_split`), critère de division (`criterion`).

### Forêt aléatoire (Random Forest)

**Fonctionnement** : Le Random Forest est une méthode d'ensemble qui construit une multitude d'arbres de décision sur des sous-ensembles aléatoires des données et des caractéristiques, puis combine leurs prédictions par vote majoritaire (classification) ou moyenne (régression).

L'algorithme introduit deux sources d'aléatoire :
1. **Bagging** (Bootstrap Aggregating) : Chaque arbre est entraîné sur un échantillon aléatoire (avec remise) des données d'entraînement.
2. **Sélection aléatoire des caractéristiques** : À chaque nœud, seulement un sous-ensemble aléatoire des caractéristiques est considéré pour la division.

Cette double randomisation rend le modèle très robuste au surapprentissage.

**Avantages** :
- Excellente précision, souvent parmi les meilleurs modèles
- Très robuste au surapprentissage
- Gère un grand nombre de caractéristiques
- Fournit une estimation de l'importance des caractéristiques
- Parallélisable facilement

**Inconvénients** :
- Plus lent et plus volumineux qu'un seul arbre de décision
- Moins interprétable qu'un arbre unique
- Peut être moins performant sur des données très déséquilibrées

**Paramètres clés** : nombre d'arbres (`n_estimators`), profondeur max (`max_depth`).

### K-plus proches voisins (KNN)

**Fonctionnement** : KNN (K-Nearest Neighbors) est un algorithme non paramétrique qui classifie un nouvel échantillon en examinant les K échantillons d'entraînement les plus proches (selon une mesure de distance, généralement euclidienne). La classe prédite est la classe majoritaire parmi ces K voisins.

KNN est un algorithme « paresseux » (lazy learner) : il ne construit pas de modèle explicite pendant l'entraînement. Tout le calcul est effectué au moment de la prédiction.

**Avantages** :
- Simple à comprendre et à implémenter
- Pas d'hypothèse sur la distribution des données
- S'adapte naturellement aux frontières de décision complexes
- Efficace avec un grand nombre d'échantillons d'entraînement

**Inconvénients** :
- Très sensible à l'échelle des caractéristiques (nécessite une normalisation)
- Lent en prédiction (doit calculer la distance à tous les échantillons)
- Sensible à la malédiction de la dimensionnalité
- Performance dépend fortement du choix de K

**Paramètres clés** : nombre de voisins (`n_neighbors`), métrique de distance (`metric`), pondération (`weights`).

### Machine à vecteurs de support (SVM)

**Fonctionnement** : SVM (Support Vector Machine) cherche à trouver l'hyperplan qui sépare le mieux les classes dans un espace de dimension éventuellement élevée. L'hyperplan optimal maximise la marge (distance) entre les échantillons des deux classes. Les échantillons les plus proches de l'hyperplan sont appelés « vecteurs de support ».

Pour les problèmes non linéairement séparables, SVM utilise l'astuce du noyau (kernel trick) : les données sont projetées dans un espace de dimension supérieure où une séparation linéaire devient possible.

**Avantages** :
- Très performant en haute dimension
- Efficace quand la marge de séparation est claire
- Robuste grâce à la maximisation de la marge
- Polyvalent grâce aux différents noyaux (linéaire, RBF, polynomial)

**Inconvénients** :
- Peut être lent à entraîner sur de grands jeux de données
- Sensible au choix du noyau et des paramètres
- Pas de probabilités natives (nécessite Platt scaling)
- Difficile à interpréter

**Paramètres clés** : type de noyau (`kernel`), paramètre de régularisation (`C`), paramètre du noyau RBF (`gamma`).

### Régression logistique (Logistic Regression)

**Fonctionnement** : Malgré son nom, la régression logistique est un modèle de classification. Elle modélise la probabilité qu'un échantillon appartienne à une classe donnée à l'aide de la fonction sigmoïde (logistique) :

P(y=1|x) = 1 / (1 + e^(-z))

où z est une combinaison linéaire des caractéristiques d'entrée. Le modèle estime les coefficients (poids) de cette combinaison linéaire en maximisant la vraisemblance des données d'entraînement.

**Avantages** :
- Très rapide à entraîner
- Probabilités bien calibrées
- Simple et interprétable (coefficients interprétables)
- Efficace quand les classes sont approximativement séparables linéairement

**Inconvénients** :
- Suppose une relation linéaire entre les caractéristiques et le log-odds
- Performance limitée si les frontières de décision sont complexes
- Sensible aux valeurs aberrantes
- Nécessite une bonne sélection des caractéristiques

**Paramètres clés** : régularisation (`C`, `penalty`), solveur (`solver`).

### Naive Bayes

**Fonctionnement** : Le classifieur Naive Bayes est basé sur le théorème de Bayes avec une hypothèse d'indépendance conditionnelle forte (naïve) : on suppose que chaque caractéristique est indépendante des autres étant donné la classe. Malgré cette hypothèse souvent irréaliste, Naive Bayes donne de bons résultats dans de nombreux cas pratiques.

La probabilité qu'un échantillon x appartienne à une classe c est donnée par :

P(c|x) = P(c) × ∏ P(xi|c) / P(x)

où P(c) est la probabilité a priori de la classe c, et P(xi|c) est la probabilité de la caractéristique xi sachant c.

**Avantages** :
- Extrêmement rapide à entraîner et à prédire
- Fonctionne avec peu de données
- Gère bien un grand nombre de caractéristiques
- Donne des probabilités bien calibrées

**Inconvénients** :
- Hypothèse d'indépendance trop forte pour des données corrélées
- Sensible à la distribution des caractéristiques (suppose une distribution gaussienne pour GaussianNB)
- Performances généralement inférieures aux modèles plus complexes

**Paramètres clés** : lissage (`var_smoothing`).

> **Note** : XGBoost et LightGBM sont présentés ici à titre de revue de littérature. Ils ne sont pas implémentés dans le projet NetGuard AI, qui se concentre sur six modèles de base de Scikit-learn. Leur ajout constituerait une amélioration future.

### XGBoost (eXtreme Gradient Boosting)

**Fonctionnement** : XGBoost est une implémentation optimisée du gradient boosting. Contrairement au Random Forest qui construit des arbres indépendants en parallèle, le boosting construit des arbres séquentiellement : chaque nouvel arbre tente de corriger les erreurs de l'arbre précédent.

XGBoost se distingue par :
- Une régularisation avancée (L1 et L2) pour éviter le surapprentissage
- Une parallélisation efficace au niveau de la construction des arbres
- La gestion automatique des valeurs manquantes
- Des optimisations matérielles (utilisation du cache)

**Avantages** :
- Très haute précision, souvent gagnant des compétitions Kaggle
- Gère naturellement les valeurs manquantes
- Intègre une régularisation puissante
- Calcul distribué possible

**Inconvénients** :
- Nombreux hyperparamètres à optimiser
- Risque de surapprentissage si mal paramétré
- Plus lent à entraîner que Random Forest
- Interprétabilité limitée

### LightGBM (Light Gradient Boosting Machine)

**Fonctionnement** : LightGBM est une variante du gradient boosting développée par Microsoft, optimisée pour la rapidité et l'efficacité mémoire. Ses innovations principales sont :

- **GOSS** (Gradient-based One-Side Sampling) : Échantillonnage adaptatif qui privilégie les échantillons avec un grand gradient (mal prédits)
- **EFB** (Exclusive Feature Bundling) : Regroupement de caractéristiques mutuellement exclusives pour réduire la dimensionnalité

**Avantages** :
- Très rapide (jusqu'à 10 fois plus rapide que XGBoost)
- Faible consommation mémoire
- Haute précision comparable à XGBoost
- Support natif des caractéristiques catégorielles

**Inconvénients** :
- Risque de surapprentissage sur petits jeux de données
- Moins adapté aux données avec très peu d'échantillons
- Paramétrage plus complexe
- Moins mature que XGBoost

---

# Chapitre 4 : Méthodologie

## 4.1 Dataset

### Choix du dataset

Le choix du dataset est une étape cruciale dans tout projet de Machine Learning. Pour ce projet, deux datasets sont utilisés :

**Dataset synthétique** (par défaut) :
- Généré automatiquement par le script `demo.py`
- 2000 échantillons de trafic réseau
- 30 caractéristiques numériques
- 1400 échantillons normaux (70%), 600 attaques (30%)
- Distribution gaussienne avec paramètres distincts pour chaque classe
- Permet un développement et des tests rapides

**CICIDS2017** (dataset réel) :
- Développé par l'Institut Canadien de Cybersécurité (CIC) de l'Université du Nouveau-Brunswick
- Considéré comme l'un des datasets de référence pour la détection d'intrusions
- Contient du trafic réseau réel capturé sur 5 jours (du lundi au vendredi)
- 80+ caractéristiques extraites par CICFlowMeter
- Types d'attaques : DoS, DDoS, Brute Force, Web Attack, Infiltration, Botnet, Port Scan

| Jour | Activité | Attaques |
|------|----------|----------|
| Lundi | Trafic normal uniquement | Aucune |
| Mardi | Attaques Brute Force | SSH, FTP Patator |
| Mercredi | Attaques DoS | DoS slowloris, DoS Hulk, DoS GoldenEye |
| Jeudi | Web + Infiltration | Web SQL Injection, XSS, Infiltration |
| Vendredi | DDoS + Port Scan + Botnet | DDoS LOIC, Port Scan, Botnet |

Le dataset synthétique est utilisé par défaut pour la démonstration et les tests. Le dataset CICIDS2017 peut être téléchargé et utilisé pour des résultats plus réalistes via la commande :

```bash
python datasets/download_cicids2017.py
```

### Structure des données

Chaque échantillon du dataset représente un flux réseau (conversation entre deux hôtes) avec les caractéristiques suivantes :

- **Caractéristiques volumétriques** : nombre de paquets, taille totale, débit
- **Caractéristiques temporelles** : durée de la connexion, temps inter-paquets
- **Caractéristiques de flux** : ratio aller/retour, taille des fenêtres TCP
- **Indicateurs de flags** : SYN, ACK, FIN, RST

La variable cible (label) est binaire :
- **0** : Trafic normal (BENIGN)
- **1** : Attaque (MALICIOUS)

![Distribution des classes](../data/processed/graphiques/01_distribution_classes.png)

*Graphique 4.1 : Distribution des classes dans les ensembles d'entraînement et de test*

## 4.2 Prétraitement des données

Le prétraitement est une étape fondamentale qui conditionne la qualité des résultats. Le pipeline de prétraitement implémenté dans ce projet comprend six étapes séquentielles :

### Étape 1 : Chargement des données

Le fichier CSV est chargé à l'aide de la bibliothèque Pandas. Les colonnes sont automatiquement détectées et typées. Les colonnes non pertinentes (identifiants, timestamps, adresses IP) sont identifiées et marquées pour suppression.

### Étape 2 : Suppression des colonnes inutiles

Certaines colonnes ne contribuent pas à la détection d'intrusions et peuvent même nuire aux performances des modèles :

- Identifiants de flux (Flow ID)
- Adresses IP source et destination
- Ports source et destination
- Timestamps
- Colonnes dupliquées

Ces colonnes sont retirées du dataset avant l'entraînement.

### Étape 3 : Suppression des doublons

Les lignes en double sont identifiées et supprimées. Les doublons peuvent biaiser l'entraînement en donnant un poids excessif à certains échantillons et n'apportent pas d'information supplémentaire.

### Étape 4 : Gestion des valeurs manquantes

Les valeurs manquantes (NaN) et infinies (Inf) sont détectées et traitées :
1. Les valeurs NaN et Inf sont converties en NaN standard
2. Les lignes contenant des NaN sont supprimées
3. Un rapport est généré indiquant le nombre de valeurs manquantes traitées

### Étape 5 : Normalisation

La normalisation (ou standardisation) est essentielle pour les algorithmes sensibles à l'échelle des caractéristiques (SVM, KNN, régression logistique). La méthode utilisée est la normalisation Z-score (StandardScaler) :

z = (x - μ) / σ

où μ est la moyenne et σ l'écart-type de la caractéristique. Après normalisation, chaque caractéristique a une moyenne de 0 et un écart-type de 1.

### Étape 6 : Encodage des variables catégorielles

Les variables catégorielles (comme le protocole ou le type de service) sont encodées numériquement à l'aide de LabelEncoder (pour la cible) ou One-Hot Encoding (pour les caractéristiques).

### Étape 7 : Séparation Train/Test

Le dataset est divisé en deux ensembles :
- **Ensemble d'entraînement (70%)** : utilisé pour entraîner les modèles
- **Ensemble de test (30%)** : utilisé pour évaluer les performances

La séparation est stratifiée : la proportion de classes normale/attaque est préservée dans les deux ensembles. Cette méthode permet une évaluation plus fiable, surtout en cas de déséquilibre des classes.

### Étape 8 : Sélection des caractéristiques (optionnelle)

Pour améliorer les performances et réduire la dimensionnalité, une sélection des caractéristiques peut être appliquée. La méthode utilisée est le test ANOVA F-test (SelectKBest) qui sélectionne les K caractéristiques les plus corrélées avec la variable cible.

![Matrice de corrélation](../data/processed/graphiques/03_matrice_correlation.png)

*Graphique 4.2 : Heatmap de corrélation entre les caractéristiques du dataset*

## 4.3 Entraînement des modèles

Six modèles de classification supervisée sont entraînés et comparés dans ce projet :

| Modèle | Bibliothèque | Classe | Hyperparamètres principaux |
|--------|-------------|-------|---------------------------|
| Decision Tree | Scikit-learn | DecisionTreeClassifier | max_depth=20, min_samples_split=5 |
| Random Forest | Scikit-learn | RandomForestClassifier | n_estimators=100, max_depth=20 |
| Logistic Regression | Scikit-learn | LogisticRegression | C=1.0, max_iter=1000 |
| SVM | Scikit-learn | SVC | kernel='rbf', probability=True |
| KNN | Scikit-learn | KNeighborsClassifier | n_neighbors=5, weights='distance' |
| Naive Bayes | Scikit-learn | GaussianNB | var_smoothing=1e-9 |

Chaque modèle est entraîné sur le même ensemble d'entraînement (X_train, y_train) avec les mêmes données prétraitées. Un chronomètre mesure le temps d'entraînement de chaque modèle pour permettre une comparaison de performance temporelle.

### Gestion des probabilités

Pour les modèles supportant `predict_proba()` (Random Forest, Logistic Regression, KNN, Naive Bayes, et SVM avec `probability=True`), les probabilités de classe sont également calculées. Ces probabilités sont utilisées pour tracer les courbes ROC et calculer l'AUC.

Les modèles qui ne supportent pas nativement les probabilités (comme SVM sans l'option `probability=True`) ne peuvent pas produire de courbes ROC complètes.

## 4.4 Évaluation des performances

L'évaluation des modèles est réalisée à l'aide de plusieurs métriques complémentaires :

### Accuracy (Exactitude)

L'accuracy mesure la proportion de prédictions correctes parmi l'ensemble des prédictions :

Accuracy = (VP + VN) / (VP + VN + FP + FN)

- VP = Vrais Positifs (attaques correctement détectées)
- VN = Vrais Négatifs (trafic normal correctement identifié)
- FP = Faux Positifs (trafic normal classé comme attaque)
- FN = Faux Négatifs (attaques non détectées)

**Limite** : L'accuracy peut être trompeuse en cas de déséquilibre des classes. Si 95% du trafic est normal, un modèle qui prédit toujours « normal » aura 95% d'accuracy mais ne détectera aucune attaque.

### Précision

La précision mesure, parmi les échantillons classés comme positifs (attaques), combien le sont réellement :

Précision = VP / (VP + FP)

Une précision élevée signifie peu de fausses alarmes. Cette métrique est importante car les fausses alarmes coûtent du temps aux analystes de sécurité.

### Rappel (Recall)

Le rappel mesure, parmi les attaques réelles, combien ont été correctement détectées :

Rappel = VP / (VP + FN)

Un rappel élevé signifie que peu d'attaques sont manquées. Cette métrique est cruciale en sécurité : une attaque non détectée peut avoir des conséquences graves.

### F1-Score

Le F1-Score est la moyenne harmonique de la précision et du rappel. Il fournit un compromis équilibré entre les deux :

F1 = 2 × (Précision × Rappel) / (Précision + Rappel)

Le F1-Score est particulièrement utile quand les classes sont déséquilibrées. Un modèle avec un F1-Score élevé équilibre bien la détection des attaques et la limitation des fausses alarmes. Dans ce projet, le F1-Score est utilisé comme métrique principale pour classer les modèles.

### Matrice de confusion

La matrice de confusion est un tableau qui croise les classes réelles (en lignes) avec les classes prédites (en colonnes) :

| | Prédit Normal | Prédit Attaque |
|---|:---:|:---:|
| **Réel Normal** | VN | FP |
| **Réel Attaque** | FN | VP |

Chaque cellule donne le nombre d'échantillons dans chaque catégorie. La matrice de confusion permet une analyse visuelle rapide des performances.

### Courbe ROC et AUC

La courbe ROC (Receiver Operating Characteristic) trace le taux de vrais positifs (TPR) en fonction du taux de faux positifs (FPR) pour différents seuils de classification.

- **TPR (Rappel)** = VP / (VP + FN) — proportion d'attaques détectées
- **FPR** = FP / (FP + VN) — proportion de faux positifs

L'AUC (Area Under the Curve) résume la performance globale du modèle en un nombre :
- AUC = 1.0 : Classifieur parfait
- AUC = 0.5 : Classifieur aléatoire
- AUC < 0.5 : Classifieur pire que le hasard

![Courbes ROC](../data/processed/graphiques/04_courbes_roc.png)

*Graphique 4.3 : Courbes ROC des 6 modèles avec valeurs d'AUC*

### Taux de détection et taux de faux positifs

Deux métriques dérivées de la matrice de confusion sont également calculées :

- **Taux de détection** (TPR) : pourcentage d'attaques correctement détectées
- **Taux de faux positifs** (FPR) : pourcentage de trafic normal classé à tort comme attaque

---

# Chapitre 5 : Réalisation

## 5.1 Outils et technologies

Le projet NetGuard AI a été développé avec les technologies suivantes :

| Technologie | Version | Rôle |
|-------------|---------|------|
| **Python** | 3.9+ | Langage de programmation principal |
| **Scikit-learn** | 1.2+ | Bibliothèque de Machine Learning (6 modèles) |
| **Pandas** | 1.5+ | Manipulation et analyse des données |
| **NumPy** | 1.24+ | Calculs numériques et matrices |
| **Matplotlib / Seaborn** | - | Visualisation des données et des résultats |
| **Streamlit** | 1.28+ | Interface utilisateur web interactive |
| **Joblib** | - | Sérialisation et désérialisation des modèles |
| **Pytest** | 7.0+ | Framework de tests unitaires |

### Justification des choix

**Python** a été choisi pour sa richesse en bibliothèques de data science et sa large adoption dans la communauté du Machine Learning. **Scikit-learn** offre une API unifiée pour tous les modèles (fit/predict), ce qui facilite l'implémentation et la comparaison. **Streamlit** permet de créer une interface web interactive sans nécessiter de compétences en développement web (HTML, CSS, JavaScript).

## 5.2 Architecture du système

L'architecture du système est modulaire, organisée en six packages principaux :

```
NetGuard-AI/
├── src/                      # Code source principal
│   ├── config.py             # Configuration centralisée
│   ├── preprocessing/        # Chargement et prétraitement
│   │   ├── data_loader.py    # Chargement des datasets
│   │   ├── pipeline.py       # Pipeline de prétraitement
│   │   └── run_pipeline.py   # Exécution du pipeline
│   ├── features/             # Extraction de caractéristiques
│   │   └── feature_extractor.py  # PCA, SelectKBest
│   ├── models/               # Modèles de Machine Learning
│   │   ├── detector.py       # 6 modèles de classification
│   │   └── compare.py        # Comparaison des modèles
│   ├── evaluation/           # Évaluation des performances
│   │   └── metrics.py        # Métriques et matrices
│   └── utils/                # Utilitaires
│       └── helpers.py        # Logging, chronomètre, sauvegarde
├── tests/                    # Tests unitaires (93 tests)
├── datasets/                 # Scripts de téléchargement
├── docs/                     # Documentation
├── app.py                    # Interface Streamlit
├── main.py                   # Interface CLI principale
├── train.py                  # Pipeline d'entraînement complet
├── demo.py                   # Démonstration rapide
└── generate_graphiques.py    # Génération des graphiques
```

### Principe de modularité

Chaque package a une responsabilité unique et bien définie :
- **config.py** : Centralise tous les paramètres (chemins, hyperparamètres, configuration des datasets)
- **preprocessing/** : Gère le chargement, le nettoyage et la transformation des données
- **features/** : Implémente la sélection et l'extraction de caractéristiques
- **models/** : Contient les 6 modèles de classification et la logique de comparaison
- **evaluation/** : Calcule toutes les métriques d'évaluation
- **utils/** : Fournit des fonctions transverses (logging, chronomètre, sauvegarde)

## 5.3 Pipeline de traitement

Le pipeline complet de traitement se déroule en sept étapes :

```
                    ┌──────────────────────┐
                    │    Dataset (CSV)     │
                    └──────────┬───────────┘
                               ↓
                    ┌──────────────────────┐
                    │      Nettoyage       │
                    │  - Suppression doublons│
                    │  - Gestion NaN/Inf   │
                    └──────────┬───────────┘
                               ↓
                    ┌──────────────────────┐
                    │    Prétraitement     │
                    │  - Normalisation     │
                    │  - Encodage          │
                    └──────────┬───────────┘
                               ↓
                    ┌──────────────────────┐
                    │  Sélection variables │
                    │  - ANOVA F-test      │
                    │  - Feature selection │
                    └──────────┬───────────┘
                               ↓
                    ┌──────────────────────┐
                    │    Entraînement      │
                    │  - 6 modèles ML      │
                    │  - Fit sur X_train   │
                    └──────────┬───────────┘
                               ↓
                    ┌──────────────────────┐
                    │     Prédiction       │
                    │  - Predict sur X_test│
                    │  - Probabilités      │
                    └──────────┬───────────┘
                               ↓
                    ┌──────────────────────┐
                    │     Évaluation       │
                    │  - Métriques         │
                    │  - Matrice confusion │
                    │  - Courbes ROC       │
                    └──────────────────────┘
```

### Détail du pipeline d'entraînement (`train.py`)

Le script `train.py` orchestre l'ensemble du pipeline d'entraînement :

1. **Chargement du dataset** : Lit le fichier CSV et extrait les colonnes pertinentes
2. **Exécution du pipeline de prétraitement** : Nettoie, normalise et sépare les données
3. **Entraînement des 6 modèles** : Chaque modèle est entraîné avec chronométrage
4. **Évaluation** : Calcul de toutes les métriques (accuracy, précision, rappel, F1, matrice)
5. **Comparaison** : Classement des modèles par F1-Score avec tableau formaté
6. **Sauvegarde** : Meilleur modèle et rapport d'entraînement sauvegardés dans `data/processed/`

### Gestion des erreurs

Le système inclut une gestion complète des erreurs :
- Détection et rapport des colonnes manquantes
- Gestion des valeurs aberrantes (NaN, Inf)
- Messages d'erreur explicites en cas d'échec
- Journalisation (logging) de toutes les étapes

## 5.4 Interface utilisateur

### Interface en ligne de commande (CLI)

L'interface CLI (`main.py`) permet d'exécuter le pipeline complet depuis le terminal :

```bash
# Pipeline complet avec le modèle par défaut (Random Forest)
python main.py

# Utiliser un modèle spécifique
python main.py --modele svm

# Utiliser le dataset CICIDS2017
python main.py --dataset cicids2017

# Comparer tous les modèles
python main.py --compare
```

### Interface web Streamlit (`app.py`)

L'interface Streamlit moderne et interactive permet :

1. **Chargement du modèle** : Sélection d'un modèle entraîné dans la barre latérale
2. **Upload de données** : Import d'un fichier CSV via `st.file_uploader`
3. **Prédiction** : Détection avec affichage codé par couleur (vert = normal, rouge = attaque)
4. **Performances** : Affichage automatique des métriques si le fichier contient une colonne label
5. **Graphiques** : Distribution des prédictions, importance des caractéristiques, matrice de confusion
6. **Export** : Téléchargement des résultats en CSV

### Script de démonstration (`demo.py`)

Le script `demo.py` génère rapidement un dataset synthétique et entraîne un modèle pour une démonstration rapide :

```bash
python demo.py
```

### Génération des graphiques (`generate_graphiques.py`)

Ce script génère automatiquement les 10 graphiques nécessaires au mémoire :

```bash
python generate_graphiques.py
```

Les graphiques sont sauvegardés dans `data/processed/graphiques/` :
1. Distribution des classes
2. Boxplots des caractéristiques
3. Matrice de corrélation
4. Courbes ROC avec AUC
5. Matrice de confusion du meilleur modèle
6. Comparaison des métriques (barres groupées)
7. Importance des caractéristiques
8. Temps d'entraînement
9. Détection vs faux positifs
10. Tableau comparatif

## 5.5 Tests unitaires

Le projet comprend **93 tests unitaires** couvrant :

- **Tests des modèles** (36 tests) : Initialisation, entraînement, prédiction, probabilités pour les 6 modèles
- **Tests de l'évaluateur** (30 tests) : Métriques, matrice de confusion, rapport, taux de détection
- **Tests du pipeline** (27 tests) : Chargement, nettoyage, normalisation, cas aux limites

Les tests utilisent le framework **Pytest** et peuvent être exécutés avec :

```bash
python -m pytest tests/ -v
```

La couverture de test est élevée, couvrant les cas normaux, les cas limites (NaN, infini, colonnes manquantes) et les cas d'erreur.

---

# Chapitre 6 : Résultats

## 6.1 Performances des modèles

Les six modèles ont été entraînés et évalués sur le dataset synthétique (2000 échantillons, 30 caractéristiques). Les résultats sont présentés ci-dessous.

### Tableau comparatif des performances

| Rang | Modèle | Accuracy | Précision | Rappel | F1-Score | Détection | Faux + |
|:----:|--------|:-------:|:---------:|:-----:|:--------:|:---------:|:-----:|
| **1** | **Random Forest** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **100%** | **0%** |
| 2 | Régression logistique | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 100% | 0% |
| 3 | SVM | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 100% | 0% |
| 4 | Naive Bayes | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 100% | 0% |
| 5 | KNN | 0.9972 | 1.0000 | 0.9944 | 0.9972 | 99.44% | 0% |
| 6 | Decision Tree | 0.9060 | 0.9298 | 0.8833 | 0.9060 | 88.33% | 0% |

![Tableau comparatif](../data/processed/graphiques/10_tableau_comparatif.png)

*Graphique 6.1 : Tableau comparatif des 6 modèles classés par F1-Score*

### Analyse détaillée par modèle

**Random Forest (F1-Score : 1.0000) — Meilleur modèle**
Le Random Forest atteint une performance parfaite sur les données synthétiques. Toutes les attaques sont détectées (Rappel = 1.0) et aucune fausse alarme n'est générée (Précision = 1.0). Ce résultat s'explique par sa capacité à combiner les décisions de multiples arbres, ce qui lui confère une grande robustesse face aux variations des données. Le Random Forest est le modèle recommandé pour ce projet.

**Régression logistique (F1-Score : 1.0000)**
La régression logistique atteint également une performance parfaite, ce qui s'explique par la nature des données synthétiques qui sont approximativement séparable linéairement. Sa rapidité d'exécution et ses probabilités bien calibrées en font une alternative intéressante.

**SVM (F1-Score : 1.0000)**
Le SVM avec noyau RBF atteint aussi une performance parfaite. Grâce à l'astuce du noyau, il peut capturer des relations non linéaires dans les données. Cependant, son temps d'entraînement est plus long que celui de la régression logistique ou du Random Forest.

**Naive Bayes (F1-Score : 1.0000)**
Malgré son hypothèse d'indépendance simpliste, Naive Bayes atteint également une performance parfaite. Cela s'explique par la distribution gaussienne des caractéristiques, qui correspond exactement à l'hypothèse sous-jacente du Gaussian Naive Bayes.

**KNN (F1-Score : 0.9972)**
KNN obtient un score quasi parfait mais légèrement inférieur aux autres. Avec 5 voisins et une pondération par la distance, il classe correctement la quasi-totalité des échantillons. Les quelques erreurs concernent des échantillons proches de la frontière de décision entre les deux classes.

**Decision Tree (F1-Score : 0.9060)**
L'arbre de décision est le modèle le moins performant avec un F1-Score de 0.9060. Bien qu'il soit rapide et interprétable, il est sujet au surapprentissage (overfitting) et ses performances sont inférieures à celles des méthodes d'ensemble comme le Random Forest. C'est un résultat attendu : les arbres de décision uniques sont rarement compétitifs face aux méthodes d'ensemble.

![Comparaison des métriques](../data/processed/graphiques/06_comparaison_metriques.png)

*Graphique 6.2 : Comparaison des métriques (Accuracy, Précision, Rappel, F1-Score) par modèle*

### Temps d'entraînement

![Temps d'entraînement](../data/processed/graphiques/08_temps_entrainement.png)

*Graphique 6.3 : Temps d'entraînement en secondes par modèle*

Les temps d'entraînement varient significativement selon les modèles :
- Les modèles les plus rapides : Naive Bayes, Decision Tree, KNN (quelques centièmes de seconde)
- Les modèles intermédiaires : Régression logistique, Random Forest (quelques dixièmes de seconde)
- Le modèle le plus lent : SVM (plusieurs secondes, dû à l'option `probability=True`)

### Matrice de confusion du meilleur modèle

![Matrice de confusion](../data/processed/graphiques/05_matrice_confusion.png)

*Graphique 6.4 : Matrice de confusion du Random Forest sur l'ensemble de test*

La matrice de confusion confirme la performance parfaite du Random Forest :
- **211 Vrais Négatifs** : trafic normal correctement classifié
- **90 Vrais Positifs** : attaques correctement détectées
- **0 Faux Positifs** : aucune fausse alerte
- **0 Faux Négatifs** : aucune attaque manquée

### Courbes ROC

![Courbes ROC](../data/processed/graphiques/04_courbes_roc.png)

*Graphique 6.5 : Courbes ROC des 6 modèles avec valeurs d'AUC*

Les courbes ROC montrent que la plupart des modèles atteignent une AUC parfaite (1.0000), indiquant une excellente capacité de discrimination entre les classes normale et malveillante. La courbe ROC du classifieur aléatoire (diagonale en pointillés, AUC = 0.5) est fournie comme référence. Les courbes ROC qui passent par le point (0,1) indiquent un classifieur parfait avec 0% de faux positifs et 100% de détection.

### Importance des caractéristiques

![Importance des caractéristiques](../data/processed/graphiques/07_importance_caracteristiques.png)

*Graphique 6.6 : Top 15 caractéristiques les plus importantes selon le Random Forest*

Le Random Forest fournit une estimation de l'importance de chaque caractéristique. Les caractéristiques les plus importantes sont celles qui contribuent le plus à la réduction de l'impureté dans les arbres de décision. Cette information est précieuse pour :
- Comprendre quels aspects du trafic réseau sont les plus discriminants
- Réduire la dimensionnalité en ne conservant que les caractéristiques les plus importantes
- Interpréter le comportement du modèle

### Taux de détection vs Faux positifs

![Détection vs Faux positifs](../data/processed/graphiques/09_detection_vs_faux_positifs.png)

*Graphique 6.7 : Taux de détection en fonction du taux de faux positifs pour chaque modèle*

Ce graphique positionne chaque modèle selon deux critères essentiels en sécurité : la capacité à détecter les attaques (TPR) et la tendance à générer de fausses alarmes (FPR). La zone idéale se situe en haut à gauche du graphique (détection maximale, faux positifs minimaux).

## 6.2 Analyse comparative

### Quel modèle est le meilleur ?

Le **Random Forest** est le meilleur modèle pour plusieurs raisons :

1. **Performance maximale** : F1-Score parfait (1.0000), 100% de détection, 0% de faux positifs
2. **Robustesse** : Méthode d'ensemble qui résiste au surapprentissage
3. **Interprétabilité** : Fournit l'importance des caractéristiques
4. **Polyvalence** : Performant sur différents types de données

### Pourquoi le Random Forest est-il le meilleur ?

Le Random Forest combine plusieurs avantages qui en font le modèle le plus adapté à la détection d'intrusions :

1. **Réduction de la variance** : En moyennant les prédictions de nombreux arbres, le Random Forest réduit considérablement la variance par rapport à un arbre unique, sans augmenter le biais.

2. **Gestion de la dimensionnalité** : Le Random Forest gère efficacement un grand nombre de caractéristiques, ce qui est crucial pour la détection d'intrusions où les données sont souvent de haute dimension.

3. **Robustesse au bruit** : La sélection aléatoire des caractéristiques à chaque nœud rend le modèle résistant au bruit et aux caractéristiques non pertinentes.

4. **Évaluation intégrée** : L'estimation de l'importance des caractéristiques et la validation croisée out-of-bag (OOB) sont fournies gratuitement.

### Comparaison avec les autres modèles

- **Decision Tree** : Trop sujet au surapprentissage pour être utilisé seul
- **Régression logistique** : Rapide et efficace mais limitée aux relations linéaires
- **SVM** : Excellent mais plus lent, moins adapté aux grands volumes de données
- **KNN** : Simple mais lent en prédiction, sensible à la dimensionnalité
- **Naive Bayes** : Très rapide mais moins performant sur des données corrélées

## 6.3 Discussion et limites

### Limites de l'étude

1. **Données synthétiques** : Les résultats présentés ont été obtenus sur un dataset synthétique avec une séparation nette entre les classes. Sur un dataset réel comme CICIDS2017, les performances seraient probablement moins parfaites, avec un déséquilibre des classes plus marqué et un bruit plus important.

2. **Taille du dataset** : Avec 2000 échantillons, le dataset synthétique est relativement petit. Un dataset plus grand (des dizaines ou centaines de milliers d'échantillons) permettrait une évaluation plus robuste des modèles.

3. **Hyperparamètres non optimisés** : Les hyperparamètres des modèles ont été fixés manuellement avec des valeurs raisonnables par défaut. Une optimisation systématique (GridSearchCV, RandomSearch) pourrait améliorer les performances.

4. **Pas de validation croisée** : L'évaluation a été réalisée avec une simple séparation train/test (70/30). Une validation croisée (k-fold cross-validation) donnerait une estimation plus fiable des performances.

5. **Classes binaires uniquement** : La classification se limite à deux classes (normal vs attaque). Une classification multi-classes (type d'attaque spécifique) serait plus informative.

6. **Pas de XGBoost/LightGBM** : Ces algorithmes avancés de boosting, mentionnés dans la revue de littérature, n'ont pas été implémentés dans le projet. Leur ajout permettrait une comparaison plus complète.

### Pistes d'amélioration

1. **Dataset réel CICIDS2017** : Utiliser le dataset complet (80+ caractéristiques, trafic réel) pour des résultats plus réalistes
2. **Optimisation des hyperparamètres** : GridSearchCV ou Optuna pour trouver les meilleurs paramètres
3. **Validation croisée** : K-fold (k=5 ou 10) pour une évaluation plus robuste
4. **Détection multi-classes** : Identifier le type d'attaque (DoS, DDoS, brute force, etc.)
5. **Deep Learning** : Réseaux de neurones pour des performances potentiellement supérieures

---

> **Remarque** : Les résultats parfaits (F1=1.0 pour 4 modèles) s'expliquent par la nature synthétique du dataset. Sur des données réelles (CICIDS2017), les performances seraient plus nuancées, bien que le Random Forest conserve généralement son avance.

# Conclusion

## Résumé du travail

Ce mémoire a présenté la conception et le développement de **NetGuard AI**, un système intelligent de détection d'intrusions réseau basé sur l'apprentissage automatique.

Le travail accompli comprend :

1. **Une revue de littérature complète** sur les réseaux informatiques, les menaces réseau, les systèmes de détection d'intrusions et les algorithmes de Machine Learning.

2. **Le développement d'un pipeline de prétraitement robuste** en huit étapes : chargement, nettoyage, gestion des valeurs manquantes, encodage, normalisation, sélection des caractéristiques, séparation train/test et sauvegarde.

3. **L'implémentation de six modèles de classification supervisée** : Decision Tree, Random Forest, Logistic Regression, SVM, KNN et Naive Bayes, chacun avec ses hyperparamètres spécifiques.

4. **Un module d'évaluation complet** calculant sept métriques : accuracy, précision, rappel, F1-score, matrice de confusion, taux de détection et taux de faux positifs.

5. **Une interface utilisateur moderne** incluant une application web Streamlit, une interface en ligne de commande et un script de démonstration.

6. **93 tests unitaires** garantissant la fiabilité du code.

7. **10 graphiques** générés automatiquement pour visualiser les résultats.

## Réponse à la problématique

**Comment concevoir un système intelligent capable de détecter efficacement les intrusions réseau grâce aux techniques de Machine Learning ?**

Ce projet démontre qu'un système de détection d'intrusions basé sur le Machine Learning est non seulement réalisable mais aussi efficace. La réponse à la problématique repose sur plusieurs piliers :

1. **Choix des algorithmes** : Le Random Forest s'impose comme le meilleur compromis entre performance (F1-Score parfait), robustesse et interprétabilité. Sa nature d'ensemble lui permet de résister au surapprentissage tout en capturant des relations complexes dans les données.

2. **Prétraitement rigoureux** : La qualité des prédictions dépend directement de la qualité du prétraitement. Le pipeline développé garantit des données propres, normalisées et prêtes pour l'entraînement.

3. **Évaluation multi-critères** : L'utilisation de métriques complémentaires (Accuracy, Précision, Rappel, F1-Score, courbe ROC) permet une évaluation nuancée des performances, essentielle en sécurité où les coûts des faux positifs et des faux négatifs sont très différents.

4. **Architecture modulaire** : La conception du système (packages séparés par responsabilité, configuration centralisée, tests unitaires) facilite la maintenance, l'extension et la réutilisation du code.

Le système développé, bien que testé principalement sur des données synthétiques, constitue une preuve de concept solide pour une solution de détection d'intrusions par Machine Learning.

## Perspectives

Plusieurs perspectives d'amélioration et d'extension du projet sont envisageables :

### Deep Learning

L'utilisation de réseaux de neurones profonds (Deep Learning) pourrait améliorer les performances, notamment sur des données réelles complexes. Les architectures prometteuses incluent :
- **MLP (Multi-Layer Perceptron)** : Réseau fully-connected pour la classification
- **CNN (Convolutional Neural Network)** : Adaptation du trafic réseau en images pour utiliser des convolutions
- **LSTM (Long Short-Term Memory)** : Capture des dépendances temporelles dans les flux réseau
- **Autoencoders** : Détection d'anomalies par reconstruction

### Détection en temps réel

Le système pourrait être adapté pour une détection en temps réel en :
- Utilisant un buffer circulaire pour l'analyse en continu des flux
- Implémentant un système de fenêtrage temporel glissant
- Optimisant les modèles pour l'inférence rapide (ONNX, TensorRT)
- Parallélisant le traitement avec multiprocessing

### Intégration avec un SIEM

L'intégration avec un Security Information and Event Management (SIEM) permettrait :
- L'envoi automatique des alertes vers une plateforme centralisée
- La corrélation avec d'autres sources de sécurité
- Le déclenchement de réponses automatiques (containment, blocking)
- La génération de rapports de sécurité automatisés

### Déploiement sur un réseau réel

Le déploiement sur un réseau réel nécessiterait :
- L'utilisation de CICIDS2017 ou d'un dataset réel pour l'entraînement
- L'adaptation au trafic spécifique du réseau cible
- La mise en place d'une boucle de feedback pour améliorer le modèle
- Le respect des contraintes de performance (latence, débit)

### Extensions fonctionnelles

- **Classification multi-classes** : Identifier le type spécifique d'attaque (DoS, DDoS, brute force, etc.)
- **Détection d'anomalies non supervisée** : Détecter des comportements inconnus sans données étiquetées
- **Explicabilité (XAI)** : Utiliser SHAP ou LIME pour expliquer les décisions du modèle
- **Dashboard temps réel** : Interface web avec graphiques dynamiques et alertes en direct

---

# Bibliographie

1. **Stuart J. Russell, Peter Norvig** (2020). *Artificial Intelligence: A Modern Approach* (4e édition). Pearson.

2. **Trevor Hastie, Robert Tibshirani, Jerome Friedman** (2009). *The Elements of Statistical Learning* (2e édition). Springer.

3. **Leo Breiman** (2001). "Random Forests". *Machine Learning*, 45(1), 5-32.

4. **Corinna Cortes, Vladimir Vapnik** (1995). "Support-Vector Networks". *Machine Learning*, 20(3), 273-297.

5. **Tianqi Chen, Carlos Guestrin** (2016). "XGBoost: A Scalable Tree Boosting System". *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*.

6. **Guolin Ke et al.** (2017). "LightGBM: A Highly Efficient Gradient Boosting Decision Tree". *Advances in Neural Information Processing Systems*, 30.

7. **Iman Sharafaldin, Arash Habibi Lashkari, Ali A. Ghorbani** (2018). "Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization". *4th International Conference on Information Systems Security and Privacy (ICISSP)*.

8. **Moustafa Nour, Slay Jill** (2015). "UNSW-NB15: A comprehensive data set for network intrusion detection systems". *Military Communications and Information Systems Conference (MilCIS)*.

9. **Mahbod Tavallaee et al.** (2009). "A detailed analysis of the KDD CUP 99 data set". *IEEE Symposium on Computational Intelligence for Security and Defense Applications*.

10. **Scikit-learn Developers** (2023). *Scikit-learn: Machine Learning in Python*. https://scikit-learn.org/stable/documentation.html

11. **Streamlit Inc.** (2023). *Streamlit Documentation*. https://docs.streamlit.io/

12. **François Chollet** (2021). *Deep Learning with Python* (2e édition). Manning Publications.

13. **Géron, Aurélien** (2022). *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow* (3e édition). O'Reilly Media.

14. **McKinney, Wes** (2022). *Python for Data Analysis* (3e édition). O'Reilly Media.

15. **Pedregosa, Fabian et al.** (2011). "Scikit-learn: Machine Learning in Python". *Journal of Machine Learning Research*, 12, 2825-2830.

16. **K. Scarfone, P. Mell** (2007). "Guide to Intrusion Detection and Prevention Systems (IDPS)". *NIST Special Publication 800-94*.

17. **R. Sommer, V. Paxson** (2010). "Outside the Closed World: On Using Machine Learning for Network Intrusion Detection". *IEEE Symposium on Security and Privacy*.

18. **M. Z. Alom, T. Taha, R. Yakopcic** (2017). "Network intrusion detection for cyber security using unsupervised deep learning approaches". *IEEE National Aerospace and Electronics Conference (NAECON)*.

---

*Document généré le 9 juillet 2026*

*Projet NetGuard AI — Master en Informatique — Année universitaire 2025-2026*
