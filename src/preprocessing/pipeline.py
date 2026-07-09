"""
Pipeline de prétraitement des données pour NetGuard AI.

Ce module implémente un pipeline linéaire et autonome qui enchaîne
les étapes de prétraitement dans un ordre précis :

    1. Chargement du dataset depuis un fichier CSV
    2. Suppression des colonnes inutiles (identifiants, timestamps)
    3. Suppression des doublons
    4. Gestion des valeurs manquantes (NaN, Infinity)
    5. Encodage des variables catégorielles
    6. Normalisation des caractéristiques numériques
    7. Séparation en ensembles d'entraînement et de test
    8. Sauvegarde des données nettoyées

Chaque étape produit des logs clairs et un résumé des transformations.
"""

import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

from src.config import (
    DOSSIER_DATA_BRUT,
    DOSSIER_DATA_TRANSFORME,
    FICHIER_DATASET,
    SEPARATEUR_CSV,
    FRACTION_ENTRAINEMENT,
    COLONNE_CIBLE,
    COLONNES_A_IGNORER,
    SEED_ALEATOIRE,
)


class PipelinePretraitement:
    """
    Pipeline de prétraitement des données réseau.

    Utilisation :
        pipeline = PipelinePretraitement()
        pipeline.charger("chemin/vers/dataset.csv")
        pipeline.supprimer_colonnes_inutiles()
        pipeline.supprimer_doublons()
        pipeline.gerer_valeurs_manquantes()
        pipeline.encoder_variables_categorielles()
        pipeline.normaliser_donnees()
        pipeline.separer_train_test()
        pipeline.sauvegarder_donnees()
        # ou bien :
        pipeline.executer_pipeline_complet()
    """

    def __init__(
        self,
        chemin_dataset: Optional[Path] = None,
        colonne_label: str = COLONNE_CIBLE,
        colonnes_a_ignorer: Optional[list] = None,
        taille_test: float = 0.3,
    ):
        """
        Initialise le pipeline de prétraitement.

        Args:
            chemin_dataset: Chemin vers le fichier CSV d'entrée.
            colonne_label: Nom de la colonne contenant les labels.
            colonnes_a_ignorer: Liste des colonnes à supprimer.
            taille_test: Proportion des données réservée au test (defaut: 0.3).
        """
        self.chemin_dataset = chemin_dataset
        self.colonne_label = colonne_label
        self.colonnes_a_ignorer = (
            colonnes_a_ignorer if colonnes_a_ignorer is not None
            else COLONNES_A_IGNORER.copy()
        )
        self.taille_test = taille_test

        # Données brutes et transformées
        self.donnees: Optional[pd.DataFrame] = None
        self.X: Optional[pd.DataFrame] = None
        self.y: Optional[pd.Series] = None
        self.X_train: Optional[np.ndarray] = None
        self.X_test: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None
        self.y_test: Optional[np.ndarray] = None

        # Transformateurs (sauvegardés pour réutilisation)
        self.encodage_label: Optional[LabelEncoder] = None
        self.standardisation: Optional[StandardScaler] = None

        # Métadonnées du pipeline
        self.metadonnees: dict = {
            "nb_lignes_initial": 0,
            "nb_colonnes_initial": 0,
            "nb_doublons_supprimes": 0,
            "nb_valeurs_manquantes_supprimees": 0,
            "colonnes_supprimees": [],
            "colonnes_categorielles_encodes": [],
            "nb_caracteristiques_final": 0,
        }

        logging.info(
            "Pipeline de prétraitement initialisé. "
            "Colonne label : '%s'", self.colonne_label
        )

    # ─── Étape 1 : Chargement du dataset ─────────────────────────────────────

    def charger(self, chemin: Optional[Path] = None) -> pd.DataFrame:
        """
        Étape 1 : Charge le dataset depuis un fichier CSV.

        Lit le fichier CSV et stocke les données dans self.donnees.
        Affiche un résumé des dimensions et des types de colonnes.

        Args:
            chemin: Chemin vers le fichier CSV. Si None, utilise
                    le chemin par défaut (data/raw/network_traffic.csv).

        Returns:
            DataFrame contenant les données brutes chargées.

        Lève:
            FileNotFoundError: Si le fichier est introuvable.
        """
        if chemin is not None:
            self.chemin_dataset = chemin

        if self.chemin_dataset is None:
            self.chemin_dataset = DOSSIER_DATA_BRUT / FICHIER_DATASET

        if not self.chemin_dataset.exists():
            raise FileNotFoundError(
                f"Fichier introuvable : {self.chemin_dataset}\n"
                "Vérifiez le chemin ou exécutez d'abord 'python demo.py' "
                "pour générer des données synthétiques."
            )

        # Affichage du nom du fichier
        nom_fichier = self.chemin_dataset.name
        taille_fichier = self.chemin_dataset.stat().st_size / 1024  # Ko
        logging.info("─" * 50)
        logging.info("ÉTAPE 1 : Chargement du dataset")
        logging.info("─" * 50)
        logging.info("Fichier : %s (%.1f Ko)", nom_fichier, taille_fichier)

        # Chargement du CSV
        self.donnees = pd.read_csv(self.chemin_dataset, sep=SEPARATEUR_CSV)

        # Enregistrement des métadonnées
        self.metadonnees["nb_lignes_initial"] = len(self.donnees)
        self.metadonnees["nb_colonnes_initial"] = len(self.donnees.columns)

        # Résumé du chargement
        logging.info("Dimensions : %d lignes × %d colonnes",
                      self.donnees.shape[0], self.donnees.shape[1])
        logging.info("Colonnes : %s", ", ".join(self.donnees.columns[:8].tolist()) +
                      ("..." if len(self.donnees.columns) > 8 else ""))
        logging.info("Types : %s",
                      dict(self.donnees.dtypes.value_counts()))
        logging.info("Chargement terminé avec succès.")

        return self.donnees

    # ─── Étape 2 : Suppression des colonnes inutiles ─────────────────────────

    def supprimer_colonnes_inutiles(self) -> pd.DataFrame:
        """
        Étape 2 : Supprime les colonnes non pertinentes.

        Colonnes supprimées :
        - Identifiants de flux (Flow ID, IP, ports)
        - Timestamps et horodatages
        - Toute colonne configurée dans colonnes_a_ignorer

        Returns:
            DataFrame sans les colonnes ignorées.
        """
        if self.donnees is None:
            raise RuntimeError(
                "Aucune donnée chargée. Appelez d'abord la méthode 'charger()'."
            )

        logging.info("─" * 50)
        logging.info("ÉTAPE 2 : Suppression des colonnes inutiles")
        logging.info("─" * 50)

        # Identification des colonnes présentes dans le dataset
        colonnes_a_supprimer = [
            col for col in self.colonnes_a_ignorer
            if col in self.donnees.columns
        ]

        if colonnes_a_supprimer:
            self.donnees.drop(columns=colonnes_a_supprimer, inplace=True)
            self.metadonnees["colonnes_supprimees"] = colonnes_a_supprimer
            logging.info("Colonnes supprimées (%d) : %s",
                          len(colonnes_a_supprimer),
                          ", ".join(colonnes_a_supprimer))
        else:
            logging.info("Aucune colonne à supprimer (déjà nettoyé).")

        logging.info("Dimensions après nettoyage : %d × %d",
                      self.donnees.shape[0], self.donnees.shape[1])

        return self.donnees

    # ─── Étape 3 : Suppression des doublons ──────────────────────────────────

    def supprimer_doublons(self) -> pd.DataFrame:
        """
        Étape 3 : Supprime les lignes en double.

        Les doublons sont identifiés sur l'ensemble des colonnes
        (sauf la colonne label si elle existe). La première occurrence
        est conservée.

        Returns:
            DataFrame sans les lignes dupliquées.
        """
        if self.donnees is None:
            raise RuntimeError(
                "Aucune donnée chargée. Appelez d'abord la méthode 'charger()'."
            )

        logging.info("─" * 50)
        logging.info("ÉTAPE 3 : Suppression des doublons")
        logging.info("─" * 50)

        nb_avant = len(self.donnees)

        # Suppression des doublons (toutes les colonnes)
        self.donnees.drop_duplicates(inplace=True)

        nb_apres = len(self.donnees)
        nb_supprimes = nb_avant - nb_apres
        self.metadonnees["nb_doublons_supprimes"] = nb_supprimes

        if nb_supprimes > 0:
            logging.info("Doublons trouvés et supprimés : %d (%.1f%%)",
                          nb_supprimes,
                          nb_supprimes / nb_avant * 100)
        else:
            logging.info("Aucun doublon détecté.")

        logging.info("Lignes restantes : %d", nb_apres)

        return self.donnees

    # ─── Étape 4 : Gestion des valeurs manquantes ────────────────────────────

    def gerer_valeurs_manquantes(self) -> pd.DataFrame:
        """
        Étape 4 : Gère les valeurs manquantes et invalides.

        Traite les cas suivants :
        - NaN (Not a Number) : suppression des lignes concernées
        - Inf / -Inf (Infinity) : conversion en NaN puis suppression
        - Valeurs aberrantes optionnelles

        Returns:
            DataFrame sans valeurs manquantes.
        """
        if self.donnees is None:
            raise RuntimeError(
                "Aucune donnée chargée. Appelez d'abord la méthode 'charger()'."
            )

        logging.info("─" * 50)
        logging.info("ÉTAPE 4 : Gestion des valeurs manquantes")
        logging.info("─" * 50)

        nb_avant = len(self.donnees)

        # 4a : Vérification des NaN
        nb_nan_total = self.donnees.isnull().sum().sum()
        if nb_nan_total > 0:
            logging.info("Valeurs manquantes (NaN) détectées : %d", nb_nan_total)

            # Afficher les colonnes concernées
            colonnes_avec_nan = self.donnees.isnull().sum()
            colonnes_avec_nan = colonnes_avec_nan[colonnes_avec_nan > 0]
            for col, count in colonnes_avec_nan.items():
                logging.info("  - %s : %d valeurs manquantes", col, count)
        else:
            logging.info("Aucune valeur NaN détectée.")

        # 4b : Conversion des valeurs Infinity en NaN
        colonnes_numeriques = self.donnees.select_dtypes(include=[np.number]).columns
        nb_inf = 0
        for col in colonnes_numeriques:
            nb_inf_col = self.donnees[col].isin([np.inf, -np.inf]).sum()
            nb_inf += nb_inf_col

        if nb_inf > 0:
            logging.info("Valeurs infinies détectées : %d", nb_inf)
            self.donnees.replace([np.inf, -np.inf], np.nan, inplace=True)
            logging.info("Valeurs infinies converties en NaN.")
        else:
            logging.info("Aucune valeur infinie détectée.")

        # 4c : Suppression des lignes avec des valeurs manquantes
        nb_nan_apres_conversion = self.donnees.isnull().sum().sum()
        if nb_nan_apres_conversion > 0:
            self.donnees.dropna(inplace=True)
            nb_supprimees = nb_avant - len(self.donnees)
            self.metadonnees["nb_valeurs_manquantes_supprimees"] = nb_supprimees
            logging.info("Lignes avec valeurs manquantes supprimées : %d",
                          nb_supprimees)
        else:
            logging.info("Aucune suppression nécessaire.")

        logging.info("Lignes restantes : %d", len(self.donnees))

        return self.donnees

    # ─── Étape 5 : Encodage des variables catégorielles ──────────────────────

    def encoder_variables_categorielles(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Étape 5 : Encode les variables catégorielles.

        Transforme les colonnes texte en valeurs numériques :
        - LabelEncoder pour la colonne cible (label)
        - Les colonnes catégorielles sont encodées en One-Hot ou supprimées

        Returns:
            Tuple (X, y) où X contient les caractéristiques numériques
            et y les labels encodés.
        """
        if self.donnees is None:
            raise RuntimeError(
                "Aucune donnée chargée. Appelez d'abord la méthode 'charger()'."
            )

        logging.info("─" * 50)
        logging.info("ÉTAPE 5 : Encodage des variables catégorielles")
        logging.info("─" * 50)

        # Séparation des caractéristiques (X) et de la cible (y)
        if self.colonne_label not in self.donnees.columns:
            raise ValueError(
                f"Colonne cible '{self.colonne_label}' introuvable.\n"
                f"Colonnes disponibles : {list(self.donnees.columns)}"
            )

        self.X = self.donnees.drop(columns=[self.colonne_label])
        self.y = self.donnees[self.colonne_label]

        # 5a : Encodage de la colonne cible (label)
        logging.info("Encodage de la colonne cible : '%s'", self.colonne_label)
        self.encodage_label = LabelEncoder()
        self.y = pd.Series(
            self.encodage_label.fit_transform(self.y),
            name=self.colonne_label
        )

        # Affichage des classes
        classes = list(self.encodage_label.classes_)
        mapping = {i: classes[i] for i in range(len(classes))}
        logging.info("Classes détectées : %s", mapping)
        logging.info("Distribution :")
        for code, label in mapping.items():
            count = (self.y == code).sum()
            logging.info("  - %s → %d : %d échantillons (%.1f%%)",
                          label, code, count,
                          count / len(self.y) * 100)

        # 5b : Encodage des colonnes catégorielles dans X
        colonnes_categorielles = self.X.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()

        if colonnes_categorielles:
            logging.info("Variables catégorielles dans X : %s",
                          ", ".join(colonnes_categorielles))

            # One-Hot Encoding pour les variables catégorielles
            self.X = pd.get_dummies(
                self.X,
                columns=colonnes_categorielles,
                drop_first=True,  # Éviter la multicolinéarité
            )

            self.metadonnees["colonnes_categorielles_encodes"] = (
                colonnes_categorielles
            )
            logging.info("Encodage One-Hot effectué pour %d variables.",
                          len(colonnes_categorielles))
        else:
            logging.info("Aucune variable catégorielle dans X détectée.")

        # Conversion explicite en float64 pour éviter les erreurs scikit-learn
        self.X = self.X.astype(np.float64)
        logging.info("Conversion en float64 effectuée.")

        logging.info("Dimensions finales : X = %s, y = %s",
                      self.X.shape, self.y.shape)

        return self.X.values, self.y.values

    # ─── Étape 6 : Normalisation des données ─────────────────────────────────

    def normaliser_donnees(self) -> np.ndarray:
        """
        Étape 6 : Normalise les caractéristiques numériques.

        Applique une standardisation (Z-score) qui centre les données
        autour de 0 avec un écart-type de 1 :
            X_norm = (X - moyenne) / écart_type

        Returns:
            Caractéristiques normalisées (numpy array).
        """
        if self.X is None:
            raise RuntimeError(
                "Aucune caractéristique. Appelez d'abord "
                "'encoder_variables_categorielles()'."
            )

        logging.info("─" * 50)
        logging.info("ÉTAPE 6 : Normalisation des données (Z-score)")
        logging.info("─" * 50)

        if self.X.shape[1] == 0:
            raise ValueError("Aucune caractéristique à normaliser.")

        # Initialisation du StandardScaler
        self.standardisation = StandardScaler()
        X_normalise = self.standardisation.fit_transform(self.X)

        # Statistiques de la normalisation
        logging.info("Moyennes avant normalisation : min=%.4f, max=%.4f",
                      self.X.values.mean(axis=0).min(),
                      self.X.values.mean(axis=0).max())
        logging.info("Écarts-types avant normalisation : min=%.4f, max=%.4f",
                      self.X.values.std(axis=0).min(),
                      self.X.values.std(axis=0).max())

        # Vérification après normalisation
        logging.info("Moyennes après normalisation : min=%.4f, max=%.4f",
                      X_normalise.mean(axis=0).min(),
                      X_normalise.mean(axis=0).max())
        logging.info("Écarts-types après normalisation : min=%.4f, max=%.4f",
                      X_normalise.std(axis=0).min(),
                      X_normalise.std(axis=0).max())

        self.metadonnees["nb_caracteristiques_final"] = X_normalise.shape[1]

        logging.info("Normalisation terminée : %d caractéristiques.",
                      X_normalise.shape[1])

        return X_normalise

    # ─── Étape 7 : Séparation entraînement / test ────────────────────────────

    def separer_train_test(
        self,
        X: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Étape 7 : Sépare les données en ensembles d'entraînement et de test.

        La séparation est stratifiée pour préserver la proportion
        de chaque classe dans les deux ensembles.

        Args:
            X: Caractéristiques (si None, utilise self.X.values).
            y: Labels (si None, utilise self.y.values).

        Returns:
            Tuple (X_train, X_test, y_train, y_test).
        """
        if X is None and self.X is not None:
            X = self.X.values
        if y is None and self.y is not None:
            y = self.y.values

        if X is None or y is None:
            raise RuntimeError(
                "Données non disponibles. Appelez d'abord "
                "'encoder_variables_categorielles()'."
            )

        logging.info("─" * 50)
        logging.info("ÉTAPE 7 : Séparation entraînement / test")
        logging.info("─" * 50)

        # Vérification de la stratificación
        classes, counts = np.unique(y, return_counts=True)
        logging.info("Distribution des classes :")
        for cls, count in zip(classes, counts):
            logging.info("  - Classe %s : %d (%.1f%%)",
                          cls, count, count / len(y) * 100)

        # Split avec stratification
        self.X_train, self.X_test, self.y_train, self.y_test = (
            train_test_split(
                X,
                y,
                test_size=self.taille_test,
                random_state=SEED_ALEATOIRE,
                stratify=y,
            )
        )

        # Résumé de la séparation
        taille_train_pct = (1 - self.taille_test) * 100
        taille_test_pct = self.taille_test * 100
        logging.info("Ensemble d'entraînement : %d échantillons (%.0f%%)",
                      len(self.X_train), taille_train_pct)
        logging.info("Ensemble de test : %d échantillons (%.0f%%)",
                      len(self.X_test), taille_test_pct)

        # Vérification de la stratification
        _, counts_train = np.unique(self.y_train, return_counts=True)
        _, counts_test = np.unique(self.y_test, return_counts=True)
        logging.info("Distribution entraînement : %s", counts_train)
        logging.info("Distribution test : %s", counts_test)

        return self.X_train, self.X_test, self.y_train, self.y_test

    # ─── Étape 8 : Sauvegarde des données nettoyées ──────────────────────────

    def sauvegarder_donnees(
        self,
        dossier_sortie: Optional[Path] = None,
        prefixe: str = "netguard_clean",
    ) -> dict:
        """
        Étape 8 : Sauvegarde les données prétraitées sur le disque.

        Sauvegarde 4 fichiers :
        - {prefixe}_X_train.csv : Caractéristiques d'entraînement
        - {prefixe}_X_test.csv  : Caractéristiques de test
        - {prefixe}_y_train.csv : Labels d'entraînement
        - {prefixe}_y_test.csv  : Labels de test
        - {prefixe}_resume.txt  : Résumé du pipeline

        Args:
            dossier_sortie: Dossier de destination (defaut: data/processed/).
            prefixe: Préfixe pour les noms de fichiers.

        Returns:
            Dictionnaire avec les chemins des fichiers sauvegardés.
        """
        if self.X_train is None or self.X_test is None:
            raise RuntimeError(
                "Données non séparées. Appelez d'abord "
                "'separer_train_test()'."
            )

        dossier_sortie = dossier_sortie or DOSSIER_DATA_TRANSFORME
        dossier_sortie.mkdir(parents=True, exist_ok=True)

        logging.info("─" * 50)
        logging.info("ÉTAPE 8 : Sauvegarde des données nettoyées")
        logging.info("─" * 50)

        # Chemins des fichiers
        chemins = {
            "X_train": dossier_sortie / f"{prefixe}_X_train.csv",
            "X_test": dossier_sortie / f"{prefixe}_X_test.csv",
            "y_train": dossier_sortie / f"{prefixe}_y_train.csv",
            "y_test": dossier_sortie / f"{prefixe}_y_test.csv",
            "resume": dossier_sortie / f"{prefixe}_resume.txt",
        }

        # Sauvegarde des fichiers CSV
        pd.DataFrame(self.X_train).to_csv(
            chemins["X_train"], index=False
        )
        logging.info("X_train sauvegardé : %s (%d × %d)",
                      chemins["X_train"].name,
                      self.X_train.shape[0], self.X_train.shape[1])

        pd.DataFrame(self.X_test).to_csv(
            chemins["X_test"], index=False
        )
        logging.info("X_test sauvegardé : %s (%d × %d)",
                      chemins["X_test"].name,
                      self.X_test.shape[0], self.X_test.shape[1])

        pd.DataFrame(self.y_train, columns=["label"]).to_csv(
            chemins["y_train"], index=False
        )
        logging.info("y_train sauvegardé : %s (%d)",
                      chemins["y_train"].name, len(self.y_train))

        pd.DataFrame(self.y_test, columns=["label"]).to_csv(
            chemins["y_test"], index=False
        )
        logging.info("y_test sauvegardé : %s (%d)",
                      chemins["y_test"].name, len(self.y_test))

        # Sauvegarde du résumé textuel
        with open(chemins["resume"], "w", encoding="utf-8") as f:
            f.write("=== Résumé du pipeline de prétraitement NetGuard AI ===\n\n")
            f.write(f"Dataset source : {self.chemin_dataset}\n")
            f.write(f"Lignes initiales : {self.metadonnees['nb_lignes_initial']}\n")
            f.write(f"Doublons supprimés : {self.metadonnees['nb_doublons_supprimes']}\n")
            f.write(f"Valeurs manquantes supprimées : {self.metadonnees['nb_valeurs_manquantes_supprimees']}\n")
            f.write(f"Colonnes supprimées : {self.metadonnees['colonnes_supprimees']}\n")
            f.write(f"Colonnes encodées : {self.metadonnees['colonnes_categorielles_encodes']}\n")
            f.write(f"Caractéristiques finales : {self.metadonnees['nb_caracteristiques_final']}\n")
            f.write(f"Taille entraînement : {len(self.X_train)}\n")
            f.write(f"Taille test : {len(self.X_test)}\n")

        logging.info("Résumé sauvegardé : %s", chemins["resume"])

        return chemins

    # ─── Exécution complète du pipeline ──────────────────────────────────────

    def executer_pipeline_complet(
        self,
        chemin_dataset: Optional[Path] = None,
        sauvegarder: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Exécute toutes les étapes du pipeline à la suite.

            Chargement → Nettoyage → Doublons → NaN →
            Encodage → Normalisation → Split → Sauvegarde

        Args:
            chemin_dataset: Chemin vers le fichier CSV.
            sauvegarder: Si True, sauvegarde les données traitées.

        Returns:
            Tuple (X_train, X_test, y_train, y_test).
        """
        logging.info("")
        logging.info("╔══════════════════════════════════════════════════╗")
        logging.info("║   PIPELINE DE PRÉTRAITEMENT NETGUARD AI         ║")
        logging.info("╚══════════════════════════════════════════════════╝")
        logging.info("")

        # Étape 1 : Chargement
        self.charger(chemin_dataset)

        # Étape 2 : Colonnes inutiles
        self.supprimer_colonnes_inutiles()

        # Étape 3 : Doublons
        self.supprimer_doublons()

        # Étape 4 : Valeurs manquantes
        self.gerer_valeurs_manquantes()

        # Étape 5 : Encodage
        self.encoder_variables_categorielles()

        # Étape 6 : Normalisation
        X_normalise = self.normaliser_donnees()

        # Étape 7 : Séparation
        self.separer_train_test(X_normalise, self.y.values)

        # Étape 8 : Sauvegarde (optionnelle)
        if sauvegarder:
            self.sauvegarder_donnees()

        # Résumé final
        logging.info("")
        logging.info("═" * 50)
        logging.info("PIPELINE TERMINÉ AVEC SUCCÈS")
        logging.info("═" * 50)
        logging.info("Dataset   : %s", self.chemin_dataset.name)
        logging.info("Échantillons : %d (train) + %d (test)",
                      len(self.X_train), len(self.X_test))
        logging.info("Caractéristiques : %d", self.X_train.shape[1])
        logging.info("Doublons supprimés : %d",
                      self.metadonnees["nb_doublons_supprimes"])
        logging.info("Valeurs manquantes traitées : %d",
                      self.metadonnees["nb_valeurs_manquantes_supprimees"])
        logging.info("═" * 50)
        logging.info("")

        return self.X_train, self.X_test, self.y_train, self.y_test
