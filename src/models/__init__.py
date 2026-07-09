"""
Package des modèles de Machine Learning pour NetGuard AI.

Contient 6 modèles de classification :
- Decision Tree
- Random Forest
- Logistic Regression
- SVM
- KNN
- Naive Bayes

Exposed via DetecteurIntrusions avec une interface unifiée.
"""

from src.models.detector import DetecteurIntrusions, creer_modele, FABRIQUE_MODELES
from src.models.compare import compare_modeles

__all__ = [
    "DetecteurIntrusions",
    "creer_modele",
    "FABRIQUE_MODELES",
    "compare_modeles",
]
