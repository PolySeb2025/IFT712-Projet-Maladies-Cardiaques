import seaborn as sns
from pathlib import Path
import time
import sys
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np # Ajouté pour les graphiques et la PCA

# Métriques Sklearn
from sklearn.metrics import f1_score, accuracy_score, classification_report
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA # Ajouté pour la visualisation

# --- Importation de nos modules ---
PROJECT_FCT = Path("../maladiescardiaques").resolve()
PROJECT_ROOT = Path("..").resolve()
sys.path.append(str(PROJECT_FCT))

print(f"Racine du projet ajoutée au PATH: {PROJECT_FCT}")

try:
    # Importer les fonctions depuis vos fichiers .py
    from traitementDonnées import load_and_split_data, create_preprocessing_pipeline
    from trainingParam import get_models_config 

    print("Modules 'traitementDonnées' et 'training' importés avec succès.")
    
except ImportError as e:
    print(f"ERREUR D'IMPORTATION : {e}")
    print("Veuillez vérifier les noms de vos fichiers dans le dossier 'maladiescardiaques/'.")


def cellule2(X_train, y_train): 
    # Cellule 3: Entraînement d'un seul modèle (DecisionTree)

    results = {}
    configs = get_models_config()
    preprocessor = create_preprocessing_pipeline()

    # 1. Sélectionner uniquement le TROISIÈME modèle (index 2)
    try:
        name, config = list(configs.items())[2] # Index 2 pour le 3ème modèle
    except IndexError:
        print("ERREUR : Impossible de trouver le troisième modèle dans les configs.")
        raise

    print(f"Début de l'entraînement d'un seul modèle : {name}...")
    print("-" * 60)

    start_global = time.time()
    start_model = time.time()
    print(f"🔄 Optimisation de : {name}...")

    full_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', config['model'])
    ])

    search = GridSearchCV(
        full_pipeline,
        config['params'],
        cv=5,
        scoring='f1',
        n_jobs=-1,
        verbose=0 # On met 0 pour ne pas polluer le notebook
    )

    search.fit(X_train, y_train)

    # Stocker le meilleur modèle et son score de CV
    results[name] = {
        'best_model': search.best_estimator_,
        'best_cv_score': search.best_score_ # Score F1 moyen en CV
    }

    end_model = time.time()
    print(f"✅ Terminé : {name} (Score CV F1: {search.best_score_:.4f}) [Durée: {end_model - start_model:.2f}s]")

    end_global = time.time()
    print("-" * 60)
    print(f"Entraînement complet terminé en {end_global - start_global:.2f} secondes.")