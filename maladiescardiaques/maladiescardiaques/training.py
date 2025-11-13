# maladiescardiaques/model_trainer.py

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Importation des modèles de scikit-learn
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

# Importation pour la validation croisée et l'optimisation
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, f1_score, accuracy_score

# Importation des fonctions de prétraitement des données
from traitementDonnées import load_and_split_data, create_preprocessing_pipeline


# Force la sortie standard en UTF-8, même sous Windows
#sys.stdout.reconfigure(encoding='utf-8')

# =============================================================================
# 1. CONFIGURATION DES MODÈLES ET HYPER-PARAMÈTRES
# =============================================================================
def get_models_config():
    """
    Retourne un dictionnaire contenant les modèles à tester et leurs
    grilles d'hyper-paramètres respectives pour GridSearchCV.
    """
    models_config = {
        # Modèle 1 : Régression Logistique
        'LogisticRegression': {
            'model': LogisticRegression(max_iter=1000, random_state=42),
            'params': {
                'classifier__C': [0.1, 1.0, 10.0],
                'classifier__solver': ['liblinear', 'lbfgs']
            }
        },
        
        # Modèle 2 : K-Plus Proches Voisins (KNN)
        'KNeighbors': {
            'model': KNeighborsClassifier(),
            'params': {
                'classifier__n_neighbors': [3, 5, 7, 9],
                'classifier__weights': ['uniform', 'distance'],
                'classifier__metric': ['euclidean', 'manhattan']
            }
        },
        
        # Modèle 3 : Arbre de Décision
        'DecisionTree': {
            'model': DecisionTreeClassifier(random_state=42),
            'params': {
                'classifier__max_depth': [None, 10, 20, 30],
                'classifier__min_samples_split': [2, 5, 10],
                'classifier__criterion': ['gini', 'entropy']
            }
        },
        
        # Modèle 4 : Forêt Aléatoire (Random Forest)
        'RandomForest': {
            'model': RandomForestClassifier(random_state=42),
            'params': {
                'classifier__n_estimators': [50, 100, 200],
                'classifier__max_depth': [None, 10, 20],
                'classifier__min_samples_split': [2, 5]
            }
        },
        
        # Modèle 5 : Support Vector Machine (SVM)
        'SVM': {
            'model': SVC(probability=True, random_state=42),
            'params': {
                'classifier__C': [0.1, 1, 10],
                'classifier__kernel': ['linear', 'rbf'],
                'classifier__gamma': ['scale', 'auto']
            }
        },
        
        # Modèle 6 : Gradient Boosting
        'GradientBoosting': {
            'model': GradientBoostingClassifier(random_state=42),
            'params': {
                'classifier__n_estimators': [50, 100],
                'classifier__learning_rate': [0.01, 0.1, 0.2],
                'classifier__max_depth': [3, 5]
            }
        }
    }
    return models_config


# =============================================================================
# 2. FONCTION D'ENTRAÎNEMENT ET DE RECHERCHE (GridSearch)
# =============================================================================
def train_and_evaluate_models(X_train, y_train):
    """
    Parcourt chaque modèle, crée un pipeline, et exécute un GridSearchCV.
    Retourne un dictionnaire avec les meilleurs modèles entraînés.
    """
    results = {}
    configs = get_models_config()
    
    print(f"Début de l'entraînement de {len(configs)} modèles avec validation croisée...")
    print("-" * 60)

    # Récupérer le préprocesseur depuis data_processor.py
    preprocessor = create_preprocessing_pipeline()

    for name, config in configs.items():
        print(f"🔄 Optimisation de : {name}...")
        
        # Création du Pipeline Complet : Prétraitement + Classifieur
        # 'preprocessor' vient de data_processor.py
        # 'classifier' est le modèle actuel de la boucle
        full_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', config['model'])
        ])
        
        # Configuration de la recherche sur grille (Cross-Validation intégrée)
        # cv=5 signifie une validation croisée à 5 plis (K-Fold)
        # scoring='f1' est souvent mieux que 'accuracy' pour les maladies (déséquilibre)
        search = GridSearchCV(
            full_pipeline,
            config['params'],
            cv=5,
            scoring='f1',  # On optimise le F1-score (équilibre précision/rappel)
            n_jobs=-1,     # Utilise tous les processeurs disponibles
            verbose=1
        )
        
        # Lancement de la recherche (Fit)
        search.fit(X_train, y_train)
        
        # Stockage des résultats
        results[name] = {
            'best_model': search.best_estimator_,
            'best_params': search.best_params_,
            'best_score': search.best_score_ # Score moyen de validation croisée
        }
        
        print(f"✅ Meilleur score CV (F1): {search.best_score_:.4f}")
        print("-" * 60)
        
    return results


# =============================================================================
# 3. MAIN : Exécution
# =============================================================================
if __name__ == '__main__':

    # On remonte de 1 niveau car on est dans maladiescardiaques/
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    DATA_PATH = PROJECT_ROOT / "data" / "raw" / "cardiaque.csv" 
    
    print("--- CHARGEMENT DES DONNÉES ---")
    data = load_and_split_data(DATA_PATH)
    
    if data:
        X_train, X_test, y_train, y_test = data
        
        # 1. Entraînement et sélection (Cross-Validation)
        model_results = train_and_evaluate_models(X_train, y_train)
        
        # 2. Sélection du "Grand Gagnant" basé sur le score de validation
        best_model_name = max(model_results, key=lambda k: model_results[k]['best_score'])
        best_pipeline = model_results[best_model_name]['best_model']
        
        print("\n" + "="*60)
        print(f"🏆 MEILLEUR MODÈLE TROUVÉ : {best_model_name}")
        print(f"   Score Validation Croisée (F1) : {model_results[best_model_name]['best_score']:.4f}")
        print(f"   Meilleurs Paramètres : {model_results[best_model_name]['best_params']}")
        print("="*60)
        
        # 3. ÉVALUATION FINALE SUR LE TEST SET (JAMAIS VU)
        # C'est ici qu'on obtient la "vraie" performance pour le rapport
        print(f"\n🧪 Évaluation finale de {best_model_name} sur le jeu de TEST...")
        y_pred = best_pipeline.predict(X_test)
        
        print("\nRapport de classification (Test Set) :")
        print(classification_report(y_test, y_pred))
        
        print(f"Accuracy finale : {accuracy_score(y_test, y_pred):.4f}")
        print(f"F1 Score final  : {f1_score(y_test, y_pred):.4f}")