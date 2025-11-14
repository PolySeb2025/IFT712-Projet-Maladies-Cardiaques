import seaborn as sns
from pathlib import Path
import time
import sys
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Métriques Sklearn
from sklearn.metrics import f1_score, accuracy_score, classification_report
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA

# --- Importation de nos modules ---
PROJECT_FCT = Path("../maladiescardiaques").resolve()
PROJECT_ROOT = Path("..").resolve()
sys.path.append(str(PROJECT_FCT))

print(f"Racine du projet ajoutée au PATH: {PROJECT_FCT}")

try:
    from traitementDonnées import load_and_split_data, create_preprocessing_pipeline
    from trainingParam import get_models_config 
    print("Modules 'traitementDonnées' et 'training' importés avec succès.")
except ImportError as e:
    print(f"ERREUR D'IMPORTATION : {e}")
    print("Veuillez vérifier les noms de vos fichiers dans le dossier 'maladiescardiaques/'.")

########################################################################################################################
# FONCTIONS D'ENTRAINEMENT ET D'ÉVALUATION
########################################################################################################################

def EntrainementModele(X_train, y_train, test_index): 
    """
    Entraîne un modèle spécifié par son index.
    RETOURNE le dictionnaire 'results'.
    """
    # results est local à cette fonction
    results = {} 
    configs = get_models_config()
    preprocessor = create_preprocessing_pipeline()

    try:
        name, config = list(configs.items())[test_index]
    except IndexError:
        print(f"ERREUR : Impossible de trouver le modèle à l'index {test_index}.")
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
        verbose=0
    )

    search.fit(X_train, y_train)

    results[name] = {
        'best_model': search.best_estimator_,
        'best_cv_score': search.best_score_
    }

    end_model = time.time()
    print(f"✅ Terminé : {name} (Score CV F1: {search.best_score_:.4f}) [Durée: {end_model - start_model:.2f}s]")

    end_global = time.time()
    print("-" * 60)
    print(f"Entraînement complet terminé en {end_global - start_model:.2f} secondes.")
    
    # Retourner le dictionnaire de résultats
    return results


def CalculPerf(results, X_test, y_test):
    """
    Calcule les performances sur le jeu de test.
    ACCEPTE 'results' et le MODIFIE en place.
    RETOURNE 'results' modifié.
    """
    print("Calcul des performances finales sur l'ensemble de Test...")

    for name, res in results.items():
        model = res['best_model']
        y_pred = model.predict(X_test)
        
        res['final_f1_score'] = f1_score(y_test, y_pred)
        res['final_accuracy'] = accuracy_score(y_test, y_pred)
        res['predictions'] = y_pred
        
        try:
            res['proba'] = model.predict_proba(X_test)[:, 1]
        except AttributeError:
            res['proba'] = None 

    print("Scores finaux calculés.")
    # Retourner le dictionnaire modifié
    return results


########################################################################################################################
# FONCTIONS DE VISUALISATION
########################################################################################################################

def _get_best_model_name_from_results(results):
    """Helper pour trouver le nom du meilleur modèle dans le dict results."""
    final_scores = {name: res['final_f1_score'] for name, res in results.items()}
    df_final_scores = pd.DataFrame.from_dict(
        final_scores, 
        orient='index', 
        columns=['F1_Score_Final_Test']
    )
    if df_final_scores.empty:
        raise ValueError("Le dictionnaire 'results' est vide ou 'CalculPerf' n'a pas été exécuté.")
        
    best_model_name = df_final_scores.idxmax().values[0]
    return best_model_name


# --- GRAPHIQUE 1 : Matrice de Confusion ---
def MatriceConfusion(results, y_test):
    """Affiche le rapport de classification et la matrice de confusion."""
    
    print(f"\n--- Visualisation 1 : Matrice de Confusion ---")
    
    try:
        best_model_name = _get_best_model_name_from_results(results)
    except ValueError as e:
        print(e)
        return

    y_pred_best = results[best_model_name]['predictions']

    print(f"\n--- Analyse détaillée du meilleur modèle : {best_model_name} ---")
    print("\nRapport de Classification (Test Set):")
    print(classification_report(y_test, y_pred_best, target_names=['No Disease (0)', 'Disease (1)']))

    cm = confusion_matrix(y_test, y_pred_best)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues', 
        xticklabels=['Prédit No Disease', 'Prédit Disease'],
        yticklabels=['Actuel No Disease', 'Actuel Disease']
    )
    plt.title(f"Matrice de Confusion pour {best_model_name}", fontsize=16)
    plt.ylabel("Valeur Actuelle")
    plt.xlabel("Valeur Prédite")
    plt.show()


# --- GRAPHIQUE 2 : Probabilités Prédites ---
def PlotProbabilites(results, y_test):
    """Affiche le scatter plot des probabilités prédites vs observations."""
    
    print(f"\n--- Visualisation 2 : Probabilités (Test Set) ---")
    
    try:
        best_model_name = _get_best_model_name_from_results(results)
    except ValueError as e:
        print(e)
        return

    y_proba_best = results[best_model_name]['proba']
    
    if y_proba_best is None:
        print(f"Impossible de tracer les probabilités pour {best_model_name} (le modèle ne supporte pas 'predict_proba').")
        return

    plt.figure(figsize=(8, 5))
    plt.scatter(y_proba_best, y_test, alpha=0.6, color='royalblue', edgecolors='k')
    plt.axvline(x=0.5, color='red', linestyle='--', linewidth=1.5, label='Seuil de décision (0.5)')
    plt.xlabel("Probabilité prédite (Classe 1 - Disease)")
    plt.ylabel("Classe observée (0=No Disease, 1=Disease)")
    plt.title(f"Prédictions vs Observations pour {best_model_name}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# --- GRAPHIQUE 3 : Courbe ROC ---
def CourbeROC(results, y_test):
    """Affiche la courbe ROC et l'AUC."""
    
    print(f"\n--- Visualisation 3 : Courbe ROC (Test Set) ---")
    
    try:
        best_model_name = _get_best_model_name_from_results(results)
    except ValueError as e:
        print(e)
        return

    y_proba_best = results[best_model_name]['proba']
    
    if y_proba_best is None:
        print(f"Impossible de tracer la courbe ROC pour {best_model_name} (le modèle ne supporte pas 'predict_proba').")
        return

    fpr, tpr, thresholds = roc_curve(y_test, y_proba_best)
    auc = roc_auc_score(y_test, y_proba_best)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"{best_model_name} (AUC = {auc:.4f})", color="darkorange", linewidth=2)
    plt.plot([0, 1], [0, 1], color="navy", linestyle="--", label="Aléatoire (AUC = 0.5)")
    plt.xlabel("Taux de faux positifs (FPR)")
    plt.ylabel("Taux de vrais positifs (TPR / Rappel)")
    plt.title(f"Courbe ROC pour {best_model_name}")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# --- GRAPHIQUE 4 : Frontière de Décision (PCA) ---

def _plot_decision_boundary_pca_sklearn(X_raw, y, pipeline):
    """Fonction helper (interne) pour tracer la frontière via PCA."""
    
    preprocessor = pipeline.named_steps['preprocessor']
    classifier = pipeline.named_steps['classifier']
    
    try:
        X_processed = preprocessor.fit_transform(X_raw)
    except Exception as e:
        print(f"Erreur lors du fit_transform du preprocessor : {e}")
        return

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_processed)
    
    x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
    y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
    
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                         np.linspace(y_min, y_max, 300))
    
    grid_points_pca = np.c_[xx.ravel(), yy.ravel()]
    grid_points_processed = pca.inverse_transform(grid_points_pca)
    
    Z = classifier.predict(grid_points_processed)
    Z = Z.reshape(xx.shape)
    
    plt.figure(figsize=(10, 7))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdYlBu)
    
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap=plt.cm.RdYlBu, 
                          edgecolor='k', alpha=0.7)
    plt.xlabel("Composante principale 1")
    plt.ylabel("Composante principale 2")
    plt.title(f"Frontière de décision (sur {classifier.__class__.__name__}) via PCA")
    plt.legend(handles=scatter.legend_elements()[0], labels=['No Disease (0)', 'Disease (1)'])
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def PlotFrontiereDecision(results, X_train, y_train):
    """Affiche la frontière de décision du meilleur modèle sur les données d'entraînement."""
    
    print(f"\n--- Visualisation 4 : Frontière de Décision (sur données d'entraînement) ---")
    
    try:
        # Note : On utilise le score CV (sur X_train) pour trouver le meilleur modèle
        # car ce graphique utilise X_train.
        cv_scores = {name: res['best_cv_score'] for name, res in results.items()}
        df_cv_scores = pd.DataFrame.from_dict(
            cv_scores, orient='index', columns=['CV_Score']
        )
        if df_cv_scores.empty:
             raise ValueError("Le dictionnaire 'results' est vide.")
        
        best_model_name = df_cv_scores.idxmax().values[0]
        
    except ValueError as e:
        print(e)
        return

    best_model_pipeline = results[best_model_name]['best_model']
    
    # Utiliser la fonction helper
    _plot_decision_boundary_pca_sklearn(X_train, y_train, best_model_pipeline)