import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# =============================================================================
# 1. Définition des Caractéristiques (basé sur l'EDA)
# =============================================================================

# Variable cible
TARGET_COL = 'num'

# Colonne non pertinente pour la modélisation
ID_COL = 'id'

# Features numériques (celles qui doivent être mises à l'échelle)
NUMERICAL_FEATURES = [
    'age', 
    'trestbps', 
    'chol', 
    'thalach', 
    'oldpeak'
]

# Features catégorielles (celles qui doivent être encodées)
CATEGORICAL_FEATURES = [
    'sex', 
    'cp', 
    'fbs', 
    'restecg', 
    'exang', 
    'slope', 
    'ca', 
    'thal'
]


# =============================================================================
# 2. Fonction de Chargement et Division des Données
# =============================================================================

def load_and_split_data(data_path: Path, test_size=0.2, random_state=42):
    """
    Charge les données brutes et les divise en ensembles d'entraînement et de test
    pour éviter toute fuite de données.
    """
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"Erreur : Fichier non trouvé à {data_path}")
        return None
    
    # Séparer les caractéristiques (X) et la cible (y)
    # Nous retirons la cible ET l'ID des caractéristiques
    X = df.drop(columns=[TARGET_COL, ID_COL], errors='ignore')
    y = df[TARGET_COL]
    
    # Division stratifiée pour conserver la proportion des classes dans y
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state, 
        stratify=y  # Important pour la classification
    )
    
    print(f"Données chargées. Entraînement : {X_train.shape}, Test : {X_test.shape}")
    return X_train, X_test, y_train, y_test


# =============================================================================
# 3. Fonction de Création du Pipeline de Prétraitement
# =============================================================================

def create_preprocessing_pipeline():
    """
    Crée le pipeline de prétraitement principal en utilisant ColumnTransformer.
    """
    
    # ----- Étape A : Pipeline pour les features NUMÉRIQUES -----
    # 1. StandardScaler : Met les données à l'échelle (moyenne 0, écart-type 1).
    #    C'est crucial pour les modèles comme la Régression Logistique ou les SVM.
    numerical_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    # ----- Étape B : Pipeline pour les features CATÉGORIELLES -----
    # 1. OneHotEncoder : Convertit les catégories (ex: 'sexe' 0/1) en colonnes binaires.
    #    'handle_unknown='ignore'' : Si une catégorie inconnue apparaît dans
    #    les données de test, elle sera ignorée (toutes les colonnes à 0).
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # ----- Étape C : Combinaison avec ColumnTransformer -----
    # Le ColumnTransformer est le "chef d'orchestre". Il applique :
    # - 'numerical_transformer' aux colonnes de NUMERICAL_FEATURES
    # - 'categorical_transformer' aux colonnes de CATEGORICAL_FEATURES
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, NUMERICAL_FEATURES),
            ('cat', categorical_transformer, CATEGORICAL_FEATURES)
        ],
        remainder='passthrough' # Garde les colonnes non spécifiées (s'il y en a)
    )
    
    return preprocessor


# =============================================================================
# 4. Bloc de Test (pour exécuter ce fichier directement)
# =============================================================================

if __name__ == '__main__':
    # Ce code s'exécute SEULEMENT si vous faites 'python src/data_processor.py'

    DATA_FILE_PATH = Path("data") / "raw" / "cardiaque.csv"

    print("--- Test du module data_processor.py ---")
    
    # Étape 1: Charger et diviser
    data = load_and_split_data(DATA_FILE_PATH.resolve())
    
    if data:
        X_train, X_test, y_train, y_test = data
        
        # Étape 2: Créer le pipeline
        preprocessor = create_preprocessing_pipeline()
        print("\nPipeline de prétraitement créé avec succès :")
        print(preprocessor)
        
        # Étape 3: Appliquer le pipeline (Fit sur train, Transform sur train/test)
        
        # fit_transform : APPREND les moyennes/écarts-types et les encodeurs
        # sur X_train, PUIS transforme X_train.
        print("\nApplication de fit_transform sur X_train...")
        X_train_processed = preprocessor.fit_transform(X_train)
        
        # transform : APPLIQUE les transformations apprises (de X_train)
        # à X_test. Il n'apprend RIEN de X_test.
        print("Application de transform sur X_test...")
        X_test_processed = preprocessor.transform(X_test)
        
        print(f"\nForme de X_train après transformation : {X_train_processed.shape}")
        print(f"Forme de X_test après transformation : {X_test_processed.shape}")
        
        print("\nExemple des 5 premières lignes de X_train transformé :")
        print(pd.DataFrame(X_train_processed, columns=preprocessor.get_feature_names_out()).head())