from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import os

def train_svm_gpu(modelname : str):
    """
    Trains an SVM classifier with RBF kernel using GPU to predict vowels/consonants
    
    Args:
        modelname: name of the model
    """

    try:
        from cuml.svm import SVC
    except ImportError:
        print("cuML library is not installed. Please install it to use GPU training.")
        return

    x_train, y_train = prepare_model_data("ciphers_train")
    print(f"Training samples: {len(x_train)}")

    svm_model = SVC(kernel='rbf', random_state=42, class_weight='balanced', C=100, gamma=0.001)
    svm_model.fit(x_train, y_train)
    print("Model trained")

    model_filename = f'{modelname}.joblib'
    try:
        joblib.dump(svm_model, model_filename)
        print(f"\nModel successfully saved to: {os.path.abspath(model_filename)}")
    except Exception as e:
        print(f"\nError saving model with joblib: {e}")
    
def load_svm_gpu(modelname : str):
    from cuml.svm import SVC

    model_filename = f'{modelname}.joblib'
    try:
        # Load the model object from the file
        loaded_model = joblib.load(model_filename)
        print(f"Model loaded successfully from {model_filename}")
        
        #Test model on ppmi validation data
        x_test, y_test = prepare_model_data("ciphers_test")

        y_pred = loaded_model.predict(x_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='binary')
        recall = recall_score(y_test, y_pred, average='binary')
        f1 = f1_score(y_test, y_pred, average='binary')

        print("\n--- Final Test Set Results ---")
        print(f"Accuracy: {accuracy:.2f}")
        print(f"Precision: {precision:.2f}")
        print(f"Recall: {recall:.2f}")
        print(f"F1-Score: {f1:.2f}")
            
    except FileNotFoundError:
        print(f"Error: Model file {model_filename} not found.")
    except Exception as e:
        print(f"Error loading model: {e}")

def prepare_model_data(train_dir: str, max_ciphers: int = None):
    """
    Prepare training data (features and binary labels) from a specified number 
    of cipher-specific files in the folder. All letters are classified as 
    Vowel (0) or Consonant (1).

    Args:
        train_dir: Path to the folder containing the separate embeddings and mappings CSV files.
        max_ciphers: The maximum number of ciphers (file pairs) to process. 
                     If None, all found ciphers are used.

    Returns:
        x_train: np.ndarray of shape (n_samples, n_features) (Embedding Vectors)
        y_train: np.ndarray of shape (n_samples,) (Binary Labels: 0=Vowel, 1=Consonant)
    """

    import os
    import glob
    import pandas as pd

    # Use glob to find all embedding files (e.g., cipher-1_embeddings.csv)
    emb_files = glob.glob(os.path.join(train_dir, '*_embeddings.csv'))

    if not emb_files:
        raise FileNotFoundError(f"No '*_embeddings.csv' files found in the directory: {train_dir}")
        
    if max_ciphers is not None and max_ciphers > 0:
        emb_files.sort()
        emb_files = emb_files[:max_ciphers]
        print(f"Limiting training to the first {len(emb_files)} ciphers.")

    list_x_train, list_y_train = [], []
    VOWELS = list('AEIOU')
    
    # Process all selected ciphers
    for emb_file in emb_files:
        # 1. Determine corresponding mapping file name
        base_name = os.path.basename(emb_file).replace('_embeddings.csv', '')
        map_file = os.path.join(train_dir, f"{base_name}_mappings.csv")

        if not os.path.exists(map_file):
            print(f"Warning: Skipping {emb_file}. Missing corresponding mapping file: {map_file}")
            continue

        # 2. Load Embeddings and Mappings
        emb_df = pd.read_csv(emb_file)
        map_df = pd.read_csv(map_file)
        
        # Clean column names
        emb_df.columns = emb_df.columns.str.strip()
        map_df.columns = map_df.columns.str.strip()
        
        CIPHER_COL = 'cipher_symbol'
        PLAINTEXT_COL = 'plaintext_symbol'

        # 3. Merge Embeddings with Plaintext Labels
        merged_df = pd.merge(emb_df, map_df, on=CIPHER_COL, how='inner')

        # 4. Extract Features (X)
        X = merged_df.drop(columns=[CIPHER_COL, PLAINTEXT_COL])
        
        # 5. Extract Labels (Y)
        Y_series = merged_df[PLAINTEXT_COL]
        
        list_x_train.append(X)
        list_y_train.append(Y_series)

    # Concatenate all data
    x_train_df = pd.concat(list_x_train, ignore_index=True)
    y_train_series = pd.concat(list_y_train, ignore_index=True)

    # Process labels: Binary classification (0=Vowel, 1=Consonant)
    y_train_binary = (~y_train_series.astype(str).str.upper().isin(VOWELS)).astype(int)

    # Convert to numpy arrays
    x_train = x_train_df.to_numpy()
    y_train = y_train_binary.to_numpy()
    
    print(f"Prepared {x_train.shape[0]} total training samples from {len(emb_files)} ciphers.")

    return x_train, y_train

def train_svm_cpu(vowels_consonants=True, hyperparam_tuning=False):
    """
    Trains an SVM classifier with RBF kernel optional hyperparameter tuning
    to predict plaintext letters (Vowels/Consonants).
    
    Args:
        vowels_consonants: if True, classify as vowel (0) or consonant (1); else classify all letters
        hyperparam_tuning: if True, perform hyperparameter tuning using GridSearchCV
    """

    x_train, y_train, x_test, y_test = prepare_data_csv("output", vowels_consonants=vowels_consonants)
    
    print(f"Training samples: {len(x_train)}, Test samples: {len(x_test)}")

    # Hyperparameters based on previous tuning experiments
    svm_base = SVC(kernel='rbf', random_state=42, class_weight='balanced', C=100, gamma=0.001)

    if hyperparam_tuning:
        # Perform Hyperparameter Tuning
        x_train_subset, _, y_train_subset, _ = train_test_split(
            x_train, y_train,
            test_size=0.8,
            random_state=42,
            stratify=y_train
        )
        print(f"Tuning subset size: {len(x_train_subset)}")
        
        param_grid = {
            'C': [0.1, 1, 10, 100], 
            'gamma': [0.001, 0.01, 0.1, 'scale'], 
            'class_weight': ['balanced']
        }
        
        grid_search = GridSearchCV(
            estimator=svm_base, 
            param_grid=param_grid, 
            scoring='f1',
            cv=3,
            verbose=2,
            n_jobs=-1
        )

        print("\nStarting Hyperparameter Grid Search (Training multiple models)...")
        grid_search.fit(x_train_subset, y_train_subset)
        print("Grid Search complete.")

        best_model = grid_search.best_estimator_

        print("\n--- Hyperparameter Tuning Results ---")
        print(f"Best Parameters Found: {grid_search.best_params_}")
        print(f"Best Cross-Validation F1-Score: {grid_search.best_score_:.2f}")

    else:
        # Skip Tuning and use the base model
        print("\nSkipping Hyperparameter Tuning. Training base RBF SVM.")
        best_model = svm_base

    # Train the final model
    best_model.fit(x_train, y_train)
    print("Model trained")

    y_pred = best_model.predict(x_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='binary')
    recall = recall_score(y_test, y_pred, average='binary')
    f1 = f1_score(y_test, y_pred, average='binary')
    
    print("\n--- Final Test Set Results ---")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1-Score: {f1:.2f}")

def prepare_data_csv(dir, vowels_consonants=True):
    """
    Prepare training data CSV from all cipher JSON files in the specified folder
    Args:
        dir: path to the folder containing embedding CSV files
        vowels_consonants: if True, classify as vowel (0) or consonant (1); else classify all letters
    Returns:
        x_train: np.ndarray of shape (n_samples, n_features)
        y_train: np.ndarray of shape (n_samples,)
        x_test: np.ndarray of shape (n_samples, n_features)
        y_test: np.ndarray of shape (n_samples,)
    """

    import os
    import glob
    import pandas as pd

    # Load embeddings
    train_files = glob.glob(os.path.join(dir, '*_train.csv'))

    if not train_files:
        raise FileNotFoundError(f"No '*_train.csv' files found in the directory: {dir}")

    # Create lists to hold the data from each file
    list_x_train, list_y_train = [], []
    list_x_test, list_y_test = [], []
    
    # Process ciphers
    for train_file in train_files:
        # Construct the paths for corresponding test files
        base_name = os.path.basename(train_file).replace('_train.csv', '')
        test_file = os.path.join(dir, f"{base_name}_test.csv")
        test_labels_file = os.path.join(dir, f"{base_name}_test_labels.csv")

        # Load training data
        train_df = pd.read_csv(train_file)
        list_x_train.append(train_df.drop(columns=['cipher_symbol', 'plaintext_symbol']))
        list_y_train.append(train_df['plaintext_symbol'])

        # Load test data and merge with labels
        if os.path.exists(test_file) and os.path.exists(test_labels_file):
            test_df = pd.read_csv(test_file)
            test_labels_df = pd.read_csv(test_labels_file)
            
            # Merge features and labels on 'cipher_symbol' to ensure correct alignment
            merged_test_df = pd.merge(test_df, test_labels_df, on='cipher_symbol')
            
            list_x_test.append(merged_test_df.drop(columns=['cipher_symbol', 'plaintext_symbol']))
            list_y_test.append(merged_test_df['plaintext_symbol'])

    # Concatenate all data into single dataframes
    x_train_df = pd.concat(list_x_train, ignore_index=True)
    y_train_series = pd.concat(list_y_train, ignore_index=True)
    x_test_df = pd.concat(list_x_test, ignore_index=True)
    y_test_series = pd.concat(list_y_test, ignore_index=True)

    # Process labels
    if vowels_consonants:
        # Binary classification: 0 for vowel, 1 for consonant
        vowels = list('AEIOU')
        y_train = (~y_train_series.str.upper().isin(vowels)).astype(int)
        y_test = (~y_test_series.str.upper().isin(vowels)).astype(int)
    else:
        # Multi-class classification
        y_train, _ = pd.factorize(y_train_series)
        y_test, _ = pd.factorize(y_test_series)

    # Convert to numpy arrays
    x_train = x_train_df.to_numpy()
    y_train = y_train.to_numpy()
    x_test = x_test_df.to_numpy()
    y_test = y_test.to_numpy()

    return x_train, y_train, x_test, y_test