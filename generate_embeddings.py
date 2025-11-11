def generate_csvs(input_dir : str = "ciphers"):
    """
    Generates separate CSV files for embeddings and mappings for training and 
    testing ciphers from the 'ciphers' directory, saving them to 'ciphers_train' 
    and 'ciphers_test', respectively.
    """
    import os
    train_dir = "ciphers_train"
    test_dir = "ciphers_test"

    # Create output directories if they don't exist
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Get all JSON files in the input directory
    json_files = [f for f in os.listdir(input_dir) if f.endswith('.json')]

    if not json_files:
        print(f"No JSON files found in the input directory '{input_dir}'")
        return

    # Separate files into train and test lists
    train_json_files = [f for f in json_files if f.startswith('cipher-')]
    test_json_files = [f for f in json_files if f.startswith('test-cipher-')]
    
    # Process both sets of files using a helper function
    _process_json_files(input_dir, train_json_files, train_dir, 'Training')
    _process_json_files(input_dir, test_json_files, test_dir, 'Testing')


def _process_json_files(input_dir, json_list, output_dir, file_type_label):
    """Helper function to load JSONs, generate data, and save two CSVs."""
    
    import json
    import os
    import pandas as pd
    
    if not json_list:
        print(f"\nNo {file_type_label} JSON files to process.")
        return
        
    print(f"\n--- Processing {file_type_label} Ciphers ({len(json_list)} files) ---")

    for json_file in json_list:
        print(f"  Processing {json_file}...")
        
        # Load the cipher JSON
        json_path = os.path.join(input_dir, json_file)
        with open(json_path, "r", encoding="utf-8") as f:
            cipher_json = json.load(f)
        
        # Build cipher to letter mapping
        cipher_to_letter = {}
        # List of mappings for the mapping CSV
        mappings_data = [] 
        for letter, symbols in cipher_json["key"].items():
            for symbol in symbols:
                cipher_symbol = str(symbol)
                plaintext_letter = letter.upper()
                cipher_to_letter[cipher_symbol] = plaintext_letter
                
                mappings_data.append({
                    "plaintext_symbol": plaintext_letter,
                    "cipher_symbol": cipher_symbol
                })
        
        # Generate embeddings (Assumes 'generate_embeddings' is available)
        # embeddings is a NumPy array, vocab is a list of symbols
        embeddings, vocab, _ = generate_embeddings(cipher_json) 
        
        # --- 1. Prepare and Save EMBEDDINGS CSV ---
        embeddings_data = []
        for i, symbol in enumerate(vocab):
            row = {"cipher_symbol": str(symbol)}
            
            # Add embedding dimensions
            for j in range(embeddings.shape[1]):
                row[f"dim_{j}"] = embeddings[i, j]
                
            embeddings_data.append(row)
        
        df_emb = pd.DataFrame(embeddings_data)
        base_name = os.path.splitext(json_file)[0]
        
        emb_csv_path = os.path.join(output_dir, f"{base_name}_embeddings.csv")
        df_emb.to_csv(emb_csv_path, index=False)

        # --- 2. Prepare and Save MAPPINGS CSV ---
        # The mapping CSV is simply built from the list collected earlier
        df_map = pd.DataFrame(mappings_data)
        map_csv_path = os.path.join(output_dir, f"{base_name}_mappings.csv")
        df_map.to_csv(map_csv_path, index=False)

        print(f"Saved {len(df_emb)} embeddings to {os.path.basename(emb_csv_path)}")
        print(f"Saved {len(df_map)} mappings to {os.path.basename(map_csv_path)}")
        print(f"Finished {json_file}")

def generate_embeddings(cipher_json, window=1, embedding_size=20, plaintext=False):
    import numpy as np
    from sklearn.decomposition import TruncatedSVD

    """
    Generate SVD-based embeddings from a ciphertext string in a cipher_json object.
    Args:
        cipher_json: dict with 'ciphertext' (space-separated string) and 'key' (letter->symbols mapping)
        window: context window size for co-occurrence
        embedding_size: number of SVD components
    Returns: embeddings, vocab, vocab_index
    """
    # 1. Load ciphertext from cipher_json
    if plaintext:
        tokens = list(cipher_json["plaintext"])
    else:
        tokens = cipher_json["ciphertext"].split()

    # 2. Build co-occurrence matrix
    vocab = sorted(set(tokens))
    vocab_index = {sym: i for i, sym in enumerate(vocab)}
    vocab_size = len(vocab)
    cooc = np.zeros((vocab_size, vocab_size), dtype=np.float64)
    for i, sym in enumerate(tokens):
        center = vocab_index[sym]
        for j in range(max(0, i - window), min(len(tokens), i + window + 1)):
            if i != j:
                context = vocab_index[tokens[j]]
                cooc[center, context] += 1

    # 3. Compute PPMI matrix
    total_cooc = np.sum(cooc)
    row_sums = np.sum(cooc, axis=1)
    col_sums = np.sum(cooc, axis=0)
    ppmi = np.zeros_like(cooc)
    for i in range(vocab_size):
        for j in range(vocab_size):
            if cooc[i, j] > 0:
                p_ij = cooc[i, j] / total_cooc
                p_i = row_sums[i] / total_cooc
                p_j = col_sums[j] / total_cooc
                pmi = np.log2(p_ij / (p_i * p_j))
                ppmi[i, j] = max(pmi, 0)

    # 4. Apply SVD to reduce dimensions
    emb_size = min(embedding_size, vocab_size)
    print(emb_size)
    svd = TruncatedSVD(n_components=emb_size, random_state=17)
    embeddings = svd.fit_transform(ppmi)

    return embeddings, vocab, vocab_index

def generate_csvs_plain_cipher(input_dir : str):
    """
    Generates embeddings for ciphertext and plaintext
    """
    import json
    import os
    import pandas as pd

    input_dir = input_dir
    output_dir = "embeddings_plain_cipher"

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Get all JSON files in the input directory
    json_files = [f for f in os.listdir(input_dir) if f.endswith('.json')]

    if not json_files:
        print("No JSON files found in the input directory")
    else:
        for json_file in json_files:
            print(f"Processing {json_file}...")
            
            # Load the cipher JSON
            with open(os.path.join(input_dir, json_file), "r", encoding="utf-8") as f:
                cipher_json = json.load(f)
            
            # Build cipher to letter mapping
            cipher_to_letter = {}
            for letter, symbols in cipher_json["key"].items():
                for symbol in symbols:
                    cipher_to_letter[str(symbol)] = letter.upper()  # ensure uppercase
            
            # Generate embeddings
            embeddings_cipher, vocab_cipher, _ = generate_embeddings(cipher_json)

            # Generate embeddings_plaintext
            embeddings_plaintext, vocab_plaintext, _ = generate_embeddings(cipher_json, plaintext=True)
            
            # Create output filename (without .json extension)
            base_name = os.path.splitext(json_file)[0]
            
            mapping_data = []
            for symbol in vocab_cipher:
                mapping_data.append({
                    "cipher_symbol": symbol,
                    "plaintext_symbol": cipher_to_letter.get(str(symbol), "Unknown")
                })

            df_mappings = pd.DataFrame(mapping_data)
            mappings_csv_path = os.path.join(output_dir, f"{base_name}_mappings.csv")
            df_mappings.to_csv(mappings_csv_path, index=False)
            print(f"Saved mappings ({len(df_mappings)} samples) to {mappings_csv_path}")

            cipher_emb_data = []
            for i, symbol in enumerate(vocab_cipher):
                row = {"cipher_symbol": symbol}
                
                # Add embedding dimensions
                for j in range(embeddings_cipher.shape[1]):
                    row[f"dim_{j}"] = embeddings_cipher[i, j]
                
                cipher_emb_data.append(row)

            df_cipher_emb = pd.DataFrame(cipher_emb_data)
            cipher_emb_csv_path = os.path.join(output_dir, f"{base_name}_cipher_embeddings.csv")
            df_cipher_emb.to_csv(cipher_emb_csv_path, index=False)
            print(f"Saved cipher embeddings ({len(df_cipher_emb)} samples) to {cipher_emb_csv_path}")

            
            plaintext_emb_data = []
            for i, letter in enumerate(vocab_plaintext):
                row = {"plaintext_letter": letter}
                
                # Add embedding dimensions
                for j in range(embeddings_plaintext.shape[1]):
                    row[f"dim_{j}"] = embeddings_plaintext[i, j]
                
                plaintext_emb_data.append(row)
            
            df_plaintext_emb = pd.DataFrame(plaintext_emb_data)
            plaintext_emb_csv_path = os.path.join(output_dir, f"{base_name}_plaintext_embeddings.csv")
            df_plaintext_emb.to_csv(plaintext_emb_csv_path, index=False)
            print(f"Saved plaintext embeddings ({len(df_plaintext_emb)} samples) to {plaintext_emb_csv_path}")

            print(f"Finished processing {json_file}\n")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Generate CSVs for cipher embeddings and mappings."
    )
    
    parser.add_argument(
        '--input_dir', 
        type=str, 
        default='ciphers', 
        help="The directory containing the cipher JSON files (default: ciphers)."
    )

    args = parser.parse_args()
    generate_csvs(input_dir=args.input_dir)