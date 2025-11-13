import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple

def find_k_similar_vecs(target_vec: np.ndarray, all_vecs: Dict[Any, np.ndarray], mappings, k: int = 5) -> Dict[Any, float]:
	"""
	Finds the k most similar vectors to the target vector using cosine similarity.
	Args:
		target_vec (np.ndarray): The target vector to compare against.
		all_vecs (Dict[Any, np.ndarray]): A dictionary mapping keys to vectors.
		k (int): The number of top similar vectors to return.
	Returns:
		Dict[Any, float]: A dictionary of the k most similar keys and their cosine similarity scores.
	"""
	from sklearn.metrics.pairwise import cosine_similarity

	# Reshape target_vec for compatibility
	target_vec = target_vec.reshape(1, -1)
	
	similarities = {}
	for key, vec in all_vecs.items():
		vec = vec.reshape(1, -1)
		sim = cosine_similarity(target_vec, vec)[0][0]
		similarities[key] = sim
	
	# Sort by similarity and get top k
	sorted_similarities = dict(sorted(similarities.items(), key=lambda item: item[1], reverse=True)[:k])

	# --- MODIFIED PRINTING LOGIC ---
	
	for cipher_symbol_key, sim in sorted_similarities.items():
		display_key = cipher_symbol_key # Default to cipher symbol key
		
		# Check if a mapping dictionary was provided
		if mappings is not None:
			# Look up the plaintext letter using the cipher symbol key
			plaintext_letter = mappings.get(cipher_symbol_key, f"Symbol {cipher_symbol_key}")
			# Format the display key as "Plaintext (Cipher Symbol)"
			display_key = f"{plaintext_letter} (Symbol: {cipher_symbol_key})"

		print(f"Match: {display_key}, Similarity: {sim:.4f}")
	
	return sorted_similarities
  
def read_embeddings(cipher_csv: str, plaintext_csv: str) -> Tuple[Dict[Any, np.ndarray], Dict[Any, np.ndarray]]:
	"""
	Reads a CSV file containing cipher symbol embeddings and returns a dictionary
	mapping each cipher symbol to its corresponding embedding vector.
	Args:
		csv_path (str): Path to the CSV file containing the embeddings.
	Returns:
		Tuple[Dict[Any, np.ndarray], Dict[Any, np.ndarray]]: A tuple containing two dictionaries:
			- A dictionary mapping cipher symbols to their embedding vectors.
			- A dictionary mapping plaintext letters to their embedding vectors.
	"""
	
	try:
		# 1. Load the CSV file into a Pandas DataFrame
		c_df = pd.read_csv(cipher_csv)
		p_df = pd.read_csv(plaintext_csv)
	except FileNotFoundError:
		print(f"Error: File not found")
		return {}
	except Exception as e:
		print(f"An error occurred during file loading: {e}")
		return {}

	# 2. Identify the embedding columns (assuming they are everything but the first column)
	# This is safer than relying on 'dim_X' names.
	symbol_col = c_df.columns[0]
	c_embedding_cols = c_df.columns[1:]
	letter_col = p_df.columns[0]
	p_embedding_cols = p_df.columns[1:]
	
	# 3. Create the dictionary by iterating through DataFrame rows
	# Convert the embedding columns to a NumPy array for each row.
	symbol_to_vector = {}
	for _, row in c_df.iterrows():
		symbol = int(row[symbol_col])
		# Extract the numerical values for the embedding vector
		vector = row[c_embedding_cols].values.astype(float)
		
		symbol_to_vector[symbol] = vector
	
	letter_to_vector = {}
	for _, row in p_df.iterrows():
		letter = row[letter_col]
		vector = row[p_embedding_cols].values.astype(float)
		
		letter_to_vector[letter] = vector

	letter_to_vector = sorted(letter_to_vector.items(), key=lambda item: item[0])
	letter_to_vector = dict(letter_to_vector)

	return symbol_to_vector, letter_to_vector

def read_mappings(mappings_csv_path: str) -> Dict[Any, str]:
	"""
	Reads the mappings CSV and creates a dictionary mapping each
	cipher symbol (as an integer) to its single corresponding plaintext letter.
	
	Args:
		mappings_csv_path (str): Path to the mappings CSV.
		
	Returns:
		Dict[int, str]: Dictionary mapping cipher symbol (int) -> plaintext letter (str).
	"""
	df = pd.read_csv(mappings_csv_path)
	
	plaintext_col = 'plaintext_symbol'
	cipher_col = 'cipher_symbol'
	
	cipher_to_plaintext_map = {}
	
	for _, row in df.iterrows():
		try:
			plaintext_letter = str(row[plaintext_col])
			cipher_symbol = int(row[cipher_col]) 
			
			cipher_to_plaintext_map[cipher_symbol] = plaintext_letter.upper()
		except ValueError:
			continue
			
	print(f"Loaded {len(cipher_to_plaintext_map)} cipher symbol mappings.")
	return cipher_to_plaintext_map

if __name__ == "__main__":
	import argparse
	from generate_embeddings import generate_csvs_plain_cipher
	
	parser = argparse.ArgumentParser(
		description="Generate CSVs for plaintext and ciphertext, allowing the user to specify a base name."
	)
	
	parser.add_argument(
		'input_dir',
		type=str
	)
	
	args = parser.parse_args()
	generate_csvs_plain_cipher(args.input_dir)
	k = 10
	j = 10
	for i in range(j):
		cipher_embs, plaintext_emb = read_embeddings(f'embeddings_plain_cipher/cipher-{i}_cipher_embeddings.csv', f'embeddings_plain_cipher/cipher-{i}_plaintext_embeddings.csv')
		mappings = read_mappings(f'embeddings_plain_cipher/cipher-{i}_mappings.csv')
		for letter in ['a', 'e', 'j']:
			print(f"Top {k} similar cipher symbols for {letter.upper()}:")
			find_k_similar_vecs(plaintext_emb[letter], cipher_embs, mappings, k)