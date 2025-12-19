import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional, List
from sklearn.metrics.pairwise import cosine_similarity

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

	# Reshape target_vec for compatibility
	target_vec = target_vec.reshape(1, -1)
	
	similarities = {}
	for key, vec in all_vecs.items():
		vec = vec.reshape(1, -1)
		sim = cosine_similarity(target_vec, vec)[0][0]
		similarities[key] = sim
	
	# Sort by similarity and get top k
	sorted_similarities = dict(sorted(similarities.items(), key=lambda item: item[1], reverse=True)[:k])
	
	for cipher_symbol_key, sim in sorted_similarities.items():
		display_key = cipher_symbol_key # Default to cipher symbol key
		
		# Check if a mapping dictionary was provided
		if mappings is not None:
			plaintext_letter = mappings.get(cipher_symbol_key, f"Symbol {cipher_symbol_key}")
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
		c_df = pd.read_csv(cipher_csv)
		p_df = pd.read_csv(plaintext_csv)
	except FileNotFoundError:
		print(f"Error: File not found")
		return {}
	except Exception as e:
		print(f"An error occurred during file loading: {e}")
		return {}

	symbol_col = c_df.columns[0]
	c_embedding_cols = c_df.columns[1:]
	letter_col = p_df.columns[0]
	p_embedding_cols = p_df.columns[1:]
	
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

def read_mappings(mappings_csv_path: str, reverse=False) -> Dict[Any, str]:
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
	
	if reverse:
		for _, row in df.iterrows():
			try:
				plaintext_letter = str(row[plaintext_col]).upper()
				cipher_symbol = int(row[cipher_col]) 

				if plaintext_letter not in cipher_to_plaintext_map:
					cipher_to_plaintext_map[plaintext_letter] = []

				cipher_to_plaintext_map[plaintext_letter].append(cipher_symbol)
				
			except ValueError:
				continue
	else:
		for _, row in df.iterrows():
			try:
				plaintext_letter = str(row[plaintext_col])
				cipher_symbol = int(row[cipher_col]) 
				cipher_to_plaintext_map[cipher_symbol] = plaintext_letter
			except ValueError:
				continue
			
	print(f"Loaded {len(cipher_to_plaintext_map)} cipher symbol mappings.")
	return cipher_to_plaintext_map

def calculate_avg_similarity_for_letter(letter: str, p_vec: np.ndarray, cipher_embs: Dict[Any, np.ndarray], mappings: Dict[str, List[Any]]) -> Optional[float]:
	"""
	Calculates the cosine similarity between a plaintext letter's embedding 
	and the average vector of all cipher symbols mapped to that letter.

	Args:
		letter (str): The plaintext letter being analyzed.
		p_vec (np.ndarray): The embedding vector for the plaintext letter.
		cipher_embs (Dict[Any, np.ndarray]): Dictionary mapping cipher symbols to their embeddings.
		mappings (Dict[str, List[Any]]): Dictionary mapping plaintext letters to lists of cipher symbols.

	Returns:
		Optional[float]: The cosine similarity score, or None if the letter cannot be processed.
	"""
	# 1. Get the cipher symbols that map to this plaintext letter
	if letter not in mappings:
		print(f"  Skipping {letter.upper()}: No cipher mapping found.")
		return None
	# 2. Collect the embeddings for the mapped cipher symbols
	cipher_vectors = []
	for symbol in mappings[letter]:
		if symbol in cipher_embs:
			cipher_vectors.append(cipher_embs[symbol])
	
	# Handle the case where the letter is mapped, but the symbol embeddings are missing
	if not cipher_vectors:
		print(f"  Skipping {letter.upper()}: No corresponding cipher embeddings found.")
		return None
	
	# 3. Calculate the average/mean vector of the cipher embeddings
	avg_cipher_vec = np.mean(np.stack(cipher_vectors), axis=0)
	
	# 4. Calculate the Cosine Similarity
	# Reshape for sklearn: (1, N) required for single vector comparison
	p_vec_reshaped = p_vec.reshape(1, -1)
	avg_cipher_vec_reshaped = avg_cipher_vec.reshape(1, -1)
	
	# Cosine similarity returns a 2D array [[sim]], so we extract the single float.
	sim = cosine_similarity(p_vec_reshaped, avg_cipher_vec_reshaped)[0][0]
	
	return sim

if __name__ == "__main__":
	import argparse
	import os
	import re
	from generate_embeddings import generate_csvs_plain_cipher
	
	parser = argparse.ArgumentParser(
		description="Generate CSVs for plaintext and ciphertext, allowing the user to specify a base name."
	)
	
	parser.add_argument(
		'--input_dir',
		default=None,
		type=str
	)

	parser.add_argument(
		'--perletter',
		action='store_true',
		default=False
	)
	
	args = parser.parse_args()
	if args.perletter is None and args.input_dir is not None:
		generate_csvs_plain_cipher(args.input_dir)
		k = 10

	emb_dir = 'glove_embeddings_plain_cipher'

	PREFIX_PATTERN = re.compile(r'(.+?)_(mappings|cipher_embeddings|plaintext_embeddings)\.csv$')

	file_prefixes = set()

	for filename in os.listdir(emb_dir):
		match = PREFIX_PATTERN.match(filename)
		if match:
			file_prefixes.add(match.group(1))

	for prefix in sorted(list(file_prefixes)):
		print(f"\nProcessing cipher set with prefix: {prefix}")

		cipher_emb_file = os.path.join(emb_dir, f'{prefix}_cipher_embeddings.csv')
		plaintext_emb_file = os.path.join(emb_dir, f'{prefix}_plaintext_embeddings.csv')
		mappings_file = os.path.join(emb_dir, f'{prefix}_mappings.csv')

		if not (os.path.exists(cipher_emb_file) and \
				os.path.exists(plaintext_emb_file) and \
				os.path.exists(mappings_file)):
			print(f"WARNING: Skipping {prefix} as one or more required files are missing.")
			continue
		cipher_embs, plaintext_emb = read_embeddings(cipher_emb_file, plaintext_emb_file)
		mappings = read_mappings(mappings_file)
		if args.perletter is None and args.input_dir is not None:
			for letter in ['a', 'e', 'j']:
				if letter in plaintext_emb:
					print(f"Top {k} similar cipher symbols for {letter.upper()} (from {prefix}):")
					find_k_similar_vecs(plaintext_emb[letter], cipher_embs, mappings, k)
				else:
					print(f"Skipping {letter.upper()}: Plaintext embedding not found for this letter.")
		else:
			inverted_mappings = read_mappings(mappings_file, reverse=True)

			print("Running Per-Letter Analysis Mode: Plaintext vs. Average Cipher Similarity.")
			
			for letter, p_vec in plaintext_emb.items():
				# PASS THE INVERTED MAPPINGS
				sim = calculate_avg_similarity_for_letter(
					letter.upper(), 
					p_vec, 
					cipher_embs, 
					inverted_mappings
				)
				
				if sim is not None:
					print(f"  {letter.upper()}: Similarity (Plain vs. Avg Cipher) = {sim:.4f}")