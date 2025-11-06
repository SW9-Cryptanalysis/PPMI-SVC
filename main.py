#from classify_embeddings import load_svm_gpu, train_svm_gpu
from embedding_comparison import read_embeddings, find_k_similar_vecs, read_mappings
from generate_embeddings import generate_csvs_plain_cipher, generate_csvs, generate_embeddings_txt_file, create_embeddings_csv

#generate_csvs()
#train_svm_gpu(vowels_consonants=True)
#load_svm_gpu()
#generate_csvs_plain_cipher()

"""
k = 10
emb_type = 'ppmi' # glove or ppmi
cipher_embs, plaintext_emb = read_embeddings(f'{emb_type}_embeddings/cipher-30_cipher_embeddings.csv', f'{emb_type}_embeddings/cipher-30_plaintext_embeddings.csv')
mappings = read_mappings(f'{emb_type}_embeddings/cipher-30_mappings.csv')
for letter in ['a', 'e', 'z']:
    print(f"Top {k} similar cipher symbols for {letter.upper()}:")
    find_k_similar_vecs(plaintext_emb[letter], cipher_embs, mappings, k)
#print(f"Top {k} similar cipher symbols for {next(iter(plaintext_emb.items()))[0].upper()}:")
#find_k_similar_vecs(next(iter(plaintext_emb.items()))[1], cipher_embs, mappings, k)
"""
#load_svm_gpu()

#create_embeddings_csv('vocab/text8', window=2, embedding_size=50)

generate_csvs_plain_cipher()