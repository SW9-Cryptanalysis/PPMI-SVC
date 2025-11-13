# PPMI-SVC
### Repository used to generate embeddings using PPMI, plotting embeddings, comparing embeddings between cipher symbols and plaintext letters, and training a support vector classifier. 

Run ```uv sync``` to install necessary packages. 

To generate PPMI embeddings run: 
``` shell
uv run generate_embeddings.py --input_dir CIPHERS_DIR
default input_dir: ciphers
```
where default CIPHERS_DIR = ciphers.

To plot cipher embeddings: 
``` shell
uv run plot_embeddings.py --filepath PATH_TO_CIPHER_FILE --simple | --cluster | --vc
```
where --simple indicates a regular plot, --cluster uses hdbscan to cluster vectors, and --vc color codes vowels/consonants.

To compare cipher & plaintext embeddings