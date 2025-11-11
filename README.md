# PPMI-SVC
Positive Pointwise Mutual Information and Support Vector Classifier

uv sync

To generate PPMI embeddings: uv run generate_embeddings.py --input_dir [name of ciphers dir]
    default input_dir: ciphers

To plot cipher embeddings: 

uv run plot_embeddings.py --filepath <PATH_TO_CIPHER_FILE> {--simple | --cluster | --vc}

To compare cipher & plaintext embeddings