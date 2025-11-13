# PPMI-SVC
### Repository used to generate embeddings using PPMI, plotting embeddings, comparing embeddings between cipher symbols and plaintext letters, and training a support vector classifier. 

#### Run ```uv sync``` to install necessary packages. 

**To generate PPMI embeddings run:** 
``` shell
uv run generate_embeddings.py --input_dir CIPHERS_DIR
```
where default input_dir = ciphers.

**To plot cipher embeddings:**
``` shell
uv run plot_embeddings.py --filepath PATH_TO_CIPHER_FILE --simple | --cluster | --vc
```
where --simple indicates a regular plot, --cluster uses hdbscan to cluster vectors, and --vc color codes vowels/consonants.

**To compare cipher & plaintext embeddings**
``` shell
uv run embedding_comparison.py CIPHERS_DIR
```

**To train/test SVC - requires generating embeddings beforehand:**
``` shell
python main.py MODEL_NAME --train | --test
```
```--train``` is used for training the model with specified name, ```--test``` to evaluate the model with given name. 

***NOTE: training or testing the model requires the cuML package from the RAPIDS library for GPU acceleration: https://docs.rapids.ai/. This cannot be installed through astral uv - see https://docs.rapids.ai/install/ for installation.***