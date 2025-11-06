def plot_embeddings_2d(embeddings, vocab, save_path=None, random_state=42, cipher_to_letter=None):
    """
    Visualize embeddings in 2D using t-SNE and matplotlib.
    Args:
        embeddings: np.ndarray, shape (n_symbols, n_features)
        vocab: list of symbols (same order as embeddings)
        save_path: if given, saves the plot to this path; else shows the plot
        random_state: random seed for reproducibility
        cipher_to_letter: dict mapping cipher symbol (as str) to letter (as str), optional
    """
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE

    tsne = TSNE(n_components=2, random_state=random_state, init='pca', perplexity=min(30, len(vocab)-1))
    emb_2d = tsne.fit_transform(embeddings)

    plt.figure(figsize=(10, 8))
    plt.scatter(emb_2d[:, 0], emb_2d[:, 1], s=60, alpha=0.7)
    for i, cipher_symbol in enumerate(vocab):
        label = str(cipher_symbol)
        if cipher_to_letter and cipher_symbol in cipher_to_letter:
            label = f"{cipher_symbol}:{cipher_to_letter[cipher_symbol]}"
        plt.text(emb_2d[i, 0], emb_2d[i, 1], label, fontsize=9, ha='center', va='center')
    plt.title('Cipher Symbol Embeddings (t-SNE 2D)')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    else:
        plt.show()

def plot_embeddings_2d_cluster(embeddings, vocab, save_path=None, random_state=42, cipher_to_letter=None, min_cluster_size=2):
    """
    Visualize embeddings in 2D using t-SNE and matplotlib, with HDBSCAN clustering.
    Args:
        embeddings: np.ndarray, shape (n_symbols, n_features)
        vocab: list of symbols (same order as embeddings)
        save_path: if given, saves the plot to this path; else shows the plot
        random_state: random seed for reproducibility
        cipher_to_letter: dict mapping cipher symbol (as str) to letter (as str)
        min_cluster_size: minimum cluster size
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    from sklearn.cluster import HDBSCAN

    clusterer = HDBSCAN(min_cluster_size=min_cluster_size, metric="euclidean", cluster_selection_method="eom", store_centers="medoid")
    labels = clusterer.fit_predict(embeddings)
    
    medoid_symbols = []

    for medoid in clusterer.medoids_:
        distances = np.linalg.norm(embeddings - medoid, axis=1)
        idx = np.argmin(distances)
        medoid_symbols.append(vocab[idx])

    for cluster_id, symbol in enumerate(medoid_symbols):
        print(f"Cluster {cluster_id}: medoid symbol = {symbol}")
    tsne = TSNE(
        n_components=2, random_state=random_state,
        init='pca', perplexity=min(30, len(vocab)-1)
    )
    emb_2d = tsne.fit_transform(embeddings)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        emb_2d[:, 0], emb_2d[:, 1],
        c=labels, cmap="tab20", s=60, alpha=0.7
    )

    for i, cipher_symbol in enumerate(vocab):
        label = str(cipher_symbol)
        if cipher_to_letter and cipher_symbol in cipher_to_letter:
            label = f"{cipher_symbol}:{cipher_to_letter[cipher_symbol]}"
        plt.text(
            emb_2d[i, 0], emb_2d[i, 1], label,
            fontsize=9, ha="center", va="center"
        )

    plt.title("Cipher Symbol Embeddings (t-SNE 2D with HDBSCAN Clusters)")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.tight_layout()

    unique_labels = sorted(set(labels))
    handles, _ = scatter.legend_elements()
    plt.legend(handles, [f"Cluster {c}" if c != -1 else "Noise" for c in unique_labels],
               title="Clusters")

    if save_path:
        plt.savefig(save_path, dpi=150)
    else:
        plt.show()

def plot_embeddings_2d_vc(embeddings, vocab, random_state=42, cipher_to_letter=None, save_path=None):
    """
    Simple function to plot embeddings in 2D with vowel/consonant coloring.
    Args:
        embeddings: np.ndarray, shape (n_symbols, n_features)
        vocab: list of symbols (same order as embeddings)
        random_state: random seed for reproducibility
        cipher_to_letter: dict mapping cipher symbol (as str) to letter (as str)
        save_path: if given, saves the plot to this path; else shows the plot
    """
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    VOWELS = {'a', 'e', 'i', 'o', 'u'}
    tsne = TSNE(
        n_components=2, random_state=random_state,
        init='pca', perplexity=min(30, len(vocab)-1)
    )
    emb_2d = tsne.fit_transform(embeddings)
    colors = []
    for symbol in vocab:
        letter = cipher_to_letter.get(symbol, None) if cipher_to_letter else None
        if letter in VOWELS:
            colors.append('red')
        else:
            colors.append('blue')
    plt.figure(figsize=(10, 8))
    plt.scatter(emb_2d[:, 0], emb_2d[:, 1], c=colors, s=60, alpha=0.7)
    for i, cipher_symbol in enumerate(vocab):
        label = str(cipher_symbol)
        if cipher_to_letter and cipher_symbol in cipher_to_letter:
            label = f"{cipher_symbol}:{cipher_to_letter[cipher_symbol]}"
        plt.text(emb_2d[i, 0], emb_2d[i, 1], label, fontsize=9, ha='center', va='center')

    plt.title("Cipher Symbol Embeddings (t-SNE 2D Vowel/Consonant)")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    else:
        plt.show()