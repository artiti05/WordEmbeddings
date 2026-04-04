import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from load_embeddings import *

def plot_tsne_comparison(glove_model, w2v_model, word_sets):
    words = []
    labels = []
    colors = ['red', 'blue', 'green']
    color_map = []

    for i, (category, word_list) in enumerate(word_sets.items()):
        for word in word_list:
            if word in glove_model and word in w2v_model:
                words.append(word)
                labels.append(category)
                color_map.append(colors[i])
            else:
                print(f"Skipping '{word}' - not found in vocabulary.")

    glove_vectors = np.array([glove_model[w] for w in words])
    w2v_vectors = np.array([w2v_model[w] for w in words])

    tsne = TSNE(n_components=2, perplexity=5, random_state=42)

    glove_2d = tsne.fit_transform(glove_vectors)

    w2v_2d = tsne.fit_transform(w2v_vectors)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    ax1.scatter(glove_2d[:, 0], glove_2d[:, 1], c=color_map, s=100, alpha=0.7)
    for i, word in enumerate(words):
        ax1.annotate(word, (glove_2d[i, 0], glove_2d[i, 1]), xytext=(5, 2),
                     textcoords='offset points', fontsize=10)

    ax1.set_title("GloVe (100d) t-SNE Clustering", fontsize=14, fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.5)

    ax2.scatter(w2v_2d[:, 0], w2v_2d[:, 1], c=color_map, s=100, alpha=0.7)
    for i, word in enumerate(words):
        ax2.annotate(word, (w2v_2d[i, 0], w2v_2d[i, 1]), xytext=(5, 2),
                     textcoords='offset points', fontsize=10)

    ax2.set_title("Word2Vec (100d) t-SNE Clustering", fontsize=14, fontweight='bold')
    ax2.grid(True, linestyle='--', alpha=0.5)

    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], marker='o', color='w', label=cat,
                              markerfacecolor=colors[i], markersize=10)
                       for i, cat in enumerate(word_sets.keys())]

    fig.legend(handles=legend_elements, loc='lower center', ncol=3, fontsize=12)

    plt.tight_layout(rect=(0.0, 0.05, 1.0, 1.0))  # Leave room for legend
    plt.show()

def main():
    glove_path_100 = "glove.2024.wikigiga.100d/wiki_giga_2024_100_MFT20_vectors_seed_2024_alpha_0.75_eta_0.05.050_combined.txt"
    glove_model_100 = load_gloveE(glove_path_100)
    w2v_model = load_word2vec_100d()

    word_sets = {
        "Countries": [
            "jordan", "egypt", "japan", "brazil", "germany", "canada",
            "france", "italy", "spain", "mexico", "china", "india"
        ],

        "Companies": [
            "microsoft", "google", "samsung", "ibm", "sony",
            "intel", "nintendo", "sega", "nvidia", "amd"
        ],

        "Animals": [
            "cat", "dog", "lion", "tiger", "elephant", "bear",
            "wolf", "shark", "eagle", "monkey", "snake", "horse"
        ]
    }
    plot_tsne_comparison(glove_model_100, w2v_model, word_sets)
if __name__ == "__main__":
    main()