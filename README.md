# WordEmbeddings
The link which has the Embeddings
https://nlp.stanford.edu/projects/glove/
from stanford university for Glove embeddings 


# WordEmbeddings

A Python project for loading, evaluating, and visualising word embedding models including **GloVe**, **Word2Vec**, and **FastText**  on semantic and syntactic analogy tasks.

## Overview

| Script | Purpose |
|---|---|
| `main_p1.py` | Compare **GloVe 300d**, **Word2Vec 300d**, and **FastText 300d** on analogy tests (Top-1 & Top-5 accuracy). |
| `main_p2.py` | Compare **GloVe** at different dimensionalities (**300d / 100d / 50d**) on the same analogy tests. |
| `main_p3.py` | Produce a **t-SNE** visualisation comparing **GloVe 100d** vs a custom **Word2Vec 100d** model across word clusters (countries, companies, animals). |
| `w2v_100d.py` | Train a custom **Word2Vec 100d** model on the *text8* corpus and save it as `wiki_text8_100d.bin`. |
| `load_embeddings.py` | Helper functions to load Word2Vec, GloVe, and FastText embeddings (via Gensim). |
| `tests.py` | Defines semantic & syntactic analogy test sets and the evaluation loop. |
| `utils.py` | Low-level utilities: cosine similarity and the analogy prediction function. |

## Requirements

- Python 3.9+
- [Gensim](https://radimrehurek.com/gensim/)
- NumPy
- scikit-learn (for t-SNE in `main_p3.py`)
- Matplotlib (for plotting in `main_p3.py`)

Install dependencies:

```bash
pip install gensim numpy scikit-learn matplotlib
```

