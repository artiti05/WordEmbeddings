from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
import gensim.downloader as api
import os

def load_word2vec():
    print("Loading Word2Vec model via gensim.downloader...")
    model = api.load('word2vec-google-news-300')
    print("Word2Vec loaded successfully!")
    return model

def load_word2vec_100d():
    print("Loading custom Word2Vec 100d model...")
    model = KeyedVectors.load_word2vec_format('wiki_text8_100d.bin', binary=True)
    print("Word2Vec 100d loaded successfully!")
    return model

def load_gloveE(glove_path):
    word2vec_output_file = glove_path + ".word2vec"
    if not os.path.exists(word2vec_output_file):
        print("Converting GloVe format to Word2Vec format...")
        glove2word2vec(glove_path, word2vec_output_file)
    print("Loading GloVe embeddings...")
    model = KeyedVectors.load_word2vec_format(word2vec_output_file, binary=False)
    print("GloVe embeddings loaded successfully!")
    return model

def load_fasttext():
    print("Loading FastText embeddings via gensim.downloader...")
    model = api.load('fasttext-wiki-news-subwords-300')
    print("FastText embeddings loaded successfully!")
    return model