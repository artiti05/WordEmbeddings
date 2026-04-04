import gensim.downloader as api
from gensim.models import Word2Vec

def train_custom_word2vec():
    dataset = api.load("text8")

    model = Word2Vec(
        sentences=dataset,
        vector_size=100,
        window=5,
        min_count=2,
        workers=11,
        epochs=15,
        sg=1
    )

    model.wv.save_word2vec_format('wiki_text8_100d.bin', binary=True)


if __name__ == "__main__":
    train_custom_word2vec()