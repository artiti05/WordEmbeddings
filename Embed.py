import os
import numpy as np
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
import gensim.downloader as api

def load_word2vec():
    print("Loading Word2Vec model via gensim.downloader...")
    model = api.load('word2vec-google-news-300')
    print("Word2Vec loaded successfully!")
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

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot_product / (norm1 * norm2)

def predict_analogy(model, a, b, c, k=5):
    for word in [a, b, c]:
        if word not in model:
            return []
            
    target_vector = model[b] - model[a] + model[c]
    
    all_vectors = model.vectors
    
    dot_products = np.dot(all_vectors, target_vector)
    
    norms_all = np.linalg.norm(all_vectors, axis=1)
    norm_target = np.linalg.norm(target_vector)
    
    similarities = dot_products / (norms_all * norm_target)
    
    top_indices = np.argsort(similarities)[-(k+3):][::-1]
    
    results = []
    for idx in top_indices:
        word = model.index_to_key[idx]
        if word not in [a, b, c]:
            results.append(word)
        if len(results) == k:
            break
            
    return results

semantic_tests = [
    ("france", "paris", "japan", "tokyo"),
    ("italy", "rome", "egypt", "cairo"),
    ("spain", "madrid", "germany", "berlin"),
    ("russia", "moscow", "china", "beijing"),
    ("canada", "ottawa", "australia", "canberra"),
    ("brazil", "brasilia", "argentina", "buenos_aires"),
    ("india", "new_delhi", "turkey", "ankara"),
    ("sweden", "stockholm", "norway", "oslo"),
    ("greece", "athens", "mexico", "mexico_city"),
    ("thailand", "bangkok", "vietnam", "hanoi")
]

syntactic_tests = [
    ("quick", "quickly", "careful", "carefully"),
    ("slow", "slowly", "loud", "loudly"),
    ("happy", "happily", "sad", "sadly"),
    ("quiet", "quietly", "gentle", "gently"),
    ("safe", "safely", "dangerous", "dangerously"),
    ("easy", "easily", "hard", "hardly"),
    ("bright", "brightly", "dim", "dimly"),
    ("smooth", "smoothly", "rough", "roughly"),
    ("bold", "boldly", "shy", "shyly"),
    ("polite", "politely", "rude", "rudely")
]

def evaluate_analogies(models, tests, test_type="Semantic"):

    model_names = ["GloVe 300", "Word2Vec 300", "FastText 300"]
    
    print(f"\n--- {test_type} Analogy Tests ---")
    header = f"{'Test #':<8} | {'Test format':<40} | {'Expected':<15} | {'GloVe Result':<15} | {'Word2Vec Result':<15} | {'FastText Result':<15}"
    print(header)
    print("-" * len(header))
    
    stats = {name: {"top1": 0, "top5": 0} for name in model_names}
    
    for i, (a, b, c, expected) in enumerate(tests, 1):
        test_str = f"{a}:{b} :: {c}:?"
        results_str = []
        
        for name, model in zip(model_names, models):
            predictions = predict_analogy(model, a, b, c, k=5)
            
            top1_result = predictions[0] if len(predictions) > 0 else "N/A"
            results_str.append(top1_result)
            
            if len(predictions) > 0 and predictions[0].lower() == expected.lower():
                stats[name]["top1"] += 1
            if any(p.lower() == expected.lower() for p in predictions):
                stats[name]["top5"] += 1
                
        print(f"{i:<8} | {test_str:<40} | {expected:<15} | {results_str[0]:<15} | {results_str[1]:<15} | {results_str[2]:<15}")
        
    return stats


def main():
    glove_path = "glove.2024.wikigiga.300d/wiki_giga_2024_300_MFT20_vectors_seed_2024_alpha_0.75_eta_0.05_combined.txt"
    
    print("Initiating Loaders. This may take a few minutes depending on memory and download speed...")
    try:
        w2v_model = load_word2vec()
        glove_model = load_gloveE(glove_path)
        fasttext_model = load_fasttext()
    except FileNotFoundError as e:
        print(f"\nERROR: Could not find GloVe model file. {e}")
        print("Please download the GloVe model and update the path in main() before running.")
        return

    models = [glove_model, w2v_model, fasttext_model]
    
    semantic_stats = evaluate_analogies(models, semantic_tests, "Semantic")
    syntactic_stats = evaluate_analogies(models, syntactic_tests, "Syntactic")
    
    print("\n\n" + "="*80)
    print("Analogy Top-1 Accuracy Summary (3 points)")
    print("="*80)
    header = f"{'Analogy Type':<15} | {'GloVe 300 Accuracy':<20} | {'Word2Vec 300 Accuracy':<25} | {'FastText 300':<20}"
    print(header)
    print("-" * len(header))
    
    s_top1 = [f"{(semantic_stats[m]['top1']/10)*100:.1f}%" for m in ["GloVe 300", "Word2Vec 300", "FastText 300"]]
    syn_top1 = [f"{(syntactic_stats[m]['top1']/10)*100:.1f}%" for m in ["GloVe 300", "Word2Vec 300", "FastText 300"]]
    
    print(f"{'Semantic':<15} | {s_top1[0]:<20} | {s_top1[1]:<25} | {s_top1[2]:<20}")
    print(f"{'Syntactic':<15} | {syn_top1[0]:<20} | {syn_top1[1]:<25} | {syn_top1[2]:<20}")

    print("\n\n" + "="*80)
    print("Analogy Top-5 Accuracy Summary (3 points)")
    print("="*80)
    print(header)
    print("-" * len(header))
    
    s_top5 = [f"{(semantic_stats[m]['top5']/10)*100:.1f}%" for m in ["GloVe 300", "Word2Vec 300", "FastText 300"]]
    syn_top5 = [f"{(syntactic_stats[m]['top5']/10)*100:.1f}%" for m in ["GloVe 300", "Word2Vec 300", "FastText 300"]]
    
    print(f"{'Semantic':<15} | {s_top5[0]:<20} | {s_top5[1]:<25} | {s_top5[2]:<20}")
    print(f"{'Syntactic':<15} | {syn_top5[0]:<20} | {syn_top5[1]:<25} | {syn_top5[2]:<20}")



main()