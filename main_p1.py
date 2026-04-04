from load_embeddings import *
from tests import *

def main():
    glove_path = "glove.2024.wikigiga.300d/wiki_giga_2024_300_MFT20_vectors_seed_2024_alpha_0.75_eta_0.05_combined.txt"

    print("Initiating Loaders.")
    try:
        w2v_model = load_word2vec()
        glove_model = load_gloveE(glove_path)
        fasttext_model = load_fasttext()
    except FileNotFoundError as e:
        print(f"\nERROR: Could not find GloVe model file. {e}")
        return

    models = {"GloVe 300":glove_model,  "Word2Vec 300":w2v_model, "FastText 300": fasttext_model}
    
    semantic_stats = evaluate_analogies(models, semantic_test(), "Semantic")
    syntactic_stats = evaluate_analogies(models, syntactic_test(), "Syntactic")
    
    print("\n\n" + "="*80)
    print("Analogy Top-1 Accuracy Summary")
    print("="*80)
    header = f"{'Analogy Type':<15} | {'GloVe 300 Accuracy':<20} | {'Word2Vec 300 Accuracy':<25} | {'FastText 300':<20}"
    print(header)
    print("-" * len(header))
    
    s_top1 = [f"{(semantic_stats[m]['top1']/10)*100:.1f}%" for m in ["GloVe 300", "Word2Vec 300", "FastText 300"]]
    syn_top1 = [f"{(syntactic_stats[m]['top1']/10)*100:.1f}%" for m in ["GloVe 300", "Word2Vec 300", "FastText 300"]]
    
    print(f"{'Semantic':<15} | {s_top1[0]:<20} | {s_top1[1]:<25} | {s_top1[2]:<20}")
    print(f"{'Syntactic':<15} | {syn_top1[0]:<20} | {syn_top1[1]:<25} | {syn_top1[2]:<20}")

    print("\n\n" + "="*80)
    print("Analogy Top-5 Accuracy Summary")
    print("="*80)
    print(header)
    print("-" * len(header))
    
    s_top5 = [f"{(semantic_stats[m]['top5']/10)*100:.1f}%" for m in ["GloVe 300", "Word2Vec 300", "FastText 300"]]
    syn_top5 = [f"{(syntactic_stats[m]['top5']/10)*100:.1f}%" for m in ["GloVe 300", "Word2Vec 300", "FastText 300"]]
    
    print(f"{'Semantic':<15} | {s_top5[0]:<20} | {s_top5[1]:<25} | {s_top5[2]:<20}")
    print(f"{'Syntactic':<15} | {syn_top5[0]:<20} | {syn_top5[1]:<25} | {syn_top5[2]:<20}")


if __name__ == "__main__":
    main()