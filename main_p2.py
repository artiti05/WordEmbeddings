from load_embeddings import *
from tests import *


def main():
    glove_path_300 = "wiki_giga_2024_300_MFT20_vectors_seed_2024_alpha_0.75_eta_0.05_combined.txt"
    glove_path_100 = "wiki_giga_2024_100_MFT20_vectors_seed_2024_alpha_0.75_eta_0.05.050_combined.txt"
    glove_path_50 = "wiki_giga_2024_50_MFT20_vectors_seed_123_alpha_0.75_eta_0.075_combined.txt"
    print("Initiating Loaders.")
    try:
        glove_model_300 = load_gloveE(glove_path_300)
        glove_model_100 = load_gloveE(glove_path_100)
        glove_model_50 = load_gloveE(glove_path_50)

    except FileNotFoundError as e:
        print(f"\nERROR: Could not find GloVe model file. {e}")
        return

    models = {"GloVe 300": glove_model_300,"GloVe 100": glove_model_100, "GloVe 50": glove_model_50}

    semantic_stats = evaluate_analogies(models, semantic_test(), "Semantic")
    syntactic_stats = evaluate_analogies(models, syntactic_test(), "Syntactic")

    print("\n\n" + "=" * 80)
    print("Analogy Top-1 Accuracy Summary")
    print("=" * 80)
    header = f"{'Analogy Type':<15} | {'GloVe 300 Accuracy':<20} | {'GloVe 100 Accuracy':<25} | {'GloVe 50 Accuracy':<20} "
    print(header)
    print("-" * len(header))

    s_top1 = [f"{(semantic_stats[m]['top1'] / 10) * 100:.1f}%" for m in ["GloVe 300","GloVe 100", "GloVe 50"]]
    syn_top1 = [f"{(syntactic_stats[m]['top1'] / 10) * 100:.1f}%" for m in
                ["GloVe 300", "GloVe 100", "GloVe 50"]]

    print(f"{'Semantic':<15} | {s_top1[0]:<20} | {s_top1[1]:<25} | {s_top1[2]:<20}")
    print(f"{'Syntactic':<15} | {syn_top1[0]:<20} | {syn_top1[1]:<25} | {syn_top1[2]:<20}")

    print("\n\n" + "=" * 80)
    print("Analogy Top-5 Accuracy Summary")
    print("=" * 80)
    print(header)
    print("-" * len(header))

    s_top5 = [f"{(semantic_stats[m]['top5'] / 10) * 100:.1f}%" for m in  ["GloVe 300","GloVe 100", "GloVe 50"]]
    syn_top5 = [f"{(syntactic_stats[m]['top5'] / 10) * 100:.1f}%" for m in
                ["GloVe 300", "GloVe 100", "GloVe 50"]]

    print(f"{'Semantic':<15} | {s_top5[0]:<20} | {s_top5[1]:<25} | {s_top5[2]:<20}")
    print(f"{'Syntactic':<15} | {syn_top5[0]:<20} | {syn_top5[1]:<25} | {syn_top5[2]:<20}")


if __name__ == "__main__":
    main()