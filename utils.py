import numpy as np

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

    top_indices = np.argsort(similarities)[-(k + 3):][::-1]

    results = []
    for idx in top_indices:
        word = model.index_to_key[idx]
        if word not in [a, b, c]:
            results.append(word)
        if len(results) == k:
            break

    return results