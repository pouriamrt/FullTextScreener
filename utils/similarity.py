from sklearn.metrics.pairwise import cosine_similarity

def compute_similar_chunks(chunks, criteria_embeddings, model, get_embedding, threshold):
    matched_chunks = []

    for chunk in chunks:
        emb = get_embedding(chunk["text"], model)
        scores = cosine_similarity([emb], criteria_embeddings)[0]
        max_idx = scores.argmax()
        if scores[max_idx] >= threshold:
            chunk["criterion_id"] = max_idx
            matched_chunks.append(chunk)

    return matched_chunks
