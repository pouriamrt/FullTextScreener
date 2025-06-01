from sklearn.metrics.pairwise import cosine_similarity
from utils.embedding import get_batch_embeddings

def compute_similar_chunks(chunks, criteria_embeddings, model, threshold):
    texts = [chunk["text"] for chunk in chunks]

    chunk_embeddings = get_batch_embeddings(texts, model)

    matched_chunks = []
    for i, (chunk, emb) in enumerate(zip(chunks, chunk_embeddings)):
        scores = cosine_similarity([emb], criteria_embeddings)[0]
        max_idx = scores.argmax()
        if scores[max_idx] >= threshold:
            chunk["criterion_id"] = max_idx
            matched_chunks.append(chunk)

    return matched_chunks
