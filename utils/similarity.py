from sklearn.metrics.pairwise import cosine_similarity
from utils.embedding import get_batch_embeddings
from tqdm import tqdm

def compute_similar_chunks(chunks, criteria_embeddings, model, threshold):
    texts = [chunk["text"] for chunk in chunks]

    chunk_embeddings = get_batch_embeddings(texts, model)

    top_scores = []
    matched_chunks = []
    for i, (chunk, emb) in enumerate(tqdm(zip(chunks, chunk_embeddings))):
        scores = cosine_similarity([emb], criteria_embeddings)[0]
        max_idx = scores.argmax()
        top_scores.append(scores[max_idx])
        if scores[max_idx] >= threshold:
            chunk["criterion_id"] = max_idx
            matched_chunks.append(chunk)

    print("Top 10 scores:", sorted(top_scores, reverse=True)[:10])
    return matched_chunks
