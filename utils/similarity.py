from sklearn.metrics.pairwise import cosine_similarity
from utils.embedding import get_batch_embeddings
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from config import CRITERIA_COLORS, PLOT_FOLDER

def compute_similar_chunks(chunks, criteria_embeddings, filename, model, thresholds):
    texts = [chunk["text"] for chunk in chunks]
    
    if len(texts) == 0:
        return None
    
    chunk_embeddings = get_batch_embeddings(texts, model)

    top_scores = []
    matched_chunks = []
    for i, (chunk, emb) in enumerate(tqdm(zip(chunks, chunk_embeddings))):
        scores = cosine_similarity([emb], criteria_embeddings)[0]
        max_idx = scores.argmax()
        top_scores.append(scores[max_idx])
        if scores[max_idx] >= thresholds[max_idx]:
            chunk["criterion_id"] = max_idx
            matched_chunks.append(chunk)

    # print("Top 10 scores:", sorted(top_scores, reverse=True)[:10])
    plot_cosine_similarity_distribution(top_scores, thresholds, filename)
    return matched_chunks


def plot_cosine_similarity_distribution(scores, thresholds, filename, bins=50, title="Cosine Similarity Score Distribution"):
    plt.figure(figsize=(10, 6))
    plt.hist(scores, bins=bins, edgecolor='black')
    for i, threshold in enumerate(thresholds):
        plt.axvline(x=threshold, color=CRITERIA_COLORS[i], linestyle='--', label=f'Threshold ({threshold})')
    plt.title(title, fontsize=14)
    plt.xlabel("Cosine Similarity Score")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if len(filename) > 60:
        filename = filename[:60] + "..."
    plt.savefig(os.path.join(PLOT_FOLDER, f"{filename}_cosine_similarity_distribution.png"))
    plt.close()
    