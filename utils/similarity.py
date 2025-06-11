from sklearn.metrics.pairwise import cosine_similarity
from utils.embedding import get_batch_embeddings
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from config import CRITERIA_COLORS, PLOT_FOLDER, CRITERIA_LABELS
import numpy as np

def compute_similar_chunks_adaptive(
    chunks,
    inclusion_criteria_embeddings,
    exclusion_criteria_embeddings,
    filename,
    model,
    thresholds=None,                 # fallback static vector
    percentile=75,                   # for percentile rule
    k=None,                          # set k (e.g. 1.5) to switch to mean+kσ rule
    min_floor=0.02,                  # absolute minimum threshold
    use_static=False,                # force old behaviour
    return_thresholds=False          # handy for debugging
):
    """
    Return list of matched chunks + (optionally) the dynamic threshold vector.

    If k is None  ➜ percentile rule
    If k is float ➜ mean + k·σ rule
    """
    texts = [c["text"] for c in chunks]
    if not texts:
        return (None, None) if return_thresholds else None

    # 1️⃣ Embed once
    chunk_emb = get_batch_embeddings(texts, model)

    # 2️⃣ Compute Δ-similarity matrix
    sim_incl = cosine_similarity(chunk_emb, inclusion_criteria_embeddings)
    sim_excl = cosine_similarity(chunk_emb, exclusion_criteria_embeddings)
    scores_matrix = sim_incl - sim_excl

    # 3️⃣ Derive dynamic thresholds -----------------------------------------
    if use_static and thresholds is not None:
        thr_dyn = np.asarray(thresholds)
    else:
        if k is None:  # percentile rule
            thr_dyn = np.percentile(scores_matrix, percentile, axis=0)
        else:          # mean + k·σ rule
            mu  = scores_matrix.mean(axis=0)
            std = scores_matrix.std(axis=0)
            thr_dyn = mu + k * std

        # enforce absolute floor
        thr_dyn = np.maximum(thr_dyn, min_floor)

    # 4️⃣ Select matched chunks ---------------------------------------------
    matched_chunks = []
    top_scores = scores_matrix.max(axis=1)       # best per chunk
    best_idxs   = scores_matrix.argmax(axis=1)   # which criterion

    for chunk, best_score, best_idx in zip(chunks, top_scores, best_idxs):
        if best_score >= thr_dyn[best_idx]:
            chunk["criterion_id"] = int(best_idx)
            matched_chunks.append(chunk)

    # 5️⃣ Plot distribution (unchanged from your code)
    plot_cosine_similarity_distribution(top_scores.tolist(), thr_dyn, filename)

    if return_thresholds:
        return matched_chunks, thr_dyn
    return matched_chunks


def compute_similar_chunks(chunks, inclusion_criteria_embeddings, exclusion_criteria_embeddings, filename, model, thresholds):
    texts = [chunk["text"] for chunk in chunks]
    
    if len(texts) == 0:
        return None
    
    chunk_embeddings = get_batch_embeddings(texts, model)

    top_scores = []
    matched_chunks = []
    scores_list = []
    for i, (chunk, emb) in enumerate(tqdm(zip(chunks, chunk_embeddings))):
        similarity_inclusion = cosine_similarity([emb], inclusion_criteria_embeddings)[0]
        similarity_exclusion = cosine_similarity([emb], exclusion_criteria_embeddings)[0]
        scores = similarity_inclusion - similarity_exclusion
        max_idx = scores.argmax()
        top_scores.append(scores[max_idx])
        if scores[max_idx] >= thresholds[max_idx]:
            chunk["criterion_id"] = max_idx
            matched_chunks.append(chunk)
        # plot_chunk_criteria_similarities(scores, CRITERIA_LABELS[max_idx], i, filename, chunk["text"], thresholds[max_idx], max_idx)
        scores_list.append(scores)
        
    # print("Top 10 scores:", sorted(top_scores, reverse=True)[:10])
    plot_cosine_similarity_distribution(top_scores, thresholds, filename)
    return matched_chunks


def plot_cosine_similarity_distribution(scores, thresholds, filename, bins=50, title="Subtracted Cosine Similarity Score Distribution"):
    plt.figure(figsize=(10, 6))
    plt.hist(scores, bins=bins, edgecolor='black')
    for i, threshold in enumerate(thresholds):
        plt.axvline(x=threshold, color=CRITERIA_COLORS[i], linestyle='--', label=f'{CRITERIA_LABELS[i]} ({threshold})')
    plt.title(title, fontsize=14)
    plt.xlabel("Subtracted Cosine Similarity Score")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if len(filename) > 70:
        filename = filename[:70] + "..."
    plt.savefig(os.path.join(PLOT_FOLDER, f"{filename}_similarity_distribution.png"))
    plt.close()


def plot_chunk_criteria_similarities(scores, criterion_label, chunk_idx, filename, chunk_text, threshold, i, top_n=10):
    plt.figure(figsize=(12, 6))
    
    plt.hist(scores, bins=40, alpha=0.5, label='Scores', color='blue')
    plt.axvline(x=threshold, color=CRITERIA_COLORS[i], linestyle='--', label=f'{CRITERIA_LABELS[i]} ({threshold})')
    
    plt.xlabel("Cosine Similarity Score")
    plt.ylabel("Frequency")
    plt.title(f"Distribution of Similarity Scores for Chunk {chunk_idx}\n{criterion_label}\n{chunk_text[:top_n]}...")
    plt.legend()
    plt.xlim(-0.05, 0.2)
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    
    pic_filename = f"{filename[:50]}_chunk{chunk_idx}_similarity_histogram_{criterion_label}.png"
    plt.savefig(os.path.join(PLOT_FOLDER, pic_filename))
    plt.close()
