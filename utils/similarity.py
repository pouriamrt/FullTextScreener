from sklearn.metrics.pairwise import cosine_similarity
from utils.embedding import get_batch_embeddings
from tqdm import tqdm
import numpy as np
import re
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from utils.plotting import (
    plot_cosine_similarity_distribution, 
    plot_chunk_criteria_similarities, 
    plot_fuzzy_output_surface, 
    plot_all_membership_functions
)

############################# Fuzzy Logic ###########################################
_SCORE = ctrl.Antecedent(np.linspace(0, 0.2, 201), 'score')      # Δ-similarity
_MARG  = ctrl.Antecedent(np.linspace(0, 0.1, 101), 'margin')     # best−second
_DEC   = ctrl.Consequent(np.linspace(0, 1,   101), 'decision')   # μ-membership

_SCORE['low']    = fuzz.trimf(_SCORE.universe, [0.00,  0.00,  0.06])
_SCORE['medium'] = fuzz.trimf(_SCORE.universe, [0.05,  0.10,  0.15])
_SCORE['high']   = fuzz.trimf(_SCORE.universe, [0.12,  0.2,  0.2])

_MARG['small']   = fuzz.trimf(_MARG.universe, [0.00,  0.00,  0.03])
_MARG['medium']  = fuzz.trimf(_MARG.universe, [0.02,  0.05,  0.08])
_MARG['large']   = fuzz.trimf(_MARG.universe, [0.07,  0.1,  0.1])

_DEC['reject'] = fuzz.trimf(_DEC.universe, [0,   0,   0.4])
_DEC['maybe']  = fuzz.trimf(_DEC.universe, [0.3, 0.5, 0.7])
_DEC['accept'] = fuzz.trimf(_DEC.universe, [0.6, 1.0, 1.0])

_RULES = [
    ctrl.Rule(_SCORE['high'] & _MARG['large'],    _DEC['accept']),
    ctrl.Rule(_SCORE['high'] & _MARG['medium'],   _DEC['accept']),
    ctrl.Rule(_SCORE['high'] & _MARG['small'],    _DEC['maybe']),
    ctrl.Rule(_SCORE['medium'] & _MARG['large'],  _DEC['accept']),
    ctrl.Rule(_SCORE['medium'] & _MARG['medium'], _DEC['maybe']),
    ctrl.Rule(_SCORE['medium'] & _MARG['small'],  _DEC['reject']),
    ctrl.Rule(_SCORE['low']    & _MARG['large'],  _DEC['accept']),
    ctrl.Rule(_SCORE['low']    & _MARG['medium'], _DEC['reject']),
    ctrl.Rule(_SCORE['low']    & _MARG['small'],  _DEC['reject'])
]
_FUZZ_CTRL = ctrl.ControlSystem(_RULES)

def compute_similar_chunks_mamdani(
    chunks,
    inclusion_criteria_embeddings,
    exclusion_criteria_embeddings,
    filename,
    model,
    mu_cutoff=0.35,           # membership degree required to accept a chunk
    percentile=95,            # still used only to *plot* a reference threshold
    k=None, min_floor=0.02,   # idem
    return_thresholds=False,
    skip_reference=True
):
    """
    Returns (matched_chunks, thr_dyn) when return_thresholds=True,
    else just matched_chunks.

    Decision logic:
        1. best Δ-score, second-best Δ-score  → Mamdani controller
        2. accept if membership μ ≥ mu_cutoff
    """
    texts = [c["text"] for c in chunks]
    if not texts:
        return (None, None) if return_thresholds else None

    # ── Embeddings & Δ-scores
    chunk_emb   = get_batch_embeddings(texts, model)
    sim_incl    = cosine_similarity(chunk_emb, inclusion_criteria_embeddings)
    sim_excl    = cosine_similarity(chunk_emb, exclusion_criteria_embeddings)
    scores_mat  = sim_incl - sim_excl                                  # [n×m]

    # ── Compute *reference* dynamic thresholds (only for plotting / debugging)
    if k is None:
        thr_dyn = np.maximum(np.percentile(scores_mat, percentile, axis=0),
                             min_floor)
    else:
        mu, std = scores_mat.mean(axis=0), scores_mat.std(axis=0)
        thr_dyn = np.maximum(mu + k*std, min_floor)

    # ── Mamdani fuzzy selection
    matched_chunks = []
    best_scores  = scores_mat.max(axis=1)
    best_idx     = scores_mat.argmax(axis=1)
    second_best  = np.partition(scores_mat, -2, axis=1)[:, -2]

    for chunk, best, second, cid in zip(chunks, best_scores, second_best, best_idx):
        if skip_reference and re.match(
            r'^(references|reference[s]? section|bibliography|works cited|literature cited)(?:\s*\n|\s+[.:])?',
            chunk["text"].lower(), re.IGNORECASE
        ):
            break
        
        _FUZZ_SIM  = ctrl.ControlSystemSimulation(_FUZZ_CTRL)
        _FUZZ_SIM.input['score']  = float(best)
        _FUZZ_SIM.input['margin'] = float(best - second)
        _FUZZ_SIM.compute()
        mu = float(_FUZZ_SIM.output['decision']) if _FUZZ_SIM.output else 0.0

        if mu >= mu_cutoff:
            cpy = chunk.copy()
            cpy["criterion_id"]  = int(cid)
            cpy["membership_mu"] = mu
            matched_chunks.append(cpy)

    # ── Optional visualisation
    plot_cosine_similarity_distribution(best_scores, thr_dyn, filename)
    # plot_all_membership_functions(_SCORE, _MARG, _DEC, mu_cutoff)
    # plot_fuzzy_output_surface(_SCORE, _MARG, _FUZZ_CTRL)
    
    return (matched_chunks, thr_dyn) if return_thresholds else matched_chunks

######################################################################################

def compute_similar_chunks_adaptive(
    chunks,
    inclusion_criteria_embeddings,
    exclusion_criteria_embeddings,
    filename,
    model,
    thresholds=None,                 # fallback static vector
    percentile=95,                   # for percentile rule
    k=None,                          # set k (e.g. 1.5) to switch to mean+kσ rule
    min_floor=0.02,                  # absolute minimum threshold
    use_static=False,                # force old behaviour
    return_thresholds=False,         # handy for debugging
    skip_reference=True              # skip reference chunks
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
    for chunk, scores in zip(chunks, scores_matrix):
        if skip_reference and re.match(r'^(references|reference[s]? section|bibliography|works cited|literature cited)(?:\s*\n|\s+[.:])?', chunk["text"].lower(), re.IGNORECASE):
            break
        
        for criterion_id, (score, threshold) in enumerate(zip(scores, thr_dyn)):
            if score >= threshold:
                chunk_copy = chunk.copy()
                chunk_copy["criterion_id"] = int(criterion_id)
                matched_chunks.append(chunk_copy)

    # 5️⃣ Plot distribution
    plot_cosine_similarity_distribution(scores_matrix.max(axis=1), thr_dyn, filename)

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

