import os
from config import *
from utils.pdf_parser import extract_chunks_with_metadata
from utils.embedding import get_embedding
from utils.similarity import compute_similar_chunks, compute_similar_chunks_adaptive, compute_similar_chunks_mamdani
from utils.pdf_highlighter import highlight_chunks
from tqdm import tqdm
from utils.check_chunk_llm import send_to_llm, send_to_llm_batch
from utils.cost_tracker import APICostTracker
from time import time

def main(overwrite=False):
    cost_tracker = APICostTracker(model=LLM_MODEL)
    
    inclusion_criteria_embeddings = [get_embedding(c, OPENAI_MODEL) for c in tqdm(INCLUSION_CRITERIA)]
    exclusion_criteria_embeddings = [get_embedding(c, OPENAI_MODEL) for c in tqdm(EXCLUSION_CRITERIA)]

    for filename in tqdm(os.listdir(PDF_FOLDER)[:4]):
        if not filename.endswith(".pdf"):
            continue
        
        output_path = os.path.join(OUTPUT_FOLDER, filename.replace(".pdf", "_highlighted.pdf"))
        
        if os.path.exists(output_path) and not overwrite:
            print(f"Skipping {filename[:100]}... - output already exists")
            continue
            
        print(f"Processing {filename[:100]}...")
        pdf_path = os.path.join(PDF_FOLDER, filename)
        
        # chunks = extract_chunks_with_metadata(pdf_path, CHUNK_SIZE, OVERLAP)
        chunks = extract_chunks_with_metadata(pdf_path, SENTENCES_PER_CHUNK, SENTENCES_OVERLAP)
        
        # matched_chunks = compute_similar_chunks(chunks, inclusion_criteria_embeddings, exclusion_criteria_embeddings, filename, OPENAI_MODEL, SIMILARITY_THRESHOLDS)
        # matched_chunks = compute_similar_chunks_adaptive(chunks, inclusion_criteria_embeddings, exclusion_criteria_embeddings, 
        #                                                  filename, OPENAI_MODEL, SIMILARITY_THRESHOLDS)
        matched_chunks = compute_similar_chunks_mamdani(chunks, inclusion_criteria_embeddings, exclusion_criteria_embeddings, 
                                                        filename, OPENAI_MODEL, MU_CUTOFF)
        
        if matched_chunks is None:
            print(f"No matched chunks found for {filename[:100]}")
            continue
        
        print(f"Found {len(matched_chunks)} matched chunks")
        
        # for i, chunk in enumerate(tqdm(matched_chunks)):
        #     label = chunk['criterion_id']
        #     description = 'INCLUSION CRITERIA: \n' + INCLUSION_CRITERIA[label] + "\n\n" + 'EXCLUSION CRITERIA: \n' + EXCLUSION_CRITERIA[label]
        #     matched_chunks[i]['llm_reason'], usage = send_to_llm(chunk['text'], CRITERIA_LABELS[label], description, LLM_MODEL)
        #     cost_tracker.update_from_usage(usage)
        
        responses, usages = send_to_llm_batch(matched_chunks, CRITERIA_LABELS, INCLUSION_CRITERIA, EXCLUSION_CRITERIA, LLM_MODEL)
        for i, r in enumerate(responses):
            matched_chunks[i]['llm_reason'] = r.content
            cost_tracker.update_from_usage(usages[i])
        
        
        highlight_chunks(pdf_path, matched_chunks, output_path)
        
    cost_tracker.report()

if __name__ == "__main__":
    start_time = time()
    main(overwrite=OVERWRITE)
    end_time = time()
    total_seconds = end_time - start_time
    print(f"\nTotal runtime: {total_seconds:.2f} seconds")
