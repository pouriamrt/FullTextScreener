import os
from config import *
from utils.pdf_parser import extract_chunks_with_metadata
from utils.embedding import get_embedding
from utils.similarity import compute_similar_chunks
from utils.pdf_highlighter import highlight_chunks
from tqdm import tqdm
from utils.check_chunk_llm import send_to_llm
from time import time

def main(overwrite=False):
    criteria_embeddings = [get_embedding(c, OPENAI_MODEL) for c in tqdm(INCLUSION_CRITERIA)]

    for filename in tqdm(os.listdir(PDF_FOLDER)):
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
        
        matched_chunks = compute_similar_chunks(chunks, criteria_embeddings, OPENAI_MODEL, SIMILARITY_THRESHOLDS)
        if matched_chunks is None:
            print(f"No matched chunks found for {filename[:100]}")
            continue
        
        print(f"Found {len(matched_chunks)} matched chunks")
        
        for i, chunk in enumerate(tqdm(matched_chunks)):
            label = chunk['criterion_id']
            description = INCLUSION_CRITERIA[label]
            matched_chunks[i]['llm_reason'] = send_to_llm(chunk['text'], CRITERIA_LABELS[label], description, LLM_MODEL)
        
        highlight_chunks(pdf_path, matched_chunks, output_path)

if __name__ == "__main__":
    start_time = time()
    main(overwrite=OVERWRITE)
    end_time = time()
    total_seconds = end_time - start_time
    print(f"\nTotal runtime: {total_seconds:.2f} seconds")
