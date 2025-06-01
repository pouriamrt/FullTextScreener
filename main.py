import os
from config import *
from utils.pdf_parser import extract_chunks_with_metadata
from utils.embedding import get_embedding
from utils.similarity import compute_similar_chunks
from utils.pdf_highlighter import highlight_chunks
from tqdm import tqdm
from utils.check_chunk_llm import send_to_llm


def main():
    criteria_embeddings = [get_embedding(c, OPENAI_MODEL) for c in tqdm(INCLUSION_CRITERIA)]

    for filename in tqdm(os.listdir(PDF_FOLDER)):
        if not filename.endswith(".pdf"):
            continue
        
        print(f"Processing {filename}")
        pdf_path = os.path.join(PDF_FOLDER, filename)
        
        # chunks = extract_chunks_with_metadata(pdf_path, CHUNK_SIZE, OVERLAP)
        chunks = extract_chunks_with_metadata(pdf_path, SENTENCES_PER_CHUNK)
        
        matched_chunks = compute_similar_chunks(chunks, criteria_embeddings, OPENAI_MODEL, SIMILARITY_THRESHOLD)
        print(f"Found {len(matched_chunks)} matched chunks")
        
        output_path = os.path.join(OUTPUT_FOLDER, filename.replace(".pdf", "_highlighted.pdf"))
        
        for i, chunk in enumerate(tqdm(matched_chunks)):
            label = chunk['criterion_id']
            description = INCLUSION_CRITERIA[label]
            matched_chunks[i]['llm_reason'] = send_to_llm(chunk['text'], CRITERIA_LABELS[label], description)
        
        highlight_chunks(pdf_path, matched_chunks, output_path)

if __name__ == "__main__":
    main()
