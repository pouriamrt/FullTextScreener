import fitz
from tqdm import tqdm
import nltk
from nltk.tokenize import sent_tokenize

def extract_chunks_with_metadata(pdf_path, sentences_per_chunk=3):
    doc = fitz.open(pdf_path)
    all_chunks = []

    for page_num, page in enumerate(doc):
        text = page.get_text("text")
        sentences = sent_tokenize(text)

        for i in range(0, len(sentences) - sentences_per_chunk + 1):
            chunk_sents = sentences[i:i + sentences_per_chunk]
            chunk_text = " ".join(s.strip() for s in chunk_sents).strip()

            if len(chunk_text.split()) < 10:  # skip very short ones
                continue

            all_chunks.append({
                "text": chunk_text,
                "page": int(page_num)
            })

    return all_chunks

# def extract_chunks_with_metadata(pdf_path, chunk_size=120, overlap=10):
#     doc = fitz.open(pdf_path)
#     all_chunks = []

#     for page_num, page in enumerate(tqdm(doc)):
#         words = page.get_text("words")
#         words.sort(key=lambda w: (w[1], w[0]))  # top-to-bottom, left-to-right
#         tokens = [w[4] for w in words]

#         step = chunk_size - overlap
#         for i in range(0, len(tokens) - chunk_size + 1, step):
#             chunk_text = " ".join(tokens[i:i + chunk_size])
#             if len(chunk_text.strip()) < 10:
#                 continue
#             all_chunks.append({
#                 "text": chunk_text,
#                 "page": page_num
#             })

#     return all_chunks
