import fitz
from tqdm import tqdm

def extract_chunks_with_metadata(pdf_path, chunk_size=120, overlap=10):
    doc = fitz.open(pdf_path)
    all_chunks = []

    for page_num, page in enumerate(tqdm(doc)):
        words = page.get_text("words")
        words.sort(key=lambda w: (w[1], w[0]))  # top-to-bottom, left-to-right
        tokens = [w[4] for w in words]

        step = chunk_size - overlap
        for i in range(0, len(tokens) - chunk_size + 1, step):
            chunk_text = " ".join(tokens[i:i + chunk_size])
            if len(chunk_text.strip()) < 10:
                continue
            all_chunks.append({
                "text": chunk_text,
                "page": page_num
            })

    return all_chunks
