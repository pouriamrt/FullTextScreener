import fitz
from collections import defaultdict
from config import CRITERIA_COLORS

def highlight_chunks(pdf_path, chunks, output_path):
    doc = fitz.open(pdf_path)

    grouped_chunks = defaultdict(list)
    for chunk in chunks:
        text = chunk["text"].strip()
        if text:
            key = (chunk["page"], text)
            grouped_chunks[key].append({
                "criterion_id": chunk["criterion_id"],
                "llm_reason": chunk["llm_reason"]
            })

    annot_types = ["add_highlight_annot", "add_underline_annot", "add_squiggly_annot", "add_strikeout_annot"]

    for (page_num, text), matches in grouped_chunks.items():
        page = doc[page_num]
        
        try:
            text_instances = page.search_for(text)
            if not text_instances:
                normalized_text = " ".join(text.split())
                if normalized_text:
                    text_instances = page.search_for(normalized_text)

            for inst in text_instances:
                for i, match in enumerate(matches):
                    annot_func_name = annot_types[i % len(annot_types)]
                    annot_func = getattr(page, annot_func_name)
                    
                    annot = annot_func(inst)
                    
                    color = CRITERIA_COLORS.get(match["criterion_id"], (1, 1, 0))
                    annot.set_colors(stroke=color)
                    annot.set_info(content=match["llm_reason"])
                    annot.update()
        except Exception as e:
            print(f"Highlight failed on page {page_num} for text '{text[:50]}...': {e}")

    doc.save(output_path, garbage=4, deflate=True)
