import fitz
import os
from config import CRITERIA_COLORS

def highlight_chunks(pdf_path, chunks, output_path):
    doc = fitz.open(pdf_path)

    for chunk in chunks:
        page = doc[chunk["page"]]
        text = chunk["text"]
        color = CRITERIA_COLORS.get(chunk["criterion_id"], (1, 1, 0))  # default yellow

        try:
            for inst in page.search_for(text.strip()):
                annot = page.add_highlight_annot(inst)
                annot.set_colors(stroke=color)
                annot.update()
        except Exception as e:
            print(f"Highlight failed on page {chunk['page']}: {e}")

    doc.save(output_path, garbage=4, deflate=True)
