# 🧠 NLP Pipeline for Full-Text Screening Using LLMs and Embeddings

This repository implements a comprehensive pipeline to **automate the full-text screening** of scientific literature, particularly research papers in PDF format. It combines **OpenAI embeddings**, **LLM-based validation (GPT-4.1-mini)**, and optional **BioBERT fine-tuning** to assess whether text segments satisfy **user-defined inclusion criteria**. The output includes **annotated PDFs** with highlighted sections and justifications for each inclusion.

---

## 📂 Repository Structure

```
.
├── main.py                       # Entry point for the entire pipeline
├── config.py                     # Inclusion criteria, colors, thresholds, API keys
├── models/
│   └── biobert_trainer.py        # Optional: Fine-tune BioBERT on labeled data
├── utils/
│   ├── check_chunk_llm.py        # LLM-based verification of semantic matches
│   ├── embedding.py              # OpenAI embedding generator with caching
│   ├── pdf_highlighter.py        # Applies highlights and tooltips to matched PDF text
│   ├── pdf_parser.py             # Extracts sentence-based text chunks from PDFs
│   └── similarity.py             # Computes cosine similarity and filters matches
├── data/
│   ├── papers/                   # Input folder: drop PDFs to be screened here
│   └── output/                   # Output folder: highlighted and annotated PDFs
└── .gitignore
```

---

## 🧠 What This Pipeline Does

### Problem
In systematic reviews or screening tasks, full-text PDF review is labor-intensive and subjective. This project automates the screening pipeline with semantic and logical checks using modern NLP tools.

### Solution Workflow
1. 📄 Input PDFs are chunked into overlapping sentences.
2. 🔢 Each chunk is embedded using OpenAI’s `text-embedding-3-large`.
3. 🧠 The chunk is semantically compared to pre-embedded inclusion criteria.
4. ✅ Chunks with high similarity are verified using GPT-4.1-mini for human-like judgment.
5. 🖍️ Validated chunks are annotated in the PDF with highlights and reasoning tooltips.

---

## 🏗️ Pipeline Architecture

### 🧭 Overview

1. 📄 **Input PDFs** placed in `data/papers/`
2. 🧩 **Sentence-Based Chunking** (sliding windows with overlap)
3. 🔢 **Embedding Generation** using `text-embedding-3-large`
4. 📐 **Cosine Similarity Scoring** vs. inclusion criteria
5. 🤖 **LLM Verification** using `GPT-4.1-mini`
6. 🧠 **Store Validated Chunks** with LLM-generated reasoning
7. 🖍️ **Highlight Matched Text** in original PDF
8. 📂 **Save Annotated PDFs** to `data/output/`

### 🗺️ Mermaid Diagram

```mermaid
flowchart TD
    A[Input PDFs (data/papers)] --> B[Sentence-Based Chunking]
    B --> C[Embedding Generation (text-embedding-3-large)]
    C --> D[Cosine Similarity Scoring vs. Inclusion Criteria]
    D -->|Above Threshold| E[LLM Verification (GPT-4.1-mini)]
    E --> F[Store Validated Chunks with Reasoning]
    F --> G[Highlight Matches in PDF]
    G --> H[Output PDFs (data/output)]
```

---

## 🔍 Inclusion Criteria Explained

All inclusion criteria are defined in `config.py`. Each one contains:
- A **descriptive paragraph** used for semantic matching.
- A **label** (e.g., "Population", "Intervention").
- A **color code** used for highlighting.

These criteria are first embedded using OpenAI and then used to compare against paper chunks.

---

## 🧪 How Matching Works (Deep Dive)

1. **Chunking**:
   - PDFs are read page-by-page using `PyMuPDF`.
   - Each page is split into sliding windows of 3–4 sentences with overlaps to ensure context continuity.
   - Each chunk is associated with its originating page number.

2. **Embedding**:
   - Both inclusion criteria and chunks are embedded using `text-embedding-3-large`.
   - Each chunk becomes a dense vector in semantic space.

3. **Similarity Scoring**:
   - Cosine similarity is calculated between each chunk vector and every criterion vector.
   - The chunk is assigned the criterion with the highest similarity.
   - Only chunks with similarity above a threshold (e.g., 0.5) are retained.

4. **LLM Verification**:
   - GPT-4.1-mini is queried via LangChain with:
     - The chunk’s text
     - The matched criterion label
     - The full inclusion criterion description
   - The model answers:
     - YES or NO
     - An explanation (1–2 sentences)

5. **Annotation**:
   - Matched and verified chunks are saved with:
     - Criterion label and ID
     - Page number
     - GPT explanation
   - The original PDF is re-opened, and the matched text is highlighted.
   - Hover annotations in the PDF show the model’s reasoning.

6. **Output**:
   - Annotated PDFs are stored in `data/output/` for human review and traceability.

---

## ⚙️ Configuration

You can modify the following in `config.py`:
- `INCLUSION_CRITERIA`: Paragraph descriptions of each criterion.
- `SIMILARITY_THRESHOLD`: Filter threshold for cosine similarity.
- `SENTENCES_PER_CHUNK`: Controls chunk granularity.
- `CRITERIA_COLORS`: Colors for each criterion’s highlight.
- `LLM_MODEL`, `OPENAI_MODEL`: Choose models to use.

---

## 🚀 Getting Started

### Step 1: Install Requirements

```bash
pip install -r requirements.txt
```

### Step 2: Set Up API Key

Create a `.env` file with:

```
OPENAI_API_KEY=your-key-here
```

### Step 3: Add PDFs

Drop PDFs into the input folder:

```
data/papers/
```

### Step 4: Run the Pipeline

```bash
python main.py
```

### Step 5: Review Annotated PDFs

Output files with highlights and LLM comments will be saved in:

```
data/output/
```

---

## 🔬 Optional: Train BioBERT on Labeled Data

If you have labeled inclusion/exclusion data:

```python
from models.biobert_trainer import train_biobert

train_biobert([
    {"text": "This is an NCD simulation model using burden-of-disease.", "label": 3},
    ...
])
```

This uses HuggingFace’s Trainer API for supervised fine-tuning.

---

## 💡 Use Cases

- Systematic review support
- Automated inclusion/exclusion filtering
- Transparent evidence triage in health modeling
- Semantic filtering in scientific NLP pipelines

---

## 📜 License

This project is released under the MIT License.

---

## 🙏 Acknowledgments

- [OpenAI](https://openai.com/)
- [LangChain](https://www.langchain.com/)
- [HuggingFace](https://huggingface.co/)
- [PyMuPDF](https://pymupdf.readthedocs.io/)