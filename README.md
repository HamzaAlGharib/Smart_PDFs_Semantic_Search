# Smart PDFs Semantic Search

**Intelligent Web-Based Search Interface for PDF Retrieval using AI**

A web application that searches a collection of PDF documents by *meaning* instead of exact wording. Upload your PDFs, ask a question in natural language, and get the most relevant passages back, each with its source document, page number, a highlighted excerpt, and a similarity score. Everything runs locally: no external API, no subscription, no internet connection, and no document content leaves your machine.

---

## Motivation

Searching inside PDFs is lexical. A PDF reader or file-search tool matches literal character strings, so a search for "interrupt handling" will not find a passage that talks about "signal handlers", even though they mean the same thing. This is the *vocabulary mismatch problem*, and it gets worse as a collection grows.

This project replaces keyword matching with **semantic retrieval**: passages and queries are converted into vectors that capture meaning, and relevance is measured by proximity in vector space.

## Features

- **Semantic search** over natural-language queries, with no keyword matching in the search pipeline
- **Multi-file upload**: index a whole collection in one request
- **Page-aware results**: every result shows its source document and page, and links directly to that page of the original PDF
- **Relevance threshold**: unrelated queries return *no results* instead of misleading ones
- **Query term highlighting** inside each excerpt, with a High / Medium relevance badge and similarity percentage
- **Duplicate protection**: documents are identified by a SHA-256 content hash, so re-uploading the same file (even renamed) never duplicates the index
- **Persistent storage**: the FAISS index, chunk metadata, and document registry are saved to disk and survive restarts
- **Admin-protected deletion**: removing a document requires an admin credential that is enforced by the server, not just hidden in the UI
- **Built-in evaluation view**: measure retrieval quality from the interface with one click
- **Fully offline**: the encoder runs from a local cache and the system makes no network calls

## How It Works

### Indexing pipeline


PDF upload -> validate & SHA-256 hash -> store original file -> page-by-page text extraction
          -> sentence-aware chunking -> all-mpnet-base-v2 embeddings (768-d)
          -> L2 normalization -> FAISS index + metadata -> persist to disk


1. **Page-aware extraction** with PyMuPDF, so each passage belongs to exactly one page.
2. **Sentence-aware chunking**: whole sentences are grouped into passages of about 500 characters with a one-sentence overlap. This replaced two earlier strategies (fixed character offsets, then word-boundary splits) that produced fragments.
3. **Embedding** with the all-mpnet-base-v2 transformer sentence encoder. The project originally used all-MiniLM-L6-v2 (384-d) and was migrated for better representation quality.
4. **Indexing** in a FAISS IndexFlatIP (exact inner-product search) over normalized vectors, which is mathematically equivalent to **cosine similarity**. An earlier Euclidean-distance index was identified as the wrong metric for this encoder and corrected.

### Search pipeline


Query -> same encoder + normalization -> top-5 nearest passages
      -> discard scores below 0.25 -> ranked results with document, page, excerpt, score


The similarity threshold of **0.25** was chosen empirically: queries about indexed material scored about 0.34 and above, while unrelated queries scored about 0.09 to 0.17, and the threshold sits in the empty gap between them.

## Tech Stack

| Component | Role |
|---|---|
| Python 3.14 | Implementation language |
| FastAPI | REST backend with automatic request validation and docs |
| PyMuPDF | Per-page PDF text extraction |
| sentence-transformers / PyTorch / transformers | Local all-mpnet-base-v2 encoder (CPU is sufficient) |
| faiss-cpu | Vector storage and exact similarity search |
| NumPy | Array handling for embeddings |
| scikit-learn | TF-IDF keyword extraction and evaluation metrics |
| Pydantic, python-multipart | Request validation and file upload parsing |
| HTML / CSS / JavaScript, Bootstrap 5 | Single-page responsive frontend (no build step) |
| hashlib, pickle | Content hashing and metadata/registry serialization |


## API Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | / | Serves the web interface |
| POST | /upload | Accepts one or more PDFs, indexes them, reports the outcome per file |
| POST | /search | Takes a natural-language query and returns ranked passages |
| GET | /documents | Lists indexed documents |
| DELETE | /documents/{filename} | Removes a document (admin credential required) |
| POST | /admin/verify | Verifies the admin credential so the UI can reveal admin controls |
| GET | /evaluate | Runs the evaluation and returns the metrics |
| GET | /pdfs/{filename} | Serves an original PDF for in-browser viewing |

Interactive API docs are available at /docs while the server is running.

## Getting Started

### Prerequisites

- Python 3 with pip
- Internet access **once**, to download the encoder model (about 420 MB). After that the system runs fully offline.

### Installation


# 1. Clone the repository
git clone https://github.com/HamzaAlGharib/Smart_PDFs_Semantic_Search.git

# 2. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Start the server
uvicorn main:app --reload


Open **http://127.0.0.1:8000** in your browser.

### Usage

1. Go to **Upload File** and select one or more PDFs.
2. Go to **Search** and type a question in plain language.
3. Click a result's document name to open the source PDF at the matching page.
4. Use **Files List** to see what is indexed. Admin mode reveals delete controls.
5. Use **Evaluation** to run the quality evaluation.

### Admin access

Document deletion is protected by a single shared credential read from an environment variable (see config.py), with a default so the project runs without configuration. **Change it before any real use.** This is proportionate to a single-user system and is not a full authentication system.

## Evaluation

Labelled ground truth is expensive, so the project uses an **automatic labelling strategy**: TF-IDF extracts the most distinctive keywords of each document (3 per document), and these become test queries whose correct answers are known in advance. A passage is labelled relevant if it contains the keyword. Every query is scored against every passage, and metrics are computed with scikit-learn. TF-IDF is used only for evaluation and plays no role in the search engine itself.

Results on a heterogeneous corpus of four PDFs (course material, reports, administrative documents):

| Metric | Value |
|---|---|
| Accuracy | 0.901 |
| Specificity | 0.984 |
| **AUC** | **0.915** |

Confusion matrix: TP = 92, FN = 97, FP = 15, TN = 924.

Because the classes are imbalanced by construction (most query/passage pairs are correct rejections), accuracy and specificity are inflated. **AUC is the principal metric**: it is threshold-independent and says the system ranks a relevant passage above an irrelevant one about 91.5% of the time.

Additional findings:

- **Keyword granularity**: AUC falls as weaker keywords are admitted (1 keyword: 0.991, 2: 0.933, 3: about 0.93, 5: 0.849, 10: 0.820). Three keywords was chosen as a balance between difficulty and sample size.
- **Threshold**: raising it rejects more irrelevant passages but discards more moderately relevant ones. It moves the operating point along the ROC curve without changing the curve itself.

## Key Design Decisions

- **Local execution** over a managed API, for privacy, zero cost, and offline use
- **Exact (flat) index** over an approximate one, since at personal-collection scale the cost is negligible and results are exact
- **No relational database**: the operation is vector similarity, which FAISS serves directly, with metadata kept alongside it in a list indexed by FAISS vector ID
- **Content-hash identification** instead of file names, making uploads idempotent
- **Server-side authorization**: the component that owns the data enforces access to it
- **Deletion by index reconstruction**: flat indexes cannot remove single vectors, so a new index is built from the retained vectors, with no re-encoding needed

## Limitations and Future Work

- **Answer generation (RAG)**: add a language model that composes written answers with citations from the retrieved passages. The retrieval component is already built for this.
- **Scanned documents**: add OCR so image-only PDFs can be indexed
- **Human relevance judgements**: complement the automatic evaluation with a human-labelled query set
- **Multi-user support**: individual accounts with hashed credentials, sessions, and per-user collections

