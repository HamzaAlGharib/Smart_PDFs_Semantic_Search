import os
import pickle

import faiss
import numpy as np


# all-mpnet-base-v2 produces 768-dimensional vectors (the old MiniLM used 384).
EMBEDDING_DIM = 768

# Minimum cosine similarity for a chunk to count as "relevant".
# Cosine similarity ranges from 0 (unrelated) to 1 (identical meaning).
# Unrelated queries usually score below ~0.25, so anything under this
# threshold is dropped — that's what stops irrelevant answers.
SIMILARITY_THRESHOLD = 0.25

# Where we save the index and its metadata on disk.
STORE_FOLDER = "vector_data"
INDEX_PATH = os.path.join(STORE_FOLDER, "index.faiss")
METADATA_PATH = os.path.join(STORE_FOLDER, "metadata.pkl")
DOCUMENTS_PATH = os.path.join(STORE_FOLDER, "documents.pkl")

os.makedirs(STORE_FOLDER, exist_ok=True)


# metadata is a plain Python list. Its position (0, 1, 2, ...) lines up
# exactly with the vector IDs FAISS assigns, because the index stores
# vectors in the order they are added. So metadata[i] always describes
# the i-th vector in the index. THIS is the ID <-> chunk mapping.
metadata = []


# documents is the "library" registry: one summary entry per uploaded PDF,
# e.g. {"filename", "hash", "pages", "chunks", "uploaded_at"}. The "hash"
# is the file's content fingerprint, used to detect duplicate uploads.
documents = []


def _load_or_create_index():
    """
    On startup: if a saved index exists on disk, load it (and its metadata)
    so previously uploaded PDFs are NOT lost. Otherwise, create a fresh
    empty index.
    """
    global metadata, documents

    if os.path.exists(INDEX_PATH) and os.path.exists(METADATA_PATH):
        index = faiss.read_index(INDEX_PATH)
        with open(METADATA_PATH, "rb") as f:
            metadata = pickle.load(f)
        # The documents registry may not exist in older saves; load if present.
        if os.path.exists(DOCUMENTS_PATH):
            with open(DOCUMENTS_PATH, "rb") as f:
                documents = pickle.load(f)
        return index

    # IndexFlatIP = inner product. When the vectors are normalized first
    # (see _normalize), inner product equals COSINE SIMILARITY, which is
    # the correct way to compare sentence-transformer embeddings.
    return faiss.IndexFlatIP(EMBEDDING_DIM)


# Load once when this file is first imported.
index = _load_or_create_index()


def _normalize(embeddings):
    """
    Converts embeddings to a float32 array and scales each vector to length 1.
    Normalizing is what turns inner product into cosine similarity.
    """
    vectors = np.array(embeddings, dtype="float32")
    faiss.normalize_L2(vectors)  # modifies the array in place
    return vectors


def add_embeddings(embeddings, chunks: list[str], pages: list[int], filename: str) -> int:
    """
    Adds one PDF's embeddings to the FAISS index and records matching
    metadata (chunk text, which PDF, and which page) in the same order.
    `chunks` and `pages` are parallel lists: pages[i] is the page number
    that chunks[i] came from.
    Appends only — existing vectors are never overwritten.
    Returns how many vectors were added.
    """
    vectors = _normalize(embeddings)
    index.add(vectors)

    for chunk, page in zip(chunks, pages):
        metadata.append({
            "filename": filename,
            "text": chunk,
            "page": page,
        })

    return len(chunks)


def save_index():
    """Writes the index, metadata and document registry to disk."""
    faiss.write_index(index, INDEX_PATH)
    with open(METADATA_PATH, "wb") as f:
        pickle.dump(metadata, f)
    with open(DOCUMENTS_PATH, "wb") as f:
        pickle.dump(documents, f)


def total_vectors() -> int:
    """How many vectors are currently stored in the index."""
    return index.ntotal


def score_all_chunks(query_embedding) -> list:
    """
    Returns one similarity score per chunk, aligned with metadata order
    (result[i] is the score for metadata[i]). Unlike search(), it applies
    NO threshold and returns EVERY chunk — needed for evaluation (to see
    true negatives and to compute AUC). Read-only: it does not change the
    index or affect normal search in any way.
    """
    if index.ntotal == 0:
        return []

    vector = _normalize(query_embedding)
    scores, indices = index.search(vector, index.ntotal)

    result = [0.0] * index.ntotal
    for score, idx in zip(scores[0], indices[0]):
        if idx != -1:
            result[idx] = float(score)
    return result


def document_exists(file_hash: str) -> bool:
    """True if a PDF with this exact content (same hash) is already indexed."""
    return any(doc["hash"] == file_hash for doc in documents)


def register_document(file_hash: str, filename: str, pages: int, chunks: int):
    """Adds one summary entry to the library registry after indexing a PDF."""
    from datetime import datetime

    documents.append({
        "filename": filename,
        "hash": file_hash,
        "pages": pages,
        "chunks": chunks,
        "uploaded_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
    })


def list_documents() -> list:
    """Returns the library: every indexed PDF and its summary info."""
    return documents


def delete_document(filename: str) -> bool:
    """
    Removes one PDF's data from the store. Because IndexFlatIP can't delete a
    single vector in place, we REBUILD the index from the surviving chunks:
    we ask FAISS to hand back each kept vector (reconstruct) and add it to a
    fresh index. No re-embedding needed. Returns False if the file isn't found.
    """
    global index, metadata, documents

    known = any(m["filename"] == filename for m in metadata) or \
        any(d["filename"] == filename for d in documents)
    if not known:
        return False

    new_index = faiss.IndexFlatIP(EMBEDDING_DIM)
    new_metadata = []

    for i, item in enumerate(metadata):
        if item["filename"] == filename:
            continue  # skip chunks that belong to the deleted file
        vector = index.reconstruct(i).reshape(1, -1)  # get the stored vector
        new_index.add(vector)
        new_metadata.append(item)

    index = new_index
    metadata = new_metadata
    documents = [d for d in documents if d["filename"] != filename]
    return True


def search(query_embedding, top_k: int = 5) -> list:
    """
    Finds the top_k most similar chunks to the query embedding.
    Uses the SAME metadata mapping: each result index points to metadata[i],
    giving back the chunk's text and which PDF it came from.
    Only returns chunks above SIMILARITY_THRESHOLD, so unrelated queries
    return nothing instead of forcing weak matches.
    Returns an empty list if nothing has been indexed yet.
    """
    if index.ntotal == 0:
        return []

    vector = _normalize(query_embedding)

    # With a normalized inner-product index, the returned "scores" are
    # cosine similarities. Higher = more similar.
    scores, indices = index.search(vector, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        # FAISS returns -1 when it has fewer vectors than top_k requested.
        if idx == -1:
            continue

        # Drop weak/irrelevant matches.
        if score < SIMILARITY_THRESHOLD:
            continue

        item = metadata[idx]
        results.append({
            "filename": item["filename"],
            "page": item.get("page", 1),
            "text": item["text"],
            "score": float(score),
        })

    return results
