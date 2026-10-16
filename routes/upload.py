import os
import hashlib

from fastapi import APIRouter, File, UploadFile

from utils.pdf_reader import extract_pages
from utils.chunker import chunk_text
from utils.embeddings import generate_embeddings
from utils.vector_store import (
    add_embeddings,
    save_index,
    total_vectors,
    document_exists,
    register_document,
)

router = APIRouter()

UPLOAD_FOLDER = "pdfs"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  


@router.post("/upload")
async def upload_pdf(files: list[UploadFile] = File(...)):  # ... means required
    results = []

    for file in files:
        if not file.filename.endswith(".pdf"):
            results.append({
                "filename": file.filename,
                "status": "skipped",
                "message": "Not a PDF file.",
            })
            continue

        
        content = await file.read()
        file_hash = hashlib.sha256(content).hexdigest()

        
        if document_exists(file_hash):
            results.append({
                "filename": file.filename,
                "status": "duplicate",
                "message": "Already indexed — skipped.",
            })
            continue

        # Save the file to disk.
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        with open(file_path, "wb") as buffer:
            buffer.write(content)

        # Read page-by-page so each chunk remembers its page number.
        pages = extract_pages(file_path)
        chunks = []
        chunk_pages = []
        for page in pages:
            for chunk in chunk_text(page["text"]):
                chunks.append(chunk)
                chunk_pages.append(page["page"])

        # Generate embeddings and add them to the FAISS index with metadata.
        chunks_added = 0
        if chunks:
            embeddings = generate_embeddings(chunks)
            chunks_added = add_embeddings(embeddings, chunks, chunk_pages, file.filename)

        # Record this PDF in the library registry.
        register_document(file_hash, file.filename, len(pages), chunks_added)

        results.append({
            "filename": file.filename,
            "status": "indexed",
            "pages": len(pages),
            "chunks_added": chunks_added,
        })

    # Save index, metadata and registry once, after all files are processed.
    save_index()

    return {
        "uploaded": len(results),
        "total_chunks_in_store": total_vectors(),
        "files": results,
    }
