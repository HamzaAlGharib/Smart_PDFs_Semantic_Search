import fitz  # PyMuPDF


def extract_text(file_path: str) -> str:
    """Opens a PDF and returns all of its text as a single string."""
    doc = fitz.open(file_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    doc.close()
    return full_text


def extract_pages(file_path: str) -> list[dict]:
    """
    Opens a PDF and returns a list of pages, each as:
        {"page": 1, "text": "...text of that page..."}
    Keeping text per-page lets us remember WHICH page each chunk came from,
    so search results can point back to the exact page.
    """
    doc = fitz.open(file_path)
    pages = []
    for page_number, page in enumerate(doc, start=1):  # pages start at 1
        pages.append({
            "page": page_number,
            "text": page.get_text(),
        })
    doc.close()
    return pages
