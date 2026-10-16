import os

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel

from utils.vector_store import list_documents, delete_document, save_index
from config import ADMIN_PASSWORD

router = APIRouter()


@router.get("/documents")
async def get_documents():
    """Returns the library: all PDFs that are currently indexed."""
    docs = list_documents()
    return {"count": len(docs), "documents": docs}


class AdminPassword(BaseModel):
    password: str


@router.post("/admin/verify")
async def verify_admin(payload: AdminPassword):
    return {"ok": payload.password == ADMIN_PASSWORD}


@router.delete("/documents/{filename}")
async def remove_document(filename: str, x_admin_password: str = Header(None)):
    
    if x_admin_password != ADMIN_PASSWORD:
        raise HTTPException(status_code=403, detail="Admin password required or incorrect.")

    safe_name = os.path.basename(filename)  # prevent path escaping

    removed = delete_document(safe_name)
    if not removed:
        raise HTTPException(status_code=404, detail="Document not found.")

    # Delete the physical PDF file too, if it exists.
    file_path = os.path.join("pdfs", safe_name)
    if os.path.isfile(file_path):
        os.remove(file_path)

    # Persist the rebuilt index/registry.
    save_index()

    return {"deleted": safe_name, "documents_left": len(list_documents())}
