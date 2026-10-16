from fastapi import APIRouter
from pydantic import BaseModel

from utils.embeddings import generate_embeddings
from utils.vector_store import search, total_vectors, SIMILARITY_THRESHOLD

router = APIRouter()


class SearchQuery(BaseModel):
    query: str


@router.post("/search")
async def search_pdfs(payload: SearchQuery):
   
    if total_vectors() == 0:
        return {
            "message": "No PDFs have been uploaded yet. Please upload PDFs first.",
            "results": [],
        }

    query_embedding = generate_embeddings([payload.query])

    
    results = search(query_embedding, top_k=5)

   
    if not results:
        return {
            "query": payload.query,
            "message": "No relevant results found for this query.",
            "results": [],
        }
    

    return {
        "query": payload.query,
        "results": results,
        "stats": {
            "indexed_chunks": total_vectors(),   
            "returned": len(results),            
            "threshold": SIMILARITY_THRESHOLD,   
        },
    }
