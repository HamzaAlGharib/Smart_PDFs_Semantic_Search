import os

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from routes.upload import router as upload_router
from routes.search import router as search_router
from routes.documents import router as documents_router
from routes.evaluate import router as evaluate_router

app = FastAPI(title="PDF Search Engine")

app.include_router(upload_router)
app.include_router(search_router)
app.include_router(documents_router)
app.include_router(evaluate_router)

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def home():
    return FileResponse("static/index.html")


@app.get("/pdfs/{filename}")
def get_pdf(filename: str):
    
    safe_name = os.path.basename(filename)
    file_path = os.path.join("pdfs", safe_name)

    if not os.path.isfile(file_path):
        raise HTTPException(status_code=404, detail="PDF not found.")

    return FileResponse(
        file_path,
        media_type="application/pdf",
        filename=safe_name,
        content_disposition_type="inline",
    )
