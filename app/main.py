from fastapi import FastAPI
from app.api.ingest import router as ingest_router
from app.api.query import router as query_router


app = FastAPI(title="RAG Intro Project")

app.include_router(ingest_router, prefix="/ingest", tags=["Ingestion"])
app.include_router(query_router, prefix="/query", tags=["Query"])

@app.get("/")
def root():
    return {"message": "Welcome to the RAG Intro Project API!"}
