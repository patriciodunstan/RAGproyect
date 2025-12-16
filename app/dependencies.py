from functools import lru_cache
from app.retrieval.retriever import Retriever
from app.ingestion.loader import DocumentLoader
from app.ingestion.splitter import TextSplitter
from app.ingestion.embedder import Embedder
from app.retrieval.vectorstore import VectorStore

# ===== Ingestion Dependencies =====

@lru_cache()
def get_document_loader() -> DocumentLoader:
    """
    Retorna instancia singleton de DocumentLoader

    @lru_cache hace que se cree solo una vez y se reutilice en todas las llamadas
    """

    return DocumentLoader()

@lru_cache()
def get_text_splitter() -> TextSplitter:
    """
    Retorna instancia singleton de TextSplitter
    """

    return TextSplitter()

@lru_cache()
def get_embedder() -> Embedder:
    """
    Retorna instancia singleton de Embedder
    """

    return Embedder()

@lru_cache()
def get_vectorstore() -> VectorStore:
    """
    Retorna instancia singleton de VectorStore
    """

    return VectorStore()

# ===== Retrieval Dependencies =====

@lru_cache()
def get_retriever() -> Retriever:
    """
    Retorna instancia singleton de Retriever

    Esta es la dependencia principal para consultas
    """

    retriever = Retriever()

    # Intenta cargar vectorstore existente
    if not retriever.initialize():
        print(" Advertencia: No hay vectorestrore. Ejecuta /ingest primero")

    return retriever