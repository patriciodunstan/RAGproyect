from langchain_openai import AzureOpenAIEmbeddings
from app.config import settings
from pydantic import SecretStr

class Embedder:

    def __init__(self):
        self.embedder = AzureOpenAIEmbeddings(
            azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
            api_key=SecretStr(settings.AZURE_OPENAI_KEY),
            model=settings.AZURE_OPENAI_EMBED_MODEL
        )

    def embed_chunks(self, chunks):
        """Genera embeddings para una lista de textos"""
        return self.embedder.embed_documents(chunks)
    
    def embed_query(self, query: str):
        """Embedding para consultas"""
        return self.embedder.embed_query(query)
    
    