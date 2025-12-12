import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    # == Azure OpenAI ==
    AZURE_OPENAI_ENDPOINT: str = os.getenv("AZURE_OPENAI_ENDPOINT", "")
    AZURE_OPENAI_KEY: str = os.getenv("AZURE_OPENAI_KEY", "")
    AZURE_OPENAI_EMBED_MODEL: str = "text-embedding-3-large"


    #=== Directorios ===
    BASE_DIR: str = os.getcwd()
    DATA_DIR: str = os.path.join(BASE_DIR, "data")
    VECTORDB_DIR: str = os.path.join(BASE_DIR, "vector_db")


    #=== Parámetros de  chunking ===
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", 800))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", 150))


settings = Settings()

os.makedirs(settings.DATA_DIR, exist_ok=True)
os.makedirs(settings.VECTORDB_DIR, exist_ok=True)