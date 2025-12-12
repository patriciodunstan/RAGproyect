import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    # == Azure OpenAI ==
    AZURE_OPENAI_ENDPOINT: str = os.getenv("AZURE_OPENAI_ENDPOINT")
    AZURE_OPENAI_KEY: str = os.getenv("AZURE_OPENAI_KEY")
    AZURE_OPENAI_EMBED_MODEL: str = "text-embedding-3-large"


    #=== Directorios ===
    DATA_DIR: str = os.path.join(os.getcwd(), "data")
    VECTORDB_DIR: str = os.path.join(os.getcwd(), "vector_db")


    #=== Parámetros de  chunking ===
    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 150


settings = Settings()