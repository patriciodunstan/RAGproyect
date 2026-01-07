import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    # ===== Google Gemini =====
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-exp")
    GEMINI_EMBED_MODEL: str = os.getenv("GEMINI_EMBED_MODEL", "models/embedding-001")


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

if not settings.GOOGLE_API_KEY:
    raise ValueError("La variable de entorno GOOGLE_API_KEY no está configurada.")
if not settings.GEMINI_MODEL:
    raise ValueError("La variable de entorno GEMINI_MODEL no está configurada.")