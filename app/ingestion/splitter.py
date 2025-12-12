from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.config import settings

class TextSplitter:
    @staticmethod
    def split_text(text: str):
        """Divide el texto en chunks usando parámetros definidos en la configuración."""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = splitter.split_text(text)
        return chunks

