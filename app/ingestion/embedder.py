from langchain_google_genai import GoogleGenerativeAIEmbeddings
from app.config import settings

class Embedder:

    """
    Genera embeddings usando Google Gemini
    
    Google Embedding API:
    - Modelo: embedding-001
    - Dimensiones: 768 (vs 3072 de OpenAI)
    - Gratuito hasta 1,500 requests/día
    - Después: $0.00001 por 1K tokens
    
    ¿Por qué menos dimensiones?
    - 768 dims es suficiente para la mayoría de casos
    - Más eficiente en cómputo y almacenamiento
    - Similar rendimiento a modelos más grandes
    """

    def __init__(self):
        """
        Inicializa el embedder de Gemini
        
        GoogleGenerativeAIEmbeddings es un wrapper de LangChain que:
        - Maneja la autenticación con Google
        - Gestiona rate limits automáticamente
        - Convierte respuestas al formato estándar de LangChain
        """
        self.embedder = GoogleGenerativeAIEmbeddings(
            model=settings.GEMINI_EMBED_MODEL,
            google_api_key=settings.GOOGLE_API_KEY # type: ignore
        )

    def embed_chunks(self, chunks):
        """
        Genera embeddings para una lista de textos
        
        Proceso interno:
        1. Valida que los textos no estén vacíos
        2. Agrupa en batches (Gemini procesa hasta 100 a la vez)
        3. Envía a la API de Google
        4. Retorna vectores de 768 dimensiones
        
        Args:
            chunks: Lista de strings a vectorizar
            
        Returns:
            Lista de vectores (cada uno con 768 números flotantes)
            
        Ejemplo:
            chunks = ["Texto 1", "Texto 2"]
            vectors = embedder.embed_chunks(chunks)
            # [[0.23, -0.11, ..., 0.45], [0.12, 0.88, ..., -0.33]]
        """

        return self.embedder.embed_documents(chunks)
    
    def embed_query(self, query: str):
        """
        Embedding para una consulta individual
        
        ¿Por qué un método separado?
        - Las queries pueden tener optimizaciones diferentes
        - Algunos modelos usan prefijos especiales para queries
        - LangChain mantiene esta distinción por compatibilidad
        
        Args:
            query: Pregunta del usuario
            
        Returns:
            Vector de 768 dimensiones
            
        Ejemplo:
            query = "¿Qué es la auditoría?"
            vector = embedder.embed_query(query)
            # [0.15, -0.22, ..., 0.67]
        """

        return self.embedder.embed_query(query)
    
    def get_embedding_dimension(self) -> int:
        """
        Retorna el número de dimensiones del modelo
        
        Útil para:
        - Configurar el vector store
        - Validar vectores
        - Debugging
        """
        return 768  # Gemini embedding-001 siempre usa 768 dimensiones