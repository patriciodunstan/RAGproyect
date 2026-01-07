""" VectorStore - Store  and search vectors
    Whe use Chroma for its simplicity and automatic persistence
"""
import os
from typing import List
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_google_genai  import GoogleGenerativeAIEmbeddings
from app.config import settings
from pydantic.v1 import SecretStr

class VectorStore:
    """
    Wrapper sobre Chroma para gestionar la base vectorial con Gemini
    
    Chroma es ideal para desarrollo porque:
    - Persiste automáticamente en disco
    - No requiere servidor externo
    - Rápido para datasets pequeños-medianos (hasta 1M de vectores)
    - Gratuito y open source
    
    Para producción grande podrías migrar a:
    - Pinecone (cloud, escalable)
    - Weaviate (más features)
    - Qdrant (open source, más rápido)
    """

    def __init__(self):
        """
        Inicializa el embedder de Gemini y prepara Chroma
        
        ¿Por qué crear embedder aquí?
        - Chroma necesita el mismo embedder para:
          1. Vectorizar documentos al guardar
          2. Vectorizar queries al buscar
        - Si usas embedders diferentes, la búsqueda no funcionará
        """
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=settings.GEMINI_EMBED_MODEL,
            google_api_key=SecretStr(settings.GOOGLE_API_KEY), # type: ignore
            task_type="retrieval_document",
        )

        # Path to persist vector DB
        self.persist_directory = settings.VECTORDB_DIR

        # Vector Store (upload from disk if exist)
        self.vectorstore = None


    def create_from_documents(self, documents: List[Document]) -> None:
        """
        Crea una nueva colección desde cero con los documentos
        
        Flujo interno de Chroma.from_documents():
        1. Por cada Document:
           - Extrae page_content
           - Llama a embeddings.embed_documents([texto])
           - Google retorna vector de 768 dims
        2. Genera un ID único para cada documento
        3. Guarda en SQLite: (id, vector, texto, metadata)
        4. Crea índice para búsqueda rápida
        5. Persiste todo en persist_directory
        
        ⚠️ IMPORTANTE: Esto SOBREESCRIBE la colección existente
        
        Args:
            documents: Lista de Documents ya chunkeados
        """


        print(f"Creating vector store with {len(documents)} chunks...")

        #Chroma automatically:
        # 1.- Extract the text from the documents
        # 2.- Create embeddings using self.embeddings
        # 3.- Store vectors and metadata 
        # 4.- Persist data to disk
        self.vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=self.persist_directory,
            collection_name="rag_collection"
        )

        print(f"Vector store created in {self.persist_directory}.")
        print(f"Total of chunks indexed: {len(documents)}")

    def load_existing(self) -> bool:
        """
        Carga un vectorstore existente desde disco
        
        Proceso:
        1. Verifica que exista el directorio
        2. Chroma lee SQLite con los vectores guardados
        3. Carga índice en memoria para búsquedas rápidas
        4. Conecta con el embedder (para futuras queries)
        
        Returns:
            True si se cargó correctamente, False si no existe
        """
        try: 
            if not os.path.exists(self.persist_directory):
                print("No existing vector store found.")
                return False
            
            #Load from disk
            self.vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings,
                collection_name="rag_collection"
            )

            #Verify if have data
            count = self.vectorstore._collection.count()
            if count == 0:
                print("Vector store is empty.")
                return False
            print(f"Loaded existing vector store with {count} chunks.")
            return True
        except Exception as e:
            print(f"Error loading vector store: {str(e)}")
            return False
    
    def add_documents(self, documents: List[Document]) -> None:
        """
        Agrega nuevos documentos a una colección existente
        
        Útil para:
        - Agregar documentos sin perder los existentes
        - Actualización incremental de la base
        
        Args:
            documents: Lista de Documents a agregar
        """
        if self.vectorstore is None:
            raise ValueError("Vector store is not initialized. Load or create a vector store first.")
        
        print(f" Adding {len(documents)} new chunks to vector store...")
        self.vectorstore.add_documents(documents)
        print("New chunks added successfully.")

    def similarity_search(self, query: str, k: int = 3) -> List[Document]:
        """
        Busca los k documentos más similares a la query
        
        Algoritmo de búsqueda (simplificado):
        1. query → embeddings.embed_query(query) → vector_query
        2. Para cada vector en la DB:
           - Calcular similitud: cosine_similarity(vector_query, vector_db)
        3. Ordenar por similitud (de mayor a menor)
        4. Retornar top K
        
        Similitud de coseno:
        - Rango: -1 a 1
        - 1 = idénticos
        - 0 = no relacionados
        - -1 = opuestos (raro en texto)
        
        Chroma internamente usa:
        - HNSW (Hierarchical Navigable Small World)
        - Búsqueda aproximada (no exacta, pero muy rápida)
        
        Args:
            query: Pregunta del usuario
            k: Número de resultados a retornar
            
        Returns:
            Lista de Documents más relevantes ordenados por similitud
        """
        if self.vectorstore is None:
            raise ValueError("Vector store is not initialized. Load or create a vector store first.")
        
        results = self.vectorstore.similarity_search(query, k=k)
        return results
    
    def similarity_search_with_score(self, query: str, k: int = 3) -> List[tuple]:
        """
        Busca documentos con sus scores de similitud
        
        Útil para:
        - Debugging (ver qué tan buenos son los matches)
        - Filtrado por threshold (ej: solo score < 0.5)
        - Logging y análisis
        
        Returns:
            Lista de tuplas: (Document, score)
            
        ⚠️ Score en Chroma:
        - Es DISTANCIA, no similitud
        - Menor score = MÁS similar
        - 0 = idéntico
        - >1 = poco similar
        """

        if self.vectorstore is None:
            raise ValueError("Vector store is not initialized. Load or create a vector store first.")
        results = self.vectorstore.similarity_search_with_score(query, k=k)
        return results
    def delete_collection(self) -> None:
        """
        Elimina completamente la colección
        
        Útil para:
        - Reset completo
        - Limpiar antes de re-indexar
        - Testing
        """
        if self.vectorstore:
            self.vectorstore.delete_collection()
            print("Vector store collection deleted.")
        else:
            print("Vector store is not initialized. Nothing to delete.")

