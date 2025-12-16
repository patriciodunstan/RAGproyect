from typing import List, Dict, Any
from langchain_core.documents import Document
from sympy import preview
from app.retrieval.vectorstore import VectorStore


class Retriever:
    """
    Maneja la recuperación de documentos relevantes
    
    Es una capa entre la API y el vectorstore que:
    - Simplifica la interfaz
    - Formatea resultados
    - Puede agregar lógica adicional (filtros, re-ranking, etc)
    """

    def __init__(self):
        """Inicializa el vectorstore"""
        self.vectorstore = VectorStore()

    def initialize(self) -> bool:
        """
        Intenta cargar un vectorstore existente

        Returns:
            True si existe y se cargó, False si no existe
        """

        return self.vectorstore.load_existing()
    
    def retrieve(self, query: str, top_k: int = 3) -> Dict[str, Any]:
        """
        Recupera los chunks más relevantes para una query
        
        Args:
            query: Pregunta del usuario
            top_k: Número de chunks a recuperar
            
        Returns:
            Dict con:
            - chunks: Lista de Documents
            - context: String con todo el contexto concatenado
            - sources: Lista de fuentes únicas
        """
        # Buscar documentos similares
        results = self.vectorstore.similarity_search(query, k=top_k)

        if not results:
            return {
                "chunks": [],
                "context": "",
                "sources": []
            }
        
        # Formatear contexto para el prompt
        #Cada chunk se numera y se identifica por su fuente
        context_parts = []
        source = set()

        for index, doc in enumerate(results, 1):
            source = doc.metadata.get('source', 'unknown source')
            page = doc.metadata.get('page', 'unknown page')
            chunk_id = doc.metadata.get('chunk_id', 'unknown id')

            source.add(source)

            #Formato: [Fuente - Págnina X - Chunk Y]
            context_parts.append(
                f"[{source} - Pág.{page} - Chunk {chunk_id}]\n"
                f"{doc.page_content}\n"
            )
        context = "\n---\n".join(context_parts)
        return {
            "chunks": results,
            "context": context,
            "sources": list(source)
        }
    
    def retrieve_with_scores(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Recupera chunks con sus scores de similitud
        
        Útil para debugging o para implementar threshold filtering
        
        Returns:
            Lista de dicts con:
            - document: Document
            - score: float (menor = más similar)
            - metadata: dict con info adicional
        """
        results = self.vectorstore.similarity_search_with_score(query, k=top_k)

        formatted_results = []
        for doc, score in results:
            formatted_results.append({
                "document": doc,
                "score": score,
                "metadata": doc.metadata,
                "preview": doc.page_content[:200] + "..."
            })
        return formatted_results
