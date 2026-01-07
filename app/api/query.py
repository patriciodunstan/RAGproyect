"""
Query API - Endpoint para consultar el RAG
"""
from fastapi import APIRouter, HTTPException, Depends
from langchain_google_genai  import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import SecretStr

from app.config import settings
from app.retrieval.retriever import Retriever
from app.dependencies import get_retriever
from models.schemas import QueryRequest, QueryResponse, SourceInfo

router = APIRouter()


# === Prompt Template para RAG ===
RAG_PROMPT_TEMPLATE = """Eres un asistente experto que responde preguntas basándose ÚNICAMENTE en el contexto proporcionado.

CONTEXTO:
{context}

INSTRUCCIONES:
- Responde SOLO con información del contexto
- Si el contexto no contiene la respuesta, di "No tengo información suficiente para responder esa pregunta"
- Sé preciso y conciso
- Cita las fuentes cuando sea posible (menciona el documento)

PREGUNTA: {question}

RESPUESTA:"""


@router.post("/ask", response_model=QueryResponse)
async def query_rag(
    request: QueryRequest,
    retriever: Retriever = Depends(get_retriever)
):
    """
    Endpoint para hacer consultas al RAG
    
    Flujo completo:
    1. Recuperar contexto relevante (similarity search)
    2. Construir prompt con contexto + pregunta
    3. Enviar a Azure OpenAI
    4. Retornar respuesta + fuentes
    
    Args:
        request: QueryRequest con question y top_k
        
    Returns:
        QueryResponse con answer, sources, chunks_used
    """
    
    # === PASO 1: Recuperar contexto relevante ===
    print(f"\n🔍 Buscando contexto para: '{request.question}'")
    
    try:
        retrieval_results = retriever.retrieve(
            query=request.question,
            top_k=request.top_k if request.top_k is not None else 5  # Usa 5 o el valor por defecto que prefieras
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error recuperando contexto: {str(e)}"
        )
    
    # Validar que haya contexto
    if not retrieval_results["context"]:
        raise HTTPException(
            status_code=404,
            detail="No se encontró información relevante. ¿Ya ejecutaste /ingest?"
        )
    
    context = retrieval_results["context"]
    chunks = retrieval_results["chunks"]
    sources = retrieval_results["sources"]
    
    print(f"✅ Encontrados {len(chunks)} chunks de {len(sources)} fuente(s)")
    
    # === PASO 2: Construir prompt ===
    prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)
    formatted_prompt = prompt.format(
        context=context,
        question=request.question
    )
    
    # === PASO 3: Consultar Azure OpenAI ===
    print("🤖 Generando respuesta con Azure OpenAI...")
    
    try:
        # Inicializar el LLM
        llm = ChatGoogleGenerativeAI(  # type: ignore[call-arg]
            model=settings.GEMINI_MODEL,
            google_api_key=settings.GOOGLE_API_KEY,
            temperature=0.3,  # Baja para respuestas determinísticas
            convert_system_message_to_human=True  # Gemini no soporta system messages nativamente
        )
        
        # Generar respuesta
        response = llm.invoke(formatted_prompt)
        answer = response.content
        if not isinstance(answer, str):
            answer = str(answer)
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generando respuesta: {str(e)}"
        )
    
    # === PASO 4: Formatear respuesta con fuentes ===
    source_infos = []
    for chunk in chunks:
        source_infos.append(SourceInfo(
            filename=chunk.metadata.get('source', 'desconocido'),
            chunk_id=chunk.metadata.get('chunk_id', 0),
            content_preview=chunk.page_content[:150] + "..."
        ))
    
    print("✅ Respuesta generada exitosamente\n")
    
    return QueryResponse(
        answer=answer,
        source=source_infos,
        chunk_used=len(chunks)
    )


@router.post("/debug", response_model=dict)
async def debug_retrieval(
    request: QueryRequest,
    retriever: Retriever = Depends(get_retriever)
):
    """
    Endpoint de debugging para ver qué chunks se recuperan
    
    Útil para entender por qué el RAG responde de cierta manera
    o para ajustar parámetros de chunking
    """
    try:
        results = retriever.retrieve_with_scores(
            query=request.question,
            top_k=request.top_k if request.top_k is not None else 5  # Usa 5 o el valor por defecto que prefieras
        )
        
        return {
            "query": request.question,
            "top_k": request.top_k,
            "results": [
                {
                    "score": r["score"],
                    "source": r["metadata"].get("source"),
                    "chunk_id": r["metadata"].get("chunk_id"),
                    "preview": r["preview"]
                }
                for r in results
            ]
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error en debug: {str(e)}"
        )