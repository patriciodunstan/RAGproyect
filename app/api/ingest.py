from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from typing import List
from pathlib import Path
import shutil

from app.config import settings
from app.ingestion.loader import DocumentLoader
from app.ingestion.splitter import TextSplitter
from app.retrieval.vectorstore import VectorStore
from app.dependencies import get_document_loader, get_text_splitter, get_vectorstore
from models.schemas import IngestResponse

router = APIRouter()


@router.post("/upload", response_model=IngestResponse)
async def ingest_documents(
    files: List[UploadFile] = File(...),
    loader: DocumentLoader = Depends(get_document_loader),
    splitter: TextSplitter = Depends(get_text_splitter),
    vectorstore: VectorStore = Depends(get_vectorstore)
):
    """
    Endpoint para subir documentos y procesarlos
    
    Flujo completo:
    1. Guardar archivos en disco (data/)
    2. Cargar documentos (PDF/TXT/DOCX → Documents)
    3. Dividir en chunks (Documents → más Documents)
    4. Vectorizar y guardar en Chroma
    
    Args:
        files: Lista de archivos subidos (multipart/form-data)
        
    Returns:
        IngestResponse con stats del procesamiento
    """

    if not files:
        raise HTTPException(status_code=400, detail="No se enviaron archivos")
    
    # === PASO 1: Guardar archivos en data/ ===
    data_dir = Path(settings.DATA_DIR)
    saved_files = []

    for file in files:
        # Validar extensión
        file_ext = Path(file.filename).suffix.lower() # pyright: ignore[reportArgumentType]
        if file_ext not in ['.pdf', '.txt', '.docx']:
            print(f"⚠️  Ignorando {file.filename} (formato no soportado)")
            continue
        
        # Guardar archivo
        file_path = data_dir / file.filename # pyright: ignore[reportOperatorIssue]
        try:
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            saved_files.append(file_path)
            print(f"💾 Guardado: {file.filename}")
        except Exception as e:
            print(f"❌ Error guardando {file.filename}: {str(e)}")
            continue
    
    if not saved_files:
        raise HTTPException(
            status_code=400, 
            detail="No se pudo guardar ningún archivo válido"
        )
    
    # === PASO 2: Cargar documentos ===
    print(" Cargando documentos...")
    all_documents = []

    for file_path in saved_files:
        try:
            docs = loader.load_file(file_path)
            all_documents.extend(docs)
        except Exception as e:
            print(f" Error cargando {file_path.name}: {str(e)}")
            continue

    if not all_documents:
        raise HTTPException(
            status_code=500,
            detail="No se pudo procesar ningún documento"
        )
    

    # === PASO 3: Dividir en chunks ===
    print(" Dividiendo en chunks...")
    all_chunks = []

    for doc in all_documents:
        #Dividir el texto
        text_chunks = splitter.split_text(doc.page_content)

        #Crear Documents para cada chunk (preservando metadata)
        for index, chunk_text in enumerate(text_chunks):
            from langchain_core.documents import Document
            chunk_doc = Document(
                page_content=chunk_text,
                metadata={
                    **doc.metadata,
                    'chuink_id': index,
                    'total_chunks': len(text_chunks)
                }
            )
            all_chunks.append(chunk_doc)
    print(f"Total de chunks: {len(all_chunks)}")

    # === PASO 4: Vectorizar y guardar en Chroma ===
    print("Vectorizando y guardando en Chroma...")
    try:
        vectorstore.create_from_documents(all_chunks)
        print("Vectorstore creado exitosamente.")
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error creando vectorstore: {str(e)}"
        )
    
    # === Respuesta ===
    return IngestResponse(
        message="Documentos procesados exitosamente",
        files_proccessed=len(saved_files),
        chunks_created=len(all_chunks)
    )


@router.delete("/clear")
async def clear_data():
    """
    Endpoint para limpiar todos los datos
    
    Elimina:
    - Archivos en data/
    - Base vectorial en vector_db/
    """
    try:
        #Limpia data/
        data_dir = Path(settings.DATA_DIR)
        if data_dir.exists():
            for item in data_dir.iterdir():
                if item.is_file():
                    item.unlink()
        #Limpiar vector_db/
        vector_db_dir = Path(settings.VECTORDB_DIR)
        if vector_db_dir.exists():
            shutil.rmtree(vector_db_dir)
            vector_db_dir.mkdir()
        return {"message": "Datos limpiados exitosamente"}
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error limpiando datos: {str(e)}"
        )

