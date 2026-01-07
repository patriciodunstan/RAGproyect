# 🤖 AGENTS.md - Guía para Agentes de IA

Este archivo proporciona contexto y directrices para agentes de IA (GitHub Copilot, Cursor, Claude, etc.) que trabajen en este proyecto.

## 📋 Descripción del Proyecto

**RAG Project** es un sistema de Retrieval Augmented Generation que:
- Ingesta documentos (PDF, TXT, DOCX)
- Los divide en chunks y genera embeddings
- Almacena vectores en ChromaDB
- Responde preguntas usando contexto recuperado + Google Gemini

## 🏗️ Arquitectura

```
app/
├── main.py              # FastAPI entry point
├── config.py            # Settings con variables de entorno
├── dependencies.py      # Inyección de dependencias (singletons)
├── api/
│   ├── ingest.py        # POST /ingest/upload
│   └── query.py         # POST /query/ask
├── ingestion/
│   ├── loader.py        # DocumentLoader (PDF, TXT, DOCX)
│   ├── splitter.py      # TextSplitter (RecursiveCharacterTextSplitter)
│   └── embedder.py      # Embedder (GoogleGenerativeAIEmbeddings)
└── retrieval/
    ├── vectorstore.py   # VectorStore (Chroma wrapper)
    └── retriever.py     # Retriever (búsqueda + formateo)
```

## 🔧 Stack Tecnológico

| Componente | Versión | Notas |
|------------|---------|-------|
| Python | 3.12 | Requerido |
| FastAPI | 0.110.0 | Framework web |
| LangChain | >=1.2.0 | Orquestación LLM |
| langchain-google-genai | >=4.1.0 | Integración Gemini |
| ChromaDB | 0.4.22 | Vector store local |
| Pydantic | >=2.7.0 | Validación de datos |

## ⚠️ Consideraciones Importantes

### Compatibilidad de Versiones

1. **NO usar `SecretStr`** para `google_api_key` en las nuevas versiones de langchain-google-genai. Pasar string directamente:
   ```python
   # ✅ Correcto
   GoogleGenerativeAIEmbeddings(
       model=settings.GEMINI_EMBED_MODEL,
       google_api_key=settings.GOOGLE_API_KEY
   )
   
   # ❌ Incorrecto (versiones antiguas)
   GoogleGenerativeAIEmbeddings(
       model=settings.GEMINI_EMBED_MODEL,
       google_api_key=SecretStr(settings.GOOGLE_API_KEY)
   )
   ```

2. **Chroma está deprecado** en `langchain_community`. El warning es esperado pero funcional. Migrar eventualmente a `langchain-chroma`.

3. **Errores de telemetría de Chroma** (`capture() takes 1 positional argument...`) son warnings inofensivos, no afectan funcionalidad.

### Patrones de Código

1. **Inyección de dependencias**: Usar `dependencies.py` con `@lru_cache()` para singletons
2. **Schemas**: Definir en `models/schemas.py` con Pydantic
3. **Configuración**: Usar `app/config.py` con `Settings` class y `.env`

### Flujo de Datos

```
Ingesta:
Archivo → DocumentLoader → TextSplitter → Embedder → VectorStore (Chroma)

Query:
Pregunta → Retriever → VectorStore.similarity_search → Prompt + Gemini → Respuesta
```

## 📁 Archivos Clave

| Archivo | Propósito |
|---------|-----------|
| `app/config.py` | Variables de entorno y configuración |
| `app/dependencies.py` | Factory de dependencias singleton |
| `app/api/query.py` | Lógica RAG completa (retrieve + generate) |
| `app/retrieval/vectorstore.py` | Wrapper de Chroma con persistencia |
| `requirements.txt` | Dependencias del proyecto |

## 🧪 Testing

```bash
# Test de conexión con Gemini
python test_gemini.py

# Ejecutar servidor de desarrollo
.\venv\Scripts\activate; uvicorn app.main:app --reload
```

## 🚨 Errores Comunes

### 1. `ModuleNotFoundError: langchain_google_genai`
**Causa**: No está activado el virtualenv
**Solución**: `.\venv\Scripts\activate`

### 2. `RESOURCE_EXHAUSTED` (429)
**Causa**: Límite de rate de la API de Google
**Solución**: Esperar unos segundos y reintentar

### 3. `Vector store is not initialized`
**Causa**: No hay documentos indexados
**Solución**: Ejecutar `/ingest/upload` primero

### 4. `ForwardRef._evaluate() missing argument`
**Causa**: Incompatibilidad Pydantic v1/v2 con Python 3.12
**Solución**: Actualizar langchain y langsmith a últimas versiones

## 📝 Convenciones

- **Idioma de código**: Inglés
- **Comentarios**: Español (proyecto educativo)
- **Docstrings**: Español con formato descriptivo
- **Nombres de variables**: snake_case
- **Clases**: PascalCase

## 🔐 Variables de Entorno

```env
GOOGLE_API_KEY=xxx          # Requerido - API key de Google AI
GEMINI_MODEL=gemini-2.0-flash-exp
GEMINI_EMBED_MODEL=text-embedding-004
CHUNK_SIZE=800
CHUNK_OVERLAP=150
```

## 🎯 Comandos Útiles

```bash
# Activar entorno
.\venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt

# Ejecutar servidor
uvicorn app.main:app --reload

# Test de Gemini
python test_gemini.py
```

## 📚 Referencias

- [LangChain Docs](https://python.langchain.com/)
- [Google AI Studio](https://aistudio.google.com/)
- [ChromaDB Docs](https://docs.trychroma.com/)
- [FastAPI Docs](https://fastapi.tiangolo.com/)
