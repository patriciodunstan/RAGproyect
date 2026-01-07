# 🔍 RAG Project - Retrieval Augmented Generation

Sistema RAG (Retrieval Augmented Generation) construido con FastAPI, LangChain y Google Gemini para responder preguntas basándose en documentos.

## 📋 Características

- **Ingesta de documentos**: Soporta PDF, TXT y DOCX
- **Vectorización**: Embeddings con Google Gemini (`text-embedding-004`)
- **Base vectorial**: ChromaDB para almacenamiento y búsqueda
- **Generación**: Respuestas con Google Gemini 2.0 Flash
- **API REST**: Endpoints documentados con Swagger/OpenAPI

## 🏗️ Arquitectura

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  Documentos │───▶│   Chunking  │───▶│  Embeddings │
│  PDF/TXT/   │    │  (800 chars)│    │   (Gemini)  │
│    DOCX     │    └─────────────┘    └──────┬──────┘
└─────────────┘                              │
                                             ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  Respuesta  │◀───│   Gemini    │◀───│  ChromaDB   │
│   + Citas   │    │  2.0 Flash  │    │  (Vectors)  │
└─────────────┘    └─────────────┘    └─────────────┘
```

## 🚀 Instalación

### 1. Clonar el repositorio
```bash
git clone https://github.com/patriciodunstan/RAGproyect.git
cd RAGproyect
```

### 2. Crear entorno virtual
```bash
python -m venv venv
.\venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac
```

### 3. Instalar dependencias
```bash
pip install -r requirements.txt
```

### 4. Configurar variables de entorno
Crear archivo `.env` en la raíz:
```env
GOOGLE_API_KEY=tu_api_key_de_google
GEMINI_MODEL=gemini-2.0-flash-exp
GEMINI_EMBED_MODEL=text-embedding-004
CHUNK_SIZE=800
CHUNK_OVERLAP=150
```

> 💡 Obtén tu API key en: https://aistudio.google.com/app/apikey

### 5. Ejecutar el servidor
```bash
.\venv\Scripts\activate; uvicorn app.main:app --reload
```

## 📡 API Endpoints

### Documentación interactiva
- **Swagger UI**: http://127.0.0.1:8000/docs
- **ReDoc**: http://127.0.0.1:8000/redoc

### Endpoints disponibles

| Método | Endpoint | Descripción |
|--------|----------|-------------|
| `GET` | `/` | Health check |
| `POST` | `/ingest/upload` | Subir y procesar documentos |
| `POST` | `/query/ask` | Hacer preguntas al RAG |

### Ejemplo de uso

#### Subir documento
```bash
curl -X POST "http://127.0.0.1:8000/ingest/upload" \
  -F "file=@documento.pdf"
```

#### Hacer una consulta
```bash
curl -X POST "http://127.0.0.1:8000/query/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "¿Qué dice el documento?", "top_k": 3}'
```

#### Respuesta ejemplo
```json
{
  "answer": "Según el documento...",
  "source": [
    {
      "filename": "documento.pdf",
      "chunk_id": 0,
      "content_preview": "Texto del chunk..."
    }
  ],
  "chunk_used": 3
}
```

## 📁 Estructura del Proyecto

```
RAGproyect/
├── app/
│   ├── main.py              # FastAPI app principal
│   ├── config.py            # Configuración y settings
│   ├── dependencies.py      # Inyección de dependencias
│   ├── api/
│   │   ├── ingest.py        # Endpoint de ingesta
│   │   └── query.py         # Endpoint de consultas
│   ├── ingestion/
│   │   ├── loader.py        # Carga de documentos
│   │   ├── splitter.py      # División en chunks
│   │   └── embedder.py      # Generación de embeddings
│   └── retrieval/
│       ├── vectorstore.py   # Gestión de ChromaDB
│       └── retriever.py     # Recuperación de contexto
├── models/
│   └── schemas.py           # Schemas Pydantic
├── data/                    # Documentos subidos
├── vector_db/               # Base de datos vectorial
├── requirements.txt
├── Dockerfile
├── azure-pipelines.yml
└── README.md
```

## 🛠️ Tecnologías

| Componente | Tecnología |
|------------|------------|
| **Framework** | FastAPI |
| **LLM** | Google Gemini 2.0 Flash |
| **Embeddings** | Google Gemini `text-embedding-004` |
| **Vector Store** | ChromaDB |
| **Orquestación** | LangChain |
| **Validación** | Pydantic |

## ⚙️ Configuración

| Variable | Descripción | Default |
|----------|-------------|---------|
| `GOOGLE_API_KEY` | API key de Google AI | (requerido) |
| `GEMINI_MODEL` | Modelo para generación | `gemini-2.0-flash-exp` |
| `GEMINI_EMBED_MODEL` | Modelo para embeddings | `text-embedding-004` |
| `CHUNK_SIZE` | Tamaño de chunks | `800` |
| `CHUNK_OVERLAP` | Solapamiento entre chunks | `150` |

## 🐳 Docker

```bash
docker build -t rag-project .
docker run -p 8000:8000 --env-file .env rag-project
```

## 📝 Licencia

MIT License

## 👤 Autor

**Patricio Dunstan**
- GitHub: [@patriciodunstan](https://github.com/patriciodunstan)