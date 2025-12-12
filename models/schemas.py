"""
 Schemas -Models of data for request/responose
 Pydantic automatic valiation entering data
"""

import json
from pydantic import BaseModel, Field
from typing import List, Optional

class IngestResponse(BaseModel):
    """Response after ingestion process"""
    message: str
    files_proccessed: int
    chunks_created: int

    class Config:
        json_schema_extra = {
            "example": {
                "message": "Ingestion completed successfully.",
                "files_proccessed": 3,
                "chunks_created": 1500
            }
        }


class QueryRequest(BaseModel):
    """Request to do a query"""
    question: str = Field(..., min_length=1, description="User Question to ask the system")
    top_k: Optional[int] = Field(3, ge=1, le=10, description="Number of chunks to retrieve")

    class Config:
        json_schema_extra = {
            "examople": {
                "question": "What is a auditory process",
                "top_k": 3
            }
        }


class SourceInfo(BaseModel):
    """One source info """
    filename: str
    chunk_id: int
    content_preview: str

class QueryResponse(BaseModel):
    """Response with model answer"""
    answer: str
    source: List[SourceInfo]
    chunk_used: int

    class Config:
        json_schema_extra = {
            "example": {
                "answer": "The process of auditory perception involves the detection and interpretation of sound waves by the ear and brain.",
                "source": [
                    {
                        "filename": "document1.pdf",
                        "chunk_id": 5,
                        "content_preview": "Auditory perception is the process by which the brain interprets sound waves..."
                    }
                ],
                "chunk_used": 3
            }
        }