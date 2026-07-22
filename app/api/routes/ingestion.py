import os
from fastapi import APIRouter, UploadFile, File, Form
from app.api.schemas import IngestionResponse
from app.ingestion.processor import DocumentProcessor

router = APIRouter()
_processor = DocumentProcessor()

@router.post("/ingest", response_model=IngestionResponse)
async def ingest(file: UploadFile = File(...), source_type: str = Form("general")):
    """Upload and ingest a PDF document."""
    # Save uploaded file temporarily
    temp_path = os.path.join("data", file.filename)
    with open(temp_path, "wb") as f:
        content = await file.read()
        f.write(content)

    _processor.process_file(
        file_path=temp_path,
        filename=file.filename,
        source_type=source_type,
    )

    count = _processor._vector_db.get_points_count()

    return IngestionResponse(
        filename=file.filename,
        source_type=source_type,
        chunks_created=count,
        vectors_upserted=count,
        status="Ingestion completed successfully",
    )
