import shutil
from fastapi import APIRouter, UploadFile, File, Depends
from langchain_core.vectorstores import VectorStore
import tempfile
import os
from app.api.deps import get_vectorstore_deps
from app.service.ingestion_service import IngestionService
from app.schema.api import ApiResponse

router = APIRouter()


@router.post("/ingest", response_model=ApiResponse)
async def ingest_documents(
    files: list[UploadFile] = File(...),
    vectorstore: VectorStore = Depends(get_vectorstore_deps),
):
    tmpdir = tempfile.mkdtemp()
    paths = []
    try:
        for file in files:
            filename = file.filename
            dest = os.path.join(tmpdir, filename)

            with open(dest, "wb") as out_file:
                out_file.write(await file.read())

            paths.append(dest)

        ingestion_service = IngestionService()
        ingestion_service.ingest(paths, vectorstore)
    finally:
        shutil.rmtree(tmpdir)

    return {"success": True, "message": "Successfully Ingested Documents"}
