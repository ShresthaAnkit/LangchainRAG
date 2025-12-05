from fastapi import APIRouter, UploadFile, File, Depends
from fastapi.responses import JSONResponse
from langchain_core.vectorstores import VectorStore
import tempfile
import os
from app.api.deps import get_vectorstore
from app.service.ingestion_service import IngestionService

router = APIRouter()

@router.post("/ingest")
async def ingest_documents(
        files: list[UploadFile] = File(...), 
        vectorstore: VectorStore = Depends(get_vectorstore)
    ):

    tmpdir = tempfile.mkdtemp()
    paths = []

    for file in files:
        filename = file.filename
        dest = os.path.join(tmpdir, filename)

        with open(dest, "wb") as out_file:
            out_file.write(await file.read())
        
        paths.append(dest)


    ingestion_service = IngestionService()
    ingestion_service.ingest(paths, vectorstore)

    return JSONResponse(content="Successfully Ingested Documents")