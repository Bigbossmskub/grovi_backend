from fastapi import APIRouter, HTTPException, status
from fastapi.responses import FileResponse
from pathlib import Path
import os

router = APIRouter(prefix="/images", tags=["images"])

@router.get("/storage/vi_images/{filename}")
async def serve_vi_image(filename: str):
    """Serve VI overlay images"""
    file_path = Path(f"backend/storage/vi_images/{filename}")
    
    if not file_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Image not found"
        )
    
    return FileResponse(
        path=file_path,
        media_type="image/png",
        filename=filename
    )
