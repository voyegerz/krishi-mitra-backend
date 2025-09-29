# api/routes/dd.py

import os
from typing import Annotated, Optional
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Query
from app.api.deps import CurrentUser
from app.services.llm_service import LLMService
from app.core.config import settings

router = APIRouter(tags=["disease_detection"])

def get_llm_service() -> LLMService:
    """Dependency to provide the LLM service instance."""
    if not settings.GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="Gemini API key is not configured.")
    return LLMService(api_key=settings.GEMINI_API_KEY)

@router.post("/detect-disease")
async def detect_crop_disease(
    file: Annotated[UploadFile, File(..., description="An image of the diseased crop.")],
    llm_service: Annotated[LLMService, Depends(get_llm_service)],
    current_user: CurrentUser, # Assuming authentication is required
    lang: Annotated[Optional[str], Query(description="The language for the response (e.g., 'Hindi', 'Marathi', 'Telugu').")] = "English",
):
    """
    Detects crop disease from an uploaded image and provides a diagnosis in the specified language.
    """
    # Create the directory for temporary images if it doesn't exist
    temp_dir = "temp_images_dd"
    os.makedirs(temp_dir, exist_ok=True)
    
    # Save the uploaded image to a temporary location
    file_path = os.path.join(temp_dir, file.filename)
    try:
        with open(file_path, "wb") as buffer:
            while content := await file.read(1024):
                buffer.write(content)

        analysis_response = await llm_service.detect_disease(image_path=file_path, language=lang)

        # Clean up the temporary file
        os.remove(file_path)

        if "error" in analysis_response:
            raise HTTPException(status_code=500, detail=analysis_response["error"])
        
        return analysis_response
    except Exception as e:
        # Ensure file is removed even if an error occurs
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=f"Failed to process image: {e}")