# api/routes/advisory.py

import os
from typing import Annotated
from fastapi import APIRouter, Depends, HTTPException, Query, UploadFile, File
from app.api.deps import CurrentUser
from app.services.llm_service import LLMService
from app.core.config import settings 

router = APIRouter(tags=["advisory"])

def get_llm_service() -> LLMService:
    """Dependency to provide the LLM service instance."""
    if not settings.GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="Gemini API key is not configured.")
    return LLMService(api_key=settings.GEMINI_API_KEY)

@router.get("/text-advisory")
async def get_text_advisory(
    user_query: Annotated[str, Query(..., description="The user's crop-related query.")],
    llm_service: Annotated[LLMService, Depends(get_llm_service)],
    current_user: CurrentUser, # Assuming authentication is required
):
    """
    Get text-based crop and soil advisory.
    """
    if not user_query:
        raise HTTPException(status_code=400, detail="Query string cannot be empty.")
    
    advisory_response = await llm_service.get_advisory(user_query)
    
    if "error" in advisory_response:
        raise HTTPException(status_code=500, detail=advisory_response["error"])
        
    return advisory_response

@router.post("/image-advisory")
async def get_image_advisory(
    file: Annotated[UploadFile, File(..., description="An image of the user's screen.")],
    user_query: Annotated[str, Query(..., description="The user's question about the app navigation.")],
    llm_service: Annotated[LLMService, Depends(get_llm_service)],
    current_user: CurrentUser, # Assuming authentication is required
):
    """
    Get image-based app navigation guidance.
    """
    # Create the directory for temporary images if it doesn't exist
    temp_dir = "temp_images"
    os.makedirs(temp_dir, exist_ok=True)
    
    # Save the uploaded image to a temporary location
    file_path = os.path.join(temp_dir, file.filename)
    try:
        with open(file_path, "wb") as buffer:
            while content := await file.read(1024):
                buffer.write(content)

        advisory_response = await llm_service.get_image_advisory(file_path, user_query)

        # Clean up the temporary file
        os.remove(file_path)

        if "error" in advisory_response:
            raise HTTPException(status_code=500, detail=advisory_response["error"])
        
        return advisory_response
    except Exception as e:
        # Ensure file is removed even if an error occurs
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=f"Failed to process image: {e}")