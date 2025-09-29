# app/services/llm_service.py

import json
from typing import Dict, Any, List, Optional
import logging
from pathlib import Path
import base64

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langchain.prompts import PromptTemplate

logger = logging.getLogger(__name__)

class LLMService:
    """Service for handling OCR and evaluation of exam answersheets using Gemini."""

    def __init__(self, api_key: str, model: str = "gemini-1.5-flash"):
        self.llm = ChatGoogleGenerativeAI(
            model=model,
            google_api_key=api_key,
            temperature=0.2,  
        )

        # The prompt is now a simple template for the LLM's instructions.
        self.evaluation_prompt = (
            "You are an exam evaluator. Evaluate the student's answer found on this image.\n\n"
            "The maximum marks for this question are {max_marks}.\n\n"
            "Provide:\n"
            "1. Marks awarded (numeric only).\n"
            "2. A short feedback (2-3 sentences).\n"
        )
    
    async def evaluate_answer(
        self, image_path: str, max_marks: int
    ) -> Dict[str, Any]:
        """
        Evaluate an answer directly from an image using the LLM.
        """
        try:
            path = Path(image_path)
            if not path.exists():
                raise FileNotFoundError(f"Image not found: {image_path}")
            
            # Read image as bytes and encode to base64
            with open(image_path, "rb") as f:
                image_bytes = f.read()
            encoded_image = base64.b64encode(image_bytes).decode("utf-8")
            
            # The prompt now includes both the evaluation instructions and the image
            message = HumanMessage(
                content=[
                    {
                        "type": "text",
                        "text": self.evaluation_prompt.format(max_marks=max_marks),
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{encoded_image}"},
                    },
                ]
            )

            response = await self.llm.ainvoke([message])
            return {"evaluation": response.content.strip()} # type: ignore

        except Exception as e:
            logger.error(f"Evaluation failed for image {image_path}: {e}")
            return {"error": str(e)}

    async def batch_evaluate(
        self, image_paths: List[str], max_marks_list: List[int]
    ) -> List[Dict[str, Any]]:
        """
        Evaluate multiple answers from a list of images.
        `image_paths` is a list of paths to student answer images.
        `max_marks_list` is a list of max marks for each question.
        """
        results = []
        for img_path, max_marks in zip(image_paths, max_marks_list):
            eval_result = await self.evaluate_answer(
                image_path=img_path, max_marks=max_marks
            )
            results.append(eval_result)
        return results
    
    async def process_images(self, image_paths: List[str], prompt: str) -> str:
        """
        Processes multiple images with a single prompt using Gemini's multimodal capabilities.
        """
        try:
            message_content = []
            
            # Add the text prompt first
            message_content.append(
                {
                    "type": "text",
                    "text": prompt,
                }
            )

            # Read each image, encode it, and add it to the message content list
            for image_path in image_paths:
                path = Path(image_path)
                if not path.exists():
                    logger.warning(f"Image not found, skipping: {image_path}")
                    continue

                with open(image_path, "rb") as f:
                    image_bytes = f.read()
                encoded_image = base64.b64encode(image_bytes).decode("utf-8")
                
                message_content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{encoded_image}"},
                    }
                )

            if not message_content:
                return json.dumps({"error": "No valid images found to process."})

            message = HumanMessage(content=message_content)

            response = await self.llm.ainvoke([message])
            
            cleaned_response = response.content.strip() # type: ignore
            if cleaned_response.startswith("```"):
                cleaned_response = cleaned_response.strip("`")
                # remove the first line (```json or ```)
                cleaned_response = "\n".join(cleaned_response.split("\n")[1:])
                # remove the last line (closing ```)
                if cleaned_response.strip().endswith("```"):
                    cleaned_response = "\n".join(cleaned_response.split("\n")[:-1])
            return cleaned_response.strip() # type: ignore

        except Exception as e:
            logger.error(f"Processing images with LLM failed: {e}")
            return json.dumps({"error": str(e)})