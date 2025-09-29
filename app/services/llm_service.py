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
    """Service for handling AI advisory and image-based assistance using Gemini."""

    def __init__(self, api_key: str, model: str = "gemini-2.5-flash"):
        self.llm = ChatGoogleGenerativeAI(
            model=model,
            google_api_key=api_key,
            temperature=0.2,  
        )
        # You can keep existing prompts or define new ones as needed.
        self.advisory_prompt = (
            "You are a smart crop advisory system for small and marginal farmers in India. "
            "Provide real-time, localized advice on crops, soil, and weather. "
            "Integrate information on mandi prices, pest alerts, and government schemes. "
            "Use simple, easy-to-understand language. "
            "Keep the response concise and actionable. "
            "Query: {query}\n\nAdvisory:"
        )

        self.image_advisory_prompt = (
            "You are an intelligent assistant for a crop advisory app. "
            "A farmer has provided a screenshot of their phone and a question. "
            "Based on the image and the query, guide the user on how to navigate the app or use a specific feature. "
            "Explain the steps clearly and concisely. "
            "Query: {query}"
        )

        # New prompt for crop disease detection
        self.disease_detection_prompt = (
            "You are a crop disease expert. A user has uploaded an image of a plant. "
            "Based on the visual evidence, identify any diseases or pests present and suggest a natural and effective treatment or solution. "
            "The response MUST be simple and easy for a farmer to understand. "
            "Generate the entire response ONLY in {language}. "
            "Provide the following in a structured format:\n"
            "1. **Diagnosis**: [Name of the disease/pest]\n"
            "2. **Symptoms**: [Description of symptoms]\n"
            "3. **Recommended Treatment**: [Clear, actionable steps for a farmer]\n"
            "Example for Hindi (language=Hindi):\n"
            "1. **निदान**: पाउडरी फफूंदी (Powdery Mildew)\n"
            "2. **लक्षण**: पत्तियों और तनों पर सफेद, पाउडर जैसा पदार्थ दिखाई देता है।\n"
            "3. **उपचार**: नीम तेल और पानी का घोल मिलाकर पौधों पर स्प्रे करें।"
        )

    async def get_advisory(self, user_query: str) -> Dict[str, Any]:
        """
        Provides text-based crop advisory using the LLM.
        """
        try:
            full_prompt = self.advisory_prompt.format(query=user_query)
            message = HumanMessage(content=full_prompt)
            response = await self.llm.ainvoke([message])
            return {"advisory": response.content.strip()}
        except Exception as e:
            logger.error(f"Text advisory failed for query '{user_query}': {e}")
            return {"error": str(e)}

    async def get_image_advisory(self, image_path: str, user_query: str) -> Dict[str, Any]:
        """
        Provides guidance on app navigation or features based on an image and text query.
        """
        try:
            path = Path(image_path)
            if not path.exists():
                raise FileNotFoundError(f"Image not found: {image_path}")
            
            with open(image_path, "rb") as f:
                image_bytes = f.read()
            encoded_image = base64.b64encode(image_bytes).decode("utf-8")
            
            full_prompt = self.image_advisory_prompt.format(query=user_query)
            
            message_content = [
                {
                    "type": "text",
                    "text": full_prompt,
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{encoded_image}"},
                },
            ]

            message = HumanMessage(content=message_content)
            response = await self.llm.ainvoke([message])
            return {"advisory": response.content.strip()}
        except Exception as e:
            logger.error(f"Image-based advisory failed for query '{user_query}': {e}")
            return {"error": str(e)}

    # New method for disease detection
    async def detect_disease(self, image_path: str, language: Optional[str] = "English") -> Dict[str, Any]:
        """
        Detects crop diseases from an image and provides diagnosis and treatment in the specified language.
        """
        try:
            path = Path(image_path)
            if not path.exists():
                raise FileNotFoundError(f"Image not found: {image_path}")

            with open(image_path, "rb") as f:
                image_bytes = f.read()
            encoded_image = base64.b64encode(image_bytes).decode("utf-8")
            
            # Format the prompt with the requested language
            full_prompt = self.disease_detection_prompt.format(language=language)

            message_content = [
                {"type": "text", "text": full_prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded_image}"}},
            ]

            message = HumanMessage(content=message_content)
            response = await self.llm.ainvoke([message])
            return {"analysis": response.content.strip()}
        except Exception as e:
            logger.error(f"Disease detection failed for image {image_path}: {e}")
            return {"error": str(e)}