# app/services/llm_service.py

import json
from typing import Dict, Any, List
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
