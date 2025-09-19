"""
Configuration module for Multi-Source RAG Engine.

This module loads environment variables from a .env file and provides
them as constants for other parts of the application. It includes validation
to ensure that critical configuration, like API keys, are present.
"""

import os
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError(
        "Google API key not found. "
        "Please create a .env file in the project root and add the following line:\n"
        "GOOGLE_API_KEY='your-api-key-here'"
    )