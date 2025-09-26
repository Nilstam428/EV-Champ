import os
from typing import List, Optional
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """
    Application settings
    """
    # API configuration
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "EV Champ API"
    PROJECT_DESCRIPTION: str = "API for EV Champ Chatbot"
    PROJECT_VERSION: str = "0.1.0"
    
    # CORS configuration
    CORS_ORIGINS: List[str] = ["*"]  # In production, replace with specific origins
    
    # Database configuration
    DATABASE_URL: str = "sqlite:///./ev_champ.db"
    
    # LLM configuration - Groq API
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    RAG_MODEL_NAME: str = "meta-llama/llama-4-maverick-17b-128e-instruct"
    CHAT_MODEL_NAME: str = "meta-llama/llama-4-maverick-17b-128e-instruct"
    
    # Legacy fields for backward compatibility (if they exist in .env)
    LLM_PROVIDER: Optional[str] = None
    LLM_MODEL_NAME: Optional[str] = None
    OPENAI_API_KEY: Optional[str] = None
    AWS_REGION: Optional[str] = None
    
    # Vector DB for RAG (optional - can use in-memory for testing)
    VECTOR_DB_PATH: str = "./vector_db"
    
    # Security
    SECRET_KEY: str = "supersecretkey"  # Change in production
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 7  # 7 days
    
    model_config = {
        "env_file": ".env",
        "case_sensitive": True,
        "extra": "ignore"  # This will ignore extra fields
    }

# Create settings instance
settings = Settings()