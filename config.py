import os
from typing import List, Optional
from pydantic import BaseModel

class Settings(BaseModel):
    aws_region: str = "us-east-1"
    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None
    aws_session_token: Optional[str] = None
    allowed_origins: List[str] = []
    bedrock_model_id: str = "mistral.mistral-large-2402-v1:0"
    dashscope_api_key: Optional[str] = None

    @classmethod
    def from_env(cls) -> "Settings":
        raw_origins = os.getenv("ALLOWED_ORIGINS", "")
        allowed_origins = [origin.strip() for origin in raw_origins.split(",") if origin.strip()]
        return cls(
            aws_region=os.getenv("AWS_REGION", "us-east-1"),
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            aws_session_token=os.getenv("AWS_SESSION_TOKEN"),
            allowed_origins=allowed_origins,
            bedrock_model_id=os.getenv("BEDROCK_MODEL_ID", "mistral.mistral-large-2402-v1:0"),
            dashscope_api_key=os.getenv("DASHSCOPE_API_KEY"),
        )

# Create a single, shared instance for the whole application to use
settings = Settings.from_env()