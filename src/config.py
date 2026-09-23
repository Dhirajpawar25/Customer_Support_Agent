"""Configuration for the Aster & Row AI Support Agent."""
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
from dotenv import dotenv_values, load_dotenv

# Load environment variables early
load_dotenv(override=True)
_file_settings = dotenv_values(Path(__file__).parent.parent / ".env")


def _setting(name: str, default: Optional[str] = None, *aliases: str) -> Optional[str]:
    for key in (name, *aliases):
        if _file_settings.get(key):
            return _file_settings[key]
    return os.getenv(name, default)

@dataclass
class Config:
    # Paths
    project_root: Path = Path(__file__).parent.parent
    knowledge_base_path: Path = Path(__file__).parent.parent / "knowledge-base"
    orders_path: Path = Path(__file__).parent.parent / "data" / "orders.json"
    evaluation_path: Path = Path(__file__).parent.parent / "evaluation" / "visible-cases.json"
    logs_path: Path = Path(__file__).parent.parent / "logs"
    
    # Provider: "openai-compatible" or "gemini"
    provider: str = field(default_factory=lambda: _setting("LLM_PROVIDER", "openai-compatible", "PROVIDER"))
    
    # Model settings
    embedding_model: str = field(default_factory=lambda: os.getenv("EMBEDDING_MODEL", "local-hashing"))
    llm_model: str = field(default_factory=lambda: _setting("LLM_MODEL", "nvidia/llama-3.1-nemotron-nano-v1.1", "GEMINI_MODEL"))
    llm_base_url: Optional[str] = field(default_factory=lambda: _setting("LLM_BASE_URL", "https://integrate.api.nvidia.com/v1", "base_url"))
    llm_temperature: float = field(default_factory=lambda: float(_setting("LLM_TEMPERATURE", "0.1")))
    
    # Retrieval settings
    chunk_size: int = 500
    chunk_overlap: int = 50
    top_k: int = 5
    min_score_threshold: float = 0.3
    
    # API Keys
    openai_api_key: Optional[str] = field(default_factory=lambda: _setting("OPENAI_API_KEY"))
    gemini_api_key: Optional[str] = field(default_factory=lambda: _setting("GEMINI_API_KEY"))
    llm_api_key: Optional[str] = field(default_factory=lambda: _setting("LLM_API_KEY", None, "NVIDIA_API_KEY", "GEMINI_API_KEY", "api_key"))
    
    # Debug
    debug_mode: bool = False
    
    def __post_init__(self):
        self.logs_path.mkdir(exist_ok=True)
        
        # Validate API key for selected provider
        if self.provider == "gemini" and not self.gemini_api_key:
            raise ValueError("GEMINI_API_KEY is required when PROVIDER=gemini")
        if self.provider in ("openai", "openai-compatible") and not (self.llm_api_key or self.openai_api_key):
            raise ValueError("LLM_API_KEY is required when LLM_PROVIDER is openai-compatible")

config = Config()