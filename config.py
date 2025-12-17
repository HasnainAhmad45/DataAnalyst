import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Configuration for the Data Analyst Agent"""
    
    # API Keys
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
    
    # Model Configuration
    GEMINI_MODEL = "gemini-2.5-flash"  # or "gemini-1.5-flash" for faster responses
    MAX_TOKENS = 8192
    TEMPERATURE = 0
    
    # RAG Configuration
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Local sentence-transformers model
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    TOP_K_RESULTS = 5
    
    # Execution Configuration
    MAX_RETRIES = 3
    EXECUTION_TIMEOUT = 30
    
    # Output Configuration
    OUTPUT_DIR = "outputs"
    PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")
    LOGS_DIR = os.path.join(OUTPUT_DIR, "logs")
    
    @classmethod
    def setup_directories(cls):
        """Create necessary directories"""
        os.makedirs(cls.OUTPUT_DIR, exist_ok=True)
        os.makedirs(cls.PLOTS_DIR, exist_ok=True)
        os.makedirs(cls.LOGS_DIR, exist_ok=True)