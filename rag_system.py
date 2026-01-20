from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict
import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

class RAGSystem:
    """Lightweight RAG system using TF-IDF (no heavy ML dependencies)"""
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
        self.documents = []
        self.document_vectors = None
        self.data_stats = {}
        logger.info("RAG System initialized with TF-IDF")
    
    def analyze_data_patterns(self, df: pd.DataFrame) -> Dict:
        """Analyze data patterns"""
        patterns = {
            "numeric_columns": list(df.select_dtypes(include=[np.number]).columns),
            "categorical_columns": list(df.select_dtypes(include=['object']).columns),
            "date_columns": list(df.select_dtypes(include=['datetime64']).columns),
        }
        self.data_stats = patterns
        return patterns
    
    def generate_smart_context_documents(self, df: pd.DataFrame, schema_info: str, data_info: str) -> List[str]:
        """Generate context documents"""
        documents = [
            f"Dataset Schema:\\n{schema_info}",
            f"Dataset Overview:\\n{data_info}",
            f"Columns: {', '.join(df.columns.tolist())}",
            f"Shape: {df.shape[0]} rows, {df.shape[1]} columns"
        ]
        return documents
    
    def index_data_context(self, df: pd.DataFrame, data_info: str, schema_info: str):
        """Index data context"""
        try:
            documents = self.generate_smart_context_documents(df, schema_info, data_info)
            self.documents = documents
            self.document_vectors = self.vectorizer.fit_transform(documents)
            logger.info(f"Indexed {len(documents)} context documents")
        except Exception as e:
            logger.error(f"Error indexing data context: {e}")
            raise
    
    def index_pdf_content(self, text_chunks: List[str], source_name: str):
        """Index PDF content"""
        try:
            self.documents = text_chunks
            self.document_vectors = self.vectorizer.fit_transform(text_chunks)
            logger.info(f"Indexed {len(text_chunks)} PDF pages")
        except Exception as e:
            logger.error(f"Error indexing PDF: {e}")
            raise
    
    def retrieve_context(self, query: str, top_k: int = 4) -> List[str]:
        """Retrieve relevant context"""
        try:
            if not self.documents or self.document_vectors is None:
                return []
            
            query_vector = self.vectorizer.transform([query])
            similarities = cosine_similarity(query_vector, self.document_vectors)[0]
            top_indices = similarities.argsort()[-top_k:][::-1]
            
            return [self.documents[i] for i in top_indices]
        except Exception as e:
            logger.error(f"Error retrieving context: {e}")
            return []
    
    def get_data_insights(self) -> Dict:
        """Return data stats"""
        return self.data_stats