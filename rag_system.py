import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional
import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

class RAGSystem:
    """Advanced RAG system for intelligent data analysis with exact answer retrieval"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.embedding_model = SentenceTransformer(model_name)
        self.chroma_client = chromadb.Client(Settings(anonymized_telemetry=False))
        self.collection = None
        self.collection_name = "data_context"
        self.data = None
        logger.info(f"RAG System initialized with model: {model_name}")
    
    def create_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Create embeddings using sentence-transformers"""
        try:
            embeddings = self.embedding_model.encode(texts, convert_to_tensor=False)
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Error creating embeddings: {e}")
            raise
    
    def index_data_context(self, data_info: str, schema_info: str, dataframe: Optional[pd.DataFrame] = None):
        """Intelligently index comprehensive data context for exact answer retrieval"""
        try:
            self.data = dataframe
            
            # Delete existing collection if it exists
            try:
                self.chroma_client.delete_collection(self.collection_name)
            except:
                pass
            
            # Create new collection
            self.collection = self.chroma_client.create_collection(
                name=self.collection_name,
                metadata={"description": "Intelligent data analysis context"}
            )
            
            documents = []
            metadatas = []
            ids = []
            doc_id = 0
            
            # 1. Schema and structure information
            documents.append(f"Dataset Schema and Structure:\n{schema_info}")
            metadatas.append({"type": "schema", "priority": "high"})
            ids.append(f"schema_{doc_id}")
            doc_id += 1
            
            documents.append(f"Dataset Overview:\n{data_info}")
            metadatas.append({"type": "overview", "priority": "high"})
            ids.append(f"overview_{doc_id}")
            doc_id += 1
            
            # 2. Detailed column information with statistics
            if dataframe is not None and len(dataframe) > 0:
                for col in dataframe.columns:
                    col_info = f"Column: {col}\n"
                    col_info += f"Data Type: {dataframe[col].dtype}\n"
                    col_info += f"Non-null Count: {dataframe[col].notna().sum()}/{len(dataframe)}\n"
                    col_info += f"Null Count: {dataframe[col].isna().sum()}\n"
                    col_info += f"Unique Values: {dataframe[col].nunique()}\n"
                    
                    # Add statistics for numeric columns
                    if pd.api.types.is_numeric_dtype(dataframe[col]):
                        col_info += f"Min: {dataframe[col].min()}\n"
                        col_info += f"Max: {dataframe[col].max()}\n"
                        col_info += f"Mean: {dataframe[col].mean():.2f}\n"
                        col_info += f"Median: {dataframe[col].median():.2f}\n"
                        col_info += f"Standard Deviation: {dataframe[col].std():.2f}\n"
                    
                    # Add sample values for categorical columns
                    if dataframe[col].dtype == 'object' or dataframe[col].nunique() < 20:
                        top_values = dataframe[col].value_counts().head(10)
                        col_info += f"Top Values:\n{top_values.to_string()}\n"
                    
                    documents.append(col_info)
                    metadatas.append({"type": "column_info", "column": col, "priority": "high"})
                    ids.append(f"col_{col}_{doc_id}")
                    doc_id += 1
                
                # 3. Data samples (first few rows)
                sample_data = dataframe.head(10).to_string()
                documents.append(f"Sample Data (First 10 rows):\n{sample_data}")
                metadatas.append({"type": "sample_data", "priority": "medium"})
                ids.append(f"sample_{doc_id}")
                doc_id += 1
                
                # 4. Statistical summaries
                if len(dataframe.select_dtypes(include=[np.number]).columns) > 0:
                    numeric_summary = dataframe.describe().to_string()
                    documents.append(f"Statistical Summary (Numeric Columns):\n{numeric_summary}")
                    metadatas.append({"type": "statistics", "priority": "high"})
                    ids.append(f"stats_{doc_id}")
                    doc_id += 1
                
                # 5. Correlation information
                numeric_cols = dataframe.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 1:
                    corr_matrix = dataframe[numeric_cols].corr()
                    # Store strong correlations
                    strong_corrs = []
                    for i in range(len(corr_matrix.columns)):
                        for j in range(i+1, len(corr_matrix.columns)):
                            corr_val = corr_matrix.iloc[i, j]
                            if abs(corr_val) > 0.5:
                                strong_corrs.append(
                                    f"{corr_matrix.columns[i]} and {corr_matrix.columns[j]}: {corr_val:.3f}"
                                )
                    if strong_corrs:
                        corr_info = "Strong Correlations (|r| > 0.5):\n" + "\n".join(strong_corrs)
                        documents.append(corr_info)
                        metadatas.append({"type": "correlations", "priority": "medium"})
                        ids.append(f"corr_{doc_id}")
                        doc_id += 1
                
                # 6. Date/time column information
                date_cols = [col for col in dataframe.columns if pd.api.types.is_datetime64_any_dtype(dataframe[col])]
                if date_cols:
                    date_info = "Date/Time Columns:\n"
                    for col in date_cols:
                        date_info += f"{col}: Range from {dataframe[col].min()} to {dataframe[col].max()}\n"
                    documents.append(date_info)
                    metadatas.append({"type": "date_info", "priority": "medium"})
                    ids.append(f"date_{doc_id}")
                    doc_id += 1
            
            # 7. Common analysis patterns and operations
            analysis_patterns = [
                "To find top N items: df.nlargest(N, 'column') or df.sort_values('column', ascending=False).head(N)",
                "To group and aggregate: df.groupby('category')['metric'].agg(['sum', 'mean', 'count'])",
                "To filter data: df[df['column'] > value] or df.query('condition')",
                "To calculate percentages: (df['col'].value_counts(normalize=True) * 100).round(2)",
                "To find unique values: df['column'].unique() or df['column'].value_counts()",
                "To handle missing values: df.dropna() or df.fillna(value) or df.isna().sum()",
                "To create visualizations: plt.figure(), sns.barplot(), df.plot(), plotly.express charts",
                "To calculate correlations: df.corr() or df['col1'].corr(df['col2'])",
                "To pivot data: df.pivot_table(values='value', index='row', columns='col', aggfunc='sum')",
                "To merge dataframes: pd.merge(df1, df2, on='key') or df1.join(df2)",
                "To extract date parts: df['date'].dt.year, df['date'].dt.month, df['date'].dt.day",
                "To calculate time differences: (df['end'] - df['start']).dt.days",
                "To find outliers: Q1 = df['col'].quantile(0.25), Q3 = df['col'].quantile(0.75), IQR = Q3 - Q1",
                "To normalize data: (df['col'] - df['col'].min()) / (df['col'].max() - df['col'].min())",
                "To create bins: pd.cut(df['col'], bins=5) or pd.qcut(df['col'], q=5)"
            ]
            
            for pattern in analysis_patterns:
                documents.append(pattern)
                metadatas.append({"type": "pattern", "priority": "low"})
                ids.append(f"pattern_{doc_id}")
                doc_id += 1
            
            # Create embeddings
            embeddings = self.create_embeddings(documents)
            
            # Add to collection
            self.collection.add(
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids
            )
            
            logger.info(f"Indexed {len(documents)} intelligent context documents")
        except Exception as e:
            logger.error(f"Error indexing data context: {e}")
            raise
    
    def retrieve_context(self, query: str, top_k: int = 5) -> List[str]:
        """Intelligently retrieve relevant context with enhanced relevance"""
        try:
            if self.collection is None:
                return []
            
            query_embedding = self.create_embeddings([query])[0]
            
            # Retrieve more results for better context
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=min(top_k * 2, 15),  # Get more, then filter
                include=['documents', 'metadatas', 'distances']
            )
            
            if not results['documents'] or len(results['documents']) == 0:
                return []
            
            documents = results['documents'][0]
            metadatas = results['metadatas'][0] if results['metadatas'] else []
            distances = results['distances'][0] if results['distances'] else []
            
            # Prioritize high-priority documents and filter by relevance
            scored_docs = []
            for i, (doc, metadata, distance) in enumerate(zip(documents, metadatas, distances)):
                score = 1.0 - distance  # Convert distance to similarity
                
                # Boost priority documents
                if metadata and metadata.get('priority') == 'high':
                    score *= 1.5
                elif metadata and metadata.get('priority') == 'medium':
                    score *= 1.2
                
                # Boost column-specific matches
                if metadata and 'column' in metadata:
                    col_name = metadata['column'].lower()
                    if col_name in query.lower():
                        score *= 1.3
                
                scored_docs.append((score, doc, metadata))
            
            # Sort by score and return top_k
            scored_docs.sort(key=lambda x: x[0], reverse=True)
            top_docs = [doc for _, doc, _ in scored_docs[:top_k]]
            
            logger.info(f"Retrieved {len(top_docs)} relevant context documents")
            return top_docs
        except Exception as e:
            logger.error(f"Error retrieving context: {e}")
            return []
    
    def get_direct_answer(self, query: str) -> Optional[str]:
        """Attempt to provide direct answer from data without code generation"""
        try:
            if self.data is None or len(self.data) == 0:
                return None
            
            query_lower = query.lower()
            
            # Pattern matching for direct answers
            # Count queries
            if any(word in query_lower for word in ['how many', 'count', 'number of', 'total number']):
                # Try to extract column name
                for col in self.data.columns:
                    if col.lower() in query_lower:
                        count = len(self.data) if 'row' in query_lower or 'record' in query_lower else self.data[col].notna().sum()
                        return f"The {col} column has {count} {'rows' if 'row' in query_lower else 'non-null values'}."
            
            # Unique values queries
            if any(word in query_lower for word in ['unique', 'distinct', 'different']):
                for col in self.data.columns:
                    if col.lower() in query_lower:
                        unique_count = self.data[col].nunique()
                        return f"There are {unique_count} unique values in the {col} column."
            
            # Min/Max queries
            if 'minimum' in query_lower or 'min' in query_lower or 'lowest' in query_lower:
                for col in self.data.select_dtypes(include=[np.number]).columns:
                    if col.lower() in query_lower:
                        min_val = self.data[col].min()
                        return f"The minimum value in {col} is {min_val}."
            
            if 'maximum' in query_lower or 'max' in query_lower or 'highest' in query_lower:
                for col in self.data.select_dtypes(include=[np.number]).columns:
                    if col.lower() in query_lower:
                        max_val = self.data[col].max()
                        return f"The maximum value in {col} is {max_val}."
            
            # Average/Mean queries
            if any(word in query_lower for word in ['average', 'mean', 'avg']):
                for col in self.data.select_dtypes(include=[np.number]).columns:
                    if col.lower() in query_lower:
                        mean_val = self.data[col].mean()
                        return f"The average value of {col} is {mean_val:.2f}."
            
            return None
        except Exception as e:
            logger.debug(f"Error getting direct answer: {e}")
            return None
