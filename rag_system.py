import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
<<<<<<< HEAD
from typing import List, Dict, Tuple
=======
from typing import List, Dict, Optional
>>>>>>> b161ff043a0535b4a953740da79412d61fd73b8a
import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

class RAGSystem:
<<<<<<< HEAD
    """Enhanced Retrieval-Augmented Generation system with intelligent context retrieval"""
=======
    """Advanced RAG system for intelligent data analysis with exact answer retrieval"""
>>>>>>> b161ff043a0535b4a953740da79412d61fd73b8a
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.embedding_model = SentenceTransformer(model_name)
        self.chroma_client = chromadb.Client(Settings(anonymized_telemetry=False))
        self.collection = None
        self.collection_name = "data_context"
<<<<<<< HEAD
        self.data_stats = {}
=======
        self.data = None
>>>>>>> b161ff043a0535b4a953740da79412d61fd73b8a
        logger.info(f"RAG System initialized with model: {model_name}")
    
    def create_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Create embeddings using sentence-transformers"""
        try:
            embeddings = self.embedding_model.encode(texts, convert_to_tensor=False)
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Error creating embeddings: {e}")
            raise
    
<<<<<<< HEAD
    def analyze_data_patterns(self, df: pd.DataFrame) -> Dict:
        """Analyze data to extract intelligent patterns and insights"""
        patterns = {
            "numeric_columns": [],
            "categorical_columns": [],
            "date_columns": [],
            "high_cardinality_columns": [],
            "potential_id_columns": [],
            "columns_with_nulls": [],
            "correlations": [],
            "outlier_columns": []
        }
        
        for col in df.columns:
            dtype = df[col].dtype
            null_count = df[col].isnull().sum()
            unique_count = df[col].nunique()
            total_count = len(df)
            
            # Identify column types
            if pd.api.types.is_numeric_dtype(dtype):
                patterns["numeric_columns"].append(col)
                
                # Check for potential ID columns
                if unique_count == total_count and null_count == 0:
                    patterns["potential_id_columns"].append(col)
                
                # Check for outliers using IQR method
                if unique_count > 10:
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    outliers = ((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))).sum()
                    if outliers > 0:
                        patterns["outlier_columns"].append({
                            "column": col,
                            "outlier_count": int(outliers),
                            "outlier_percentage": round(outliers / total_count * 100, 2)
                        })
            
            elif pd.api.types.is_datetime64_any_dtype(dtype):
                patterns["date_columns"].append(col)
            
            else:
                patterns["categorical_columns"].append(col)
                
                # High cardinality check
                if unique_count > total_count * 0.5 and unique_count > 10:
                    patterns["high_cardinality_columns"].append({
                        "column": col,
                        "unique_count": int(unique_count),
                        "unique_ratio": round(unique_count / total_count, 2)
                    })
            
            # Track columns with nulls
            if null_count > 0:
                patterns["columns_with_nulls"].append({
                    "column": col,
                    "null_count": int(null_count),
                    "null_percentage": round(null_count / total_count * 100, 2)
                })
        
        # Calculate correlations for numeric columns
        numeric_df = df[patterns["numeric_columns"]]
        if len(patterns["numeric_columns"]) > 1:
            corr_matrix = numeric_df.corr()
            
            # Find strong correlations (> 0.7 or < -0.7)
            for i in range(len(corr_matrix.columns)):
                for j in range(i + 1, len(corr_matrix.columns)):
                    corr_val = corr_matrix.iloc[i, j]
                    if abs(corr_val) > 0.7:
                        patterns["correlations"].append({
                            "column1": corr_matrix.columns[i],
                            "column2": corr_matrix.columns[j],
                            "correlation": round(float(corr_val), 3)
                        })
        
        self.data_stats = patterns
        return patterns
    
    def generate_smart_context_documents(self, df: pd.DataFrame, schema_info: str, data_info: str) -> List[str]:
        """Generate intelligent context documents based on data analysis"""
        patterns = self.analyze_data_patterns(df)
        documents = []
        
        # Document 1: Schema and basic info
        documents.append(f"Dataset Schema Information:\n{schema_info}\n\nDataset Overview:\n{data_info}")
        
        # Document 2: Column type information
        col_types_doc = "Column Types:\n"
        col_types_doc += f"- Numeric columns ({len(patterns['numeric_columns'])}): {', '.join(patterns['numeric_columns'][:10])}\n"
        col_types_doc += f"- Categorical columns ({len(patterns['categorical_columns'])}): {', '.join(patterns['categorical_columns'][:10])}\n"
        col_types_doc += f"- Date/Time columns ({len(patterns['date_columns'])}): {', '.join(patterns['date_columns'])}\n"
        if patterns['potential_id_columns']:
            col_types_doc += f"- Potential ID columns: {', '.join(patterns['potential_id_columns'])}\n"
        documents.append(col_types_doc)
        
        # Document 3: Data quality insights
        quality_doc = "Data Quality Information:\n"
        if patterns['columns_with_nulls']:
            quality_doc += "Columns with missing values:\n"
            for item in patterns['columns_with_nulls'][:5]:
                quality_doc += f"  - {item['column']}: {item['null_count']} nulls ({item['null_percentage']}%)\n"
        else:
            quality_doc += "No missing values detected.\n"
        
        if patterns['outlier_columns']:
            quality_doc += "\nColumns with outliers:\n"
            for item in patterns['outlier_columns'][:5]:
                quality_doc += f"  - {item['column']}: {item['outlier_count']} outliers ({item['outlier_percentage']}%)\n"
        documents.append(quality_doc)
        
        # Document 4: Correlation insights
        if patterns['correlations']:
            corr_doc = "Strong Correlations Found:\n"
            for corr in patterns['correlations'][:10]:
                corr_doc += f"  - {corr['column1']} ↔ {corr['column2']}: {corr['correlation']}\n"
            corr_doc += "\nThese correlations may indicate relationships useful for analysis, prediction, or feature engineering."
            documents.append(corr_doc)
        
        # Document 5: Recommended analyses based on data types
        recommendations = "Recommended Analysis Types:\n"
        
        if len(patterns['numeric_columns']) > 0:
            recommendations += "- Statistical analysis: mean, median, std, min, max for numeric columns\n"
            recommendations += "- Distribution analysis: histograms, box plots for numeric data\n"
        
        if len(patterns['categorical_columns']) > 0:
            recommendations += "- Frequency analysis: value_counts() for categorical columns\n"
            recommendations += "- Group-by analysis: aggregate numeric columns by categorical groups\n"
        
        if len(patterns['date_columns']) > 0:
            recommendations += "- Time series analysis: trends over time, resampling, rolling windows\n"
            recommendations += "- Date-based filtering: year, month, quarter extraction\n"
        
        if len(patterns['numeric_columns']) > 1:
            recommendations += "- Correlation analysis: identify relationships between numeric variables\n"
            recommendations += "- Regression analysis: predict one variable from others\n"
        
        documents.append(recommendations)
        
        # Document 6: Common pandas operations
        documents.append("""Common Pandas Operations:
- Filtering: df[df['column'] > value], df[df['column'].isin(list)]
- Grouping: df.groupby('column').agg({'col': ['mean', 'sum', 'count']})
- Sorting: df.sort_values('column', ascending=False)
- Merging: pd.merge(df1, df2, on='key', how='inner/left/right/outer')
- Pivoting: df.pivot_table(values='val', index='idx', columns='col', aggfunc='mean')
- Date operations: df['date'].dt.year, df['date'].dt.month, df['date'].dt.day_name()""")
        
        # Document 7: Visualization best practices
        documents.append("""Visualization Guidelines:
- Line plots: Time series data, trends over time
- Bar plots: Comparing categories, frequency distributions
- Scatter plots: Relationships between two numeric variables
- Histograms: Distribution of a single numeric variable
- Box plots: Distribution and outliers across categories
- Heatmaps: Correlation matrices, pivot tables
- Pie charts: Proportions (use sparingly, bar charts often better)
Always include: title, axis labels, legend (if needed), appropriate colors""")
        
        # Document 8: Statistical analysis methods
        documents.append("""Statistical Analysis Methods:
- Descriptive statistics: df.describe(), df.mean(), df.median(), df.std()
- Correlation analysis: df.corr(), df.corrwith()
- Hypothesis testing: scipy.stats.ttest_ind, chi2_contingency
- Outlier detection: IQR method, Z-score method
- Missing data handling: dropna(), fillna(method='ffill/bfill'), interpolate()
- Normalization: (df - df.mean()) / df.std(), MinMaxScaler""")
        
        # Document 9: Advanced pandas techniques
        documents.append("""Advanced Pandas Techniques:
- Window functions: df.rolling(window=7).mean(), df.expanding().sum()
- Apply custom functions: df.apply(lambda x: custom_func(x)), df.applymap()
- String operations: df['col'].str.contains(), str.upper(), str.split()
- Category optimization: df['col'].astype('category') for memory efficiency
- Multi-indexing: df.set_index(['col1', 'col2']), df.unstack()
- Query method: df.query('column > @variable and other_col == "value"')""")
        
        return documents
    
    def index_data_context(self, df: pd.DataFrame, data_info: str, schema_info: str):
        """Index enhanced data context with intelligent document generation"""
        try:
            # Delete existing collection
=======
    def index_data_context(self, data_info: str, schema_info: str, dataframe: Optional[pd.DataFrame] = None):
        """Intelligently index comprehensive data context for exact answer retrieval"""
        try:
            self.data = dataframe
            
            # Delete existing collection if it exists
>>>>>>> b161ff043a0535b4a953740da79412d61fd73b8a
            try:
                self.chroma_client.delete_collection(self.collection_name)
            except:
                pass
            
            # Create new collection
            self.collection = self.chroma_client.create_collection(
                name=self.collection_name,
<<<<<<< HEAD
                metadata={"description": "Enhanced data analysis context with intelligent insights"}
            )
            
            # Generate smart documents
            documents = self.generate_smart_context_documents(df, schema_info, data_info)
=======
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
>>>>>>> b161ff043a0535b4a953740da79412d61fd73b8a
            
            # Create embeddings
            embeddings = self.create_embeddings(documents)
            
<<<<<<< HEAD
            # Add to collection with metadata
            self.collection.add(
                documents=documents,
                embeddings=embeddings,
                ids=[f"doc_{i}" for i in range(len(documents))],
                metadatas=[{"type": f"context_{i}", "priority": i} for i in range(len(documents))]
=======
            # Add to collection
            self.collection.add(
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids
>>>>>>> b161ff043a0535b4a953740da79412d61fd73b8a
            )
            
            logger.info(f"Indexed {len(documents)} intelligent context documents")
        except Exception as e:
            logger.error(f"Error indexing data context: {e}")
            raise
    
<<<<<<< HEAD
    def retrieve_context(self, query: str, top_k: int = 4) -> List[str]:
        """Retrieve relevant context with intelligent ranking"""
=======
    def retrieve_context(self, query: str, top_k: int = 5) -> List[str]:
        """Intelligently retrieve relevant context with enhanced relevance"""
>>>>>>> b161ff043a0535b4a953740da79412d61fd73b8a
        try:
            if self.collection is None:
                return []
            
            query_embedding = self.create_embeddings([query])[0]
            
<<<<<<< HEAD
            # Retrieve more results than needed for re-ranking
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=min(top_k + 2, 9)
            )
            
            if not results['documents']:
                return []
            
            # Extract documents and distances
            documents = results['documents'][0]
            
            # Apply query-specific boosting
            boosted_docs = self._boost_relevant_docs(query, documents)
            
            return boosted_docs[:top_k]
=======
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
>>>>>>> b161ff043a0535b4a953740da79412d61fd73b8a
        except Exception as e:
            logger.error(f"Error retrieving context: {e}")
            return []
    
<<<<<<< HEAD
    def _boost_relevant_docs(self, query: str, documents: List[str]) -> List[str]:
        """Boost document relevance based on query keywords"""
        query_lower = query.lower()
        
        # Define keyword categories
        keyword_map = {
            "correlation": ["correlation", "relationship", "related", "corr"],
            "visualization": ["plot", "chart", "graph", "visualize", "show", "display"],
            "statistics": ["mean", "average", "median", "std", "statistics", "summary"],
            "grouping": ["group", "aggregate", "by category", "per"],
            "time_series": ["trend", "over time", "time series", "date", "month", "year"],
            "missing": ["missing", "null", "nan", "empty"],
            "outlier": ["outlier", "anomaly", "extreme", "unusual"]
        }
        
        # Score documents based on keyword matches
        doc_scores = []
        for doc in documents:
            score = 0
            doc_lower = doc.lower()
            
            for category, keywords in keyword_map.items():
                if any(kw in query_lower for kw in keywords):
                    if category in doc_lower or any(kw in doc_lower for kw in keywords):
                        score += 2
            
            doc_scores.append((doc, score))
        
        # Sort by score (descending) and return documents
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, score in doc_scores]
    
    def get_data_insights(self) -> Dict:
        """Return analyzed data patterns"""
        return self.data_stats
=======
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
>>>>>>> b161ff043a0535b4a953740da79412d61fd73b8a
