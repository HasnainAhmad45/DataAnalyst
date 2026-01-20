import google.generativeai as genai
import re
from typing import Dict, Any, List, Tuple, Optional
import logging
import json

logger = logging.getLogger(__name__)

class AnalystAgent:
    """Enhanced AI agent with advanced reasoning and self-correction capabilities"""
    
    def __init__(self, api_key: str, model: str = "gemini-2.5-flash"):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model)
        self.conversation_history = []
        self.analysis_cache = {}
        logger.info(f"Analyst Agent initialized with model: {model}")
    
    def analyze_query(self, query: str, data_context: str, retrieved_context: List[str], 
                     data_profile: Dict = None, source_name: str = "Unknown Source") -> str:
        """Analyze user query and generate analysis plan + code"""
        
        # Build comprehensive context
        context_str = "\n\n".join(retrieved_context) if retrieved_context else "No specific context retrieved."
        
        # Detect query intent
        query_intent = self._detect_query_intent(query)
        
        # Build specialized prompt based on intent
        prompt = self._build_specialized_prompt(
            query=query,
            data_context=data_context,
            context_str=context_str,
            query_intent=query_intent,
            data_profile=data_profile,
            source_name=source_name
        )
        
        try:
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1,  # Low temperature for consistent code generation
                    max_output_tokens=8192,
                    top_p=0.95,
                    top_k=40
                )
            )
            
            # Store in conversation history
            self.conversation_history.append({
                "query": query,
                "response": response.text,
                "intent": query_intent
            })
            
            return response.text
            
        except Exception as e:
            logger.error(f"Error calling Gemini API: {e}")
            raise
    
    def _build_context_string(self, retrieved_context: List[str], 
                             data_profile: Dict = None) -> str:
        """Build comprehensive context string"""
        context_parts = []
        
        # Add retrieved RAG context
        if retrieved_context:
            context_parts.append("=== RELEVANT KNOWLEDGE ===")
            for i, ctx in enumerate(retrieved_context, 1):
                context_parts.append(f"\n[Context {i}]\n{ctx}")
        
        # Add data profile insights
        if data_profile:
            context_parts.append("\n=== DATA PROFILE ===")
            
            if data_profile.get("numeric_columns"):
                context_parts.append(f"Numeric Columns: {', '.join(data_profile['numeric_columns'][:10])}")
            
            if data_profile.get("categorical_columns"):
                context_parts.append(f"Categorical Columns: {', '.join(data_profile['categorical_columns'][:10])}")
            
            if data_profile.get("date_columns"):
                context_parts.append(f"Date Columns: {', '.join(data_profile['date_columns'])}")
            
            # Add summary stats for key columns
            if data_profile.get("summary_stats"):
                context_parts.append("\nKey Statistics:")
                for col, stats in list(data_profile["summary_stats"].items())[:5]:
                    if "mean" in stats and stats["mean"] is not None:
                        context_parts.append(
                            f"  {col}: mean={stats['mean']:.2f}, "
                            f"range=[{stats['min']:.2f}, {stats['max']:.2f}]"
                        )
        
        return "\n".join(context_parts)
    
    def _detect_query_intent(self, query: str) -> Dict[str, bool]:
        """Detect the intent of the user's query"""
        query_lower = query.lower()
        
        intent = {
            "visualization": any(kw in query_lower for kw in 
                ['plot', 'chart', 'graph', 'visualiz', 'show', 'display', 'draw']),
            
            "statistics": any(kw in query_lower for kw in 
                ['mean', 'average', 'median', 'std', 'statistics', 'summary', 'describe']),
            
            "correlation": any(kw in query_lower for kw in 
                ['correlation', 'corr', 'relationship', 'relate', 'associated']),
            
            "grouping": any(kw in query_lower for kw in 
                ['group', 'by', 'per', 'each', 'aggregate', 'sum by', 'count by']),
            
            "filtering": any(kw in query_lower for kw in 
                ['filter', 'where', 'only', 'exclude', 'select', 'subset']),
            
            "time_series": any(kw in query_lower for kw in 
                ['trend', 'over time', 'time series', 'temporal', 'year', 'month', 'date']),
            
            "comparison": any(kw in query_lower for kw in 
                ['compare', 'difference', 'vs', 'versus', 'between', 'contrast']),
            
            "ranking": any(kw in query_lower for kw in 
                ['top', 'bottom', 'highest', 'lowest', 'best', 'worst', 'rank']),
            
            "missing_data": any(kw in query_lower for kw in 
                ['missing', 'null', 'nan', 'empty', 'blank']),
            
            "outliers": any(kw in query_lower for kw in 
                ['outlier', 'anomaly', 'unusual', 'extreme', 'abnormal'])
        }
        
        return intent
    
    def _build_specialized_prompt(self, query: str, data_context: str, 
                                 context_str: str, query_intent: Dict,
                                 data_profile: Dict = None, source_name: str = "Unknown Source") -> str:
        """Build specialized prompt based on query intent"""
        
        base_prompt = f"""You are an expert Data Analyst AI working inside a Retrieval-Augmented Generation (RAG) system.

The active data source you are analyzing is: **{source_name}**

Your task is to answer user queries written in natural language by retrieving and analyzing data from:
1. A structured database (SQL tables)
2. CSV files
3. PDF documents

Follow these rules strictly:

1. **Cite your source**: Always start or end your response by explicitly mentioning that you are using data from '{source_name}'.
2. Understand the user’s question and identify:
   - Required metrics (count, sum, average, trends, comparison, etc.)
   - Relevant columns, tables, or documents

2. Retrieve ONLY the most relevant data chunks using semantic similarity.
   - Use embeddings to fetch matching rows, CSV sections, or PDF text
   - Do NOT hallucinate or assume missing data

3. If the query requires:
   - Numerical analysis → perform calculations correctly
   - Filtering → apply exact conditions
   - Aggregation → use correct statistical operations
   - Comparison → clearly compare values
   - Trends → analyze changes over time

4. If data is insufficient or missing:
   - Clearly state: "Insufficient data to answer the query"

5. Return the final answer in clear, simple natural language.
   - Include tables or bullet points if helpful
   - Keep explanations concise and accurate
   - Mention the data source used (Database / CSV / PDF)

6. NEVER expose:
   - SQL queries
   - Embedding logic
   - Internal retrieval steps

7. Be deterministic, factual, and professional.

User Query:
{query}

Retrieved Context:
{context_str}

Dataset Context:
{data_context}

---
IMPORTANT: To perform the analysis, you MUST generate executable Python code.
- Wrap your code in ```python blocks.
- The code will be executed in an environment with pandas (pd), numpy (np), matplotlib.pyplot (plt), seaborn (sns), and plotly.express (px).
- The dataset is available as a pandas DataFrame named `df`.
- Save any requested plots to files (the system handles display).
- Store the FINAL ANSWER inside the code as a variable named `result` (if it's a number/table) or print it.
"""
        return base_prompt
    
    def extract_code(self, response: str) -> List[str]:
        """Extract Python code blocks from response"""
        pattern = r'```python\n(.*?)```'
        matches = re.findall(pattern, response, re.DOTALL)
        
        # Clean up code
        cleaned_matches = []
        for code in matches:
            # Remove any common issues
            code = code.strip()
            cleaned_matches.append(code)
        
        return cleaned_matches
    
    def self_correct(self, code: str, error: str, query: str, 
                    data_context: str, attempt: int = 1, source_name: str = "Unknown Source") -> str:
        """Generate corrected code with enhanced error analysis"""
        
        # Analyze the error
        error_analysis = self._analyze_error(error)
        
        correction_prompt = f"""You are debugging code that produced an error. Analyze carefully and fix it.

The active data source is: **{source_name}**

Follow these rules:
1. **Cite your source**: Always mention that you are analyzing data from '{source_name}'.
2. Provide a clear explanation of what went wrong.
3. Provide the corrected Python code block.

Original Query: {query}

Dataset Context:
{data_context}

Code that failed (Attempt #{attempt}):
```python
{code}
```

Error Message:
{error}

Error Analysis:
{error_analysis}

DEBUGGING CHECKLIST:
1. Column names: Are they spelled correctly? Case-sensitive? Do they exist?
2. Data types: Are you using the right operations for the data type?
3. Missing values: Could NaN/None values be causing issues?
4. Index alignment: Are you working with aligned indices?
5. Empty results: Could a filter be returning an empty DataFrame?
6. Division by zero: Are denominators non-zero?
7. Date operations: Are date columns in datetime format?
8. Syntax: Are parentheses, brackets, quotes balanced?

COMMON FIXES:
- KeyError: Check df.columns to see exact column names
- TypeError: Check df.dtypes and convert types if needed
- AttributeError: Verify the method exists for that data type
- ValueError: Check if values are in expected format/range
- IndexError: Verify list/array has enough elements

Please provide:
1. Brief explanation of what caused the error
2. The corrected Python code
3. What you changed and why

Provide ONLY the corrected Python code in a ```python code block."""

        try:
            response = self.model.generate_content(
                correction_prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1,
                    max_output_tokens=8192,
                )
            )
            
            return response.text
            
        except Exception as e:
            logger.error(f"Error in self-correction: {e}")
            raise
    
    def _analyze_error(self, error: str) -> str:
        """Provide intelligent error analysis"""
        error_lower = error.lower()
        analysis_parts = []
        
        # Categorize and provide guidance
        if "keyerror" in error_lower:
            analysis_parts.append("❌ KeyError: Column name doesn't exist or is misspelled")
            analysis_parts.append("→ Check column names are exact (case-sensitive)")
            analysis_parts.append("→ Use df.columns to list all available columns")
        
        elif "typeerror" in error_lower:
            analysis_parts.append("❌ TypeError: Operation not supported for this data type")
            analysis_parts.append("→ Check data types with df.dtypes")
            analysis_parts.append("→ Convert types using astype() or pd.to_numeric()")
        
        elif "valueerror" in error_lower:
            analysis_parts.append("❌ ValueError: Value is not in expected format or range")
            analysis_parts.append("→ Verify input values are valid")
            analysis_parts.append("→ Check for NaN/None values that need handling")
        
        elif "attributeerror" in error_lower:
            analysis_parts.append("❌ AttributeError: Method/attribute doesn't exist")
            analysis_parts.append("→ Verify correct method name for the object type")
            analysis_parts.append("→ Check if you're calling a method on the right object")
        
        elif "indexerror" in error_lower:
            analysis_parts.append("❌ IndexError: Index out of range")
            analysis_parts.append("→ Check array/list bounds")
            analysis_parts.append("→ Verify data isn't empty after filtering")
        
        elif "zerodivisionerror" in error_lower:
            analysis_parts.append("❌ ZeroDivisionError: Division by zero")
            analysis_parts.append("→ Add check to prevent division by zero")
            analysis_parts.append("→ Filter out zero values before division")
        
        elif "datetime" in error_lower or "date" in error_lower:
            analysis_parts.append("❌ Date/Time Error: Issue with date operations")
            analysis_parts.append("→ Convert to datetime: pd.to_datetime(df['date'])")
            analysis_parts.append("→ Use .dt accessor for date operations")
        
        else:
            analysis_parts.append("❌ Error detected - analyzing context...")
        
        # Extract specific error details
        if "'" in error and "'" in error[error.find("'")+1:]:
            # Try to extract the problematic value
            match = re.search(r"'([^']*)'", error)
            if match:
                analysis_parts.append(f"→ Problem with: '{match.group(1)}'")
        
        return "\n".join(analysis_parts)
    
    def suggest_next_steps(self, query: str, result: Any) -> List[str]:
        """Suggest logical next analysis steps"""
        suggestions = []
        query_lower = query.lower()
        
        # Based on what was done, suggest follow-ups
        if "correlation" in query_lower:
            suggestions.append("Visualize top correlations with a heatmap")
            suggestions.append("Investigate causal relationships for strong correlations")
        
        elif any(kw in query_lower for kw in ["plot", "chart", "visualize"]):
            suggestions.append("Add statistical annotations to the plot")
            suggestions.append("Create interactive version with plotly")
        
        elif "group" in query_lower:
            suggestions.append("Visualize the grouped results with a bar chart")
            suggestions.append("Perform statistical tests between groups")
        
        elif "trend" in query_lower or "time" in query_lower:
            suggestions.append("Add moving averages to smooth the trend")
            suggestions.append("Forecast future values using regression")
        
        # General suggestions
        suggestions.append("Filter the data by specific criteria")
        suggestions.append("Compare with other time periods or categories")
        
        return suggestions[:3]  # Return top 3 suggestions
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []
        logger.info("Conversation history cleared")
    
    def get_history(self, limit: int = 5) -> List[Dict]:
        """Get recent conversation history"""
        return self.conversation_history[-limit:]