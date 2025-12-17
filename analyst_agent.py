import google.generativeai as genai
import re
<<<<<<< HEAD
from typing import Dict, Any, List, Tuple, Optional
import logging
import json
=======
from typing import Dict, Any, List, Optional
import logging
>>>>>>> b161ff043a0535b4a953740da79412d61fd73b8a

logger = logging.getLogger(__name__)

class AnalystAgent:
<<<<<<< HEAD
    """Enhanced AI agent with advanced reasoning and self-correction capabilities"""
    
    def __init__(self, api_key: str, model: str = "gemini-2.5-flash"):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model)
        self.conversation_history = []
        self.analysis_cache = {}
        logger.info(f"Analyst Agent initialized with model: {model}")
    
    def analyze_query(self, query: str, data_context: str, retrieved_context: List[str],
                     data_profile: Dict = None) -> str:
        """Analyze user query with enhanced context and reasoning"""
        
        # Build comprehensive context
        context_str = self._build_context_string(retrieved_context, data_profile)
        
        # Detect query intent
        query_intent = self._detect_query_intent(query)
        
        # Build specialized prompt based on intent
        prompt = self._build_specialized_prompt(
            query=query,
            data_context=data_context,
            context_str=context_str,
            query_intent=query_intent,
            data_profile=data_profile
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
                                 data_profile: Dict = None) -> str:
        """Build specialized prompt based on query intent"""
        
        base_prompt = f"""You are an expert data scientist and analyst with deep knowledge of Python, Pandas, and data visualization.

{context_str}
=======
    """AI agent that analyzes data and generates code using Google Gemini"""
    
    def __init__(self, api_key: str, model: str = "gemini-1.5-pro"):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model)
        self.conversation_history = []
        logger.info(f"Analyst Agent initialized with model: {model}")
    
    def analyze_query(self, query: str, data_context: str, retrieved_context: List[str], direct_answer: Optional[str] = None) -> str:
        """Analyze user query with intelligent context and provide exact answers"""
        
        context_str = "\n\n".join(retrieved_context) if retrieved_context else "No additional context"
        
        direct_answer_section = ""
        if direct_answer:
            direct_answer_section = f"\n\nDIRECT ANSWER FROM DATA:\n{direct_answer}\n\nUse this information to provide an exact answer in your response."
        
        prompt = f"""You are an expert data analyst with deep knowledge of Python, Pandas, and data visualization. Your goal is to provide EXACT, ACCURATE answers based on the actual data.
>>>>>>> b161ff043a0535b4a953740da79412d61fd73b8a

Dataset Context:
{data_context}

<<<<<<< HEAD
User Query: {query}

"""
        
        # Add intent-specific guidance
        if query_intent.get("visualization"):
            base_prompt += """
VISUALIZATION GUIDANCE:
- Choose the right plot type for the data and question
- Use clear titles, labels, and legends
- Apply appropriate color schemes
- Consider using seaborn or plotly for professional aesthetics
- For time series: line plots
- For distributions: histograms, box plots, violin plots
- For categories: bar plots, count plots
- For relationships: scatter plots, pair plots
- For correlations: heatmaps
"""
        
        if query_intent.get("time_series"):
            base_prompt += """
TIME SERIES GUIDANCE:
- Always verify date columns are datetime type first
- Use pd.to_datetime() if conversion needed
- Extract components using .dt accessor (year, month, day, etc.)
- Consider resampling for aggregation ('D', 'W', 'M', 'Y')
- Use rolling windows for smoothing trends
- Handle missing dates appropriately
"""
        
        if query_intent.get("grouping"):
            base_prompt += """
GROUPING GUIDANCE:
- Use df.groupby() with appropriate aggregation functions
- Consider multiple grouping columns for hierarchical analysis
- Use .agg() for multiple aggregations
- Reset index after groupby for easier manipulation
- Sort results for better insights
"""
        
        if query_intent.get("correlation"):
            base_prompt += """
CORRELATION GUIDANCE:
- Select only numeric columns for correlation
- Use df.corr() for correlation matrix
- Visualize with heatmap (sns.heatmap or px.imshow)
- Interpret correlation values: |r| > 0.7 is strong, 0.3-0.7 is moderate
- Remember: correlation does not imply causation
"""
        
        if query_intent.get("missing_data"):
            base_prompt += """
MISSING DATA GUIDANCE:
- Use df.isnull().sum() or df.isna().sum() to count missing values
- Calculate percentages: (df.isnull().sum() / len(df)) * 100
- Visualize with heatmap: sns.heatmap(df.isnull())
- Consider imputation strategies if needed
"""
        
        if query_intent.get("outliers"):
            base_prompt += """
OUTLIER DETECTION GUIDANCE:
- Use IQR method: Q1 - 1.5*IQR and Q3 + 1.5*IQR
- Visualize with box plots
- Use z-score method for normally distributed data
- Consider domain knowledge before removing outliers
"""
        
        # Add core instructions
        base_prompt += """

Your task is to:
1. Understand the user's request thoroughly
2. Plan the analysis approach step by step
3. Generate clean, efficient, well-commented Python code
4. Create appropriate visualizations when needed
5. Provide clear insights and interpretation

CODE REQUIREMENTS:
- Use 'df' as the DataFrame variable name
- Include detailed comments explaining each step
- Store final results in a 'result' variable
- Use matplotlib/seaborn for static plots, plotly for interactive
- Handle potential errors with try-except where appropriate
- ALWAYS verify data types before operations
- Use .copy() when modifying dataframes to avoid SettingWithCopyWarning
- Print key insights, statistics, and findings
- Format output nicely with clear labels

DATA TYPE HANDLING (CRITICAL):
- Check column data types before operations: df.dtypes
- For date operations, ensure datetime: df['date'] = pd.to_datetime(df['date'], errors='coerce')
- For numeric operations, ensure numeric type: df['col'] = pd.to_numeric(df['col'], errors='coerce')
- For filtering by year: df['date'].dt.year == 2023
- Handle missing values before aggregations

ERROR PREVENTION:
- Check if columns exist before using them
- Handle division by zero
- Verify non-empty dataframes before operations
- Use .notna() or .dropna() when needed

Always wrap your Python code in ```python code blocks.

Provide your analysis in this structure:
1. Brief explanation of the approach
2. The Python code to perform the analysis
3. Expected insights or interpretation (after seeing results)
"""
        
        return base_prompt
=======
Retrieved Knowledge (Use this for exact column names, data types, and statistics):
{context_str}
{direct_answer_section}

User Query: {query}

CRITICAL INSTRUCTIONS:
1. **Provide Exact Answers First**: Before generating code, use the retrieved context to give a direct, exact answer to the user's question
2. **Use Actual Column Names**: The retrieved context contains the EXACT column names in the dataset - use them precisely (case-sensitive)
3. **Reference Actual Statistics**: If the context contains relevant statistics (min, max, mean, counts), mention them in your answer
4. **Be Specific**: Give concrete numbers, values, and facts from the data context
5. **Then Generate Code**: After providing the answer, generate code to verify or visualize it

Code Guidelines:
- Use 'df' as the DataFrame variable name
- Use EXACT column names from the retrieved context (they are case-sensitive)
- Include comments explaining each step
- Store final results in a 'result' variable
- Print the exact answer/result clearly
- Use matplotlib/seaborn for static plots, plotly for interactive
- Handle errors gracefully with try-except
- Always check for missing values before operations
- Use .copy() when modifying dataframes
- For date columns: Check dtype first, convert with pd.to_datetime() if needed
- For filtering by year: Use df['Date'].dt.year after ensuring datetime type
- IMPORTANT: Verify column names match exactly what's in the context

Response Format:
1. Start with a direct answer using information from the context
2. Explain what you found
3. Provide Python code in ```python code blocks to verify/visualize
4. Print the exact results in the code

Example:
"Based on the dataset, [exact answer from context]. The [column] has [specific value/statistic]. Here's the code to verify:"

Always wrap your Python code in ```python code blocks."""

        try:
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0,
                    max_output_tokens=8192,
                )
            )
            
            return response.text
        except Exception as e:
            logger.error(f"Error calling Gemini API: {e}")
            raise
>>>>>>> b161ff043a0535b4a953740da79412d61fd73b8a
    
    def extract_code(self, response: str) -> List[str]:
        """Extract Python code blocks from response"""
        pattern = r'```python\n(.*?)```'
        matches = re.findall(pattern, response, re.DOTALL)
<<<<<<< HEAD
        
        # Clean up code
        cleaned_matches = []
        for code in matches:
            # Remove any common issues
            code = code.strip()
            cleaned_matches.append(code)
        
        return cleaned_matches
    
    def self_correct(self, code: str, error: str, query: str, 
                    data_context: str, attempt: int = 1) -> str:
        """Generate corrected code with enhanced error analysis"""
        
        # Analyze the error
        error_analysis = self._analyze_error(error)
        
        correction_prompt = f"""You are debugging code that produced an error. Analyze carefully and fix it.
=======
        return matches
    
    def self_correct(self, code: str, error: str, query: str, data_context: str) -> str:
        """Generate corrected code based on error"""
        
        correction_prompt = f"""The following code produced an error. Please fix it.
>>>>>>> b161ff043a0535b4a953740da79412d61fd73b8a

Original Query: {query}

Dataset Context:
{data_context}

<<<<<<< HEAD
Code that failed (Attempt #{attempt}):
=======
Code that failed:
>>>>>>> b161ff043a0535b4a953740da79412d61fd73b8a
```python
{code}
```

<<<<<<< HEAD
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
=======
Error:
{error}

Please analyze the error and provide corrected code that:
1. Fixes the specific error
2. Achieves the original goal
3. Is robust and handles edge cases
4. Includes error handling where appropriate

Common issues to check:
- Column names (case-sensitive, spaces)
- Data types (numeric vs string)
- Missing values
- Index alignment
- Division by zero

Provide ONLY the corrected Python code in a ```python code block with brief explanation of what was fixed."""
>>>>>>> b161ff043a0535b4a953740da79412d61fd73b8a

        try:
            response = self.model.generate_content(
                correction_prompt,
                generation_config=genai.types.GenerationConfig(
<<<<<<< HEAD
                    temperature=0.1,
=======
                    temperature=0,
>>>>>>> b161ff043a0535b4a953740da79412d61fd73b8a
                    max_output_tokens=8192,
                )
            )
            
            return response.text
<<<<<<< HEAD
            
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
=======
        except Exception as e:
            logger.error(f"Error in self-correction: {e}")
            raise
>>>>>>> b161ff043a0535b4a953740da79412d61fd73b8a
