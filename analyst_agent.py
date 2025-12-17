import google.generativeai as genai
import re
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)

class AnalystAgent:
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

Dataset Context:
{data_context}

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
    
    def extract_code(self, response: str) -> List[str]:
        """Extract Python code blocks from response"""
        pattern = r'```python\n(.*?)```'
        matches = re.findall(pattern, response, re.DOTALL)
        return matches
    
    def self_correct(self, code: str, error: str, query: str, data_context: str) -> str:
        """Generate corrected code based on error"""
        
        correction_prompt = f"""The following code produced an error. Please fix it.

Original Query: {query}

Dataset Context:
{data_context}

Code that failed:
```python
{code}
```

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

        try:
            response = self.model.generate_content(
                correction_prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0,
                    max_output_tokens=8192,
                )
            )
            
            return response.text
        except Exception as e:
            logger.error(f"Error in self-correction: {e}")
            raise