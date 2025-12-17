import sys
import io
import traceback
from contextlib import redirect_stdout, redirect_stderr
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
<<<<<<< HEAD
from typing import Tuple, Any, Dict, List
import logging
import time
import ast
import re
from datetime import datetime

logger = logging.getLogger(__name__)

class CodeExecutor:
    """Safely executes Python/Pandas code with enhanced security and monitoring"""
=======
from typing import Tuple, Any
import logging

logger = logging.getLogger(__name__)
class CodeExecutor:
    """Safely executes Python/Pandas code for data analysis"""
>>>>>>> b161ff043a0535b4a953740da79412d61fd73b8a
    
    def __init__(self, data: pd.DataFrame, output_dir: str):
        self.data = data
        self.output_dir = output_dir
        self.execution_count = 0
<<<<<<< HEAD
        self.execution_history = []
        
        # Set visualization styles
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 6)
        plt.rcParams['font.size'] = 10
        
        logger.info(f"CodeExecutor initialized with data shape: {data.shape}")
    
    def execute(self, code: str, timeout: int = 30) -> Tuple[bool, str, Any, str]:
        """
        Execute Python code with comprehensive safety checks and monitoring
        Returns: (success, output, result, plot_filename)
        """
        self.execution_count += 1
        execution_start = time.time()
        
        # Validate code first
        is_valid, validation_msg = self.validate_code(code)
        if not is_valid:
            logger.warning(f"Code validation failed: {validation_msg}")
            return False, f"Security Error: {validation_msg}", None, None
        
        # Check for syntax errors
        try:
            ast.parse(code)
        except SyntaxError as e:
            error_msg = f"Syntax Error: {str(e)}"
            logger.error(f"Syntax error in code: {error_msg}")
            return False, error_msg, None, None
        
        # Prepare secure execution environment
        local_vars = self._prepare_execution_environment()
        
        # Capture output
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        plot_filename = None
        result = None
        
        try:
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                # Execute code with timeout protection
                exec(code, {"__builtins__": __builtins__}, local_vars)
                
                # Handle matplotlib plots
                if plt.get_fignums():
                    plot_filename = f"plot_{self.execution_count}_{int(time.time())}.png"
                    plot_path = f"{self.output_dir}/{plot_filename}"
                    plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
                    plt.close('all')
                    logger.info(f"Saved matplotlib plot to {plot_path}")
                
                # Handle plotly plots
                if 'fig' in local_vars and hasattr(local_vars['fig'], 'write_html'):
                    plot_filename = f"plot_{self.execution_count}_{int(time.time())}.html"
                    plot_path = f"{self.output_dir}/{plot_filename}"
                    local_vars['fig'].write_html(plot_path)
                    logger.info(f"Saved plotly plot to {plot_path}")
            
            # Capture output and result
            output = stdout_capture.getvalue()
            stderr_output = stderr_capture.getvalue()
            
            if stderr_output:
                output += f"\nWarnings/Errors:\n{stderr_output}"
            
            # Get result variable if it exists
            result = local_vars.get('result', None)
            
            # Convert result to serializable format
            result = self._serialize_result(result)
            
            execution_time = time.time() - execution_start
            
            # Log execution
            self._log_execution(code, True, output, execution_time)
            
            logger.info(f"Code execution #{self.execution_count} successful in {execution_time:.2f}s")
            return True, output, result, plot_filename
            
        except Exception as e:
            execution_time = time.time() - execution_start
            error_msg = f"Execution Error: {str(e)}\n\n{traceback.format_exc()}"
            
            # Enhanced error message with hints
            error_msg = self._enhance_error_message(error_msg, code)
            
            self._log_execution(code, False, error_msg, execution_time)
            
            logger.error(f"Code execution #{self.execution_count} failed: {str(e)}")
            return False, error_msg, None, None
        
        finally:
            # Cleanup
            plt.close('all')
    
    def _prepare_execution_environment(self) -> Dict:
        """Prepare a secure execution environment with necessary imports"""
        return {
=======
    
    def execute(self, code: str) -> Tuple[bool, str, Any, str]:
        """
        Execute Python code and return results
        Returns: (success, output, result, plot_filename)
        """
        self.execution_count += 1
        
        # Prepare execution environment
        local_vars = {
>>>>>>> b161ff043a0535b4a953740da79412d61fd73b8a
            'pd': pd,
            'np': np,
            'plt': plt,
            'sns': sns,
            'px': px,
            'go': go,
            'df': self.data.copy(),
<<<<<<< HEAD
            'data': self.data.copy(),
            'datetime': datetime,
            're': re,
            # Add safe utility functions
            'print': print,
            'len': len,
            'sum': sum,
            'max': max,
            'min': min,
            'sorted': sorted,
            'enumerate': enumerate,
            'zip': zip,
            'range': range,
            'list': list,
            'dict': dict,
            'set': set,
            'tuple': tuple,
            'str': str,
            'int': int,
            'float': float,
            'bool': bool,
        }
    
    def validate_code(self, code: str) -> Tuple[bool, str]:
        """Enhanced code validation with comprehensive security checks"""
        
        # Dangerous patterns to block
        dangerous_patterns = [
            (r'\bimport\s+os\b', "Importing 'os' module is not allowed"),
            (r'\bimport\s+sys\b', "Importing 'sys' module is not allowed"),
            (r'\bimport\s+subprocess\b', "Importing 'subprocess' module is not allowed"),
            (r'\bimport\s+socket\b', "Importing 'socket' module is not allowed"),
            (r'\b__import__\s*\(', "Dynamic imports are not allowed"),
            (r'\beval\s*\(', "Use of eval() is not allowed"),
            (r'\bexec\s*\(', "Nested exec() is not allowed"),
            (r'\bopen\s*\(', "File operations are restricted"),
            (r'\bfile\s*\(', "File operations are restricted"),
            (r'\binput\s*\(', "User input is not allowed"),
            (r'\braw_input\s*\(', "User input is not allowed"),
            (r'\bcompile\s*\(', "Code compilation is not allowed"),
            (r'__builtins__', "Access to __builtins__ is restricted"),
            (r'__globals__', "Access to __globals__ is restricted"),
            (r'__locals__', "Access to __locals__ is restricted"),
            (r'\bgetattr\s*\(', "Dynamic attribute access is restricted"),
            (r'\bsetattr\s*\(', "Dynamic attribute modification is restricted"),
            (r'\bdelattr\s*\(', "Dynamic attribute deletion is restricted"),
            (r'\bglobals\s*\(', "Access to globals() is restricted"),
            (r'\blocals\s*\(', "Access to locals() is restricted"),
            (r'\bvars\s*\(', "Access to vars() is restricted"),
            (r'\bdir\s*\(', "Access to dir() is restricted for security"),
        ]
        
        code_lower = code.lower()
        
        # Check each dangerous pattern
        for pattern, message in dangerous_patterns:
            if re.search(pattern, code, re.IGNORECASE):
                return False, message
        
        # Check for suspicious keywords
        suspicious_keywords = ['shell', 'system', 'popen', 'spawn']
        for keyword in suspicious_keywords:
            if keyword in code_lower:
                return False, f"Suspicious keyword '{keyword}' detected"
        
        # Validate imports are only from allowed modules
        allowed_imports = {
            'pandas', 'pd', 'numpy', 'np', 'matplotlib', 'pyplot', 'plt',
            'seaborn', 'sns', 'plotly', 'express', 'px', 'graph_objects', 'go',
            'datetime', 're', 'math', 'statistics', 'collections', 'itertools',
            'scipy', 'sklearn', 'json'
        }
        
        # Find all import statements
        import_pattern = r'\bimport\s+(\w+)|from\s+(\w+)\s+import'
        imports = re.findall(import_pattern, code)
        
        for imp in imports:
            module = imp[0] or imp[1]
            if module and module not in allowed_imports:
                return False, f"Import of '{module}' module is not allowed"
        
        return True, "Code validation passed"
    
    def _serialize_result(self, result: Any) -> Any:
        """Convert result to JSON-serializable format"""
        if result is None:
            return None
        
        # Handle pandas objects
        if isinstance(result, pd.DataFrame):
            return {
                "type": "dataframe",
                "data": result.head(100).to_dict(orient='records'),
                "shape": result.shape,
                "columns": list(result.columns)
            }
        elif isinstance(result, pd.Series):
            return {
                "type": "series",
                "data": result.head(100).to_dict(),
                "length": len(result)
            }
        
        # Handle numpy arrays
        elif isinstance(result, np.ndarray):
            return {
                "type": "ndarray",
                "data": result.tolist()[:100],
                "shape": result.shape
            }
        
        # Handle numeric types
        elif isinstance(result, (np.integer, np.floating)):
            return float(result)
        
        # Handle dictionaries
        elif isinstance(result, dict):
            return {k: self._serialize_result(v) for k, v in list(result.items())[:100]}
        
        # Handle lists
        elif isinstance(result, list):
            return [self._serialize_result(item) for item in result[:100]]
        
        # Handle basic types
        elif isinstance(result, (str, int, float, bool)):
            return result
        
        # Fallback to string representation
        else:
            return str(result)[:1000]
    
    def _enhance_error_message(self, error_msg: str, code: str) -> str:
        """Add helpful hints to error messages"""
        enhanced_msg = error_msg
        
        # Common error patterns and hints
        error_hints = {
            "KeyError": "Hint: Check that the column name exists and is spelled correctly (case-sensitive).",
            "AttributeError": "Hint: Verify that you're using the correct method for this data type.",
            "TypeError": "Hint: Check data types - you may need to convert types using astype().",
            "ValueError": "Hint: Check if values are in the expected format or range.",
            "IndexError": "Hint: Check array/list bounds - index may be out of range.",
            "NameError": "Hint: Variable may not be defined or column name needs quotes.",
            "SyntaxError": "Hint: Check for missing parentheses, brackets, or quotes.",
            "ZeroDivisionError": "Hint: Add a check to prevent division by zero.",
            "MemoryError": "Hint: Dataset may be too large - try filtering or sampling first.",
        }
        
        for error_type, hint in error_hints.items():
            if error_type in error_msg:
                enhanced_msg += f"\n\n{hint}"
                break
        
        # Check for common mistakes in code
        if "KeyError" in error_msg and "'" not in code.count("df["):
            enhanced_msg += "\n\nTip: Use df['column_name'] with quotes for column access."
        
        if "datetime" in error_msg.lower():
            enhanced_msg += "\n\nTip: Convert date columns using pd.to_datetime(df['date_column'])"
        
        return enhanced_msg
    
    def _log_execution(self, code: str, success: bool, output: str, execution_time: float):
        """Log execution history for debugging and analysis"""
        self.execution_history.append({
            "execution_number": self.execution_count,
            "timestamp": datetime.now().isoformat(),
            "code": code[:500],  # Truncate long code
            "success": success,
            "output": output[:1000],  # Truncate long output
            "execution_time": execution_time
        })
        
        # Keep only last 50 executions
        if len(self.execution_history) > 50:
            self.execution_history.pop(0)
    
    def get_execution_stats(self) -> Dict:
        """Get statistics about code executions"""
        if not self.execution_history:
            return {
                "total_executions": 0,
                "successful_executions": 0,
                "failed_executions": 0,
                "success_rate": 0,
                "average_execution_time": 0
            }
        
        successful = sum(1 for ex in self.execution_history if ex['success'])
        total = len(self.execution_history)
        avg_time = sum(ex['execution_time'] for ex in self.execution_history) / total
        
        return {
            "total_executions": total,
            "successful_executions": successful,
            "failed_executions": total - successful,
            "success_rate": round(successful / total * 100, 2),
            "average_execution_time": round(avg_time, 3)
        }
    
    def get_execution_history(self, limit: int = 10) -> List[Dict]:
        """Get recent execution history"""
        return self.execution_history[-limit:]
=======
            'data': self.data.copy()
        }
        
        # Capture output
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        plot_filename = None
        
        try:
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                # Execute code
                exec(code, local_vars)
                
                # Save any matplotlib figures
                if plt.get_fignums():
                    plot_filename = f"plot_{self.execution_count}.png"
                    plot_path = f"{self.output_dir}/{plot_filename}"
                    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                    plt.close('all')
                    logger.info(f"Saved plot to {plot_path}")
            
            output = stdout_capture.getvalue()
            result = local_vars.get('result', None)
            
            logger.info(f"Code execution #{self.execution_count} successful")
            return True, output, result, plot_filename
            
        except Exception as e:
            error_msg = f"Error: {str(e)}\n{traceback.format_exc()}"
            logger.error(f"Code execution #{self.execution_count} failed: {error_msg}")
            return False, error_msg, None, None
    
    def validate_code(self, code: str) -> Tuple[bool, str]:
        """Validate code before execution"""
        dangerous_patterns = [
            'import os',
            'import sys',
            'import subprocess',
            '__import__',
            'eval(',
            'exec(',
            'open(',
            'file(',
            'input(',
            'raw_input(',
            'compile(',
            '__builtins__'
        ]
        
        code_lower = code.lower()
        for pattern in dangerous_patterns:
            if pattern in code_lower:
                return False, f"Dangerous operation detected: {pattern}"
        
        return True, "Code validation passed"
>>>>>>> b161ff043a0535b4a953740da79412d61fd73b8a
