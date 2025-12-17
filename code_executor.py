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
from typing import Tuple, Any
import logging

logger = logging.getLogger(__name__)
class CodeExecutor:
    """Safely executes Python/Pandas code for data analysis"""
    
    def __init__(self, data: pd.DataFrame, output_dir: str):
        self.data = data
        self.output_dir = output_dir
        self.execution_count = 0
    
    def execute(self, code: str) -> Tuple[bool, str, Any, str]:
        """
        Execute Python code and return results
        Returns: (success, output, result, plot_filename)
        """
        self.execution_count += 1
        
        # Prepare execution environment
        local_vars = {
            'pd': pd,
            'np': np,
            'plt': plt,
            'sns': sns,
            'px': px,
            'go': go,
            'df': self.data.copy(),
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