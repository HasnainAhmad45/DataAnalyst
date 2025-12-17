import pytest
import pandas as pd
import numpy as np
from data_loader import DataLoader
from code_executor import CodeExecutor
import tempfile
import os

def test_data_loader_csv():
    """Test CSV loading"""
    # Create temporary CSV
    df = pd.DataFrame({
        'A': [1, 2, 3],
        'B': ['x', 'y', 'z']
    })
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        df.to_csv(f.name, index=False)
        temp_path = f.name
    
    try:
        loader = DataLoader()
        loaded_df = loader.load_csv(temp_path)
        
        assert loaded_df.shape == (3, 2)
        assert list(loaded_df.columns) == ['A', 'B']
    finally:
        os.unlink(temp_path)

def test_code_executor_success():
    """Test successful code execution"""
    df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    executor = CodeExecutor(df, tempfile.gettempdir())
    
    code = """
result = df['A'].sum()
print(f"Sum: {result}")
"""
    
    success, output, result = executor.execute(code)
    
    assert success == True
    assert result == 6
    assert "Sum: 6" in output

def test_code_executor_error():
    """Test error handling"""
    df = pd.DataFrame({'A': [1, 2, 3]})
    executor = CodeExecutor(df, tempfile.gettempdir())
    
    code = "result = df['NonExistent'].sum()"
    
    success, output, result = executor.execute(code)
    
    assert success == False
    assert "Error" in output

def test_code_validation():
    """Test code validation"""
    df = pd.DataFrame()
    executor = CodeExecutor(df, tempfile.gettempdir())
    
    # Safe code
    is_valid, msg = executor.validate_code("df.head()")
    assert is_valid == True
    
    # Dangerous code
    is_valid, msg = executor.validate_code("import os; os.system('ls')")
    assert is_valid == False

print("\n" + "="*80)
print("ALL FILES CREATED SUCCESSFULLY!")
print("="*80)
print("\nFiles included:")
print("  1. requirements.txt - Python dependencies")
print("  2. .env.example - Environment variable template")
print("  3. config.py - Configuration settings")
print("  4. data_loader.py - Data loading from CSV/SQL")
print("  5. rag_system.py - RAG system for context retrieval")
print("  6. code_executor.py - Safe code execution engine")
print("  7. analyst_agent.py - AI agent using Claude")
print("  8. main.py - Main orchestrator and entry point")
print("  9. example_usage.py - Usage examples")
print(" 10. README.md - Documentation")
print(" 11. test_analyst.py - Unit tests")
print("\nNext steps:")
print("  1. Copy each section into separate files")
print("  2. Create .env file with your API keys")
print("  3. Install dependencies: pip install -r requirements.txt")
print("  4. Run: python main.py your_data.csv")
print("="*80)