<<<<<<< HEAD
import os
import logging
from typing import Dict, Any, Optional
import time
import re
from datetime import datetime
from dotenv import load_dotenv

# Import custom modules
=======
from config import Config
>>>>>>> b161ff043a0535b4a953740da79412d61fd73b8a
from data_loader import DataLoader
from rag_system import RAGSystem
from code_executor import CodeExecutor
from analyst_agent import AnalystAgent
<<<<<<< HEAD

# Load environment variables
load_dotenv()

# Create necessary directories first
os.makedirs('logs', exist_ok=True)
os.makedirs('outputs/plots', exist_ok=True)
os.makedirs('uploads', exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/analyst.log'),
        logging.StreamHandler()
    ]
=======
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
>>>>>>> b161ff043a0535b4a953740da79412d61fd73b8a
)
logger = logging.getLogger(__name__)

class AutonomousDataAnalyst:
<<<<<<< HEAD
    """
    Enhanced Autonomous Data Analyst with intelligent reasoning,
    self-correction, and comprehensive error handling
    """
    
    def __init__(self, api_key: str = None, output_dir: str = "outputs/plots", 
                 model: str = "models/gemini-2.5-flash"):
        """
        Initialize the Autonomous Data Analyst
        
        Args:
            api_key: Google Gemini API key (if not in env)
            output_dir: Directory for output plots
            model: Gemini model name (default: models/gemini-2.5-flash)
        """
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        
        if not self.api_key:
            raise ValueError(
                "Gemini API key not found. Set GEMINI_API_KEY environment variable "
                "or pass it to the constructor."
            )
        
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs("logs", exist_ok=True)
        
        # Initialize components
        self.data_loader = DataLoader()
        self.rag_system = RAGSystem()
        self.code_executor = None
        self.analyst_agent = AnalystAgent(api_key=self.api_key, model=model)
        
        self.analysis_history = []
        self.performance_metrics = {
            "total_analyses": 0,
            "successful_analyses": 0,
            "failed_analyses": 0,
            "total_execution_time": 0,
            "average_execution_time": 0
        }
        
        logger.info("AutonomousDataAnalyst initialized successfully")
    
    def load_data(self, source: str, source_type: str = "csv", **kwargs) -> None:
        """
        Load data from various sources
        
        Args:
            source: File path or connection string
            source_type: Type of source ("csv" or "sql")
            **kwargs: Additional arguments for data loading
        """
        try:
            logger.info(f"Loading data from {source_type}: {source}")
            
            if source_type == "csv":
                self.data_loader.load_csv(source)
            elif source_type == "sql":
                table_name = kwargs.get("table_name")
                query = kwargs.get("query")
                self.data_loader.load_sql(source, query=query, table_name=table_name)
            else:
                raise ValueError(f"Unsupported source type: {source_type}")
            
            # Initialize code executor with loaded data
            self.code_executor = CodeExecutor(
                data=self.data_loader.data,
                output_dir=self.output_dir
            )
            
            # Index data in RAG system
            self._index_data_context()
            
            logger.info(f"Data loaded successfully: shape={self.data_loader.data.shape}")
            
        except Exception as e:
            logger.error(f"Error loading data: {e}", exc_info=True)
            raise
    
    def _index_data_context(self):
        """Index data context in RAG system with enhanced intelligence"""
        try:
            data_info = self.data_loader.get_sample_data(None)  # Use all data for context
            schema_info = self.data_loader.get_schema_info()
            
            # Use enhanced indexing with data profiling
            self.rag_system.index_data_context(
                df=self.data_loader.data,
                data_info=data_info,
                schema_info=schema_info
            )
            
            logger.info("Data context indexed in RAG system")
        except Exception as e:
            logger.error(f"Error indexing data context: {e}")
            raise
    
    def analyze(self, query: str, max_retries: int = 2, verbose: bool = True) -> Dict[str, Any]:
        """
        Analyze data based on user query with intelligent retry and self-correction
        
        Args:
            query: Natural language query
            max_retries: Maximum number of retry attempts
            verbose: Whether to log detailed information
            
        Returns:
            Dictionary containing analysis results
        """
        if self.data_loader.data is None:
            raise ValueError("No data loaded. Please load data first using load_data()")
        
        analysis_start = time.time()
        self.performance_metrics["total_analyses"] += 1
        
        try:
            logger.info(f"Starting analysis for query: {query}")
            
            # Retrieve relevant context from RAG system
            retrieved_context = self.rag_system.retrieve_context(query, top_k=4)
            
            if verbose:
                logger.info(f"Retrieved {len(retrieved_context)} relevant context documents")
            
            # Get data context
            data_context = self._build_data_context()
            
            # Get data profile
            data_profile = self.data_loader.get_data_profile()
            
            # Generate analysis plan and code
            response = self.analyst_agent.analyze_query(
                query=query,
                data_context=data_context,
                retrieved_context=retrieved_context,
                data_profile=data_profile
            )
            
            # Extract code from response
            code_blocks = self.analyst_agent.extract_code(response)
            
            if not code_blocks:
                logger.warning("No code blocks found in AI response")
                return {
                    "success": False,
                    "summary": response,
                    "output": "No executable code generated",
                    "result": None,
                    "plot_filename": None,
                    "execution_time": time.time() - analysis_start
                }
            
            # Try executing code with retries
            success, output, result, plot_filename = self._execute_with_retry(
                code_blocks[0],
                query,
                data_context,
                max_retries
            )
            
            execution_time = time.time() - analysis_start
            
            # Update metrics
            if success:
                self.performance_metrics["successful_analyses"] += 1
            else:
                self.performance_metrics["failed_analyses"] += 1
            
            self.performance_metrics["total_execution_time"] += execution_time
            self.performance_metrics["average_execution_time"] = (
                self.performance_metrics["total_execution_time"] / 
                self.performance_metrics["total_analyses"]
            )
            
            # Extract explanation from response
            explanation = self._extract_explanation(response)
            
            # Log to history
            self._log_analysis(query, success, execution_time)
            
            analysis_result = {
                "success": success,
                "summary": explanation,
                "output": output,
                "result": result,
                "plot_filename": plot_filename,
                "code_executed": code_blocks[0] if success else None,
                "execution_time": round(execution_time, 2),
                "timestamp": datetime.now().isoformat()
            }
            
            # Suggest next steps if successful
            if success:
                suggestions = self.analyst_agent.suggest_next_steps(query, result)
                analysis_result["suggestions"] = suggestions
            
            logger.info(f"Analysis completed: success={success}, time={execution_time:.2f}s")
            
            return analysis_result
            
        except Exception as e:
            execution_time = time.time() - analysis_start
            self.performance_metrics["failed_analyses"] += 1
            
            logger.error(f"Analysis failed: {e}", exc_info=True)
            
            return {
                "success": False,
                "summary": f"Analysis failed: {str(e)}",
                "output": str(e),
                "result": None,
                "plot_filename": None,
                "execution_time": round(execution_time, 2),
                "timestamp": datetime.now().isoformat()
            }
    
    def _execute_with_retry(self, code: str, query: str, 
                           data_context: str, max_retries: int) -> tuple:
        """
        Execute code with intelligent retry and self-correction
        
        Args:
            code: Python code to execute
            query: Original user query
            data_context: Data context string
            max_retries: Maximum retry attempts
            
        Returns:
            Tuple of (success, output, result, plot_filename)
        """
        attempt = 0
        last_error = None
        
        while attempt <= max_retries:
            attempt += 1
            
            logger.info(f"Execution attempt {attempt}/{max_retries + 1}")
            
            # Validate code before execution
            is_valid, validation_msg = self.code_executor.validate_code(code)
            
            if not is_valid:
                logger.warning(f"Code validation failed: {validation_msg}")
                
                if attempt > max_retries:
                    return False, f"Code validation failed: {validation_msg}", None, None
                
                # Try to fix validation issues
                try:
                    corrected_response = self.analyst_agent.self_correct(
                        code=code,
                        error=f"Validation Error: {validation_msg}",
                        query=query,
                        data_context=data_context,
                        attempt=attempt
                    )
                    
                    corrected_blocks = self.analyst_agent.extract_code(corrected_response)
                    if corrected_blocks:
                        code = corrected_blocks[0]
                        logger.info("Code corrected after validation failure")
                        continue
                    else:
                        return False, "Failed to generate valid corrected code", None, None
                        
                except Exception as e:
                    logger.error(f"Self-correction failed: {e}")
                    return False, f"Self-correction failed: {str(e)}", None, None
            
            # Execute code
            success, output, result, plot_filename = self.code_executor.execute(code)
            
            if success:
                logger.info(f"Code execution successful on attempt {attempt}")
                return success, output, result, plot_filename
            
            # Execution failed
            last_error = output
            logger.warning(f"Execution failed on attempt {attempt}: {output[:200]}")
            
            if attempt > max_retries:
                break
            
            # Try self-correction
            try:
                logger.info(f"Attempting self-correction (attempt {attempt})")
                
                corrected_response = self.analyst_agent.self_correct(
                    code=code,
                    error=output,
                    query=query,
                    data_context=data_context,
                    attempt=attempt
                )
                
                corrected_blocks = self.analyst_agent.extract_code(corrected_response)
                
                if corrected_blocks:
                    code = corrected_blocks[0]
                    logger.info("Self-correction generated new code")
                else:
                    logger.warning("Self-correction did not generate code blocks")
                    break
                    
            except Exception as e:
                logger.error(f"Self-correction failed: {e}")
                break
        
        # All retries exhausted
        return False, last_error or "Execution failed after all retries", None, None
    
    def _build_data_context(self) -> str:
        """Build comprehensive data context string"""
        context_parts = [
            f"Dataset Shape: {self.data_loader.data.shape}",
            f"\nAvailable Columns: {list(self.data_loader.data.columns)}",
            f"\nData Types:\n{self.data_loader.data.dtypes.to_string()}",
            f"\n\nAll Rows:\n{self.data_loader.data.to_string()}"  # Show ALL rows
        ]
        
        return "\n".join(context_parts)
    
    def _extract_explanation(self, response: str) -> str:
        """Extract explanation text from AI response"""
        # Remove code blocks
        explanation = response
        code_pattern = r'```python.*?```'
        explanation = re.sub(code_pattern, '', explanation, flags=re.DOTALL)
        
        # Clean up
        explanation = explanation.strip()
        
        # Limit length
        if len(explanation) > 1000:
            explanation = explanation[:1000] + "..."
        
        return explanation
    
    def _log_analysis(self, query: str, success: bool, execution_time: float):
        """Log analysis to history"""
        self.analysis_history.append({
            "query": query,
            "success": success,
            "execution_time": execution_time,
            "timestamp": datetime.now().isoformat()
        })
        
        # Keep only last 100 analyses
        if len(self.analysis_history) > 100:
            self.analysis_history.pop(0)
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        return self.performance_metrics.copy()
    
    def get_analysis_history(self, limit: int = 10) -> list:
        """Get recent analysis history"""
        return self.analysis_history[-limit:]
    
    def get_data_insights(self) -> Dict[str, Any]:
        """Get comprehensive data insights"""
        if self.data_loader.data is None:
            return {"error": "No data loaded"}
        
        insights = {
            "shape": self.data_loader.data.shape,
            "columns": list(self.data_loader.data.columns),
            "dtypes": {str(k): str(v) for k, v in self.data_loader.data.dtypes.to_dict().items()},
            "memory_usage_mb": self.data_loader.metadata.get("memory_usage_mb", 0),
            "missing_data": self.data_loader.metadata.get("missing_data", {}),
            "data_profile": self.data_loader.get_data_profile(),
            "rag_insights": self.rag_system.get_data_insights()
        }
        
        return insights
    
    def reset(self):
        """Reset the analyst state"""
        self.data_loader = DataLoader()
        self.code_executor = None
        self.analyst_agent.clear_history()
        self.analysis_history = []
        
        logger.info("Analyst state reset")
    
    def export_analysis_report(self, filepath: str = "analysis_report.json"):
        """Export comprehensive analysis report"""
        import json
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "data_info": {
                "shape": self.data_loader.data.shape if self.data_loader.data is not None else None,
                "columns": list(self.data_loader.data.columns) if self.data_loader.data is not None else [],
                "metadata": self.data_loader.metadata
            },
            "performance_metrics": self.performance_metrics,
            "analysis_history": self.analysis_history,
            "execution_stats": self.code_executor.get_execution_stats() if self.code_executor else {}
        }
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Analysis report exported to {filepath}")


# Example usage
if __name__ == "__main__":
    import sys
    
    # Initialize analyst
    analyst = AutonomousDataAnalyst()
    
    # Example: Load CSV
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
        analyst.load_data(csv_file, source_type="csv")
        
        # Interactive analysis
        print("\n" + "="*70)
        print("Autonomous Data Analyst - Interactive Mode")
        print("="*70)
        print("\nData loaded successfully!")
        print(f"Shape: {analyst.data_loader.data.shape}")
        print(f"\nSample data:\n{analyst.data_loader.get_sample_data(5)}")
        
        while True:
            print("\n" + "-"*70)
            query = input("\nEnter your analysis query (or 'quit' to exit): ")
            
            if query.lower() in ['quit', 'exit', 'q']:
                break
            
            if not query.strip():
                continue
            
            print("\nAnalyzing...")
            result = analyst.analyze(query)
            
            print("\n" + "="*70)
            print("RESULTS")
            print("="*70)
            print(f"\nExplanation:\n{result['summary']}")
            print(f"\nOutput:\n{result['output']}")
            
            if result.get('plot_filename'):
                print(f"\nPlot saved: {result['plot_filename']}")
            
            if result.get('suggestions'):
                print("\nSuggested next steps:")
                for i, suggestion in enumerate(result['suggestions'], 1):
                    print(f"  {i}. {suggestion}")
        
        # Export report
        analyst.export_analysis_report()
        print("\nAnalysis report exported!")
    
    else:
        print("Usage: python main.py <csv_file_path>")
        print("\nOr import and use programmatically:")
        print("  from main import AutonomousDataAnalyst")
        print("  analyst = AutonomousDataAnalyst()")
        print("  analyst.load_data('data.csv', source_type='csv')")
        print("  result = analyst.analyze('Show me sales trends over time')")
=======
    """Main orchestrator for the autonomous data analyst system"""
    
    def __init__(self):
        Config.setup_directories()
        
        self.data_loader = DataLoader()
        self.rag_system = RAGSystem(Config.EMBEDDING_MODEL)
        self.analyst_agent = AnalystAgent(Config.GEMINI_API_KEY, Config.GEMINI_MODEL)
        self.code_executor = None
        
        logger.info("Autonomous Data Analyst initialized with Gemini API")
    
    def load_data(self, source: str, source_type: str = None, **kwargs):
        """Intelligently load data from source with auto-detection"""
        import os
        from pathlib import Path
        
        # Auto-detect source type if not provided
        if source_type is None:
            if os.path.exists(source):
                # It's a file path
                file_ext = Path(source).suffix.lower()
                if file_ext in ['.csv', '.tsv']:
                    source_type = "csv"
                elif file_ext in ['.xlsx', '.xls']:
                    source_type = "excel"
                elif file_ext in ['.json', '.jsonl']:
                    source_type = "json"
                else:
                    source_type = "file"  # Will use intelligent file loader
            elif '://' in source or source.startswith('mysql') or source.startswith('postgresql'):
                # It's a database connection string
                source_type = "sql"
            else:
                # Default to CSV
                source_type = "csv"
        
        logger.info(f"Loading data from {source} (detected type: {source_type})")
        
        if source_type == "csv":
            data = self.data_loader.load_csv(source, **kwargs)
        elif source_type == "excel":
            data = self.data_loader.load_excel(source, **kwargs)
        elif source_type == "json":
            data = self.data_loader.load_json(source, **kwargs)
        elif source_type == "file":
            data = self.data_loader.load_file(source, **kwargs)
        elif source_type == "sql":
            data = self.data_loader.load_sql(source, **kwargs)
        else:
            raise ValueError(f"Unsupported source type: {source_type}")
        
        # Initialize code executor with loaded data
        self.code_executor = CodeExecutor(data, Config.PLOTS_DIR)
        
        # Index comprehensive data context in RAG system with actual dataframe
        schema_info = self.data_loader.get_schema_info()
        data_info = f"Dataset loaded from {source}\nShape: {data.shape}\nColumns: {list(data.columns)}"
        self.rag_system.index_data_context(data_info, schema_info, dataframe=data)
        
        logger.info("Data loaded and indexed successfully")
        print("\n" + "="*60)
        print("DATA LOADED SUCCESSFULLY")
        print("="*60)
        print(schema_info)
        
        return data
    
    def analyze(self, query: str, max_retries: int = 3):
        """Analyze data based on user query with self-correction"""
        logger.info(f"Processing query: {query}")
        print("\n" + "="*60)
        print(f"QUERY: {query}")
        print("="*60)
        
        if self.code_executor is None:
            raise ValueError("No data loaded. Please load data first.")
        
        # Retrieve relevant context with enhanced RAG
        retrieved_context = self.rag_system.retrieve_context(query, top_k=8)
        data_context = self.data_loader.get_schema_info()
        
        # Try to get direct answer first
        direct_answer = self.rag_system.get_direct_answer(query)
        
        # Generate intelligent response with exact answers
        response = self.analyst_agent.analyze_query(query, data_context, retrieved_context, direct_answer)
        print("\n" + "-"*60)
        print("AGENT RESPONSE:")
        print("-"*60)
        print(response)
        
        # Extract and execute code
        code_blocks = self.analyst_agent.extract_code(response)
        
        if not code_blocks:
            print("\nNo code generated. Providing explanation only.")
            return {
                "summary": response,
                "output": "",
                "result": None,
                "plot_filename": None
            }
        
        # Store execution results
        all_outputs = []
        all_results = []
        plot_filenames = []
        
        for i, code in enumerate(code_blocks):
            print(f"\n{'='*60}")
            print(f"EXECUTING CODE BLOCK {i+1}/{len(code_blocks)}")
            print("="*60)
            print(code)
            
            retry_count = 0
            current_code = code
            
            while retry_count < max_retries:
                # Validate code
                is_valid, validation_msg = self.code_executor.validate_code(current_code)
                if not is_valid:
                    print(f"\n⚠️  Code validation failed: {validation_msg}")
                    break
                
                # Execute code
                success, output, result, plot_filename = self.code_executor.execute(current_code)
                
                if success:
                    print(f"\n✅ Execution successful!")
                    if output:
                        print("\nOutput:")
                        print(output)
                        all_outputs.append(output)
                    if result is not None:
                        print("\nResult:")
                        print(result)
                        all_results.append(result)
                    if plot_filename:
                        plot_filenames.append(plot_filename)
                    break
                else:
                    print(f"\n❌ Execution failed (attempt {retry_count + 1}/{max_retries})")
                    print(f"\nError:\n{output}")
                    
                    if retry_count < max_retries - 1:
                        print("\n🔄 Attempting self-correction...")
                        corrected_response = self.analyst_agent.self_correct(
                            current_code, output, query, data_context
                        )
                        corrected_codes = self.analyst_agent.extract_code(corrected_response)
                        
                        if corrected_codes:
                            current_code = corrected_codes[0]
                            print("\nCorrected code:")
                            print(current_code)
                        else:
                            print("⚠️  Failed to generate corrected code")
                            break
                    
                    retry_count += 1
            
            if retry_count == max_retries:
                print(f"\n⚠️  Max retries reached for code block {i+1}")
        
        print("\n" + "="*60)
        print("ANALYSIS COMPLETE")
        print("="*60)
        
        # Combine all outputs
        combined_output = "\n".join(all_outputs)
        final_result = all_results[-1] if all_results else None
        
        return {
            "summary": response,
            "output": combined_output,
            "result": str(final_result) if final_result is not None else None,
            "plot_filename": plot_filenames[0] if plot_filenames else None
        }
    
    def interactive_mode(self):
        """Run in interactive mode"""
        print("\n" + "="*60)
        print("AUTONOMOUS DATA ANALYST - INTERACTIVE MODE")
        print("Powered by Google Gemini")
        print("="*60)
        print("\nCommands:")
        print("  'load <filepath>' - Load a new CSV file")
        print("  'mysql <table_name>' - Load data from MySQL table")
        print("  'sqlquery <query>' - Execute custom SQL query")
        print("  'info' - Show dataset information")
        print("  'help' - Show this help message")
        print("="*60 + "\n")
        
        while True:
            try:
                query = input("📊 Your query: ").strip()
                
                if not query:
                    continue
                
                if query.lower() in ['exit', 'quit']:
                    print("\n👋 Goodbye!")
                    break
                
                if query.lower() == 'help':
                    print("\nCommands:")
                    print("  'load <filepath>' - Load a new CSV file")
                    print("  'mysql <table_name>' - Load from MySQL table")
                    print("  'sqlquery <query>' - Execute custom SQL query")
                    print("  'info' - Show dataset information")
                    print("  'exit' or 'quit' - Exit the program")
                    continue
                
                if query.lower().startswith('load '):
                    filepath = query[5:].strip()
                    try:
                        self.load_data(filepath)
                    except Exception as e:
                        print(f"❌ Error loading data: {e}")
                    continue
                
                if query.lower().startswith('mysql '):
                    table_name = query[6:].strip()
                    try:
                        # MySQL connection string
                        connection_string = "mysql+pymysql://analyst:Hasnain123%40@localhost:3306/sales_data"
                        self.load_data(connection_string, source_type="sql", table_name=table_name)
                    except Exception as e:
                        print(f"❌ Error loading from MySQL: {e}")
                    continue
                
                if query.lower().startswith('sqlquery '):
                    sql_query = query[9:].strip()
                    try:
                        connection_string = "mysql+pymysql://analyst:your_password@localhost:3306/sales_data"
                        self.load_data(connection_string, source_type="sql", query=sql_query)
                    except Exception as e:
                        print(f"❌ Error executing SQL query: {e}")
                    continue
                
                if query.lower() == 'info':
                    if self.data_loader.data is not None:
                        print(self.data_loader.get_schema_info())
                    else:
                        print("No data loaded.")
                    continue
                
                self.analyze(query)
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                logger.error(f"Error in interactive mode: {e}")
                print(f"\n❌ Error: {e}")

def main():
    """Main entry point"""
    analyst = AutonomousDataAnalyst()
    
    if len(sys.argv) > 1:
        # Load data from command line argument
        data_path = sys.argv[1]
        try:
            analyst.load_data(data_path)
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            sys.exit(1)
    
    # Start interactive mode
    analyst.interactive_mode()

if __name__ == "__main__":
    main()
>>>>>>> b161ff043a0535b4a953740da79412d61fd73b8a
