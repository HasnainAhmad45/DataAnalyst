from config import Config
from data_loader import DataLoader
from rag_system import RAGSystem
from code_executor import CodeExecutor
from analyst_agent import AnalystAgent
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AutonomousDataAnalyst:
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