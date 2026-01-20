from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from main import AutonomousDataAnalyst
import os
import json
import logging
from datetime import datetime
from urllib.parse import quote_plus

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend-backend communication
analyst = None

# Configuration
app.static_folder = 'frontend/static'
FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "frontend")
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "uploads")
LOGS_FOLDER = os.path.join(os.path.dirname(__file__), "logs")

# Create necessary directories
for folder in [UPLOAD_FOLDER, LOGS_FOLDER, "outputs/plots"]:
    os.makedirs(folder, exist_ok=True)

@app.route("/")
def index():
    """Serve the main frontend page"""
    return send_from_directory(FRONTEND_DIR, "index.html")

@app.route("/api/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "analyst_loaded": analyst is not None
    })

@app.route("/api/upload_csv", methods=["POST"])
def api_upload_csv():
    """Upload and load CSV file"""
    global analyst
    try:
        if 'file' not in request.files:
            return jsonify({"success": False, "msg": "No file part in request"}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"success": False, "msg": "No file selected"}), 400
        
        # Validate file extension
        if not file.filename.lower().endswith('.csv'):
            return jsonify({"success": False, "msg": "Only CSV files are supported"}), 400
        
        # Save file with timestamp to avoid conflicts
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_filename = f"{timestamp}_{file.filename}"
        filepath = os.path.join(UPLOAD_FOLDER, safe_filename)
        file.save(filepath)
        
        logger.info(f"CSV file uploaded: {safe_filename}")
        
        # Initialize analyst and load data
        analyst = AutonomousDataAnalyst()
        analyst.load_data(filepath, source_type="csv")
        
        # Get data summary
        data_info = analyst.data_loader.get_schema_info()
        sample_data = analyst.data_loader.get_sample_data(5)
        
        return jsonify({
            "success": True,
            "msg": "CSV loaded successfully",
            "filename": safe_filename,
            "shape": analyst.data_loader.metadata.get("shape"),
            "columns": analyst.data_loader.metadata.get("columns"),
            "preview": sample_data
        })
    
    except Exception as e:
        logger.error(f"Error uploading CSV: {str(e)}", exc_info=True)
        return jsonify({"success": False, "msg": f"Error: {str(e)}"}), 500

@app.route("/api/upload_pdf", methods=["POST"])
def api_upload_pdf():
    """Upload and process PDF file"""
    global analyst
    try:
        if 'file' not in request.files:
            return jsonify({"success": False, "msg": "No file part in request"}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"success": False, "msg": "No file selected"}), 400
        
        # Validate file extension
        if not file.filename.lower().endswith('.pdf'):
            return jsonify({"success": False, "msg": "Only PDF files are supported"}), 400
        
        # Save file with timestamp to avoid conflicts
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_filename = f"{timestamp}_{file.filename}"
        filepath = os.path.join(UPLOAD_FOLDER, safe_filename)
        file.save(filepath)
        
        logger.info(f"PDF file uploaded: {safe_filename}")
        
        # Initialize analyst
        analyst = AutonomousDataAnalyst()
        
        # Load PDF text
        text_chunks = analyst.data_loader.load_pdf(filepath)
        
        # Index in RAG
        analyst.rag_system.index_pdf_content(text_chunks, safe_filename)
        
        return jsonify({
            "success": True,
            "msg": "PDF uploaded and indexed successfully",
            "filename": safe_filename,
            "pages": len(text_chunks)
        })
    
    except Exception as e:
        logger.error(f"Error uploading PDF: {str(e)}", exc_info=True)
        return jsonify({"success": False, "msg": f"Error: {str(e)}"}), 500

@app.route("/api/load_db", methods=["POST"])
def api_load_db():
    """Load data from MySQL or PostgreSQL database"""
    global analyst
    try:
        data = request.get_json() or {}
        db_type = data.get("db_type", "mysql")
        table_name = data.get("table", "sales")
        
        # Build connection string based on database type
        if db_type == "postgresql":
            # PostgreSQL connection using provided URL
            postgres_url = data.get("postgres_url")
            if not postgres_url:
                return jsonify({"success": False, "msg": "PostgreSQL URL is required"}), 400
            conn_str = postgres_url
            logger.info(f"Loading PostgreSQL table: {table_name}")
        else:
            # MySQL connection (default)
            host = data.get("host", "localhost")
            port = data.get("port", "3306")
            user = data.get("user", "root")
            password = data.get("password", "")
            database = data.get("database", "sales_data")
            conn_str = f"mysql+pymysql://{user}:{password}@{host}:{port}/{database}"
            logger.info(f"Loading MySQL table: {table_name} from {host}")
        
        analyst = AutonomousDataAnalyst()
        analyst.load_data(conn_str, source_type="sql", table_name=table_name)
        
        # Get data summary
        data_info = analyst.data_loader.get_schema_info()
        sample_data = analyst.data_loader.get_sample_data(5)
        
        return jsonify({
            "success": True,
            "msg": f"{db_type.upper()} table '{table_name}' loaded successfully",
            "shape": analyst.data_loader.metadata.get("shape"),
            "columns": analyst.data_loader.metadata.get("columns"),
            "preview": sample_data
        })
    
    except Exception as e:
        logger.error(f"Error loading database: {str(e)}", exc_info=True)
        return jsonify({"success": False, "msg": f"Database error: {str(e)}"}), 500

@app.route("/api/sql_query", methods=["POST"])
def api_sql_query():
    """Execute custom SQL query against MySQL database"""
    global analyst
    try:
        payload = request.get_json()
        sql = (payload or {}).get("sql", "").strip()
        
        if not sql:
            return jsonify({"success": False, "msg": "No SQL query provided"}), 400
        
        # Security: Basic SQL injection prevention (reject dangerous keywords)
        dangerous_keywords = ["DROP", "DELETE", "TRUNCATE", "ALTER", "CREATE", "INSERT", "UPDATE"]
        sql_upper = sql.upper()
        for keyword in dangerous_keywords:
            if keyword in sql_upper:
                return jsonify({
                    "success": False,
                    "msg": f"Dangerous SQL keyword detected: {keyword}. Only SELECT queries are allowed."
                }), 403
        
        logger.info(f"Executing SQL query: {sql[:100]}...")
        
        conn_str = get_mysql_conn_str()
        analyst = analyst or AutonomousDataAnalyst()
        
        df = analyst.load_data(conn_str, source_type="sql", query=sql)
        rows = df.to_dict(orient="records")
        
        return jsonify({
            "success": True,
            "columns": list(df.columns),
            "rows": rows,
            "row_count": len(rows),
            "dtypes": {str(k): str(v) for k, v in df.dtypes.to_dict().items()}
        })
    
    except Exception as e:
        logger.error(f"SQL query error: {str(e)}", exc_info=True)
        return jsonify({"success": False, "msg": f"Query error: {str(e)}"}), 500

@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    """Analyze data based on user query"""
    global analyst
    try:
        if not analyst:
            return jsonify({"success": False, "msg": "No data loaded. Please upload CSV or load database first."}), 400
        
        data = request.get_json()
        query = data.get("query", "").strip()
        
        if not query:
            return jsonify({"success": False, "msg": "No query provided"}), 400
        
        logger.info(f"Analyzing query: {query}")
        
        # Perform analysis with error handling and retry
        result = analyst.analyze(query, max_retries=2)
        
        if isinstance(result, dict):
            response_data = {
                "success": True,
                "summary": result.get("summary", ""),
                "output": result.get("output", ""),
                "result": result.get("result"),
                "plot_filename": result.get("plot_filename"),
                "code_executed": result.get("code_executed", ""),
                "execution_time": result.get("execution_time", 0)
            }
        else:
            # Fallback for older format
            response_data = {
                "success": True,
                "summary": str(result),
                "output": "",
                "result": None,
                "plot_filename": None
            }
        
        logger.info(f"Analysis completed successfully")
        return jsonify(response_data)
    
    except Exception as e:
        logger.error(f"Analysis error: {str(e)}", exc_info=True)
        return jsonify({
            "success": False,
            "msg": f"Analysis error: {str(e)}"
        }), 500

@app.route("/api/data_info", methods=["GET"])
def api_data_info():
    """Get information about loaded data"""
    global analyst
    try:
        if not analyst or not analyst.data_loader.data is not None:
            return jsonify({"success": False, "msg": "No data loaded"}), 400
        
        schema_info = analyst.data_loader.get_schema_info()
        sample_data = analyst.data_loader.get_sample_data(10)
        
        return jsonify({
            "success": True,
            "schema": schema_info,
            "sample": sample_data,
            "metadata": analyst.data_loader.metadata
        })
    
    except Exception as e:
        logger.error(f"Error getting data info: {str(e)}", exc_info=True)
        return jsonify({"success": False, "msg": str(e)}), 500

@app.route("/api/plot/<plotfile>")
def api_plot(plotfile):
    """Serve generated plot files"""
    try:
        plot_path = os.path.join("outputs", "plots")
        return send_from_directory(plot_path, plotfile)
    except Exception as e:
        logger.error(f"Error serving plot: {str(e)}")
        return jsonify({"error": "Plot not found"}), 404

@app.route("/api/clear_data", methods=["POST"])
def api_clear_data():
    """Clear loaded data and reset analyst"""
    global analyst
    analyst = None
    logger.info("Data cleared")
    return jsonify({"success": True, "msg": "Data cleared successfully"})

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(e):
    logger.error(f"Internal server error: {str(e)}", exc_info=True)
    return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    logger.info("Starting Autonomous Data Analyst Flask server...")
    app.run(host='0.0.0.0', port=port, debug=True)