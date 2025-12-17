from flask import Flask, request, jsonify, send_from_directory
from main import AutonomousDataAnalyst
from config import Config
import os
import json
import urllib.parse
from datetime import datetime

app = Flask(__name__)

# Global state management
app_state = {
    "analyst": None,
    "data_loaded": False,
    "data_path": None,
    "last_analysis": None,
    "metadata": {}
}

# Directories
FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "frontend")
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "uploads")
PLOTS_DIR = getattr(Config, 'PLOTS_DIR', os.path.join(os.path.dirname(__file__), "outputs", "plots"))

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

app.static_folder = 'frontend/static'

def get_mysql_conn_str():
    """Generate MySQL connection string with proper encoding"""
    user = "root"
    password = "Hasnain123@"
    host = "localhost"
    port = 3306
    database = "sales_data"
    
    password_encoded = urllib.parse.quote(password, safe='')
    user_encoded = urllib.parse.quote(user, safe='')
    database_encoded = urllib.parse.quote(database, safe='')
    
    return f"mysql+pymysql://{user_encoded}:{password_encoded}@{host}:{port}/{database_encoded}"

def get_schema_info():
    """Get comprehensive schema information from loaded data"""
    if not app_state["analyst"] or not app_state["data_loaded"]:
        return None
    
    try:
        schema_info = app_state["analyst"].data_loader.get_schema_info()
        return schema_info
    except Exception as e:
        return f"Error retrieving schema: {str(e)}"

def get_recent_plots(limit=5):
    """Get recently generated plot files"""
    if not os.path.exists(PLOTS_DIR):
        return []
    
    try:
        plot_files = [f for f in os.listdir(PLOTS_DIR) if f.endswith(('.png', '.jpg', '.jpeg'))]
        plot_files_with_time = [
            (f, os.path.getmtime(os.path.join(PLOTS_DIR, f))) 
            for f in plot_files
        ]
        plot_files_with_time.sort(key=lambda x: x[1], reverse=True)
        return [f[0] for f in plot_files_with_time[:limit]]
    except Exception as e:
        print(f"Error getting plots: {e}")
        return []

@app.route("/")
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")

@app.route("/debug/static-list")
def debug_static_list():
    files_path = os.path.join(FRONTEND_DIR, "static")
    if not os.path.exists(files_path):
        return {"error": f"Path not found: {files_path}"}
    return {"files": os.listdir(files_path)}

@app.route("/api/status", methods=["GET"])
def api_status():
    """Get current application state and metadata"""
    return jsonify({
        "success": True,
        "data_loaded": app_state["data_loaded"],
        "data_path": app_state["data_path"],
        "metadata": app_state["metadata"],
        "analyst_initialized": app_state["analyst"] is not None,
        "recent_plots": get_recent_plots(3)
    })

@app.route("/api/schema", methods=["GET"])
def api_schema():
    """Get detailed schema information about loaded dataset"""
    if not app_state["data_loaded"]:
        return jsonify({
            "success": False,
            "msg": "No data loaded. Please load a dataset first."
        })
    
    schema_info = get_schema_info()
    
    if schema_info:
        return jsonify({
            "success": True,
            "schema": schema_info,
            "metadata": app_state["metadata"]
        })
    else:
        return jsonify({
            "success": False,
            "msg": "Could not retrieve schema information"
        })

@app.route("/api/upload_csv", methods=["POST"])
def api_upload_csv():
    """Upload and load CSV/Excel/JSON file with intelligent detection"""
    if 'file' not in request.files:
        return jsonify({"success": False, "msg": "No file part"})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"success": False, "msg": "No selected file"})
    
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)
    
    # Initialize analyst
    app_state["analyst"] = AutonomousDataAnalyst()
    
    try:
        # Use intelligent file loading - auto-detects CSV, Excel, JSON, etc.
        app_state["analyst"].load_data(filepath)
        
        # Extract metadata
        file_type = app_state["analyst"].data_loader.metadata.get("type", "file")
        shape = app_state["analyst"].data_loader.metadata.get("shape", [0, 0])
        columns = app_state["analyst"].data_loader.metadata.get("columns", [])
        
        # Update state
        app_state["data_loaded"] = True
        app_state["data_path"] = filepath
        app_state["metadata"] = {
            "filename": file.filename,
            "file_type": file_type,
            "shape": shape,
            "columns": columns,
            "rows": shape[0] if len(shape) > 0 else 0,
            "cols": shape[1] if len(shape) > 1 else 0,
            "loaded_at": datetime.now().isoformat()
        }
        
        # Get schema info for richer response
        schema_info = get_schema_info()
        
        return jsonify({
            "success": True, 
            "msg": f"{file_type.upper()} file loaded successfully!", 
            "metadata": app_state["metadata"],
            "schema_preview": schema_info[:500] if schema_info else None  # First 500 chars
        })
    except Exception as e:
        app_state["data_loaded"] = False
        return jsonify({"success": False, "msg": f"Error loading file: {str(e)}"})

@app.route("/api/load_db", methods=["POST"])
def api_load_db():
    """Load data from database with intelligent connection handling"""
    data = request.get_json() or {}
    
    # Get connection parameters from request or use defaults
    host = data.get("host") or "localhost"
    port = data.get("port") or 3306
    user = data.get("user") or "root"
    password = data.get("password") or "Hasnain123@"
    database = data.get("database") or "sales_data"
    table_name = data.get("table_name") or "sales"
    database_type = data.get("database_type", "mysql")
    
    app_state["analyst"] = AutonomousDataAnalyst()
    
    try:
        # Build connection string based on database type
        if database_type == "mysql":
            password_encoded = urllib.parse.quote(password, safe='')
            user_encoded = urllib.parse.quote(user, safe='')
            database_encoded = urllib.parse.quote(database, safe='')
            conn_str = f"mysql+pymysql://{user_encoded}:{password_encoded}@{host}:{port}/{database_encoded}"
        elif database_type == "postgresql":
            password_encoded = urllib.parse.quote(password, safe='')
            user_encoded = urllib.parse.quote(user, safe='')
            database_encoded = urllib.parse.quote(database, safe='')
            conn_str = f"postgresql://{user_encoded}:{password_encoded}@{host}:{port}/{database_encoded}"
        elif database_type == "sqlite":
            conn_str = f"sqlite:///{database}"
        else:
            conn_str = data.get("connection_string")
            if not conn_str:
                return jsonify({"success": False, "msg": f"Unsupported database type: {database_type}"})
        
        app_state["analyst"].load_data(conn_str, source_type="sql", table_name=table_name, database_type=database_type)
        
        # Extract metadata
        shape = app_state["analyst"].data_loader.metadata.get("shape", [0, 0])
        columns = app_state["analyst"].data_loader.metadata.get("columns", [])
        
        # Update state
        app_state["data_loaded"] = True
        app_state["data_path"] = f"{database_type}://{host}/{database}/{table_name}"
        app_state["metadata"] = {
            "source": "database",
            "database_type": database_type,
            "database": database,
            "table": table_name,
            "shape": shape,
            "columns": columns,
            "rows": shape[0] if len(shape) > 0 else 0,
            "cols": shape[1] if len(shape) > 1 else 0,
            "loaded_at": datetime.now().isoformat()
        }
        
        # Get schema info
        schema_info = get_schema_info()
        
        return jsonify({
            "success": True, 
            "msg": f"{database_type.upper()} database loaded successfully!",
            "metadata": app_state["metadata"],
            "schema_preview": schema_info[:500] if schema_info else None
        })
    except Exception as e:
        app_state["data_loaded"] = False
        return jsonify({"success": False, "msg": f"Database connection error: {str(e)}"})

@app.route("/api/sql_query", methods=["POST"])
def api_sql_query():
    """Execute custom SQL query with intelligent result handling"""
    payload = request.get_json()
    sql = (payload or {}).get("sql")
    
    if not sql or not sql.strip():
        return jsonify({"success": False, "msg": "No SQL provided."})

    conn_str = get_mysql_conn_str()
    
    if not app_state["analyst"]:
        app_state["analyst"] = AutonomousDataAnalyst()
    
    try:
        df = app_state["analyst"].load_data(conn_str, source_type="sql", query=sql)
        rows = df.to_dict(orient="records")
        
        return jsonify({
            "success": True,
            "columns": list(df.columns),
            "rows": rows,
            "row_count": len(rows),
            "query": sql
        })
    except Exception as e:
        return jsonify({"success": False, "msg": f"SQL execution error: {str(e)}"})

@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    """Perform intelligent data analysis with comprehensive response"""
    if not app_state["analyst"] or not app_state["data_loaded"]:
        return jsonify({
            "success": False, 
            "msg": "No data loaded. Please load a dataset first."
        })
    
    data = request.get_json()
    query = data.get("query", "").strip()
    
    if not query:
        return jsonify({
            "success": False,
            "msg": "Please enter a query."
        })
    
    try:
        # Perform analysis
        result = app_state["analyst"].analyze(query)
        
        # Get recent plots generated during analysis
        recent_plots = get_recent_plots(5)
        
        # Store last analysis
        app_state["last_analysis"] = {
            "query": query,
            "timestamp": datetime.now().isoformat(),
            "plots": recent_plots
        }
        
        # Handle different result formats
        if isinstance(result, dict):
            response_data = {
                "success": True,
                "query": query,
                "summary": result.get("summary", ""),
                "output": result.get("output", ""),
                "result": result.get("result"),
                "plot_filename": result.get("plot_filename"),
                "recent_plots": recent_plots,
                "timestamp": datetime.now().isoformat()
            }
        else:
            # Fallback for simple string responses
            response_data = {
                "success": True,
                "query": query,
                "summary": str(result),
                "output": "",
                "result": None,
                "plot_filename": None,
                "recent_plots": recent_plots,
                "timestamp": datetime.now().isoformat()
            }
        
        return jsonify(response_data)
        
    except Exception as e:
        return jsonify({
            "success": False, 
            "msg": f"Analysis error: {str(e)}",
            "query": query
        })

@app.route("/api/plots", methods=["GET"])
def api_plots_list():
    """Get list of all available plots"""
    recent_plots = get_recent_plots(10)
    return jsonify({
        "success": True,
        "plots": recent_plots,
        "count": len(recent_plots)
    })

@app.route("/api/plot/<plotfile>")
def api_plot(plotfile):
    """Serve plot images"""
    return send_from_directory(PLOTS_DIR, plotfile)

@app.route("/api/reset", methods=["POST"])
def api_reset():
    """Reset application state"""
    app_state["analyst"] = None
    app_state["data_loaded"] = False
    app_state["data_path"] = None
    app_state["last_analysis"] = None
    app_state["metadata"] = {}
    
    return jsonify({
        "success": True,
        "msg": "Application state reset successfully"
    })

@app.route("/api/clear_plots", methods=["POST"])
def api_clear_plots():
    """Clear all generated plots"""
    try:
        if os.path.exists(PLOTS_DIR):
            for file in os.listdir(PLOTS_DIR):
                if file.endswith(('.png', '.jpg', '.jpeg')):
                    os.remove(os.path.join(PLOTS_DIR, file))
        return jsonify({
            "success": True,
            "msg": "All plots cleared successfully"
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "msg": f"Error clearing plots: {str(e)}"
        })

if __name__ == "__main__":
    app.run(debug=True, port=5000)