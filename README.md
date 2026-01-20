# 🤖 Autonomous Data Analyst

A production-ready AI-powered data analysis platform with a modern web interface. Connect CSV files, PDF documents, or MySQL databases and ask questions in natural language. The AI generates and executes Python code to provide insights and visualizations.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-3.0-green.svg)
![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## ✨ Features

- 🎨 **Modern Dark UI** - Glassmorphic design with chat interface
- 📊 **Multi-Source Support** - CSV, PDF, and MySQL databases
- 🤖 **AI-Powered Analysis** - Google Gemini generates executable Python code
- 🔄 **Self-Correction** - Automatic error detection and retry
- 🧠 **RAG System** - Intelligent context retrieval using ChromaDB
- 📈 **Rich Visualizations** - Matplotlib, Seaborn, and Plotly charts
- � **Docker Ready** - One-command deployment
- � **Secure Execution** - Sandboxed code environment

## � Quick Start

### Prerequisites
- Python 3.9+
- Google Gemini API key ([Get one free](https://makersuite.google.com/app/apikey))
- MySQL (optional, for database analysis)

### Local Development

1. **Clone and install dependencies:**
```bash
git clone <your-repo-url>
cd "Ai Project"
pip install -r requirements.txt
```

2. **Set up environment:**
```bash
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY
```

3. **Run the application:**
```bash
python app.py
```

4. **Open your browser:**
```
http://localhost:5000
```

### Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose up --build

# Access at http://localhost:5000
```

## 📖 Usage

### 1. Connect a Data Source

**Upload Files:**
- Navigate to "Data Sources" tab
- Drag & drop CSV or PDF files
- Or click "Browse Files"

**Connect Database:**
- Click "Connect Database"
- Enter credentials (host, port, user, password, database, table)
- Click "Connect"

### 2. Ask Questions

Switch to "Analysis" tab and type queries like:

**For CSV/Database:**
- "Show me sales trends over time"
- "What are the top 5 products by revenue?"
- "Create a correlation heatmap"
- "Find outliers in the price column"

**For PDF:**
- "Summarize the main findings"
- "What are the key statistics mentioned?"
- "Extract information about [topic]"

### 3. View Results

- Text answers appear in chat bubbles
- Plots are embedded inline
- Tables are formatted in markdown

## 🏗️ Architecture

```
┌─────────────────┐
│   Frontend      │  HTML/CSS/JS (Dark Theme)
│   (Browser)     │
└────────┬────────┘
         │ HTTP/JSON
┌────────▼────────┐
│   Flask API     │  app.py
└────────┬────────┘
         │
┌────────▼────────────────────────┐
│  AutonomousDataAnalyst (main.py)│
└─┬──────┬──────┬──────┬──────────┘
  │      │      │      │
  ▼      ▼      ▼      ▼
┌────┐┌────┐┌────┐┌────┐
│Data││RAG ││AI  ││Code│
│Load││Sys ││Agnt││Exec│
└────┘└────┘└────┘└────┘
  │      │      │      │
  ▼      ▼      ▼      ▼
[CSV] [Vector] [Gemini] [Python]
[PDF]  [DB]              [Plots]
[SQL]
```

## 🔧 Configuration

### Environment Variables (.env)

```bash
# Required
GEMINI_API_KEY=your_api_key_here

# Optional (for database defaults)
DB_HOST=localhost
DB_PORT=3306
DB_USER=root
DB_PASSWORD=your_password
DB_NAME=your_database
```

### Key Files

- `app.py` - Flask web server and API endpoints
- `main.py` - Core analyst orchestration
- `analyst_agent.py` - AI agent with Gemini integration
- `data_loader.py` - CSV/SQL/PDF data ingestion
- `rag_system.py` - Context retrieval system
- `code_executor.py` - Safe Python code execution
- `frontend/` - Web UI (HTML/CSS/JS)

## 🐳 Docker Configuration

The included `docker-compose.yml` sets up:
- Flask application container
- MySQL database container (optional)
- Persistent volumes for data

Customize environment variables in `docker-compose.yml` as needed.

## 🔒 Security

**Code Execution:**
- Blocks dangerous imports (os, sys, subprocess)
- Prevents file system access
- Validates code before execution
- Sandboxed environment

**Database:**
- Credentials sent over HTTPS (use SSL in production)
- Consider read-only database users
- Whitelist IP addresses for remote databases

## 📦 Dependencies

**Core:**
- Flask - Web framework
- pandas - Data manipulation
- google-generativeai - Gemini AI
- chromadb - Vector database
- sentence-transformers - Embeddings

**Visualization:**
- matplotlib, seaborn, plotly

**Database:**
- pymysql, sqlalchemy

**PDF:**
- pypdf

See `requirements.txt` for complete list.

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📝 License

MIT License - see LICENSE file for details

## 🆘 Troubleshooting

**API Key Error:**
- Ensure `GEMINI_API_KEY` is set in `.env`
- Verify key is valid at [Google AI Studio](https://makersuite.google.com/)

**Database Connection Failed:**
- Check credentials (host, port, user, password)
- Ensure MySQL is running
- Verify network access/firewall rules
- Avoid special characters in passwords

**Import Errors:**
- Run `pip install -r requirements.txt`
- Use Python 3.9 or higher

**Memory Issues:**
- Reduce dataset size
- Use sampling for large files
- Increase Docker memory limits

## 🎯 Roadmap

- [ ] Support for Excel files
- [ ] PostgreSQL support
- [ ] User authentication
- [ ] Query history and favorites
- [ ] Export results to PDF/Excel
- [ ] Multi-language support

## 📧 Support

For issues and questions, please open a GitHub issue.

---

**Built with ❤️ using Google Gemini AI**