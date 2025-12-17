# Autonomous Data Analyst Agent with Google Gemini

A self-correcting AI-powered data analyst that reads CSV/SQL datasets, writes Python/Pandas code, 
and executes it to visualize data with built-in error correction using Google's Gemini API.

## Features

- 📊 **Multi-Source Data Loading**: CSV files and SQL databases
- 🤖 **AI-Powered Analysis**: Uses Google Gemini 1.5 Pro for intelligent code generation
- 🔄 **Self-Correction**: Automatically fixes errors in generated code
- 🧠 **RAG System**: Retrieves relevant context using local embeddings (no API costs)
- 📈 **Advanced Visualizations**: Matplotlib, Seaborn, and Plotly support
- 💬 **Interactive Mode**: Chat-based interface for exploratory analysis
- 🛡️ **Safe Execution**: Sandboxed code execution environment
- 💰 **Cost Effective**: Uses local embeddings instead of API-based embeddings

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables:
```bash
cp .env.example .env
# Edit .env and add your Gemini API key
```

3. Get your Gemini API key:
   - Visit https://makersuite.google.com/app/apikey
   - Create a new API key
   - Add it to your .env file

## Quick Start

### Command Line
```bash
python main.py data.csv
```

### Programmatic Usage
```python
from main import AutonomousDataAnalyst

analyst = AutonomousDataAnalyst()
analyst.load_data("sales_data.csv")
analyst.analyze("Show me sales trends over time")
```

### Interactive Mode
```python
analyst.interactive_mode()
```

## Example Queries

- "Show me the distribution of values in column X"
- "Create a correlation heatmap"
- "Find outliers in the dataset"
- "Group by category and calculate mean sales"
- "Create a time series visualization"
- "What are the top 10 items by revenue?"
- "Show me missing value patterns"
- "Perform statistical analysis on numeric columns"
- "Compare distributions across different categories"

## Architecture

1. **Data Loader**: Handles CSV and SQL data ingestion
2. **RAG System**: Indexes and retrieves relevant context using sentence-transformers (local)
3. **Analyst Agent**: Generates Python/Pandas code using Google Gemini
4. **Code Executor**: Safely executes code with output capture
5. **Self-Correction**: Automatically fixes errors up to max retries

## Configuration

Edit `config.py` to customize:
- Model selection (gemini-1.5-pro or gemini-1.5-flash)
- Retry limits
- Output directories
- RAG parameters
- Embedding model

## Requirements

- Python 3.8+
- Google Gemini API key (free tier available)
- Internet connection for Gemini API calls
- Local compute for embeddings (no API costs)

## Safety

The code executor includes validation to prevent:
- File system access
- Network requests
- System command execution
- Dangerous imports
- Arbitrary code evaluation

## Advantages Over OpenAI/Anthropic Version

1. **Cost Effective**: Uses local embeddings (sentence-transformers) instead of OpenAI embeddings
2. **Gemini 1.5 Pro**: Access to Google's latest model with large context window
3. **Free Tier**: Gemini offers generous free tier for API usage
4. **No Multiple API Keys**: Only need Gemini API key

## Troubleshooting

If you encounter issues:

1. **API Key Error**: Make sure GEMINI_API_KEY is set in .env file
2. **Import Errors**: Run `pip install -r requirements.txt`
3. **Model Errors**: Check if you have access to gemini-1.5-pro (try gemini-1.5-flash instead)
4. **Memory Issues**: Reduce dataset size or use sampling for large files

## License

MIT License