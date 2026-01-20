"""
Quick script to check PostgreSQL tables
"""
import pandas as pd
from sqlalchemy import create_engine, inspect
import os
from dotenv import load_dotenv

load_dotenv()

# Get PostgreSQL URL from .env
POSTGRES_URL = os.getenv("DATABASE_URL")

if not POSTGRES_URL:
    print("❌ DATABASE_URL not found in .env file")
    exit(1)

# Connect
print("🔌 Connecting to PostgreSQL...")
engine = create_engine(POSTGRES_URL)

# List all tables
print("\n📋 Tables in database:")
inspector = inspect(engine)
tables = inspector.get_table_names()

if not tables:
    print("  ⚠️  No tables found!")
else:
    for table in tables:
        print(f"  ✓ {table}")
        
        # Get row count
        try:
            count = pd.read_sql(f"SELECT COUNT(*) as count FROM {table}", engine).iloc[0]['count']
            print(f"    → {count} rows")
            
            # Show first 3 rows
            df = pd.read_sql(f"SELECT * FROM {table} LIMIT 3", engine)
            print(f"    → Columns: {', '.join(df.columns.tolist())}")
            print()
        except Exception as e:
            print(f"    ✗ Error reading table: {e}\n")

print("✅ Check complete!")
