"""
Data Migration Script: MySQL to PostgreSQL
Migrates data from local MySQL to Render PostgreSQL
"""

import pandas as pd
from sqlalchemy import create_engine, text
import os
from dotenv import load_dotenv

load_dotenv()

# ============================================================================
# CONFIGURATION
# ============================================================================

# Source: Local MySQL
MYSQL_HOST = "localhost"
MYSQL_PORT = 3306
MYSQL_USER = "root"
MYSQL_PASSWORD = os.getenv("DB_PASSWORD", "your_local_password")  # Update this
MYSQL_DATABASE = "sales_data"  # Your local database name

# Destination: Render PostgreSQL
# Get this from Render dashboard after creating the database
POSTGRES_URL = os.getenv("DATABASE_URL")  # Set this in .env or paste directly

# Tables to migrate (add all your table names here)
TABLES_TO_MIGRATE = [
    "sales",
    # Add more table names as needed
]

# ============================================================================
# MIGRATION FUNCTIONS
# ============================================================================

def create_mysql_engine():
    """Create MySQL connection"""
    connection_string = f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DATABASE}"
    return create_engine(connection_string)

def create_postgres_engine():
    """Create PostgreSQL connection"""
    if not POSTGRES_URL:
        raise ValueError("DATABASE_URL not set! Get it from Render dashboard.")
    return create_engine(POSTGRES_URL)

def migrate_table(table_name, mysql_engine, postgres_engine):
    """Migrate a single table from MySQL to PostgreSQL"""
    print(f"\n📊 Migrating table: {table_name}")
    
    try:
        # Read from MySQL
        print(f"  ↓ Reading from MySQL...")
        df = pd.read_sql_table(table_name, mysql_engine)
        print(f"  ✓ Read {len(df)} rows")
        
        # Write to PostgreSQL
        print(f"  ↑ Writing to PostgreSQL...")
        df.to_sql(table_name, postgres_engine, if_exists='replace', index=False)
        print(f"  ✓ Migrated {len(df)} rows successfully!")
        
        return True
    except Exception as e:
        print(f"  ✗ Error migrating {table_name}: {e}")
        return False

def verify_migration(table_name, mysql_engine, postgres_engine):
    """Verify data was migrated correctly"""
    try:
        mysql_count = pd.read_sql(f"SELECT COUNT(*) as count FROM {table_name}", mysql_engine).iloc[0]['count']
        postgres_count = pd.read_sql(f"SELECT COUNT(*) as count FROM {table_name}", postgres_engine).iloc[0]['count']
        
        if mysql_count == postgres_count:
            print(f"  ✓ Verification passed: {mysql_count} rows in both databases")
            return True
        else:
            print(f"  ✗ Verification failed: MySQL={mysql_count}, PostgreSQL={postgres_count}")
            return False
    except Exception as e:
        print(f"  ✗ Verification error: {e}")
        return False

# ============================================================================
# MAIN MIGRATION
# ============================================================================

def main():
    print("=" * 60)
    print("🚀 MySQL to PostgreSQL Migration")
    print("=" * 60)
    
    # Create connections
    print("\n🔌 Connecting to databases...")
    try:
        mysql_engine = create_mysql_engine()
        postgres_engine = create_postgres_engine()
        print("✓ Connected to both databases")
    except Exception as e:
        print(f"✗ Connection failed: {e}")
        return
    
    # Migrate tables
    print(f"\n📦 Migrating {len(TABLES_TO_MIGRATE)} table(s)...")
    
    success_count = 0
    for table in TABLES_TO_MIGRATE:
        if migrate_table(table, mysql_engine, postgres_engine):
            if verify_migration(table, mysql_engine, postgres_engine):
                success_count += 1
    
    # Summary
    print("\n" + "=" * 60)
    print(f"✅ Migration Complete: {success_count}/{len(TABLES_TO_MIGRATE)} tables migrated")
    print("=" * 60)
    
    if success_count == len(TABLES_TO_MIGRATE):
        print("\n🎉 All tables migrated successfully!")
    else:
        print(f"\n⚠️  {len(TABLES_TO_MIGRATE) - success_count} table(s) failed to migrate")

if __name__ == "__main__":
    main()
