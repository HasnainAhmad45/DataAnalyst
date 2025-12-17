import pandas as pd
from sqlalchemy import create_engine, inspect
from typing import Dict, Any, Optional, List, Tuple
import logging
import os
import chardet
import csv
import json
import re
import time
import urllib.parse
from pathlib import Path
from collections import Counter

logger = logging.getLogger(__name__)

class DataLoader:
    """Highly intelligent data loader with advanced parsing and type inference"""
    
    def __init__(self):
        self.data = None
        self.connection = None
        self.metadata = {}
        # Common delimiters to try (ordered by frequency)
        self.delimiters = [',', ';', '\t', '|', ':', ' ', '\x00']
        # Extended encodings list with BOM handling
        self.encodings = [
            'utf-8-sig',  # UTF-8 with BOM (try first)
            'utf-8',      # UTF-8 without BOM
            'utf-16',     # UTF-16
            'utf-16le',   # UTF-16 Little Endian
            'utf-16be',   # UTF-16 Big Endian
            'latin-1',    # ISO-8859-1
            'iso-8859-1', # Alternative name
            'windows-1252', # Windows Western European
            'cp1252',     # Windows-1252 alternative
            'ascii',      # ASCII
            'gb2312',     # Chinese Simplified
            'gbk',        # Chinese Extended
            'shift_jis',  # Japanese
            'euc-kr',     # Korean
        ]
    
    def _detect_encoding(self, filepath: str) -> str:
        """Advanced encoding detection with BOM handling and multiple strategies"""
        # Check for BOM (Byte Order Mark) first
        try:
            with open(filepath, 'rb') as f:
                first_bytes = f.read(4)
                
                # Check for UTF-8 BOM
                if first_bytes.startswith(b'\xef\xbb\xbf'):
                    logger.info("Detected UTF-8 BOM")
                    return 'utf-8-sig'
                # Check for UTF-16 LE BOM
                elif first_bytes.startswith(b'\xff\xfe'):
                    logger.info("Detected UTF-16 LE BOM")
                    return 'utf-16le'
                # Check for UTF-16 BE BOM
                elif first_bytes.startswith(b'\xfe\xff'):
                    logger.info("Detected UTF-16 BE BOM")
                    return 'utf-16be'
        except Exception as e:
            logger.debug(f"BOM detection failed: {e}")
        
        # Use chardet for intelligent detection
        try:
            with open(filepath, 'rb') as f:
                raw_data = f.read(50000)  # Read more data for better detection
                result = chardet.detect(raw_data)
                encoding = result['encoding']
                confidence = result['confidence']
                
                if confidence > 0.75:
                    logger.info(f"Detected encoding: {encoding} (confidence: {confidence:.2f})")
                    # Verify the encoding works
                    try:
                        with open(filepath, 'r', encoding=encoding) as test_f:
                            test_f.read(1000)
                        return encoding
                    except (UnicodeDecodeError, UnicodeError):
                        logger.warning(f"Detected encoding {encoding} failed verification")
        except Exception as e:
            logger.debug(f"Chardet detection failed: {e}")
        
        # Fallback: try encodings in order of likelihood
        for enc in self.encodings:
            try:
                with open(filepath, 'r', encoding=enc) as f:
                    # Read more to ensure it works
                    sample = f.read(5000)
                    # Check if we can decode without errors
                    if sample:
                        logger.info(f"Successfully verified encoding: {enc}")
                        return enc
            except (UnicodeDecodeError, UnicodeError, LookupError):
                continue
        
        # Last resort: try with error handling
        logger.warning("Using UTF-8 with error handling as fallback")
        return 'utf-8'
    
    def _detect_delimiter(self, filepath: str, encoding: str) -> str:
        """Advanced delimiter detection with multiple strategies"""
        try:
            with open(filepath, 'r', encoding=encoding, errors='replace') as f:
                # Read multiple lines for better detection
                lines = [f.readline() for _ in range(20)]
                sample = ''.join(lines)
                
                if not sample.strip():
                    return ','
                
                # Strategy 1: Use CSV sniffer
                try:
                    sniffer = csv.Sniffer()
                    delimiter = sniffer.sniff(sample, delimiters=''.join(self.delimiters)).delimiter
                    logger.info(f"CSV sniffer detected delimiter: '{delimiter}'")
                    return delimiter
                except Exception:
                    pass
                
                # Strategy 2: Count delimiter occurrences
                delimiter_counts = {}
                for delim in self.delimiters:
                    if delim == ' ':  # Skip space as it's too common
                        continue
                    count = sample.count(delim)
                    if count > 0:
                        delimiter_counts[delim] = count
                
                if delimiter_counts:
                    # Find delimiter with most consistent occurrence per line
                    best_delimiter = max(delimiter_counts.items(), key=lambda x: x[1])[0]
                    logger.info(f"Frequency-based detection: '{best_delimiter}' ({delimiter_counts[best_delimiter]} occurrences)")
                    return best_delimiter
                
        except Exception as e:
            logger.warning(f"Delimiter detection failed: {e}")
        
        # Default to comma
        logger.info("Using comma as default delimiter")
        return ','
    
    def _detect_header_row(self, filepath: str, encoding: str, delimiter: str) -> Optional[int]:
        """Detect if file has a header row"""
        try:
            with open(filepath, 'r', encoding=encoding) as f:
                first_line = f.readline().strip()
                second_line = f.readline().strip()
                
                # Check if first line looks like headers (non-numeric, unique values)
                first_cols = first_line.split(delimiter)
                second_cols = second_line.split(delimiter)
                
                # If first line has non-numeric values and second line has numeric values, first is likely header
                try:
                    # Try to convert first line to numbers
                    float(first_cols[0])
                    # If successful, no header
                    return None
                except (ValueError, IndexError):
                    # First line likely contains headers
                    return 0
        except Exception as e:
            logger.warning(f"Header detection failed: {e}, assuming header in first row")
            return 0
    
    def _clean_numeric_string(self, value: str) -> str:
        """Clean numeric strings by removing common formatting"""
        if pd.isna(value) or value == '':
            return value
        value_str = str(value).strip()
        # Remove currency symbols, thousands separators, etc.
        value_str = re.sub(r'[^\d\.\-\+]', '', value_str)
        return value_str
    
    def _detect_date_formats(self, series: pd.Series) -> List[str]:
        """Detect common date formats in a series"""
        common_formats = [
            '%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y', '%Y/%m/%d',
            '%d-%m-%Y', '%m-%d-%Y', '%Y.%m.%d', '%d.%m.%Y',
            '%Y-%m-%d %H:%M:%S', '%m/%d/%Y %H:%M:%S',
            '%d/%m/%Y %H:%M:%S', '%Y-%m-%dT%H:%M:%S',
            '%Y-%m-%d %H:%M:%S.%f', '%Y%m%d', '%d%m%Y'
        ]
        return common_formats
    
    def _smart_type_conversion(self, df: pd.DataFrame) -> tuple:
        """Advanced intelligent type conversion with multiple strategies"""
        date_columns = []
        numeric_columns = []
        boolean_columns = []
        
        for col in df.columns:
            if len(df) == 0:
                continue
                
            col_lower = str(col).lower()
            original_dtype = df[col].dtype
            
            # Skip if already properly typed
            if original_dtype in [pd.Int64Dtype(), pd.Float64Dtype(), 'int64', 'float64', 'bool']:
                continue
            
            # Strategy 1: Detect and convert boolean columns
            if df[col].dtype == 'object':
                unique_vals = df[col].dropna().unique()
                if len(unique_vals) <= 2:
                    # Check if values look like booleans
                    str_vals = [str(v).lower().strip() for v in unique_vals]
                    bool_patterns = [
                        {'true', 'false'}, {'yes', 'no'}, {'y', 'n'},
                        {'1', '0'}, {'t', 'f'}, {'on', 'off'}
                    ]
                    for pattern in bool_patterns:
                        if set(str_vals).issubset(pattern):
                            df[col] = df[col].astype(str).str.lower().str.strip().map({
                                'true': True, 'false': False,
                                'yes': True, 'no': False,
                                'y': True, 'n': False,
                                '1': True, '0': False,
                                't': True, 'f': False,
                                'on': True, 'off': False
                            })
                            boolean_columns.append(col)
                            logger.info(f"Converted column '{col}' to boolean")
                            break
            
            # Strategy 2: Detect and convert date columns
            date_keywords = ['date', 'time', 'timestamp', 'created', 'updated', 'modified', 
                           'dob', 'birth', 'start', 'end', 'expiry', 'expire']
            
            if any(keyword in col_lower for keyword in date_keywords) or df[col].dtype == 'object':
                try:
                    # Try pandas automatic conversion first
                    converted = pd.to_datetime(df[col], errors='coerce', infer_datetime_format=True)
                    success_rate = converted.notna().sum() / len(df) if len(df) > 0 else 0
                    
                    if success_rate > 0.6:  # 60% success threshold
                        df[col] = converted
                        date_columns.append(col)
                        logger.info(f"Converted column '{col}' to datetime ({success_rate:.1%} success rate)")
                    elif success_rate > 0.3:  # Try manual format detection
                        # Try common date formats
                        for fmt in self._detect_date_formats(df[col]):
                            try:
                                converted = pd.to_datetime(df[col], format=fmt, errors='coerce')
                                if converted.notna().sum() / len(df) > 0.6:
                                    df[col] = converted
                                    date_columns.append(col)
                                    logger.info(f"Converted column '{col}' to datetime using format {fmt}")
                                    break
                            except:
                                continue
                except Exception as e:
                    logger.debug(f"Date conversion failed for '{col}': {e}")
            
            # Strategy 3: Convert numeric columns (only if not already converted to date/boolean)
            if col not in date_columns and col not in boolean_columns and df[col].dtype == 'object':
                try:
                    # First, try direct conversion
                    numeric_series = pd.to_numeric(df[col], errors='coerce')
                    success_rate = numeric_series.notna().sum() / len(df) if len(df) > 0 else 0
                    
                    if success_rate > 0.85:  # 85% success threshold for direct conversion
                        df[col] = numeric_series
                        numeric_columns.append(col)
                        logger.info(f"Converted column '{col}' to numeric ({success_rate:.1%} success rate)")
                    elif success_rate > 0.5:  # Try cleaning the data first
                        # Clean numeric strings
                        cleaned = df[col].apply(self._clean_numeric_string)
                        numeric_series = pd.to_numeric(cleaned, errors='coerce')
                        new_success_rate = numeric_series.notna().sum() / len(df) if len(df) > 0 else 0
                        
                        if new_success_rate > 0.8:
                            df[col] = numeric_series
                            numeric_columns.append(col)
                            logger.info(f"Converted column '{col}' to numeric after cleaning ({new_success_rate:.1%} success rate)")
                except Exception as e:
                    logger.debug(f"Numeric conversion failed for '{col}': {e}")
        
        return df, date_columns
    
    def load_csv(self, filepath: str, **kwargs) -> pd.DataFrame:
        """Intelligently load data from CSV file with auto-detection"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        
        logger.info(f"Loading CSV file: {filepath}")
        
        # Detect encoding
        encoding = kwargs.get('encoding') or self._detect_encoding(filepath)
        
        # Try multiple strategies to load the CSV
        # Strategy 1: Use C engine (faster, default) with detected encoding and delimiter
        # Strategy 2: Use Python engine (more flexible) if C engine fails
        
        detected_delimiter = self._detect_delimiter(filepath, encoding)
        
        # Build comprehensive strategies with different quote and escape handling
        strategies = [
            # Strategy 1: C engine with detected delimiter (fastest, standard quotes)
            {
                'encoding': encoding,
                'delimiter': detected_delimiter,
                'engine': 'c',
                'low_memory': False,
                'quotechar': '"',
                'quoting': csv.QUOTE_MINIMAL
            },
            # Strategy 2: C engine with comma delimiter
            {
                'encoding': encoding,
                'delimiter': ',',
                'engine': 'c',
                'low_memory': False,
                'quotechar': '"',
                'quoting': csv.QUOTE_MINIMAL
            },
            # Strategy 3: Python engine with detected delimiter (more flexible)
            {
                'encoding': encoding,
                'delimiter': detected_delimiter,
                'engine': 'python',
                'quotechar': '"',
                'quoting': csv.QUOTE_MINIMAL
            },
            # Strategy 4: Python engine with comma delimiter
            {
                'encoding': encoding,
                'delimiter': ',',
                'engine': 'python',
                'quotechar': '"',
                'quoting': csv.QUOTE_MINIMAL
            },
            # Strategy 5: Python engine with semicolon (common in European CSVs)
            {
                'encoding': encoding,
                'delimiter': ';',
                'engine': 'python',
                'quotechar': '"',
                'quoting': csv.QUOTE_MINIMAL
            },
            # Strategy 6: Python engine with tab delimiter
            {
                'encoding': encoding,
                'delimiter': '\t',
                'engine': 'python',
                'quotechar': '"',
                'quoting': csv.QUOTE_MINIMAL
            },
            # Strategy 7: Python engine with no quotes (for files without quoting)
            {
                'encoding': encoding,
                'delimiter': detected_delimiter,
                'engine': 'python',
                'quoting': csv.QUOTE_NONE
            },
            # Strategy 8: Python engine with single quotes
            {
                'encoding': encoding,
                'delimiter': detected_delimiter,
                'engine': 'python',
                'quotechar': "'",
                'quoting': csv.QUOTE_MINIMAL
            },
        ]
        
        # Add on_bad_lines for pandas >= 1.3.0 (only for python engine)
        try:
            import pandas as pd
            if hasattr(pd, '__version__'):
                version_parts = pd.__version__.split('.')
                if int(version_parts[0]) > 1 or (int(version_parts[0]) == 1 and int(version_parts[1]) >= 3):
                    # Add to python engine strategies only
                    for strategy in strategies:
                        if strategy.get('engine') == 'python':
                            strategy['on_bad_lines'] = 'skip'
        except:
            pass  # Fallback if version check fails
        
        last_error = None
        loaded_successfully = False
        
        for i, strategy in enumerate(strategies):
            try:
                logger.info(f"Trying strategy {i+1}: encoding={strategy['encoding']}, delimiter='{strategy['delimiter']}', engine={strategy.get('engine', 'c')}")
                
                # Filter out kwargs that conflict with strategy
                filtered_kwargs = {k: v for k, v in kwargs.items() if k not in strategy}
                self.data = pd.read_csv(filepath, **strategy, **filtered_kwargs)
                
                # If we got data (even if empty, it's a valid load), break
                if self.data is not None:
                    logger.info(f"Successfully loaded CSV using strategy {i+1} - Shape: {self.data.shape}")
                    loaded_successfully = True
                    break
            except Exception as e:
                last_error = e
                logger.warning(f"Strategy {i+1} failed: {e}")
                continue
        
        if not loaded_successfully or self.data is None:
            error_msg = f"Failed to load CSV file after trying {len(strategies)} strategies."
            if last_error:
                error_msg += f" Last error: {last_error}"
            raise ValueError(error_msg)
        
        # Check if dataframe is empty
        if len(self.data) == 0:
            logger.warning("CSV file loaded but contains no data rows")
        
        # Advanced column name cleaning and duplicate handling
        self.data.columns = self._clean_column_names(self.data.columns)
        
        # Data quality checks
        self._perform_data_quality_checks()
        
        # Smart type conversion
        self.data, date_columns = self._smart_type_conversion(self.data)
        
        # Store comprehensive metadata
        self.metadata = {
            "source": filepath,
            "type": "csv",
            "shape": self.data.shape,
            "columns": list(self.data.columns),
            "dtypes": {str(k): str(v) for k, v in self.data.dtypes.to_dict().items()},
            "date_columns": date_columns,
            "encoding": encoding,
            "delimiter": detected_delimiter,
            "null_counts": {col: int(self.data[col].isnull().sum()) for col in self.data.columns},
            "null_percentages": {col: float(self.data[col].isnull().sum() / len(self.data) * 100) 
                                for col in self.data.columns if len(self.data) > 0},
            "unique_counts": {col: int(self.data[col].nunique()) for col in self.data.columns},
            "memory_usage_mb": float(self.data.memory_usage(deep=True).sum() / 1024**2),
            "duplicate_rows": int(self.data.duplicated().sum()),
            "load_timestamp": pd.Timestamp.now().isoformat()
        }
        
        logger.info(f"Successfully loaded CSV: {filepath} with shape {self.data.shape}")
        return self.data
    
    def load_excel(self, filepath: str, sheet_name: Optional[str] = None, **kwargs) -> pd.DataFrame:
        """Load data from Excel file (.xlsx, .xls)"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        
        logger.info(f"Loading Excel file: {filepath}")
        
        try:
            # Try to read Excel file
            if sheet_name:
                self.data = pd.read_excel(filepath, sheet_name=sheet_name, **kwargs)
            else:
                # Try to read first sheet
                excel_file = pd.ExcelFile(filepath)
                sheet_name = excel_file.sheet_names[0]
                logger.info(f"Reading first sheet: {sheet_name}")
                self.data = pd.read_excel(filepath, sheet_name=sheet_name, **kwargs)
            
            # Clean column names
            self.data.columns = self._clean_column_names(self.data.columns)
            
            # Data quality checks
            self._perform_data_quality_checks()
            
            # Smart type conversion
            self.data, date_columns = self._smart_type_conversion(self.data)
            
            self.metadata = {
                "source": filepath,
                "type": "excel",
                "sheet": sheet_name,
                "shape": self.data.shape,
                "columns": list(self.data.columns),
                "dtypes": {str(k): str(v) for k, v in self.data.dtypes.to_dict().items()},
                "date_columns": date_columns,
                "null_counts": {col: int(self.data[col].isnull().sum()) for col in self.data.columns},
                "null_percentages": {col: float(self.data[col].isnull().sum() / len(self.data) * 100) 
                                    for col in self.data.columns if len(self.data) > 0},
                "unique_counts": {col: int(self.data[col].nunique()) for col in self.data.columns},
                "memory_usage_mb": float(self.data.memory_usage(deep=True).sum() / 1024**2),
                "duplicate_rows": int(self.data.duplicated().sum()),
                "load_timestamp": pd.Timestamp.now().isoformat()
            }
            
            logger.info(f"Successfully loaded Excel: {filepath} with shape {self.data.shape}")
            return self.data
        except Exception as e:
            logger.error(f"Error loading Excel file: {e}")
            raise
    
    def load_json(self, filepath: str, **kwargs) -> pd.DataFrame:
        """Load data from JSON file"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        
        logger.info(f"Loading JSON file: {filepath}")
        
        try:
            # Try different JSON formats
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                data = json.loads(content)
            
            # Handle different JSON structures
            if isinstance(data, list):
                self.data = pd.json_normalize(data)
            elif isinstance(data, dict):
                # If it's a dict with a list, try to find it
                for key, value in data.items():
                    if isinstance(value, list):
                        self.data = pd.json_normalize(value)
                        break
                else:
                    # If no list found, normalize the dict itself
                    self.data = pd.json_normalize([data])
            else:
                raise ValueError(f"Unsupported JSON structure: {type(data)}")
            
            # Clean column names
            self.data.columns = self._clean_column_names(self.data.columns)
            
            # Data quality checks
            self._perform_data_quality_checks()
            
            # Smart type conversion
            self.data, date_columns = self._smart_type_conversion(self.data)
            
            self.metadata = {
                "source": filepath,
                "type": "json",
                "shape": self.data.shape,
                "columns": list(self.data.columns),
                "dtypes": {str(k): str(v) for k, v in self.data.dtypes.to_dict().items()},
                "date_columns": date_columns
            }
            
            logger.info(f"Successfully loaded JSON: {filepath} with shape {self.data.shape}")
            return self.data
        except Exception as e:
            logger.error(f"Error loading JSON file: {e}")
            raise
    
    def _clean_column_names(self, columns: pd.Index) -> pd.Index:
        """Clean column names and handle duplicates"""
        cleaned = []
        seen = {}
        
        for col in columns:
            # Clean the column name
            clean_col = str(col).strip()
            # Replace spaces and dots with underscores
            clean_col = re.sub(r'[^\w]', '_', clean_col)
            # Remove multiple underscores
            clean_col = re.sub(r'_+', '_', clean_col)
            # Remove leading/trailing underscores
            clean_col = clean_col.strip('_')
            # Handle empty column names
            if not clean_col or clean_col == '':
                clean_col = 'unnamed_column'
            
            # Handle duplicates
            original_col = clean_col
            counter = 1
            while clean_col in seen:
                clean_col = f"{original_col}_{counter}"
                counter += 1
            
            seen[clean_col] = True
            cleaned.append(clean_col)
        
        return pd.Index(cleaned)
    
    def _perform_data_quality_checks(self):
        """Perform data quality checks and log warnings"""
        if self.data is None or len(self.data) == 0:
            return
        
        quality_issues = []
        
        # Check for completely empty columns
        empty_cols = [col for col in self.data.columns if self.data[col].isna().all()]
        if empty_cols:
            quality_issues.append(f"Completely empty columns: {empty_cols}")
            logger.warning(f"Found {len(empty_cols)} completely empty columns")
        
        # Check for high null percentage
        for col in self.data.columns:
            null_pct = self.data[col].isna().sum() / len(self.data) * 100
            if null_pct > 50:
                quality_issues.append(f"Column '{col}' has {null_pct:.1f}% null values")
                logger.warning(f"Column '{col}' has {null_pct:.1f}% null values")
        
        # Check for duplicate rows
        duplicate_count = self.data.duplicated().sum()
        if duplicate_count > 0:
            duplicate_pct = duplicate_count / len(self.data) * 100
            quality_issues.append(f"Found {duplicate_count} duplicate rows ({duplicate_pct:.1f}%)")
            logger.warning(f"Found {duplicate_count} duplicate rows")
        
        # Store quality issues in metadata
        if quality_issues:
            self.metadata['quality_issues'] = quality_issues
    
    def _detect_database_type(self, connection_string: str) -> str:
        """Detect database type from connection string"""
        conn_lower = connection_string.lower()
        if 'mysql' in conn_lower or 'mariadb' in conn_lower:
            return 'mysql'
        elif 'postgresql' in conn_lower or 'postgres' in conn_lower:
            return 'postgresql'
        elif 'sqlite' in conn_lower:
            return 'sqlite'
        elif 'mssql' in conn_lower or 'sqlserver' in conn_lower:
            return 'mssql'
        elif 'oracle' in conn_lower:
            return 'oracle'
        else:
            return 'unknown'
    
    def _check_database_driver(self, database_type: str) -> bool:
        """Check if required database driver is installed"""
        try:
            if database_type == 'mysql':
                import pymysql
                return True
            elif database_type == 'postgresql':
                import psycopg2
                return True
            elif database_type == 'sqlite':
                # sqlite3 is built-in
                return True
            elif database_type == 'mssql':
                try:
                    import pyodbc
                    return True
                except ImportError:
                    return False
            return True  # Unknown type, let it try
        except ImportError:
            return False
    
    def load_sql(self, connection_string: str, query: str = None, table_name: str = None, 
                 database_type: str = None, **kwargs) -> pd.DataFrame:
        """Intelligently load data from SQL database with support for multiple database types"""
        logger.info(f"Loading data from SQL database")
        
        try:
            # Detect database type if not provided
            if not database_type:
                database_type = self._detect_database_type(connection_string)
                logger.info(f"Detected database type: {database_type}")
            
            # Check if required driver is installed
            if not self._check_database_driver(database_type):
                driver_map = {
                    'mysql': 'pymysql',
                    'postgresql': 'psycopg2-binary',
                    'mssql': 'pyodbc'
                }
                driver_name = driver_map.get(database_type, 'database driver')
                error_msg = (
                    f"Required database driver '{driver_name}' is not installed. "
                    f"Please install it using: pip install {driver_name}"
                )
                logger.error(error_msg)
                raise ImportError(error_msg)
            
            # Build connection string with proper driver if needed
            if database_type == 'mssql' and '://' in connection_string and 'mssql' not in connection_string.lower():
                # Add proper MSSQL driver
                if 'mssql+pyodbc' not in connection_string.lower():
                    connection_string = connection_string.replace('mssql://', 'mssql+pyodbc://')
            
            # Connection retry logic
            max_retries = 3
            retry_delay = 1  # seconds
            
            for attempt in range(max_retries):
                try:
                    # Create engine with connection pooling
                    engine = create_engine(
                        connection_string,
                        pool_pre_ping=True,  # Verify connections before using
                        pool_recycle=3600,   # Recycle connections after 1 hour
                        connect_args={'connect_timeout': 10},  # 10 second timeout
                        **kwargs
                    )
                    self.connection = engine
                    
                    # Test connection with retry
                    connection_success = False
                    for conn_attempt in range(3):
                        try:
                            try:
                                from sqlalchemy import text
                                with engine.connect() as conn:
                                    conn.execute(text("SELECT 1"))
                            except ImportError:
                                # Fallback for older SQLAlchemy versions
                                with engine.connect() as conn:
                                    conn.execute("SELECT 1")
                            connection_success = True
                            break
                        except Exception as conn_e:
                            if conn_attempt < 2:
                                logger.warning(f"Connection test failed (attempt {conn_attempt + 1}/3): {conn_e}")
                                time.sleep(retry_delay)
                            else:
                                raise
                    
                    if connection_success:
                        logger.info("Database connection established successfully")
                        break
                        
                except Exception as e:
                    if attempt < max_retries - 1:
                        logger.warning(f"Connection attempt {attempt + 1}/{max_retries} failed: {e}. Retrying...")
                        time.sleep(retry_delay * (attempt + 1))  # Exponential backoff
                    else:
                        logger.error(f"Failed to connect after {max_retries} attempts")
                        raise
            else:
                raise ConnectionError(f"Failed to establish database connection after {max_retries} attempts")
            
            # Load data (after successful connection)
            if query:
                logger.info(f"Executing SQL query")
                try:
                    from sqlalchemy import text
                    self.data = pd.read_sql(text(query), engine)
                except ImportError:
                    # Older SQLAlchemy version
                    self.data = pd.read_sql(query, engine)
            elif table_name:
                logger.info(f"Loading table: {table_name}")
                # Try to get all data from table
                try:
                    self.data = pd.read_sql_table(table_name, engine)
                except Exception as e:
                    # Fallback to SELECT * query
                    logger.warning(f"read_sql_table failed: {e}, using SELECT query")
                    try:
                        from sqlalchemy import text
                        self.data = pd.read_sql(text(f"SELECT * FROM {table_name}"), engine)
                    except ImportError:
                        # Older SQLAlchemy version
                        self.data = pd.read_sql(f"SELECT * FROM {table_name}", engine)
            else:
                # If no query or table, try to list tables
                inspector = inspect(engine)
                tables = inspector.get_table_names()
                if tables:
                    logger.info(f"No query/table specified, loading first table: {tables[0]}")
                    try:
                        from sqlalchemy import text
                        self.data = pd.read_sql(text(f"SELECT * FROM {tables[0]}"), engine)
                    except ImportError:
                        # Older SQLAlchemy version
                        self.data = pd.read_sql(f"SELECT * FROM {tables[0]}", engine)
                    table_name = tables[0]
                else:
                    raise ValueError("No tables found in database. Please specify a query or table_name.")
            
            # Smart type conversion for dates
            self.data, date_columns = self._smart_type_conversion(self.data)
            
            self.metadata = {
                "source": connection_string,
                "type": "sql",
                "database_type": database_type,
                "table_name": table_name,
                "shape": self.data.shape,
                "columns": list(self.data.columns),
                "dtypes": {str(k): str(v) for k, v in self.data.dtypes.to_dict().items()},
                "date_columns": date_columns,
                "null_counts": {col: int(self.data[col].isnull().sum()) for col in self.data.columns},
                "null_percentages": {col: float(self.data[col].isnull().sum() / len(self.data) * 100) 
                                    for col in self.data.columns if len(self.data) > 0},
                "unique_counts": {col: int(self.data[col].nunique()) for col in self.data.columns},
                "memory_usage_mb": float(self.data.memory_usage(deep=True).sum() / 1024**2),
                "duplicate_rows": int(self.data.duplicated().sum()),
                "load_timestamp": pd.Timestamp.now().isoformat()
            }
            
            logger.info(f"Successfully loaded SQL data with shape {self.data.shape}")
            return self.data
        except Exception as e:
            error_msg = str(e)
            
            # Provide user-friendly error messages for common database errors
            if "1045" in error_msg or "Access denied" in error_msg:
                friendly_msg = (
                    "❌ Database Authentication Failed\n\n"
                    "The username or password is incorrect. Please check:\n"
                    "• Username is correct\n"
                    "• Password is correct (check for typos)\n"
                    "• User has access to the specified database\n"
                    "• Special characters in password are properly encoded\n\n"
                    f"Technical details: {error_msg}"
                )
                logger.error(friendly_msg)
                raise ConnectionError(friendly_msg)
            elif "2003" in error_msg or "Can't connect" in error_msg:
                friendly_msg = (
                    "❌ Database Connection Failed\n\n"
                    "Cannot connect to the database server. Please check:\n"
                    "• Database server is running\n"
                    "• Host and port are correct\n"
                    "• Firewall allows connections\n"
                    "• Network connectivity is available\n\n"
                    f"Technical details: {error_msg}"
                )
                logger.error(friendly_msg)
                raise ConnectionError(friendly_msg)
            elif "1049" in error_msg or "Unknown database" in error_msg:
                friendly_msg = (
                    "❌ Database Not Found\n\n"
                    "The specified database does not exist. Please check:\n"
                    "• Database name is correct\n"
                    "• Database has been created\n"
                    "• User has permissions to access the database\n\n"
                    f"Technical details: {error_msg}"
                )
                logger.error(friendly_msg)
                raise ConnectionError(friendly_msg)
            elif "1146" in error_msg or "Table doesn't exist" in error_msg:
                friendly_msg = (
                    "❌ Table Not Found\n\n"
                    "The specified table does not exist in the database. Please check:\n"
                    "• Table name is correct\n"
                    "• Table exists in the database\n"
                    "• Use correct case sensitivity\n\n"
                    f"Technical details: {error_msg}"
                )
                logger.error(friendly_msg)
                raise ConnectionError(friendly_msg)
            else:
                logger.error(f"Error loading SQL data: {error_msg}")
                raise
    
    def load_file(self, filepath: str, **kwargs) -> pd.DataFrame:
        """Intelligently load any supported file type based on extension"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        
        file_ext = Path(filepath).suffix.lower()
        
        if file_ext in ['.csv', '.tsv']:
            return self.load_csv(filepath, **kwargs)
        elif file_ext in ['.xlsx', '.xls']:
            return self.load_excel(filepath, **kwargs)
        elif file_ext in ['.json', '.jsonl']:
            return self.load_json(filepath, **kwargs)
        else:
            # Try CSV as default
            logger.warning(f"Unknown file extension {file_ext}, trying CSV format")
            return self.load_csv(filepath, **kwargs)
    
    def get_schema_info(self) -> str:
        """Get detailed schema information"""
        if self.data is None:
            return "No data loaded"
        
        info = f"Dataset Shape: {self.data.shape}\n\n"
        info += "Column Information:\n"
        info += "-" * 50 + "\n"
        
        for col in self.data.columns:
            dtype = self.data[col].dtype
            null_count = self.data[col].isnull().sum()
            unique_count = self.data[col].nunique()
            
            info += f"Column: {col}\n"
            info += f"  Type: {dtype}\n"
            info += f"  Null Count: {null_count}\n"
            info += f"  Unique Values: {unique_count}\n"
            
            if dtype in ['int64', 'float64']:
                info += f"  Min: {self.data[col].min()}\n"
                info += f"  Max: {self.data[col].max()}\n"
                info += f"  Mean: {self.data[col].mean():.2f}\n"
            
            info += "\n"
        
        return info
    
    def get_sample_data(self, n: int = 5) -> str:
        """Get sample rows as string"""
        if self.data is None:
            return "No data loaded"
        return self.data.head(n).to_string()