import pandas as pd
from sqlalchemy import create_engine, inspect
from typing import Dict, Any, List, Optional, Union
import logging
import numpy as np
from datetime import datetime
import warnings
from pypdf import PdfReader
import numpy as np
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

class DataLoader:
    """Enhanced data loader with intelligent preprocessing and validation"""
    
    def __init__(self):
        self.data = None
        self.connection = None
        self.metadata = {}
        self.data_profile = {}
    
    def load_csv(self, filepath: str, encoding: str = 'utf-8', 
                 auto_optimize: bool = True) -> pd.DataFrame:
        """Load data from CSV file with intelligent preprocessing"""
        try:
            logger.info(f"Loading CSV file: {filepath}")
            
            # Try different encodings if UTF-8 fails
            encodings_to_try = [encoding, 'utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
            
            for enc in encodings_to_try:
                try:
                    self.data = pd.read_csv(filepath, encoding=enc)
                    logger.info(f"Successfully loaded CSV with encoding: {enc}")
                    break
                except UnicodeDecodeError:
                    if enc == encodings_to_try[-1]:
                        raise
                    continue
            
            # Intelligent date parsing
            date_columns = self._auto_detect_dates()
            
            # Optimize data types
            if auto_optimize:
                self.data = self._optimize_datatypes()
            
            # Handle missing values intelligently
            self._analyze_missing_data()
            
            # Create metadata
            self.metadata = {
                "source": filepath,
                "type": "csv",
                "shape": self.data.shape,
                "columns": list(self.data.columns),
                "dtypes": {str(k): str(v) for k, v in self.data.dtypes.to_dict().items()},
                "date_columns": date_columns,
                "memory_usage_mb": round(self.data.memory_usage(deep=True).sum() / 1024**2, 2),
                "loaded_at": datetime.now().isoformat()
            }
            
            # Profile the data
            self._profile_data()
            
            logger.info(f"CSV loaded successfully: shape={self.data.shape}, "
                       f"memory={self.metadata['memory_usage_mb']}MB")
            
            return self.data
            
        except FileNotFoundError:
            error_msg = f"File not found: {filepath}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        except pd.errors.EmptyDataError:
            error_msg = f"Empty CSV file: {filepath}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        except Exception as e:
            logger.error(f"Error loading CSV: {str(e)}", exc_info=True)
            raise
    
    def load_sql(self, connection_string: str, query: str = None, 
                 table_name: str = None, auto_optimize: bool = True) -> pd.DataFrame:
        """Load data from SQL database with enhanced error handling"""
        try:
            logger.info(f"Connecting to database...")
            engine = create_engine(connection_string)
            self.connection = engine
            
            # Test connection
            with engine.connect() as conn:
                logger.info("Database connection successful")
            
            # Load data
            if query:
                logger.info(f"Executing custom query: {query[:100]}...")
                self.data = pd.read_sql(query, engine)
            elif table_name:
                logger.info(f"Loading table: {table_name}")
                
                # Check if table exists
                inspector = inspect(engine)
                available_tables = inspector.get_table_names()
                
                if table_name not in available_tables:
                    raise ValueError(
                        f"Table '{table_name}' not found. "
                        f"Available tables: {', '.join(available_tables)}"
                    )
                
                self.data = pd.read_sql_table(table_name, engine)
            else:
                raise ValueError("Either query or table_name must be provided")
            
            # Intelligent date parsing
            date_columns = self._auto_detect_dates()
            
            # Optimize data types
            if auto_optimize:
                self.data = self._optimize_datatypes()
            
            # Analyze missing data
            self._analyze_missing_data()
            
            # Create metadata
            self.metadata = {
                "source": connection_string.split('@')[-1] if '@' in connection_string else connection_string,
                "type": "sql",
                "table_name": table_name,
                "query": query[:200] if query else None,
                "shape": self.data.shape,
                "columns": list(self.data.columns),
                "dtypes": {str(k): str(v) for k, v in self.data.dtypes.to_dict().items()},
                "date_columns": date_columns,
                "memory_usage_mb": round(self.data.memory_usage(deep=True).sum() / 1024**2, 2),
                "loaded_at": datetime.now().isoformat()
            }
            
            # Profile the data
            self._profile_data()
            
            logger.info(f"SQL data loaded successfully: shape={self.data.shape}, "
                       f"memory={self.metadata['memory_usage_mb']}MB")
            
            return self.data
            
        except Exception as e:
            logger.error(f"Error loading SQL data: {str(e)}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"Error loading SQL data: {str(e)}", exc_info=True)
            raise
    
    def load_pdf(self, filepath: str) -> List[str]:
        """Load text from PDF file"""
        try:
            logger.info(f"Loading PDF file: {filepath}")
            reader = PdfReader(filepath)
            text_chunks = []
            
            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                if text.strip():
                    text_chunks.append(f"Page {i+1}:\n{text}")
            
            self.metadata = {
                "source": filepath,
                "type": "pdf",
                "pages": len(reader.pages),
                "loaded_at": datetime.now().isoformat()
            }
            
            logger.info(f"PDF loaded successfully: {len(text_chunks)} pages extracted")
            return text_chunks
            
        except Exception as e:
            logger.error(f"Error loading PDF: {str(e)}", exc_info=True)
            raise

    def _auto_detect_dates(self) -> List[str]:
        """Intelligently detect and parse date columns"""
        date_columns = []
        
        for col in self.data.columns:
            # Skip if already datetime
            if pd.api.types.is_datetime64_any_dtype(self.data[col]):
                date_columns.append(col)
                continue
            
            col_lower = col.lower()
            
            # Check column name patterns
            date_patterns = ['date', 'time', 'day', 'month', 'year', 'timestamp', 
                           'created', 'updated', 'modified', 'dt', 'period']
            
            if any(pattern in col_lower for pattern in date_patterns):
                try:
                    # Try parsing as datetime
                    original_dtype = self.data[col].dtype
                    parsed = pd.to_datetime(self.data[col], errors='coerce')
                    
                    # Check if parsing was successful (not all NaT)
                    valid_dates = parsed.notna().sum()
                    total_non_null = self.data[col].notna().sum()
                    
                    if valid_dates > 0 and (valid_dates / max(total_non_null, 1)) > 0.5:
                        self.data[col] = parsed
                        date_columns.append(col)
                        logger.info(f"Auto-parsed date column: {col} "
                                  f"({valid_dates}/{total_non_null} valid dates)")
                    else:
                        logger.debug(f"Column {col} looks like a date but couldn't parse reliably")
                        
                except Exception as e:
                    logger.debug(f"Could not parse {col} as date: {e}")
            
            # Also check data patterns for non-obvious date columns
            elif self.data[col].dtype == 'object':
                try:
                    sample = self.data[col].dropna().head(10)
                    if len(sample) > 0:
                        # Try to parse sample
                        parsed_sample = pd.to_datetime(sample, errors='coerce')
                        if parsed_sample.notna().sum() / len(sample) > 0.7:
                            # Looks like dates, parse full column
                            parsed = pd.to_datetime(self.data[col], errors='coerce')
                            valid_dates = parsed.notna().sum()
                            
                            if valid_dates > len(self.data) * 0.5:
                                self.data[col] = parsed
                                date_columns.append(col)
                                logger.info(f"Auto-detected date pattern in column: {col}")
                except:
                    pass
        
        return date_columns
    
    def _optimize_datatypes(self) -> pd.DataFrame:
        """Optimize memory usage by converting to appropriate dtypes"""
        logger.info("Optimizing data types for memory efficiency...")
        
        initial_memory = self.data.memory_usage(deep=True).sum() / 1024**2
        
        for col in self.data.columns:
            col_type = self.data[col].dtype
            
            # Skip datetime columns
            if pd.api.types.is_datetime64_any_dtype(col_type):
                continue
            
            # Optimize integers
            if pd.api.types.is_integer_dtype(col_type):
                c_min = self.data[col].min()
                c_max = self.data[col].max()
                
                if c_min >= 0:  # Unsigned integers
                    if c_max < 255:
                        self.data[col] = self.data[col].astype(np.uint8)
                    elif c_max < 65535:
                        self.data[col] = self.data[col].astype(np.uint16)
                    elif c_max < 4294967295:
                        self.data[col] = self.data[col].astype(np.uint32)
                else:  # Signed integers
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        self.data[col] = self.data[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        self.data[col] = self.data[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        self.data[col] = self.data[col].astype(np.int32)
            
            # Optimize floats
            # Skipped float32 conversion to maintain precision for financial/analytical accuracy
            # elif pd.api.types.is_float_dtype(col_type):
            #     self.data[col] = self.data[col].astype(np.float32)
            
            # Convert low-cardinality strings to category
            elif col_type == 'object':
                num_unique = self.data[col].nunique()
                num_total = len(self.data[col])
                
                if num_unique / num_total < 0.5:  # Less than 50% unique
                    self.data[col] = self.data[col].astype('category')
                    logger.debug(f"Converted {col} to category (cardinality: {num_unique})")
        
        final_memory = self.data.memory_usage(deep=True).sum() / 1024**2
        reduction = (1 - final_memory / initial_memory) * 100
        
        logger.info(f"Memory optimization: {initial_memory:.2f}MB → {final_memory:.2f}MB "
                   f"({reduction:.1f}% reduction)")
        
        return self.data
    
    def _analyze_missing_data(self):
        """Analyze patterns in missing data"""
        missing_info = {}
        
        for col in self.data.columns:
            missing_count = self.data[col].isnull().sum()
            if missing_count > 0:
                missing_info[col] = {
                    "count": int(missing_count),
                    "percentage": round(missing_count / len(self.data) * 100, 2)
                }
        
        self.metadata['missing_data'] = missing_info
        
        if missing_info:
            logger.info(f"Found missing data in {len(missing_info)} columns")
    
    def _profile_data(self):
        """Create comprehensive data profile"""
        profile = {
            "numeric_columns": [],
            "categorical_columns": [],
            "date_columns": [],
            "text_columns": [],
            "summary_stats": {}
        }
        
        for col in self.data.columns:
            dtype = self.data[col].dtype
            
            if pd.api.types.is_numeric_dtype(dtype):
                profile["numeric_columns"].append(col)
                
                # Get summary statistics
                profile["summary_stats"][col] = {
                    "min": float(self.data[col].min()) if not self.data[col].isna().all() else None,
                    "max": float(self.data[col].max()) if not self.data[col].isna().all() else None,
                    "mean": float(self.data[col].mean()) if not self.data[col].isna().all() else None,
                    "median": float(self.data[col].median()) if not self.data[col].isna().all() else None,
                    "std": float(self.data[col].std()) if not self.data[col].isna().all() else None,
                }
            
            elif pd.api.types.is_datetime64_any_dtype(dtype):
                profile["date_columns"].append(col)
                
                # Date range info
                profile["summary_stats"][col] = {
                    "min_date": str(self.data[col].min()) if not self.data[col].isna().all() else None,
                    "max_date": str(self.data[col].max()) if not self.data[col].isna().all() else None,
                    "date_range_days": int((self.data[col].max() - self.data[col].min()).days) 
                                      if not self.data[col].isna().all() else None
                }
            
            elif dtype == 'category' or self.data[col].nunique() < 50:
                profile["categorical_columns"].append(col)
                
                # Category distribution
                value_counts = self.data[col].value_counts()
                profile["summary_stats"][col] = {
                    "unique_values": int(self.data[col].nunique()),
                    "top_values": {str(k): v for k, v in value_counts.head(5).to_dict().items()},
                    "mode": str(self.data[col].mode()[0]) if len(self.data[col].mode()) > 0 else None
                }
            
            else:  # Text columns
                profile["text_columns"].append(col)
                profile["summary_stats"][col] = {
                    "unique_values": int(self.data[col].nunique()),
                }
        
        self.data_profile = profile
        logger.info(f"Data profiling complete: "
                   f"{len(profile['numeric_columns'])} numeric, "
                   f"{len(profile['categorical_columns'])} categorical, "
                   f"{len(profile['date_columns'])} date columns")
    
    def get_schema_info(self) -> str:
        """Get detailed schema information as formatted string"""
        if self.data is None:
            return "No data loaded"
        
        info = f"Dataset Shape: {self.data.shape[0]:,} rows × {self.data.shape[1]} columns\n"
        info += f"Memory Usage: {self.metadata.get('memory_usage_mb', 0):.2f} MB\n\n"
        
        info += "=" * 70 + "\n"
        info += "COLUMN INFORMATION\n"
        info += "=" * 70 + "\n\n"
        
        for col in self.data.columns:
            dtype = self.data[col].dtype
            null_count = self.data[col].isnull().sum()
            null_pct = (null_count / len(self.data)) * 100
            unique_count = self.data[col].nunique()
            
            info += f"Column: {col}\n"
            info += f"  Type: {dtype}\n"
            info += f"  Non-Null: {len(self.data) - null_count:,} ({100 - null_pct:.1f}%)\n"
            info += f"  Unique Values: {unique_count:,}\n"
            
            # Add statistics based on type
            if col in self.data_profile.get("summary_stats", {}):
                stats = self.data_profile["summary_stats"][col]
                
                if pd.api.types.is_numeric_dtype(dtype):
                    if stats.get("mean") is not None:
                        info += f"  Range: [{stats['min']:.2f}, {stats['max']:.2f}]\n"
                        info += f"  Mean: {stats['mean']:.2f}, Median: {stats['median']:.2f}\n"
                
                elif pd.api.types.is_datetime64_any_dtype(dtype):
                    if stats.get("min_date"):
                        info += f"  Date Range: {stats['min_date']} to {stats['max_date']}\n"
                        info += f"  Span: {stats['date_range_days']} days\n"
                
                elif unique_count < 50:
                    if "top_values" in stats and stats["top_values"]:
                        info += f"  Top Values: {list(stats['top_values'].keys())[:3]}\n"
            
            info += "\n"
        
        return info
    
    def get_sample_data(self, n: int = None) -> str:
        """
        Get sample rows as formatted string
        
        Args:
            n: Number of rows to display. If None, shows all rows.
        """
        if self.data is None:
            return "No data loaded"
        
        if n is None:
            # Show all rows
            sample = self.data
        else:
            sample = self.data.head(n)
        
        return sample.to_string()
    
    def get_data_profile(self) -> Dict:
        """Get the complete data profile"""
        return self.data_profile
    
    def get_available_tables(self) -> List[str]:
        """Get list of available tables from SQL connection"""
        if self.connection is None:
            return []
        
        try:
            inspector = inspect(self.connection)
            return inspector.get_table_names()
        except Exception as e:
            logger.error(f"Error getting table list: {e}")
            return []