class MySQLConfig:
    """MySQL Database Configuration"""
    
    # MySQL Connection Settings
    MYSQL_HOST = "localhost"
    MYSQL_PORT = 3306
    MYSQL_USER = "root"  # or "root"
    MYSQL_PASSWORD = "Hasnain123@"
    MYSQL_DATABASE = "sales_data"
    
    @classmethod
    def get_connection_string(cls):
        """Get MySQL connection string"""
        return f"mysql+pymysql://{cls.MYSQL_USER}:{cls.MYSQL_PASSWORD}@{cls.MYSQL_HOST}:{cls.MYSQL_PORT}/{cls.MYSQL_DATABASE}"