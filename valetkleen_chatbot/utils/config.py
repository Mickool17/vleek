"""Configuration management"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Application configuration"""
    
    # Flask Configuration
    SECRET_KEY = os.getenv('FLASK_SECRET_KEY', 'dev-secret-key-change-in-production')
    FLASK_ENV = os.getenv('FLASK_ENV', 'development')
    DEBUG = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    
    # Database Configuration
    SQLALCHEMY_DATABASE_URI = os.getenv('DATABASE_URL', 'sqlite:///valetkleen.db')
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # API Keys
    GROQ_API_KEY = os.getenv('GROQ_API_KEY')
    
    # Application Settings
    APP_HOST = os.getenv('APP_HOST', '0.0.0.0')
    APP_PORT = int(os.getenv('APP_PORT', 5000))
    
    # Business Logic Settings
    MAX_CART_ITEMS = 50
    SESSION_TIMEOUT_HOURS = 24
    PICKUP_DAYS_AHEAD = 7
    EXCLUDE_PICKUP_DAYS = [6]  # Sunday = 6
    
    # Time Slots
    PICKUP_TIME_SLOTS = [
        "8:00 AM - 10:00 AM",
        "10:00 AM - 12:00 PM",
        "12:00 PM - 2:00 PM",
        "2:00 PM - 4:00 PM",
        "4:00 PM - 6:00 PM"
    ]
    
    @classmethod
    def validate(cls):
        """Validate required configuration"""
        if not cls.GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY is required. Please set it in your .env file")
        
        if cls.FLASK_ENV == 'production' and cls.SECRET_KEY == 'dev-secret-key-change-in-production':
            raise ValueError("Please set a secure SECRET_KEY in production")
        
        return True