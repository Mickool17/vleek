"""Database configuration and initialization"""

from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate

db = SQLAlchemy()
migrate = None

def init_db(app):
    """Initialize database with Flask app"""
    global migrate
    db.init_app(app)
    migrate = Migrate(app, db)
    
    with app.app_context():
        db.create_all()
    
    return db