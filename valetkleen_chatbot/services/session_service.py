"""Session management service"""

from typing import Dict, Optional
from datetime import datetime, timedelta
import uuid
from ..models import db, Session, Customer
from ..utils.config import Config
import logging

class SessionService:
    """Service for managing user sessions"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def create_session(self, session_id: str = None) -> Session:
        """Create new session"""
        try:
            if not session_id:
                session_id = str(uuid.uuid4())
            
            # Check if session already exists
            existing = Session.query.get(session_id)
            if existing:
                return existing
            
            session = Session(
                id=session_id,
                current_step='welcome'
            )
            
            db.session.add(session)
            db.session.commit()
            
            return session
            
        except Exception as e:
            db.session.rollback()
            self.logger.error(f"Error creating session: {e}")
            raise
    
    def get_session(self, session_id: str) -> Optional[Session]:
        """Get session by ID"""
        try:
            return Session.query.get(session_id)
        except Exception as e:
            self.logger.error(f"Error getting session: {e}")
            return None
    
    def update_step(self, session_id: str, step: str) -> bool:
        """Update current step in session"""
        try:
            session = Session.query.get(session_id)
            if session:
                session.current_step = step
                session.updated_at = datetime.utcnow()
                db.session.commit()
                return True
            return False
        except Exception as e:
            db.session.rollback()
            self.logger.error(f"Error updating session step: {e}")
            return False
    
    def save_customer_info(self, session_id: str, info: Dict) -> bool:
        """Save or update customer information"""
        try:
            session = Session.query.get(session_id)
            if not session:
                return False
            
            # Check if customer already exists for this session
            customer = Customer.query.filter_by(session_id=session_id).first()
            
            if not customer:
                customer = Customer(session_id=session_id)
                db.session.add(customer)
            
            # Update customer fields
            for field, value in info.items():
                if hasattr(customer, field):
                    setattr(customer, field, value)
            
            customer.updated_at = datetime.utcnow()
            db.session.commit()
            
            return True
            
        except Exception as e:
            db.session.rollback()
            self.logger.error(f"Error saving customer info: {e}")
            return False
    
    def get_customer_info(self, session_id: str) -> Optional[Dict]:
        """Get customer information for session"""
        try:
            customer = Customer.query.filter_by(session_id=session_id).first()
            return customer.to_dict() if customer else None
        except Exception as e:
            self.logger.error(f"Error getting customer info: {e}")
            return None
    
    def add_conversation_entry(self, session_id: str, 
                             role: str, message: str) -> bool:
        """Add entry to conversation history"""
        try:
            session = Session.query.get(session_id)
            if session:
                entry = {
                    'role': role,
                    'message': message,
                    'timestamp': datetime.utcnow().isoformat()
                }
                session.add_to_history(entry)
                db.session.commit()
                return True
            return False
        except Exception as e:
            db.session.rollback()
            self.logger.error(f"Error adding conversation entry: {e}")
            return False
    
    def cleanup_old_sessions(self) -> int:
        """Clean up sessions older than configured timeout"""
        try:
            cutoff_time = datetime.utcnow() - timedelta(
                hours=Config.SESSION_TIMEOUT_HOURS
            )
            
            old_sessions = Session.query.filter(
                Session.updated_at < cutoff_time
            ).all()
            
            count = len(old_sessions)
            
            for session in old_sessions:
                db.session.delete(session)
            
            db.session.commit()
            
            self.logger.info(f"Cleaned up {count} old sessions")
            return count
            
        except Exception as e:
            db.session.rollback()
            self.logger.error(f"Error cleaning up sessions: {e}")
            return 0
    
    def get_session_data(self, session_id: str) -> Optional[Dict]:
        """Get complete session data"""
        try:
            session = Session.query.get(session_id)
            if session:
                return session.to_dict()
            return None
        except Exception as e:
            self.logger.error(f"Error getting session data: {e}")
            return None