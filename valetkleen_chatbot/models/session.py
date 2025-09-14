"""Session model for tracking user sessions"""

from datetime import datetime
from .database import db
import json

class Session(db.Model):
    __tablename__ = 'sessions'
    
    id = db.Column(db.String(36), primary_key=True)  # UUID
    current_step = db.Column(db.String(50), default='welcome')
    conversation_history = db.Column(db.Text, default='[]')
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    cart_items = db.relationship('CartItem', backref='session', cascade='all, delete-orphan')
    customer = db.relationship('Customer', backref='session', uselist=False, cascade='all, delete-orphan')
    
    def add_to_history(self, entry):
        """Add entry to conversation history"""
        history = json.loads(self.conversation_history or '[]')
        history.append(entry)
        self.conversation_history = json.dumps(history)
        
    def get_history(self):
        """Get conversation history as list"""
        return json.loads(self.conversation_history or '[]')
    
    def to_dict(self):
        """Convert session to dictionary"""
        return {
            'session_id': self.id,
            'current_step': self.current_step,
            'conversation_history': self.get_history(),
            'cart': [item.to_dict() for item in self.cart_items],
            'customer_info': self.customer.to_dict() if self.customer else {},
            'created_at': self.created_at.isoformat() if self.created_at else None
        }