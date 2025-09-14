"""Customer model for storing customer information"""

from datetime import datetime
from .database import db

class Customer(db.Model):
    __tablename__ = 'customers'
    
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.String(36), db.ForeignKey('sessions.id'), nullable=False)
    name = db.Column(db.String(100))
    email = db.Column(db.String(100))
    phone = db.Column(db.String(20))
    address = db.Column(db.Text)
    pickup_date = db.Column(db.String(50))
    pickup_time = db.Column(db.String(50))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationship
    orders = db.relationship('Order', backref='customer', lazy='dynamic')
    
    def to_dict(self):
        """Convert customer to dictionary"""
        return {
            'name': self.name,
            'email': self.email,
            'phone': self.phone,
            'address': self.address,
            'pickup_date': self.pickup_date,
            'pickup_time': self.pickup_time
        }