"""Cart item model for tracking items in cart"""

from datetime import datetime
from .database import db
import json

class CartItem(db.Model):
    __tablename__ = 'cart_items'
    
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.String(36), db.ForeignKey('sessions.id'), nullable=False)
    service_type = db.Column(db.String(50), nullable=False)
    item_key = db.Column(db.String(100), nullable=False)
    name = db.Column(db.String(200), nullable=False)
    price = db.Column(db.Float, nullable=False)
    quantity = db.Column(db.Integer, default=1)
    options = db.Column(db.Text, default='[]')  # JSON array of selected options
    total = db.Column(db.Float, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def get_options(self):
        """Get options as list"""
        return json.loads(self.options or '[]')
    
    def set_options(self, options_list):
        """Set options from list"""
        self.options = json.dumps(options_list or [])
    
    def calculate_total(self):
        """Calculate and update total price"""
        self.total = self.price * self.quantity
        return self.total
    
    def to_dict(self):
        """Convert cart item to dictionary"""
        return {
            'id': self.id,
            'service_type': self.service_type,
            'item_key': self.item_key,
            'name': self.name,
            'price': self.price,
            'quantity': self.quantity,
            'options': self.get_options(),
            'total': self.total
        }