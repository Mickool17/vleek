"""Order model for tracking completed orders"""

from datetime import datetime
from .database import db
import json

class Order(db.Model):
    __tablename__ = 'orders'
    
    id = db.Column(db.Integer, primary_key=True)
    order_number = db.Column(db.String(20), unique=True, nullable=False)
    customer_id = db.Column(db.Integer, db.ForeignKey('customers.id'), nullable=False)
    items = db.Column(db.Text)  # JSON array of order items
    total_amount = db.Column(db.Float, nullable=False)
    status = db.Column(db.String(50), default='pending')  # pending, processing, completed, cancelled
    pickup_date = db.Column(db.String(50))
    pickup_time = db.Column(db.String(50))
    delivery_date = db.Column(db.String(50))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def get_items(self):
        """Get items as list"""
        return json.loads(self.items or '[]')
    
    def set_items(self, items_list):
        """Set items from list"""
        self.items = json.dumps(items_list or [])
    
    @staticmethod
    def generate_order_number():
        """Generate unique order number"""
        from datetime import datetime
        import random
        timestamp = datetime.now().strftime('%Y%m%d%H%M')
        random_suffix = str(random.randint(100, 999))
        return f"VK{timestamp}{random_suffix}"
    
    def to_dict(self):
        """Convert order to dictionary"""
        return {
            'order_number': self.order_number,
            'items': self.get_items(),
            'total_amount': self.total_amount,
            'status': self.status,
            'pickup_date': self.pickup_date,
            'pickup_time': self.pickup_time,
            'delivery_date': self.delivery_date,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }