"""Order management service"""

from typing import Dict, List, Optional
from datetime import datetime
from ..models import db, Order, Customer, CartItem
import logging

class OrderService:
    """Service for managing orders"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def create_order(self, session_id: str) -> Dict:
        """Create order from cart"""
        try:
            # Get customer
            customer = Customer.query.filter_by(session_id=session_id).first()
            if not customer:
                return {
                    'success': False,
                    'error': 'Customer information not found'
                }
            
            # Get cart items
            cart_items = CartItem.query.filter_by(session_id=session_id).all()
            if not cart_items:
                return {
                    'success': False,
                    'error': 'Cart is empty'
                }
            
            # Calculate total
            total_amount = sum(item.total for item in cart_items)
            
            # Create order
            order = Order(
                order_number=Order.generate_order_number(),
                customer_id=customer.id,
                total_amount=total_amount,
                status='pending',
                pickup_date=customer.pickup_date,
                pickup_time=customer.pickup_time
            )
            
            # Convert cart items to order items
            order_items = []
            for item in cart_items:
                order_items.append({
                    'service_type': item.service_type,
                    'item_key': item.item_key,
                    'name': item.name,
                    'price': item.price,
                    'quantity': item.quantity,
                    'options': item.get_options(),
                    'total': item.total
                })
            
            order.set_items(order_items)
            
            db.session.add(order)
            
            # Clear cart after creating order
            for item in cart_items:
                db.session.delete(item)
            
            db.session.commit()
            
            return {
                'success': True,
                'order_number': order.order_number,
                'total': total_amount,
                'message': 'Order created successfully'
            }
            
        except Exception as e:
            db.session.rollback()
            self.logger.error(f"Error creating order: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_order(self, order_number: str) -> Optional[Dict]:
        """Get order by order number"""
        try:
            order = Order.query.filter_by(order_number=order_number).first()
            if order:
                return order.to_dict()
            return None
        except Exception as e:
            self.logger.error(f"Error getting order: {e}")
            return None
    
    def update_order_status(self, order_number: str, status: str) -> bool:
        """Update order status"""
        try:
            valid_statuses = ['pending', 'processing', 'completed', 'cancelled']
            if status not in valid_statuses:
                return False
            
            order = Order.query.filter_by(order_number=order_number).first()
            if order:
                order.status = status
                order.updated_at = datetime.utcnow()
                db.session.commit()
                return True
            return False
            
        except Exception as e:
            db.session.rollback()
            self.logger.error(f"Error updating order status: {e}")
            return False
    
    def get_customer_orders(self, customer_email: str) -> List[Dict]:
        """Get all orders for a customer"""
        try:
            customer = Customer.query.filter_by(email=customer_email).first()
            if customer:
                orders = Order.query.filter_by(customer_id=customer.id).all()
                return [order.to_dict() for order in orders]
            return []
        except Exception as e:
            self.logger.error(f"Error getting customer orders: {e}")
            return []
    
    def cancel_order(self, order_number: str) -> Dict:
        """Cancel an order"""
        try:
            order = Order.query.filter_by(order_number=order_number).first()
            if not order:
                return {
                    'success': False,
                    'error': 'Order not found'
                }
            
            if order.status in ['completed', 'cancelled']:
                return {
                    'success': False,
                    'error': f'Cannot cancel {order.status} order'
                }
            
            order.status = 'cancelled'
            order.updated_at = datetime.utcnow()
            db.session.commit()
            
            return {
                'success': True,
                'message': f'Order {order_number} cancelled successfully'
            }
            
        except Exception as e:
            db.session.rollback()
            self.logger.error(f"Error cancelling order: {e}")
            return {
                'success': False,
                'error': str(e)
            }