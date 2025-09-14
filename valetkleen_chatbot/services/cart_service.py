"""Cart management service with full CRUD operations"""

from typing import Dict, List, Optional
from ..models import db, CartItem, Session
import logging

class CartService:
    """Service for managing shopping cart operations"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def add_item(self, session_id: str, service_type: str, item_key: str, 
                 name: str, price: float, quantity: int = 1, 
                 options: List[str] = None) -> Dict:
        """Add item to cart"""
        try:
            # Check if session exists
            session = Session.query.get(session_id)
            if not session:
                return {'success': False, 'error': 'Session not found'}
            
            # Check if item already exists in cart
            existing_item = CartItem.query.filter_by(
                session_id=session_id,
                service_type=service_type,
                item_key=item_key
            ).first()
            
            if existing_item:
                # Update quantity instead of adding duplicate
                existing_item.quantity += quantity
                existing_item.calculate_total()
            else:
                # Create new cart item
                cart_item = CartItem(
                    session_id=session_id,
                    service_type=service_type,
                    item_key=item_key,
                    name=name,
                    price=price,
                    quantity=quantity,
                    total=price * quantity
                )
                
                if options:
                    cart_item.set_options(options)
                
                db.session.add(cart_item)
            
            db.session.commit()
            
            return {
                'success': True,
                'message': f'Added {quantity}x {name} to cart',
                'cart_total': self.get_cart_total(session_id)
            }
            
        except Exception as e:
            db.session.rollback()
            self.logger.error(f"Error adding item to cart: {e}")
            return {'success': False, 'error': str(e)}
    
    def remove_item(self, session_id: str, item_id: int) -> Dict:
        """Remove specific item from cart"""
        try:
            cart_item = CartItem.query.filter_by(
                session_id=session_id,
                id=item_id
            ).first()
            
            if not cart_item:
                return {'success': False, 'error': 'Item not found in cart'}
            
            item_name = cart_item.name
            db.session.delete(cart_item)
            db.session.commit()
            
            return {
                'success': True,
                'message': f'Removed {item_name} from cart',
                'cart_total': self.get_cart_total(session_id)
            }
            
        except Exception as e:
            db.session.rollback()
            self.logger.error(f"Error removing item from cart: {e}")
            return {'success': False, 'error': str(e)}
    
    def update_quantity(self, session_id: str, item_id: int, 
                       new_quantity: int) -> Dict:
        """Update item quantity in cart"""
        try:
            if new_quantity < 1:
                return self.remove_item(session_id, item_id)
            
            cart_item = CartItem.query.filter_by(
                session_id=session_id,
                id=item_id
            ).first()
            
            if not cart_item:
                return {'success': False, 'error': 'Item not found in cart'}
            
            cart_item.quantity = new_quantity
            cart_item.calculate_total()
            db.session.commit()
            
            return {
                'success': True,
                'message': f'Updated {cart_item.name} quantity to {new_quantity}',
                'cart_total': self.get_cart_total(session_id)
            }
            
        except Exception as e:
            db.session.rollback()
            self.logger.error(f"Error updating item quantity: {e}")
            return {'success': False, 'error': str(e)}
    
    def clear_cart(self, session_id: str) -> Dict:
        """Clear all items from cart"""
        try:
            CartItem.query.filter_by(session_id=session_id).delete()
            db.session.commit()
            
            return {
                'success': True,
                'message': 'Cart cleared successfully'
            }
            
        except Exception as e:
            db.session.rollback()
            self.logger.error(f"Error clearing cart: {e}")
            return {'success': False, 'error': str(e)}
    
    def get_cart_items(self, session_id: str) -> List[Dict]:
        """Get all items in cart"""
        try:
            items = CartItem.query.filter_by(session_id=session_id).all()
            return [item.to_dict() for item in items]
        except Exception as e:
            self.logger.error(f"Error getting cart items: {e}")
            return []
    
    def get_cart_total(self, session_id: str) -> float:
        """Calculate total cart value"""
        try:
            items = CartItem.query.filter_by(session_id=session_id).all()
            return sum(item.total for item in items)
        except Exception as e:
            self.logger.error(f"Error calculating cart total: {e}")
            return 0.0
    
    def get_cart_summary(self, session_id: str) -> Dict:
        """Get complete cart summary"""
        try:
            items = self.get_cart_items(session_id)
            total = self.get_cart_total(session_id)
            
            return {
                'items': items,
                'item_count': len(items),
                'total_items': sum(item['quantity'] for item in items),
                'total': total,
                'formatted_total': f"${total:.2f}"
            }
        except Exception as e:
            self.logger.error(f"Error getting cart summary: {e}")
            return {
                'items': [],
                'item_count': 0,
                'total_items': 0,
                'total': 0.0,
                'formatted_total': "$0.00"
            }
    
    def apply_discount(self, session_id: str, discount_code: str) -> Dict:
        """Apply discount code to cart (future feature)"""
        # Placeholder for discount functionality
        discounts = {
            'FIRST10': 0.10,  # 10% off
            'WELCOME15': 0.15,  # 15% off
            'VIP20': 0.20  # 20% off
        }
        
        if discount_code.upper() in discounts:
            discount_rate = discounts[discount_code.upper()]
            original_total = self.get_cart_total(session_id)
            discount_amount = original_total * discount_rate
            new_total = original_total - discount_amount
            
            return {
                'success': True,
                'message': f'Discount {discount_code} applied',
                'discount_amount': discount_amount,
                'original_total': original_total,
                'new_total': new_total
            }
        else:
            return {
                'success': False,
                'error': 'Invalid discount code'
            }