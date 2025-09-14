"""
ValetKleen Professional Chatbot System - Render Compatible Version
Enhanced with all V2 fixes but in a single file for deployment compatibility
"""

import json
import re
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging

# NLP and ML imports
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Groq LLM integration
from groq import Groq

# Web framework
from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import uuid

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('punkt_tab', quiet=True)
except:
    pass

# Input validation functions
def validate_email(email: str) -> Tuple[bool, str]:
    """Validate email address format"""
    if not email:
        return False, "Email is required"
    
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    
    if not re.match(pattern, email):
        return False, "Please enter a valid email address (e.g., john@example.com)"
    
    if email.count('@') != 1:
        return False, "Email must contain exactly one @ symbol"
    
    local, domain = email.split('@')
    
    if len(local) > 64:
        return False, "Email username is too long"
    
    if len(domain) > 255:
        return False, "Email domain is too long"
    
    return True, ""

def validate_phone(phone: str) -> Tuple[bool, str]:
    """Validate phone number format"""
    if not phone:
        return False, "Phone number is required"
    
    # Remove common formatting characters
    cleaned = re.sub(r'[\s\-\(\)\.]', '', phone)
    
    # US phone number patterns
    us_pattern = r'^(\+1)?[2-9]\d{9}$'
    intl_pattern = r'^\+\d{10,15}$'
    
    if re.match(us_pattern, cleaned) or re.match(intl_pattern, cleaned):
        return True, ""
    elif len(cleaned) < 10:
        return False, "Phone number is too short. Please include area code."
    elif len(cleaned) > 15:
        return False, "Phone number is too long."
    else:
        return False, "Please enter a valid phone number (e.g., 555-123-4567)"

def validate_address(address: str) -> Tuple[bool, str]:
    """Validate physical address"""
    if not address:
        return False, "Address is required"
    
    if len(address.strip()) < 10:
        return False, "Please enter a complete address"
    
    has_number = any(char.isdigit() for char in address)
    if not has_number:
        return False, "Please include a street number in your address"
    
    return True, ""

def sanitize_input(text: str, max_length: int = 500) -> str:
    """Sanitize user input"""
    if not text:
        return ""
    
    # Remove HTML/script tags
    text = re.sub(r'<[^>]*>', '', text)
    
    # Limit length
    text = text[:max_length]
    
    # Remove excessive whitespace
    text = ' '.join(text.split())
    
    return text.strip()

class ValetKleenChatbotRender:
    def __init__(self):
        """Initialize the chatbot"""
        
        self.logger = logging.getLogger(__name__)
        
        # Get API key from environment
        api_key = os.getenv('GROQ_API_KEY')
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable is required")
            
        self.groq_client = Groq(api_key=api_key)
        
        # Initialize NLP components
        self.lemmatizer = WordNetLemmatizer()
        try:
            self.stop_words = set(stopwords.words('english'))
        except:
            self.stop_words = set()
        
        # Load knowledge base
        self.knowledge_base = self.load_knowledge_base()
        
        # Initialize TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        self.prepare_intent_matching()
        
        # Service catalog
        self.service_catalog = self.load_service_catalog()
        
        # In-memory storage (enhanced with IDs for cart management)
        self.customer_sessions = {}
        self.cart_item_counter = 0
        
        self.logger.info("ValetKleen Chatbot initialized successfully!")
    
    def load_knowledge_base(self) -> Dict:
        """Load knowledge base from files"""
        try:
            data_path = "enhanced_wordpress_data/chatbot_training_data.json"
            if not os.path.exists(data_path):
                data_path = "wordpress_scraped_data/chatbot_training_data.json"
            
            if os.path.exists(data_path):
                with open(data_path, 'r', encoding='utf-8') as f:
                    scraped_data = json.load(f)
                
                knowledge = {
                    'about': "",
                    'services': "",
                    'faq': "",
                    'contact': "",
                    'process': "",
                    'pricing': "",
                    'all_content': []
                }
                
                for item in scraped_data:
                    content = item.get('content', '').lower()
                    title = item.get('title', '').lower()
                    
                    if any(word in title for word in ['about', 'about us']):
                        knowledge['about'] += f" {item.get('content', '')}"
                    elif any(word in title for word in ['service', 'dry cleaning', 'laundry']):
                        knowledge['services'] += f" {item.get('content', '')}"
                    
                    knowledge['all_content'].append({
                        'title': item.get('title', ''),
                        'content': item.get('content', ''),
                        'type': item.get('type', ''),
                        'url': item.get('url', '')
                    })
                
                return knowledge
                
        except Exception as e:
            self.logger.error(f"Error loading knowledge base: {e}")
        
        return {
            'about': "ValetKleen provides convenient pickup and delivery laundry services.",
            'services': "We offer dry cleaning, laundry, pickup and delivery services.",
            'all_content': []
        }
    
    def load_service_catalog(self) -> Dict:
        """Load service catalog"""
        return {
            'dry_cleaning': {
                'name': 'Dry Cleaning Services',
                'description': 'Professional dry cleaning for specialty items',
                'items': {
                    'office_shirt': {'name': 'Office Shirt (Dry-Clean)', 'price': 5.50, 'options': []},
                    'pants': {'name': 'Pants', 'price': 7.50, 'options': ['Crease', 'No crease']},
                    'dress_standard': {'name': 'Standard Dress', 'price': 12.00, 'options': []},
                    'coat_short': {'name': 'Short Coat', 'price': 12.00, 'options': []},
                    'wedding_dress': {'name': 'Wedding Dress', 'price': 180.00, 'options': ['Box', 'No box']}
                }
            },
            'laundry': {
                'name': 'Laundry Services',
                'description': 'Full laundry service with wash and fold',
                'items': {
                    'bag_small': {'name': 'Small Bag (12 lb)', 'price': 22.00, 'options': []},
                    'bag_medium': {'name': 'Medium Bag (18 lb)', 'price': 33.00, 'options': []},
                    'bag_large': {'name': 'Large Bag (25 lb)', 'price': 46.00, 'options': []},
                    'comforter_queen': {'name': 'Comforter (Queen/King)', 'price': 30.00, 'options': []},
                    'blanket_queen': {'name': 'Blanket (Queen/King)', 'price': 25.00, 'options': []}
                }
            }
        }
    
    def prepare_intent_matching(self):
        """Prepare intent matching"""
        self.intents = {
            'greeting': ['hello', 'hi', 'hey', 'good morning'],
            'place_order': ['place order', 'make order', 'book service'],
            'view_cart': ['view cart', 'show cart', 'my cart', 'cart items'],
            'remove_item': ['remove item', 'delete item', 'remove from cart'],
            'update_quantity': ['update quantity', 'change quantity', 'modify amount'],
            'clear_cart': ['clear cart', 'empty cart', 'remove all'],
            'checkout': ['checkout', 'complete order', 'finish order']
        }
        
        self.intent_texts = []
        self.intent_labels = []
        
        for intent, phrases in self.intents.items():
            for phrase in phrases:
                self.intent_texts.append(phrase)
                self.intent_labels.append(intent)
        
        try:
            self.intent_vectors = self.vectorizer.fit_transform(self.intent_texts)
        except:
            self.intent_vectors = None
    
    def detect_intent(self, user_input: str) -> Tuple[str, float]:
        """Detect user intent"""
        user_input_lower = user_input.lower()
        
        # Direct keyword matching for cart operations
        if any(keyword in user_input_lower for keyword in ['view cart', 'show cart', 'my cart']):
            return 'view_cart', 0.95
        elif any(keyword in user_input_lower for keyword in ['remove item', 'delete item']):
            return 'remove_item', 0.95
        elif any(keyword in user_input_lower for keyword in ['update quantity', 'change quantity']):
            return 'update_quantity', 0.95
        elif any(keyword in user_input_lower for keyword in ['clear cart', 'empty cart']):
            return 'clear_cart', 0.95
        elif any(keyword in user_input_lower for keyword in ['checkout', 'complete order']):
            return 'checkout', 0.95
        elif any(keyword in user_input_lower for keyword in ['place order', 'make order']):
            return 'place_order', 0.95
        
        return 'unknown', 0.0
    
    def create_customer_session(self, session_id: str = None) -> str:
        """Create or get customer session"""
        if not session_id:
            session_id = str(uuid.uuid4())
        
        if session_id not in self.customer_sessions:
            self.customer_sessions[session_id] = {
                'cart': [],
                'customer_info': {},
                'conversation_history': [],
                'current_step': 'welcome',
                'created_at': datetime.now().isoformat()
            }
        
        return session_id
    
    def add_to_cart(self, session_id: str, service_type: str, item_key: str, 
                   quantity: int = 1, selected_options: List[str] = None) -> Dict:
        """Add item to cart with enhanced management"""
        if session_id not in self.customer_sessions:
            return {'success': False, 'error': 'Session not found'}
        
        if service_type not in self.service_catalog:
            return {'success': False, 'error': 'Invalid service type'}
        
        if item_key not in self.service_catalog[service_type]['items']:
            return {'success': False, 'error': 'Item not found'}
        
        item_info = self.service_catalog[service_type]['items'][item_key]
        selected_options = selected_options or []
        
        # Generate unique cart item ID
        self.cart_item_counter += 1
        cart_item_id = self.cart_item_counter
        
        cart_item = {
            'id': cart_item_id,
            'service_type': service_type,
            'item_key': item_key,
            'name': item_info['name'],
            'price': item_info['price'],
            'quantity': quantity,
            'options': selected_options,
            'total': item_info['price'] * quantity
        }
        
        self.customer_sessions[session_id]['cart'].append(cart_item)
        
        return {
            'success': True,
            'message': f'Added {quantity}x {item_info["name"]} to cart',
            'cart_total': self.get_cart_total(session_id)
        }
    
    def remove_from_cart(self, session_id: str, item_id: int) -> Dict:
        """Remove specific item from cart"""
        if session_id not in self.customer_sessions:
            return {'success': False, 'error': 'Session not found'}
        
        cart = self.customer_sessions[session_id]['cart']
        
        for i, item in enumerate(cart):
            if item['id'] == item_id:
                removed_item = cart.pop(i)
                return {
                    'success': True,
                    'message': f'Removed {removed_item["name"]} from cart',
                    'cart_total': self.get_cart_total(session_id)
                }
        
        return {'success': False, 'error': 'Item not found in cart'}
    
    def update_cart_quantity(self, session_id: str, item_id: int, new_quantity: int) -> Dict:
        """Update item quantity in cart"""
        if session_id not in self.customer_sessions:
            return {'success': False, 'error': 'Session not found'}
        
        if new_quantity < 1:
            return self.remove_from_cart(session_id, item_id)
        
        cart = self.customer_sessions[session_id]['cart']
        
        for item in cart:
            if item['id'] == item_id:
                item['quantity'] = new_quantity
                item['total'] = item['price'] * new_quantity
                return {
                    'success': True,
                    'message': f'Updated {item["name"]} quantity to {new_quantity}',
                    'cart_total': self.get_cart_total(session_id)
                }
        
        return {'success': False, 'error': 'Item not found in cart'}
    
    def clear_cart(self, session_id: str) -> Dict:
        """Clear all items from cart"""
        if session_id not in self.customer_sessions:
            return {'success': False, 'error': 'Session not found'}
        
        self.customer_sessions[session_id]['cart'] = []
        
        return {
            'success': True,
            'message': 'Cart cleared successfully'
        }
    
    def get_cart_total(self, session_id: str) -> float:
        """Calculate total cart value"""
        if session_id not in self.customer_sessions:
            return 0.0
        
        cart = self.customer_sessions[session_id]['cart']
        return sum(item['total'] for item in cart)
    
    def get_cart_summary(self, session_id: str) -> str:
        """Generate cart summary"""
        session = self.customer_sessions.get(session_id, {})
        cart = session.get('cart', [])
        
        if not cart:
            return "🛒 Your cart is empty."
        
        summary = "🛒 **YOUR CART:**\n\n"
        total = 0
        
        for item in cart:
            item_total = item['total']
            total += item_total
            
            summary += f"**#{item['id']}** {item['quantity']}x {item['name']}\n"
            if item['options']:
                summary += f"   Options: {', '.join(item['options'])}\n"
            summary += f"   Price: ${item['price']:.2f} each\n"
            summary += f"   Subtotal: ${item_total:.2f}\n\n"
        
        summary += f"**TOTAL: ${total:.2f}**\n\n"
        summary += "To remove: 'Remove item #ID'\nTo update: 'Update item #ID to 3'"
        
        return summary
    
    def generate_response(self, user_input: str, session_id: str = None) -> Dict:
        """Generate chatbot response"""
        
        # Sanitize input
        user_input = sanitize_input(user_input)
        
        # Create or get session
        if not session_id:
            session_id = self.create_customer_session()
        else:
            self.create_customer_session(session_id)
        
        session = self.customer_sessions[session_id]
        
        # Add to conversation history
        session['conversation_history'].append({
            'user': user_input,
            'timestamp': datetime.now().isoformat()
        })
        
        # Check for specific cart commands first
        user_input_lower = user_input.lower()
        
        # Handle view cart
        if any(keyword in user_input_lower for keyword in ['view cart', 'show cart', 'my cart']):
            cart_summary = self.get_cart_summary(session_id)
            cart = session.get('cart', [])
            
            if not cart:
                response = {
                    'message': "🛒 Your cart is empty. Let's add some items!",
                    'type': 'cart_empty',
                    'suggestions': ["Place an Order", "View Services"]
                }
            else:
                response = {
                    'message': cart_summary,
                    'type': 'cart_view',
                    'suggestions': ["Checkout", "Add More Items", "Clear Cart", "Remove Item"]
                }
        
        # Handle remove item
        elif any(keyword in user_input_lower for keyword in ['remove item', 'delete item']):
            numbers = re.findall(r'\d+', user_input)
            if numbers:
                item_id = int(numbers[0])
                result = self.remove_from_cart(session_id, item_id)
                if result['success']:
                    response = {
                        'message': f"✅ {result['message']}\n\nNew Total: ${result['cart_total']:.2f}",
                        'type': 'cart_update',
                        'suggestions': ["View Cart", "Checkout", "Add More Items"]
                    }
                else:
                    response = {
                        'message': f"❌ {result['error']}",
                        'type': 'error',
                        'suggestions': ["View Cart"]
                    }
            else:
                response = {
                    'message': "Please specify item ID to remove (e.g., 'Remove item #1')",
                    'type': 'error',
                    'suggestions': ["View Cart"]
                }
        
        # Handle update quantity
        elif any(keyword in user_input_lower for keyword in ['update item', 'change quantity']):
            numbers = re.findall(r'\d+', user_input)
            if len(numbers) >= 2:
                item_id = int(numbers[0])
                new_quantity = int(numbers[1])
                result = self.update_cart_quantity(session_id, item_id, new_quantity)
                if result['success']:
                    response = {
                        'message': f"✅ {result['message']}\n\nNew Total: ${result['cart_total']:.2f}",
                        'type': 'cart_update',
                        'suggestions': ["View Cart", "Checkout", "Add More Items"]
                    }
                else:
                    response = {
                        'message': f"❌ {result['error']}",
                        'type': 'error',
                        'suggestions': ["View Cart"]
                    }
            else:
                response = {
                    'message': "Please specify item ID and new quantity (e.g., 'Update item #1 to 3')",
                    'type': 'error',
                    'suggestions': ["View Cart"]
                }
        
        # Handle clear cart
        elif any(keyword in user_input_lower for keyword in ['clear cart', 'empty cart']):
            result = self.clear_cart(session_id)
            response = {
                'message': "✅ Cart cleared successfully!",
                'type': 'cart_cleared',
                'suggestions': ["Place an Order", "View Services"]
            }
        
        # Handle checkout
        elif any(keyword in user_input_lower for keyword in ['checkout', 'complete order', 'proceed to checkout']):
            response = self.handle_checkout(session_id)
        
        # Handle order flow steps
        elif session.get('current_step') == 'collecting_info':
            response = self.handle_info_collection(user_input, session_id)
        elif session.get('current_step') == 'selecting_service':
            response = self.handle_service_selection(user_input, session_id)
        elif session.get('current_step') == 'selecting_items':
            response = self.handle_item_selection(user_input, session_id)
        
        # Handle other intents
        elif any(keyword in user_input_lower for keyword in ['place order', 'make order', 'start order']):
            response = self.start_order_process(session_id)
        elif any(keyword in user_input_lower for keyword in ['hello', 'hi', 'hey']):
            response = self.handle_greeting()
        else:
            response = self.handle_general_inquiry(user_input)
        
        # Add bot response to history
        session['conversation_history'].append({
            'bot': response.get('message', ''),
            'timestamp': datetime.now().isoformat()
        })
        
        response['session_id'] = session_id
        return response
    
    def handle_greeting(self) -> Dict:
        """Handle greeting"""
        return {
            'message': "👋 Welcome to ValetKleen! I'm here to help with laundry and dry cleaning.\n\nHow can I assist you?",
            'type': 'greeting',
            'suggestions': [
                "📋 Place an Order",
                "🛒 View Cart", 
                "💰 Pricing Information",
                "📞 Contact Information"
            ]
        }
    
    def start_order_process(self, session_id: str) -> Dict:
        """Start order process"""
        session = self.customer_sessions[session_id]
        session['current_step'] = 'collecting_info'
        
        return {
            'message': "🛍️ Let's start your order!\n\nFirst, I'll need some information.\n\n👤 **Your Name:**",
            'type': 'info_collection',
            'collecting': 'name'
        }
    
    def handle_info_collection(self, user_input: str, session_id: str) -> Dict:
        """Handle customer information collection with validation"""
        session = self.customer_sessions[session_id]
        customer_info = session.get('customer_info', {})
        
        if 'name' not in customer_info:
            customer_info['name'] = user_input.strip()
            session['customer_info'] = customer_info
            return {
                'message': f"Thank you, {customer_info['name']}! 📧 **Your Email:**",
                'type': 'info_collection',
                'collecting': 'email'
            }
        
        elif 'email' not in customer_info:
            is_valid, error_msg = validate_email(user_input.strip())
            if not is_valid:
                return {
                    'message': f"❌ {error_msg}\n\nPlease enter a valid email:",
                    'type': 'info_collection',
                    'collecting': 'email'
                }
            
            customer_info['email'] = user_input.strip()
            session['customer_info'] = customer_info
            return {
                'message': "Perfect! 🏠 **Your Address (for pickup & delivery):**",
                'type': 'info_collection',
                'collecting': 'address'
            }
        
        elif 'address' not in customer_info:
            is_valid, error_msg = validate_address(user_input.strip())
            if not is_valid:
                return {
                    'message': f"❌ {error_msg}\n\nPlease enter your complete address:",
                    'type': 'info_collection',
                    'collecting': 'address'
                }
            
            customer_info['address'] = user_input.strip()
            session['customer_info'] = customer_info
            return {
                'message': "Great! 📱 **Your Phone Number:**",
                'type': 'info_collection',
                'collecting': 'phone'
            }
        
        elif 'phone' not in customer_info:
            is_valid, error_msg = validate_phone(user_input.strip())
            if not is_valid:
                return {
                    'message': f"❌ {error_msg}\n\nPlease enter your phone number:",
                    'type': 'info_collection',
                    'collecting': 'phone'
                }
            
            customer_info['phone'] = user_input.strip()
            session['customer_info'] = customer_info
            
            # Generate pickup dates (next 7 days, excluding Sunday)
            available_dates = []
            today = datetime.now()
            for i in range(1, 8):
                date = today + timedelta(days=i)
                if date.weekday() != 6:  # Exclude Sunday
                    available_dates.append(date.strftime("%A, %B %d, %Y"))
            
            return {
                'message': "📅 **When would you like pickup?**\n\nAvailable dates:",
                'type': 'info_collection',
                'collecting': 'pickup_date',
                'suggestions': available_dates[:5]
            }
        
        elif 'pickup_date' not in customer_info:
            customer_info['pickup_date'] = user_input.strip()
            session['customer_info'] = customer_info
            
            time_slots = [
                "8:00 AM - 10:00 AM",
                "10:00 AM - 12:00 PM", 
                "12:00 PM - 2:00 PM",
                "2:00 PM - 4:00 PM",
                "4:00 PM - 6:00 PM"
            ]
            
            return {
                'message': f"🕐 **What time works best on {customer_info['pickup_date']}?**",
                'type': 'info_collection',
                'collecting': 'pickup_time',
                'suggestions': time_slots
            }
        
        elif 'pickup_time' not in customer_info:
            customer_info['pickup_time'] = user_input.strip()
            session['customer_info'] = customer_info
            session['current_step'] = 'selecting_service'
            
            return {
                'message': f"Perfect! Pickup scheduled for **{customer_info['pickup_date']}** at **{customer_info['pickup_time']}**.\n\nWhich service do you need?",
                'type': 'service_selection',
                'suggestions': ["👔 Dry Cleaning", "🧺 Laundry Services"]
            }
    
    def handle_service_selection(self, user_input: str, session_id: str) -> Dict:
        """Handle service selection"""
        session = self.customer_sessions[session_id]
        
        if 'dry' in user_input.lower():
            session['selected_service'] = 'dry_cleaning'
            session['current_step'] = 'selecting_items'
            return self.show_service_menu('dry_cleaning')
        elif 'laundry' in user_input.lower():
            session['selected_service'] = 'laundry'
            session['current_step'] = 'selecting_items'
            return self.show_service_menu('laundry')
        else:
            return {
                'message': "Please select a service:",
                'type': 'service_selection',
                'suggestions': ["👔 Dry Cleaning", "🧺 Laundry Services"]
            }
    
    def show_service_menu(self, service_type: str) -> Dict:
        """Show service menu"""
        service = self.service_catalog.get(service_type, {})
        items = service.get('items', {})
        
        menu_text = f"**{service.get('name', '')}**\n\n"
        suggestions = []
        
        for i, (key, item) in enumerate(items.items(), 1):
            menu_text += f"{i}. **{item['name']}** - ${item['price']:.2f}\n"
            if i <= 6:
                suggestions.append(f"{i}. {item['name']}")
        
        menu_text += "\n💬 Say 'Add 2 shirts' or select from menu:"
        
        return {
            'message': menu_text,
            'type': 'item_selection',
            'service': service_type,
            'suggestions': suggestions + ["View Cart", "Checkout"]
        }
    
    def handle_item_selection(self, user_input: str, session_id: str) -> Dict:
        """Handle item selection"""
        session = self.customer_sessions[session_id]
        service_type = session.get('selected_service', 'dry_cleaning')
        
        # Extract quantity
        quantity = 1
        numbers = re.findall(r'\d+', user_input)
        if numbers:
            quantity = int(numbers[0])
        
        # Find matching item
        for item_key, item_info in self.service_catalog[service_type]['items'].items():
            if any(word in user_input.lower() for word in item_key.split('_')):
                # Add to cart
                result = self.add_to_cart(session_id, service_type, item_key, quantity)
                
                if result['success']:
                    return {
                        'message': f"✅ {result['message']}\n\n**Cart Total: ${result['cart_total']:.2f}**\n\nWhat would you like to do?",
                        'type': 'cart_update',
                        'suggestions': [
                            "Add More Items",
                            "View Cart", 
                            "Checkout",
                            "Clear Cart"
                        ]
                    }
        
        return {
            'message': "Sorry, I couldn't find that item. Please try again or select from menu.",
            'type': 'item_selection',
            'suggestions': ["View Menu", "View Cart"]
        }
    
    def handle_checkout(self, session_id: str) -> Dict:
        """Handle checkout"""
        session = self.customer_sessions.get(session_id, {})
        cart = session.get('cart', [])
        
        if not cart:
            return {
                'message': "Your cart is empty! Please add items first.",
                'type': 'error',
                'suggestions': ["Place an Order"]
            }
        
        # Calculate total
        total = sum(item['total'] for item in cart)
        customer_info = session.get('customer_info', {})
        
        # Generate order number
        order_number = f"VK{datetime.now().strftime('%Y%m%d%H%M')}{len(cart):02d}"
        
        # Create order summary
        order_summary = f"🎉 **ORDER CONFIRMED!**\n\n"
        order_summary += f"**Order #:** {order_number}\n"
        order_summary += f"**Total:** ${total:.2f}\n\n"
        order_summary += f"**Pickup Details:**\n"
        order_summary += f"Date: {customer_info.get('pickup_date', 'TBD')}\n"
        order_summary += f"Time: {customer_info.get('pickup_time', 'TBD')}\n"
        order_summary += f"Address: {customer_info.get('address', 'TBD')}\n\n"
        order_summary += "✅ We'll pick up your items as scheduled!\n"
        order_summary += "📧 Confirmation email sent!\n\n"
        order_summary += "Thank you for choosing ValetKleen!"
        
        # Clear cart after successful checkout
        session['cart'] = []
        session['current_step'] = 'welcome'
        
        return {
            'message': order_summary,
            'type': 'checkout_success',
            'order_number': order_number,
            'suggestions': ["Place Another Order", "View Services"]
        }
    
    def handle_general_inquiry(self, user_input: str) -> Dict:
        """Handle general inquiries"""
        return {
            'message': "I can help you with:\n• Placing orders\n• Managing your cart\n• Pricing information\n• Service details\n\nWhat would you like to do?",
            'type': 'help',
            'suggestions': [
                "Place an Order",
                "View Cart",
                "Pricing Info", 
                "Our Services"
            ]
        }

# Create Flask application
app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'valetkleen_chatbot_secret_key_2024')
CORS(app)

# Initialize chatbot
chatbot = ValetKleenChatbotRender()

# HTML template
CHATBOT_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ValetKleen Chatbot</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); height: 100vh; display: flex; align-items: center; justify-content: center; }
        .chat-container { width: 90%; max-width: 800px; height: 90vh; background: white; border-radius: 15px; box-shadow: 0 10px 30px rgba(0,0,0,0.3); display: flex; flex-direction: column; overflow: hidden; }
        .chat-header { background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%); color: white; padding: 20px; text-align: center; }
        .chat-messages { flex: 1; padding: 20px; overflow-y: auto; background: #f8f9fa; }
        .message { margin-bottom: 15px; animation: fadeIn 0.3s ease-in; }
        @keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
        .message.bot { text-align: left; }
        .message.user { text-align: right; }
        .message-bubble { display: inline-block; max-width: 80%; padding: 12px 16px; border-radius: 18px; word-wrap: break-word; }
        .message.bot .message-bubble { background: #e3f2fd; color: #1565c0; border-bottom-left-radius: 5px; }
        .message.user .message-bubble { background: #4CAF50; color: white; border-bottom-right-radius: 5px; }
        .suggestions { margin-top: 10px; display: flex; flex-wrap: wrap; gap: 8px; }
        .suggestion-btn { background: #fff; border: 2px solid #4CAF50; color: #4CAF50; padding: 8px 12px; border-radius: 20px; cursor: pointer; font-size: 12px; transition: all 0.3s ease; }
        .suggestion-btn:hover { background: #4CAF50; color: white; }
        .chat-input { background: white; padding: 20px; border-top: 1px solid #e0e0e0; display: flex; gap: 10px; }
        .chat-input input { flex: 1; padding: 12px 16px; border: 2px solid #e0e0e0; border-radius: 25px; outline: none; font-size: 14px; }
        .chat-input input:focus { border-color: #4CAF50; }
        .send-btn { background: #4CAF50; color: white; border: none; padding: 12px 20px; border-radius: 25px; cursor: pointer; font-size: 14px; transition: background 0.3s ease; }
        .send-btn:hover { background: #45a049; }
    </style>
</head>
<body>
    <div class="chat-container">
        <div class="chat-header">
            <h1>🧼 ValetKleen Chatbot</h1>
            <p>Professional Laundry & Dry Cleaning Services</p>
        </div>
        <div class="chat-messages" id="chatMessages"></div>
        <div class="chat-input">
            <input type="text" id="messageInput" placeholder="Type your message..." onkeypress="handleKeyPress(event)">
            <button class="send-btn" onclick="sendMessage()">Send</button>
        </div>
    </div>

    <script>
        let sessionId = null;

        function handleKeyPress(event) {
            if (event.key === 'Enter') {
                sendMessage();
            }
        }

        function sendMessage(text = null) {
            const input = document.getElementById('messageInput');
            const message = text || input.value.trim();
            
            if (!message) return;
            
            addMessage(message, 'user');
            input.value = '';
            
            fetch('/chat', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({message: message, session_id: sessionId})
            })
            .then(response => response.json())
            .then(data => {
                sessionId = data.session_id;
                addMessage(data.message, 'bot', data.suggestions);
            })
            .catch(error => {
                console.error('Error:', error);
                addMessage('Sorry, there was an error. Please try again.', 'bot');
            });
        }

        function addMessage(text, sender, suggestions = null) {
            const container = document.getElementById('chatMessages');
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${sender}`;
            
            const bubbleDiv = document.createElement('div');
            bubbleDiv.className = 'message-bubble';
            
            const formattedText = text.replace(/\\*\\*(.*?)\\*\\*/g, '<strong>$1</strong>').replace(/\\n/g, '<br>');
            bubbleDiv.innerHTML = formattedText;
            messageDiv.appendChild(bubbleDiv);
            
            if (suggestions && suggestions.length > 0) {
                const suggestionsDiv = document.createElement('div');
                suggestionsDiv.className = 'suggestions';
                
                suggestions.forEach(suggestion => {
                    const btn = document.createElement('button');
                    btn.className = 'suggestion-btn';
                    btn.textContent = suggestion;
                    btn.onclick = () => sendMessage(suggestion);
                    suggestionsDiv.appendChild(btn);
                });
                
                messageDiv.appendChild(suggestionsDiv);
            }
            
            container.appendChild(messageDiv);
            container.scrollTop = container.scrollHeight;
        }

        // Initial greeting
        window.onload = function() {
            addMessage("👋 Welcome to ValetKleen! I'm here to help with laundry and dry cleaning.\\n\\nHow can I assist you?", 'bot', [
                "📋 Place an Order",
                "🛒 View Cart", 
                "💰 Pricing Information",
                "📞 Contact Information"
            ]);
        };
    </script>
</body>
</html>"""

@app.route('/')
def index():
    """Serve chatbot interface"""
    return render_template_string(CHATBOT_HTML)

@app.route('/chat', methods=['POST'])
def chat():
    """Handle chat messages"""
    try:
        data = request.json
        message = data.get('message', '')
        session_id = data.get('session_id')
        
        response = chatbot.generate_response(message, session_id)
        return jsonify(response)
    
    except Exception as e:
        app.logger.error(f"Chat error: {e}")
        return jsonify({
            'message': 'Sorry, there was an error. Please try again.',
            'type': 'error'
        })

@app.route('/api/cart/<session_id>')
def get_cart(session_id):
    """Get cart contents"""
    session = chatbot.customer_sessions.get(session_id, {})
    cart = session.get('cart', [])
    total = sum(item['total'] for item in cart)
    
    return jsonify({
        'cart': cart,
        'total': total,
        'item_count': len(cart)
    })

@app.route('/health')
def health():
    """Health check"""
    return jsonify({'status': 'healthy', 'version': '2.0-render'})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)