"""
ValetKleen Professional Chatbot System V2
Enhanced with modular architecture, database persistence, and improved features
"""

import os
import json
import re
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

# Flask imports
from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS

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

# Import our modules
from valetkleen_chatbot.models import db, init_db
from valetkleen_chatbot.services import CartService, SessionService, OrderService
from valetkleen_chatbot.utils import validate_email, validate_phone, validate_address, Config
from valetkleen_chatbot.utils.validators import sanitize_input

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('punkt_tab', quiet=True)
except:
    pass

class ValetKleenChatbotV2:
    """Enhanced ValetKleen chatbot with database persistence and modular design"""
    
    def __init__(self):
        """Initialize the enhanced chatbot"""
        
        self.logger = logging.getLogger(__name__)
        
        # Validate configuration
        Config.validate()
        
        # Initialize services
        self.cart_service = CartService()
        self.session_service = SessionService()
        self.order_service = OrderService()
        
        # Initialize Groq LLM client with environment variable
        self.groq_client = Groq(api_key=Config.GROQ_API_KEY)
        
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
        
        # Load service catalog
        self.service_catalog = self.load_service_catalog()
        
        self.logger.info("ValetKleen Chatbot V2 initialized successfully!")
    
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
        """Load service catalog with pricing"""
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
        """Prepare intent matching system"""
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
        
        # Fallback to vectorized matching
        if self.intent_vectors is not None:
            try:
                user_vector = self.vectorizer.transform([user_input_lower])
                similarities = cosine_similarity(user_vector, self.intent_vectors)[0]
                
                if len(similarities) > 0:
                    best_match_idx = np.argmax(similarities)
                    confidence = similarities[best_match_idx]
                    
                    if best_match_idx < len(self.intent_labels):
                        return self.intent_labels[best_match_idx], confidence
            except:
                pass
        
        return 'unknown', 0.0
    
    def generate_response(self, user_input: str, session_id: str = None) -> Dict:
        """Generate chatbot response"""
        
        # Sanitize input
        user_input = sanitize_input(user_input)
        
        # Create or get session
        if not session_id:
            session = self.session_service.create_session()
            session_id = session.id
        else:
            session = self.session_service.get_session(session_id)
            if not session:
                session = self.session_service.create_session(session_id)
        
        # Add to conversation history
        self.session_service.add_conversation_entry(session_id, 'user', user_input)
        
        # Detect intent
        intent, confidence = self.detect_intent(user_input)
        
        # Generate response based on intent
        response = self.handle_intent(intent, user_input, session_id, confidence)
        
        # Add bot response to history
        self.session_service.add_conversation_entry(
            session_id, 'bot', response.get('message', '')
        )
        
        response['session_id'] = session_id
        return response
    
    def handle_intent(self, intent: str, user_input: str, 
                     session_id: str, confidence: float) -> Dict:
        """Handle different intents"""
        
        session_data = self.session_service.get_session_data(session_id)
        current_step = session_data.get('current_step', 'welcome') if session_data else 'welcome'
        
        # Handle order flow steps
        if current_step == 'collecting_info':
            return self.handle_info_collection(user_input, session_id)
        elif current_step == 'selecting_service':
            return self.handle_service_selection(user_input, session_id)
        elif current_step == 'selecting_items':
            return self.handle_item_selection(user_input, session_id)
        
        # Handle specific intents
        if intent == 'greeting':
            return self.handle_greeting()
        elif intent == 'place_order':
            return self.start_order_process(session_id)
        elif intent == 'view_cart':
            return self.handle_view_cart(session_id)
        elif intent == 'remove_item':
            return self.handle_remove_item(user_input, session_id)
        elif intent == 'update_quantity':
            return self.handle_update_quantity(user_input, session_id)
        elif intent == 'clear_cart':
            return self.handle_clear_cart(session_id)
        elif intent == 'checkout':
            return self.handle_checkout(session_id)
        else:
            return self.handle_general_inquiry(user_input)
    
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
        self.session_service.update_step(session_id, 'collecting_info')
        
        return {
            'message': "🛍️ Let's start your order!\n\nFirst, I'll need some information.\n\n👤 **Your Name:**",
            'type': 'info_collection',
            'collecting': 'name'
        }
    
    def handle_info_collection(self, user_input: str, session_id: str) -> Dict:
        """Handle customer information collection with validation"""
        customer_info = self.session_service.get_customer_info(session_id) or {}
        
        if 'name' not in customer_info:
            customer_info['name'] = user_input.strip()
            self.session_service.save_customer_info(session_id, customer_info)
            return {
                'message': f"Thank you, {customer_info['name']}! 📧 **Your Email:**",
                'type': 'info_collection',
                'collecting': 'email'
            }
        
        elif 'email' not in customer_info:
            # Validate email
            is_valid, error_msg = validate_email(user_input.strip())
            if not is_valid:
                return {
                    'message': f"❌ {error_msg}\n\nPlease enter a valid email:",
                    'type': 'info_collection',
                    'collecting': 'email'
                }
            
            customer_info['email'] = user_input.strip()
            self.session_service.save_customer_info(session_id, customer_info)
            return {
                'message': "Perfect! 🏠 **Your Address (for pickup & delivery):**",
                'type': 'info_collection',
                'collecting': 'address'
            }
        
        elif 'address' not in customer_info:
            # Validate address
            is_valid, error_msg = validate_address(user_input.strip())
            if not is_valid:
                return {
                    'message': f"❌ {error_msg}\n\nPlease enter your complete address:",
                    'type': 'info_collection',
                    'collecting': 'address'
                }
            
            customer_info['address'] = user_input.strip()
            self.session_service.save_customer_info(session_id, customer_info)
            return {
                'message': "Great! 📱 **Your Phone Number:**",
                'type': 'info_collection',
                'collecting': 'phone'
            }
        
        elif 'phone' not in customer_info:
            # Validate phone
            is_valid, error_msg = validate_phone(user_input.strip())
            if not is_valid:
                return {
                    'message': f"❌ {error_msg}\n\nPlease enter your phone number:",
                    'type': 'info_collection',
                    'collecting': 'phone'
                }
            
            customer_info['phone'] = user_input.strip()
            self.session_service.save_customer_info(session_id, customer_info)
            
            # Generate pickup dates
            available_dates = []
            today = datetime.now()
            for i in range(1, Config.PICKUP_DAYS_AHEAD + 1):
                date = today + timedelta(days=i)
                if date.weekday() not in Config.EXCLUDE_PICKUP_DAYS:
                    available_dates.append(date.strftime("%A, %B %d, %Y"))
            
            return {
                'message': "📅 **When would you like pickup?**\n\nAvailable dates:",
                'type': 'info_collection',
                'collecting': 'pickup_date',
                'suggestions': available_dates[:5]
            }
        
        elif 'pickup_date' not in customer_info:
            customer_info['pickup_date'] = user_input.strip()
            self.session_service.save_customer_info(session_id, customer_info)
            
            return {
                'message': f"🕐 **What time works best on {customer_info['pickup_date']}?**",
                'type': 'info_collection',
                'collecting': 'pickup_time',
                'suggestions': Config.PICKUP_TIME_SLOTS
            }
        
        elif 'pickup_time' not in customer_info:
            customer_info['pickup_time'] = user_input.strip()
            self.session_service.save_customer_info(session_id, customer_info)
            self.session_service.update_step(session_id, 'selecting_service')
            
            return {
                'message': f"Perfect! Pickup scheduled for **{customer_info['pickup_date']}** at **{customer_info['pickup_time']}**.\n\nWhich service do you need?",
                'type': 'service_selection',
                'suggestions': ["👔 Dry Cleaning", "🧺 Laundry Services"]
            }
    
    def handle_service_selection(self, user_input: str, session_id: str) -> Dict:
        """Handle service selection"""
        if 'dry' in user_input.lower():
            self.session_service.update_step(session_id, 'selecting_items')
            return self.show_service_menu('dry_cleaning')
        elif 'laundry' in user_input.lower():
            self.session_service.update_step(session_id, 'selecting_items')
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
        # Simple parsing for demo
        service_type = 'dry_cleaning'  # Default
        
        # Extract quantity
        quantity = 1
        numbers = re.findall(r'\d+', user_input)
        if numbers:
            quantity = int(numbers[0])
        
        # Find matching item
        for service_key, service in self.service_catalog.items():
            for item_key, item_info in service['items'].items():
                if any(word in user_input.lower() for word in item_key.split('_')):
                    # Add to cart
                    result = self.cart_service.add_item(
                        session_id, service_key, item_key,
                        item_info['name'], item_info['price'], quantity
                    )
                    
                    if result['success']:
                        cart_summary = self.cart_service.get_cart_summary(session_id)
                        return {
                            'message': f"✅ {result['message']}\n\n**Cart Total: ${cart_summary['total']:.2f}**\n\nWhat would you like to do?",
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
    
    def handle_view_cart(self, session_id: str) -> Dict:
        """Handle view cart request"""
        cart_summary = self.cart_service.get_cart_summary(session_id)
        
        if not cart_summary['items']:
            return {
                'message': "🛒 Your cart is empty. Let's add some items!",
                'type': 'cart_empty',
                'suggestions': ["Place an Order", "View Services"]
            }
        
        cart_message = "🛒 **YOUR CART:**\n\n"
        
        for i, item in enumerate(cart_summary['items'], 1):
            cart_message += f"**#{i}** {item['quantity']}x {item['name']}\n"
            cart_message += f"   Price: ${item['price']:.2f} each\n"
            cart_message += f"   Subtotal: ${item['total']:.2f}\n"
            cart_message += f"   ID: {item['id']}\n\n"
        
        cart_message += f"**TOTAL: ${cart_summary['total']:.2f}**\n\n"
        cart_message += "To remove item, say 'Remove item #ID'\n"
        cart_message += "To update quantity, say 'Update item #ID to 3'"
        
        return {
            'message': cart_message,
            'type': 'cart_view',
            'suggestions': [
                "Checkout",
                "Add More Items",
                "Clear Cart",
                "Remove Item"
            ]
        }
    
    def handle_remove_item(self, user_input: str, session_id: str) -> Dict:
        """Handle remove item from cart"""
        # Extract item ID
        numbers = re.findall(r'\d+', user_input)
        if not numbers:
            return {
                'message': "Please specify item ID to remove (e.g., 'Remove item #1')",
                'type': 'error'
            }
        
        item_id = int(numbers[0])
        result = self.cart_service.remove_item(session_id, item_id)
        
        if result['success']:
            return {
                'message': f"✅ {result['message']}\n\n**New Total: ${result['cart_total']:.2f}**",
                'type': 'cart_update',
                'suggestions': ["View Cart", "Checkout", "Add More Items"]
            }
        else:
            return {
                'message': f"❌ {result['error']}",
                'type': 'error',
                'suggestions': ["View Cart"]
            }
    
    def handle_update_quantity(self, user_input: str, session_id: str) -> Dict:
        """Handle update item quantity"""
        # Extract item ID and new quantity
        numbers = re.findall(r'\d+', user_input)
        if len(numbers) < 2:
            return {
                'message': "Please specify item ID and new quantity (e.g., 'Update item #1 to 3')",
                'type': 'error'
            }
        
        item_id = int(numbers[0])
        new_quantity = int(numbers[1])
        
        result = self.cart_service.update_quantity(session_id, item_id, new_quantity)
        
        if result['success']:
            return {
                'message': f"✅ {result['message']}\n\n**New Total: ${result['cart_total']:.2f}**",
                'type': 'cart_update',
                'suggestions': ["View Cart", "Checkout", "Add More Items"]
            }
        else:
            return {
                'message': f"❌ {result['error']}",
                'type': 'error',
                'suggestions': ["View Cart"]
            }
    
    def handle_clear_cart(self, session_id: str) -> Dict:
        """Handle clear cart"""
        result = self.cart_service.clear_cart(session_id)
        
        if result['success']:
            return {
                'message': "✅ Cart cleared successfully!",
                'type': 'cart_cleared',
                'suggestions': ["Place an Order", "View Services"]
            }
        else:
            return {
                'message': f"❌ {result['error']}",
                'type': 'error'
            }
    
    def handle_checkout(self, session_id: str) -> Dict:
        """Handle checkout"""
        cart_summary = self.cart_service.get_cart_summary(session_id)
        
        if not cart_summary['items']:
            return {
                'message': "Your cart is empty! Please add items first.",
                'type': 'error',
                'suggestions': ["Place an Order"]
            }
        
        # Create order
        result = self.order_service.create_order(session_id)
        
        if result['success']:
            customer_info = self.session_service.get_customer_info(session_id) or {}
            
            message = f"🎉 **ORDER CONFIRMED!**\n\n"
            message += f"**Order #:** {result['order_number']}\n"
            message += f"**Total:** ${result['total']:.2f}\n\n"
            message += f"**Pickup Details:**\n"
            message += f"Date: {customer_info.get('pickup_date', 'TBD')}\n"
            message += f"Time: {customer_info.get('pickup_time', 'TBD')}\n"
            message += f"Address: {customer_info.get('address', 'TBD')}\n\n"
            message += "✅ We'll pick up your items as scheduled!\n"
            message += "📧 Confirmation email sent!\n\n"
            message += "Thank you for choosing ValetKleen!"
            
            return {
                'message': message,
                'type': 'checkout_success',
                'order_number': result['order_number'],
                'suggestions': ["Place Another Order", "View Services"]
            }
        else:
            return {
                'message': f"❌ Checkout failed: {result['error']}",
                'type': 'error',
                'suggestions': ["View Cart", "Try Again"]
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
app.config.from_object(Config)
CORS(app)

# Initialize database
init_db(app)

# Initialize chatbot
chatbot = ValetKleenChatbotV2()

# HTML template (same as before but with cart management buttons)
CHATBOT_HTML = """<!DOCTYPE html>
<html>
<head>
    <title>ValetKleen Chatbot V2</title>
    <style>
        /* Same styles as before */
        body { font-family: Arial; background: linear-gradient(135deg, #667eea, #764ba2); }
        .chat-container { max-width: 800px; margin: 50px auto; background: white; border-radius: 15px; }
        /* ... rest of styles ... */
    </style>
</head>
<body>
    <div class="chat-container">
        <h1>🧼 ValetKleen Chatbot V2</h1>
        <div id="chat-messages"></div>
        <div class="chat-input">
            <input type="text" id="message-input" placeholder="Type your message...">
            <button onclick="sendMessage()">Send</button>
        </div>
    </div>
    <script>
        let sessionId = null;
        
        function sendMessage(text) {
            const input = document.getElementById('message-input');
            const message = text || input.value;
            if (!message) return;
            
            // Add user message to chat
            addMessage(message, 'user');
            input.value = '';
            
            // Send to backend
            fetch('/chat', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({message: message, session_id: sessionId})
            })
            .then(r => r.json())
            .then(data => {
                sessionId = data.session_id;
                addMessage(data.message, 'bot', data.suggestions);
            });
        }
        
        function addMessage(text, sender, suggestions) {
            const container = document.getElementById('chat-messages');
            const messageDiv = document.createElement('div');
            messageDiv.className = 'message ' + sender;
            messageDiv.innerHTML = text.replace(/\\*\\*(.*?)\\*\\*/g, '<strong>$1</strong>');
            
            if (suggestions) {
                const sugDiv = document.createElement('div');
                suggestions.forEach(s => {
                    const btn = document.createElement('button');
                    btn.textContent = s;
                    btn.onclick = () => sendMessage(s);
                    sugDiv.appendChild(btn);
                });
                messageDiv.appendChild(sugDiv);
            }
            
            container.appendChild(messageDiv);
            container.scrollTop = container.scrollHeight;
        }
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
            'message': 'Sorry, an error occurred. Please try again.',
            'type': 'error'
        })

@app.route('/api/cart/<session_id>')
def get_cart(session_id):
    """Get cart contents"""
    cart_summary = chatbot.cart_service.get_cart_summary(session_id)
    return jsonify(cart_summary)

@app.route('/api/order/<order_number>')
def get_order(order_number):
    """Get order details"""
    order = chatbot.order_service.get_order(order_number)
    if order:
        return jsonify(order)
    return jsonify({'error': 'Order not found'}), 404

@app.route('/health')
def health():
    """Health check"""
    return jsonify({'status': 'healthy', 'version': '2.0.0'})

if __name__ == '__main__':
    print("🚀 Starting ValetKleen Chatbot V2...")
    print("✅ Database persistence enabled")
    print("✅ Input validation active")
    print("✅ Cart management features ready")
    print("🌐 Access at: http://localhost:5000")
    
    app.run(host=Config.APP_HOST, port=Config.APP_PORT, debug=Config.DEBUG)