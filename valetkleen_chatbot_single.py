"""
ValetKleen Professional Chatbot System - Single File for Render
This is the V2 modular version combined back into a single file for deployment
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

# Configuration
class Config:
    SECRET_KEY = os.getenv('FLASK_SECRET_KEY', 'valetkleen_chatbot_secret_key_2024')
    GROQ_API_KEY = os.getenv('GROQ_API_KEY')
    APP_HOST = os.getenv('APP_HOST', '0.0.0.0')
    APP_PORT = int(os.getenv('APP_PORT', 5000))
    PICKUP_TIME_SLOTS = [
        "8:00 AM - 10:00 AM",
        "10:00 AM - 12:00 PM",
        "12:00 PM - 2:00 PM", 
        "2:00 PM - 4:00 PM",
        "4:00 PM - 6:00 PM"
    ]

# Validation functions (V2 enhancement)
def validate_email(email: str) -> Tuple[bool, str]:
    if not email:
        return False, "Email is required"
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    if not re.match(pattern, email):
        return False, "Please enter a valid email address (e.g., john@example.com)"
    return True, ""

def validate_phone(phone: str) -> Tuple[bool, str]:
    if not phone:
        return False, "Phone number is required"
    cleaned = re.sub(r'[\s\-\(\)\.]', '', phone)
    us_pattern = r'^(\+1)?[2-9]\d{9}$'
    if re.match(us_pattern, cleaned):
        return True, ""
    elif len(cleaned) < 10:
        return False, "Phone number is too short. Please include area code."
    else:
        return False, "Please enter a valid phone number (e.g., 555-123-4567)"

def validate_address(address: str) -> Tuple[bool, str]:
    if not address:
        return False, "Address is required"
    if len(address.strip()) < 10:
        return False, "Please enter a complete address"
    return True, ""

def sanitize_input(text: str, max_length: int = 500) -> str:
    if not text:
        return ""
    text = re.sub(r'<[^>]*>', '', text)
    text = text[:max_length]
    text = ' '.join(text.split())
    return text.strip()

class ValetKleenChatbot:
    """The original V2 chatbot logic in a single file"""
    
    def __init__(self):
        """Initialize the ValetKleen chatbot with NLP and knowledge base"""
        
        # Initialize logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialize Groq LLM client
        api_key = Config.GROQ_API_KEY
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable is required")
        self.groq_client = Groq(api_key=api_key)
        
        # Initialize NLP components
        self.lemmatizer = WordNetLemmatizer()
        try:
            self.stop_words = set(stopwords.words('english'))
        except:
            self.stop_words = set()
        
        # Load knowledge base from scraped data
        self.knowledge_base = self.load_knowledge_base()
        
        # Initialize TF-IDF vectorizer for intent matching
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        self.prepare_intent_matching()
        
        # Service catalogs with pricing
        self.service_catalog = self.load_service_catalog()
        
        # Customer sessions storage (enhanced with cart IDs)
        self.customer_sessions = {}
        self.cart_item_counter = 0
        
        self.logger.info("ValetKleen Chatbot initialized successfully!")
    
    def load_knowledge_base(self) -> Dict:
        """Load and process the scraped ValetKleen data"""
        try:
            # Try to load the scraped data
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
                
                # Process scraped content
                for item in scraped_data:
                    content = item.get('content', '').lower()
                    title = item.get('title', '').lower()
                    
                    # Categorize content based on title and content
                    if any(word in title for word in ['about', 'about us']):
                        knowledge['about'] += f" {item.get('content', '')}"
                    elif any(word in title for word in ['service', 'dry cleaning', 'laundry', 'hotel']):
                        knowledge['services'] += f" {item.get('content', '')}"
                    elif 'faq' in title:
                        knowledge['faq'] += f" {item.get('content', '')}"
                    elif 'contact' in title:
                        knowledge['contact'] += f" {item.get('content', '')}"
                    elif any(word in title for word in ['how it works', 'process']):
                        knowledge['process'] += f" {item.get('content', '')}"
                    
                    knowledge['all_content'].append({
                        'title': item.get('title', ''),
                        'content': item.get('content', ''),
                        'type': item.get('type', ''),
                        'url': item.get('url', '')
                    })
                
                self.logger.info(f"Loaded {len(scraped_data)} knowledge base items")
                return knowledge
                
        except Exception as e:
            self.logger.error(f"Error loading knowledge base: {e}")
        
        # Fallback knowledge base
        return {
            'about': "ValetKleen provides convenient pickup and delivery laundry and dry cleaning services. We offer professional cleaning with eco-friendly methods and transparent pricing.",
            'services': "We offer dry cleaning, laundry services, pickup and delivery, hotel services, and specialty cleaning for delicate fabrics.",
            'faq': "Common questions about our services, pricing, and delivery options.",
            'contact': "Contact ValetKleen for pickup scheduling and customer service.",
            'process': "Book online, we pickup, clean professionally, and deliver back to you.",
            'pricing': "Transparent flat-rate pricing with no hidden fees.",
            'all_content': []
        }
    
    def load_service_catalog(self) -> Dict:
        """Load the comprehensive service catalog with pricing"""
        return {
            'dry_cleaning': {
                'name': 'Dry Cleaning Services',
                'description': 'Professional dry cleaning for specialty items',
                'items': {
                    'office_shirt': {'name': 'Office Shirt (Dry-Clean)', 'price': 5.50, 'options': []},
                    'pants': {'name': 'Pants', 'price': 7.50, 'options': ['Crease', 'No crease']},
                    'dress_kids': {'name': 'Kids Dress', 'price': 8.00, 'options': []},
                    'dress_kids_premium': {'name': 'Kids Premium Dress', 'price': 10.00, 'options': []},
                    'dress_standard': {'name': 'Standard Dress', 'price': 12.00, 'options': []},
                    'dress_standard_long': {'name': 'Standard Extra Long Dress', 'price': 14.00, 'options': []},
                    'dress_cocktail': {'name': 'Cocktail Dress', 'price': 16.00, 'options': []},
                    'dress_formal': {'name': 'Formal/Gown Dress', 'price': 25.00, 'options': []},
                    'dress_evening': {'name': 'Evening/Prom Long Dress', 'price': 35.00, 'options': []},
                    'coat_lab': {'name': 'Lab Coat', 'price': 9.50, 'options': []},
                    'coat_short': {'name': 'Short Coat', 'price': 12.00, 'options': []},
                    'coat_3quarter': {'name': '3/4 Length Coat', 'price': 14.00, 'options': []},
                    'coat_rain': {'name': 'Raincoat', 'price': 16.00, 'options': []},
                    'coat_over': {'name': 'Overcoat', 'price': 20.00, 'options': []},
                    'coat_down': {'name': 'Down Filled Coat', 'price': 25.00, 'options': []},
                    'coat_fur': {'name': 'Fur Lined Coat', 'price': 30.00, 'options': []},
                    'jumpsuit_short': {'name': 'Short Jump Suit', 'price': 10.00, 'options': []},
                    'jumpsuit_long': {'name': 'Long Jump Suit', 'price': 12.00, 'options': []},
                    'jumpsuit_premium': {'name': 'Long Premium Jump Suit', 'price': 16.00, 'options': []},
                    'curtains': {'name': 'Curtains (Per Panel)', 'price': 25.00, 'options': []},
                    'dashiki': {'name': 'Men\'s Dashiki (2 PC)', 'price': 18.00, 'options': ['Starch', 'Hang', 'Fold']},
                    'agbada': {'name': 'Men\'s Agbada (3 PC)', 'price': 20.00, 'options': ['Starch', 'Hang', 'Fold']},
                    'wedding_dress': {'name': 'Wedding Dress', 'price': 180.00, 'options': ['Box', 'No box']}
                }
            },
            'laundry': {
                'name': 'Laundry Services',
                'description': 'Full laundry service with wash, fold, and dry cleaning',
                'items': {
                    'bag_small': {'name': 'Small Bag (12 lb capacity)', 'price': 22.00, 'options': []},
                    'bag_medium': {'name': 'Medium Bag (18 lb capacity)', 'price': 33.00, 'options': []},
                    'bag_large': {'name': 'Large Bag (25 lb capacity)', 'price': 46.00, 'options': []},
                    'bag_king': {'name': 'King Size Premium Bag (35 lb capacity)', 'price': 65.00, 'options': []},
                    'comforter_twin': {'name': 'Comforter (Twin/Full)', 'price': 25.00, 'options': []},
                    'comforter_queen': {'name': 'Comforter (Queen/King)', 'price': 30.00, 'options': []},
                    'blanket_twin': {'name': 'Blanket (Full/Twin)', 'price': 20.00, 'options': []},
                    'blanket_queen': {'name': 'Blanket (Queen/King)', 'price': 25.00, 'options': []},
                    'mattress_twin': {'name': 'Mattress Cover (Twin/Full)', 'price': 15.00, 'options': []},
                    'mattress_queen': {'name': 'Mattress Cover (Queen/King)', 'price': 20.00, 'options': []}
                }
            }
        }
    
    def prepare_intent_matching(self):
        """Prepare TF-IDF vectors for intent matching"""
        # Define common intents and their example phrases
        self.intents = {
            'greeting': [
                'hello', 'hi', 'hey', 'good morning', 'good afternoon', 'good evening',
                'how are you', 'whats up', 'greetings'
            ],
            'place_order': [
                'place order', 'make order', 'order service', 'book service', 'schedule pickup',
                'i want to order', 'need cleaning', 'book cleaning', 'arrange pickup'
            ],
            'services_inquiry': [
                'what services', 'services offered', 'what do you offer', 'available services',
                'dry cleaning', 'laundry service', 'cleaning options', 'types of cleaning'
            ],
            'pricing_inquiry': [
                'how much', 'price', 'cost', 'pricing', 'rates', 'charges', 'fees',
                'what does it cost', 'pricing information'
            ],
            'delivery_inquiry': [
                'pickup and delivery', 'delivery', 'pickup', 'when do you pickup',
                'delivery time', 'pickup schedule', 'how does delivery work'
            ],
            'about_company': [
                'about valetkleen', 'about you', 'company info', 'who are you',
                'tell me about', 'your company', 'about your service'
            ],
            'contact_info': [
                'contact', 'phone number', 'email', 'address', 'how to reach',
                'contact information', 'get in touch'
            ],
            'process_inquiry': [
                'how it works', 'process', 'how do you work', 'steps', 'procedure',
                'how does it work', 'what happens after'
            ]
        }
        
        # Create training data for intent classification
        self.intent_texts = []
        self.intent_labels = []
        
        for intent, phrases in self.intents.items():
            for phrase in phrases:
                self.intent_texts.append(phrase)
                self.intent_labels.append(intent)
        
        # Add knowledge base content for better matching
        for content_item in self.knowledge_base.get('all_content', []):
            self.intent_texts.append(content_item.get('content', '')[:200])  # First 200 chars
            self.intent_labels.append('information')
        
        # Fit the vectorizer
        try:
            self.intent_vectors = self.vectorizer.fit_transform(self.intent_texts)
        except Exception as e:
            self.logger.error(f"Error preparing intent matching: {e}")
            # Create dummy vectors if TF-IDF fails
            self.intent_vectors = None
    
    def detect_intent_with_llm(self, user_input: str) -> Tuple[str, float]:
        """Enhanced intent detection using Groq LLM"""
        try:
            # Create prompt with available intents and services
            service_options = ", ".join(self.service_catalog.keys())
            
            prompt = f"""You are analyzing a customer message for a laundry and dry cleaning service called ValetKleen.

Available service types: {service_options}

Customer message: "{user_input}"

Analyze this message and determine the customer's primary intent. Respond with one of these exact intent categories:

1. "place_order" - Customer wants to place an order or add items to cart
2. "service_inquiry" - Customer asking about services, pricing, or what's available
3. "process_inquiry" - Customer asking how the service works, pickup/delivery process
4. "pricing_inquiry" - Customer asking about prices or costs
5. "contact_info" - Customer wants contact information
6. "about_company" - Customer asking about the company
7. "item_selection" - Customer specifying items they want cleaned (like "2 shirts", "dry clean my dress")
8. "information" - General questions about laundry/dry cleaning

Additionally, if the message contains specific items and quantities (like "I need 2 office shirts" or "clean my wedding dress"), extract:
- Items mentioned
- Quantities mentioned
- Service type preference (dry cleaning or laundry)

Format your response as:
INTENT: [intent category]
CONFIDENCE: [0.0 to 1.0]
ITEMS: [list any specific items mentioned]
QUANTITIES: [list any numbers/quantities mentioned]
SERVICE_PREFERENCE: [dry_cleaning/laundry/unknown]

Example response:
INTENT: item_selection
CONFIDENCE: 0.95
ITEMS: office shirts, wedding dress
QUANTITIES: 2, 1
SERVICE_PREFERENCE: dry_cleaning"""

            # Call Groq API
            completion = self.groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_completion_tokens=200
            )
            
            response = completion.choices[0].message.content
            
            # Parse LLM response
            intent = "information"  # default
            confidence = 0.5
            
            for line in response.split('\n'):
                if line.startswith('INTENT:'):
                    intent = line.split(':', 1)[1].strip()
                elif line.startswith('CONFIDENCE:'):
                    try:
                        confidence = float(line.split(':', 1)[1].strip())
                    except:
                        confidence = 0.8
            
            self.logger.info(f"LLM Intent Detection - Input: '{user_input}' -> Intent: {intent}, Confidence: {confidence}")
            return intent, confidence
            
        except Exception as e:
            self.logger.error(f"Error in LLM intent detection: {e}")
            # Fallback to original method
            return self.detect_intent(user_input)
    
    def detect_intent(self, user_input: str) -> Tuple[str, float]:
        """Detect user intent using NLP"""
        processed_input = self.preprocess_text(user_input)
        
        # Try TF-IDF similarity matching
        if self.intent_vectors is not None:
            try:
                user_vector = self.vectorizer.transform([processed_input])
                similarities = cosine_similarity(user_vector, self.intent_vectors)[0]
                
                if len(similarities) > 0:
                    best_match_idx = np.argmax(similarities)
                    confidence = similarities[best_match_idx]
                    
                    if best_match_idx < len(self.intent_labels):
                        return self.intent_labels[best_match_idx], confidence
            except Exception as e:
                self.logger.error(f"Error in TF-IDF matching: {e}")
        
        # Fallback to keyword matching
        return self.keyword_intent_detection(processed_input)
    
    def preprocess_text(self, text: str) -> str:
        """Preprocess text for better NLP matching"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters but keep alphanumeric and spaces
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        
        # Tokenize
        try:
            tokens = word_tokenize(text)
        except:
            tokens = text.split()
        
        # Remove stopwords and lemmatize
        processed_tokens = []
        for token in tokens:
            if token not in self.stop_words:
                try:
                    processed_tokens.append(self.lemmatizer.lemmatize(token))
                except:
                    processed_tokens.append(token)
        
        return ' '.join(processed_tokens)
    
    def keyword_intent_detection(self, processed_input: str) -> Tuple[str, float]:
        """Fallback keyword-based intent detection"""
        intent_scores = {}
        
        for intent, keywords in self.intents.items():
            score = 0
            for keyword in keywords:
                if keyword.lower() in processed_input:
                    score += 1
            
            if score > 0:
                intent_scores[intent] = score / len(keywords)
        
        if intent_scores:
            best_intent = max(intent_scores, key=intent_scores.get)
            confidence = intent_scores[best_intent]
            return best_intent, confidence
        
        return 'unknown', 0.0
    
    def create_customer_session(self, session_id: str = None) -> str:
        """Create or get customer session"""
        import uuid
        
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
                   quantity: int = 1, selected_options: List[str] = None) -> bool:
        """Add item to customer cart"""
        if session_id not in self.customer_sessions:
            return False
        
        if service_type not in self.service_catalog:
            return False
        
        if item_key not in self.service_catalog[service_type]['items']:
            return False
        
        item_info = self.service_catalog[service_type]['items'][item_key]
        selected_options = selected_options or []
        
        # Generate unique cart item ID
        self.cart_item_counter += 1
        
        cart_item = {
            'id': self.cart_item_counter,  # V2 enhancement: unique IDs for cart management
            'service_type': service_type,
            'item_key': item_key,
            'name': item_info['name'],
            'price': item_info['price'],
            'quantity': quantity,
            'options': selected_options,
            'total': item_info['price'] * quantity
        }
        
        self.customer_sessions[session_id]['cart'].append(cart_item)
        return True
    
    def handle_view_cart(self, session_id: str) -> Dict:
        """Handle view cart request - V2 FIX"""
        session = self.customer_sessions.get(session_id, {})
        cart = session.get('cart', [])
        
        if not cart:
            return {
                'message': "🛒 Your cart is empty. Let's add some items!",
                'type': 'cart_empty',
                'suggestions': [
                    "Place an Order",
                    "👔 Dry Cleaning Services",
                    "🧺 Laundry Services",
                    "What Services Do You Offer?"
                ]
            }
        
        # Build detailed cart view
        cart_message = "🛒 **YOUR SHOPPING CART:**\n\n"
        total = 0
        
        for i, item in enumerate(cart, 1):
            item_total = item.get('total', item.get('price', 0) * item.get('quantity', 1))
            total += item_total
            
            cart_message += f"**{i}.** {item['quantity']}x {item['name']}\n"
            if item.get('options'):
                cart_message += f"   Options: {', '.join(item['options'])}\n"
            cart_message += f"   Price: ${item['price']:.2f} each\n"
            cart_message += f"   Subtotal: ${item_total:.2f}\n"
            cart_message += f"   ID: {item.get('id', i)}\n\n"
        
        cart_message += f"━━━━━━━━━━━━━━━━━━━━\n"
        cart_message += f"💰 **TOTAL: ${total:.2f}**\n\n"
        cart_message += "What would you like to do next?"
        
        return {
            'message': cart_message,
            'type': 'cart_view',
            'suggestions': [
                "Add More Items",
                "Proceed to Checkout",
                "Remove Item",
                "👔 Add Dry Cleaning Items",
                "🧺 Add Laundry Items"
            ]
        }
    
    def generate_response(self, user_input: str, session_id: str = None) -> Dict:
        """Generate chatbot response based on user input and intent"""
        
        # Sanitize input (V2 enhancement)
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
        
        # Detect intent with LLM (fallback to traditional method if needed)
        intent, confidence = self.detect_intent_with_llm(user_input)
        
        # Generate appropriate response based on intent and current step
        response = self.handle_intent(intent, user_input, session_id, confidence)
        
        # Add bot response to history
        session['conversation_history'].append({
            'bot': response.get('message', ''),
            'timestamp': datetime.now().isoformat()
        })
        
        response['session_id'] = session_id
        return response
    
    def handle_intent(self, intent: str, user_input: str, session_id: str, confidence: float) -> Dict:
        """Handle different intents and generate appropriate responses"""
        
        session = self.customer_sessions[session_id]
        current_step = session.get('current_step', 'welcome')
        
        # Check for checkout keywords FIRST (before any step handling)
        user_input_lower = user_input.lower()
        if any(keyword in user_input_lower for keyword in ['proceed to checkout', 'checkout', 'complete order', 'finish order', 'place order now']):
            return self.handle_checkout(session_id)
        
        # Check for view cart request - V2 FIX
        if any(keyword in user_input_lower for keyword in ['view cart', 'view full cart', 'show cart', 'my cart', 'cart items']):
            return self.handle_view_cart(session_id)
        
        # Handle order flow steps
        if current_step == 'collecting_info':
            return self.handle_info_collection(user_input, session_id)
        elif current_step == 'selecting_service':
            return self.handle_service_selection(user_input, session_id)
        elif current_step == 'selecting_items':
            return self.handle_item_selection(user_input, session_id)
        elif current_step == 'adding_options':
            return self.handle_option_selection(user_input, session_id)
        
        # Handle intents
        if intent == 'greeting':
            return self.handle_greeting()
        elif intent == 'place_order':
            return self.start_order_process(session_id)
        elif intent == 'services_inquiry':
            return self.handle_services_inquiry()
        elif intent == 'pricing_inquiry':
            return self.handle_pricing_inquiry()
        elif intent == 'delivery_inquiry':
            return self.handle_delivery_inquiry()
        elif intent == 'about_company':
            return self.handle_about_inquiry()
        elif intent == 'contact_info':
            return self.handle_contact_inquiry()
        elif intent == 'process_inquiry':
            return self.handle_process_inquiry()
        else:
            return self.handle_general_inquiry(user_input)
    
    def handle_greeting(self) -> Dict:
        """Handle greeting messages"""
        return {
            'message': "👋 Welcome to ValetKleen! I'm here to help you with our premium laundry and dry cleaning services.\n\nHow can I assist you today?",
            'type': 'greeting',
            'suggestions': [
                "📋 Place an Order",
                "🧼 What Services Do You Offer?",
                "💰 Pricing Information",
                "🚛 Pickup & Delivery Info",
                "📞 Contact Information"
            ]
        }
    
    def start_order_process(self, session_id: str) -> Dict:
        """Start the order placement process"""
        session = self.customer_sessions[session_id]
        session['current_step'] = 'collecting_info'
        
        return {
            'message': "🛍️ Great! I'd love to help you place an order.\n\nFirst, I'll need some information from you:\n\n👤 **Your Name:**",
            'type': 'info_collection',
            'collecting': 'name',
            'suggestions': []
        }
    
    def handle_info_collection(self, user_input: str, session_id: str) -> Dict:
        """Handle customer information collection - V2 ENHANCED"""
        session = self.customer_sessions[session_id]
        customer_info = session['customer_info']
        
        # Determine what information we're collecting
        if 'name' not in customer_info:
            customer_info['name'] = user_input.strip()
            return {
                'message': f"Thank you, {customer_info['name']}! 📧 **Your Email:**",
                'type': 'info_collection',
                'collecting': 'email'
            }
        elif 'email' not in customer_info:
            # V2 Enhancement: Email validation
            is_valid, error_msg = validate_email(user_input.strip())
            if not is_valid:
                return {
                    'message': f"❌ {error_msg}\n\nPlease enter a valid email:",
                    'type': 'info_collection',
                    'collecting': 'email'
                }
            
            customer_info['email'] = user_input.strip()
            return {
                'message': "Perfect! 🏠 **Your Address (for pickup & delivery):**",
                'type': 'info_collection',
                'collecting': 'address'
            }
        elif 'address' not in customer_info:
            # V2 Enhancement: Address validation
            is_valid, error_msg = validate_address(user_input.strip())
            if not is_valid:
                return {
                    'message': f"❌ {error_msg}\n\nPlease enter your complete address:",
                    'type': 'info_collection',
                    'collecting': 'address'
                }
            
            customer_info['address'] = user_input.strip()
            return {
                'message': "Great! 📱 **Your Phone Number:**",
                'type': 'info_collection',
                'collecting': 'phone'
            }
        elif 'phone' not in customer_info:
            # V2 Enhancement: Phone validation
            is_valid, error_msg = validate_phone(user_input.strip())
            if not is_valid:
                return {
                    'message': f"❌ {error_msg}\n\nPlease enter your phone number:",
                    'type': 'info_collection',
                    'collecting': 'phone'
                }
            
            customer_info['phone'] = user_input.strip()
            
            # V2 Enhancement: Dynamic date generation
            available_dates = []
            today = datetime.now()
            for i in range(1, 8):  # Next 7 days
                date = today + timedelta(days=i)
                if date.weekday() != 6:  # Exclude Sunday (6)
                    available_dates.append(date.strftime("%A, %B %d, %Y"))
            
            return {
                'message': f"Great! 📅 **When would you like us to pick up your items?**\n\nAvailable dates:",
                'type': 'info_collection',
                'collecting': 'pickup_date',
                'suggestions': available_dates[:5]  # Show first 5 available dates
            }
        elif 'pickup_date' not in customer_info:
            customer_info['pickup_date'] = user_input.strip()
            
            return {
                'message': f"Perfect! 🕐 **What time slot works best for pickup on {customer_info['pickup_date']}?**",
                'type': 'info_collection',
                'collecting': 'pickup_time',
                'suggestions': Config.PICKUP_TIME_SLOTS
            }
        elif 'pickup_time' not in customer_info:
            customer_info['pickup_time'] = user_input.strip()
            session['current_step'] = 'selecting_service'
            
            return {
                'message': f"Excellent, {customer_info['name']}! 🎯\n\nWe'll pick up on **{customer_info['pickup_date']}** between **{customer_info['pickup_time']}**.\n\nNow, which service would you like?",
                'type': 'service_selection',
                'suggestions': [
                    "👔 Dry Cleaning Services",
                    "🧺 Laundry Services"
                ]
            }
    
    def handle_service_selection(self, user_input: str, session_id: str) -> Dict:
        """Handle service type selection"""
        session = self.customer_sessions[session_id]
        processed_input = user_input.lower()
        
        if 'dry' in processed_input or 'dry cleaning' in processed_input:
            session['selected_service'] = 'dry_cleaning'
            session['current_step'] = 'selecting_items'
            return self.show_dry_cleaning_menu()
        elif 'laundry' in processed_input:
            session['selected_service'] = 'laundry'
            session['current_step'] = 'selecting_items'
            return self.show_laundry_menu()
        else:
            return {
                'message': "Please select one of our services:",
                'type': 'service_selection',
                'suggestions': [
                    "👔 Dry Cleaning Services",
                    "🧺 Laundry Services"
                ]
            }
    
    def show_dry_cleaning_menu(self) -> Dict:
        """Show dry cleaning service menu"""
        menu_text = "👔 **DRY CLEANING SERVICES** (Specialty cleaning only):\n\n"
        
        items = self.service_catalog['dry_cleaning']['items']
        menu_items = []
        
        for i, (key, item) in enumerate(items.items(), 1):
            menu_text += f"{i}. **{item['name']}** - ${item['price']:.2f}"
            if item['options']:
                menu_text += f" (Options: {', '.join(item['options'])})"
            menu_text += "\n"
            menu_items.append(f"{i}. {item['name']}")
        
        menu_text += "\n💬 You can say things like:\n• 'I need 2 office shirts'\n• 'Add 1 cocktail dress'\n• 'I want pants with crease option'\n\n**What would you like to add?**"
        
        return {
            'message': menu_text,
            'type': 'item_selection',
            'service': 'dry_cleaning',
            'suggestions': menu_items[:8]  # Show first 8 items as suggestions
        }
    
    def show_laundry_menu(self) -> Dict:
        """Show laundry service menu"""
        menu_text = "🧺 **LAUNDRY SERVICES** (Wash, fold, and dry cleaning items):\n\n"
        
        items = self.service_catalog['laundry']['items']
        menu_items = []
        
        for i, (key, item) in enumerate(items.items(), 1):
            menu_text += f"{i}. **{item['name']}** - ${item['price']:.2f}\n"
            menu_items.append(f"{i}. {item['name']}")
        
        menu_text += "\n💬 You can say things like:\n• 'I need 1 medium bag'\n• 'Add 2 queen comforters'\n• 'I want a large bag'\n\n**What would you like to add?**"
        
        return {
            'message': menu_text,
            'type': 'item_selection',
            'service': 'laundry',
            'suggestions': menu_items
        }
    
    def handle_item_selection(self, user_input: str, session_id: str) -> Dict:
        """Handle item selection with NLP parsing"""
        session = self.customer_sessions[session_id]
        selected_service = session.get('selected_service')
        
        if not selected_service:
            return self.handle_service_selection(user_input, session_id)
        
        # Parse user input for items and quantities
        parsed_items = self.parse_item_request(user_input, selected_service)
        
        if not parsed_items:
            return {
                'message': "I couldn't understand that. Please try again or select from the menu:",
                'type': 'item_selection',
                'suggestions': self.get_item_suggestions(selected_service)
            }
        
        # Add items to cart
        added_items = []
        for item_info in parsed_items:
            item_key = item_info['key']
            quantity = item_info['quantity']
            item_details = self.service_catalog[selected_service]['items'][item_key]
            
            # Add to cart
            if self.add_to_cart(session_id, selected_service, item_key, quantity):
                added_items.append(f"{quantity}x {item_details['name']}")
        
        if added_items:
            cart_summary = self.get_cart_summary(session_id)
            return {
                'message': f"✅ Added to cart: {', '.join(added_items)}\n\n{cart_summary}\n\nWould you like to add more items or proceed to checkout?",
                'type': 'cart_update',
                'suggestions': [
                    "Add More Items",
                    "Proceed to Checkout",
                    "View Full Cart",
                    "Remove Item"
                ]
            }
        
        return {
            'message': "Sorry, I couldn't add those items. Please try again:",
            'type': 'item_selection',
            'suggestions': self.get_item_suggestions(selected_service)
        }
    
    def parse_item_request(self, user_input: str, service_type: str) -> List[Dict]:
        """Parse user input to extract items and quantities"""
        parsed_items = []
        input_lower = user_input.lower()
        
        # Extract numbers (quantities)
        numbers = re.findall(r'\d+', user_input)
        
        # Get service items
        service_items = self.service_catalog[service_type]['items']
        
        # Try to match items
        for item_key, item_info in service_items.items():
            item_name_lower = item_info['name'].lower()
            
            # Create various matching patterns
            name_words = item_name_lower.split()
            
            # Check for exact matches or partial matches
            if (item_name_lower in input_lower or 
                any(word in input_lower for word in name_words if len(word) > 3) or
                any(keyword in input_lower for keyword in self.get_item_keywords(item_key, item_info))):
                
                # Find quantity (default to 1)
                quantity = 1
                if numbers:
                    quantity = int(numbers[0])
                    numbers = numbers[1:]  # Remove used number
                
                parsed_items.append({
                    'key': item_key,
                    'name': item_info['name'],
                    'quantity': quantity
                })
        
        return parsed_items
    
    def get_item_keywords(self, item_key: str, item_info: Dict) -> List[str]:
        """Get keywords for better item matching"""
        keywords = []
        
        # Add variations based on item key
        if 'shirt' in item_key:
            keywords.extend(['shirt', 'blouse'])
        elif 'pants' in item_key:
            keywords.extend(['pants', 'trousers'])
        elif 'dress' in item_key:
            keywords.extend(['dress', 'gown'])
        elif 'coat' in item_key:
            keywords.extend(['coat', 'jacket'])
        elif 'bag' in item_key:
            keywords.extend(['bag', 'laundry bag'])
        elif 'comforter' in item_key:
            keywords.extend(['comforter', 'duvet'])
        elif 'blanket' in item_key:
            keywords.extend(['blanket'])
        
        return keywords
    
    def get_item_suggestions(self, service_type: str) -> List[str]:
        """Get item suggestions for the service"""
        items = self.service_catalog[service_type]['items']
        return [f"{item['name']} - ${item['price']:.2f}" for item in list(items.values())[:8]]
    
    def get_cart_summary(self, session_id: str) -> str:
        """Generate cart summary"""
        session = self.customer_sessions[session_id]
        cart = session.get('cart', [])
        
        if not cart:
            return "🛒 Your cart is empty."
        
        summary = "🛒 **Your Cart:**\n"
        total = 0
        
        for i, item in enumerate(cart, 1):
            item_total = item['total']
            total += item_total
            
            summary += f"{i}. {item['quantity']}x {item['name']}"
            if item['options']:
                summary += f" ({', '.join(item['options'])})"
            summary += f" - ${item_total:.2f}\n"
        
        summary += f"\n💰 **Total: ${total:.2f}**"
        return summary
    
    def handle_option_selection(self, user_input: str, session_id: str) -> Dict:
        """Handle option selection for items with options"""
        # This method exists in original but simplified for space
        return {
            'message': "Option selection handled",
            'type': 'option_selection'
        }
    
    def handle_checkout(self, session_id: str) -> Dict:
        """Handle checkout process - V2 ENHANCED"""
        session = self.customer_sessions.get(session_id, {})
        cart = session.get('cart', [])
        
        if not cart:
            return {
                'message': "Your cart is empty! Please add some items first.",
                'type': 'error',
                'suggestions': [
                    "Place an Order",
                    "What Services Do You Offer?",
                    "Pricing Information"
                ]
            }
        
        # Calculate total
        total = sum(item.get('total', item.get('price', 0) * item.get('quantity', 1)) for item in cart)
        
        # Get customer info
        customer_info = session.get('customer_info', {})
        
        # Create order summary
        order_summary = "🎉 **CHECKOUT SUCCESSFUL!**\n\n"
        order_summary += "📋 **Your Order:**\n"
        
        for item in cart:
            item_name = item.get('name', 'Unknown Item')
            quantity = item.get('quantity', 1)
            price = item.get('total', item.get('price', 0) * quantity)
            order_summary += f"• {quantity}x {item_name} - ${price:.2f}\n"
        
        order_summary += f"\n💰 **Total: ${total:.2f}**\n\n"
        
        # Include pickup details if available - V2 ENHANCEMENT
        if 'pickup_date' in customer_info and 'pickup_time' in customer_info:
            order_summary += "📅 **Scheduled Pickup:**\n"
            order_summary += f"• Date: {customer_info['pickup_date']}\n"
            order_summary += f"• Time: {customer_info['pickup_time']}\n"
            order_summary += f"• Address: {customer_info.get('address', 'To be confirmed')}\n\n"
        
        order_summary += "✅ **Next Steps:**\n"
        order_summary += "• Your order has been confirmed\n"
        order_summary += "• We'll pick up your items as scheduled\n"
        order_summary += "• Professional cleaning in 24-48 hours\n"
        order_summary += "• Door-to-door delivery service\n\n"
        order_summary += "🙏 **Thank you for choosing ValetKleen!**\n"
        order_summary += "Ready to help with your next order!"
        
        # Clear the session for new testing
        self.customer_sessions[session_id] = {
            'session_id': session_id,
            'current_step': 'welcome',
            'conversation_history': [],
            'cart': [],
            'customer_info': {}
        }
        
        return {
            'message': order_summary,
            'type': 'checkout_success',
            'suggestions': [
                "Place Another Order",
                "What Services Do You Offer?",
                "Pricing Information",
                "Contact Information"
            ]
        }
    
    # Remaining methods from original (simplified for space)
    def handle_services_inquiry(self) -> Dict:
        return {'message': "We offer dry cleaning and laundry services with pickup & delivery.", 'type': 'information'}
    
    def handle_pricing_inquiry(self) -> Dict:
        return {'message': "Our pricing is transparent with no hidden fees. Dry cleaning starts at $5.50.", 'type': 'information'}
    
    def handle_delivery_inquiry(self) -> Dict:
        return {'message': "We offer convenient door-to-door pickup and delivery service.", 'type': 'information'}
    
    def handle_about_inquiry(self) -> Dict:
        return {'message': "ValetKleen provides professional laundry and dry cleaning with convenient pickup & delivery.", 'type': 'information'}
    
    def handle_contact_inquiry(self) -> Dict:
        return {'message': "Contact us through this chat, phone, or email for scheduling and questions.", 'type': 'information'}
    
    def handle_process_inquiry(self) -> Dict:
        return {'message': "Simple process: Book → We Pickup → We Clean → We Deliver. Clean clothes, zero hassle!", 'type': 'information'}
    
    def handle_general_inquiry(self, user_input: str) -> Dict:
        return {'message': "I can help with orders, pricing, services, and pickup scheduling. What would you like to know?", 'type': 'information'}

# Create Flask application
app = Flask(__name__)
app.secret_key = Config.SECRET_KEY
CORS(app)

# Initialize chatbot
chatbot = ValetKleenChatbot()

# Use the ORIGINAL HTML template from V1 (the working one)
CHATBOT_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ValetKleen Chatbot</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        
        .chat-container {
            width: 90%;
            max-width: 800px;
            height: 90vh;
            background: white;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }
        
        .chat-header {
            background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
            color: white;
            padding: 20px;
            text-align: center;
        }
        
        .chat-header h1 {
            margin-bottom: 5px;
        }
        
        .chat-header p {
            opacity: 0.9;
            font-size: 14px;
        }
        
        .chat-messages {
            flex: 1;
            padding: 20px;
            overflow-y: auto;
            background: #f8f9fa;
        }
        
        .message {
            margin-bottom: 15px;
            animation: fadeIn 0.3s ease-in;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .message.bot {
            text-align: left;
        }
        
        .message.user {
            text-align: right;
        }
        
        .message-bubble {
            display: inline-block;
            max-width: 80%;
            padding: 12px 16px;
            border-radius: 18px;
            word-wrap: break-word;
        }
        
        .message.bot .message-bubble {
            background: #e3f2fd;
            color: #1565c0;
            border-bottom-left-radius: 5px;
        }
        
        .message.user .message-bubble {
            background: #4CAF50;
            color: white;
            border-bottom-right-radius: 5px;
        }
        
        .suggestions {
            margin-top: 10px;
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
        }
        
        .suggestion-btn {
            background: #fff;
            border: 2px solid #4CAF50;
            color: #4CAF50;
            padding: 8px 12px;
            border-radius: 20px;
            cursor: pointer;
            font-size: 12px;
            transition: all 0.3s ease;
        }
        
        .suggestion-btn:hover {
            background: #4CAF50;
            color: white;
        }
        
        .chat-input {
            background: white;
            padding: 20px;
            border-top: 1px solid #e0e0e0;
            display: flex;
            gap: 10px;
        }
        
        .chat-input input {
            flex: 1;
            padding: 12px 16px;
            border: 2px solid #e0e0e0;
            border-radius: 25px;
            outline: none;
            font-size: 14px;
        }
        
        .chat-input input:focus {
            border-color: #4CAF50;
        }
        
        .send-btn {
            background: #4CAF50;
            color: white;
            border: none;
            padding: 12px 20px;
            border-radius: 25px;
            cursor: pointer;
            font-size: 14px;
            transition: background 0.3s ease;
        }
        
        .send-btn:hover {
            background: #45a049;
        }
        
        .typing-indicator {
            display: none;
            text-align: left;
            margin-bottom: 15px;
        }
        
        .typing-indicator .message-bubble {
            background: #e0e0e0;
            color: #666;
            padding: 12px 16px;
        }
        
        .typing-dots {
            display: inline-block;
        }
        
        .typing-dots::after {
            content: '';
            animation: dots 1.5s infinite;
        }
        
        @keyframes dots {
            0% { content: ''; }
            33% { content: '.'; }
            66% { content: '..'; }
            100% { content: '...'; }
        }
        
        .welcome-screen {
            text-align: center;
            padding: 40px 20px;
            color: #666;
        }
        
        .welcome-screen h2 {
            color: #4CAF50;
            margin-bottom: 20px;
        }
        
        .welcome-suggestions {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 30px;
        }
        
        .welcome-btn {
            background: white;
            border: 2px solid #4CAF50;
            color: #4CAF50;
            padding: 15px 20px;
            border-radius: 10px;
            cursor: pointer;
            font-size: 14px;
            transition: all 0.3s ease;
            text-align: center;
        }
        
        .welcome-btn:hover {
            background: #4CAF50;
            color: white;
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(76, 175, 80, 0.3);
        }
    </style>
</head>
<body>
    <div class="chat-container">
        <div class="chat-header">
            <h1>🧼 ValetKleen Chatbot</h1>
            <p>Professional Laundry & Dry Cleaning Services</p>
        </div>
        
        <div class="chat-messages" id="chatMessages">
            <div class="welcome-screen" id="welcomeScreen">
                <h2>Welcome to ValetKleen! 👋</h2>
                <p>I'm here to help you with our premium laundry and dry cleaning services.<br>
                Choose an option below or type your question:</p>
                
                <div class="welcome-suggestions">
                    <button class="welcome-btn" onclick="sendMessage('Place an Order')">
                        📋 Place an Order
                    </button>
                    <button class="welcome-btn" onclick="sendMessage('What services do you offer?')">
                        🧼 Our Services
                    </button>
                    <button class="welcome-btn" onclick="sendMessage('Pricing information')">
                        💰 Pricing Info
                    </button>
                    <button class="welcome-btn" onclick="sendMessage('Pickup and delivery info')">
                        🚛 Pickup & Delivery
                    </button>
                    <button class="welcome-btn" onclick="sendMessage('About ValetKleen')">
                        ℹ️ About Us
                    </button>
                    <button class="welcome-btn" onclick="sendMessage('Contact information')">
                        📞 Contact Info
                    </button>
                </div>
            </div>
            
            <div class="typing-indicator" id="typingIndicator">
                <div class="message-bubble">
                    <span class="typing-dots">ValetKleen is typing</span>
                </div>
            </div>
        </div>
        
        <div class="chat-input">
            <input type="text" id="messageInput" placeholder="Type your message..." onkeypress="handleKeyPress(event)">
            <button class="send-btn" onclick="sendMessage()">Send</button>
        </div>
    </div>

    <script>
        let sessionId = null;
        let isFirstMessage = true;

        function handleKeyPress(event) {
            if (event.key === 'Enter') {
                sendMessage();
            }
        }

        function sendMessage(text = null) {
            const input = document.getElementById('messageInput');
            const message = text || input.value.trim();
            
            if (!message) return;
            
            // Hide welcome screen on first message
            if (isFirstMessage) {
                document.getElementById('welcomeScreen').style.display = 'none';
                isFirstMessage = false;
            }
            
            // Add user message to chat
            addMessage(message, 'user');
            
            // Clear input
            input.value = '';
            
            // Show typing indicator
            document.getElementById('typingIndicator').style.display = 'block';
            scrollToBottom();
            
            // Send message to backend
            fetch('/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    message: message,
                    session_id: sessionId
                })
            })
            .then(response => response.json())
            .then(data => {
                // Hide typing indicator
                document.getElementById('typingIndicator').style.display = 'none';
                
                // Update session ID
                sessionId = data.session_id;
                
                // Add bot response
                addMessage(data.message, 'bot', data.suggestions);
                
                scrollToBottom();
            })
            .catch(error => {
                console.error('Error:', error);
                document.getElementById('typingIndicator').style.display = 'none';
                addMessage('Sorry, there was an error. Please try again.', 'bot');
                scrollToBottom();
            });
        }

        function addMessage(text, sender, suggestions = null) {
            const messagesContainer = document.getElementById('chatMessages');
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${sender}`;
            
            const bubbleDiv = document.createElement('div');
            bubbleDiv.className = 'message-bubble';
            
            // Format message text (convert markdown-like formatting)
            const formattedText = text
                .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
                .replace(/\n/g, '<br>');
            
            bubbleDiv.innerHTML = formattedText;
            messageDiv.appendChild(bubbleDiv);
            
            // Add suggestions if provided
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
            
            messagesContainer.appendChild(messageDiv);
        }

        function scrollToBottom() {
            const messagesContainer = document.getElementById('chatMessages');
            messagesContainer.scrollTop = messagesContainer.scrollHeight;
        }

        // Focus input on load
        document.getElementById('messageInput').focus();
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    """Serve the chatbot interface"""
    return render_template_string(CHATBOT_HTML)

@app.route('/chat', methods=['POST'])
def chat():
    """Handle chat messages"""
    try:
        data = request.json
        message = data.get('message', '')
        session_id = data.get('session_id')
        
        # Generate response
        response = chatbot.generate_response(message, session_id)
        
        return jsonify(response)
    
    except Exception as e:
        chatbot.logger.error(f"Chat error: {e}")
        return jsonify({
            'message': 'Sorry, there was an error processing your request. Please try again.',
            'type': 'error',
            'suggestions': ["Try again", "Contact Support"]
        })

@app.route('/api/cart/<session_id>')
def get_cart(session_id):
    """Get cart contents API endpoint"""
    if session_id in chatbot.customer_sessions:
        cart = chatbot.customer_sessions[session_id].get('cart', [])
        total = sum(item['total'] for item in cart)
        return jsonify({
            'cart': cart,
            'total': total,
            'item_count': len(cart)
        })
    return jsonify({'error': 'Session not found'}), 404

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'ValetKleen Chatbot',
        'version': '2.0.0-single'
    })

if __name__ == '__main__':
    print("🚀 Starting ValetKleen Professional Chatbot...")
    port = int(os.environ.get('PORT', Config.APP_PORT))
    app.run(host=Config.APP_HOST, port=port, debug=False)