"""
ValetKleen Professional Chatbot System
A comprehensive chatbot with NLP capabilities, order management, and deployment readiness.
Built from scraped ValetKleen website data.
"""

import json
import re
import os
import smtplib
import hashlib
import hmac
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import logging
import stripe

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
from flask import Flask, request, jsonify, render_template_string, session
import uuid

# HTML parsing for website content extraction
from bs4 import BeautifulSoup

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('punkt_tab', quiet=True)
except:
    pass

class EmailService:
    """Professional email service using Hostinger SMTP"""
    
    def __init__(self):
        # Hostinger SMTP Configuration
        self.smtp_server = "smtp.hostinger.com"
        self.smtp_port = 465  # SSL
        self.email_username = "orders@valetkleen.com"
        self.email_password = os.getenv('SMTP_PASSWORD')  # Email password from environment
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
    
    def send_order_notification(self, order_data: dict, payment_info: dict = None):
        """Send professional order notification email to company"""
        try:
            # Create message
            msg = MIMEMultipart()
            msg['From'] = self.email_username
            msg['To'] = self.email_username  # Send to same address for now
            msg['Subject'] = f"🚚 New ValetKleen Logistics Order - ${order_data.get('cost', '20.00')} {'PAID' if payment_info else 'PENDING'}"
            
            # Create professional email body
            email_body = self._create_order_email_body(order_data, payment_info)
            msg.attach(MIMEText(email_body, 'html'))
            
            # Send email
            with smtplib.SMTP_SSL(self.smtp_server, self.smtp_port) as server:
                server.login(self.email_username, self.email_password)
                server.send_message(msg)
            
            self.logger.info(f"Order notification email sent successfully for order {order_data.get('order_id', 'Unknown')}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to send order notification email: {e}")
            return False
    
    def _create_order_email_body(self, order_data: dict, payment_info: dict = None) -> str:
        """Create professional HTML email body"""
        timestamp = order_data.get('timestamp', datetime.now().isoformat())
        order_id = order_data.get('order_id', f"VK{datetime.now().strftime('%Y%m%d%H%M%S')}")
        service_type = order_data.get('service_type', 'regular_order')
        
        payment_status = "✅ CONFIRMED" if payment_info else "⏳ PENDING"
        
        # Check if this is a logistics order or regular order
        if service_type == 'logistics_service':
            return self._create_logistics_email_template(order_data, payment_info, payment_status, order_id, timestamp)
        else:
            return self._create_regular_order_email_template(order_data, payment_info, payment_status, order_id, timestamp)
    
    def _create_logistics_email_template(self, order_data: dict, payment_info: dict, payment_status: str, order_id: str, timestamp: str) -> str:
        """Create email template for logistics service"""
        logistics_info = order_data.get('customer_info', {})
        payment_amount = payment_info.get('amount', 20.00) if payment_info else order_data.get('cost', 20.00)
        
        return f"""
<!DOCTYPE html>
<html>
<head>
    <style>
        body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 0; padding: 20px; background-color: #f8f9fa; }}
        .container {{ max-width: 600px; margin: 0 auto; background: white; border-radius: 10px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); }}
        .header {{ background: linear-gradient(135deg, #2c3e50, #3498db); color: white; padding: 30px 20px; text-align: center; border-radius: 10px 10px 0 0; }}
        .content {{ padding: 30px; }}
        .section {{ margin-bottom: 25px; }}
        .section-title {{ color: #2c3e50; font-size: 18px; font-weight: bold; margin-bottom: 10px; padding-bottom: 5px; border-bottom: 2px solid #3498db; }}
        .info-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin-bottom: 15px; }}
        .info-item {{ background: #f8f9fa; padding: 15px; border-radius: 8px; border-left: 4px solid #3498db; }}
        .info-label {{ font-weight: bold; color: #2c3e50; margin-bottom: 5px; }}
        .info-value {{ color: #5a6c7d; }}
        .payment-status {{ text-align: center; padding: 20px; margin: 20px 0; border-radius: 10px; }}
        .payment-confirmed {{ background: linear-gradient(135deg, #27ae60, #2ecc71); color: white; }}
        .payment-pending {{ background: linear-gradient(135deg, #f39c12, #e67e22); color: white; }}
        .footer {{ background: #2c3e50; color: white; text-align: center; padding: 20px; border-radius: 0 0 10px 10px; }}
        .highlight {{ background: #fff3cd; padding: 15px; border-radius: 8px; border-left: 4px solid #ffc107; margin: 15px 0; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>ValetKleen Order Notification</h1>
            <h2>🚚 Logistics Service Request</h2>
        </div>
        
        <div class="content">
            <div class="payment-status {'payment-confirmed' if payment_info else 'payment-pending'}">
                <h2>💰 Payment Status: {payment_status}</h2>
                <h3>Amount: ${payment_amount}</h3>
            </div>
            
            <div class="highlight">
                <strong>📋 Order ID:</strong> {order_id}<br>
                <strong>📅 Order Date:</strong> {datetime.fromisoformat(timestamp.replace('Z', '+00:00')).strftime('%B %d, %Y at %I:%M %p')}<br>
                <strong>🚚 Service Type:</strong> Logistics - Pickup & Delivery
            </div>
            
            <div class="section">
                <div class="section-title">👤 Customer Information</div>
                <div class="info-grid">
                    <div class="info-item">
                        <div class="info-label">Full Name</div>
                        <div class="info-value">{logistics_info.get('full_name', 'N/A')}</div>
                    </div>
                    <div class="info-item">
                        <div class="info-label">Email</div>
                        <div class="info-value">{logistics_info.get('email', 'N/A')}</div>
                    </div>
                    <div class="info-item">
                        <div class="info-label">Cell Phone</div>
                        <div class="info-value">{logistics_info.get('cell_phone', 'N/A')}</div>
                    </div>
                    <div class="info-item">
                        <div class="info-label">Home Phone</div>
                        <div class="info-value">{logistics_info.get('home_phone', 'N/A')}</div>
                    </div>
                </div>
                <div class="info-item">
                    <div class="info-label">Home Address</div>
                    <div class="info-value">{logistics_info.get('home_address', 'N/A')}, {logistics_info.get('zip_code', 'N/A')}</div>
                </div>
            </div>
            
            <div class="section">
                <div class="section-title">📅 Pickup Schedule</div>
                <div class="info-grid">
                    <div class="info-item">
                        <div class="info-label">Pickup Date</div>
                        <div class="info-value">{logistics_info.get('pickup_date', 'N/A')}</div>
                    </div>
                    <div class="info-item">
                        <div class="info-label">Pickup Time</div>
                        <div class="info-value">{logistics_info.get('pickup_time', 'N/A')}</div>
                    </div>
                </div>
            </div>
            
            <div class="section">
                <div class="section-title">🏪 Dry Cleaning/Laundry Mart Details</div>
                <div class="info-item">
                    <div class="info-label">Business Name</div>
                    <div class="info-value">{logistics_info.get('mart_name', 'N/A')}</div>
                </div>
                <div class="info-item">
                    <div class="info-label">Address</div>
                    <div class="info-value">{logistics_info.get('mart_address', 'N/A')}</div>
                </div>
                <div class="info-item">
                    <div class="info-label">Phone Number</div>
                    <div class="info-value">{logistics_info.get('mart_phone', 'N/A')}</div>
                </div>
            </div>
            
            <div class="highlight">
                <strong>⚡ Action Required:</strong> Schedule pickup service between customer and cleaning mart.
            </div>
        </div>
        
        <div class="footer">
            <p><strong>ValetKleen Professional Services</strong></p>
            <p>📧 orders@valetkleen.com | 🌐 valetkleen.com</p>
        </div>
    </div>
</body>
</html>
        """
    
    def _create_regular_order_email_template(self, order_data: dict, payment_info: dict, payment_status: str, order_id: str, timestamp: str) -> str:
        """Create email template for regular dry cleaning/laundry orders"""
        customer_info = order_data.get('customer_info', {})
        cart = order_data.get('cart', [])
        pickup_info = order_data.get('pickup_info', {})
        total = order_data.get('total', 0.00)
        status = order_data.get('status', 'pending_payment')
        
        # Create cart items HTML
        cart_items_html = ""
        if cart:
            for item in cart:
                cart_items_html += f"""
                <div class="info-item">
                    <div class="info-label">{item.get('quantity', 1)}x {item.get('name', 'Unknown Item')}</div>
                    <div class="info-value">${item.get('total_price', item.get('price', 0)):.2f}</div>
                </div>
                """
        else:
            cart_items_html = '<div class="info-item"><div class="info-label">No items</div><div class="info-value">N/A</div></div>'
        
        return f"""
<!DOCTYPE html>
<html>
<head>
    <style>
        body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 0; padding: 20px; background-color: #f8f9fa; }}
        .container {{ max-width: 600px; margin: 0 auto; background: white; border-radius: 10px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); }}
        .header {{ background: linear-gradient(135deg, #2c3e50, #3498db); color: white; padding: 30px 20px; text-align: center; border-radius: 10px 10px 0 0; }}
        .content {{ padding: 30px; }}
        .section {{ margin-bottom: 25px; }}
        .section-title {{ color: #2c3e50; font-size: 18px; font-weight: bold; margin-bottom: 10px; padding-bottom: 5px; border-bottom: 2px solid #3498db; }}
        .info-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin-bottom: 15px; }}
        .info-item {{ background: #f8f9fa; padding: 15px; border-radius: 8px; border-left: 4px solid #3498db; }}
        .info-label {{ font-weight: bold; color: #2c3e50; margin-bottom: 5px; }}
        .info-value {{ color: #5a6c7d; }}
        .payment-status {{ text-align: center; padding: 20px; margin: 20px 0; border-radius: 10px; }}
        .payment-confirmed {{ background: linear-gradient(135deg, #27ae60, #2ecc71); color: white; }}
        .payment-pending {{ background: linear-gradient(135deg, #f39c12, #e67e22); color: white; }}
        .footer {{ background: #2c3e50; color: white; text-align: center; padding: 20px; border-radius: 0 0 10px 10px; }}
        .highlight {{ background: #fff3cd; padding: 15px; border-radius: 8px; border-left: 4px solid #ffc107; margin: 15px 0; }}
        .total {{ background: linear-gradient(135deg, #e74c3c, #c0392b); color: white; padding: 20px; text-align: center; border-radius: 10px; font-size: 20px; font-weight: bold; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>ValetKleen Order Notification</h1>
            <h2>🧼 Dry Cleaning & Laundry Service</h2>
        </div>
        
        <div class="content">
            <div class="payment-status {'payment-confirmed' if payment_info else 'payment-pending'}">
                <h2>💰 Payment Status: {payment_status}</h2>
                <h3>Order Status: {status.title().replace('_', ' ')}</h3>
            </div>
            
            <div class="highlight">
                <strong>📋 Order ID:</strong> {order_id}<br>
                <strong>📅 Order Date:</strong> {datetime.fromisoformat(timestamp.replace('Z', '+00:00')).strftime('%B %d, %Y at %I:%M %p')}<br>
                <strong>🧼 Service Type:</strong> Professional Dry Cleaning & Laundry
            </div>
            
            <div class="section">
                <div class="section-title">👤 Customer Information</div>
                <div class="info-grid">
                    <div class="info-item">
                        <div class="info-label">Name</div>
                        <div class="info-value">{customer_info.get('name', 'N/A')}</div>
                    </div>
                    <div class="info-item">
                        <div class="info-label">Email</div>
                        <div class="info-value">{customer_info.get('email', 'N/A')}</div>
                    </div>
                    <div class="info-item">
                        <div class="info-label">Phone</div>
                        <div class="info-value">{customer_info.get('phone', 'N/A')}</div>
                    </div>
                    <div class="info-item">
                        <div class="info-label">Address</div>
                        <div class="info-value">{customer_info.get('address', 'N/A')}</div>
                    </div>
                </div>
            </div>
            
            <div class="section">
                <div class="section-title">🛍️ Order Items</div>
                {cart_items_html}
            </div>
            
            <div class="total">
                💰 Total Amount: ${total:.2f}
            </div>
            
            <div class="section">
                <div class="section-title">📅 Pickup & Delivery Schedule</div>
                <div class="info-grid">
                    <div class="info-item">
                        <div class="info-label">Pickup Date</div>
                        <div class="info-value">{pickup_info.get('pickup_date', 'N/A')}</div>
                    </div>
                    <div class="info-item">
                        <div class="info-label">Pickup Time</div>
                        <div class="info-value">{pickup_info.get('pickup_time', 'N/A')}</div>
                    </div>
                    <div class="info-item">
                        <div class="info-label">Delivery Date</div>
                        <div class="info-value">{pickup_info.get('delivery_date', 'TBD')}</div>
                    </div>
                    <div class="info-item">
                        <div class="info-label">Delivery Time</div>
                        <div class="info-value">{pickup_info.get('delivery_time', 'TBD')}</div>
                    </div>
                </div>
            </div>
            
            <div class="highlight">
                <strong>⚡ Action Required:</strong> {'Process payment and schedule pickup service.' if status == 'pending_payment' else 'Schedule pickup service with customer.'}
            </div>
        </div>
        
        <div class="footer">
            <p><strong>ValetKleen Professional Services</strong></p>
            <p>📧 orders@valetkleen.com | 🌐 valetkleen.com</p>
        </div>
    </div>
</body>
</html>
        """

class ValetKleenChatbot:
    def __init__(self):
        """Initialize the ValetKleen chatbot with NLP and knowledge base"""
        
        # Initialize logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialize Groq LLM client
        self.groq_client = Groq(api_key=os.getenv('GROQ_API_KEY'))
        
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
        
        # Customer sessions storage
        self.customer_sessions = {}
        
        # Enhanced website content knowledge base
        self.website_knowledge = self.extract_website_content()
        
        self.logger.info("ValetKleen Chatbot V2 initialized successfully with FRESH CODE!")
        # Initialize email service
        self.email_service = EmailService()
        
        self.logger.info(f"Extracted {len(self.website_knowledge)} website content sections")
        print("FRESH CODE LOADED: Laundry Service & Wedding Dress Options fixes active!")
    
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
                    'dashiki': {'name': 'Men\'s Dashiki (2 PC)', 'price': 18.00, 'options': ['No Starch', 'Light Starch', 'Medium Starch', 'Heavy Starch', 'Hanger', 'Fold']},
                    'agbada': {'name': 'Men\'s Agbada (3 PC)', 'price': 20.00, 'options': ['No Starch', 'Light Starch', 'Medium Starch', 'Heavy Starch', 'Hanger', 'Fold']},
                    'wedding_dress': {'name': 'Wedding Dress', 'price': 180.00, 'options': ['Boxed', 'No Box']},
                    # New items
                    'jacket': {'name': 'Jacket', 'price': 9.50, 'options': []},
                    'hood': {'name': 'Hood', 'price': 7.00, 'options': []},
                    'tuxedo': {'name': 'Tuxedo', 'price': 18.00, 'options': []},
                    'suit_2piece': {'name': '2 Piece Suit', 'price': 18.00, 'options': []},
                    'tie': {'name': 'Tie', 'price': 4.00, 'options': []},
                    'sport_coat': {'name': 'Sport Coat', 'price': 9.50, 'options': []},
                    'blouse': {'name': 'Blouse', 'price': 6.50, 'options': []},
                    'polo_shirt': {'name': 'Polo Shirt', 'price': 5.50, 'options': []},
                    'blazer': {'name': 'Blazer', 'price': 7.00, 'options': []},
                    'suit_3piece': {'name': '3 Piece Suit', 'price': 20.00, 'options': []},
                    'skirt': {'name': 'Skirt', 'price': 6.50, 'options': []},
                    'tuxedo_shirt': {'name': 'Tuxedo Shirt', 'price': 6.00, 'options': []},
                    'ladies_shirt': {'name': 'Ladies Shirt', 'price': 6.00, 'options': []},
                    'robe': {'name': 'Robe', 'price': 9.00, 'options': []},
                    'scarf': {'name': 'Scarf', 'price': 4.50, 'options': []},
                    'chef_coat': {'name': 'Chef Coat', 'price': 6.50, 'options': []},
                    'sweater': {'name': 'Sweater', 'price': 6.50, 'options': []},
                    'apron': {'name': 'Apron', 'price': 5.00, 'options': []}
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
                max_completion_tokens=200,
                top_p=1,
                stream=False
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
    
    def parse_item_request_with_llm(self, user_input: str, service_type: str) -> List[Dict]:
        """Enhanced item parsing using Groq LLM"""
        try:
            # Get available items for the service type
            service_items = self.service_catalog[service_type]['items']
            
            # Create list of available items for the prompt
            available_items = []
            for key, item in service_items.items():
                available_items.append(f"- {item['name']} (${item['price']:.2f})")
            
            available_items_text = '\n'.join(available_items)
            
            prompt = f"""You are helping parse a customer order for {service_type} services.

Available {service_type} items:
{available_items_text}

Customer message: "{user_input}"

Parse this message to extract:
1. Which items the customer wants
2. How many of each item
3. Match items to the exact names from the available list above

IMPORTANT: Only match items from the {service_type} service list above. Do not match items from other services.

Respond in this exact JSON format:
{{
    "parsed_items": [
        {{
            "item_name": "exact name from available list",
            "quantity": number,
            "confidence": 0.0-1.0
        }}
    ]
}}

Rules:
- Only include items that clearly match the {service_type} available list
- Use exact item names from the list above
- Default quantity is 1 if not specified
- Be flexible with partial matches - if user says "small bag", match to "Small Bag (12 lb capacity)"
- For laundry bags: "small bag" = "Small Bag", "medium bag" = "Medium Bag", "large bag" = "Large Bag", "king bag" = "King Size Premium Bag"
- For comforters: match size keywords like "twin", "full", "queen", "king"
- For blankets: match size keywords like "twin", "full", "queen", "king"  
- For mattress covers: match size keywords like "twin", "full", "queen", "king"
- For dry cleaning: match clothing items like "shirt", "dress", "coat", "pants"
- Consider plural/singular forms and size variations
- If multiple quantities mentioned for different items, parse each separately"""

            # Call Groq API
            completion = self.groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_completion_tokens=300,
                top_p=1,
                stream=False
            )
            
            response = completion.choices[0].message.content
            
            # Parse JSON response
            try:
                import json
                result = json.loads(response)
                parsed_items = []
                
                for item_data in result.get('parsed_items', []):
                    item_name = item_data.get('item_name', '')
                    quantity = item_data.get('quantity', 1)
                    confidence = item_data.get('confidence', 0.8)
                    
                    # Find matching item key in service catalog
                    for key, item_info in service_items.items():
                        if item_info['name'].lower() == item_name.lower():
                            item_dict = {
                                'key': key,
                                'name': item_info['name'],
                                'quantity': quantity,
                                'price': item_info['price'],
                                'confidence': confidence
                            }
                            
                            # DON'T add 'options' key unless user has already selected options
                            # This allows the options logic to detect items that need options
                            
                            parsed_items.append(item_dict)
                            break
                
                self.logger.info(f"LLM Item Parsing - Input: '{user_input}' -> Found {len(parsed_items)} items")
                
                # DEBUG: Log what the LLM actually returned
                for i, item in enumerate(parsed_items):
                    self.logger.info(f"DEBUG LLM ITEM {i}: {item}")
                
                return parsed_items
                
            except json.JSONDecodeError:
                self.logger.error(f"Failed to parse LLM JSON response: {response}")
                
        except Exception as e:
            self.logger.error(f"Error in LLM item parsing: {e}")
        
        # Fallback to original method
        return self.parse_item_request(user_input, service_type)
    
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
        
        # Calculate dynamic pricing based on options
        base_price = item_info['price']
        
        # Special pricing for wedding dress based on boxing option
        if item_key == 'wedding_dress' and selected_options:
            if 'Boxed' in selected_options:
                base_price = 180.00
            elif 'No Box' in selected_options:
                base_price = 150.00
        
        cart_item = {
            'service_type': service_type,
            'item_key': item_key,
            'name': item_info['name'],
            'price': base_price,
            'quantity': quantity,
            'options': selected_options,
            'total': base_price * quantity
        }
        
        self.customer_sessions[session_id]['cart'].append(cart_item)
        return True
    
    def generate_response(self, user_input: str, session_id: str = None) -> Dict:
        """Generate chatbot response based on user input and intent"""
        
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
        self.logger.info(f"DEBUG: Current step = '{current_step}', Intent = '{intent}', Input = '{user_input}'")
        
        # Check for payment keywords FIRST (before any step handling)
        user_input_lower = user_input.lower()
        if 'pay now' in user_input_lower:
            return self.handle_payment(session_id)
        elif any(keyword in user_input_lower for keyword in ['proceed to checkout', 'checkout', 'complete order', 'finish order', 'place order now']):
            return self.handle_checkout(session_id)
        
        # Handle order flow steps
        if current_step == 'selecting_service_type':
            return self.handle_service_type_selection(user_input, session_id)
        elif current_step == 'collecting_info':
            return self.handle_info_collection(user_input, session_id)
        elif current_step == 'collecting_logistics_info':
            return self.handle_logistics_info_collection(user_input, session_id)
        elif current_step == 'collecting_pickup_info':
            return self.handle_pickup_info_collection(user_input, session_id)
        elif current_step == 'selecting_service':
            self.logger.info(f"DEBUG: In selecting_service step, input: '{user_input}'")
            return self.handle_service_selection(user_input, session_id)
        elif current_step == 'selecting_items':
            self.logger.info(f"DEBUG: In selecting_items step, input: '{user_input}'")
            return self.handle_item_selection(user_input, session_id)
        elif current_step == 'adding_options':
            self.logger.info(f"DEBUG: In adding_options step, input: '{user_input}'")
            return self.handle_option_selection(user_input, session_id)
        elif current_step == 'logistics_confirmation':
            return self.handle_logistics_confirmation(user_input, session_id)
        
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
        """Start the order placement process with service selection"""
        session = self.customer_sessions[session_id]
        session['current_step'] = 'selecting_service_type'
        
        return {
            'message': "🛍️ Great! I'd love to help you place an order.\n\n**Please choose which service you need:**",
            'type': 'service_type_selection',
            'suggestions': [
                "🧺 Our Laundry Services",
                "👔 Our Dry-Cleaning Services", 
                "🚚 Logistics Service"
            ]
        }
    
    def handle_info_collection(self, user_input: str, session_id: str) -> Dict:
        """Handle customer information collection"""
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
            # Basic email validation
            if '@' in user_input and '.' in user_input:
                customer_info['email'] = user_input.strip()
                return {
                    'message': "Perfect! 🏠 **Your Address (for pickup & delivery):**",
                    'type': 'info_collection',
                    'collecting': 'address'
                }
            else:
                return {
                    'message': "Please enter a valid email address (e.g., john@example.com):",
                    'type': 'info_collection',
                    'collecting': 'email'
                }
        elif 'address' not in customer_info:
            customer_info['address'] = user_input.strip()
            return {
                'message': "Great! 📱 **Your Phone Number:**",
                'type': 'info_collection',
                'collecting': 'phone'
            }
        elif 'phone' not in customer_info:
            customer_info['phone'] = user_input.strip()
            
            # Check if user has already selected a service
            selected_service_type = session.get('selected_service_type')
            if selected_service_type:
                # User already selected service, go directly to item selection
                session['current_step'] = 'selecting_items'
                session['selected_service'] = selected_service_type  # Set this for consistency
                
                if selected_service_type == 'dry_cleaning':
                    return self.show_dry_cleaning_menu()
                elif selected_service_type == 'laundry':
                    return self.show_laundry_menu()
                else:
                    # Fallback for any other service type
                    service_name = "dry cleaning" if selected_service_type == "dry_cleaning" else "laundry"
                    return {
                        'message': f"Perfect! All set, {customer_info['name']}! 🎯\n\nNow, what {service_name} items would you like? You can tell me specifically (e.g., '2 dress shirts') or choose from the menu:",
                        'type': 'item_selection',
                        'suggestions': self.get_item_suggestions(selected_service_type)
                    }
            else:
                # No service selected yet, ask for service selection
                session['current_step'] = 'selecting_service'
                return {
                    'message': f"Perfect! All set, {customer_info['name']}! 🎯\n\nNow, which service would you like?",
                    'type': 'service_selection',
                    'suggestions': [
                        "👔 Dry Cleaning Services", 
                        "🧺 Laundry Services"
                    ]
                }
    
    def handle_service_type_selection(self, user_input: str, session_id: str) -> Dict:
        """Handle initial service type selection (Laundry, Dry-Cleaning, or Logistics)"""
        session = self.customer_sessions[session_id]
        processed_input = user_input.lower()
        
        if 'logistics' in processed_input or 'pickup and delivery' in processed_input:
            session['selected_service_type'] = 'logistics'
            session['current_step'] = 'collecting_logistics_info'
            # Initialize logistics info structure
            session['logistics_info'] = {}
            
            return {
                'message': "🚚 **Logistics Service Selected**\n\n*Do you have a cleaner or laundry service you prefer?*\n*Already have a laundry? Let ValetKleen handle just the pickup and delivery.*\n\nPlease provide your information:\n\n👤 **Full Name:**",
                'type': 'logistics_info_collection',
                'suggestions': []
            }
        elif 'laundry' in processed_input:
            session['selected_service_type'] = 'laundry'
            session['current_step'] = 'collecting_info'
            
            return {
                'message': "🧺 **Laundry Services Selected**\n\nFirst, I'll need some information from you:\n\n👤 **Your Name:**",
                'type': 'info_collection',
                'collecting': 'name',
                'suggestions': []
            }
        elif 'dry' in processed_input or 'cleaning' in processed_input:
            session['selected_service_type'] = 'dry_cleaning'
            session['current_step'] = 'collecting_info'
            
            return {
                'message': "👔 **Dry-Cleaning Services Selected**\n\nFirst, I'll need some information from you:\n\n👤 **Your Name:**",
                'type': 'info_collection',
                'collecting': 'name',
                'suggestions': []
            }
        else:
            return {
                'message': "Please select one of our service types:",
                'type': 'service_type_selection',
                'suggestions': [
                    "🧺 Our Laundry Services",
                    "👔 Our Dry-Cleaning Services", 
                    "🚚 Logistics Service"
                ]
            }
    
    def handle_logistics_info_collection(self, user_input: str, session_id: str) -> Dict:
        """Handle logistics service information collection"""
        session = self.customer_sessions[session_id]
        logistics_info = session['logistics_info']
        
        # Determine what information we're collecting
        if 'full_name' not in logistics_info:
            logistics_info['full_name'] = user_input.strip()
            return {
                'message': f"Thank you, {logistics_info['full_name']}!\n\n🏠 **Home Address:**",
                'type': 'logistics_info_collection',
                'collecting': 'home_address'
            }
        elif 'home_address' not in logistics_info:
            logistics_info['home_address'] = user_input.strip()
            return {
                'message': "📮 **Zip Code:**",
                'type': 'logistics_info_collection',
                'collecting': 'zip_code'
            }
        elif 'zip_code' not in logistics_info:
            logistics_info['zip_code'] = user_input.strip()
            return {
                'message': "📱 **Cell Phone:**",
                'type': 'logistics_info_collection',
                'collecting': 'cell_phone'
            }
        elif 'cell_phone' not in logistics_info:
            logistics_info['cell_phone'] = user_input.strip()
            return {
                'message': "🏠 **Home Phone:**",
                'type': 'logistics_info_collection',
                'collecting': 'home_phone'
            }
        elif 'home_phone' not in logistics_info:
            logistics_info['home_phone'] = user_input.strip()
            return {
                'message': "📧 **Email:**",
                'type': 'logistics_info_collection',
                'collecting': 'email'
            }
        elif 'email' not in logistics_info:
            if '@' in user_input and '.' in user_input:
                logistics_info['email'] = user_input.strip()
                return {
                    'message': "📅 **Pickup Date (e.g., Monday, Dec 15 or 12/15/2024):**",
                    'type': 'logistics_info_collection',
                    'collecting': 'pickup_date'
                }
            else:
                return {
                    'message': "Please enter a valid email address:",
                    'type': 'logistics_info_collection',
                    'collecting': 'email'
                }
        elif 'pickup_date' not in logistics_info:
            logistics_info['pickup_date'] = user_input.strip()
            return {
                'message': "🕐 **Pickup Time (e.g., 2:00 PM or 14:00):**",
                'type': 'logistics_info_collection',
                'collecting': 'pickup_time'
            }
        elif 'pickup_time' not in logistics_info:
            logistics_info['pickup_time'] = user_input.strip()
            return {
                'message': "🏪 **Name of the dry cleaning or laundry mart:**",
                'type': 'logistics_info_collection',
                'collecting': 'mart_name'
            }
        elif 'mart_name' not in logistics_info:
            logistics_info['mart_name'] = user_input.strip()
            return {
                'message': "📍 **Address of the dry cleaning or laundry mart:**",
                'type': 'logistics_info_collection',
                'collecting': 'mart_address'
            }
        elif 'mart_address' not in logistics_info:
            logistics_info['mart_address'] = user_input.strip()
            return {
                'message': "📞 **Phone number of the dry cleaning or laundry mart:**",
                'type': 'logistics_info_collection',
                'collecting': 'mart_phone'
            }
        elif 'mart_phone' not in logistics_info:
            logistics_info['mart_phone'] = user_input.strip()
            session['current_step'] = 'logistics_confirmation'
            
            # Create confirmation summary
            summary = f"""📋 **Please review your logistics service information:**

👤 **Customer Information:**
• Full Name: {logistics_info['full_name']}
• Home Address: {logistics_info['home_address']}
• Zip Code: {logistics_info['zip_code']}
• Cell Phone: {logistics_info['cell_phone']}
• Home Phone: {logistics_info['home_phone']}
• Email: {logistics_info['email']}

📅 **Pickup Information:**
• Pickup Date: {logistics_info['pickup_date']}
• Pickup Time: {logistics_info['pickup_time']}

🏪 **Dry Cleaning/Laundry Mart:**
• Name: {logistics_info['mart_name']}
• Address: {logistics_info['mart_address']}
• Phone: {logistics_info['mart_phone']}

Please type **"confirm"** to proceed with your logistics service request."""

            return {
                'message': summary,
                'type': 'logistics_confirmation',
                'suggestions': ['Confirm', 'Edit Information']
            }
        else:
            return {
                'message': "All information collected. Please confirm to proceed.",
                'type': 'logistics_confirmation',
                'suggestions': ['Confirm']
            }
    
    def handle_pickup_info_collection(self, user_input: str, session_id: str) -> Dict:
        """Handle pickup information collection for dry cleaning and laundry orders"""
        session = self.customer_sessions[session_id]
        pickup_info = session['pickup_info']
        
        # Determine what information we're collecting
        collecting = pickup_info.get('collecting')
        
        if collecting == 'pickup_date':
            pickup_info['pickup_date'] = user_input.strip()
            pickup_info['collecting'] = 'pickup_time'
            return {
                'message': f"📅 Pickup date set: **{pickup_info['pickup_date']}**\n\n⏰ **Please provide your preferred pickup time** (e.g., 9:00 AM, 2:30 PM, Morning, Afternoon):",
                'type': 'pickup_scheduling',
                'suggestions': [
                    "9:00 AM",
                    "2:00 PM", 
                    "Morning",
                    "Afternoon"
                ]
            }
        
        elif collecting == 'pickup_time':
            pickup_info['pickup_time'] = user_input.strip()
            
            # Pickup scheduling complete, now create Stripe checkout
            return self.create_stripe_checkout(session_id)
            
        return {
            'message': "Please provide the requested information.",
            'type': 'pickup_scheduling'
        }
    
    def create_stripe_checkout(self, session_id: str) -> Dict:
        """Create Stripe checkout session for order payment"""
        try:
            session = self.customer_sessions[session_id]
            cart = session.get('cart', [])
            customer_info = session.get('customer_info', {})
            pickup_info = session.get('pickup_info', {})
            
            # Calculate total and create line items
            line_items = []
            total_amount = 0
            
            for item in cart:
                item_name = item.get('name', 'Unknown Item')
                quantity = item.get('quantity', 1)
                unit_price = item.get('price', 0)
                total_price = item.get('total_price', unit_price * quantity)
                total_amount += total_price
                
                line_items.append({
                    'price_data': {
                        'currency': 'usd',
                        'product_data': {
                            'name': f"{item_name} x{quantity}",
                            'description': item.get('description', f'ValetKleen {item_name} - Professional laundry and dry cleaning service')
                        },
                        'unit_amount': int(total_price * 100),  # Convert to cents
                    },
                    'quantity': 1,
                })
            
            # Generate unique order ID
            order_id = f"VK_{datetime.now().strftime('%Y%m%d%H%M%S')}_{session_id[:8]}"
            
            # Set Stripe API key - using LIVE key for production payments
            stripe.api_key = os.getenv('STRIPE_SECRET_KEY')
            
            # Create checkout session
            checkout_session = stripe.checkout.Session.create(
                payment_method_types=['card'],
                line_items=line_items,
                mode='payment',
                success_url=f"http://localhost:5000/payment-success?session_id={session_id}&order_id={order_id}",
                cancel_url=f"http://localhost:5000/payment-cancel?session_id={session_id}",
                metadata={
                    'order_id': order_id,
                    'session_id': session_id,
                    'customer_email': customer_info.get('email', ''),
                    'customer_name': customer_info.get('name', ''),
                    'order_type': 'regular_order',
                    'pickup_date': pickup_info.get('pickup_date', ''),
                    'pickup_time': pickup_info.get('pickup_time', ''),
                    'delivery_date': pickup_info.get('delivery_date', ''),
                    'delivery_time': pickup_info.get('delivery_time', '')
                }
            )
            
            # Store checkout session info
            session['stripe_session_id'] = checkout_session.id
            session['order_id'] = order_id
            self.customer_sessions[session_id] = session
            
            # Send email notification to company (order pending payment)
            try:
                order_data = {
                    'order_id': order_id,
                    'customer_info': customer_info,
                    'cart': cart,
                    'pickup_info': pickup_info,
                    'total': total_amount,
                    'status': 'pending_payment',
                    'service_type': 'regular_order'
                }
                self.email_service.send_order_notification(order_data)
                self.logger.info(f"Order pending notification email sent for {order_id}")
            except Exception as e:
                self.logger.error(f"Failed to send order pending notification email for {order_id}: {e}")
            
            return {
                'message': f"✅ **Order Ready for Payment!**\n\n📋 **Order Summary:**\n{self.format_order_summary(cart, pickup_info)}\n\n💰 **Total: ${total_amount:.2f}**\n\n🔗 **Payment Link:**\n[Click here to complete your secure payment →]({checkout_session.url})\n\n*Secure payment powered by Stripe*",
                'type': 'payment_ready',
                'payment_url': checkout_session.url,
                'order_id': order_id,
                'total': total_amount,
                'suggestions': []  # No suggestions needed - user should click the payment link
            }
            
        except stripe.error.StripeError as e:
            self.logger.error(f"Stripe error: {str(e)}")
            return {
                'message': f"❌ **Payment Error**\n\nSorry, there was an issue setting up payment. Please try again or contact customer service.\n\nError: {str(e)}",
                'type': 'payment_error',
                'suggestions': [
                    "Try Again",
                    "Contact Support",
                    "Cancel Order"
                ]
            }
        except Exception as e:
            self.logger.error(f"Checkout creation error: {str(e)}")
            return {
                'message': "❌ **Order Error**\n\nSorry, there was an issue processing your order. Please try again.",
                'type': 'order_error',
                'suggestions': [
                    "Try Again",
                    "Start Over"
                ]
            }
    
    def format_order_summary(self, cart: List[Dict], pickup_info: Dict) -> str:
        """Format order summary for display"""
        summary = ""
        for item in cart:
            item_name = item.get('name', 'Unknown Item')
            quantity = item.get('quantity', 1)
            price = item.get('total_price', item.get('price', 0) * quantity)
            summary += f"• {quantity}x {item_name} - ${price:.2f}\n"
        
        summary += f"\n📅 **Pickup:** {pickup_info.get('pickup_date', 'TBD')} at {pickup_info.get('pickup_time', 'TBD')}"
        summary += f"\n🚛 **Delivery:** {pickup_info.get('delivery_date', 'TBD')} at {pickup_info.get('delivery_time', 'TBD')}"
        
        return summary
    
    def handle_logistics_confirmation(self, user_input: str, session_id: str) -> Dict:
        """Handle logistics service confirmation"""
        session = self.customer_sessions[session_id]
        user_input_lower = user_input.lower()
        
        # Handle Edit Information request
        if 'edit information' in user_input_lower or 'edit info' in user_input_lower:
            # Reset logistics info and go back to collection step
            session['logistics_info'] = {}
            session['current_step'] = 'collecting_logistics_info'
            
            return {
                'message': "📝 **Let's Update Your Information**\n\nI'll help you re-enter your details. Let's start fresh:\n\n**Please provide your full name:**",
                'type': 'logistics_info_collection',
                'suggestions': ['Cancel', 'Contact Support']
            }
        
        elif 'confirm' in user_input_lower:
            logistics_info = session['logistics_info']
            
            # Create order data with unique ID
            order_id = f"LOG{datetime.now().strftime('%Y%m%d%H%M%S')}{session_id[:4]}"
            order_data = {
                'order_id': order_id,
                'service_type': 'logistics',
                'customer_info': logistics_info,
                'cost': 20.00,
                'status': 'confirmed',
                'timestamp': datetime.now().isoformat()
            }
            
            # Store the order
            session['logistics_order'] = order_data
            
            # Send email notification to company
            try:
                self.email_service.send_order_notification(order_data)
                self.logger.info(f"Order notification email sent for {order_id}")
            except Exception as e:
                self.logger.error(f"Failed to send order notification email for {order_id}: {e}")
            
            return {
                'message': f"✅ **Logistics Service Confirmed!**\n\n📋 **Order ID:** {order_id}\n\n🚚 Your pickup and delivery service has been scheduled.\n\n💰 **Total Cost: $20.00**\n\nWe'll handle the pickup and delivery between you and your preferred dry cleaning/laundry service.\n\n📧 Order details have been sent to our team for processing.",
                'type': 'logistics_completed',
                'suggestions': [
                    'Pay Now',
                    'Place Another Order',
                    'Contact Support'
                ]
            }
        else:
            return {
                'message': "Please type **'confirm'** to proceed with your logistics service request, or let me know what you'd like to change.",
                'type': 'logistics_confirmation',
                'suggestions': ['Confirm', 'Edit Information']
            }
    
    def handle_payment(self, session_id: str) -> Dict:
        """Handle payment requests - redirect to Stripe for logistics service"""
        session = self.customer_sessions[session_id]
        
        # Check if this is for logistics service
        if 'logistics_order' in session:
            return {
                'message': "💳 **Redirecting to Payment**\n\n🚚 **Logistics Service - $20.00**\n\nClick the link below to complete your payment securely through Stripe:\n\n🔗 **[Pay Now - $20.00](https://buy.stripe.com/eVqcN79RAfNgacKcATbfO01)**\n\nAfter payment, you'll receive a confirmation email with all service details.",
                'type': 'payment_redirect',
                'payment_url': 'https://buy.stripe.com/eVqcN79RAfNgacKcATbfO01',
                'suggestions': [
                    'Place Another Order',
                    'Contact Support',
                    'Need Help?'
                ]
            }
        elif session.get('cart', []):
            # For regular services (future implementation)
            total = sum(item['total'] for item in session.get('cart', []))
            return {
                'message': f"💳 **Payment processing for regular services will be available soon.**\n\n💰 **Total: ${total:.2f}**\n\nFor now, please contact our customer service to complete your order.",
                'type': 'payment_pending',
                'suggestions': [
                    'Contact Support',
                    'Place Another Order'
                ]
            }
        else:
            return {
                'message': "❗ No active order found to process payment.\n\nPlease place an order first before proceeding to payment.",
                'type': 'no_order',
                'suggestions': [
                    'Place an Order',
                    'Contact Support'
                ]
            }
    
    def handle_service_selection(self, user_input: str, session_id: str) -> Dict:
        """Handle service type selection"""
        session = self.customer_sessions[session_id]
        processed_input = user_input.lower()
        
        self.logger.info(f"DEBUG: Service selection - Input: '{user_input}', Processed: '{processed_input}'")
        self.logger.info(f"DEBUG: 'dry' in processed_input = {('dry' in processed_input)}")
        self.logger.info(f"DEBUG: 'laundry' in processed_input = {('laundry' in processed_input)}")
        
        if 'dry cleaning' in processed_input or ('dry' in processed_input and 'laundry' not in processed_input):
            session['selected_service'] = 'dry_cleaning'
            session['current_step'] = 'selecting_items'
            self.logger.info("DEBUG: Selected DRY CLEANING service")
            return self.show_dry_cleaning_menu()
        elif 'laundry' in processed_input:
            session['selected_service'] = 'laundry'
            session['current_step'] = 'selecting_items'
            self.logger.info("DEBUG: Selected LAUNDRY service")
            return self.show_laundry_menu()
        else:
            self.logger.info("DEBUG: NO SERVICE MATCH - going to else clause")
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
        
        # Parse user input for items and quantities using LLM
        parsed_items = self.parse_item_request_with_llm(user_input, selected_service)
        
        if not parsed_items:
            return {
                'message': "I couldn't understand that. Please try again or select from the menu:",
                'type': 'item_selection',
                'suggestions': self.get_item_suggestions(selected_service)
            }
        
        # Separate items into those needing options and those ready to add
        items_needing_options = []
        items_ready_to_add = []
        
        self.logger.info(f"DEBUG OPTIONS: Checking {len(parsed_items)} parsed items for options")
        for item_info in parsed_items:
            item_key = item_info['key']
            item_details = self.service_catalog[selected_service]['items'][item_key]
            
            self.logger.info(f"DEBUG OPTIONS: Item '{item_details['name']}' has options: {item_details['options']}")
            self.logger.info(f"DEBUG OPTIONS: 'options' in item_info: {'options' in item_info}")
            
            # If item has options and no options selected yet, queue it
            if item_details['options'] and 'options' not in item_info:
                items_needing_options.append(item_info)
                self.logger.info(f"DEBUG OPTIONS: Queued {item_details['name']} for options")
            else:
                items_ready_to_add.append(item_info)
                self.logger.info(f"DEBUG OPTIONS: {item_details['name']} ready to add")
        
        # If there are items needing options, handle them one by one
        if items_needing_options:
            # Store all items in session for processing
            session['items_needing_options'] = items_needing_options[1:]  # Store remaining items
            session['items_ready_to_add'] = items_ready_to_add  # Store items ready to add
            session['pending_item'] = items_needing_options[0]  # Process first item
            session['current_step'] = 'adding_options'
            
            # Ask for options for the first item
            item_info = items_needing_options[0]
            item_key = item_info['key']
            item_details = self.service_catalog[selected_service]['items'][item_key]
            
            self.logger.info(f"DEBUG OPTIONS: Asking for options for {item_details['name']}")
            
            # Special formatting for items with multiple option categories
            if item_key in ['agbada', 'dashiki']:
                message = f"Perfect! I found: **{item_info['quantity']}x {item_details['name']}**\n\n"
                message += "**Starch Level Options:**\n"
                message += "• No Starch\n• Light Starch\n• Medium Starch\n• Heavy Starch\n\n"
                message += "**Cleaning Instructions:**\n"
                message += "• Hanger\n• Fold\n\n"
                message += "**Please specify:** Do you want it starched? And should we hang or fold it?\n"
                message += "Example: \"Medium Starch and Hanger\" or \"No Starch and Fold\""
            elif item_key == 'wedding_dress':
                message = f"Perfect! I found: **{item_info['quantity']}x {item_details['name']}**\n\n"
                message += "**Boxing Options:**\n"
                message += "• **Boxed** - $180.00 (Professional preservation box)\n"
                message += "• **No Box** - $150.00 (Standard cleaning only)\n\n"
                message += "**Which option would you prefer?**"
            else:
                message = f"Perfect! I found: **{item_info['quantity']}x {item_details['name']}**\n\n"
                message += f"This item has these options:\n\n"
                message += "\n".join([f"• {option}" for option in item_details['options']])
                message += "\n\n**Which option would you prefer?**"
            
            return {
                'message': message,
                'type': 'option_selection',
                'suggestions': item_details['options'] + ['None']
            }
        
        # No items need options, add all items to cart
        added_items = []
        for item_info in items_ready_to_add:
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
    
    def handle_option_selection(self, user_input: str, session_id: str) -> Dict:
        """Handle option selection for items with options"""
        session = self.customer_sessions[session_id]
        pending_item = session.get('pending_item')
        
        if not pending_item:
            return {
                'message': "Sorry, there's no pending item for option selection. Please start your order again.",
                'type': 'error',
                'suggestions': ["Place an Order", "View Services"]
            }
        
        selected_service = session.get('selected_service')
        item_key = pending_item['key']
        quantity = pending_item['quantity']
        
        # Parse selected options
        selected_options = []
        user_input_lower = user_input.lower()
        
        if 'none' not in user_input_lower:
            # Get available options for this item
            item_details = self.service_catalog[selected_service]['items'][item_key]
            available_options = item_details['options']
            
            # Special handling for agbada/dashiki - need both starch and cleaning instruction
            if item_key in ['agbada', 'dashiki']:
                # Parse starch level
                starch_options = ['No Starch', 'Light Starch', 'Medium Starch', 'Heavy Starch']
                cleaning_options = ['Hanger', 'Fold']
                
                starch_selected = None
                cleaning_selected = None
                
                for starch in starch_options:
                    if starch.lower() in user_input_lower:
                        starch_selected = starch
                        break
                
                for cleaning in cleaning_options:
                    if cleaning.lower() in user_input_lower:
                        cleaning_selected = cleaning
                        break
                
                # Add selected options
                if starch_selected:
                    selected_options.append(starch_selected)
                if cleaning_selected:
                    selected_options.append(cleaning_selected)
                
                # If user didn't select both, ask them to clarify
                if not starch_selected or not cleaning_selected:
                    return {
                        'message': f"Please specify both:\n\n• **Starch level:** No Starch, Light Starch, Medium Starch, or Heavy Starch\n• **Cleaning instruction:** Hanger or Fold\n\nExample: \"Medium Starch and Hanger\"",
                        'type': 'option_selection',
                        'suggestions': ['No Starch and Hanger', 'Light Starch and Fold', 'Medium Starch and Hanger', 'Heavy Starch and Fold']
                    }
            else:
                # Regular option parsing for other items
                for option in available_options:
                    if option.lower() in user_input_lower:
                        selected_options.append(option)
        
        # Add current item to cart with selected options
        if self.add_to_cart(session_id, selected_service, item_key, quantity, selected_options):
            item_name = self.service_catalog[selected_service]['items'][item_key]['name']
            options_text = f" ({', '.join(selected_options)})" if selected_options else ""
            
            # Check if there are more items needing options
            items_needing_options = session.get('items_needing_options', [])
            
            if items_needing_options:
                # Process next item needing options
                next_item = items_needing_options[0]
                session['items_needing_options'] = items_needing_options[1:]  # Remove processed item
                session['pending_item'] = next_item  # Set next item as pending
                
                # Ask for options for next item
                next_item_key = next_item['key']
                next_item_details = self.service_catalog[selected_service]['items'][next_item_key]
                
                self.logger.info(f"DEBUG OPTIONS: Next item needing options: {next_item_details['name']}")
                
                # Special formatting for items with multiple option categories
                if next_item_key in ['agbada', 'dashiki']:
                    next_message = f"✅ Added: {quantity}x {item_name}{options_text}\n\n"
                    next_message += f"Next item: **{next_item['quantity']}x {next_item_details['name']}**\n\n"
                    next_message += "**Starch Level Options:**\n"
                    next_message += "• No Starch\n• Light Starch\n• Medium Starch\n• Heavy Starch\n\n"
                    next_message += "**Cleaning Instructions:**\n"
                    next_message += "• Hanger\n• Fold\n\n"
                    next_message += "**Please specify:** Do you want it starched? And should we hang or fold it?\n"
                    next_message += "Example: \"Medium Starch and Hanger\" or \"No Starch and Fold\""
                elif next_item_key == 'wedding_dress':
                    next_message = f"✅ Added: {quantity}x {item_name}{options_text}\n\n"
                    next_message += f"Next item: **{next_item['quantity']}x {next_item_details['name']}**\n\n"
                    next_message += "**Boxing Options:**\n"
                    next_message += "• **Boxed** - $180.00 (Professional preservation box)\n"
                    next_message += "• **No Box** - $150.00 (Standard cleaning only)\n\n"
                    next_message += "**Which option would you prefer?**"
                else:
                    next_message = f"✅ Added: {quantity}x {item_name}{options_text}\n\n"
                    next_message += f"Next item: **{next_item['quantity']}x {next_item_details['name']}**\n\n"
                    next_message += f"This item has these options:\n\n"
                    next_message += "\n".join([f"• {option}" for option in next_item_details['options']])
                    next_message += "\n\n**Which option would you prefer?**"
                
                return {
                    'message': next_message,
                    'type': 'option_selection',
                    'suggestions': next_item_details['options'] + ['None']
                }
            
            # No more items needing options, add all remaining items ready to add
            items_ready_to_add = session.get('items_ready_to_add', [])
            added_items = [f"{quantity}x {item_name}{options_text}"]  # Include current item
            
            for item_info in items_ready_to_add:
                ready_item_key = item_info['key']
                ready_quantity = item_info['quantity']
                ready_item_details = self.service_catalog[selected_service]['items'][ready_item_key]
                
                if self.add_to_cart(session_id, selected_service, ready_item_key, ready_quantity):
                    added_items.append(f"{ready_quantity}x {ready_item_details['name']}")
            
            # Clear all pending items and reset session
            session['pending_item'] = None
            session['items_needing_options'] = []
            session['items_ready_to_add'] = []
            session['current_step'] = 'selecting_items'
            
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
        else:
            return {
                'message': "Sorry, I couldn't add that item. Please try again.",
                'type': 'error',
                'suggestions': ["Try Again", "Start Over"]
            }
    
    def parse_item_request(self, user_input: str, service_type: str) -> List[Dict]:
        """Parse user input to extract items and quantities using NLP"""
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
    
    def handle_services_inquiry(self) -> Dict:
        """Handle services inquiry"""
        message = f"🧼 **What Services Do You Offer?**\n\nAt ValetKleen, we're proud to offer a range of convenient and high-quality services to make your life easier. Our main services include:\n\n**1. 🧺 Laundry Services** - We provide full-service wash, dry, and fold services using premium detergents and fabric softeners.\n\n**2. 👔 Dry Cleaning Services** - Our expert team offers professional dry cleaning for suits, dresses, and delicate fabrics.\n\n**3. 🚛 Pickup and Delivery** - Our convenient door-to-door service allows you to schedule pickups and deliveries at a time that fits your busy schedule."
        
        return {
            'message': message,
            'type': 'information',
            'suggestions': [
                "Place an Order",
                "View Dry Cleaning Prices",
                "View Laundry Prices",
                "How Does Pickup Work?"
            ]
        }
    
    def handle_pricing_inquiry(self) -> Dict:
        """Handle pricing inquiry"""
        message = "💰 **Our Pricing:**\n\n🔹 **Transparent flat-rate pricing**\n🔹 **No hidden fees**\n🔹 **Delivery included**\n🔹 **First-time customer discounts**\n\n"
        
        # Add sample prices
        message += "**Sample Dry Cleaning Prices:**\n"
        message += "• Office Shirt - $5.50\n"
        message += "• Pants - $7.50\n"
        message += "• Standard Dress - $12.00\n"
        message += "• Short Coat - $12.00\n\n"
        
        message += "**Laundry Bag Packages:**\n"
        message += "• Small Bag (12 lb) - $22.00\n"
        message += "• Medium Bag (18 lb) - $33.00\n"
        message += "• Large Bag (25 lb) - $46.00\n"
        
        return {
            'message': message,
            'type': 'information',
            'suggestions': [
                "Place an Order",
                "View Full Dry Cleaning Menu",
                "View Full Laundry Menu",
                "About ValetKleen"
            ]
        }
    
    def handle_delivery_inquiry(self) -> Dict:
        """Handle delivery and pickup inquiry"""
        delivery_info = self.knowledge_base.get('process', '')
        
        message = "🚛 **Pickup & Delivery:**\n\n"
        message += "✅ **Convenient door-to-door service**\n"
        message += "✅ **Easy online scheduling**\n"
        message += "✅ **Same-day or next-day options**\n"
        message += "✅ **Timely pickups and fast returns**\n"
        message += "✅ **No need to leave home or office**\n\n"
        
        message += "**How it works:**\n"
        message += "1️⃣ Book with us (online or call)\n"
        message += "2️⃣ We pick up from your location\n"
        message += "3️⃣ Professional cleaning\n"
        message += "4️⃣ We deliver back to you\n"
        
        return {
            'message': message,
            'type': 'information',
            'suggestions': [
                "Place an Order",
                "What Services Do You Offer?",
                "Pricing Information",
                "Contact Information"
            ]
        }
    
    def handle_about_inquiry(self) -> Dict:
        """Handle about company inquiry"""
        about_info = self.knowledge_base.get('about', '')
        
        message = "ℹ️ **About ValetKleen:**\n\n"
        
        if about_info:
            message += f"{about_info[:400]}...\n\n"
        
        message += "🌟 **Why Choose Us:**\n"
        message += "• Professional dry cleaning & laundry\n"
        message += "• Convenient pickup & delivery\n"
        message += "• Eco-friendly cleaning agents\n"
        message += "• Transparent, flat-rate pricing\n"
        message += "• Expert stain removal\n"
        message += "• Same-day/next-day service\n"
        message += "• 100% satisfaction guarantee\n"
        
        return {
            'message': message,
            'type': 'information',
            'suggestions': [
                "Place an Order",
                "What Services Do You Offer?",
                "Pickup & Delivery Info",
                "Contact Information"
            ]
        }
    
    def handle_contact_inquiry(self) -> Dict:
        """Handle contact information inquiry"""
        
        message = "📞 **Contact ValetKleen:**\n\n"
        message += "**Phone:** 1-844-750-2444\n"
        message += "**Email:** info@valetkleen.com\n"
        message += "**Website:** valetkleen.com\n\n"
        message += "**Customer Support Hours:**\n"
        message += "Monday - Friday: 8:00 AM - 6:00 PM\n"
        message += "Saturday: 9:00 AM - 4:00 PM\n"
        message += "Sunday: Closed\n\n"
        message += "We're here to help with scheduling, questions, and order tracking!"
        
        return {
            'message': message,
            'type': 'information',
            'suggestions': [
                "Place an Order",
                "Schedule Pickup",
                "What Services Do You Offer?",
                "Pricing Information"
            ]
        }
    
    def handle_checkout(self, session_id: str) -> Dict:
        """Handle checkout process - collect pickup info then create payment"""
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
        
        # Check if pickup scheduling is needed for regular orders
        pickup_info = session.get('pickup_info', {})
        
        # If no pickup info collected yet, start pickup scheduling
        if not pickup_info.get('pickup_date'):
            session['current_step'] = 'collecting_pickup_info'
            session['pickup_info'] = {'collecting': 'pickup_date'}
            self.customer_sessions[session_id] = session
            
            return {
                'message': "📅 **Pickup Scheduling**\n\nGreat! Let's schedule your pickup.\n\n**Please provide your pickup date** (e.g., 2024-01-15, Tomorrow, Monday):",
                'type': 'pickup_scheduling',
                'suggestions': [
                    "Tomorrow",
                    "Monday", 
                    "2024-01-15",
                    "Next Week"
                ]
            }
        
        # Calculate total
        total = sum(item.get('total_price', item.get('price', 0) * item.get('quantity', 1)) for item in cart)
        
        # Create order summary
        order_summary = "🎉 **CHECKOUT SUCCESSFUL!**\n\n"
        order_summary += "📋 **Your Order:**\n"
        
        for item in cart:
            item_name = item.get('name', 'Unknown Item')
            quantity = item.get('quantity', 1)
            price = item.get('total_price', item.get('price', 0) * quantity)
            order_summary += f"• {quantity}x {item_name} - ${price:.2f}\n"
        
        order_summary += f"\n💰 **Total: ${total:.2f}**\n\n"
        order_summary += "✅ **Next Steps:**\n"
        order_summary += "• Your order has been submitted\n"
        order_summary += "• We'll contact you for pickup scheduling\n"
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
    
    def handle_process_inquiry(self) -> Dict:
        """Handle process/how it works inquiry"""
        process_info = self.knowledge_base.get('process', '')
        
        message = "⚙️ **How ValetKleen Works:**\n\n"
        
        message += "**Our Simple 4-Step Process:**\n\n"
        message += "1️⃣ **Book With Us**\n"
        message += "Schedule pickup online or by phone - quick and easy!\n\n"
        
        message += "2️⃣ **We Pick Up**\n"
        message += "We come to you at your convenience - home or office.\n\n"
        
        message += "3️⃣ **We Clean**\n"
        message += "Professional cleaning with eco-friendly methods.\n\n"
        
        message += "4️⃣ **We Deliver**\n"
        message += "Fresh, clean clothes delivered back to your door.\n\n"
        
        message += "✨ **Clean clothes, zero hassle!**"
        
        return {
            'message': message,
            'type': 'information',
            'suggestions': [
                "Place an Order",
                "What Services Do You Offer?",
                "Pricing Information",
                "About ValetKleen"
            ]
        }
    
    def handle_general_inquiry(self, user_input: str) -> Dict:
        """Handle general inquiries using enhanced LLM with website knowledge"""
        try:
            # First try traditional knowledge base search
            processed_input = self.preprocess_text(user_input)
            best_match = ""
            best_score = 0
            
            for content_item in self.knowledge_base.get('all_content', []):
                content = content_item.get('content', '').lower()
                input_words = processed_input.split()
                content_words = content.split()
                
                matches = sum(1 for word in input_words if word in content_words)
                score = matches / len(input_words) if input_words else 0
                
                if score > best_score and score > 0.1:
                    best_score = score
                    best_match = content[:200] + "..." if len(content) > 200 else content
            
            # Use enhanced LLM response for better answers
            context = f"Knowledge base match: {best_match}" if best_match else "No specific match found"
            llm_response = self.enhanced_llm_response(user_input, context)
            
            # If LLM response is too generic, fall back to structured response
            if "rephrase your question" in llm_response.lower():
                message = "I'd be happy to help! Here's what I can assist you with regarding ValetKleen's services:"
            else:
                message = llm_response
            
            return {
                'message': message,
                'type': 'information',
                'suggestions': [
                    "Place an Order",
                    "What Services Do You Offer?",
                    "Pricing Information",
                    "Pickup & Delivery Info",
                    "Contact Information"
                ]
            }
            
        except Exception as e:
            self.logger.error(f"General inquiry error: {e}")
            return {
                'message': "I'd be happy to help with your laundry and dry cleaning needs! What would you like to know?",
                'type': 'information',
                'suggestions': [
                    "Place an Order",
                    "What Services Do You Offer?",
                    "Pricing Information",
                    "Pickup & Delivery Info",
                    "Contact Information"
                ]
            }
    
    def extract_website_content(self) -> Dict[str, str]:
        """Extract content from the website HTML file for enhanced knowledge base"""
        website_content = {}
        
        try:
            # Read the website HTML file
            website_path = os.path.join(os.path.dirname(__file__), 'website.html')
            if os.path.exists(website_path):
                with open(website_path, 'r', encoding='utf-8') as file:
                    html_content = file.read()
                
                # Parse HTML with BeautifulSoup
                soup = BeautifulSoup(html_content, 'html.parser')
                
                # Extract hero section
                hero = soup.find('section', class_='hero')
                if hero:
                    hero_title = hero.find('h1')
                    hero_desc = hero.find('p')
                    website_content['hero'] = {
                        'title': hero_title.text.strip() if hero_title else '',
                        'description': hero_desc.text.strip() if hero_desc else '',
                        'content': f"{hero_title.text.strip() if hero_title else ''} - {hero_desc.text.strip() if hero_desc else ''}"
                    }
                
                # Extract service cards
                services = soup.find_all('div', class_='service-card')
                service_info = []
                for service in services:
                    title = service.find('h3')
                    desc = service.find('p')
                    if title and desc:
                        service_info.append({
                            'title': title.text.strip(),
                            'description': desc.text.strip(),
                            'content': f"{title.text.strip()}: {desc.text.strip()}"
                        })
                
                website_content['services'] = service_info
                
                # Extract navigation items
                nav_items = soup.find_all('a', href=re.compile(r'^#'))
                nav_content = []
                for item in nav_items:
                    if item.text.strip():
                        nav_content.append(item.text.strip())
                
                website_content['navigation'] = nav_content
                
                # Extract meta information
                title_tag = soup.find('title')
                if title_tag:
                    website_content['title'] = title_tag.text.strip()
                
                # Build comprehensive text for LLM context
                all_text = []
                if 'hero' in website_content:
                    all_text.append(website_content['hero']['content'])
                
                for service in website_content.get('services', []):
                    all_text.append(service['content'])
                
                website_content['full_context'] = ' '.join(all_text)
                
                self.logger.info(f"Successfully extracted website content: {len(website_content)} sections")
                
            else:
                self.logger.warning("website.html not found, using default content")
                website_content = self.get_default_website_content()
                
        except Exception as e:
            self.logger.error(f"Error extracting website content: {e}")
            website_content = self.get_default_website_content()
        
        return website_content
    
    def get_default_website_content(self) -> Dict[str, str]:
        """Fallback website content if HTML file not available"""
        return {
            'hero': {
                'title': 'Premium Laundry & Dry Cleaning',
                'description': 'Professional cleaning services with convenient pickup and delivery',
                'content': 'Premium Laundry & Dry Cleaning - Professional cleaning services with convenient pickup and delivery'
            },
            'services': [
                {
                    'title': 'Dry Cleaning',
                    'description': 'Professional dry cleaning for suits, dresses, and delicate fabrics with expert care and attention.',
                    'content': 'Dry Cleaning: Professional dry cleaning for suits, dresses, and delicate fabrics with expert care and attention.'
                },
                {
                    'title': 'Laundry Service',
                    'description': 'Full-service wash, dry, and fold with premium detergents and fabric softeners.',
                    'content': 'Laundry Service: Full-service wash, dry, and fold with premium detergents and fabric softeners.'
                },
                {
                    'title': 'Pickup & Delivery',
                    'description': 'Convenient door-to-door service that fits your busy schedule. Same-day pickup available.',
                    'content': 'Pickup & Delivery: Convenient door-to-door service that fits your busy schedule. Same-day pickup available.'
                }
            ],
            'title': 'ValetKleen - Professional Laundry & Dry Cleaning Services',
            'full_context': 'Premium Laundry & Dry Cleaning - Professional cleaning services with convenient pickup and delivery. Dry Cleaning: Professional dry cleaning for suits, dresses, and delicate fabrics with expert care and attention. Laundry Service: Full-service wash, dry, and fold with premium detergents and fabric softeners. Pickup & Delivery: Convenient door-to-door service that fits your busy schedule. Same-day pickup available.'
        }
    
    def enhanced_llm_response(self, user_input: str, context: str = "") -> str:
        """Generate enhanced responses using LLM with website context"""
        try:
            # Build context with website information
            website_context = self.website_knowledge.get('full_context', '')
            
            system_prompt = f"""You are ValetKleen's professional customer service assistant. You help customers with laundry and dry cleaning services.

REAL COMPANY INFORMATION:
- Company: Valetkleen (official spelling)
- Tagline: "We pick up, clean, and deliver right to your door"
- Phone: 1-844-750-2444
- Email: info@valetkleen.com
- Website: valetkleen.com

BUSINESS HOURS:
- Monday-Friday: 7am to 7pm
- Saturday: 9am to 5pm
- 24/7 call service available

SERVICE AREAS:
We serve 20+ cities in Georgia including:
- Atlanta, Alpharetta, Brookhaven, Duluth, Dunwoody, Roswell, Sandy Springs, and more

SERVICES OFFERED:
1. Laundry Services (wash, dry, fold)
2. Dry Cleaning Services (suits, dresses, delicate fabrics)  
3. Pickup and Delivery (door-to-door convenience)

KEY FEATURES:
- 98% On-Time Delivery
- 95% Customer Satisfaction
- 99% Laundry Completion Rate
- 94% Eco-Friendly Practices
- Door-to-door convenience
- Reliable & insured staff
- Transparent pricing
- Flexible scheduling
- Swift delivery
- Hassle-free online booking

WEBSITE CONTEXT:
{website_context}

Provide helpful, professional, and friendly responses using this accurate company information. Always mention relevant services and contact details when appropriate. Keep responses conversational and informative.
"""

            # Additional context if provided
            if context:
                system_prompt += f"\n\nADDITIONAL CONTEXT: {context}"

            completion = self.groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_input}
                ],
                temperature=0.7,
                max_completion_tokens=500,
                top_p=1,
                stream=False
            )
            
            return completion.choices[0].message.content.strip()
            
        except Exception as e:
            self.logger.error(f"LLM generation error: {e}")
            return "I'm here to help with your laundry and dry cleaning needs. Could you please rephrase your question?"

# Flask Web Application
app = Flask(__name__)
app.secret_key = 'valetkleen_chatbot_secret_key_2024'

# Initialize chatbot
chatbot = ValetKleenChatbot()

# HTML Template for the chatbot interface
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
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            background: transparent;
            height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            margin: 0;
            padding: 0;
        }
        
        .chat-container {
            width: 100%;
            height: 100vh;
            background: transparent;
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }
        
        /* Header removed since website handles branding */
        
        .chat-messages {
            flex: 1;
            padding: 15px;
            overflow-y: auto;
            background: transparent;
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
            background: #f8f9fa;
            color: #2c3e50;
            border-bottom-left-radius: 5px;
            border: 1px solid #e9ecef;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        
        .message.user .message-bubble {
            background: linear-gradient(135deg, #007bff, #0056b3);
            color: white;
            border-bottom-right-radius: 5px;
            box-shadow: 0 2px 6px rgba(0,123,255,0.3);
        }
        
        .suggestions {
            margin-top: 10px;
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
        }
        
        .suggestion-btn {
            background: linear-gradient(135deg, #ffffff, #f8f9fa);
            border: 1px solid #dee2e6;
            color: #495057;
            padding: 10px 16px;
            border-radius: 25px;
            cursor: pointer;
            font-size: 13px;
            font-weight: 500;
            transition: all 0.4s cubic-bezier(0.25, 0.8, 0.25, 1);
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
            backdrop-filter: blur(10px);
            position: relative;
            overflow: hidden;
        }
        
        .suggestion-btn::before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(255,255,255,0.6), transparent);
            transition: left 0.6s;
            pointer-events: none;
        }
        
        .suggestion-btn:hover::before {
            left: 100%;
        }
        
        .suggestion-btn:hover {
            background: linear-gradient(135deg, #007bff, #0056b3);
            color: white;
            transform: translateY(-2px) scale(1.02);
            box-shadow: 0 8px 25px rgba(0,123,255,0.3);
            border-color: transparent;
        }
        
        .chat-input {
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(20px);
            padding: 20px;
            border-top: 1px solid rgba(233, 236, 239, 0.3);
            display: flex;
            gap: 12px;
            position: relative;
        }
        
        .chat-input input {
            flex: 1;
            padding: 14px 20px;
            border: 2px solid #e9ecef;
            border-radius: 30px;
            outline: none;
            font-size: 15px;
            font-family: inherit;
            transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
            background: rgba(255, 255, 255, 0.9);
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        }
        
        .chat-input input:focus {
            border-color: #007bff;
            box-shadow: 0 0 0 3px rgba(0,123,255,0.1);
            background: white;
        }
        
        .send-btn {
            background: linear-gradient(135deg, #007bff, #0056b3);
            color: white;
            border: none;
            padding: 14px 24px;
            border-radius: 30px;
            cursor: pointer;
            font-size: 14px;
            font-weight: 600;
            transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
            box-shadow: 0 4px 15px rgba(0,123,255,0.3);
            position: relative;
            overflow: hidden;
        }
        
        .send-btn::before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
            transition: left 0.5s;
            pointer-events: none;
        }
        
        .send-btn:hover::before {
            left: 100%;
        }
        
        .send-btn:hover {
            background: linear-gradient(135deg, #0056b3, #004085);
            transform: translateY(-1px);
            box-shadow: 0 6px 20px rgba(0,123,255,0.4);
        }
        
        .send-btn:active {
            transform: translateY(0);
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
        
        .bot-message {
            background: #f1f3f4;
            padding: 12px 16px;
            border-radius: 18px 18px 18px 4px;
            margin-bottom: 16px;
            max-width: 85%;
            animation: messageSlideIn 0.3s ease;
            font-size: 15px;
            line-height: 1.4;
            color: #333;
        }
        
        .bot-avatar {
            width: 32px;
            height: 32px;
            background: linear-gradient(135deg, #4CAF50, #45a049);
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-size: 16px;
            margin-bottom: 8px;
            animation: avatarBounce 0.5s ease;
        }
        
        .suggestions-container {
            margin-top: 12px;
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
        }
        
        .suggestion-pill {
            background: linear-gradient(135deg, #f8f9fa, #ffffff);
            border: 1px solid #e9ecef;
            border-radius: 30px;
            padding: 12px 18px;
            cursor: pointer;
            font-size: 14px;
            font-weight: 600;
            color: #495057;
            transition: all 0.5s cubic-bezier(0.175, 0.885, 0.32, 1.275);
            animation: suggestionFadeIn 0.6s ease both;
            display: inline-flex;
            align-items: center;
            gap: 8px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.08);
            backdrop-filter: blur(20px);
            position: relative;
            overflow: hidden;
            letter-spacing: 0.3px;
            margin: 6px;
        }
        
        .suggestion-pill::before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(0,123,255,0.1), transparent);
            transition: left 0.8s ease;
            pointer-events: none;
        }
        
        .suggestion-pill:hover::before {
            left: 100%;
        }
        
        .suggestion-pill:hover {
            background: linear-gradient(135deg, #007bff, #0056b3);
            color: white;
            border-color: #007bff;
            transform: translateY(-3px) scale(1.05);
            box-shadow: 0 12px 30px rgba(0,123,255,0.4);
        }
        
        .suggestion-pill:active {
            transform: translateY(-1px) scale(1.02);
            transition: all 0.1s ease;
        }
        
        @keyframes messageSlideIn {
            from {
                opacity: 0;
                transform: translateY(10px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        @keyframes avatarBounce {
            0%, 20%, 50%, 80%, 100% {
                transform: translateY(0);
            }
            40% {
                transform: translateY(-4px);
            }
            60% {
                transform: translateY(-2px);
            }
        }
        
        @keyframes suggestionFadeIn {
            from {
                opacity: 0;
                transform: scale(0.8);
            }
            to {
                opacity: 1;
                transform: scale(1);
            }
        }
    </style>
</head>
<body>
    <div class="chat-container">
        <div class="chat-messages" id="chatMessages">
            <div class="welcome-screen" id="welcomeScreen">
                <div class="bot-avatar">🤖</div>
                <div class="bot-message" id="welcomeMessage">
                    Hi there! 👋 How can I help you today?
                </div>
                
                <div class="suggestions-container" id="suggestionsContainer">
                    <div class="suggestion-pill" data-message="Place an Order" style="animation-delay: 0.1s; cursor: pointer;">
                        📋 Place an Order
                    </div>
                    <div class="suggestion-pill" data-message="What services do you offer?" style="animation-delay: 0.2s; cursor: pointer;">
                        🧼 Our Services
                    </div>
                    <div class="suggestion-pill" data-message="Pricing information" style="animation-delay: 0.3s; cursor: pointer;">
                        💰 Pricing
                    </div>
                    <div class="suggestion-pill" data-message="How does pickup work?" style="animation-delay: 0.4s; cursor: pointer;">
                        🚛 Pickup & Delivery
                    </div>
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
        // Generate or retrieve a unique session ID for this user
        function getOrCreateSessionId() {
            let storedId = localStorage.getItem('valetkleen_session_id');
            if (!storedId) {
                // Generate a new UUID for this user
                storedId = 'sess_' + Math.random().toString(36).substr(2, 9) + '_' + Date.now();
                localStorage.setItem('valetkleen_session_id', storedId);
            }
            return storedId;
        }
        
        // Initialize session on page load
        let sessionId = getOrCreateSessionId();
        let isFirstMessage = true;
        
        console.log('Session initialized:', sessionId);

        function handleKeyPress(event) {
            if (event.key === 'Enter') {
                sendMessage();
            }
        }


        function sendMessage(text = null) {
            console.log('sendMessage called with:', text);
            const input = document.getElementById('messageInput');
            const message = text || input.value.trim();
            console.log('Final message:', message);
            
            if (!message) {
                console.log('No message, returning');
                return;
            }
            
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
                
                // Update session ID if server provides a different one
                if (data.session_id && data.session_id !== sessionId) {
                    sessionId = data.session_id;
                    localStorage.setItem('valetkleen_session_id', sessionId);
                    console.log('Session updated:', sessionId);
                }
                
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
                .replace(/\\*\\*(.*?)\\*\\*/g, '<strong>$1</strong>')
                .replace(/\\[([^\\]]+)\\]\\(([^\\)]+)\\)/g, '<a href="$2" target="_blank" rel="noopener noreferrer" style="color: #007bff; text-decoration: underline;">$1</a>')
                .replace(/\\n/g, '<br>');
            
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
        
        // Function to clear session and start fresh (useful for testing)
        function clearSession() {
            if (confirm('This will clear your cart and start a new session. Continue?')) {
                localStorage.removeItem('valetkleen_session_id');
                sessionId = getOrCreateSessionId();
                location.reload();
            }
        }

        // Add click event listeners to suggestion pills
        document.querySelectorAll('.suggestion-pill').forEach(pill => {
            pill.addEventListener('click', function(e) {
                e.preventDefault();
                e.stopPropagation();
                console.log('Pill clicked:', this.textContent);
                const message = this.getAttribute('data-message') || this.textContent.trim();
                console.log('Sending message:', message);
                sendMessage(message);
            });
        });
        
        // Focus input on load
        document.getElementById('messageInput').focus();
        
        // Test function
        window.testClick = function() {
            console.log('Test function works!');
            sendMessage('Hello test');
        };
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

@app.route('/webhook/stripe', methods=['POST'])
def stripe_webhook():
    """Handle Stripe payment webhook notifications"""
    try:
        # Get the raw payload
        payload = request.get_data()
        sig_header = request.headers.get('stripe-signature')
        
        # For now, we'll process all payment success events
        # In production, you should verify the webhook signature
        event_data = request.get_json()
        
        if event_data and event_data.get('type') == 'payment_intent.succeeded':
            # Handle direct payment intent success (logistics service)
            payment_intent = event_data.get('data', {}).get('object', {})
            amount = payment_intent.get('amount', 0) / 100  # Convert from cents
            payment_id = payment_intent.get('id', '')
            
            # Log the successful payment
            chatbot.logger.info(f"Stripe payment successful: {payment_id}, Amount: ${amount}")
            
            # For logistics service (amount = $20.00), find and send email
            if amount == 20.00:
                # In a real app, you'd match this to a pending order
                # For now, we'll send a general payment confirmation email
                payment_info = {
                    'payment_id': payment_id,
                    'amount': amount,
                    'status': 'succeeded',
                    'timestamp': datetime.now().isoformat()
                }
                
                # Create dummy order data for email (in real app, fetch from database)
                order_data = {
                    'order_id': f"STRIPE_{payment_id[:8]}",
                    'service_type': 'logistics',
                    'customer_info': {
                        'full_name': 'Payment Confirmed',
                        'email': 'customer@email.com',
                        'cell_phone': 'Contact customer for details',
                        'home_phone': 'N/A',
                        'home_address': 'Address from Stripe metadata',
                        'zip_code': 'N/A',
                        'pickup_date': 'To be scheduled',
                        'pickup_time': 'To be scheduled',
                        'mart_name': 'Customer preferred location',
                        'mart_address': 'To be confirmed',
                        'mart_phone': 'To be confirmed'
                    },
                    'cost': amount,
                    'status': 'paid',
                    'timestamp': datetime.now().isoformat()
                }
                
                # Send payment confirmation email
                try:
                    chatbot.email_service.send_order_notification(order_data, payment_info)
                    chatbot.logger.info(f"Payment confirmation email sent for {payment_id}")
                except Exception as e:
                    chatbot.logger.error(f"Failed to send payment confirmation email: {e}")
        
        elif event_data and event_data.get('type') == 'checkout.session.completed':
            # Handle Stripe Checkout session completion (regular orders)
            checkout_session = event_data.get('data', {}).get('object', {})
            session_id = checkout_session.get('id', '')
            amount_total = checkout_session.get('amount_total', 0) / 100  # Convert from cents
            
            # Get metadata from the checkout session
            metadata = checkout_session.get('metadata', {})
            order_id = metadata.get('order_id', '')
            customer_email = metadata.get('customer_email', '')
            customer_name = metadata.get('customer_name', '')
            order_type = metadata.get('order_type', 'regular_order')
            
            # Log the successful checkout
            chatbot.logger.info(f"Stripe checkout completed: {session_id}, Order: {order_id}, Amount: ${amount_total}")
            
            # Create comprehensive order data from metadata and checkout session
            order_data = {
                'order_id': order_id,
                'service_type': order_type,
                'customer_info': {
                    'full_name': customer_name,
                    'email': customer_email,
                    'cell_phone': 'From customer profile',
                    'home_phone': 'N/A',
                    'home_address': 'From customer profile',
                    'zip_code': 'N/A',
                    'pickup_date': metadata.get('pickup_date', 'TBD'),
                    'pickup_time': metadata.get('pickup_time', 'TBD'),
                    'delivery_date': metadata.get('delivery_date', 'TBD'),
                    'delivery_time': metadata.get('delivery_time', 'TBD')
                },
                'cost': amount_total,
                'status': 'paid',
                'timestamp': datetime.now().isoformat(),
                'items': f"Order details for {order_id}"  # In real app, fetch from session
            }
            
            # Create payment info
            payment_info = {
                'payment_id': session_id,
                'amount': amount_total,
                'status': 'succeeded',
                'timestamp': datetime.now().isoformat(),
                'payment_method': 'Stripe Checkout'
            }
            
            # Send order confirmation email
            try:
                chatbot.email_service.send_order_notification(order_data, payment_info)
                chatbot.logger.info(f"Order confirmation email sent for {order_id}")
            except Exception as e:
                chatbot.logger.error(f"Failed to send order confirmation email: {e}")
        
        return jsonify({'status': 'success'}), 200
        
    except Exception as e:
        chatbot.logger.error(f"Stripe webhook error: {e}")
        return jsonify({'error': 'webhook processing failed'}), 400

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'ValetKleen Chatbot',
        'version': '1.0.0',
        'email_service': 'configured',
        'stripe_webhook': '/webhook/stripe'
    })

if __name__ == '__main__':
    print("🚀 Starting ValetKleen Professional Chatbot...")
    print("=" * 50)
    print("🤖 Chatbot Features:")
    print("   ✅ Advanced NLP with intent detection")
    print("   ✅ Complete order management system")
    print("   ✅ Customer information collection")
    print("   ✅ Interactive service catalogs")
    print("   ✅ Smart cart functionality")
    print("   ✅ Professional web interface")
    print("   ✅ RESTful APIs for integration")
    print()
    print("📊 Loaded Knowledge Base:")
    print(f"   ✅ {len(chatbot.knowledge_base.get('all_content', []))} content items")
    print(f"   ✅ {len(chatbot.service_catalog)} service categories")
    print(f"   ✅ {sum(len(cat['items']) for cat in chatbot.service_catalog.values())} total items")
    print()
    print("🌐 Access your chatbot at: http://localhost:5000")
    print("🔗 API Health Check: http://localhost:5000/api/health")
    print("=" * 50)
    
    # Start the Flask application
    app.run(host='0.0.0.0', port=5000, debug=False)