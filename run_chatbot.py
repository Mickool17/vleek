"""
Quick launcher for the ValetKleen Chatbot
"""

import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import app at module level for gunicorn
from valetkleen_chatbot_v2 import app, chatbot

if __name__ == "__main__":
    print("LAUNCHING ValetKleen Professional Chatbot...")
    print("-" * 50)
    
    try:
        
        print("SUCCESS: Chatbot system loaded successfully!")
        print(f"Knowledge base: {len(chatbot.knowledge_base.get('all_content', []))} items")
        print(f"Service catalog: {sum(len(cat['items']) for cat in chatbot.service_catalog.values())} items")
        print()
        print("Your chatbot will be available at:")
        print("   > http://localhost:5000")
        print("   > http://127.0.0.1:5000")
        print()
        print("Starting web server...")
        print("-" * 50)
        
        # Run the application
        app.run(host='0.0.0.0', port=5000, debug=False)
        
    except ImportError as e:
        print(f"❌ Error importing modules: {e}")
        print("💡 Please make sure all packages are installed:")
        print("   pip install flask nltk scikit-learn python-dotenv")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error starting chatbot: {e}")
        sys.exit(1)