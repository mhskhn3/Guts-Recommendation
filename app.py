import os
import gdown
import joblib
import sys
import traceback
from flask import Flask, render_template, request, jsonify
from model import TicketRecommendationSystem
import nltk

# Download NLTK data
print("="*50)
print("Downloading NLTK resources...")
print("="*50)
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
print("✅ NLTK resources downloaded!")

app = Flask(__name__)

# Global recommender object
recommender = None

def download_from_drive():
    """Download model from Google Drive"""
    print("="*50)
    print("Checking for model file from Google Drive...")
    print("="*50)
    
    # Your Google Drive file ID (from your code)
    MODEL_FILE_ID = "10dtSRjjsE-42sLFDnfOj8WKLu99YuZ-A"
    
    try:
        # Download model file if not exists
        if not os.path.exists("ticket_recommender_model.joblib"):
            print(f"Downloading 300MB model from Google Drive...")
            print(f"File ID: 10dtSRjjsE-42sLFDnfOj8WKLu99YuZ-A")
            model_url = f"https://drive.google.com/uc?id=10dtSRjjsE-42sLFDnfOj8WKLu99YuZ-A"
            gdown.download(model_url, "ticket_recommender_model.joblib", quiet=False)
            
            if os.path.exists("ticket_recommender_model.joblib"):
                file_size = os.path.getsize("ticket_recommender_model.joblib") / (1024*1024)
                print(f"✅ Model downloaded! Size: {file_size:.2f} MB")
                return True
            else:
                print("❌ Model download failed - file not found after download")
                return False
        else:
            file_size = os.path.getsize("ticket_recommender_model.joblib") / (1024*1024)
            print(f"✅ Model already exists locally (Size: {file_size:.2f} MB)")
            return True
            
    except Exception as e:
        print(f"❌ Download error: {str(e)}")
        traceback.print_exc()
        return False

def load_recommender():
    """Load the recommendation model"""
    global recommender
    try:
        print("="*50)
        print("Loading recommendation system...")
        print("="*50)
        
        if not os.path.exists("ticket_recommender_model.joblib"):
            print("❌ Model file not found!")
            return False
        
        file_size = os.path.getsize("ticket_recommender_model.joblib") / (1024*1024)
        print(f"Model file size: {file_size:.2f} MB")
        
        # Initialize and load model
        recommender = TicketRecommendationSystem()
        recommender.load_model("ticket_recommender_model.joblib")
        
        print("✅ Model loaded successfully!")
        print(f"✅ Number of tickets in model: {len(recommender.ticket_ids)}")
        return True
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        traceback.print_exc()
        return False

@app.route('/')
def home():
    """Serve the HTML frontend"""
    try:
        return render_template('index.html')
    except Exception as e:
        print(f"Error serving index.html: {e}")
        traceback.print_exc()
        return "Error loading page. Please check that index.html is in the templates folder.", 500

@app.route('/api/find_similar_tickets', methods=['POST'])
def find_similar_tickets():
    """API endpoint to find similar tickets"""
    global recommender
    
    if recommender is None:
        return jsonify({"error": "Recommendation system not loaded. Please check server logs."}), 500
    
    try:
        data = request.get_json()
        
        if not data or 'description' not in data:
            return jsonify({"error": "No description provided"}), 400
        
        description = data['description'].strip()
        
        if not description:
            return jsonify({"error": "Empty description"}), 400
        
        if len(description) < 10:
            return jsonify({"error": "Description too short (minimum 10 characters)"}), 400
        
        print(f"🔍 Processing query: {description[:50]}...")
        
        # Find similar tickets
        results = recommender.find_similar_tickets(
            query_description=description,
            top_n=10,
            similarity_threshold=0.5
        )
        
        # Check if results is a dict with error
        if isinstance(results, dict) and 'error' in results:
            print(f"⚠️ Error from model: {results['error']}")
            return jsonify([])
        
        print(f"✅ Found {len(results)} similar tickets")
        return jsonify(results)
    
    except Exception as e:
        print(f"❌ Error processing request: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy" if recommender is not None else "degraded",
        "model_loaded": recommender is not None,
        "model_file_exists": os.path.exists("ticket_recommender_model.joblib"),
        "templates_folder_exists": os.path.exists("templates"),
        "index_html_exists": os.path.exists("templates/index.html") if os.path.exists("templates") else False
    })

@app.route('/debug', methods=['GET'])
def debug_info():
    """Debug endpoint to check file structure"""
    files = os.listdir(".")
    templates_files = os.listdir("templates") if os.path.exists("templates") else []
    
    return jsonify({
        "current_directory": os.getcwd(),
        "files_in_root": files,
        "templates_folder_exists": os.path.exists("templates"),
        "files_in_templates": templates_files if os.path.exists("templates") else [],
        "model_file_exists": os.path.exists("ticket_recommender_model.joblib"),
    })

if __name__ == '__main__':
    print("="*50)
    print("🚀 Starting Ticket Recommendation System")
    print("="*50)
    
    # Check if templates folder exists
    if not os.path.exists("templates"):
        print("⚠️  Warning: 'templates' folder not found!")
    elif not os.path.exists("templates/index.html"):
        print("⚠️  Warning: 'templates/index.html' not found!")
    else:
        print("✅ Templates folder and index.html found")
    
    # Download model from Google Drive
    if download_from_drive():
        print("✅ File download completed")
    else:
        print("⚠️  File download had issues - continuing anyway...")
    
    # Load the recommender
    if load_recommender():
        port = int(os.environ.get('PORT', 5000))
        print(f"✅ Server starting on port {port}")
        print(f"🌐 Access your app at: http://localhost:{port} or your Render URL")
        app.run(host='0.0.0.0', port=port, debug=False)
    else:
        print("❌ Failed to load recommender. Exiting.")
        exit(1)
