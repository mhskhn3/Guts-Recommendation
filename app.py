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
print("STEP 1: Downloading NLTK resources...")
print("="*50)
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    print("✅ NLTK resources downloaded!")
except Exception as e:
    print(f"❌ NLTK download error: {e}")

app = Flask(__name__)

# Global recommender object
recommender = None

def download_from_drive():
    """Download model from Google Drive"""
    print("="*50)
    print("STEP 2: Checking for model file from Google Drive...")
    print("="*50)
    
    # Your Google Drive file ID
    MODEL_FILE_ID = "10dtSRjjsE-42sLFDnfOj8WKLu99YuZ-A"
    
    try:
        # Download model file if not exists
        if not os.path.exists("ticket_recommender_model.joblib"):
            print(f"📥 Downloading 300MB model from Google Drive...")
            print(f"File ID: 10dtSRjjsE-42sLFDnfOj8WKLu99YuZ-A")
            model_url = f"https://drive.google.com/uc?id=10dtSRjjsE-42sLFDnfOj8WKLu99YuZ-A"
            
            # Try downloading with gdown
            try:
                gdown.download(model_url, "ticket_recommender_model.joblib", quiet=False)
            except Exception as e:
                print(f"❌ gdown download failed: {e}")
                # Try alternative download method
                import requests
                print("Trying alternative download method...")
                response = requests.get(model_url, stream=True)
                with open("ticket_recommender_model.joblib", "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                print("Alternative download complete")
            
            if os.path.exists("ticket_recommender_model.joblib"):
                file_size = os.path.getsize("ticket_recommender_model.joblib") / (1024*1024)
                print(f"✅ Model downloaded! Size: {file_size:.2f} MB")
                return True
            else:
                print("❌ Model download failed - file not found after download")
                # List all files in current directory
                print(f"Files in directory: {os.listdir('.')}")
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
        print("STEP 3: Loading recommendation system...")
        print("="*50)
        
        # Check if model file exists
        if not os.path.exists("ticket_recommender_model.joblib"):
            print("❌ Model file not found!")
            print(f"Current directory: {os.getcwd()}")
            print(f"Files: {os.listdir('.')}")
            return False
        
        file_size = os.path.getsize("ticket_recommender_model.joblib") / (1024*1024)
        print(f"📁 Model file found! Size: {file_size:.2f} MB")
        
        # Try to load the model with detailed error catching
        print("🔄 Initializing TicketRecommendationSystem...")
        recommender = TicketRecommendationSystem()
        
        print("🔄 Attempting to load model...")
        try:
            recommender.load_model("ticket_recommender_model.joblib")
        except Exception as e:
            print(f"❌ Error in recommender.load_model: {e}")
            traceback.print_exc()
            return False
        
        # Verify model loaded correctly
        if recommender.ticket_ids is not None:
            print(f"✅ Model loaded successfully!")
            print(f"✅ Number of tickets in model: {len(recommender.ticket_ids)}")
            return True
        else:
            print("❌ Model loaded but ticket_ids is None")
            return False
        
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
        print("❌ API called but recommender is None")
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
        "model_file_size": f"{os.path.getsize('ticket_recommender_model.joblib') / (1024*1024):.2f} MB" if os.path.exists("ticket_recommender_model.joblib") else "N/A",
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
        "model_file_size": f"{os.path.getsize('ticket_recommender_model.joblib') / (1024*1024):.2f} MB" if os.path.exists("ticket_recommender_model.joblib") else "N/A",
    })

if __name__ == '__main__':
    print("="*50)
    print("🚀 Starting Ticket Recommendation System")
    print("="*50)
    print(f"Python version: {sys.version}")
    print(f"Current directory: {os.getcwd()}")
    print(f"Files in directory: {os.listdir('.')}")
    
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

