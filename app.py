# app.py - With your specific file IDs
import os
import gdown
import joblib
import sys
import traceback
from flask import Flask, render_template, request, jsonify
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
    """Download model and CSV from Google Drive"""
    print("="*50)
    print("STEP 2: Downloading files from Google Drive")
    print("="*50)
    
    # Your specific file IDs (hardcoded for simplicity)
    MODEL_FILE_ID = "10dtSRjjsE-42sLFDnfOj8WKLu99YuZ-A"
    CSV_FILE_ID = "1JsSCqQomfIelhSfO9eHlbyDGbuRqfzMX"
    
    success = True
    
    # Download model file
    if not os.path.exists("ticket_recommender_model.joblib"):
        print("📥 Downloading model file...")
        model_url = f"https://drive.google.com/uc?id={MODEL_FILE_ID}"
        try:
            gdown.download(model_url, "ticket_recommender_model.joblib", quiet=False)
            if os.path.exists("ticket_recommender_model.joblib"):
                file_size = os.path.getsize("ticket_recommender_model.joblib") / (1024*1024)
                print(f"✅ Model downloaded! Size: {file_size:.2f} MB")
            else:
                print("❌ Model download failed")
                success = False
        except Exception as e:
            print(f"❌ Model download error: {e}")
            success = False
    else:
        file_size = os.path.getsize("ticket_recommender_model.joblib") / (1024*1024)
        print(f"✅ Model already exists (Size: {file_size:.2f} MB)")
    
    # Download CSV file
    if not os.path.exists("merged_data.csv"):
        print("📥 Downloading CSV file...")
        csv_url = f"https://drive.google.com/uc?id={CSV_FILE_ID}"
        try:
            gdown.download(csv_url, "merged_data.csv", quiet=False)
            if os.path.exists("merged_data.csv"):
                file_size = os.path.getsize("merged_data.csv") / (1024*1024)
                print(f"✅ CSV downloaded! Size: {file_size:.2f} MB")
            else:
                print("❌ CSV download failed")
                success = False
        except Exception as e:
            print(f"❌ CSV download error: {e}")
            success = False
    else:
        file_size = os.path.getsize("merged_data.csv") / (1024*1024)
        print(f"✅ CSV already exists (Size: {file_size:.2f} MB)")
    
    return success

def check_and_prepare_model():
    """Check if model exists, if not prepare to train"""
    print("="*50)
    print("STEP 3: Checking model status")
    print("="*50)
    
    if os.path.exists("ticket_recommender_model.joblib"):
        file_size = os.path.getsize("ticket_recommender_model.joblib") / (1024*1024)
        print(f"✅ Model file found! Size: {file_size:.2f} MB")
        
        # Try to load it to verify it's not corrupted
        try:
            test_load = joblib.load("ticket_recommender_model.joblib")
            print("✅ Model file is valid and can be loaded")
            return "load"
        except Exception as e:
            print(f"⚠️ Existing model file is corrupted: {e}")
            print("Will train new model instead")
            os.remove("ticket_recommender_model.joblib")
            return "train"
    else:
        print("ℹ️ No existing model file found")
        if os.path.exists("merged_data.csv"):
            print("✅ CSV file found - will train new model")
            return "train"
        else:
            print("❌ No CSV file found either!")
            return "failed"

def train_model():
    """Train the model using CSV"""
    print("="*50)
    print("STEP 4: Training new model")
    print("="*50)
    
    try:
        from train_on_render import train_model_on_render
        if train_model_on_render():
            print("✅ Model training successful!")
            return True
        else:
            print("❌ Model training failed")
            return False
    except Exception as e:
        print(f"❌ Error during training: {e}")
        traceback.print_exc()
        return False

def load_recommender():
    """Load the recommendation model"""
    global recommender
    try:
        print("="*50)
        print("STEP 5: Loading recommendation system")
        print("="*50)
        
        from model import TicketRecommendationSystem
        
        if not os.path.exists("ticket_recommender_model.joblib"):
            print("❌ Model file not found!")
            return False
        
        file_size = os.path.getsize("ticket_recommender_model.joblib") / (1024*1024)
        print(f"📁 Loading model file ({file_size:.2f} MB)...")
        
        recommender = TicketRecommendationSystem()
        recommender.load_model("ticket_recommender_model.joblib")
        
        if recommender.ticket_ids is not None:
            print(f"✅ Model loaded successfully!")
            print(f"✅ Number of tickets in model: {len(recommender.ticket_ids)}")
            if len(recommender.ticket_ids) > 0:
                print(f"✅ Sample ticket ID: {recommender.ticket_ids[0]}")
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
        return "Error loading page. Check that index.html is in templates folder.", 500

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
        
        results = recommender.find_similar_tickets(
            query_description=description,
            top_n=10,
            similarity_threshold=0.5
        )
        
        if isinstance(results, dict) and 'error' in results:
            print(f"⚠️ Error from model: {results['error']}")
            return jsonify([])
        
        print(f"✅ Found {len(results)} similar tickets")
        return jsonify(results)
    
    except Exception as e:
        print(f"❌ Error processing request: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/debug', methods=['GET'])
def debug():
    """Debug endpoint to check file status"""
    files = {}
    for f in ['ticket_recommender_model.joblib', 'merged_data.csv']:
        if os.path.exists(f):
            files[f] = f"{os.path.getsize(f) / (1024*1024):.2f} MB"
        else:
            files[f] = "Not found"
    
    return jsonify({
        "files": files,
        "recommender_loaded": recommender is not None,
        "current_directory": os.getcwd(),
        "all_files": os.listdir('.')
    })

if __name__ == '__main__':
    print("="*60)
    print("🚀 Starting Ticket Recommendation System")
    print("="*60)
    
    # Step 1: Download files from Google Drive
    if download_from_drive():
        print("✅ All downloads completed")
        
        # Step 2: Check model status
        status = check_and_prepare_model()
        
        if status == "load":
            # Try to load existing model
            if load_recommender():
                print("✅ Ready to serve requests!")
            else:
                print("⚠️ Could not load model, will train new one")
                if train_model():
                    load_recommender()
        
        elif status == "train":
            # Train new model
            if train_model():
                load_recommender()
            else:
                print("❌ Training failed")
        
        else:
            print("❌ Cannot proceed - no model or CSV")
    
    else:
        print("❌ Failed to download files from Google Drive")
    
    # Always start the server (even if model failed)
    port = int(os.environ.get('PORT', 5000))
    print(f"🌍 Server starting on port {port}")
    print(f"🔍 Debug endpoint: /debug")
    app.run(host='0.0.0.0', port=port, debug=False)
