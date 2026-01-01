from flask import Flask, render_template, request, jsonify
import os
import uuid
from PIL import Image
try:
    from ml_models import (
        GenderCNN, 
        VoiceSentimentModel, 
        VoiceQAModel, 
        TextGenModel, 
        TranslationModel
    )
except ImportError as e:
    print(f"IMPORT ERROR: {e}")
    # Fallback to empty mocks if file is missing (during setup)
    class TranslationModel: 
        def translate(self, t): return "Import Error"

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize models
gender_engine = GenderCNN()
sentiment_engine = VoiceSentimentModel()
qa_engine = VoiceQAModel()
gen_engine = TextGenModel()
trans_engine = TranslationModel()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/image-classification')
def image_classification_page():
    return render_template('image_classification.html')

@app.route('/sentiment-analysis')
def sentiment_analysis_page():
    return render_template('sentiment_analysis.html')

@app.route('/qa')
def qa_page():
    return render_template('qa.html')

@app.route('/text-generation')
def text_generation_page():
    return render_template('text_generation.html')

@app.route('/translation')
def translation_page():
    return render_template('translation.html')

# --- API Endpoints ---

@app.route('/api/translate', methods=['POST'])
def translate_api():
    print("Translation Request Received")
    try:
        data = request.get_json(silent=True) or {}
        text = data.get('text', '')
        print(f"Input Text: {text}")
        
        if not text:
            return jsonify({'translated_text': 'Please enter some text.'})

        urdu = trans_engine.translate(text)
        print(f"Output Urdu: {urdu}")
        return jsonify({'translated_text': urdu})
    except Exception as e:
        print(f"Error in /api/translate: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/classify-image', methods=['POST'])
def classify_image():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file'}), 400
        file = request.files['file']
        img = Image.open(file.stream).convert('RGB')
        results = gender_engine.process(img)
        return jsonify({
            'label': results[0]['label'],
            'score': results[0]['score']
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/analyze-sentiment', methods=['POST'])
def analyze_sentiment():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file'}), 400
        file = request.files['file']
        audio_path = os.path.join(app.config['UPLOAD_FOLDER'], f"temp_s_{uuid.uuid4()}.wav")
        file.save(audio_path)
        text, sentiment = sentiment_engine.process(audio_path)
        if os.path.exists(audio_path):
            os.remove(audio_path)
        return jsonify({'transcription': text, 'sentiment': sentiment})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/voice-qa', methods=['POST'])
def voice_qa():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file'}), 400
        file = request.files['file']
        audio_path = os.path.join(app.config['UPLOAD_FOLDER'], f"temp_q_{uuid.uuid4()}.wav")
        file.save(audio_path)
        q, a, audio_file = qa_engine.ask(audio_path, app.config['UPLOAD_FOLDER'])
        if os.path.exists(audio_path):
            os.remove(audio_path)
        return jsonify({'question': q, 'answer': a, 'audio_file': audio_file})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/generate-text', methods=['POST'])
def generate_text():
    try:
        data = request.get_json(silent=True) or {}
        prompt = data.get('prompt', '')
        text = gen_engine.generate(prompt)
        return jsonify({'generated_text': text})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
