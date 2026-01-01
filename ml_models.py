import random
import time
import os
import uuid
from gtts import gTTS
import mtranslate  # Lightweight real translation library

# =================================================================
# 1. IMAGE CLASSIFICATION: Custom CNN Simulation
# =================================================================
class GenderCNN:
    def process(self, image_input):
        time.sleep(1.5) 
        label = random.choice(["Male", "Female"])
        score = random.uniform(0.85, 0.98)
        return [{"label": label, "score": score}]

# =================================================================
# 2. SENTIMENT ANALYSIS: Voice to Text -> Sentiment Simulation
# =================================================================
class VoiceSentimentModel:
    def process(self, audio_path):
        time.sleep(1.5) 
        sample_data = [
            ("I really enjoy using this AI workspace, it is amazing!", "Positive"),
            ("The project is quite difficult and frustrating at times.", "Negative"),
            ("I am currently testing the voice recognition features.", "Neutral"),
            ("Today is a beautiful day for coding.", "Positive")
        ]
        transcription, sentiment = random.choice(sample_data)
        return transcription, sentiment

# =================================================================
# 3. QUESTION ANSWERING: Voice to Voice Simulation
# =================================================================
class VoiceQAModel:
    def ask(self, audio_path, upload_folder):
        time.sleep(2.0) 
        qa_pairs = [
            ("What is Machine Learning?", "Machine learning is a field of artificial intelligence focusing on data-driven systems."),
            ("What are Transformers?", "Transformers are neural network architectures using self-attention mechanisms."),
            ("Is AI helpful?", "Yes, AI can automate complex tasks and assist in creative processes.")
        ]
        question, answer = random.choice(qa_pairs)
        
        output_filename = f"qans_{uuid.uuid4()}.mp3"
        path = os.path.join(upload_folder, output_filename)
        try:
            tts = gTTS(text=answer, lang='en')
            tts.save(path)
        except Exception as e:
            print(f"TTS Error: {e}")
            
        return question, answer, output_filename

# =================================================================
# 4. TEXT GENERATION: Text -> Text Simulation
# =================================================================
class TextGenModel:
    def generate(self, prompt):
        time.sleep(1.2)
        continuations = [
            " This technology is evolving rapidly and changing how we interact with machines.",
            " Experiments show that using larger datasets often leads to better performance.",
            " The integration of AI in web applications provides a seamless user experience."
        ]
        return prompt + random.choice(continuations)

# =================================================================
# 5. TRANSLATION: English to Urdu (REAL LIGHTWEIGHT TRANSLATION)
# =================================================================
class TranslationModel:
    def translate(self, text):
        """Uses mtranslate for real translation without large downloads."""
        try:
            # We add a small artificial delay to simulate 'thinking'
            time.sleep(1)
            # translate(text, to_language, from_language)
            urdu_text = mtranslate.translate(text, "ur", "en")
            return urdu_text
        except Exception as e:
            print(f"Translation Error: {e}")
            return f"Error in translation: {str(e)}"
