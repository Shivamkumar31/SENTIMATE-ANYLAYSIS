"""
Sentiment Analysis Utility Functions
Contains NLP preprocessing, model loading, and inference functions
"""

import re
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import Dict, List, Tuple, Union
import json
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============= EMOTION MAPPINGS =============

EMOTION_LABELS = {
    0: "😢 Sadness",
    1: "😨 Fear",
    2: "😠 Anger",
    3: "😊 Joy",
    4: "😐 Neutral",
    5: "😲 Surprise",
    6: "🤢 Disgust"
}

SENTIMENT_LABELS = {
    0: "Negative",
    1: "Neutral",
    2: "Positive"
}

# Color mapping for visualization
SENTIMENT_COLORS = {
    "Negative": "#FF6B6B",
    "Neutral": "#FFA500",
    "Positive": "#4CAF50"
}

EMOTION_COLORS = {
    0: "#4A90E2",   # Sadness - Blue
    1: "#7B68EE",   # Fear - Purple
    2: "#DC143C",   # Anger - Crimson
    3: "#FFD700",   # Joy - Gold
    4: "#808080",   # Neutral - Gray
    5: "#FF69B4",   # Surprise - Pink
    6: "#228B22"    # Disgust - Forest Green
}

# ============= TEXT PREPROCESSING =============

class TextPreprocessor:
    """Preprocess text for NLP models"""
    
    @staticmethod
    def clean_text(text: str) -> str:
        """
        Clean text by removing special characters, extra spaces, etc.
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove mentions and hashtags but keep the text
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'#(\w+)', r'\1', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^a-zA-Z0-9\s\.\!\?\']', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    @staticmethod
    def preprocess(text: str, clean: bool = True) -> str:
        """Preprocess text for model input"""
        if clean:
            text = TextPreprocessor.clean_text(text)
        
        # Remove multiple punctuation
        text = re.sub(r'([.!?])\1+', r'\1', text)
        
        # Normalize repeated characters (remove_repetitions)
        text = re.sub(r'(.)\1{2,}', r'\1\1', text)
        
        return text.strip()

# ============= MODEL LOADING =============

class SentimentModel:
    """Load and use pre-trained sentiment analysis models"""
    
    def __init__(self, model_name: str = "distilbert-base-uncased-finetuned-sst-2-english"):
        """
        Initialize sentiment model
        
        Available models:
        - distilbert-base-uncased-finetuned-sst-2-english (3 sentiments, fast)
        - bert-base-uncased (requires fine-tuning)
        - roberta-base (better accuracy)
        """
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info(f"Loading model: {model_name}")
        logger.info(f"Using device: {self.device}")
        
        try:
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.model = self.model.to(self.device)
            self.model.eval()
            
            logger.info("✅ Model loaded successfully")
        except Exception as e:
            logger.error(f"❌ Error loading model: {e}")
            raise
    
    def predict_sentiment(self, text: str, return_confidence: bool = True) -> Dict:
        """
        Predict sentiment for text
        
        Returns:
            {
                'text': original text,
                'sentiment': 'Positive/Negative/Neutral',
                'confidence': 0.95,
                'scores': {'Negative': 0.05, 'Neutral': 0.05, 'Positive': 0.90}
            }
        """
        # Preprocess text
        processed_text = TextPreprocessor.preprocess(text)
        
        # Tokenize
        inputs = self.tokenizer(
            processed_text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        ).to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
        
        # Get probabilities
        probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
        predicted_class = np.argmax(probs)
        confidence = float(probs[predicted_class])
        
        # Map to labels
        sentiment_label = SENTIMENT_LABELS.get(predicted_class, "Unknown")
        
        # Create scores dictionary
        scores = {
            SENTIMENT_LABELS[i]: float(probs[i]) 
            for i in range(len(probs))
        }
        
        return {
            'text': text,
            'processed_text': processed_text,
            'sentiment': sentiment_label,
            'confidence': round(confidence, 4),
            'scores': scores,
            'color': SENTIMENT_COLORS.get(sentiment_label, "#808080")
        }
    
    def predict_batch(self, texts: List[str]) -> List[Dict]:
        """Predict sentiment for multiple texts"""
        results = []
        for text in texts:
            result = self.predict_sentiment(text)
            results.append(result)
        return results

# ============= EMOTION DETECTION =============

class EmotionModel:
    """Detect emotions in text"""
    
    def __init__(self, model_name: str = "j-hartmann/emotion-english-distilroberta-base"):
        """
        Initialize emotion detection model
        
        Model: Detects 7 emotions - sadness, fear, anger, joy, neutral, surprise, disgust
        """
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info(f"Loading emotion model: {model_name}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.model = self.model.to(self.device)
            self.model.eval()
            
            logger.info("✅ Emotion model loaded successfully")
        except Exception as e:
            logger.error(f"❌ Error loading emotion model: {e}")
            raise
    
    def predict_emotion(self, text: str) -> Dict:
        """
        Predict emotions in text
        
        Returns:
            {
                'text': original text,
                'emotion': 'Joy',
                'confidence': 0.92,
                'all_emotions': {...},
                'color': '#FFD700'
            }
        """
        # Preprocess text
        processed_text = TextPreprocessor.preprocess(text)
        
        # Tokenize
        inputs = self.tokenizer(
            processed_text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        ).to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
        
        # Get probabilities
        probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
        predicted_class = np.argmax(probs)
        confidence = float(probs[predicted_class])
        
        # Map to emotions
        emotion_label = EMOTION_LABELS.get(predicted_class, "Unknown")
        
        # Create scores dictionary
        emotion_scores = {
            EMOTION_LABELS[i]: float(probs[i]) 
            for i in range(len(probs))
        }
        
        return {
            'text': text,
            'processed_text': processed_text,
            'emotion': emotion_label,
            'confidence': round(confidence, 4),
            'all_emotions': emotion_scores,
            'color': EMOTION_COLORS.get(predicted_class, "#808080")
        }
    
    def predict_batch(self, texts: List[str]) -> List[Dict]:
        """Predict emotions for multiple texts"""
        results = []
        for text in texts:
            result = self.predict_emotion(text)
            results.append(result)
        return results

# ============= COMBINED ANALYSIS =============

class SentimentEmotionAnalyzer:
    """Combined sentiment + emotion analysis"""
    
    def __init__(
        self,
        sentiment_model_name: str = "distilbert-base-uncased-finetuned-sst-2-english",
        emotion_model_name: str = "j-hartmann/emotion-english-distilroberta-base"
    ):
        """Initialize both models"""
        self.sentiment_model = SentimentModel(sentiment_model_name)
        self.emotion_model = EmotionModel(emotion_model_name)
    
    def analyze(self, text: str) -> Dict:
        """
        Analyze text for both sentiment and emotion
        
        Returns: Combined analysis result
        """
        sentiment_result = self.sentiment_model.predict_sentiment(text)
        emotion_result = self.emotion_model.predict_emotion(text)
        
        return {
            'text': text,
            'processed_text': sentiment_result['processed_text'],
            'sentiment': {
                'label': sentiment_result['sentiment'],
                'confidence': sentiment_result['confidence'],
                'scores': sentiment_result['scores'],
                'color': sentiment_result['color']
            },
            'emotion': {
                'label': emotion_result['emotion'],
                'confidence': emotion_result['confidence'],
                'all_emotions': emotion_result['all_emotions'],
                'color': emotion_result['color']
            }
        }
    
    def analyze_batch(self, texts: List[str]) -> List[Dict]:
        """Analyze multiple texts"""
        results = []
        for text in texts:
            result = self.analyze(text)
            results.append(result)
        return results

# ============= UTILITY FUNCTIONS =============

def format_confidence(confidence: float) -> str:
    """Format confidence as percentage"""
    return f"{confidence * 100:.1f}%"

def get_sentiment_emoji(sentiment: str) -> str:
    """Get emoji for sentiment"""
    emojis = {
        "Positive": "😊",
        "Neutral": "😐",
        "Negative": "😢"
    }
    return emojis.get(sentiment, "❓")

def get_emotion_emoji(emotion: str) -> str:
    """Get emoji from emotion label"""
    return emotion.split()[0]  # Get emoji from label like "😢 Sadness"

def batch_analyze_to_json(results: List[Dict], pretty: bool = True) -> str:
    """Convert analysis results to JSON"""
    json_results = []
    for result in results:
        json_results.append(result)
    
    if pretty:
        return json.dumps(json_results, indent=2)
    else:
        return json.dumps(json_results)

def calculate_statistics(results: List[Dict]) -> Dict:
    """Calculate statistics from batch analysis"""
    if not results:
        return {}
    
    sentiments = [r['sentiment']['label'] for r in results]
    emotions = [r['emotion']['label'].split()[-1] for r in results]  # Extract emotion name
    
    sentiment_counts = {}
    emotion_counts = {}
    
    for sentiment in sentiments:
        sentiment_counts[sentiment] = sentiment_counts.get(sentiment, 0) + 1
    
    for emotion in emotions:
        emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
    
    return {
        'total_texts': len(results),
        'sentiment_distribution': sentiment_counts,
        'emotion_distribution': emotion_counts,
        'avg_sentiment_confidence': np.mean([r['sentiment']['confidence'] for r in results]),
        'avg_emotion_confidence': np.mean([r['emotion']['confidence'] for r in results])
    }

# ============= EXAMPLE USAGE =============

if __name__ == "__main__":
    # Example texts
    examples = [
        "I love this product! It's amazing!",
        "This is terrible and I hate it.",
        "The weather is okay today.",
        "I'm so excited about the trip!",
        "I'm feeling sad and lonely.",
        "This is shocking news!"
    ]
    
    print("Loading models...")
    analyzer = SentimentEmotionAnalyzer()
    
    print("\n" + "="*60)
    print("SENTIMENT + EMOTION ANALYSIS")
    print("="*60)
    
    for text in examples:
        result = analyzer.analyze(text)
        print(f"\n📝 Text: {text}")
        print(f"😊 Sentiment: {result['sentiment']['label']} ({result['sentiment']['confidence']:.2%})")
        print(f"💭 Emotion: {result['emotion']['label']} ({result['emotion']['confidence']:.2%})")
