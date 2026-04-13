"""
FastAPI Backend for Sentiment Analysis
Provides REST API endpoints for sentiment and emotion detection
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sentiment_utils import SentimentEmotionAnalyzer, calculate_statistics
import json
import csv
from io import StringIO
import logging
from typing import List, Optional
from datetime import datetime
import uvicorn

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============= FASTAPI SETUP =============

app = FastAPI(
    title="Sentiment Analysis API",
    description="Analyze sentiment and emotions in text using NLP",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============= LOAD MODELS (GLOBAL) =============

try:
    analyzer = SentimentEmotionAnalyzer()
    logger.info("✅ Models loaded successfully")
except Exception as e:
    logger.error(f"❌ Error loading models: {e}")
    analyzer = None

# ============= PYDANTIC MODELS =============

class TextInput(BaseModel):
    """Input model for single text analysis"""
    text: str = Field(..., min_length=1, max_length=1000, description="Text to analyze")
    clean_text: bool = Field(True, description="Whether to clean the text")

class BatchTextInput(BaseModel):
    """Input model for batch text analysis"""
    texts: List[str] = Field(..., min_items=1, max_items=100, description="List of texts")
    clean_text: bool = Field(True, description="Whether to clean texts")

class AnalysisResponse(BaseModel):
    """Response model for single analysis"""
    text: str
    sentiment: dict
    emotion: dict
    timestamp: str

class BatchAnalysisResponse(BaseModel):
    """Response model for batch analysis"""
    results: List[dict]
    statistics: dict
    timestamp: str
    count: int

# ============= HEALTH CHECK ENDPOINTS =============

@app.get("/")
async def root():
    """Root endpoint with API info"""
    return {
        "name": "Sentiment Analysis API",
        "version": "1.0.0",
        "status": "🟢 Running",
        "models_loaded": analyzer is not None,
        "endpoints": {
            "health": "/health",
            "analyze_single": "/api/analyze",
            "analyze_batch": "/api/analyze-batch",
            "analyze_csv": "/api/analyze-csv",
            "health_check": "/health",
            "models_info": "/api/models-info",
            "api_docs": "/docs"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "models_loaded": analyzer is not None,
        "timestamp": datetime.now().isoformat()
    }

# ============= SINGLE TEXT ANALYSIS =============

@app.post("/api/analyze", response_model=dict)
async def analyze_text(input_data: TextInput):
    """
    Analyze sentiment and emotion for a single text
    
    Parameters:
    - text: Text to analyze (max 1000 characters)
    - clean_text: Whether to clean the text (default: true)
    
    Returns:
    - text: Original text
    - sentiment: Sentiment analysis result
    - emotion: Emotion detection result
    - timestamp: Analysis timestamp
    """
    
    if not analyzer:
        raise HTTPException(status_code=500, detail="Models not loaded")
    
    try:
        # Analyze
        result = analyzer.analyze(input_data.text)
        
        return {
            "status": "success",
            "text": result['text'],
            "processed_text": result['processed_text'],
            "sentiment": result['sentiment'],
            "emotion": result['emotion'],
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Error in analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============= BATCH ANALYSIS =============

@app.post("/api/analyze-batch", response_model=dict)
async def analyze_batch(input_data: BatchTextInput):
    """
    Analyze multiple texts
    
    Parameters:
    - texts: List of texts to analyze (max 100 texts)
    - clean_text: Whether to clean texts (default: true)
    
    Returns:
    - results: List of analysis results
    - statistics: Summary statistics
    - count: Number of texts analyzed
    - timestamp: Analysis timestamp
    """
    
    if not analyzer:
        raise HTTPException(status_code=500, detail="Models not loaded")
    
    if len(input_data.texts) == 0:
        raise HTTPException(status_code=400, detail="No texts provided")
    
    if len(input_data.texts) > 100:
        raise HTTPException(status_code=400, detail="Maximum 100 texts allowed")
    
    try:
        # Analyze all texts
        results = analyzer.analyze_batch(input_data.texts)
        
        # Calculate statistics
        stats = calculate_statistics(results)
        
        return {
            "status": "success",
            "results": results,
            "statistics": {
                "total_texts": stats['total_texts'],
                "sentiment_distribution": stats['sentiment_distribution'],
                "emotion_distribution": stats['emotion_distribution'],
                "avg_sentiment_confidence": round(stats['avg_sentiment_confidence'], 4),
                "avg_emotion_confidence": round(stats['avg_emotion_confidence'], 4)
            },
            "count": len(results),
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Error in batch analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============= CSV ANALYSIS =============

@app.post("/api/analyze-csv")
async def analyze_csv(file: UploadFile = File(...)):
    """
    Analyze texts from CSV file
    
    CSV Format (with header):
    - text: Column containing text to analyze
    - [Optional] id: Identifier for each row
    
    Returns:
    - CSV file with analysis results
    """
    
    if not analyzer:
        raise HTTPException(status_code=500, detail="Models not loaded")
    
    try:
        # Read CSV
        contents = await file.read()
        csv_reader = csv.DictReader(StringIO(contents.decode('utf-8')))
        
        rows = list(csv_reader)
        if not rows:
            raise HTTPException(status_code=400, detail="CSV is empty")
        
        if 'text' not in rows[0]:
            raise HTTPException(
                status_code=400,
                detail="CSV must have 'text' column"
            )
        
        # Analyze all texts
        results = []
        for row in rows:
            text = row['text']
            analysis = analyzer.analyze(text)
            
            result_row = {
                'original_text': text,
                'processed_text': analysis['processed_text'],
                'sentiment': analysis['sentiment']['label'],
                'sentiment_confidence': analysis['sentiment']['confidence'],
                'emotion': analysis['emotion']['label'].split()[-1],
                'emotion_confidence': analysis['emotion']['confidence']
            }
            
            # Add ID if present
            if 'id' in row:
                result_row['id'] = row['id']
            
            results.append(result_row)
        
        # Create output CSV
        output = StringIO()
        if results:
            writer = csv.DictWriter(output, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        
        return JSONResponse(
            content={
                "status": "success",
                "rows_processed": len(results),
                "results": results,
                "timestamp": datetime.now().isoformat()
            }
        )
    
    except Exception as e:
        logger.error(f"Error processing CSV: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============= COMPARISON ENDPOINT =============

@app.post("/api/compare")
async def compare_texts(input_data: BatchTextInput):
    """
    Compare sentiment and emotion across multiple texts
    
    Returns:
    - Detailed comparison of all texts
    - Rankings by sentiment/emotion confidence
    """
    
    if not analyzer:
        raise HTTPException(status_code=500, detail="Models not loaded")
    
    if len(input_data.texts) < 2:
        raise HTTPException(status_code=400, detail="Need at least 2 texts to compare")
    
    try:
        results = analyzer.analyze_batch(input_data.texts)
        
        # Sort by sentiment confidence
        sorted_by_sentiment = sorted(
            results,
            key=lambda x: x['sentiment']['confidence'],
            reverse=True
        )
        
        # Sort by emotion confidence
        sorted_by_emotion = sorted(
            results,
            key=lambda x: x['emotion']['confidence'],
            reverse=True
        )
        
        return {
            "status": "success",
            "total_texts": len(results),
            "results": results,
            "rankings": {
                "by_sentiment_confidence": [
                    {
                        "text": r['text'][:100],
                        "sentiment": r['sentiment']['label'],
                        "confidence": r['sentiment']['confidence']
                    }
                    for r in sorted_by_sentiment
                ],
                "by_emotion_confidence": [
                    {
                        "text": r['text'][:100],
                        "emotion": r['emotion']['label'],
                        "confidence": r['emotion']['confidence']
                    }
                    for r in sorted_by_emotion
                ]
            },
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Error in comparison: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============= INFORMATION ENDPOINTS =============

@app.get("/api/models-info")
async def models_info():
    """Get information about loaded models"""
    return {
        "sentiment_model": "distilbert-base-uncased-finetuned-sst-2-english",
        "emotion_model": "j-hartmann/emotion-english-distilroberta-base",
        "sentiment_classes": 3,
        "emotion_classes": 7,
        "sentiment_labels": ["Negative", "Neutral", "Positive"],
        "emotion_labels": [
            "😢 Sadness",
            "😨 Fear",
            "😠 Anger",
            "😊 Joy",
            "😐 Neutral",
            "😲 Surprise",
            "🤢 Disgust"
        ],
        "max_length": 512,
        "device": "GPU" if analyzer else "CPU"
    }

@app.get("/api/stats")
async def get_stats():
    """Get API statistics"""
    return {
        "api_version": "1.0.0",
        "models_loaded": analyzer is not None,
        "endpoints_available": 8,
        "max_batch_size": 100,
        "max_text_length": 1000,
        "supported_formats": ["JSON", "CSV"],
        "documentation": "/docs"
    }

# ============= EXAMPLE ENDPOINT =============

@app.get("/api/examples")
async def get_examples():
    """Get example texts for testing"""
    examples = [
        "I love this product! It's amazing!",
        "This is terrible and I hate it.",
        "The weather is okay today.",
        "I'm so excited about the trip!",
        "I'm feeling sad and lonely.",
        "This is shocking news!",
        "I'm very angry about this!",
        "That's absolutely disgusting!"
    ]
    
    return {
        "status": "success",
        "examples": examples,
        "count": len(examples),
        "description": "Use these texts to test the API"
    }

# ============= ERROR HANDLERS =============

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail, "status": "error"},
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unexpected error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "status": "error"},
    )

# ============= RUN SERVER =============

if __name__ == "__main__":
    # Run with: uvicorn api_sentiment.py --host 0.0.0.0 --port 8000 --reload
    uvicorn.run(
        "api_sentiment:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
