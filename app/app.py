# app.py

import os
import sys
import logging
import argparse
import shutil
import uvicorn
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, File, UploadFile, Form, Depends
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
import time
import math
from pathlib import Path
import json
import uuid
import traceback

# Configure comprehensive logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('api.log') if not os.getenv('DISABLE_FILE_LOGGING') else logging.NullHandler()
    ]
)

logger = logging.getLogger(__name__)

# Try to import scipy for statistical calculations
try:
    from scipy.stats import norm
    logger.info("Successfully imported scipy for statistical calculations")
except ImportError:
    logger.warning("scipy not available, using numpy for statistical calculations")
    norm = None

# Import your existing modules with enhanced fallback handling
try:
    from .dependencies import device
    logger.info(f"Successfully imported device from dependencies: {device}")
except ImportError:
    logger.warning("Failed to import device from dependencies, using CPU")
    device = 'cpu'

try:
    from .config import Config
    logger.info("Successfully imported Config")
except ImportError as e:
    logger.error(f"Failed to import Config: {e}")
    class Config:
        INFERENCE_BATCH_SIZE = 16
        MODEL_NAME = "hate-speech-detector"
        MAX_LENGTH = 512
        USE_MINIMAL_PREPROCESSING = False
        REMOVE_STOPWORDS = False
        USE_STEMMING = False
        NORMALIZE_SLANG = False
        API_VERSION = "2.0.0"
        ENVIRONMENT = "development"
    logger.warning("Using fallback Config class")

try:
    from .predictor import HateSpeechPredictor
    logger.info("Successfully imported HateSpeechPredictor")
except ImportError as e:
    logger.error(f"Failed to import HateSpeechPredictor: {e}")
    
    # Enhanced dummy predictor for development/testing
    class HateSpeechPredictor:
        def __init__(self, config=None):
            self.config = config
            self.model_loaded = False
            self.model_info = {
                'model_name': 'hate-speech-detector-demo',
                'model_type': 'transformer',
                'status': 'demo_mode',
                'num_parameters': 124000000,
                'version': '1.0.0',
                'accuracy': 89.44,
                'precision': 89.50,
                'recall': 89.44,
                'f1_score': 89.44,
                'auc': 96.16
            }
            
        def load_model(self, model_path):
            logger.warning("Using dummy predictor - no actual model loading")
            self.model_loaded = False
            
        def predict(self, text, return_probabilities=True, preprocess=True):
            import random
            import re
            
            # Simple heuristic for demo purposes
            hate_indicators = ['bodoh', 'tolol', 'bangsat', 'bego', 'idiot', 'sampah']
            text_lower = text.lower()
            
            # Check for hate indicators
            hate_score = 0
            for indicator in hate_indicators:
                if indicator in text_lower:
                    hate_score += 0.3
            
            # Add some randomness
            hate_score += random.uniform(0, 0.4)
            hate_score = min(hate_score, 1.0)
            
            is_hate = hate_score > 0.5
            hate_prob = hate_score if is_hate else random.uniform(0.0, 0.4)
            safe_prob = 1 - hate_prob
            
            confidence = max(hate_prob, safe_prob)
            
            return {
                'original_text': text,
                'processed_text': re.sub(r'[^\w\s]', '', text.lower()),
                'prediction': 'Ujaran Kebencian' if is_hate else 'Bukan Ujaran Kebencian',
                'is_hate_speech': is_hate,
                'probabilities': {
                    'hate': round(hate_prob, 4),
                    'safe': round(safe_prob, 4)
                },
                'confidence': round(confidence, 4),
                'analysis': {
                    'risk_assessment': self._assess_risk(hate_prob, confidence),
                    'text_length': len(text)
                }
            }
            
        def predict_batch(self, texts, batch_size=16, return_probabilities=True, preprocess=True):
            results = []
            for text in texts:
                result = self.predict(text, return_probabilities, preprocess)
                results.append(result)
            return results
            
        def _assess_risk(self, hate_prob, confidence):
            if hate_prob < 0.3:
                return 'Low'
            elif hate_prob < 0.6:
                return 'Low-Medium'
            elif hate_prob < 0.8:
                return 'Medium'
            else:
                return 'High'
            
        def get_model_info(self):
            return self.model_info
    
    logger.warning("Using fallback HateSpeechPredictor class")

# =============================================================================
# 1. Enhanced Pydantic Models for Request and Response Validation
# =============================================================================

class TextPredictionRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000, description="Text to analyze for hate speech.")
    include_confidence_interval: bool = Field(False, description="Whether to include confidence interval in the response.")
    confidence_level: float = Field(0.95, ge=0.01, le=0.99, description="Confidence level for the interval (e.g., 0.95 for 95%).")

class BatchPredictionRequest(BaseModel):
    texts: List[str] = Field(..., min_items=1, max_items=100, description="List of texts to analyze for hate speech.")
    include_statistics: bool = Field(True, description="Whether to include batch statistics in the response.")
    include_confidence_interval: bool = Field(False, description="Whether to include confidence interval for each prediction.")
    confidence_level: float = Field(0.95, ge=0.01, le=0.99, description="Confidence level for the interval (e.g., 0.95 for 95%).")

class PredictionResponse(BaseModel):
    success: bool = True
    prediction: Dict[str, Any]
    processing_time_ms: float
    model_info: Dict[str, Any] = {}
    message: str = "Analysis completed successfully."

class BatchPredictionResponse(BaseModel):
    success: bool = True
    predictions: List[Dict[str, Any]]
    statistics: Optional[Dict[str, Any]] = None
    processing_time_ms: float
    total_samples: int
    message: str = "Batch analysis completed successfully."

class CSVPredictionResponse(BaseModel):
    success: bool
    message: str
    total_samples: int
    processing_time_ms: float
    download_url: str
    statistics: Optional[Dict[str, Any]] = None

class ModelInfoResponse(BaseModel):
    success: bool = True
    model_info: Dict[str, Any]
    config: Dict[str, Any]
    message: str = "Model information retrieved successfully."

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_load_error: Optional[str] = None
    api_version: str
    timestamp: str
    device: str
    mode: str
    environment: Optional[str] = None

# =============================================================================
# 2. Enhanced Statistical Analyzer Class
# =============================================================================

class StatisticalAnalyzer:
    """
    Enhanced class for performing comprehensive statistical analysis on prediction results.
    """
    def __init__(self):
        self.logger = logger

    def calculate_confidence_interval(self, p_hat: float, n: int, confidence_level: float = 0.95) -> Dict[str, float]:
        """
        Calculates the Wilson score confidence interval for a proportion.
        """
        if not (0 <= p_hat <= 1):
            self.logger.warning(f"Invalid p_hat value for CI calculation: {p_hat}. Must be between 0 and 1.")
            return {"lower_bound": 0.0, "upper_bound": 1.0, "margin_of_error": 0.5}
        if n <= 0:
            self.logger.warning(f"Invalid n value for CI calculation: {n}. Must be positive.")
            return {"lower_bound": 0.0, "upper_bound": 1.0, "margin_of_error": 0.5}
        if not (0.01 <= confidence_level <= 0.99):
            self.logger.warning(f"Invalid confidence_level: {confidence_level}. Using 0.95.")
            confidence_level = 0.95

        try:
            # Use scipy if available, otherwise fallback to numpy approximation
            if norm is not None:
                z = norm.ppf(1 - (1 - confidence_level) / 2)
            else:
                # Approximation using numpy for common confidence levels
                if confidence_level >= 0.99:
                    z = 2.576
                elif confidence_level >= 0.95:
                    z = 1.96
                elif confidence_level >= 0.90:
                    z = 1.645
                else:
                    z = 1.96  # Default to 95%
            
            z_squared = z**2
            denominator = 1 + z_squared / n
            center = (p_hat + z_squared / (2 * n)) / denominator
            margin = (z / denominator) * math.sqrt(p_hat * (1 - p_hat) / n + z_squared / (4 * n**2))

            lower_bound = max(0.0, center - margin)
            upper_bound = min(1.0, center + margin)
            margin_of_error = (upper_bound - lower_bound) / 2

            return {
                "lower_bound": round(lower_bound, 4),
                "upper_bound": round(upper_bound, 4),
                "margin_of_error": round(margin_of_error, 4)
            }
        except Exception as e:
            self.logger.error(f"Error calculating confidence interval: {e}")
            return {"lower_bound": 0.0, "upper_bound": 1.0, "margin_of_error": 0.5}

    def _create_histogram_data(self, data: List[float], num_bins: int = 10) -> Dict[str, List[float]]:
        """Helper to create histogram data (bins and counts)."""
        if not data:
            return {"bins": [], "counts": []}
        
        data_np = np.array(data)
        min_val = np.min(data_np)
        max_val = np.max(data_np)
        
        if min_val == max_val:
            return {"bins": [float(min_val)], "counts": [len(data)]}

        counts, bin_edges = np.histogram(data_np, bins=num_bins, range=(min_val, max_val))
        bin_centers = [(bin_edges[i] + bin_edges[i+1]) / 2 for i in range(len(bin_edges) - 1)]
        
        return {"bins": [float(b) for b in bin_centers], "counts": [int(c) for c in counts.tolist()]}

    def generate_batch_statistics(self, predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generates comprehensive statistics for a batch of predictions.
        """
        if not predictions:
            self.logger.warning("No predictions provided for batch statistics generation.")
            return {}

        hate_speech_count = sum(1 for p in predictions if p.get('is_hate_speech'))
        safe_speech_count = len(predictions) - hate_speech_count

        hate_probabilities = [p['probabilities']['hate'] for p in predictions if 'probabilities' in p and 'hate' in p['probabilities']]
        safe_probabilities = [p['probabilities']['safe'] for p in predictions if 'probabilities' in p and 'safe' in p['probabilities']]
        confidences = [p['confidence'] for p in predictions if 'confidence' in p]

        stats = {
            "summary": {
                "total_samples": len(predictions),
                "hate_speech_count": hate_speech_count,
                "safe_speech_count": safe_speech_count,
                "hate_speech_percentage": round((hate_speech_count / len(predictions)) * 100, 2) if predictions else 0,
                "safe_speech_percentage": round((safe_speech_count / len(predictions)) * 100, 2) if predictions else 0,
            },
            "distributions": {
                "hate_probabilities_stats": {
                    "mean": round(float(np.mean(hate_probabilities)), 4) if hate_probabilities else 0,
                    "std": round(float(np.std(hate_probabilities)), 4) if hate_probabilities else 0,
                    "min": round(float(np.min(hate_probabilities)), 4) if hate_probabilities else 0,
                    "max": round(float(np.max(hate_probabilities)), 4) if hate_probabilities else 0,
                    "median": round(float(np.median(hate_probabilities)), 4) if hate_probabilities else 0,
                },
                "safe_probabilities_stats": {
                    "mean": round(float(np.mean(safe_probabilities)), 4) if safe_probabilities else 0,
                    "std": round(float(np.std(safe_probabilities)), 4) if safe_probabilities else 0,
                    "min": round(float(np.min(safe_probabilities)), 4) if safe_probabilities else 0,
                    "max": round(float(np.max(safe_probabilities)), 4) if safe_probabilities else 0,
                    "median": round(float(np.median(safe_probabilities)), 4) if safe_probabilities else 0,
                },
                "confidence_stats": {
                    "mean": round(float(np.mean(confidences)), 4) if confidences else 0,
                    "std": round(float(np.std(confidences)), 4) if confidences else 0,
                    "min": round(float(np.min(confidences)), 4) if confidences else 0,
                    "max": round(float(np.max(confidences)), 4) if confidences else 0,
                    "median": round(float(np.median(confidences)), 4) if confidences else 0,
                },
                "confidence_bins": self._create_histogram_data(confidences, num_bins=10),
                "hate_probability_bins": self._create_histogram_data(hate_probabilities, num_bins=10),
            },
            "risk_assessment_counts": {
                "High": sum(1 for p in predictions if self._assess_risk(p) == "High"),
                "Medium": sum(1 for p in predictions if self._assess_risk(p) == "Medium"),
                "Low-Medium": sum(1 for p in predictions if self._assess_risk(p) == "Low-Medium"),
                "Low": sum(1 for p in predictions if self._assess_risk(p) == "Low"),
            }
        }
        return stats

    def _assess_risk(self, prediction: Dict[str, Any]) -> str:
        """Assess risk level based on prediction probabilities and confidence."""
        if not prediction.get('is_hate_speech'):
            return 'Low'

        hate_prob = prediction.get('probabilities', {}).get('hate', 0)
        confidence = prediction.get('confidence', 0)

        if hate_prob >= 0.8 and confidence >= 0.8:
            return 'High'
        elif hate_prob >= 0.6 or confidence >= 0.6:
            return 'Medium'
        elif hate_prob >= 0.3:
            return 'Low-Medium'
        else:
            return 'Low'

# =============================================================================
# 3. Enhanced FastAPI Application Setup
# =============================================================================

class HateSpeechAPI:
    def __init__(self):
        self.app = FastAPI(
            title="Advanced Hate Speech Detection API",
            description="Enhanced API for detecting hate speech in Indonesian text with comprehensive statistical analysis and visualization support.",
            version="2.0.0",
            docs_url="/docs",
            redoc_url="/redoc",
            openapi_tags=[
                {"name": "Health", "description": "Health check and status endpoints"},
                {"name": "Model", "description": "Model information and configuration"},
                {"name": "Prediction", "description": "Text prediction endpoints"},
                {"name": "Batch", "description": "Batch processing endpoints"},
                {"name": "CSV", "description": "CSV file processing endpoints"},
                {"name": "Static", "description": "Static file serving"}
            ]
        )
        
        # Enhanced CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=[
                "http://localhost:3000",
                "http://localhost:8080", 
                "http://127.0.0.1:3000",
                "http://127.0.0.1:8080",
                "*"  # In production, replace with specific origins
            ],
            allow_credentials=True,
            allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
            allow_headers=["*"],
        )
        
        self.config = Config()
        self.predictor = None
        self.analyzer = StatisticalAnalyzer()
        self.model_loaded = False
        self.model_load_error = None
        self.startup_time = datetime.now()

        # Enhanced directories - using absolute paths for Docker compatibility
        self.BASE_DIR = Path(__file__).parent.absolute()
        self.RESULTS_DIR = self.BASE_DIR / "results"
        self.UPLOADS_DIR = self.BASE_DIR / "uploads"
        self.PLOTS_DIR = self.BASE_DIR / "plots"
        self.STATIC_DIR = self.BASE_DIR / "static"
        self.LOGS_DIR = self.BASE_DIR / "logs"

        self.setup_directories()
        self._add_routes()
        
        # Initialize model after setting up routes
        self.initialize_model()

    def setup_directories(self):
        """Create necessary directories with proper permissions."""
        for directory in [self.RESULTS_DIR, self.UPLOADS_DIR, self.PLOTS_DIR, self.STATIC_DIR, self.LOGS_DIR]:
            try:
                directory.mkdir(parents=True, exist_ok=True)
                logger.info(f"Directory '{directory}' ensured.")
            except OSError as e:
                logger.error(f"Failed to create directory '{directory}': {e}")

    def initialize_model(self):
        """Load the ML model during application startup with enhanced error handling."""
        model_path = os.getenv("MODEL_PATH", "./model/optimized")
        logger.info(f"Attempting to load model from: {model_path}")
        try:
            self.predictor = HateSpeechPredictor(config=self.config)
            self.predictor.load_model(model_path)
            self.model_loaded = getattr(self.predictor, 'model_loaded', True)
            
            if self.model_loaded:
                logger.info("Model loaded successfully for API.")
            else:
                logger.warning("Model predictor created but model not properly loaded.")
        except Exception as e:
            self.model_loaded = False
            self.model_load_error = str(e)
            logger.warning(f"Failed to load model from {model_path}: {e}")
            logger.info("API will run with dummy predictor for testing.")

    def _add_routes(self):
        """Define all API endpoints with enhanced functionality."""

        # Serve static files if directory exists
        if self.STATIC_DIR.exists():
            self.app.mount("/static", StaticFiles(directory=str(self.STATIC_DIR)), name="static")
        if self.RESULTS_DIR.exists():
            self.app.mount("/results", StaticFiles(directory=str(self.RESULTS_DIR)), name="results")

        @self.app.get("/", response_class=HTMLResponse, tags=["Static"])
        async def read_root():
            """Enhanced main HTML page for the frontend application."""
            index_path = self.STATIC_DIR / "index.html"
            if index_path.exists():
                return FileResponse(str(index_path))
            
            # Enhanced HTML page with modern styling
            return HTMLResponse(content=f"""
            <!DOCTYPE html>
            <html lang="id">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>🤖 Enhanced Hate Speech Detection API</title>
                <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
                <style>
                    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
                    body {{ 
                        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; 
                        background: linear-gradient(135deg, #1a202c 0%, #2d3748 100%);
                        color: white; min-height: 100vh; display: flex; align-items: center; justify-content: center;
                    }}
                    .container {{ 
                        max-width: 900px; width: 90%; background: rgba(45, 55, 72, 0.9); 
                        padding: 40px; border-radius: 20px; backdrop-filter: blur(10px);
                        box-shadow: 0 20px 40px rgba(0,0,0,0.4); text-align: center;
                    }}
                    h1 {{ color: #4299e1; font-size: 2.5rem; margin-bottom: 20px; }}
                    .subtitle {{ font-size: 1.2rem; color: #a0aec0; margin-bottom: 30px; }}
                    .status {{ 
                        margin: 30px 0; padding: 20px; border-radius: 15px; display: inline-block;
                        min-width: 200px; font-weight: bold; font-size: 1.1rem;
                    }}
                    .status.loaded {{ background: linear-gradient(45deg, #48bb78, #38a169); }}
                    .status.error {{ background: linear-gradient(45deg, #f56565, #e53e3e); }}
                    .links {{ margin-top: 40px; display: flex; gap: 20px; justify-content: center; flex-wrap: wrap; }}
                    .links a {{ 
                        display: inline-flex; align-items: center; gap: 10px; padding: 15px 25px; 
                        background: linear-gradient(45deg, #4299e1, #3182ce); color: white; 
                        text-decoration: none; border-radius: 10px; transition: all 0.3s ease;
                        font-weight: 500;
                    }}
                    .links a:hover {{ transform: translateY(-3px); box-shadow: 0 10px 20px rgba(66, 153, 225, 0.3); }}
                    .features {{ margin-top: 40px; display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; }}
                    .feature {{ padding: 20px; background: rgba(255,255,255,0.05); border-radius: 10px; }}
                    .feature i {{ font-size: 2rem; color: #4299e1; margin-bottom: 10px; }}
                    .stats {{ margin: 30px 0; display: flex; justify-content: center; gap: 30px; flex-wrap: wrap; }}
                    .stat {{ text-align: center; }}
                    .stat-value {{ font-size: 2rem; font-weight: bold; color: #4299e1; }}
                    .stat-label {{ font-size: 0.9rem; color: #a0aec0; }}
                </style>
            </head>
            <body>
                <div class="container">
                    <h1><i class="fas fa-robot"></i> Enhanced Hate Speech Detection API</h1>
                    <p class="subtitle">Advanced Indonesian Text Analysis with AI-Powered Detection & Comprehensive Analytics</p>
                    
                    <div class="status {'loaded' if self.model_loaded else 'error'}">
                        <i class="fas fa-{'check-circle' if self.model_loaded else 'exclamation-triangle'}"></i>
                        {'✅ Model Ready & Operational' if self.model_loaded else '⚠️ Demo Mode Active - Model Not Loaded'}
                    </div>

                    <div class="stats">
                        <div class="stat">
                            <div class="stat-value">v{self.config.API_VERSION}</div>
                            <div class="stat-label">API Version</div>
                        </div>
                        <div class="stat">
                            <div class="stat-value">{'Production' if self.model_loaded else 'Demo'}</div>
                            <div class="stat-label">Mode</div>
                        </div>
                        <div class="stat">
                            <div class="stat-value">{self.startup_time.strftime('%H:%M')}</div>
                            <div class="stat-label">Started</div>
                        </div>
                    </div>

                    <div class="features">
                        <div class="feature">
                            <i class="fas fa-search"></i>
                            <h3>Single Text Analysis</h3>
                            <p>Real-time hate speech detection with confidence scoring</p>
                        </div>
                        <div class="feature">
                            <i class="fas fa-layer-group"></i>
                            <h3>Batch Processing</h3>
                            <p>Process multiple texts with comprehensive statistics</p>
                        </div>
                        <div class="feature">
                            <i class="fas fa-file-csv"></i>
                            <h3>CSV Analysis</h3>
                            <p>Upload and analyze CSV files with detailed reporting</p>
                        </div>
                        <div class="feature">
                            <i class="fas fa-chart-line"></i>
                            <h3>Advanced Analytics</h3>
                            <p>Interactive visualizations and statistical insights</p>
                        </div>
                    </div>
                    
                    <div class="links">
                        <a href="/docs"><i class="fas fa-book"></i> Interactive API Docs</a>
                        <a href="/health"><i class="fas fa-heart"></i> System Health</a>
                        <a href="/model/info"><i class="fas fa-info-circle"></i> Model Information</a>
                        <a href="/redoc"><i class="fas fa-file-alt"></i> API Reference</a>
                    </div>
                </div>
            </body>
            </html>
            """)

        @self.app.get("/health", response_model=HealthResponse, tags=["Health"])
        async def health_check():
            """Enhanced health check endpoint with detailed system information."""
            return HealthResponse(
                status="ok" if self.model_loaded else "degraded",
                model_loaded=self.model_loaded,
                model_load_error=self.model_load_error,
                api_version=self.app.version,
                timestamp=datetime.now().isoformat(),
                device=str(device),
                mode="production" if self.model_loaded else "demo",
                environment=getattr(self.config, 'ENVIRONMENT', 'development')
            )

        @self.app.get("/model/info", response_model=ModelInfoResponse, tags=["Model"])
        async def get_model_info():
            """Enhanced model information endpoint."""
            try:
                model_info = self.predictor.get_model_info() if self.predictor else {
                    "error": "No predictor available",
                    "status": "unavailable"
                }
                
                config_dict = {
                    k: v for k, v in self.config.__dict__.items() 
                    if not k.startswith('_') and not callable(v)
                }
                
                return ModelInfoResponse(
                    model_info=model_info,
                    config=config_dict,
                    message="Model information retrieved successfully."
                )
            except Exception as e:
                logger.error(f"Error retrieving model info: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to retrieve model info: {e}")

        @self.app.post("/predict/single", response_model=PredictionResponse, tags=["Prediction"])
        async def predict_single_text(request: TextPredictionRequest):
            """Enhanced single text prediction endpoint."""
            if not self.predictor:
                raise HTTPException(status_code=503, detail="Predictor not available.")

            start_time = time.perf_counter()
            try:
                prediction_result = self.predictor.predict(
                    request.text,
                    return_probabilities=True,
                    preprocess=True
                )

                # Add confidence interval if requested
                if request.include_confidence_interval and 'probabilities' in prediction_result:
                    p_hat = prediction_result['probabilities']['hate']
                    n_samples = 100  # Sample size for CI calculation
                    prediction_result['confidence_interval'] = self.analyzer.calculate_confidence_interval(
                        p_hat, n_samples, request.confidence_level
                    )
                
                # Ensure analysis field is present
                if 'analysis' not in prediction_result:
                    prediction_result['analysis'] = {
                        'text_length': len(request.text),
                        'risk_assessment': self._assess_risk(prediction_result)
                    }

                end_time = time.perf_counter()
                processing_time_ms = (end_time - start_time) * 1000

                return PredictionResponse(
                    prediction=prediction_result,
                    processing_time_ms=round(processing_time_ms, 2),
                    model_info=self.predictor.get_model_info() if self.predictor else {}
                )
            except Exception as e:
                logger.error(f"Error during single text prediction: {e}")
                logger.error(traceback.format_exc())
                raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

        @self.app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Batch"])
        async def predict_batch_texts(request: BatchPredictionRequest):
            """Enhanced batch text prediction endpoint."""
            if not self.predictor:
                raise HTTPException(status_code=503, detail="Predictor not available.")

            start_time = time.perf_counter()
            try:
                predictions = self.predictor.predict_batch(
                    request.texts,
                    batch_size=getattr(self.config, 'INFERENCE_BATCH_SIZE', 16),
                    return_probabilities=True,
                    preprocess=True
                )

                # Add confidence intervals and analysis if requested
                for pred in predictions:
                    if request.include_confidence_interval and 'probabilities' in pred:
                        p_hat = pred['probabilities']['hate']
                        n_samples = 100
                        pred['confidence_interval'] = self.analyzer.calculate_confidence_interval(
                            p_hat, n_samples, request.confidence_level
                        )
                    
                    # Ensure analysis field is present
                    if 'analysis' not in pred:
                        pred['analysis'] = {
                            'text_length': len(pred.get('original_text', '')),
                            'risk_assessment': self._assess_risk(pred)
                        }

                statistics = None
                if request.include_statistics:
                    statistics = self.analyzer.generate_batch_statistics(predictions)

                end_time = time.perf_counter()
                processing_time_ms = (end_time - start_time) * 1000

                return BatchPredictionResponse(
                    predictions=predictions,
                    statistics=statistics,
                    processing_time_ms=round(processing_time_ms, 2),
                    total_samples=len(request.texts)
                )
            except Exception as e:
                logger.error(f"Error during batch text prediction: {e}")
                logger.error(traceback.format_exc())
                raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

        @self.app.post("/predict/csv", response_model=CSVPredictionResponse, tags=["CSV"])
        async def process_csv_file(
            file: UploadFile = File(...),
            text_column: str = Form(...),
            include_confidence_interval: bool = Form(False),
            confidence_level: float = Form(0.95)
        ):
            """Enhanced CSV file processing endpoint."""
            if not self.predictor:
                raise HTTPException(status_code=503, detail="Predictor not available.")

            # Validate file type
            if not file.filename or not file.filename.lower().endswith('.csv'):
                raise HTTPException(status_code=400, detail="Invalid file type. Only CSV files are allowed.")

            # Size limit check
            MAX_FILE_SIZE_MB = 50
            content = await file.read()
            if len(content) > MAX_FILE_SIZE_MB * 1024 * 1024:
                raise HTTPException(
                    status_code=413, 
                    detail=f"File too large. Maximum size is {MAX_FILE_SIZE_MB}MB."
                )

            # Save uploaded file temporarily with unique name
            file_id = str(uuid.uuid4())
            upload_path = self.UPLOADS_DIR / f"{file_id}_{file.filename}"
            
            try:
                with open(upload_path, "wb") as buffer:
                    buffer.write(content)
                logger.info(f"Uploaded CSV saved to {upload_path}")
            except Exception as e:
                logger.error(f"Failed to save uploaded file: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to save uploaded file: {str(e)}")

            # Read and validate CSV
            try:
                df = pd.read_csv(upload_path, encoding='utf-8')
            except UnicodeDecodeError:
                try:
                    df = pd.read_csv(upload_path, encoding='latin-1')
                except Exception as e:
                    logger.error(f"Failed to read CSV with multiple encodings: {e}")
                    if upload_path.exists():
                        upload_path.unlink()
                    raise HTTPException(status_code=400, detail=f"Failed to read CSV file: {str(e)}")
            except Exception as e:
                logger.error(f"Failed to read CSV file: {e}")
                if upload_path.exists():
                    upload_path.unlink()
                raise HTTPException(status_code=400, detail=f"Failed to read CSV file: {str(e)}")
            finally:
                # Clean up uploaded file
                if upload_path.exists():
                    try:
                        upload_path.unlink()
                    except Exception as e:
                        logger.warning(f"Failed to cleanup uploaded file: {e}")

            # Validate text column
            if text_column not in df.columns:
                available_columns = ', '.join(df.columns.tolist())
                raise HTTPException(
                    status_code=400, 
                    detail=f"Text column '{text_column}' not found in CSV. Available columns: {available_columns}"
                )

            # Prepare texts for prediction
            texts_to_predict = df[text_column].fillna('').astype(str).tolist()
            total_samples = len(texts_to_predict)

            # Row limit check
            MAX_CSV_ROWS = 50000
            if total_samples > MAX_CSV_ROWS:
                raise HTTPException(
                    status_code=413, 
                    detail=f"Too many rows. Maximum {MAX_CSV_ROWS} rows allowed, found {total_samples}."
                )

            # Filter out empty texts
            non_empty_texts = [(i, text) for i, text in enumerate(texts_to_predict) if text.strip()]
            if not non_empty_texts:
                raise HTTPException(status_code=400, detail="No valid text content found in the specified column.")

            start_time = time.perf_counter()
            try:
                # Process predictions
                predictions = self.predictor.predict_batch(
                    [text for _, text in non_empty_texts],
                    batch_size=getattr(self.config, 'INFERENCE_BATCH_SIZE', 16),
                    return_probabilities=True,
                    preprocess=True
                )

                # Add confidence intervals and analysis
                for pred in predictions:
                    if include_confidence_interval and 'probabilities' in pred:
                        p_hat = pred['probabilities']['hate']
                        pred['confidence_interval'] = self.analyzer.calculate_confidence_interval(
                            p_hat, 100, confidence_level
                        )
                    
                    # Ensure analysis field is present
                    if 'analysis' not in pred:
                        pred['analysis'] = {
                            'text_length': len(pred.get('original_text', '')),
                            'risk_assessment': self._assess_risk(pred)
                        }

                # Create comprehensive results DataFrame
                results_data = []
                pred_index = 0
                
                for i, original_text in enumerate(texts_to_predict):
                    if original_text.strip() and pred_index < len(predictions):
                        pred = predictions[pred_index]
                        row = {
                            'row_index': i + 1,
                            'original_text': original_text,
                            'processed_text': pred.get('processed_text', ''),
                            'prediction': pred.get('prediction', ''),
                            'is_hate_speech': pred.get('is_hate_speech', False),
                            'hate_probability': pred.get('probabilities', {}).get('hate', 0),
                            'safe_probability': pred.get('probabilities', {}).get('safe', 0),
                            'confidence': pred.get('confidence', 0),
                            'risk_assessment': pred.get('analysis', {}).get('risk_assessment', 'Unknown'),
                            'text_length': len(original_text),
                        }
                        
                        if include_confidence_interval and 'confidence_interval' in pred:
                            ci = pred['confidence_interval']
                            row.update({
                                'ci_lower_bound': ci.get('lower_bound', 0),
                                'ci_upper_bound': ci.get('upper_bound', 0),
                                'ci_margin_of_error': ci.get('margin_of_error', 0)
                            })
                        
                        pred_index += 1
                    else:
                        # Handle empty rows
                        row = {
                            'row_index': i + 1,
                            'original_text': original_text,
                            'processed_text': '',
                            'prediction': 'Skipped (Empty)',
                            'is_hate_speech': False,
                            'hate_probability': 0,
                            'safe_probability': 0,
                            'confidence': 0,
                            'risk_assessment': 'N/A',
                            'text_length': len(original_text),
                        }
                        
                        if include_confidence_interval:
                            row.update({
                                'ci_lower_bound': 0,
                                'ci_upper_bound': 0,
                                'ci_margin_of_error': 0
                            })
                    
                    results_data.append(row)

                results_df = pd.DataFrame(results_data)

                # Generate comprehensive statistics
                valid_predictions = [pred for pred in predictions if pred.get('original_text', '').strip()]
                statistics = self.analyzer.generate_batch_statistics(valid_predictions) if valid_predictions else {}

                # Save results with timestamp
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                output_filename = f"hate_speech_results_{timestamp}_{file_id[:8]}.csv"
                output_filepath = self.RESULTS_DIR / output_filename
                
                results_df.to_csv(output_filepath, index=False, encoding='utf-8-sig')
                logger.info(f"CSV results saved to {output_filepath}")

                end_time = time.perf_counter()
                processing_time_ms = (end_time - start_time) * 1000

                return CSVPredictionResponse(
                    success=True,
                    message=f"CSV processed successfully. {len(valid_predictions)} texts analyzed, {total_samples - len(valid_predictions)} skipped.",
                    total_samples=total_samples,
                    processing_time_ms=round(processing_time_ms, 2),
                    download_url=f"/results/{output_filename}",
                    statistics=statistics
                )

            except Exception as e:
                logger.error(f"Error processing CSV file: {e}")
                logger.error(traceback.format_exc())
                raise HTTPException(status_code=500, detail=f"CSV processing failed: {str(e)}")

        # Additional utility endpoints
        @self.app.get("/api/stats", tags=["Analytics"])
        async def get_api_stats():
            """Get API usage statistics."""
            try:
                uptime = datetime.now() - self.startup_time
                return {
                    "success": True,
                    "uptime_seconds": int(uptime.total_seconds()),
                    "uptime_formatted": str(uptime).split('.')[0],
                    "model_status": "loaded" if self.model_loaded else "not_loaded",
                    "api_version": self.app.version,
                    "endpoints_count": len([route for route in self.app.routes]),
                    "timestamp": datetime.now().isoformat()
                }
            except Exception as e:
                logger.error(f"Error getting API stats: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")

        @self.app.post("/api/validate", tags=["Utilities"])
        async def validate_text(text: str):
            """Validate text input for processing."""
            try:
                validation_result = {
                    "valid": True,
                    "length": len(text),
                    "word_count": len(text.split()),
                    "char_count": len(text),
                    "has_content": bool(text.strip()),
                    "issues": []
                }
                
                if not text.strip():
                    validation_result["valid"] = False
                    validation_result["issues"].append("Text is empty or only whitespace")
                
                if len(text) > 2000:
                    validation_result["valid"] = False
                    validation_result["issues"].append(f"Text too long ({len(text)} > 2000 characters)")
                
                if len(text.split()) > 500:
                    validation_result["issues"].append("Text contains many words, processing may be slower")
                
                return {
                    "success": True,
                    "validation": validation_result
                }
            except Exception as e:
                logger.error(f"Error validating text: {e}")
                raise HTTPException(status_code=500, detail=f"Validation failed: {str(e)}")

        @self.app.get("/api/supported-formats", tags=["Utilities"])
        async def get_supported_formats():
            """Get information about supported input formats."""
            return {
                "success": True,
                "formats": {
                    "single_text": {
                        "max_length": 2000,
                        "encoding": "UTF-8",
                        "supported": True
                    },
                    "batch_text": {
                        "max_items": 100,
                        "max_length_per_item": 2000,
                        "supported": True
                    },
                    "csv_upload": {
                        "max_file_size_mb": 50,
                        "max_rows": 50000,
                        "supported_encodings": ["UTF-8", "Latin-1"],
                        "required_columns": ["text_column (user-specified)"],
                        "supported": True
                    }
                }
            }

    def _assess_risk(self, prediction: Dict[str, Any]) -> str:
        """Enhanced risk assessment based on prediction."""
        if not prediction.get('is_hate_speech', False):
            return 'Low'

        hate_prob = prediction.get('probabilities', {}).get('hate', 0)
        confidence = prediction.get('confidence', 0)

        if hate_prob >= 0.8 and confidence >= 0.8:
            return 'High'
        elif hate_prob >= 0.6 or confidence >= 0.6:
            return 'Medium'  
        elif hate_prob >= 0.3:
            return 'Low-Medium'
        else:
            return 'Low'

# =============================================================================
# 4. Enhanced Application Factory with Error Handling
# =============================================================================

def create_app():
    """Enhanced factory function to create and return the FastAPI application instance."""
    try:
        api = HateSpeechAPI()
        logger.info("Enhanced FastAPI application created successfully")
        
        # Add startup event
        @api.app.on_event("startup")
        async def startup_event():
            logger.info("🚀 Enhanced Hate Speech Detection API starting up...")
            logger.info(f"📊 Model Status: {'Loaded' if api.model_loaded else 'Demo Mode'}")
            logger.info(f"🔧 Environment: {getattr(api.config, 'ENVIRONMENT', 'development')}")
            logger.info(f"⚡ API Version: {api.app.version}")
        
        # Add shutdown event  
        @api.app.on_event("shutdown")
        async def shutdown_event():
            logger.info("🛑 Enhanced Hate Speech Detection API shutting down...")
            # Cleanup logic here if needed
            
        return api.app
    except Exception as e:
        logger.error(f"Failed to create Enhanced FastAPI application: {e}")
        logger.error(traceback.format_exc())
        
        # Return a minimal FastAPI app for error cases
        error_app = FastAPI(
            title="Hate Speech Detection API - Error Mode",
            description="API failed to initialize properly",
            version="2.0.0-error"
        )
        
        @error_app.get("/")
        async def error_root():
            return {
                "error": "Application failed to initialize",
                "detail": str(e),
                "status": "error",
                "timestamp": datetime.now().isoformat()
            }
            
        @error_app.get("/health")
        async def error_health():
            return {
                "status": "error", 
                "message": f"Application failed to initialize: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }
            
        return error_app

# =============================================================================
# 5. Enhanced Main Execution Block
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Enhanced Hate Speech Detection API",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--host", type=str, default="0.0.0.0", 
                       help="Host address to bind to.")
    parser.add_argument("--port", type=int, default=8000, 
                       help="Port to listen on.")
    parser.add_argument("--reload", action="store_true", 
                       help="Enable auto-reload (for development).")
    parser.add_argument("--model_path", type=str, default="./model/optimized",
                       help="Path to the trained model directory.")
    parser.add_argument("--log-level", type=str, default="info",
                       choices=["debug", "info", "warning", "error"],
                       help="Logging level.")
    parser.add_argument("--workers", type=int, default=1,
                       help="Number of worker processes.")
    
    args = parser.parse_args()

    # Set environment variable for model path
    os.environ["MODEL_PATH"] = args.model_path
    
    # Configure logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level.upper()))

    logger.info("🚀 Starting Enhanced Hate Speech Detection API")
    logger.info(f"📁 Model path: {args.model_path}")
    logger.info(f"🌐 Server: http://{args.host}:{args.port}")
    logger.info(f"📚 Documentation: http://{args.host}:{args.port}/docs")
    logger.info(f"🔧 Log level: {args.log_level}")
    
    try:
        uvicorn.run(
            "app:create_app",
            host=args.host,
            port=args.port,
            reload=args.reload,
            factory=True,
            log_level=args.log_level,
            workers=args.workers if not args.reload else 1,
            access_log=True,
            use_colors=True
        )
    except KeyboardInterrupt:
        logger.info("🛑 Server stopped by user")
    except Exception as e:
        logger.error(f"❌ Failed to start server: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)
