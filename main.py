from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import cv2
import numpy as np
import base64
from typing import List, Dict, Optional
import json
import os
from datetime import datetime, timedelta
import hashlib
from io import BytesIO
from PIL import Image
import sqlite3
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="TCGinspector Pokemon API",
    description="Pokemon card authenticity checker using visual heuristics",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for your extension domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class CardAnalysisRequest(BaseModel):
    image_base64: str
    debug_mode: Optional[bool] = False

class CardAnalysisResponse(BaseModel):
    result: str  # "authentic", "possibly_fake", "likely_fake"
    confidence_score: float
    issues_detected: List[str]
    suggested_action: str
    debug_info: Optional[Dict] = None

# Rate limiting configuration
RATE_LIMIT_PER_DAY = 100
RATE_LIMIT_RESET_HOUR = 0  # Reset at midnight UTC

# Database initialization
def init_db():
    """Initialize SQLite database"""
    conn = sqlite3.connect('pokemon_inspector.db')
    cursor = conn.cursor()
    
    # Analysis history table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS analysis_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ip_address TEXT,
            image_hash TEXT,
            result TEXT,
            confidence_score REAL,
            issues_detected TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Rate limiting table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS rate_limits (
            ip_address TEXT PRIMARY KEY,
            daily_count INTEGER DEFAULT 0,
            last_reset_date DATE DEFAULT CURRENT_DATE
        )
    ''')
    
    conn.commit()
    conn.close()

# Initialize database
init_db()

def check_rate_limit(ip_address: str) -> bool:
    """Check if IP address has exceeded daily limit"""
    conn = sqlite3.connect('pokemon_inspector.db')
    cursor = conn.cursor()
    
    # Get today's date
    today = datetime.now().date()
    
    # Get current usage
    cursor.execute(
        "SELECT daily_count, last_reset_date FROM rate_limits WHERE ip_address = ?",
        (ip_address,)
    )
    result = cursor.fetchone()
    
    if result:
        daily_count, last_reset_date = result
        last_reset = datetime.strptime(last_reset_date, '%Y-%m-%d').date()
        
        # Reset counter if new day
        if last_reset < today:
            cursor.execute(
                "UPDATE rate_limits SET daily_count = 0, last_reset_date = ? WHERE ip_address = ?",
                (today, ip_address)
            )
            daily_count = 0
    else:
        # First time user
        cursor.execute(
            "INSERT INTO rate_limits (ip_address, daily_count, last_reset_date) VALUES (?, 0, ?)",
            (ip_address, today)
        )
        daily_count = 0
    
    conn.commit()
    conn.close()
    
    return daily_count < RATE_LIMIT_PER_DAY

def increment_usage(ip_address: str):
    """Increment usage count for IP address"""
    conn = sqlite3.connect('pokemon_inspector.db')
    cursor = conn.cursor()
    
    cursor.execute(
        "UPDATE rate_limits SET daily_count = daily_count + 1 WHERE ip_address = ?",
        (ip_address,)
    )
    
    conn.commit()
    conn.close()

class ImageProcessor:
    """Image processing utilities for Pokemon cards"""
    
    @staticmethod
    def base64_to_cv2(base64_string: str) -> np.ndarray:
        """Convert base64 string to OpenCV image"""
        try:
            if base64_string.startswith('data:image'):
                base64_string = base64_string.split(',')[1]
            
            image_bytes = base64.b64decode(base64_string)
            pil_image = Image.open(BytesIO(image_bytes))
            
            # Convert to RGB if needed
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            
            cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            return cv_image
        except Exception as e:
            logger.error(f"Error converting base64 to image: {e}")
            raise HTTPException(status_code=400, detail="Invalid image format")
    
    @staticmethod
    def calculate_image_hash(image: np.ndarray) -> str:
        """Calculate hash of image for deduplication"""
        # Resize to small size for consistent hashing
        small = cv2.resize(image, (32, 32))
        image_bytes = cv2.imencode('.jpg', small)[1].tobytes()
        return hashlib.md5(image_bytes).hexdigest()
    
    @staticmethod
    def preprocess_image(image: np.ndarray) -> np.ndarray:
        """Preprocess image for analysis"""
        height, width = image.shape[:2]
        
        # Resize while maintaining aspect ratio
        if max(height, width) > 800:
            if width > height:
                new_width = 800
                new_height = int(height * (800 / width))
            else:
                new_height = 800
                new_width = int(width * (800 / height))
            
            image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
        
        return image

class PokemonAnalyzer:
    """Pokemon card specific analysis"""
    
    def __init__(self):
        self.issues = []
    
    def analyze_blue_border(self, image: np.ndarray) -> float:
        """Analyze Pokemon's characteristic blue border"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        height, width = hsv.shape[:2]
        
        # Create border mask (outer 25 pixels)
        border_mask = np.zeros((height, width), dtype=np.uint8)
        border_width = min(25, min(height, width) // 15)
        
        border_mask[:border_width, :] = 255  # Top
        border_mask[-border_width:, :] = 255  # Bottom
        border_mask[:, :border_width] = 255  # Left
        border_mask[:, -border_width:] = 255  # Right
        
        # Pokemon blue color range (HSV)
        lower_blue = np.array([200, 120, 80])
        upper_blue = np.array([220, 255, 200])
        
        # Create blue mask
        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
        
        # Combine with border mask
        border_blue = cv2.bitwise_and(blue_mask, border_mask)
        
        # Calculate blue percentage in border
        blue_pixels = np.sum(border_blue > 0)
        total_border_pixels = np.sum(border_mask > 0)
        
        blue_ratio = blue_pixels / total_border_pixels if total_border_pixels > 0 else 0
        
        if blue_ratio < 0.4:
            self.issues.append("Blue border color doesn't match authentic Pokemon cards")
        
        return min(blue_ratio * 1.5, 1.0)  # Boost good scores
    
    def analyze_print_quality(self, image: np.ndarray) -> float:
        """Analyze overall print quality and sharpness"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Calculate sharpness using Laplacian variance
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness = laplacian.var()
        
        # Normalize sharpness score
        sharpness_score = min(sharpness / 800.0, 1.0)
        
        if sharpness_score < 0.4:
            self.issues.append("Image appears blurry - possible low-quality print or photo")
        
        return sharpness_score
    
    def analyze_color_saturation(self, image: np.ndarray) -> float:
        """Analyze color saturation levels"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        saturation = hsv[:, :, 1]
        mean_saturation = np.mean(saturation) / 255.0
        
        # Pokemon cards typically have good color saturation
        if mean_saturation < 0.3:
            self.issues.append("Color saturation unusually low for Pokemon cards")
            return mean_saturation / 0.3
        elif mean_saturation > 0.95:
            self.issues.append("Color saturation unnaturally high - possible digital manipulation")
            return (1.0 - mean_saturation) / 0.05
        
        return min(mean_saturation * 1.2, 1.0)
    
    def analyze_border_consistency(self, image: np.ndarray) -> float:
        """Analyze border thickness and consistency"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        height, width = edges.shape
        border_width = min(30, min(height, width) // 12)
        
        # Sample border regions
        top_border = edges[:border_width, :]
        bottom_border = edges[-border_width:, :]
        left_border = edges[:, :border_width]
        right_border = edges[:, -border_width:]
        
        # Calculate edge density for each border
        densities = [
            np.sum(top_border) / top_border.size,
            np.sum(bottom_border) / bottom_border.size,
            np.sum(left_border) / left_border.size,
            np.sum(right_border) / right_border.size
        ]
        
        # Check consistency
        mean_density = np.mean(densities)
        std_density = np.std(densities)
        
        if mean_density > 0:
            consistency = 1.0 - (std_density / mean_density)
        else:
            consistency = 0.5
        
        if consistency < 0.7:
            self.issues.append("Border thickness appears inconsistent")
        
        return max(consistency, 0.0)
    
    def analyze_aspect_ratio(self, image: np.ndarray) -> float:
        """Check if image has correct Pokemon card aspect ratio"""
        height, width = image.shape[:2]
        actual_ratio = width / height
        
        # Pokemon cards: 63mm x 88mm = 0.716 ratio
        expected_ratio = 0.716
        ratio_diff = abs(actual_ratio - expected_ratio) / expected_ratio
        
        ratio_score = max(0.0, 1.0 - ratio_diff * 2)
        
        if ratio_score < 0.6:
            self.issues.append(f"Card dimensions don't match Pokemon card proportions (got {actual_ratio:.3f}, expected ~{expected_ratio:.3f})")
        
        return ratio_score
    
    def analyze_energy_symbols(self, image: np.ndarray) -> float:
        """Look for energy symbols (simplified detection)"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Look for circular shapes that could be energy symbols
        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, dp=1, minDist=30,
            param1=50, param2=30, minRadius=8, maxRadius=40
        )
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            # Score based on reasonable number of energy symbols
            energy_score = min(len(circles) / 6.0, 1.0)
        else:
            energy_score = 0.3  # Neutral score if no circles detected
        
        return energy_score
    
    def analyze_card(self, image: np.ndarray, debug_mode: bool = False) -> Dict:
        """Perform complete Pokemon card analysis"""
        # Reset issues for new analysis
        self.issues = []
        
        # Preprocess image
        processed_image = ImageProcessor.preprocess_image(image)
        
        # Run all analyses
        scores = {
            'blue_border': self.analyze_blue_border(processed_image),
            'print_quality': self.analyze_print_quality(processed_image),
            'color_saturation': self.analyze_color_saturation(processed_image),
            'border_consistency': self.analyze_border_consistency(processed_image),
            'aspect_ratio': self.analyze_aspect_ratio(processed_image),
            'energy_symbols': self.analyze_energy_symbols(processed_image)
        }
        
        # Define weights for different factors
        weights = {
            'blue_border': 0.25,      # Most important for Pokemon
            'print_quality': 0.20,
            'color_saturation': 0.20,
            'border_consistency': 0.15,
            'aspect_ratio': 0.15,
            'energy_symbols': 0.05
        }
        
        # Calculate weighted confidence score
        confidence = sum(scores[key] * weights[key] for key in scores.keys())
        
        # Determine result
        if confidence >= 0.75:
            result = "authentic"
            suggested_action = "Card appears authentic based on visual analysis"
        elif confidence >= 0.55:
            result = "possibly_fake"
            suggested_action = "Some concerns detected. Compare with known authentic Pokemon cards or consult a professional grader"
        else:
            result = "likely_fake"
            suggested_action = "Multiple red flags detected. High probability of counterfeit - avoid purchase"
        
        response = {
            "result": result,
            "confidence_score": round(confidence, 3),
            "issues_detected": self.issues,
            "suggested_action": suggested_action
        }
        
        if debug_mode:
            response["debug_info"] = {
                "individual_scores": {k: round(v, 3) for k, v in scores.items()},
                "weights_used": weights,
                "image_dimensions": processed_image.shape[:2]
            }
        
        return response

# Global analyzer instance
pokemon_analyzer = PokemonAnalyzer()

@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """Rate limiting middleware"""
    # Skip rate limiting for health checks and docs
    if request.url.path in ["/health", "/docs", "/openapi.json"]:
        response = await call_next(request)
        return response
    
    # Get client IP
    client_ip = request.client.host
    if hasattr(request, 'headers') and 'x-forwarded-for' in request.headers:
        client_ip = request.headers['x-forwarded-for'].split(',')[0].strip()
    
    # Check rate limit for analysis endpoint
    if request.url.path == "/analyze" and request.method == "POST":
        if not check_rate_limit(client_ip):
            return JSONResponse(
                status_code=429,
                content={
                    "detail": "Daily analysis limit exceeded. You get 1 free analysis per day.",
                    "reset_time": "Midnight UTC"
                }
            )
    
    response = await call_next(request)
    return response

@app.post("/analyze", response_model=CardAnalysisResponse)
async def analyze_pokemon_card(request: CardAnalysisRequest, http_request: Request):
    """Analyze Pokemon card authenticity"""
    try:
        # Get client IP for logging
        client_ip = http_request.client.host
        if hasattr(http_request, 'headers') and 'x-forwarded-for' in http_request.headers:
            client_ip = http_request.headers['x-forwarded-for'].split(',')[0].strip()
        
        # Convert base64 to image
        image = ImageProcessor.base64_to_cv2(request.image_base64)
        
        # Calculate image hash
        image_hash = ImageProcessor.calculate_image_hash(image)
        
        # Check if we've analyzed this exact image before
        conn = sqlite3.connect('pokemon_inspector.db')
        cursor = conn.cursor()
        cursor.execute(
            "SELECT result, confidence_score, issues_detected FROM analysis_history WHERE image_hash = ?",
            (image_hash,)
        )
        cached_result = cursor.fetchone()
        
        if cached_result:
            logger.info(f"Returning cached result for hash: {image_hash}")
            # Don't increment usage for cached results
            return CardAnalysisResponse(
                result=cached_result[0],
                confidence_score=cached_result[1],
                issues_detected=json.loads(cached_result[2]),
                suggested_action="Cached analysis result"
            )
        
        # Perform new analysis
        analysis_result = pokemon_analyzer.analyze_card(image, request.debug_mode)
        
        # Increment usage count
        increment_usage(client_ip)
        
        # Store result in database
        cursor.execute(
            """INSERT INTO analysis_history 
               (ip_address, image_hash, result, confidence_score, issues_detected) 
               VALUES (?, ?, ?, ?, ?)""",
            (client_ip, image_hash, analysis_result["result"], 
             analysis_result["confidence_score"], json.dumps(analysis_result["issues_detected"]))
        )
        conn.commit()
        conn.close()
        
        return CardAnalysisResponse(**analysis_result)
        
    except Exception as e:
        logger.error(f"Error analyzing Pokemon card: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Pokemon Card Inspector",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/stats")
async def get_stats():
    """Get usage statistics"""
    conn = sqlite3.connect('pokemon_inspector.db')
    cursor = conn.cursor()
    
    # Total analyses
    cursor.execute("SELECT COUNT(*) FROM analysis_history")
    total_analyses = cursor.fetchone()[0]
    
    # Results breakdown
    cursor.execute("SELECT result, COUNT(*) FROM analysis_history GROUP BY result")
    results_breakdown = dict(cursor.fetchall())
    
    # Today's usage
    today = datetime.now().date()
    cursor.execute("SELECT COUNT(*) FROM analysis_history WHERE DATE(timestamp) = ?", (today,))
    today_analyses = cursor.fetchone()[0]
    
    conn.close()
    
    return {
        "total_analyses": total_analyses,
        "today_analyses": today_analyses,
        "results_breakdown": results_breakdown,
        "rate_limit": f"{RATE_LIMIT_PER_DAY} per day"
    }

@app.get("/")
async def root():
    """Root endpoint with API info"""
    return {
        "service": "Pokemon Card Inspector API",
        "description": "Analyze Pokemon card authenticity using visual heuristics",
        "version": "1.0.0",
        "endpoints": {
            "analyze": "POST /analyze - Analyze a Pokemon card image",
            "health": "GET /health - Service health check",
            "stats": "GET /stats - Usage statistics",
            "docs": "GET /docs - API documentation"
        },
        "rate_limit": f"{RATE_LIMIT_PER_DAY} free analysis per day"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
