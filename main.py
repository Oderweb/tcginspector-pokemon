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
    analysis_id: Optional[str] = None
    debug_info: Optional[Dict] = None

class FeedbackRequest(BaseModel):
    analysis_id: str  # The ID of the analysis being corrected
    actual_result: str  # "authentic", "possibly_fake", or "likely_fake" 
    user_confidence: int  # 1-5 scale how confident user is
    notes: Optional[str] = None  # Optional user comments

class FeedbackResponse(BaseModel):
    message: str
    feedback_id: int

class TrainingDataResponse(BaseModel):
    total_examples: int
    authentic_count: int
    fake_count: int
    examples: List[Dict]

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
    
    # Feedback table for training data collection
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            analysis_id INTEGER,
            image_hash TEXT,
            original_prediction TEXT,
            original_confidence REAL,
            actual_result TEXT,
            user_confidence INTEGER,
            notes TEXT,
            ip_address TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (analysis_id) REFERENCES analysis_history (id)
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
    """Pokemon card specific analysis with improved heuristics"""
    
    def __init__(self):
        self.issues = []
    
    def analyze_blue_border(self, image: np.ndarray) -> float:
        """Analyze Pokemon's characteristic blue border - IMPROVED"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        height, width = hsv.shape[:2]
        
        # Create border mask (outer pixels)
        border_mask = np.zeros((height, width), dtype=np.uint8)
        border_width = min(25, min(height, width) // 15)
        
        border_mask[:border_width, :] = 255  # Top
        border_mask[-border_width:, :] = 255  # Bottom
        border_mask[:, :border_width] = 255  # Left
        border_mask[:, -border_width:] = 255  # Right
        
        # EXPANDED Pokemon blue color range (HSV) - more permissive
        lower_blue1 = np.array([200, 80, 60])   # Darker blue
        upper_blue1 = np.array([230, 255, 255]) # Lighter blue
        
        # Also check for darker blue variants
        lower_blue2 = np.array([180, 50, 40])
        upper_blue2 = np.array([200, 200, 200])
        
        # Create blue masks
        blue_mask1 = cv2.inRange(hsv, lower_blue1, upper_blue1)
        blue_mask2 = cv2.inRange(hsv, lower_blue2, upper_blue2)
        blue_mask = cv2.bitwise_or(blue_mask1, blue_mask2)
        
        # Combine with border mask
        border_blue = cv2.bitwise_and(blue_mask, border_mask)
        
        # Calculate blue percentage in border
        blue_pixels = np.sum(border_blue > 0)
        total_border_pixels = np.sum(border_mask > 0)
        
        blue_ratio = blue_pixels / total_border_pixels if total_border_pixels > 0 else 0
        
        # MORE LENIENT threshold for blue detection
        if blue_ratio < 0.15:  # Changed from 0.4 to 0.15
            self.issues.append(f"Blue border color doesn't match authentic Pokemon cards (detected: {blue_ratio:.3f})")
        
        # More generous scoring
        return min(blue_ratio * 3.0, 1.0)  # Increased multiplier
    
    def analyze_print_quality(self, image: np.ndarray) -> float:
        """Analyze overall print quality and sharpness"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Calculate sharpness using Laplacian variance
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness = laplacian.var()
        
        # Normalize sharpness score - more lenient
        sharpness_score = min(sharpness / 500.0, 1.0)  # Reduced from 800
        
        if sharpness_score < 0.3:  # Reduced from 0.4
            self.issues.append("Image appears blurry - possible low-quality print or photo")
        
        return sharpness_score
    
    def analyze_color_saturation(self, image: np.ndarray) -> float:
        """Analyze color saturation levels"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        saturation = hsv[:, :, 1]
        mean_saturation = np.mean(saturation) / 255.0
        
        # More lenient Pokemon card saturation check
        if mean_saturation < 0.2:  # Reduced from 0.3
            self.issues.append("Color saturation unusually low for Pokemon cards")
            return mean_saturation / 0.2
        elif mean_saturation > 0.98:  # Increased from 0.95
            self.issues.append("Color saturation unnaturally high - possible digital manipulation")
            return (1.0 - mean_saturation) / 0.02
        
        return min(mean_saturation * 1.5, 1.0)  # More generous
    
    def analyze_border_consistency(self, image: np.ndarray) -> float:
        """Analyze border thickness and consistency - IMPROVED"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Use gentler edge detection
        edges = cv2.Canny(gray, 30, 100)  # Reduced from 50, 150
        
        height, width = edges.shape
        border_width = min(20, min(height, width) // 15)  # Reduced from 30
        
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
        
        # Check consistency - more lenient
        mean_density = np.mean(densities)
        std_density = np.std(densities)
        
        if mean_density > 0:
            consistency = 1.0 - (std_density / mean_density)
        else:
            consistency = 0.8  # Give benefit of doubt
        
        # More lenient threshold
        if consistency < 0.4:  # Changed from 0.7
            self.issues.append(f"Border thickness appears inconsistent (score: {consistency:.3f})")
        
        return max(consistency, 0.3)  # Minimum score of 0.3
    
    def analyze_aspect_ratio(self, image: np.ndarray) -> float:
        """Check if image has correct Pokemon card aspect ratio"""
        height, width = image.shape[:2]
        actual_ratio = width / height
        
        # Pokemon cards: 63mm x 88mm = 0.716 ratio
        expected_ratio = 0.716
        ratio_diff = abs(actual_ratio - expected_ratio) / expected_ratio
        
        # More lenient ratio checking
        ratio_score = max(0.0, 1.0 - ratio_diff * 1.5)  # Reduced from 2.0
        
        if ratio_score < 0.5:  # Reduced from 0.6
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
            energy_score = 0.5  # Neutral score if no circles detected (was 0.3)
        
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
        
        # UPDATED weights - reduce emphasis on problematic checks
        weights = {
            'blue_border': 0.20,        # Reduced from 0.25
            'print_quality': 0.25,      # Increased from 0.20
            'color_saturation': 0.25,   # Increased from 0.20
            'border_consistency': 0.10, # Reduced from 0.15
            'aspect_ratio': 0.15,       # Same
            'energy_symbols': 0.05      # Same
        }
        
        # Calculate weighted confidence score
        confidence = sum(scores[key] * weights[key] for key in scores.keys())
        
        # MORE LENIENT result determination
        if confidence >= 0.65:  # Reduced from 0.75
            result = "authentic"
            suggested_action = "Card appears authentic based on visual analysis"
        elif confidence >= 0.45:  # Reduced from 0.55
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
    if request.url.path in ["/health", "/docs", "/openapi.json", "/", "/stats", "/training-data", "/analysis-accuracy"]:
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
                    "detail": "Daily analysis limit exceeded. You get 100 free analyses per day.",
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
            "SELECT id, result, confidence_score, issues_detected FROM analysis_history WHERE image_hash = ?",
            (image_hash,)
        )
        cached_result = cursor.fetchone()
        
        if cached_result:
            logger.info(f"Returning cached result for hash: {image_hash}")
            # Don't increment usage for cached results
            return CardAnalysisResponse(
                result=cached_result[1],
                confidence_score=cached_result[2],
                issues_detected=json.loads(cached_result[3]),
                suggested_action="Cached analysis result",
                analysis_id=str(cached_result[0])
            )
        
        # Perform new analysis
        analysis_result = pokemon_analyzer.analyze_card(image, request.debug_mode)
        
        # Increment usage count
        increment_usage(client_ip)
        
        # Store result in database and get the ID
        cursor.execute(
            """INSERT INTO analysis_history 
               (ip_address, image_hash, result, confidence_score, issues_detected) 
               VALUES (?, ?, ?, ?, ?)""",
            (client_ip, image_hash, analysis_result["result"], 
             analysis_result["confidence_score"], json.dumps(analysis_result["issues_detected"]))
        )
        
        # Get the analysis ID
        analysis_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        # Add analysis_id to the response
        analysis_result["analysis_id"] = str(analysis_id)
        
        return CardAnalysisResponse(**analysis_result)
        
    except Exception as e:
        logger.error(f"Error analyzing Pokemon card: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(feedback: FeedbackRequest, request: Request):
    """Submit feedback to improve our analysis"""
    try:
        client_ip = request.client.host
        if hasattr(request, 'headers') and 'x-forwarded-for' in request.headers:
            client_ip = request.headers['x-forwarded-for'].split(',')[0].strip()
        
        conn = sqlite3.connect('pokemon_inspector.db')
        cursor = conn.cursor()
        
        # Get original analysis details
        cursor.execute(
            "SELECT image_hash, result, confidence_score FROM analysis_history WHERE id = ?",
            (feedback.analysis_id,)
        )
        original_analysis = cursor.fetchone()
        
        if not original_analysis:
            raise HTTPException(status_code=404, detail="Analysis not found")
        
        image_hash, original_prediction, original_confidence = original_analysis
        
        # Insert feedback
        cursor.execute(
            """INSERT INTO feedback 
               (analysis_id, image_hash, original_prediction, original_confidence, 
                actual_result, user_confidence, notes, ip_address) 
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (feedback.analysis_id, image_hash, original_prediction, original_confidence,
             feedback.actual_result, feedback.user_confidence, feedback.notes, client_ip)
        )
        
        feedback_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        logger.info(f"Feedback received: Analysis {feedback.analysis_id} -> {feedback.actual_result}")
        
        return FeedbackResponse(
            message="Thank you for your feedback! This helps improve our analysis.",
            feedback_id=feedback_id
        )
        
    except Exception as e:
        logger.error(f"Error submitting feedback: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to submit feedback: {str(e)}")

@app.get("/training-data")
async def get_training_data():
    """Get aggregated training data for analysis improvement"""
    try:
        conn = sqlite3.connect('pokemon_inspector.db')
        cursor = conn.cursor()
        
        # Get feedback statistics
        cursor.execute("SELECT COUNT(*) FROM feedback")
        total_feedback = cursor.fetchone()[0]
        
        cursor.execute("SELECT actual_result, COUNT(*) FROM feedback GROUP BY actual_result")
        result_breakdown = dict(cursor.fetchall())
        
        # Get recent feedback examples (anonymized)
        cursor.execute("""
            SELECT f.original_prediction, f.original_confidence, f.actual_result, 
                   f.user_confidence, f.notes, f.timestamp
            FROM feedback f
            ORDER BY f.timestamp DESC
            LIMIT 20
        """)
        
        recent_feedback = []
        for row in cursor.fetchall():
            recent_feedback.append({
                "original_prediction": row[0],
                "original_confidence": row[1],
                "actual_result": row[2],
                "user_confidence": row[3],
                "notes": row[4],
                "timestamp": row[5]
            })
        
        # Calculate accuracy
        cursor.execute("""
            SELECT 
                SUM(CASE WHEN original_prediction = actual_result THEN 1 ELSE 0 END) as correct,
                COUNT(*) as total
            FROM feedback
        """)
        accuracy_data = cursor.fetchone()
        accuracy = (accuracy_data[0] / accuracy_data[1] * 100) if accuracy_data[1] > 0 else 0
        
        conn.close()
        
        return {
            "total_feedback": total_feedback,
            "result_breakdown": result_breakdown,
            "current_accuracy": f"{accuracy:.1f}%",
            "recent_feedback": recent_feedback,
            "authentic_count": result_breakdown.get("authentic", 0),
            "fake_count": result_breakdown.get("likely_fake", 0) + result_breakdown.get("possibly_fake", 0)
        }
        
    except Exception as e:
        logger.error(f"Error getting training data: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get training data: {str(e)}")

@app.get("/analysis-accuracy")
async def get_analysis_accuracy():
    """Get accuracy metrics for our analysis"""
    try:
        conn = sqlite3.connect('pokemon_inspector.db')
        cursor = conn.cursor()
        
        # Overall accuracy
        cursor.execute("""
            SELECT 
                original_prediction,
                actual_result,
                COUNT(*) as count,
                AVG(original_confidence) as avg_confidence
            FROM feedback
            GROUP BY original_prediction, actual_result
            ORDER BY original_prediction, actual_result
        """)
        
        confusion_matrix = []
        for row in cursor.fetchall():
            confusion_matrix.append({
                "predicted": row[0],
                "actual": row[1],
                "count": row[2],
                "avg_confidence": round(row[3], 3)
            })
        
        # Most common mistakes
        cursor.execute("""
            SELECT original_prediction, actual_result, COUNT(*) as mistakes
            FROM feedback
            WHERE original_prediction != actual_result
            GROUP BY original_prediction, actual_result
            ORDER BY mistakes DESC
            LIMIT 10
        """)
        
        common_mistakes = []
        for row in cursor.fetchall():
            common_mistakes.append({
                "predicted": row[0],
                "should_be": row[1],
                "frequency": row[2]
            })
        
        conn.close()
        
        return {
            "confusion_matrix": confusion_matrix,
            "common_mistakes": common_mistakes
        }
        
    except Exception as e:
        logger.error(f"Error getting accuracy data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

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
    
    # Feedback stats
    cursor.execute("SELECT COUNT(*) FROM feedback")
    total_feedback = cursor.fetchone()[0]
    
    conn.close()
    
    return {
        "total_analyses": total_analyses,
        "today_analyses": today_analyses,
        "results_breakdown": results_breakdown,
        "total_feedback": total_feedback,
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
            "feedback": "POST /feedback - Submit feedback on analysis results",
            "training-data": "GET /training-data - View collected training data",
            "analysis-accuracy": "GET /analysis-accuracy - View accuracy metrics",
            "health": "GET /health - Service health check",
            "stats": "GET /stats - Usage statistics",
            "docs": "GET /docs - API documentation"
        },
        "rate_limit": f"{RATE_LIMIT_PER_DAY} free analyses per day"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
