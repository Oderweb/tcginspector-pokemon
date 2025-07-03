import requests
import base64
from PIL import Image, ImageDraw
import io

def create_test_pokemon_card():
    """Create a simple test image that looks like a Pokemon card"""
    # Create image with Pokemon-like proportions (63mm x 88mm ratio)
    img = Image.new('RGB', (400, 560), color='white')
    draw = ImageDraw.Draw(img)
    
    # Add blue border (Pokemon's signature feature)
    draw.rectangle([0, 0, 399, 559], outline='#0066CC', width=15)
    
    # Add inner content area
    draw.rectangle([20, 20, 379, 539], outline='black', width=2)
    
    # Add some Pokemon-like text
    draw.text((50, 50), "PIKACHU", fill='black')
    draw.text((50, 100), "Lightning Pokemon", fill='gray')
    draw.text((50, 150), "HP 60", fill='red')
    
    # Add some circular areas (simulate energy symbols)
    draw.ellipse([300, 50, 330, 80], outline='yellow', width=3)
    draw.ellipse([340, 50, 370, 80], outline='yellow', width=3)
    
    # Convert to base64
    buffer = io.BytesIO()
    img.save(buffer, format='JPEG')
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/jpeg;base64,{img_str}"

def test_api():
    """Test the Pokemon card analysis API"""
    base64_image = create_test_pokemon_card()
    
    print("🔍 Testing Pokemon Card API...")
    print("📊 Created test card with blue border and energy symbols")
    
    try:
        response = requests.post("http://localhost:8000/analyze", json={
            "image_base64": base64_image,
            "debug_mode": True
        })
        
        print(f"\n📡 Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ API Test Successful!")
            print(f"🎯 Result: {result['result']}")
            print(f"📈 Confidence: {result['confidence_score']}")
            print(f"⚠️  Issues Detected: {len(result['issues_detected'])}")
            
            for issue in result['issues_detected']:
                print(f"   - {issue}")
            
            print(f"💡 Suggested Action: {result['suggested_action']}")
            
            if 'debug_info' in result:
                print(f"\n🔧 Debug Info:")
                scores = result['debug_info']['individual_scores']
                for score_name, score_value in scores.items():
                    print(f"   {score_name}: {score_value}")
                    
        elif response.status_code == 429:
            print("🚫 Rate limit exceeded - you've used your 1 daily analysis")
            print("💡 Wait until midnight UTC or test with a different IP")
        else:
            print(f"❌ API Error: {response.status_code}")
            print(f"📝 Error details: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("❌ Connection Error: Make sure your API is running on localhost:8000")
        print("💡 Run 'python main.py' in another terminal first")
    except Exception as e:
        print(f"❌ Test Error: {e}")

def test_rate_limiting():
    """Test the rate limiting by making a second request"""
    print("\n🔄 Testing rate limiting with second request...")
    
    # Create a different test image
    img = Image.new('RGB', (400, 560), color='lightgray')
    buffer = io.BytesIO()
    img.save(buffer, format='JPEG')
    img_str = base64.b64encode(buffer.getvalue()).decode()
    base64_image = f"data:image/jpeg;base64,{img_str}"
    
    try:
        response = requests.post("http://localhost:8000/analyze", json={
            "image_base64": base64_image
        })
        
        if response.status_code == 429:
            print("✅ Rate limiting works! Second request blocked")
            print(f"📝 Response: {response.json()}")
        elif response.status_code == 200:
            print("ℹ️  Second request allowed (maybe cached or different IP)")
        else:
            print(f"🤔 Unexpected response: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Rate limit test error: {e}")

if __name__ == "__main__":
    test_api()
    test_rate_limiting()