// Listen for messages from background script
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.action === "analyzeImage") {
    analyzeImage(request.imageUrl);
  }
});

async function analyzeImage(imageUrl) {
  try {
    // Convert image to base64
    const base64Image = await imageToBase64(imageUrl);
    
    // Call your API
    const response = await fetch('https://pokemon-inspector.onrender.com/analyze', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        image_base64: base64Image,
        debug_mode: false
      })
    });
    
    const result = await response.json();
    
    if (response.status === 429) {
      showNotification("Daily limit reached", "You've used your daily analyses!");
      return;
    }
    
    // Show result to user
    showAnalysisResult(result);
    
  } catch (error) {
    showNotification("Error", "Failed to analyze card: " + error.message);
  }
}

function imageToBase64(imageUrl) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.onload = function() {
      const canvas = document.createElement('canvas');
      const ctx = canvas.getContext('2d');
      canvas.width = img.width;
      canvas.height = img.height;
      ctx.drawImage(img, 0, 0);
      const dataURL = canvas.toDataURL('image/jpeg');
      resolve(dataURL);
    };
    img.onerror = reject;
    img.src = imageUrl;
  });
}

function showAnalysisResult(result) {
  // Create a styled notification overlay
  const overlay = document.createElement('div');
  overlay.style.cssText = `
    position: fixed;
    top: 20px;
    right: 20px;
    background: white;
    border: 2px solid #0066cc;
    border-radius: 10px;
    padding: 20px;
    max-width: 300px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    z-index: 10000;
    font-family: Arial, sans-serif;
  `;
  
  const resultColor = result.result === 'authentic' ? '#28a745' : 
                     result.result === 'possibly_fake' ? '#ffc107' : '#dc3545';

  // Create feedback section
  const feedbackSection = `
    <div style="margin-top: 15px; border-top: 1px solid #eee; padding-top: 15px;">
      <div style="font-size: 12px; color: #666; margin-bottom: 10px;">
        Was this analysis correct?
      </div>
      <button onclick="window.submitFeedback('${result.analysis_id}', 'authentic')" 
              style="background: #28a745; color: white; border: none; padding: 4px 8px; margin: 2px; border-radius: 3px; font-size: 11px; cursor: pointer;">✓ Authentic</button>
      <button onclick="window.submitFeedback('${result.analysis_id}', 'likely_fake')" 
              style="background: #dc3545; color: white; border: none; padding: 4px 8px; margin: 2px; border-radius: 3px; font-size: 11px; cursor: pointer;">✗ Fake</button>
      <button onclick="window.submitFeedback('${result.analysis_id}', 'possibly_fake')" 
              style="background: #ffc107; color: black; border: none; padding: 4px 8px; margin: 2px; border-radius: 3px; font-size: 11px; cursor: pointer;">? Unsure</button>
    </div>
  `;
  
  // Build the complete overlay content
  overlay.innerHTML = `
    <div style="border-bottom: 1px solid #eee; padding-bottom: 10px; margin-bottom: 10px;">
      <h3 style="margin: 0; color: #0066cc;">🔍 Pokemon Card Analysis</h3>
    </div>
    <div style="margin-bottom: 15px;">
      <strong style="color: ${resultColor};">${result.result.toUpperCase()}</strong>
      <div style="font-size: 14px; color: #666;">
        Confidence: ${(result.confidence_score * 100).toFixed(1)}%
      </div>
    </div>
    ${result.issues_detected.length > 0 ? `
      <div style="margin-bottom: 15px;">
        <strong>Issues:</strong>
        <ul style="margin: 5px 0; padding-left: 20px; font-size: 12px;">
          ${result.issues_detected.map(issue => `<li>${issue}</li>`).join('')}
        </ul>
      </div>
    ` : ''}
    <div style="font-size: 12px; color: #666; margin-bottom: 15px;">
      ${result.suggested_action}
    </div>
    ${result.analysis_id ? feedbackSection : ''}
    <button onclick="this.parentElement.remove()" style="
      background: #0066cc; 
      color: white; 
      border: none; 
      padding: 8px 16px; 
      border-radius: 5px; 
      cursor: pointer;
      float: right;
      margin-top: 10px;
    ">Close</button>
  `;

  // Define the feedback function globally
  window.submitFeedback = async function(analysisId, actualResult) {
    try {
      const response = await fetch('https://pokemon-inspector.onrender.com/feedback', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          analysis_id: analysisId,
          actual_result: actualResult,
          user_confidence: 4  // High confidence
        })
      });
      
      if (response.ok) {
        // Replace feedback buttons with thank you message
        const feedbackDiv = overlay.querySelector('div[style*="border-top"]');
        if (feedbackDiv) {
          feedbackDiv.innerHTML = '<div style="font-size: 12px; color: #28a745; text-align: center;">✓ Thank you for your feedback!</div>';
        }
      } else {
        console.error('Feedback submission failed:', response.status);
      }
    } catch (error) {
      console.error('Feedback error:', error);
    }
  };
  
  // Add overlay to page
  document.body.appendChild(overlay);
  
  // Auto-remove after 15 seconds
  setTimeout(() => {
    if (overlay.parentElement) {
      overlay.remove();
    }
  }, 15000);
}

function showNotification(title, message) {
  // Simple notification for errors
  const notification = document.createElement('div');
  notification.style.cssText = `
    position: fixed;
    top: 20px;
    right: 20px;
    background: #dc3545;
    color: white;
    padding: 15px;
    border-radius: 5px;
    z-index: 10000;
    max-width: 300px;
    font-family: Arial, sans-serif;
  `;
  notification.innerHTML = `<strong>${title}</strong><br>${message}`;
  document.body.appendChild(notification);
  
  setTimeout(() => notification.remove(), 5000);
}
