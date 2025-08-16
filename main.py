from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from recommendation import ForYouPageSystem
import traceback
import pandas as pd

app = FastAPI(title="FlatZ Recommendation API", version="1.0.0")

# Allow frontend to call API (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (HTML, CSS, JS)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize systems
recommendation_system = ForYouPageSystem()

# Request/Response models
class FeedbackRequest(BaseModel):
    user_id: int
    item_id: int
    feedback_type: str  # 'like', 'dislike', 'not_interested'

class RecommendationResponse(BaseModel):
    item_id: int
    title: str
    description: str
    item_type: str
    price: float
    rating: float
    reason: str
    score: float

@app.get("/")
async def root():
    """Serve the main page"""
    return {"message": "FlatZ Recommendation API is running!"}

@app.get("/v1/reco/homefeed/{user_name}")
async def get_homefeed(user_name: str, limit: int = 10):
    """Get personalized recommendations for user"""
    try:
        # Get recommendations from your existing system
        result = recommendation_system.get_for_you_page(user_name)
        
        if not result:
            raise HTTPException(status_code=404, detail=f"User '{user_name}' not found")
        
        user = result['user']
        recommendations = result['recommendations']
        
        # Debug logging
        print(f"🔍 Debug: User {user_name} found")
        print(f"🔍 Debug: User community: {user.community}")
        print(f"🔍 Debug: User interests: {user.interests}")
        print(f"🔍 Debug: Raw recommendations count: {len(recommendations)}")
        
        # No safety filtering needed - your recommendation system is already safe
        safe_recommendations = recommendations
        print(f"🔍 Debug: After safety filter: {len(safe_recommendations)}")
        
        # Limit results
        safe_recommendations = safe_recommendations[:limit]
        
        # Format response
        formatted_recs = []
        for rec in safe_recommendations:
            # Clean NaN values before JSON serialization
            clean_price = rec['price'] if pd.notna(rec['price']) else 0.0
            clean_rating = rec['rating'] if pd.notna(rec['rating']) else 0.0
            clean_score = rec.get('ranking_score', rec['score'])
            if pd.isna(clean_score):
                clean_score = 0.0
            
            formatted_recs.append({
                "item_id": rec['item_id'],
                "title": rec['title'],
                "description": rec['description'][:150] + "..." if len(rec['description']) > 150 else rec['description'],
                "item_type": rec['item_type'],
                "price": clean_price,
                "rating": clean_rating,
                "reason": rec['reason'],
                "score": clean_score
            })
        
        print(f"🔍 Debug: Final formatted recommendations: {len(formatted_recs)}")
        
        return {
            "user_name": user['name'],
            "user_community": user.community,
            "recommendations": formatted_recs,
            "total_count": len(formatted_recs)
        }
        
    except Exception as e:
        print(f"Error in homefeed: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/v1/reco/feedback")
async def submit_feedback(feedback: FeedbackRequest):
    """Submit user feedback on recommendations"""
    try:
        # Here you would log feedback to database
        # For now, we'll just return success
        
        # In a real system, you'd do:
        # 1. Insert feedback into database
        # 2. Update user preferences
        # 3. Potentially retrain models
        
        print(f"Feedback received: User {feedback.user_id} gave {feedback.feedback_type} to item {feedback.item_id}")
        
        return {
            "status": "success", 
            "message": f"Feedback '{feedback.feedback_type}' recorded for item {feedback.item_id}"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error recording feedback: {str(e)}")

@app.get("/v1/reco/explanations/{user_name}/{item_id}")
async def get_explanation(user_name: str, item_id: int):
    """Get detailed explanation for why item was recommended"""
    try:
        # Get user
        user = recommendation_system.get_user_by_name(user_name)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        # Extract detailed features
        features = recommendation_system._extract_features(user.id, item_id)
        
        explanation = {
            "item_id": item_id,
            "user_name": user_name,
            "explanation_factors": {
                "content_similarity": f"{features['content_similarity']:.2f} - How well this matches your interests",
                "community_match": "Yes" if features['community_match'] > 0 else "No",
                "recency_score": f"{features['recency_score']:.2f} - How recent this item is",
                "popularity_score": f"{features['popularity_score']:.2f} - How popular with other users",
                "tag_overlap": f"{features['tag_overlap']:.2f} - Tag similarity with your interests"
            }
        }
        
        return explanation
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating explanation: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "FlatZ Recommendations"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)