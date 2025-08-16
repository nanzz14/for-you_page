from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import traceback
import pandas as pd
import json

app = FastAPI(title="FlatZ Recommendation API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (HTML, CSS, JS)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Request/Response models
class FeedbackRequest(BaseModel):
    user_id: int
    item_id: int
    feedback_type: str  # 'like', 'dislike', 'not_interested'

class ColdStartRequest(BaseModel):
    user_name: str
    community: str
    age_group: str

class RecommendationResponse(BaseModel):
    item_id: int
    title: str
    description: str
    item_type: str
    price: float
    rating: float
    reason: str
    score: float

# Mock data for testing
MOCK_USERS = [
    {"id": 1, "name": "John Doe", "community": "Maple Grove", "age_group": "26-35"},
    {"id": 2, "name": "Jane Smith", "community": "Downtown Heights", "age_group": "36-45"},
]

MOCK_ITEMS = [
    {"id": 1, "title": "Community Garden Workshop", "description": "Learn about sustainable gardening practices", "item_type": "workshop", "price": 0, "rating": 4.5, "community": "Maple Grove", "tags": "gardening,community,outdoor"},
    {"id": 2, "title": "Yoga in the Park", "description": "Join us for morning yoga sessions", "item_type": "class", "price": 10, "rating": 4.8, "community": "Maple Grove", "tags": "fitness,yoga,health"},
    {"id": 3, "title": "Tech Meetup", "description": "Network with local tech professionals", "item_type": "meetup", "price": 5, "rating": 4.2, "community": "Downtown Heights", "tags": "technology,networking,career"},
    {"id": 4, "title": "Art & Craft Session", "description": "Express your creativity with local artists", "item_type": "workshop", "price": 15, "rating": 4.6, "community": "Downtown Heights", "tags": "art,creativity,crafts"},
]

# Load actual data from CSV files
def load_csv_data():
    """Load actual data from CSV files"""
    try:
        users_df = pd.read_csv('users.csv')
        items_df = pd.read_csv('items.csv')
        
        # Convert users to list of dicts
        users = []
        for _, row in users_df.iterrows():
            users.append({
                "id": row['user_id'],
                "name": row['name'],
                "community": row['community'],
                "age_group": row['age_group']
            })
        
        # Convert items to list of dicts
        items = []
        for _, row in items_df.iterrows():
            items.append({
                "id": row['item_id'],
                "title": row['title'],
                "description": row['description'],
                "item_type": row['item_type'],
                "price": row['price'] if pd.notna(row['price']) else 0,
                "rating": row['rating'] if pd.notna(row['rating']) else 0,
                "community": row['community'],
                "tags": row['tags'] if pd.notna(row['tags']) else "community"
            })
        
        return users, items
    except Exception as e:
        print(f"Error loading CSV data: {e}")
        print("Falling back to mock data...")
        return MOCK_USERS, MOCK_ITEMS

# Load actual data
USERS, ITEMS = load_csv_data()

def get_age_group_interests(age_group):
    """Get typical interests for an age group"""
    age_interests = {
        '18-25': ['gaming', 'music', 'fitness', 'technology', 'social media', 'sports', 'art', 'travel'],
        '26-35': ['fitness', 'cooking', 'parenting', 'career', 'networking', 'outdoor activities', 'music', 'art'],
        '36-45': ['parenting', 'fitness', 'cooking', 'gardening', 'community service', 'outdoor activities', 'art', 'music'],
        '46-55': ['gardening', 'cooking', 'community service', 'outdoor activities', 'art', 'music', 'volunteer work', 'health'],
        '56+': ['gardening', 'community service', 'volunteer work', 'art', 'music', 'reading', 'health', 'social activities']
    }
    return age_interests.get(age_group, ['community', 'social activities', 'health', 'art'])

@app.get("/")
async def root():
    """Serve the main page"""
    return {"message": "FlatZ Recommendation API is running!"}

@app.get("/v1/reco/homefeed/{user_name}")
async def get_homefeed(user_name: str, limit: int = 10):
    """Get personalized recommendations for user"""
    try:
        # Check if user exists
        user = next((u for u in USERS if u["name"].lower() == user_name.lower()), None)
        
        if not user:
            raise HTTPException(status_code=404, detail=f"User '{user_name}' not found")
        
        # Get items for user's community
        community_items = [item for item in ITEMS if item["community"] == user["community"]]
        
        # Simple recommendation logic
        recommendations = []
        for item in community_items[:limit]:
            recommendations.append({
                "item_id": item["id"],
                "title": item["title"],
                "description": item["description"],
                "item_type": item["item_type"],
                "price": item["price"],
                "rating": item["rating"],
                "reason": f"Popular in {user['community']}",
                "score": item["rating"] / 5.0
            })
        
        return {
            "user_name": user["name"],
            "user_community": user["community"],
            "recommendations": recommendations,
            "total_count": len(recommendations)
        }
        
    except HTTPException:
        # Re-raise HTTP exceptions (like 404) without wrapping them
        raise
    except Exception as e:
        print(f"Error in homefeed: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/v1/reco/coldstart")
async def get_coldstart_recommendations(coldstart: ColdStartRequest, limit: int = 10):
    """Get cold start recommendations for new users"""
    try:
        # Get items from the specified community
        community_items = [item for item in ITEMS if item["community"] == coldstart.community]
        
        if not community_items:
            raise HTTPException(status_code=404, detail=f"No items found for community: {coldstart.community}")
        
        # Get age group interests
        age_interests = get_age_group_interests(coldstart.age_group)
        
        # Score items based on age group relevance
        scored_items = []
        for item in community_items:
            item_tags = [tag.strip().lower() for tag in item["tags"].split(",")]
            age_relevance = sum(1 for interest in age_interests if interest.lower() in item_tags)
            age_score = age_relevance / len(age_interests) if age_interests else 0.5
            
            # Combined score
            combined_score = (age_score * 0.6 + (item["rating"] / 5.0) * 0.4)
            
            # Find matching interest for reason
            matched_interest = None
            for interest in age_interests:
                if interest.lower() in item_tags:
                    matched_interest = interest
                    break
            
            reason_str = f"Popular in {coldstart.community} and matches {coldstart.age_group} interests"
            if matched_interest:
                reason_str = f"Popular in {coldstart.community} and matches your age group's interest in {matched_interest}"
            
            scored_items.append({
                "item_id": item["id"],
                "title": item["title"],
                "description": item["description"],
                "item_type": item["item_type"],
                "price": item["price"],
                "rating": item["rating"],
                "reason": reason_str,
                "score": combined_score
            })
        
        # Sort by score and limit
        scored_items.sort(key=lambda x: x["score"], reverse=True)
        recommendations = scored_items[:limit]
        
        return {
            "user_name": coldstart.user_name,
            "user_community": coldstart.community,
            "recommendations": recommendations,
            "total_count": len(recommendations),
            "is_cold_start": True
        }
        
    except Exception as e:
        print(f"Error in coldstart: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/v1/reco/feedback")
async def submit_feedback(feedback: FeedbackRequest):
    """Submit user feedback on recommendations"""
    try:
        print(f"Feedback received: User {feedback.user_id} gave {feedback.feedback_type} to item {feedback.item_id}")
        
        return {
            "status": "success", 
            "message": f"Feedback '{feedback.feedback_type}' recorded for item {feedback.item_id}"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error recording feedback: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "FlatZ Recommendations"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
