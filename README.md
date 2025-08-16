# For- You Recommendation System

A personalized recommendation system for community events and activities using collaborative filtering, content-based filtering, and ML ranking. Provides both CLI and REST API interfaces.

## Features

- **Multi-Strategy Recommendations**: Content-based, collaborative filtering, and XGBoost ranking
- **Semantic Understanding**: Uses sentence transformers for content similarity
- **Cold-Start Handling**: Fallbacks for new users and items
- **FastAPI Web Interface**: REST API with automatic documentation
- **Rich Dataset**: 500 users, 1000 items, 3000 interactions across 10 communities

## Quick Start

1. **Setup Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Generate Data & Train Models**
   ```bash
   python data.py      # Creates database and sample data
   python training.py  # Trains and saves ML models
   ```

3. **Run the System**
   ```bash
   # CLI Interface
   python recommendation.py
   
   # Web API (visit http://localhost:8000/docs for API docs)
   python main.py
   ```

## Usage

### Command Line
```python
from recommendation import ForYouPageSystem
system = ForYouPageSystem()
system.display_for_you_page("John Doe")
```

### REST API
```bash
# Get recommendations
curl "http://localhost:8000/v1/reco/homefeed/John%20Doe?limit=5"

# Submit feedback
curl -X POST "http://localhost:8000/v1/reco/feedback" \
     -H "Content-Type: application/json" \
     -d '{"user_id": 1, "item_id": 5, "feedback_type": "like"}'
```

## Project Structure

```
├── data.py                  # Dataset generation
├── training.py              # Model training
├── recommendation.py        # Core system (CLI)
├── main.py                  # FastAPI server
├── requirements.txt         # Dependencies
├── flatZ_enhanced.db       # SQLite database
└── *.pkl                   # Trained models
```

## Dataset

- **10 Communities**: Downtown Heights, Sunset Valley, Oakwood Gardens, etc.
- **25+ Interests**: Fitness, pets, gardening, sports, cooking, arts, etc.
- **Item Types**: Events, services, workshops, classes, meetups

## Architecture

1. **Candidate Generation**: Content-based + collaborative filtering + recency + popularity
2. **Feature Extraction**: Content similarity, tag overlap, community match, recency, popularity
3. **ML Ranking**: XGBoost model scores and ranks candidates
4. **Personalization**: Adapts to user interests and behavior

## API Endpoints

- `GET /v1/reco/homefeed/{user_name}` - Get personalized recommendations
- `POST /v1/reco/feedback` - Submit user feedback
- `GET /health` - Health check

## Dependencies

Core: `fastapi`, `pandas`, `numpy`, `sqlalchemy`, `sentence-transformers`, `surprise`, `xgboost`, `scikit-learn`

## Troubleshooting

- **Models not found**: Run `python training.py` first
- **Database missing**: Run `python data.py` first
- **Port issues**: Use `uvicorn main:app --port 8001`