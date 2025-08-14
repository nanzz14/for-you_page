import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sentence_transformers import SentenceTransformer
import numpy as np
import json
from surprise import Reader, Dataset, SVD

# Database setup
engine = create_engine('sqlite:///flatZ.db')
Session = sessionmaker(bind=engine)
session = Session()

# Load AI model for embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load and train collaborative filtering model (run once)
interactions = pd.read_sql('SELECT user_id, item_id, action FROM interactions', engine)
# Map action to rating (like=5, view=3)
interactions['rating'] = interactions['action'].map({'like': 5, 'view': 3})
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(interactions[['user_id', 'item_id', 'rating']], reader)
trainset = data.build_full_trainset()
algo = SVD()
algo.fit(trainset)

def get_candidates(user_id):
    try:
        # Step 1: Get user data
        user = pd.read_sql(f'SELECT * FROM users WHERE id = {user_id}', engine)
        if user.empty:
            print(f"No user found with id {user_id}")
            return []
        community = user['community'].iloc[0]
        interests = user['interests'].iloc[0]

        # Step 2: Content-based candidates
        interest_embedding = model.encode(interests).tolist()  # Embed user interests
        items = pd.read_sql(f'SELECT * FROM items WHERE community = "{community}"', engine)
        content_candidates = []
        for _, item in items.iterrows():
            item_embedding = json.loads(item['embedding'])  # Load item embedding
            # Compute cosine similarity
            similarity = np.dot(interest_embedding, item_embedding) / (
                np.linalg.norm(interest_embedding) * np.linalg.norm(item_embedding)
            )
            content_candidates.append({
                'item_id': item['id'],
                'title': item['title'],
                'similarity': float(similarity),
                'reason': f"Matches your interest in {interests.split(',')[0]}"
            })
        # Sort by similarity, take top 5
        content_candidates = sorted(content_candidates, key=lambda x: x['similarity'], reverse=True)[:5]

        # Step 3: Collaborative filtering candidates
        # Get user's interacted items to avoid recommending seen ones
        user_interactions = pd.read_sql(f'SELECT item_id FROM interactions WHERE user_id = {user_id}', engine)
        interacted_items = set(user_interactions['item_id'])
        # Predict for unseen items (all items 1-150)
        unseen_items = [i for i in range(1, 151) if i not in interacted_items]
        if unseen_items:
            testset = [(user_id, iid, 4) for iid in unseen_items]  # Dummy rating
            predictions = algo.test(testset)
            cf_candidates = sorted(predictions, key=lambda x: x.est, reverse=True)[:5]
            cf_candidates = [
                {
                    'item_id': pred.iid,
                    'title': pd.read_sql(f'SELECT title FROM items WHERE id = {pred.iid}', engine)['title'].iloc[0],
                    'similarity': pred.est,
                    'reason': 'Recommended based on similar users'
                }
                for pred in cf_candidates
            ]
        else:
            cf_candidates = []  # Cold-start: No interactions, skip

        # Step 4: Recency-based candidates
        recent_items = pd.read_sql(
            f'SELECT * FROM items WHERE community = "{community}" ORDER BY creation_date DESC LIMIT 5',
            engine
        )
        recency_candidates = [
            {
                'item_id': row['id'],
                'title': row['title'],
                'similarity': 0.0,
                'reason': 'Recent in your community'
            }
            for _, row in recent_items.iterrows()
        ]

        # Step 5: Combine and deduplicate
        candidates = content_candidates + cf_candidates + recency_candidates
        seen = set()
        unique_candidates = []
        for cand in candidates:
            if cand['item_id'] not in seen:
                unique_candidates.append(cand)
                seen.add(cand['item_id'])

        return unique_candidates[:10]  # Return up to 10 candidates
    except Exception as e:
        print(f"Error: {e}")
        return []
    finally:
        session.close()

# Test the function
if __name__ == "__main__":
    candidates = get_candidates(1)  # Test with user_id=1
    for cand in candidates:
        print(f"Item {cand['item_id']}: {cand['title']} - {cand['reason']} (Similarity/Predicted Score: {cand['similarity']:.2f})")