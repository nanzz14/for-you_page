import pandas as pd
import pickle
from sqlalchemy import create_engine
from surprise import Reader, Dataset, SVD
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import xgboost as xgb
import json
from datetime import datetime
from sentence_transformers import SentenceTransformer
import numpy as np

def train_and_save_models(db_path='sqlite:///flatZ_enhanced.db'):
    """Train and save collaborative filtering and ranking models."""
    engine = create_engine(db_path)

    # Train Collaborative Filtering Model
    print("🔄 Training Collaborative Filtering model...")
    interactions = pd.read_sql('SELECT user_id, item_id, action, rating FROM interactions', engine)
    action_ratings = {'view': 2.0, 'like': 4.0, 'bookmark': 4.5, 'share': 4.5, 'join': 5.0, 'comment': 3.5}
    interactions['final_rating'] = interactions.apply(
        lambda x: x['rating'] if pd.notna(x['rating']) else action_ratings.get(x['action'], 3.0), axis=1
    )
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(interactions[['user_id', 'item_id', 'final_rating']], reader)
    trainset = data.build_full_trainset()
    cf_algo = SVD(n_factors=50, n_epochs=30, random_state=42)
    cf_algo.fit(trainset)

    # Save Collaborative Filtering Model
    with open("cf_model.pkl", "wb") as f:
        pickle.dump(cf_algo, f)
    print("✅ Collaborative Filtering model saved as 'cf_model.pkl'.")

    # Train Ranking Model
    print("🔄 Training Ranking model...")
    interactions = pd.read_sql("""
        SELECT i.user_id, i.item_id, i.action,
               CASE 
                   WHEN i.action = 'join' THEN 5.0
                   WHEN i.action = 'like' THEN 4.0
                   WHEN i.action = 'bookmark' THEN 4.5
                   WHEN i.action = 'share' THEN 4.5
                   WHEN i.action = 'comment' THEN 3.5
                   ELSE 2.0 
               END as rating
        FROM interactions i
        LIMIT 1000
    """, engine)

    training_data = []
    for _, row in interactions.iterrows():
        features = extract_features(engine, row['user_id'], row['item_id'])
        features['target'] = row['rating']
        training_data.append(features)

    df = pd.DataFrame(training_data)
    feature_cols = [col for col in df.columns if col != 'target']
    X = df[feature_cols].fillna(0)
    y = df['target']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    ranking_model = xgb.XGBRegressor(n_estimators=50, max_depth=4, random_state=42)
    ranking_model.fit(X_train, y_train)

    # Save Ranking Model and Scaler
    with open("ranking_model.pkl", "wb") as f:
        pickle.dump(ranking_model, f)
    with open("scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    print("✅ Ranking model and scaler saved as 'ranking_model.pkl' and 'scaler.pkl'.")

def extract_features(engine, user_id, item_id):
    """Extract features for ranking model."""
    try:
        user = pd.read_sql(f'SELECT * FROM users WHERE id = {user_id}', engine).iloc[0]
        item = pd.read_sql(f'SELECT * FROM items WHERE id = {item_id}', engine).iloc[0]

        features = {}

        # Content similarity
        user_interests = [i.strip() for i in user.interests.split(',')]
        item_embedding = json.loads(item.embedding)
        interest_embeddings = [SentenceTransformer('all-MiniLM-L6-v2').encode(interest) for interest in user_interests]
        user_embedding = np.mean(interest_embeddings, axis=0)
        similarity = np.dot(user_embedding, item_embedding) / (
            np.linalg.norm(user_embedding) * np.linalg.norm(item_embedding)
        )
        features['content_similarity'] = float(similarity)

        # Tag overlap
        item_tags = set(tag.strip().lower() for tag in item.tags.split(','))
        user_tags = set(interest.strip().lower() for interest in user_interests)
        features['tag_overlap'] = len(item_tags.intersection(user_tags)) / len(item_tags) if item_tags else 0

        # Community match
        features['community_match'] = 1.0 if user.community == item.community else 0.0

        # Recency
        creation_date = datetime.fromisoformat(item.creation_date)
        days_old = (datetime.now() - creation_date).days
        features['recency_score'] = max(0, 1 - (days_old / 30))

        # Popularity
        popularity = pd.read_sql(f'SELECT COUNT(*) as count FROM interactions WHERE item_id = {item_id}', engine).iloc[0].count
        features['popularity_score'] = min(1.0, popularity / 10)

        # Price preference
        features['price_match'] = 1.0 if item.price == 0 else 0.5

        return features

    except Exception:
        return {'content_similarity': 0.5, 'tag_overlap': 0.0, 'community_match': 0.0, 
                'recency_score': 0.5, 'popularity_score': 0.5, 'price_match': 0.5}

if __name__ == "__main__":
    train_and_save_models()