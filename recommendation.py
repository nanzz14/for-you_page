import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sentence_transformers import SentenceTransformer
import json
from datetime import datetime, timedelta
from collections import defaultdict
import warnings
import traceback
import os
import pickle
from surprise import Reader, Dataset, SVD
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')

class ForYouPageSystem:
    def __init__(self, db_path='sqlite:///flatZ_enhanced.db'):
        self.engine = create_engine(db_path)
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.cf_algo = None
        self.ranking_model = None
        self.scaler = None
        
        # Load pre-trained models
        self._load_models()
    
    def _load_models(self):
        """Load pre-trained models."""
        print("🔄 Loading pre-trained models...")
        try:
            with open("cf_model.pkl", "rb") as f:
                self.cf_algo = pickle.load(f)
            with open("ranking_model.pkl", "rb") as f:
                self.ranking_model = pickle.load(f)
            with open("scaler.pkl", "rb") as f:
                self.scaler = pickle.load(f)
            print("✅ Models loaded successfully!")
        except FileNotFoundError:
            print("❌ Pre-trained models not found. Please run 'model_training.py' to train and save models.")
            raise
    
    def _initialize_models(self):
        """Initialize CF and ranking models"""
        print("🔄 Loading recommendation models...")
        
        # Initialize Collaborative Filtering
        interactions = pd.read_sql('SELECT user_id, item_id, action, rating FROM interactions', self.engine)
        
        action_ratings = {'view': 2.0, 'like': 4.0, 'bookmark': 4.5, 'share': 4.5, 'join': 5.0, 'comment': 3.5}
        interactions['final_rating'] = interactions.apply(
            lambda x: x['rating'] if pd.notna(x['rating']) else action_ratings.get(x['action'], 3.0), axis=1
        )
        
        reader = Reader(rating_scale=(1, 5))
        data = Dataset.load_from_df(interactions[['user_id', 'item_id', 'final_rating']], reader)
        trainset = data.build_full_trainset()
        
        self.cf_algo = SVD(n_factors=50, n_epochs=30, random_state=42)
        self.cf_algo.fit(trainset)
        
        # Initialize Ranking Model
        self._train_ranking_model()
        print("✅ Models loaded successfully!")
    
    def _train_ranking_model(self):
        """Train the ranking model"""
        # Prepare training data
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
        """, self.engine)
        
        # Extract features for training
        training_data = []
        for _, row in interactions.iterrows():
            features = self._extract_features(row['user_id'], row['item_id'])
            features['target'] = row['rating']
            training_data.append(features)
        
        df = pd.DataFrame(training_data)
        feature_cols = [col for col in df.columns if col != 'target']
        
        X = df[feature_cols].fillna(0)
        y = df['target']
        
        X_scaled = self.scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        
        self.ranking_model = xgb.XGBRegressor(n_estimators=50, max_depth=4, random_state=42)
        self.ranking_model.fit(X_train, y_train)
    
    def _extract_features(self, user_id, item_id):
        """Extract ranking features"""
        try:
            # Get user and item data
            user = pd.read_sql(f'SELECT * FROM users WHERE id = {user_id}', self.engine).iloc[0]
            item = pd.read_sql(f'SELECT * FROM items WHERE id = {item_id}', self.engine).iloc[0]
            
            features = {}
            
            # Content similarity - handle missing or malformed embeddings
            try:
                if pd.notna(user.interests) and user.interests:
                    user_interests = [i.strip() for i in user.interests.split(',') if i.strip()]
                else:
                    user_interests = []
                
                if pd.notna(item.embedding) and item.embedding:
                    item_embedding = json.loads(item.embedding)
                    if user_interests:
                        interest_embeddings = [self.model.encode(interest) for interest in user_interests]
                        user_embedding = np.mean(interest_embeddings, axis=0)
                        
                        similarity = np.dot(user_embedding, item_embedding) / (
                            np.linalg.norm(user_embedding) * np.linalg.norm(item_embedding)
                        )
                        features['content_similarity'] = float(similarity)
                    else:
                        features['content_similarity'] = 0.5
                else:
                    features['content_similarity'] = 0.5
            except Exception:
                features['content_similarity'] = 0.5
            
            # Tag overlap - handle missing tags
            try:
                if pd.notna(item.tags) and item.tags:
                    item_tags = set(tag.strip().lower() for tag in item.tags.split(',') if tag.strip())
                else:
                    item_tags = set()
                
                if pd.notna(user.interests) and user.interests:
                    user_tags = set(interest.strip().lower() for interest in user.interests.split(',') if interest.strip())
                else:
                    user_tags = set()
                
                features['tag_overlap'] = len(item_tags.intersection(user_tags)) / len(item_tags) if item_tags else 0
            except Exception:
                features['tag_overlap'] = 0.0
            
            # Community match
            try:
                features['community_match'] = 1.0 if (pd.notna(user.community) and pd.notna(item.community) and user.community == item.community) else 0.0
            except Exception:
                features['community_match'] = 0.0
            
            # Recency - handle date parsing errors
            try:
                if pd.notna(item.creation_date) and item.creation_date:
                    creation_date = datetime.fromisoformat(item.creation_date)
                    days_old = (datetime.now() - creation_date).days
                    features['recency_score'] = max(0, 1 - (days_old / 30))
                else:
                    features['recency_score'] = 0.5
            except Exception:
                features['recency_score'] = 0.5
            
            # Popularity
            try:
                popularity = pd.read_sql(f'SELECT COUNT(*) as count FROM interactions WHERE item_id = {item_id}', self.engine).iloc[0].count
                features['popularity_score'] = min(1.0, popularity / 10)
            except Exception:
                features['popularity_score'] = 0.5
            
            # Price preference
            try:
                features['price_match'] = 1.0 if (pd.notna(item.price) and item.price == 0) else 0.5
            except Exception:
                features['price_match'] = 0.5
            
            return features
            
        except Exception as e:
            print(f"Error in _extract_features: {e}")
            return {'content_similarity': 0.5, 'tag_overlap': 0.0, 'community_match': 0.0, 
                   'recency_score': 0.5, 'popularity_score': 0.5, 'price_match': 0.5}
    
    def get_user_by_name(self, username):
        """Find user by name"""
        users = pd.read_sql(f"SELECT * FROM users WHERE name LIKE '%{username}%'", self.engine)
        return users.iloc[0] if not users.empty else None
    
    def generate_candidates(self, user_id, num_candidates=15):
        """Generate candidate recommendations"""
        user = pd.read_sql(f'SELECT * FROM users WHERE id = {user_id}', self.engine).iloc[0]
        community = user.community
        interests = user.interests.split(',')
        
        # Get community items
        items = pd.read_sql(f'SELECT * FROM items WHERE community = "{community}"', self.engine)
        
        # Get user interactions to exclude
        user_interactions = pd.read_sql(f'SELECT item_id FROM interactions WHERE user_id = {user_id}', self.engine)
        interacted_items = set(user_interactions.item_id.values) if not user_interactions.empty else set()
        
        candidates = []
        
        # Content-based candidates
        for _, item in items.iterrows():
            if item.id in interacted_items:
                continue

            try:
                item_embedding = json.loads(item.embedding)
                interest_embeddings = [self.model.encode(interest.strip()) for interest in interests]
                user_embedding = np.mean(interest_embeddings, axis=0)

                similarity = np.dot(user_embedding, item_embedding) / (
                    np.linalg.norm(user_embedding) * np.linalg.norm(item_embedding)
                )

                # Find the best matching interest/tag for reason
                item_tags = [tag.strip().lower() for tag in item.tags.split(',')]
                matched_interest = None
                for interest in interests:
                    if interest.strip().lower() in item_tags:
                        matched_interest = interest.strip()
                        break

                if similarity > 0.3:
                    reason_str = f"Matches your interest in {matched_interest}" if matched_interest else "Matches your interests"
                    candidates.append({
                        'item_id': item.id,
                        'title': item.title,
                        'description': item.description,
                        'item_type': item.item_type,
                        'price': item.price,
                        'rating': item.rating,
                        'tags': item.tags,
                        'score': similarity,
                        'strategy': 'content',
                        'reason': reason_str
                    })
            except:
                continue
        
        # Collaborative filtering candidates
        unseen_items = [item.id for _, item in items.iterrows() if item.id not in interacted_items]
        if unseen_items:
            testset = [(user_id, iid, 4) for iid in unseen_items[:20]]
            predictions = self.cf_algo.test(testset)
            
            for pred in predictions:
                if pred.est > 3.5:
                    item_info = items[items.id == pred.iid]
                    if not item_info.empty:
                        item = item_info.iloc[0]
                        candidates.append({
                            'item_id': pred.iid,
                            'title': item.title,
                            'description': item.description,
                            'item_type': item.item_type,
                            'price': item.price,
                            'rating': item.rating,
                            'tags': item.tags,
                            'score': pred.est,
                            'strategy': 'collaborative',
                            'reason': "Recommended for you based on similar users"
                        })
        
        # Recent items
        recent_items = pd.read_sql(f"""
            SELECT * FROM items 
            WHERE community = '{community}' 
            ORDER BY creation_date DESC 
            LIMIT 5
        """, self.engine)
        
        for _, item in recent_items.iterrows():
            if item.id not in interacted_items:
                candidates.append({
                    'item_id': item.id,
                    'title': item.title,
                    'description': item.description,
                    'item_type': item.item_type,
                    'price': item.price,
                    'rating': item.rating,
                    'tags': item.tags,
                    'score': 0.8,
                    'strategy': 'recent',
                    'reason': "Recently added in your community"
                })
        
        # Remove duplicates
        seen = set()
        unique_candidates = []
        for cand in candidates:
            if cand['item_id'] not in seen:
                unique_candidates.append(cand)
                seen.add(cand['item_id'])
        
        return unique_candidates[:num_candidates]
    
    def rank_candidates(self, user_id, candidates):
        """Rank candidates using ML model"""
        if not candidates:
            return candidates
        
        ranking_features = []
        for candidate in candidates:
            features = self._extract_features(user_id, candidate['item_id'])
            ranking_features.append(features)
        
        df = pd.DataFrame(ranking_features)
        feature_cols = [col for col in df.columns]
        X = df[feature_cols].fillna(0)
        X_scaled = self.scaler.transform(X)
        
        predicted_scores = self.ranking_model.predict(X_scaled)
        
        for i, candidate in enumerate(candidates):
            candidate['ranking_score'] = float(predicted_scores[i])
        
        return sorted(candidates, key=lambda x: x['ranking_score'], reverse=True)
    
    def get_for_you_page(self, username):
        """Generate For You page for a user"""
        # Find user
        user = self.get_user_by_name(username)
        if user is None:
            return None
        
        user_id = user.id
        
        # Generate candidates
        candidates = self.generate_candidates(user_id, 15)
        
        # Rank candidates
        ranked_candidates = self.rank_candidates(user_id, candidates)
        
        return {
            'user': user,
            'recommendations': ranked_candidates[:10]
        }
    
    def display_for_you_page(self, username):
        """Display the For You page"""
        result = self.get_for_you_page(username)
        
        if result is None:
            print(f"User '{username}' not found.")
            return
        
        user = result['user']
        recommendations = result['recommendations']
        
        print(f"\nFor You Page - {user['name']}")
        print("=" * 60)
        print(f"Community: {user.community} | Age Group: {user.age_group} | Activity Level: {user.activity_level}")
        print(f"Your interests: {user.interests}")
        print(f"\nPersonalized recommendations:")
        print("-" * 60)
        
        for i, rec in enumerate(recommendations, 1):
            price_str = "Free" if rec['price'] == 0 else f"Rs {rec['price']:.0f}"
            rating_str = "No rating" if pd.isna(rec['rating']) else f"{rec['rating']:.1f}"
            
            print(f"{i}. {rec['title']}")
            print(f"   {rec['description'][:100]}{'...' if len(rec['description']) > 100 else ''}")
            print(f"   Type: {rec['item_type'].title()} • {price_str} • Rating: {rating_str}")
            print(f"   Reason: {rec['reason']}")
            print()

def main():
    """Main function for For You page"""
    print("🌟 Welcome to Your Personalized For You Page!")
    print("=" * 50)
    
    # Initialize system
    system = ForYouPageSystem()
    
    while True:
        print(f"\n👋 Enter your name to see personalized recommendations")
        username = input("Your name: ").strip()
        
        if not username:
            print("Please enter a valid name.")
            continue
        
        if username.lower() in ['exit', 'quit', 'q']:
            print("👋 Thanks for using the For You Page!")
            break
        
        print(f"\n🔍 Finding recommendations for {username}...")
        system.display_for_you_page(username)
        
        # Ask if they want to continue
        continue_choice = input("\n🔄 Try another user? (y/N): ").strip().lower()
        if continue_choice not in ['y', 'yes']:
            print("👋 Thanks for using the For You Page!")
            break

if __name__ == "__main__":
    main()