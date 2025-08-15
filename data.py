import pandas as pd
from faker import Faker
import random
from datetime import datetime, timedelta
from sqlalchemy import create_engine, Column, Integer, String, Text, Float
from sqlalchemy.orm import declarative_base, sessionmaker
from sentence_transformers import SentenceTransformer
import json
import numpy as np

# Step 1: Generate fake data with more variety
fake = Faker()

# Expanded communities (10 diverse communities)
communities = [
    'Downtown Heights', 'Sunset Valley', 'Oakwood Gardens', 'Riverside Park',
    'Mountain View', 'Cedar Creek', 'Lakeside Manor', 'Pine Ridge',
    'Willow Brook', 'Maple Grove'
]

# Expanded interests with subcategories (25+ interests)
interests_list = [
    'yoga', 'pilates', 'meditation', 'fitness', 'crossfit', 'running', 'cycling',
    'pets', 'dogs', 'cats', 'pet training', 'pet grooming',
    'gardening', 'urban farming', 'composting', 'plant care',
    'badminton', 'tennis', 'basketball', 'soccer', 'volleyball', 'swimming',
    'kids play date', 'family events', 'parenting', 'child care',
    'cooking', 'baking', 'food tasting', 'wine tasting',
    'book club', 'writing', 'poetry', 'literature',
    'art', 'painting', 'photography', 'crafts',
    'music', 'guitar', 'piano', 'singing',
    'technology', 'coding', 'gaming', 'board games',
    'hiking', 'camping', 'outdoor adventures', 'nature walks',
    'volunteer work', 'community service', 'charity events',
    'business', 'networking', 'entrepreneurship',
    'language learning', 'cultural exchange'
]

# Age groups for more realistic user profiles
age_groups = ['18-25', '26-35', '36-45', '46-55', '56+']

# Activity levels
activity_levels = ['very_active', 'active', 'moderate', 'occasional']

# Function to generate more sophisticated titles based on tags and community
def generate_title(tags, community, activity_type='event'):
    primary_tag = tags[0]
    secondary_tag = tags[1] if len(tags) > 1 else None
    
    title_templates = {
        'yoga': [
            f"Sunrise Yoga in {community}",
            f"{community} Weekly Yoga Class",
            f"Mindful Yoga Session - {community}",
            f"Beginner Yoga Workshop"
        ],
        'pilates': [
            f"Pilates Class in {community}",
            f"Core Strength Pilates",
            f"{community} Pilates Group"
        ],
        'fitness': [
            f"Outdoor Fitness Bootcamp",
            f"{community} Fitness Group",
            f"HIIT Training Session",
            f"Morning Fitness Class"
        ],
        'pets': [
            f"Dog Walking Group - {community}",
            f"Pet Care Workshop",
            f"{community} Pet Owners Meetup",
            f"Dog Training Class"
        ],
        'gardening': [
            f"{community} Community Garden",
            f"Seasonal Gardening Tips",
            f"Organic Gardening Workshop",
            f"Plant Swap Event"
        ],
        'basketball': [
            f"Sunday Basketball Game",
            f"{community} Basketball League",
            f"Pickup Basketball Session",
            f"Youth Basketball Training"
        ],
        'kids play date': [
            f"Toddler Playdate at {community} Park",
            f"Family Fun Day",
            f"Kids Activity Session",
            f"Playground Meetup"
        ],
        'cooking': [
            f"Community Cooking Class",
            f"{community} Potluck Dinner",
            f"Healthy Cooking Workshop",
            f"International Cuisine Night"
        ],
        'book club': [
            f"{community} Book Discussion",
            f"Monthly Book Club Meeting",
            f"Literary Society Gathering",
            f"Author Reading Event"
        ],
        'hiking': [
            f"{community} Nature Hike",
            f"Weekend Trail Adventure",
            f"Beginner Hiking Group",
            f"Sunrise Hike Club"
        ]
    }
    
    if primary_tag in title_templates:
        return random.choice(title_templates[primary_tag])
    else:
        return f"{primary_tag.title()} Group in {community}"

def generate_description(tags, community):
    primary_tag = tags[0]
    
    description_templates = {
        'yoga': [
            "Join us for a relaxing yoga session suitable for all levels. We'll focus on breathing techniques and gentle stretches.",
            "Experience the benefits of yoga in a supportive community environment. Mats provided for beginners.",
            "Start your week right with our Monday morning yoga class. Perfect for stress relief and flexibility."
        ],
        'fitness': [
            "High-energy workout session combining cardio and strength training. All fitness levels welcome!",
            "Transform your fitness journey with our supportive group. Equipment provided.",
            "Outdoor fitness class that makes working out fun and engaging. Weather permitting."
        ],
        'pets': [
            "Calling all pet lovers! Share tips, stories, and enjoy the company of fellow animal enthusiasts.",
            "Professional pet trainer will guide us through basic obedience techniques. Bring your furry friends!",
            "Social hour for pets and their humans. Great for socialization and making new friends."
        ],
        'gardening': [
            "Learn sustainable gardening practices from experienced community members. Tools and seeds provided.",
            "Seasonal workshop covering planting, maintenance, and harvesting techniques for local climate.",
            "Hands-on gardening experience with emphasis on organic and eco-friendly methods."
        ],
        'cooking': [
            "Explore new flavors and cooking techniques in a fun, collaborative environment. Ingredients provided.",
            "Learn to prepare healthy, delicious meals that fit your busy lifestyle. Recipes included!",
            "Cultural cooking experience featuring traditional dishes and modern interpretations."
        ]
    }
    
    if primary_tag in description_templates:
        base_desc = random.choice(description_templates[primary_tag])
        return f"{base_desc} {fake.paragraph(nb_sentences=2)}"
    else:
        return f"Join our {primary_tag} group for an engaging community experience. {fake.paragraph(nb_sentences=3)}"

# Generate Users (500 users for better diversity)
users = []
for i in range(500):
    num_interests = random.randint(3, 6)  # More interests per user
    user_interests = random.sample(interests_list, num_interests)
    
    users.append({
        'user_id': i+1,
        'name': fake.name(),
        'community': random.choice(communities),
        'interests': ','.join(user_interests),
        'age_group': random.choice(age_groups),
        'activity_level': random.choice(activity_levels),
        'join_date': (datetime.now() - timedelta(days=random.randint(1, 365))).isoformat(),
        'email': fake.email()
    })

pd.DataFrame(users).to_csv('users.csv', index=False)

# Generate Items (1000 items with more variety)
items = []
item_types = ['event', 'service', 'group', 'workshop', 'class', 'meetup']

for i in range(1000):
    num_tags = random.randint(2, 4)  # 2-4 tags per item
    item_tags = random.sample(interests_list, num_tags)
    community = random.choice(communities)
    item_type = random.choice(item_types)
    
    title = generate_title(item_tags, community, item_type)
    description = generate_description(item_tags, community)
    
    # Add pricing information
    price = 0 if random.random() < 0.6 else random.randint(5, 50)  # 60% free events
    
    # Add capacity and current participants
    max_capacity = random.randint(5, 50)
    current_participants = random.randint(0, max_capacity)
    
    # Add event timing
    event_date = datetime.now() + timedelta(days=random.randint(1, 60))
    duration_hours = random.choice([1, 1.5, 2, 2.5, 3])
    
    items.append({
        'item_id': i+1,
        'title': title,
        'description': description,
        'tags': ','.join(item_tags),
        'community': community,
        'creator_id': random.randint(1, 500),
        'creation_date': (datetime.now() - timedelta(days=random.randint(1, 90))).isoformat(),
        'item_type': item_type,
        'price': price,
        'max_capacity': max_capacity,
        'current_participants': current_participants,
        'event_date': event_date.isoformat(),
        'duration_hours': duration_hours,
        'is_recurring': random.choice([True, False]),
        'rating': round(random.uniform(3.5, 5.0), 1) if random.random() < 0.7 else None
    })

pd.DataFrame(items).to_csv('items.csv', index=False)

# Generate Interactions (3000 interactions for better patterns)
interactions = []
interaction_types = ['like', 'view', 'join', 'comment', 'share', 'bookmark']

for i in range(3000):
    user_id = random.randint(1, 500)
    item_id = random.randint(1, 1000)
    action = random.choice(interaction_types)
    
    # Add weighted probability for more realistic interactions
    if action == 'view':
        weight = 0.5  # Most common
    elif action in ['like', 'bookmark']:
        weight = 0.25
    elif action == 'join':
        weight = 0.1
    else:
        weight = 0.05
    
    # Add interaction strength/rating for some actions
    rating = None
    if action in ['like', 'join'] and random.random() < 0.3:
        rating = random.randint(1, 5)
    
    interactions.append({
        'id': i+1,
        'user_id': user_id,
        'item_id': item_id,
        'action': action,
        'timestamp': (datetime.now() - timedelta(days=random.randint(1, 90))).isoformat(),
        'rating': rating,
        'session_duration': random.randint(10, 300) if action == 'view' else None  # seconds
    })

pd.DataFrame(interactions).to_csv('interactions.csv', index=False)

# Generate Reviews (500 reviews for additional feedback data)
reviews = []
review_sentiments = ['positive', 'neutral', 'negative']
positive_reviews = [
    "Great experience! Highly recommended.",
    "Loved the atmosphere and met wonderful people.",
    "Excellent organization and friendly participants.",
    "Will definitely join again. Five stars!",
    "Exceeded my expectations. Well worth it."
]
neutral_reviews = [
    "It was okay, nothing special but decent.",
    "Average experience, could be better organized.",
    "Not bad, but room for improvement.",
    "Decent activity, would consider again."
]
negative_reviews = [
    "Disappointed with the organization.",
    "Not what I expected from the description.",
    "Could have been much better.",
    "Wouldn't recommend to others."
]

for i in range(500):
    sentiment = random.choice(review_sentiments)
    if sentiment == 'positive':
        review_text = random.choice(positive_reviews)
        rating = random.randint(4, 5)
    elif sentiment == 'neutral':
        review_text = random.choice(neutral_reviews)
        rating = random.randint(2, 4)
    else:
        review_text = random.choice(negative_reviews)
        rating = random.randint(1, 3)
    
    reviews.append({
        'review_id': i+1,
        'user_id': random.randint(1, 500),
        'item_id': random.randint(1, 1000),
        'rating': rating,
        'review_text': review_text,
        'review_date': (datetime.now() - timedelta(days=random.randint(1, 60))).isoformat()
    })

pd.DataFrame(reviews).to_csv('reviews.csv', index=False)

# Step 2: Enhanced SQLite database schema
Base = declarative_base()

class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True)
    name = Column(String)
    community = Column(String)
    interests = Column(String)
    age_group = Column(String)
    activity_level = Column(String)
    join_date = Column(String)
    email = Column(String)

class Item(Base):
    __tablename__ = 'items'
    id = Column(Integer, primary_key=True)
    title = Column(String)
    description = Column(Text)
    tags = Column(String)
    community = Column(String)
    creator_id = Column(Integer)
    creation_date = Column(String)
    item_type = Column(String)
    price = Column(Float)
    max_capacity = Column(Integer)
    current_participants = Column(Integer)
    event_date = Column(String)
    duration_hours = Column(Float)
    is_recurring = Column(String)
    rating = Column(Float)
    embedding = Column(Text)

class Interaction(Base):
    __tablename__ = 'interactions'
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer)
    item_id = Column(Integer)
    action = Column(String)
    timestamp = Column(String)
    rating = Column(Integer)
    session_duration = Column(Integer)

class Review(Base):
    __tablename__ = 'reviews'
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer)
    item_id = Column(Integer)
    rating = Column(Integer)
    review_text = Column(Text)
    review_date = Column(String)

# Create enhanced database
engine = create_engine('sqlite:///flatZ_enhanced.db')
Base.metadata.drop_all(engine)  # Clear existing tables
Base.metadata.create_all(engine)  # Recreate tables
Session = sessionmaker(bind=engine)
session = Session()

# Step 3: Load data and compute embeddings
try:
    print("Loading users...")
    users_df = pd.read_csv('users.csv')
    for _, row in users_df.iterrows():
        session.add(User(
            id=row['user_id'], name=row['name'], community=row['community'], 
            interests=row['interests'], age_group=row['age_group'], 
            activity_level=row['activity_level'], join_date=row['join_date'],
            email=row['email']
        ))

    print("Loading items and computing embeddings...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    items_df = pd.read_csv('items.csv')
    for i, row in items_df.iterrows():
        if i % 100 == 0:
            print(f"Processing item {i+1}/{len(items_df)}")
        
        # Create richer text for embedding
        text = f"{row['title']} {row['description']} {row['tags'].replace(',', ' ')} {row['community']} {row['item_type']}"
        embedding = model.encode(text).tolist()
        
        session.add(Item(
            id=row['item_id'], title=row['title'], description=row['description'],
            tags=row['tags'], community=row['community'], creator_id=row['creator_id'],
            creation_date=row['creation_date'], item_type=row['item_type'],
            price=row['price'], max_capacity=row['max_capacity'],
            current_participants=row['current_participants'], event_date=row['event_date'],
            duration_hours=row['duration_hours'], is_recurring=str(row['is_recurring']),
            rating=row['rating'] if pd.notna(row['rating']) else None,
            embedding=json.dumps(embedding)
        ))

    print("Loading interactions...")
    interactions_df = pd.read_csv('interactions.csv')
    for _, row in interactions_df.iterrows():
        session.add(Interaction(
            id=row['id'], user_id=row['user_id'], item_id=row['item_id'], 
            action=row['action'], timestamp=row['timestamp'],
            rating=row['rating'] if pd.notna(row['rating']) else None,
            session_duration=row['session_duration'] if pd.notna(row['session_duration']) else None
        ))

    print("Loading reviews...")
    reviews_df = pd.read_csv('reviews.csv')
    for _, row in reviews_df.iterrows():
        session.add(Review(
            id=row['review_id'], user_id=row['user_id'], item_id=row['item_id'],
            rating=row['rating'], review_text=row['review_text'],
            review_date=row['review_date']
        ))

    # Commit all data
    print("Saving to database...")
    session.commit()
    
    print("✅ Enhanced dataset created successfully!")
    print(f"📊 Dataset Summary:")
    print(f"   • Users: {len(users_df):,}")
    print(f"   • Items: {len(items_df):,}")
    print(f"   • Interactions: {len(interactions_df):,}")
    print(f"   • Reviews: {len(reviews_df):,}")
    print(f"   • Communities: {len(communities)}")
    print(f"   • Interest Categories: {len(interests_list)}")
    
except Exception as e:
    print(f"❌ Error: {e}")
    session.rollback()
finally:
    session.close()