import pandas as pd
from faker import Faker
import random
from datetime import datetime, timedelta
from sqlalchemy import create_engine, Column, Integer, String, Text
from sqlalchemy.orm import declarative_base, sessionmaker
from sentence_transformers import SentenceTransformer
import json

# Step 1: Generate fake data
fake = Faker()
communities = ['Community1', 'Community2', 'Community3']
interests_list = ['yoga', 'pets', 'gardening', 'badminton', 'kids play date', 'basketball', 'fitness', 'events']

# Function to generate relevant titles based on tags
def generate_title(tags):
    primary_tag = tags[0]  # Use first tag for title
    if primary_tag == 'badminton':
        return f"Badminton Tournament in {random.choice(communities)}"
    elif primary_tag == 'kids play date':
        return f"Kids Play Date Event"
    elif primary_tag == 'basketball':
        return f"Basketball Game This Weekend"
    elif primary_tag == 'yoga':
        return f"Yoga Session in {random.choice(communities)}"
    elif primary_tag == 'pets':
        return f"Pet Care Workshop"
    elif primary_tag == 'gardening':
        return f"Community Gardening Meetup"
    elif primary_tag == 'fitness':
        return f"Fitness Bootcamp"
    elif primary_tag == 'events':
        return f"Community Social Event"
    return fake.sentence(nb_words=5)  # Fallback

# Users
users = []
for i in range(100):
    users.append({
        'user_id': i+1,
        'name': fake.name(),
        'community': random.choice(communities),
        'interests': ','.join(random.sample(interests_list, random.randint(2, 3)))
    })
pd.DataFrame(users).to_csv('users.csv', index=False)

# Items
items = []
for i in range(150):
    item_tags = random.sample(interests_list, random.randint(2, 3))
    description = fake.paragraph(nb_sentences=3)
    if 'badminton' in item_tags:
        description = f"Join our badminton tournament! {description}"
    elif 'kids play date' in item_tags:
        description = f"Organize a kids play date in the park. {description}"
    elif 'basketball' in item_tags:
        description = f"Basketball game this weekend! {description}"
    elif 'yoga' in item_tags:
        description = f"Relax with a yoga session. {description}"
    elif 'pets' in item_tags:
        description = f"Fun pet care workshop for animal lovers! {description}"
    elif 'gardening' in item_tags:
        description = f"Learn gardening tips at our meetup. {description}"
    elif 'fitness' in item_tags:
        description = f"Join our fitness bootcamp! {description}"
    elif 'events' in item_tags:
        description = f"Attend our community social event. {description}"
    items.append({
        'item_id': i+1,
        'title': generate_title(item_tags),
        'description': description,
        'tags': ','.join(item_tags),
        'community': random.choice(communities),
        'creator_id': random.randint(1, 100),
        'creation_date': (datetime.now() - timedelta(days=random.randint(1, 30))).isoformat()
    })
pd.DataFrame(items).to_csv('items.csv', index=False)

# Interactions
interactions = []
for i in range(500):
    interactions.append({
        'id': i+1,
        'user_id': random.randint(1, 100),
        'item_id': random.randint(1, 150),
        'action': random.choice(['like', 'view']),
        'timestamp': (datetime.now() - timedelta(days=random.randint(1, 30))).isoformat()
    })
pd.DataFrame(interactions).to_csv('interactions.csv', index=False)

# Step 2: Set up SQLite database
Base = declarative_base()

class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True)
    name = Column(String)
    community = Column(String)
    interests = Column(String)

class Item(Base):
    __tablename__ = 'items'
    id = Column(Integer, primary_key=True)
    title = Column(String)
    description = Column(Text)
    tags = Column(String)
    community = Column(String)
    creator_id = Column(Integer)
    creation_date = Column(String)
    embedding = Column(Text)

class Interaction(Base):
    __tablename__ = 'interactions'
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer)
    item_id = Column(Integer)
    action = Column(String)
    timestamp = Column(String)

# Create database
engine = create_engine('sqlite:///flatZ.db')
Base.metadata.drop_all(engine)  # Clear existing tables
Base.metadata.create_all(engine)  # Recreate tables
Session = sessionmaker(bind=engine)
session = Session()

# Step 3: Load data and compute embeddings
try:
    # Users
    users_df = pd.read_csv('users.csv')
    for _, row in users_df.iterrows():
        session.add(User(id=row['user_id'], name=row['name'], community=row['community'], interests=row['interests']))

    # Items with embeddings
    model = SentenceTransformer('all-MiniLM-L6-v2')
    items_df = pd.read_csv('items.csv')
    for _, row in items_df.iterrows():
        # Combine title and description for embedding
        text = f"{row['title']} {row['description']}"
        embedding = model.encode(text).tolist()
        session.add(Item(
            id=row['item_id'], title=row['title'], description=row['description'],
            tags=row['tags'], community=row['community'], creator_id=row['creator_id'],
            creation_date=row['creation_date'], embedding=json.dumps(embedding)
        ))

    # Interactions
    interactions_df = pd.read_csv('interactions.csv')
    for _, row in interactions_df.iterrows():
        session.add(Interaction(id=row['id'], user_id=row['user_id'], item_id=row['item_id'], action=row['action'], timestamp=row['timestamp']))

    # Save to database
    session.commit()
    print("Data ingested successfully!")
except Exception as e:
    print(f"Error: {e}")
    session.rollback()
finally:
    session.close()