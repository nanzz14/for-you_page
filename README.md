# FlatZ Recommendation System

A sophisticated recommendation system that provides personalized "For You" page recommendations for community events, services, and activities. Built with Python, this system combines collaborative filtering, content-based filtering, and machine learning ranking to deliver highly relevant recommendations.

## 🌟 Features

### Core Recommendation Engine
- **Multi-Strategy Approach**: Combines content-based, collaborative filtering, and ML ranking
- **Personalized Recommendations**: Based on user interests, community, and behavior patterns
- **Real-time Ranking**: Uses XGBoost for intelligent candidate ranking
- **Semantic Understanding**: Leverages sentence transformers for content similarity

### Recommendation Strategies
1. **Content-Based Filtering**: Matches user interests with item tags and descriptions
2. **Collaborative Filtering**: Uses SVD algorithm to find similar users and their preferences
3. **Recency-Based**: Includes recently added items in user's community
4. **ML Ranking**: Advanced ranking using extracted features and trained models

### Smart Features
- **Interest Matching**: Semantic similarity between user interests and item content
- **Community Alignment**: Prioritizes items from user's local community
- **Behavioral Analysis**: Considers user interaction patterns (likes, joins, bookmarks)
- **Dynamic Pricing**: Adapts to user's price sensitivity
- **Popularity Scoring**: Factors in item popularity and engagement

## 🚀 Quick Start

### Prerequisites
- Python 3.7+
- pip package manager
- Virtual environment (recommended)

### Installation

1. **Clone or download the project files**
   ```bash
   # Ensure you have all the required files in your project directory
   ```

2. **Create and activate a virtual environment**
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install required dependencies**
   ```bash
   pip install pandas numpy sqlalchemy sentence-transformers surprise xgboost scikit-learn faker
   ```

4. **Generate the dataset and database**
   ```bash
   python data.py
   ```

5. **Train the recommendation models**
   ```bash
   python training.py
   ```

6. **Run the recommendation system**
   ```bash
   python recommendation.py
   ```

## 📁 Project Structure

```
flatz_assesment/
├── README.md                 # This file
├── data.py                  # Dataset generation and database setup
├── training.py              # Model training and saving
├── recommendation.py        # Main recommendation system
├── users.csv               # Generated user data
├── items.csv               # Generated item data
├── interactions.csv         # Generated interaction data
├── reviews.csv             # Generated review data
├── flatZ_enhanced.db       # SQLite database (created after running data.py)
├── cf_model.pkl            # Collaborative filtering model (created after training)
├── ranking_model.pkl       # ML ranking model (created after training)
├── scaler.pkl              # Feature scaler (created after training)
└── venv/                   # Virtual environment directory
```

## 🔧 How It Works

### 1. Data Generation (`data.py`)
- Creates realistic user profiles with diverse interests and communities
- Generates community events, services, and activities
- Simulates user interactions and reviews
- Computes semantic embeddings for content similarity

### 2. Model Training (`training.py`)
- Trains collaborative filtering model using SVD algorithm
- Builds XGBoost ranking model with extracted features
- Saves trained models for quick loading

### 3. Recommendation System (`recommendation.py`)
- **Candidate Generation**: Creates diverse recommendation candidates
- **Feature Extraction**: Computes ranking features for each candidate
- **ML Ranking**: Uses trained model to score and rank candidates
- **Personalization**: Adapts recommendations to user preferences

## 🎯 Usage Examples

### Basic Usage
```python
from recommendation import ForYouPageSystem

# Initialize the system
system = ForYouPageSystem()

# Get personalized recommendations for a user
result = system.get_for_you_page("John Doe")

# Display the recommendations
system.display_for_you_page("John Doe")
```

### Interactive Mode
Run the main script for an interactive experience:
```bash
python recommendation.py
```

The system will prompt you to enter a username and display personalized recommendations.

## 📊 Dataset Details

### Generated Data
- **500 Users**: Diverse profiles with realistic interests and demographics
- **1000 Items**: Community events, services, and activities
- **3000 Interactions**: User engagement patterns
- **500 Reviews**: User feedback and ratings

### Communities
10 diverse communities including:
- Downtown Heights, Sunset Valley, Oakwood Gardens
- Riverside Park, Mountain View, Cedar Creek
- Lakeside Manor, Pine Ridge, Willow Brook, Maple Grove

### Interest Categories
25+ interest categories covering:
- Fitness & Wellness (yoga, pilates, running, cycling)
- Pets & Animals (dog walking, pet training)
- Outdoor Activities (hiking, camping, gardening)
- Social & Cultural (book clubs, cooking classes, art workshops)
- Sports & Recreation (basketball, tennis, swimming)

## 🧠 Technical Architecture

### Recommendation Pipeline
1. **User Lookup**: Find user by name and extract profile
2. **Candidate Generation**: Create diverse recommendation candidates
3. **Feature Extraction**: Compute ranking features for each candidate
4. **ML Ranking**: Score candidates using trained XGBoost model
5. **Result Presentation**: Display top-ranked recommendations

### Key Algorithms
- **SVD (Singular Value Decomposition)**: Collaborative filtering
- **Sentence Transformers**: Content similarity computation
- **XGBoost**: Feature-based ranking
- **Feature Engineering**: Multi-dimensional user-item matching

### Database Schema
- **Users**: Profile information, interests, community
- **Items**: Event details, tags, embeddings, metadata
- **Interactions**: User behavior patterns
- **Reviews**: User feedback and ratings

## 🔍 Customization

### Adding New Features
The system is designed to be extensible. You can:
- Add new ranking features in `_extract_features()`
- Implement additional recommendation strategies
- Customize the candidate generation logic
- Modify the feature engineering process

### Model Parameters
Adjust model hyperparameters in the training scripts:
- SVD factors and epochs
- XGBoost parameters (n_estimators, max_depth)
- Feature scaling and preprocessing

## 📚 Dependencies

### Core Libraries
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **sqlalchemy**: Database ORM and management
- **sentence-transformers**: Semantic text embeddings
- **surprise**: Recommendation algorithms
- **xgboost**: Gradient boosting for ranking
- **scikit-learn**: Machine learning utilities

### Data Generation
- **faker**: Realistic fake data generation