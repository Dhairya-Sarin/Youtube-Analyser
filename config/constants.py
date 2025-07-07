from typing import Dict, List

# Clickbait keywords and phrases
CLICKBAIT_PHRASES = [
    'you won\'t believe', 'shocking', 'amazing', 'incredible', 'must watch',
    'gone wrong', 'exposed', 'secret', 'truth', 'revealed', 'hack',
    'trick', 'tips', 'how to', 'tutorial', 'review', 'reaction',
    'vs', 'challenge', 'experiment', 'test', 'comparison', 'ultimate',
    'best', 'worst', 'top', 'epic', 'fail', 'success', 'viral',
    'trending', 'new', 'latest', 'update', 'breaking', 'exclusive'
]

# Emotional words for sentiment analysis
EMOTIONAL_WORDS = {
    'positive': ['love', 'amazing', 'awesome', 'great', 'excellent', 'fantastic', 'wonderful'],
    'negative': ['hate', 'terrible', 'awful', 'bad', 'horrible', 'disgusting', 'worst'],
    'excitement': ['exciting', 'thrilling', 'incredible', 'unbelievable', 'stunning'],
    'urgency': ['now', 'today', 'urgent', 'quickly', 'immediately', 'fast', 'limited']
}

# Time-related keywords
TIME_WORDS = [
    'today', 'now', 'new', 'latest', 'recent', 'current', 'live',
    'breaking', '2024', '2025', 'this week', 'this month'
]

# Action words that suggest tutorials or how-to content
ACTION_WORDS = [
    'how', 'make', 'create', 'build', 'learn', 'master', 'get', 'achieve',
    'improve', 'increase', 'boost', 'enhance', 'optimize', 'fix', 'solve'
]

# Video category mappings (YouTube category IDs)
VIDEO_CATEGORIES = {
    1: 'Film & Animation',
    2: 'Autos & Vehicles',
    10: 'Music',
    15: 'Pets & Animals',
    17: 'Sports',
    19: 'Travel & Events',
    20: 'Gaming',
    22: 'People & Blogs',
    23: 'Comedy',
    24: 'Entertainment',
    25: 'News & Politics',
    26: 'Howto & Style',
    27: 'Education',
    28: 'Science & Technology'
}

# Prime time hours (when most people are online)
PRIME_TIME_HOURS = list(range(18, 23))  # 6 PM to 10 PM

# Weekend days
WEEKEND_DAYS = [5, 6]  # Saturday, Sunday

# Color analysis constants
COLOR_RANGES = {
    'red': ([0, 100, 100], [10, 255, 255]),
    'orange': ([11, 100, 100], [25, 255, 255]),
    'yellow': ([26, 100, 100], [35, 255, 255]),
    'green': ([36, 100, 100], [85, 255, 255]),
    'blue': ([86, 100, 100], [125, 255, 255]),
    'purple': ([126, 100, 100], [165, 255, 255])
}

# Thumbnail analysis thresholds
THUMBNAIL_THRESHOLDS = {
    'brightness_high': 150,
    'brightness_low': 50,
    'contrast_high': 50,
    'contrast_low': 20,
    'face_area_threshold': 0.1,
    'text_density_threshold': 0.3
}

# Feature importance categories
FEATURE_CATEGORIES = {
    'metadata': ['title_length', 'description_word_count', 'duration_seconds', 'tag_count'],
    'temporal': ['publish_hour', 'publish_day_of_week', 'is_weekend', 'is_prime_time'],
    'engagement': ['views', 'likes', 'comments', 'engagement_score', 'like_ratio'],
    'title_nlp': ['title_sentiment_polarity', 'clickbait_score', 'emotional_word_count'],
    'thumbnail': ['thumbnail_brightness', 'thumbnail_face_count', 'thumbnail_text_density'],
    'channel': ['channel_age_days', 'channel_total_subscribers', 'upload_frequency']
}

# Machine learning constants
ML_CONSTANTS = {
    'min_samples_for_training': 50,
    'max_features_ratio': 0.8,
    'feature_selection_threshold': 0.01,
    'shap_sample_size': 100,
    'cross_validation_folds': 5
}

# Export formats
EXPORT_FORMATS = {
    'excel': '.xlsx',
    'csv': '.csv',
    'json': '.json',
    'parquet': '.parquet'
}

# Holidays and special days (simplified)
SPECIAL_DAYS = {
    'new_year': [(1, 1)],
    'valentine': [(2, 14)],
    'christmas': [(12, 25)],
    'halloween': [(10, 31)],
    'thanksgiving': [(11, 24)],  # Approximate
    'black_friday': [(11, 25)]   # Approximate
}