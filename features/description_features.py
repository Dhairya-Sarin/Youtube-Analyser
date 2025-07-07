import re
from typing import Dict, Any, List
from textblob import TextBlob
from .base_extractor import BaseFeatureExtractor


class DescriptionFeatureExtractor(BaseFeatureExtractor):
    """Extract features from video descriptions and tags"""

    def __init__(self, config=None):
        super().__init__(config)
        self.feature_names = [
            'description_word_count', 'description_char_count', 'description_line_count',
            'description_has_links', 'description_link_count', 'description_has_hashtags',
            'description_hashtag_count', 'description_has_email', 'description_has_phone',
            'description_has_social_media', 'description_has_affiliate_link',
            'description_sentiment_polarity', 'description_sentiment_subjectivity',
            'description_call_to_action_count', 'description_timestamp_count',
            'tag_count', 'tag_avg_length', 'tag_overlap_with_title',
            'tag_overlap_with_description', 'tag_uniqueness_score'
        ]

    def extract(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract description and tag features"""
        description = data.get('description', '')
        tags = data.get('tags', [])
        title = data.get('title', '')

        features = {}

        # Description features
        features.update(self._extract_description_features(description))

        # Tag features
        features.update(self._extract_tag_features(tags, title, description))

        return features

    def _extract_description_features(self, description: str) -> Dict[str, Any]:
        """Extract features from video description"""
        if not description:
            return {name: 0 for name in self.feature_names if name.startswith('description_')}

        features = {}
        description_lower = description.lower()

        # Basic text statistics
        words = description.split()
        lines = description.split('\n')

        features['description_word_count'] = len(words)
        features['description_char_count'] = len(description)
        features['description_line_count'] = len(lines)

        # Link detection
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        links = re.findall(url_pattern, description)
        features['description_has_links'] = 1 if links else 0
        features['description_link_count'] = len(links)

        # Hashtag detection
        hashtags = re.findall(r'#\w+', description)
        features['description_has_hashtags'] = 1 if hashtags else 0
        features['description_hashtag_count'] = len(hashtags)

        # Contact information
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(email_pattern, description)
        features['description_has_email'] = 1 if emails else 0

        phone_pattern = r'\b(?:\+?1[-.\s]?)?(?:\(?[0-9]{3}\)?[-.\s]?)?[0-9]{3}[-.\s]?[0-9]{4}\b'
        phones = re.findall(phone_pattern, description)
        features['description_has_phone'] = 1 if phones else 0

        # Social media mentions
        social_patterns = ['twitter', 'instagram', 'facebook', 'tiktok', 'discord', 'telegram']
        features['description_has_social_media'] = 1 if any(
            pattern in description_lower for pattern in social_patterns) else 0

        # Affiliate link indicators
        affiliate_patterns = ['affiliate', 'sponsored', 'ad', 'promo', 'discount', 'coupon', 'deal']
        features['description_has_affiliate_link'] = 1 if any(
            pattern in description_lower for pattern in affiliate_patterns) else 0

        # Sentiment analysis
        try:
            blob = TextBlob(description)
            features['description_sentiment_polarity'] = blob.sentiment.polarity
            features['description_sentiment_subjectivity'] = blob.sentiment.subjectivity
        except:
            features['description_sentiment_polarity'] = 0.0
            features['description_sentiment_subjectivity'] = 0.0

        # Call to action
        cta_patterns = ['subscribe', 'like', 'comment', 'share', 'follow', 'click', 'watch', 'check out']
        cta_count = sum(description_lower.count(pattern) for pattern in cta_patterns)
        features['description_call_to_action_count'] = cta_count

        # Timestamps (for chapters/navigation)
        timestamp_pattern = r'\b(?:[0-9]{1,2}:)?[0-9]{1,2}:[0-9]{2}\b'
        timestamps = re.findall(timestamp_pattern, description)
        features['description_timestamp_count'] = len(timestamps)

        return features

    def _extract_tag_features(self, tags: List[str], title: str, description: str) -> Dict[str, Any]:
        """Extract features from video tags"""
        if not tags:
            return {
                'tag_count': 0,
                'tag_avg_length': 0,
                'tag_overlap_with_title': 0,
                'tag_overlap_with_description': 0,
                'tag_uniqueness_score': 0
            }

        features = {}

        # Basic tag statistics
        features['tag_count'] = len(tags)
        features['tag_avg_length'] = sum(len(tag) for tag in tags) / len(tags)

        # Tag overlap with title
        title_words = set(title.lower().split())
        tag_words = set(' '.join(tags).lower().split())

        if title_words:
            overlap_title = len(title_words.intersection(tag_words)) / len(title_words)
            features['tag_overlap_with_title'] = overlap_title
        else:
            features['tag_overlap_with_title'] = 0

        # Tag overlap with description
        description_words = set(description.lower().split())
        if description_words:
            overlap_desc = len(description_words.intersection(tag_words)) / len(description_words)
            features['tag_overlap_with_description'] = overlap_desc
        else:
            features['tag_overlap_with_description'] = 0

        # Tag uniqueness (how specific vs generic the tags are)
        # Simple heuristic: longer, more specific tags get higher scores
        avg_tag_specificity = sum(len(tag.split()) for tag in tags) / len(tags)
        features['tag_uniqueness_score'] = min(avg_tag_specificity / 3.0, 1.0)  # Normalize

        return features

    def get_feature_names(self) -> List[str]:
        return self.feature_names