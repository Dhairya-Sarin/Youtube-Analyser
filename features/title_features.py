import re
import string
from collections import Counter
from typing import Dict, Any, List, Optional
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from textblob import TextBlob
from .base_extractor import BaseFeatureExtractor
from config.constants import CLICKBAIT_PHRASES, EMOTIONAL_WORDS, TIME_WORDS, ACTION_WORDS

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('vader_lexicon', quiet=True)
except:
    pass


class TitleNLPExtractor(BaseFeatureExtractor):
    """Extract NLP and psychological features from video titles"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.feature_names = [
            # Basic text statistics
            'title_word_count', 'title_length', 'title_char_count',
            'title_question_count', 'title_exclamation_count', 'title_number_count',
            'title_uppercase_ratio', 'title_has_numbers', 'title_stopword_ratio',

            # NLP features
            'title_sentiment_polarity', 'title_sentiment_subjectivity',
            'title_named_entities_count', 'title_personal_reference_count',

            # Psychological features
            'title_clickbait_score', 'title_emotional_word_count',
            'title_urgency_score', 'title_promise_score', 'title_fear_words',
            'title_power_words', 'title_superlative_count',

            # Content type indicators
            'is_how_to', 'is_review', 'is_reaction', 'is_vs_comparison',
            'is_list_video', 'is_challenge', 'is_tutorial',

            # POS tag features
            'title_noun_ratio', 'title_verb_ratio', 'title_adjective_ratio',
            'title_adverb_ratio'
        ]

        # Load stopwords
        try:
            self.stop_words = set(stopwords.words('english'))
        except:
            self.stop_words = set()

    def extract(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract title NLP features"""
        title = data.get('title', '')
        if not title:
            return {name: 0 for name in self.feature_names}

        features = {}

        # Basic text statistics
        features.update(self._extract_basic_stats(title))

        # NLP features
        features.update(self._extract_nlp_features(title))

        # Psychological features
        features.update(self._extract_psychological_features(title))

        # Content type indicators
        features.update(self._extract_content_type_features(title))

        # POS tag features
        features.update(self._extract_pos_features(title))

        return features

    def _extract_basic_stats(self, title: str) -> Dict[str, Any]:
        """Extract basic text statistics"""
        words = title.split()

        return {
            'title_word_count': len(words),
            'title_length': len(title),
            'title_char_count': len(title),
            'title_question_count': title.count('?'),
            'title_exclamation_count': title.count('!'),
            'title_number_count': sum(1 for char in title if char.isdigit()),
            'title_uppercase_ratio': sum(1 for char in title if char.isupper()) / max(len(title), 1),
            'title_has_numbers': 1 if any(char.isdigit() for char in title) else 0,
            'title_stopword_ratio': sum(1 for word in words if word.lower() in self.stop_words) / max(len(words), 1)
        }

    def _extract_nlp_features(self, title: str) -> Dict[str, Any]:
        """Extract NLP features"""
        features = {}

        # Sentiment analysis
        try:
            blob = TextBlob(title)
            features['title_sentiment_polarity'] = blob.sentiment.polarity
            features['title_sentiment_subjectivity'] = blob.sentiment.subjectivity
        except:
            features['title_sentiment_polarity'] = 0.0
            features['title_sentiment_subjectivity'] = 0.0

        # Named entities (simplified)
        # Note: For production, you might want to use spaCy or more advanced NER
        capitalized_words = re.findall(r'\b[A-Z][a-z]+\b', title)
        features['title_named_entities_count'] = len(capitalized_words)

        # Personal references
        personal_pronouns = ['you', 'your', 'yours', 'we', 'our', 'us', 'my', 'me', 'i']
        title_lower = title.lower()
        features['title_personal_reference_count'] = sum(
            title_lower.count(pronoun) for pronoun in personal_pronouns
        )

        return features

    def _extract_psychological_features(self, title: str) -> Dict[str, Any]:
        """Extract psychological and clickbait features"""
        title_lower = title.lower()
        features = {}

        # Clickbait score
        clickbait_count = sum(1 for phrase in CLICKBAIT_PHRASES if phrase in title_lower)
        features['title_clickbait_score'] = clickbait_count

        # Emotional words
        emotional_count = 0
        for category, words in EMOTIONAL_WORDS.items():
            emotional_count += sum(1 for word in words if word in title_lower)
        features['title_emotional_word_count'] = emotional_count

        # Urgency indicators
        urgency_words = EMOTIONAL_WORDS.get('urgency', [])
        features['title_urgency_score'] = sum(1 for word in urgency_words if word in title_lower)

        # Promise indicators (how-to, tips, etc.)
        promise_patterns = ['how to', 'tips', 'secrets', 'guide', 'tutorial', 'learn']
        features['title_promise_score'] = sum(1 for pattern in promise_patterns if pattern in title_lower)

        # Fear words
        fear_words = ['dangerous', 'scary', 'terrifying', 'shocking', 'warning', 'avoid']
        features['title_fear_words'] = sum(1 for word in fear_words if word in title_lower)

        # Power words
        power_words = ['ultimate', 'best', 'top', 'amazing', 'incredible', 'unbelievable']
        features['title_power_words'] = sum(1 for word in power_words if word in title_lower)

        # Superlatives
        superlative_patterns = ['most', 'least', 'best', 'worst', 'fastest', 'slowest', 'biggest', 'smallest']
        features['title_superlative_count'] = sum(1 for pattern in superlative_patterns if pattern in title_lower)

        return features

    def _extract_content_type_features(self, title: str) -> Dict[str, Any]:
        """Extract content type indicators"""
        title_lower = title.lower()

        return {
            'is_how_to': 1 if 'how to' in title_lower else 0,
            'is_review': 1 if 'review' in title_lower else 0,
            'is_reaction': 1 if 'reaction' in title_lower else 0,
            'is_vs_comparison': 1 if ' vs ' in title_lower or ' versus ' in title_lower else 0,
            'is_list_video': 1 if any(
                pattern in title_lower for pattern in ['top ', 'best ', 'worst ', 'reasons']) else 0,
            'is_challenge': 1 if 'challenge' in title_lower else 0,
            'is_tutorial': 1 if 'tutorial' in title_lower or 'how to' in title_lower else 0
        }

    def _extract_pos_features(self, title: str) -> Dict[str, Any]:
        """Extract part-of-speech features"""
        try:
            words = word_tokenize(title)
            pos_tags = pos_tag(words)

            total_words = len(pos_tags)
            if total_words == 0:
                return {
                    'title_noun_ratio': 0,
                    'title_verb_ratio': 0,
                    'title_adjective_ratio': 0,
                    'title_adverb_ratio': 0
                }

            noun_count = sum(1 for word, pos in pos_tags if pos.startswith('NN'))
            verb_count = sum(1 for word, pos in pos_tags if pos.startswith('VB'))
            adj_count = sum(1 for word, pos in pos_tags if pos.startswith('JJ'))
            adv_count = sum(1 for word, pos in pos_tags if pos.startswith('RB'))

            return {
                'title_noun_ratio': noun_count / total_words,
                'title_verb_ratio': verb_count / total_words,
                'title_adjective_ratio': adj_count / total_words,
                'title_adverb_ratio': adv_count / total_words
            }
        except:
            return {
                'title_noun_ratio': 0,
                'title_verb_ratio': 0,
                'title_adjective_ratio': 0,
                'title_adverb_ratio': 0
            }

    def get_feature_names(self) -> List[str]:
        return self.feature_names
