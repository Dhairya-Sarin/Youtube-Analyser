import re
import string
from typing import List, Dict, Set
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer


class TextProcessor:
    """Utility class for text processing operations"""

    def __init__(self):
        # Download required NLTK data
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('wordnet', quiet=True)
            nltk.download('omw-1.4', quiet=True)
        except:
            pass

        # Initialize processors
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()

        try:
            self.stop_words = set(stopwords.words('english'))
        except:
            self.stop_words = set()

    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        if not text:
            return ""

        # Convert to lowercase
        text = text.lower()

        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)

        # Remove email addresses
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', text)

        # Remove special characters but keep spaces
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def tokenize(self, text: str, remove_stopwords: bool = True) -> List[str]:
        """Tokenize text into words"""
        if not text:
            return []

        # Clean text
        cleaned_text = self.clean_text(text)

        # Tokenize
        try:
            tokens = word_tokenize(cleaned_text)
        except:
            tokens = cleaned_text.split()

        # Remove stopwords if requested
        if remove_stopwords and self.stop_words:
            tokens = [token for token in tokens if token not in self.stop_words]

        # Remove single characters and empty strings
        tokens = [token for token in tokens if len(token) > 1]

        return tokens

    def extract_keywords(self, text: str, max_keywords: int = 50) -> List[Dict[str, any]]:
        """Extract keywords with frequency from text"""
        if not text:
            return []

        # Tokenize
        tokens = self.tokenize(text, remove_stopwords=True)

        # Count frequency
        from collections import Counter
        word_freq = Counter(tokens)

        # Create keyword list with frequencies
        keywords = []
        for word, freq in word_freq.most_common(max_keywords):
            keywords.append({
                'keyword': word,
                'frequency': freq,
                'relative_frequency': freq / len(tokens) if tokens else 0
            })

        return keywords

    def extract_ngrams(self, text: str, n: int = 2, max_ngrams: int = 20) -> List[Dict[str, any]]:
        """Extract n-grams from text"""
        if not text:
            return []

        tokens = self.tokenize(text, remove_stopwords=True)

        if len(tokens) < n:
            return []

        # Generate n-grams
        ngrams = []
        for i in range(len(tokens) - n + 1):
            ngram = ' '.join(tokens[i:i + n])
            ngrams.append(ngram)

        # Count frequency
        from collections import Counter
        ngram_freq = Counter(ngrams)

        # Create n-gram list
        result = []
        for ngram, freq in ngram_freq.most_common(max_ngrams):
            result.append({
                'ngram': ngram,
                'frequency': freq,
                'relative_frequency': freq / len(ngrams) if ngrams else 0
            })

        return result

    def calculate_readability(self, text: str) -> Dict[str, float]:
        """Calculate text readability metrics"""
        if not text:
            return {'flesch_score': 0, 'avg_sentence_length': 0, 'avg_word_length': 0}

        # Split into sentences
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        # Split into words
        words = self.tokenize(text, remove_stopwords=False)

        if not sentences or not words:
            return {'flesch_score': 0, 'avg_sentence_length': 0, 'avg_word_length': 0}

        # Calculate metrics
        avg_sentence_length = len(words) / len(sentences)
        avg_word_length = sum(len(word) for word in words) / len(words)

        # Simplified Flesch Reading Ease (approximation)
        flesch_score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_word_length)
        flesch_score = max(0, min(100, flesch_score))  # Clamp between 0-100

        return {
            'flesch_score': flesch_score,
            'avg_sentence_length': avg_sentence_length,
            'avg_word_length': avg_word_length
        }

    def detect_language(self, text: str) -> str:
        """Detect text language (simplified)"""
        if not text:
            return 'unknown'

        # Simple heuristic based on common English words
        english_indicators = ['the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with']

        words = self.tokenize(text, remove_stopwords=False)
        if not words:
            return 'unknown'

        english_word_count = sum(1 for word in words if word.lower() in english_indicators)
        english_ratio = english_word_count / len(words)

        return 'english' if english_ratio > 0.1 else 'other'
