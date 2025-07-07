import pytest
import pandas as pd
from unittest.mock import Mock, AsyncMock
from core.analyzer import YouTubeAnalyzer
from config.settings import get_default_settings


@pytest.fixture
def analyzer():
    settings = get_default_settings()
    settings.youtube.api_key = "test_key"
    return YouTubeAnalyzer(settings)


@pytest.fixture
def sample_video_data():
    return [
        {
            'id': 'test_video_1',
            'title': 'Test Video 1',
            'published_at': '2024-01-01T12:00:00Z',
            'view_count': 1000,
            'like_count': 100,
            'comment_count': 10,
            'duration': 'PT5M30S'
        }
    ]


def test_analyzer_initialization(analyzer):
    """Test analyzer initialization"""
    assert analyzer is not None
    assert analyzer.settings.youtube.api_key == "test_key"
    assert len(analyzer.feature_extractors) > 0


@pytest.mark.asyncio
async def test_feature_extraction(analyzer, sample_video_data):
    """Test feature extraction from video data"""
    channel_info = {'title': 'Test Channel', 'subscriber_count': 1000}

    features = await analyzer._extract_all_features(sample_video_data, channel_info)

    assert len(features) == 1
    assert 'video_id' in features[0]
    assert features[0]['video_id'] == 'test_video_1'


# tests/test_features/test_extractors.py
import pytest
from features.title_features import TitleNLPExtractor
from features.temporal_features import TemporalFeatureExtractor
from features.engagement_metrics import EngagementMetricsExtractor


def test_title_nlp_extractor():
    """Test title NLP feature extraction"""
    extractor = TitleNLPExtractor()

    test_data = {
        'title': 'How to Build Amazing Apps in 2024! Must Watch Tutorial'
    }

    features = extractor.extract(test_data)

    assert 'title_length' in features
    assert 'title_word_count' in features
    assert 'clickbait_score' in features
    assert 'is_tutorial' in features
    assert features['is_tutorial'] == 1
    assert features['title_exclamation_count'] == 1


def test_temporal_extractor():
    """Test temporal feature extraction"""
    extractor = TemporalFeatureExtractor()

    test_data = {
        'published_at': '2024-07-15T14:30:00Z'
    }

    features = extractor.extract(test_data)

    assert 'publish_hour' in features
    assert 'publish_day_of_week' in features
    assert 'is_weekend' in features
    assert features['publish_hour'] == 14
    assert features['is_afternoon'] == 1


def test_engagement_extractor():
    """Test engagement metrics extraction"""
    extractor = EngagementMetricsExtractor()

    test_data = {
        'viewCount': 10000,
        'likeCount': 500,
        'commentCount': 50,
        'days_since_upload': 30
    }

    features = extractor.extract(test_data)

    assert 'views' in features
    assert 'engagement_score' in features
    assert 'views_per_day' in features
    assert features['views'] == 10000
    assert features['engagement_score'] == (500 + 50) / 10000
