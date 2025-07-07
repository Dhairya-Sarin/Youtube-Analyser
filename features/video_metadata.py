from datetime import datetime
from typing import Dict, Any, List, Optional
import re
from .base_extractor import BaseFeatureExtractor
from config.constants import VIDEO_CATEGORIES


class VideoMetadataExtractor(BaseFeatureExtractor):
    """Extract basic video metadata features"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.feature_names = [
            'video_id_hash', 'title_length', 'description_length',
            'duration_seconds', 'category_id', 'tag_count',
            'is_live', 'is_short', 'is_monetized', 'is_premiere',
            'has_captions', 'definition_hd', 'projection_360'
        ]

    def extract(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract metadata features"""
        features = {}

        # Basic identifiers
        features['video_id_hash'] = hash(data.get('video_id', '')) % 10000
        features['channel_id_hash'] = hash(data.get('channel_id', '')) % 10000

        # Text metadata
        title = data.get('title', '')
        description = data.get('description', '')
        tags = data.get('tags', [])

        features['title_length'] = len(title)
        features['description_length'] = len(description)
        features['tag_count'] = len(tags) if isinstance(tags, list) else 0

        # Duration
        duration = data.get('duration', '')
        features['duration_seconds'] = self._parse_duration(duration)

        # Video type classification
        features['is_short'] = 1 if features['duration_seconds'] < 60 else 0
        features['is_long'] = 1 if features['duration_seconds'] > 1800 else 0  # 30 min

        # Category
        features['category_id'] = data.get('categoryId', 0)
        features['category_name'] = VIDEO_CATEGORIES.get(features['category_id'], 'Unknown')

        # Video properties
        features['is_live'] = 1 if data.get('liveBroadcastContent') == 'live' else 0
        features['is_premiere'] = 1 if data.get('liveBroadcastContent') == 'upcoming' else 0

        # Technical properties
        definition = data.get('contentDetails', {}).get('definition', 'sd')
        features['definition_hd'] = 1 if definition == 'hd' else 0

        projection = data.get('contentDetails', {}).get('projection', 'rectangular')
        features['projection_360'] = 1 if projection == '360' else 0

        # Captions
        features['has_captions'] = 1 if data.get('contentDetails', {}).get('caption') == 'true' else 0

        # Monetization indicators (approximate)
        features['is_monetized'] = self._estimate_monetization(data)

        return features

    def _parse_duration(self, duration_str: str) -> int:
        """Parse ISO 8601 duration to seconds"""
        if not duration_str:
            return 0

        try:
            duration_str = duration_str.replace('PT', '')

            hours = 0
            minutes = 0
            seconds = 0

            if 'H' in duration_str:
                hours = int(duration_str.split('H')[0])
                duration_str = duration_str.split('H')[1]

            if 'M' in duration_str:
                minutes = int(duration_str.split('M')[0])
                duration_str = duration_str.split('M')[1]

            if 'S' in duration_str:
                seconds = int(duration_str.split('S')[0])

            return hours * 3600 + minutes * 60 + seconds
        except:
            return 0

    def _estimate_monetization(self, data: Dict[str, Any]) -> int:
        """Estimate if video is monetized based on available data"""
        # This is an approximation - actual monetization data isn't in public API
        description = data.get('description', '').lower()

        # Look for monetization indicators
        monetization_indicators = [
            'sponsor', 'affiliate', 'discount', 'promo code',
            'advertisement', 'paid promotion', 'brand deal'
        ]

        return 1 if any(indicator in description for indicator in monetization_indicators) else 0

    def get_feature_names(self) -> List[str]:
        return self.feature_names
