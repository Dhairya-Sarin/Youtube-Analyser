from datetime import datetime
from typing import Dict, Any, List
import numpy as np
from .base_extractor import BaseFeatureExtractor


class ChannelFeatureExtractor(BaseFeatureExtractor):
    """Extract channel-level features"""

    def __init__(self, config=None):
        super().__init__(config)
        self.feature_names = [
            'channel_age_days', 'channel_total_views', 'channel_total_subscribers',
            'channel_total_videos', 'channel_avg_views_per_video', 'channel_subscriber_view_ratio',
            'channel_upload_frequency', 'channel_recent_upload_gap', 'channel_consistency_score',
            'channel_growth_rate_proxy', 'channel_engagement_rate', 'channel_content_diversity',
            'channel_description_length', 'channel_has_custom_url', 'channel_verified_status',
            'channel_country_specified', 'channel_branding_consistency'
        ]

    def extract(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract channel-level features"""
        # Channel info should be passed in data
        channel_info = {
            'published_at': data.get('published_at'),
            'subscriber_count': data.get('subscriber_count', 0),
            'video_count': data.get('video_count', 0),
            'view_count': data.get('view_count', 0),
            'description': data.get('description', ''),
            'custom_url': data.get('custom_url', ''),
            'country': data.get('country', ''),
            'title': data.get('channel_title', data.get('title', ''))
        }

        features = {}

        # Basic channel metrics
        features.update(self._extract_basic_metrics(channel_info))

        # Channel behavior patterns
        features.update(self._extract_behavior_patterns(channel_info))

        # Channel branding and setup
        features.update(self._extract_branding_features(channel_info))

        return features

    def _extract_basic_metrics(self, channel_info: Dict[str, Any]) -> Dict[str, Any]:
        """Extract basic channel metrics"""
        features = {}

        # Channel age
        if channel_info.get('published_at'):
            try:
                created_date = datetime.fromisoformat(channel_info['published_at'].replace('Z', '+00:00'))
                age_days = (datetime.now() - created_date.replace(tzinfo=None)).days
                features['channel_age_days'] = age_days
            except:
                features['channel_age_days'] = 0
        else:
            features['channel_age_days'] = 0

        # Basic counts
        total_views = channel_info.get('view_count', 0)
        total_subscribers = channel_info.get('subscriber_count', 0)
        total_videos = channel_info.get('video_count', 0)

        features['channel_total_views'] = total_views
        features['channel_total_subscribers'] = total_subscribers
        features['channel_total_videos'] = total_videos

        # Derived metrics
        features['channel_avg_views_per_video'] = total_views / max(total_videos, 1)
        features['channel_subscriber_view_ratio'] = total_subscribers / max(total_views, 1)

        return features

    def _extract_behavior_patterns(self, channel_info: Dict[str, Any]) -> Dict[str, Any]:
        """Extract channel behavior and activity patterns"""
        features = {}

        # Upload frequency (videos per day since creation)
        age_days = features.get('channel_age_days', 1)
        total_videos = channel_info.get('video_count', 0)

        features['channel_upload_frequency'] = total_videos / max(age_days, 1)

        # Engagement rate proxy (simplified)
        # Note: Real engagement rate would require video-level data aggregation
        total_views = channel_info.get('view_count', 0)
        total_subscribers = channel_info.get('subscriber_count', 0)

        if total_subscribers > 0:
            features['channel_engagement_rate'] = min(total_views / (total_subscribers * max(total_videos, 1)), 10.0)
        else:
            features['channel_engagement_rate'] = 0

        # Growth rate proxy (views per subscriber)
        features['channel_growth_rate_proxy'] = total_views / max(total_subscribers, 1)

        # Consistency score (simplified - based on regular upload pattern assumption)
        # In real implementation, this would analyze actual upload dates
        expected_uploads = age_days * 0.1  # Assume ~1 video per 10 days as baseline
        actual_uploads = total_videos
        features['channel_consistency_score'] = min(actual_uploads / max(expected_uploads, 1), 2.0)

        # Recent upload gap (would need video dates in real implementation)
        # For now, use a proxy based on total activity
        features['channel_recent_upload_gap'] = max(30 - (total_videos / max(age_days / 30, 1)), 0)

        # Content diversity proxy (would need topic analysis in real implementation)
        # Simple proxy based on video count and age
        features['channel_content_diversity'] = min(np.log1p(total_videos) / 5.0, 1.0)

        return features

    def _extract_branding_features(self, channel_info: Dict[str, Any]) -> Dict[str, Any]:
        """Extract branding and setup features"""
        features = {}

        # Description
        description = channel_info.get('description', '')
        features['channel_description_length'] = len(description)

        # Custom URL
        features['channel_has_custom_url'] = 1 if channel_info.get('custom_url') else 0

        # Country
        features['channel_country_specified'] = 1 if channel_info.get('country') else 0

        # Verified status (would need additional API call in real implementation)
        features['channel_verified_status'] = 0  # Placeholder

        # Branding consistency (simplified heuristic)
        # Check if channel name appears in description
        channel_title = channel_info.get('title', '').lower()
        description_lower = description.lower()

        branding_score = 0
        if channel_title and channel_title in description_lower:
            branding_score += 0.3
        if len(description) > 100:  # Has substantial description
            branding_score += 0.3
        if channel_info.get('custom_url'):  # Has custom URL
            branding_score += 0.4

        features['channel_branding_consistency'] = branding_score

        return features

    def get_feature_names(self) -> List[str]:
        return self.feature_names
