import numpy as np
from typing import Dict, Any, List, Optional
from .base_extractor import BaseFeatureExtractor


class EngagementMetricsExtractor(BaseFeatureExtractor):
    """Extract engagement and performance metrics"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.feature_names = [
            'views', 'likes', 'dislikes', 'comments', 'favorites',
            'engagement_score', 'like_ratio', 'dislike_ratio', 'comment_ratio',
            'views_per_day', 'likes_per_day', 'comments_per_day',
            'viral_potential_score', 'retention_proxy', 'shareability_score'
        ]

    def extract(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract engagement features"""
        features = {}

        # Raw metrics
        views = int(data.get('viewCount', 0))
        likes = int(data.get('likeCount', 0))
        dislikes = int(data.get('dislikeCount', 0))  # Note: Not available after 2021
        comments = int(data.get('commentCount', 0))
        favorites = int(data.get('favoriteCount', 0))

        features['views'] = views
        features['likes'] = likes
        features['dislikes'] = dislikes
        features['comments'] = comments
        features['favorites'] = favorites

        # Engagement ratios (handle division by zero)
        features['engagement_score'] = (likes + comments) / max(views, 1)
        features['like_ratio'] = likes / max(likes + dislikes, 1)
        features['dislike_ratio'] = dislikes / max(likes + dislikes, 1)
        features['comment_ratio'] = comments / max(views, 1)

        # Time-normalized metrics
        days_since_upload = data.get('days_since_upload', 1)
        days_since_upload = max(days_since_upload, 1)  # Avoid division by zero

        features['views_per_day'] = views / days_since_upload
        features['likes_per_day'] = likes / days_since_upload
        features['comments_per_day'] = comments / days_since_upload

        # Advanced engagement scores
        features['viral_potential_score'] = self._calculate_viral_potential(
            views, likes, comments, days_since_upload
        )

        features['retention_proxy'] = self._estimate_retention(
            views, likes, comments, data.get('duration_seconds', 1)
        )

        features['shareability_score'] = self._calculate_shareability(
            likes, comments, views
        )

        return features

    def _calculate_viral_potential(self, views: int, likes: int, comments: int, days: int) -> float:
        """Calculate viral potential score"""
        if views == 0 or days == 0:
            return 0.0

        # Combination of engagement rate and velocity
        engagement_rate = (likes + comments * 2) / views  # Comments weighted more
        velocity = views / days

        # Normalize and combine
        viral_score = np.log1p(velocity) * engagement_rate
        return min(viral_score, 10.0)  # Cap at 10

    def _estimate_retention(self, views: int, likes: int, comments: int, duration: int) -> float:
        """Estimate retention based on engagement patterns"""
        if views == 0 or duration == 0:
            return 0.0

        # Assumption: higher engagement suggests better retention
        engagement_rate = (likes + comments) / views

        # Longer videos with high engagement suggest good retention
        duration_factor = min(duration / 600, 2.0)  # Normalize to 10 minutes, cap at 2x

        retention_proxy = engagement_rate * duration_factor
        return min(retention_proxy, 1.0)

    def _calculate_shareability(self, likes: int, comments: int, views: int) -> float:
        """Calculate how shareable content appears to be"""
        if views == 0:
            return 0.0

        # High like-to-view ratio suggests shareability
        like_share_factor = likes / views

        # Comments suggest discussion-worthy content
        comment_share_factor = comments / views

        # Combine with weights
        shareability = (like_share_factor * 0.7) + (comment_share_factor * 0.3)
        return min(shareability, 1.0)

    def get_feature_names(self) -> List[str]:
        return self.feature_names