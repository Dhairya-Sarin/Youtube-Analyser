import pandas as pd
from datetime import datetime
from typing import Dict, Any, List, Optional
from .base_extractor import BaseFeatureExtractor
from config.constants import PRIME_TIME_HOURS, WEEKEND_DAYS, SPECIAL_DAYS


class TemporalFeatureExtractor(BaseFeatureExtractor):
    """Extract time-based features from video upload time"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.feature_names = [
            'publish_hour', 'publish_day_of_week', 'publish_month', 'publish_year',
            'is_weekend', 'is_prime_time', 'is_morning', 'is_afternoon',
            'is_evening', 'is_night', 'days_since_upload', 'upload_season',
            'is_holiday_season', 'is_summer_break', 'is_back_to_school',
            'upload_daylight_flag', 'upload_on_special_day'
        ]

    def extract(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract temporal features"""
        features = {}

        published_at = data.get('published_at') or data.get('publishedAt', '')
        if not published_at:
            return {name: 0 for name in self.feature_names}

        try:
            # Parse datetime
            if isinstance(published_at, str):
                # Handle ISO format with Z
                published_at = published_at.replace('Z', '+00:00')
                dt = datetime.fromisoformat(published_at)
            else:
                dt = published_at

            # Basic time features
            features['publish_hour'] = dt.hour
            features['publish_day_of_week'] = dt.weekday()  # 0=Monday
            features['publish_month'] = dt.month
            features['publish_year'] = dt.year

            # Time of day categories
            features['is_morning'] = 1 if 6 <= dt.hour < 12 else 0
            features['is_afternoon'] = 1 if 12 <= dt.hour < 18 else 0
            features['is_evening'] = 1 if 18 <= dt.hour < 24 else 0
            features['is_night'] = 1 if 0 <= dt.hour < 6 else 0

            # Weekend/weekday
            features['is_weekend'] = 1 if dt.weekday() in WEEKEND_DAYS else 0
            features['is_prime_time'] = 1 if dt.hour in PRIME_TIME_HOURS else 0

            # Days since upload
            features['days_since_upload'] = (datetime.now() - dt.replace(tzinfo=None)).days

            # Season
            features['upload_season'] = self._get_season(dt.month)

            # Holiday and special periods
            features['is_holiday_season'] = 1 if dt.month in [11, 12] else 0
            features['is_summer_break'] = 1 if dt.month in [6, 7, 8] else 0
            features['is_back_to_school'] = 1 if dt.month in [8, 9] else 0

            # Daylight (Northern Hemisphere approximation)
            features['upload_daylight_flag'] = self._is_daylight_time(dt)

            # Special days
            features['upload_on_special_day'] = self._is_special_day(dt)

        except Exception as e:
            print(f"Error parsing temporal features: {e}")
            features = {name: 0 for name in self.feature_names}

        return features

    def _get_season(self, month: int) -> int:
        """Get season number (0-3) from month"""
        if month in [12, 1, 2]:
            return 0  # Winter
        elif month in [3, 4, 5]:
            return 1  # Spring
        elif month in [6, 7, 8]:
            return 2  # Summer
        else:
            return 3  # Fall

    def _is_daylight_time(self, dt: datetime) -> int:
        """Approximate daylight saving time (US)"""
        # Simplified: March to November
        return 1 if 3 <= dt.month <= 11 else 0

    def _is_special_day(self, dt: datetime) -> int:
        """Check if upload date is on a special day"""
        day_month = (dt.month, dt.day)

        for holiday, dates in SPECIAL_DAYS.items():
            if day_month in dates:
                return 1

        return 0

    def get_feature_names(self) -> List[str]:
        return self.feature_names