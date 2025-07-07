import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
import logging
from core.exceptions import ValidationError


class DataValidator:
    """Validates and cleans video data"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # Required fields for analysis
        self.required_fields = [
            'id', 'title', 'published_at', 'view_count', 'like_count'
        ]

        # Numeric fields that should be positive
        self.positive_numeric_fields = [
            'view_count', 'like_count', 'comment_count', 'duration_seconds'
        ]

    def validate_video_data(self, videos: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Validate and clean list of video data"""
        if not videos:
            raise ValidationError("No video data provided")

        validated_videos = []

        for i, video in enumerate(videos):
            try:
                cleaned_video = self._validate_single_video(video)
                if cleaned_video:
                    validated_videos.append(cleaned_video)
            except Exception as e:
                self.logger.warning(f"Skipping invalid video at index {i}: {e}")
                continue

        if not validated_videos:
            raise ValidationError("No valid videos after validation")

        self.logger.info(f"Validated {len(validated_videos)} out of {len(videos)} videos")
        return validated_videos

    def _validate_single_video(self, video: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Validate and clean a single video record"""
        # Check required fields
        for field in self.required_fields:
            if field not in video or video[field] is None:
                raise ValidationError(f"Missing required field: {field}")

        # Create cleaned copy
        cleaned = video.copy()

        # Validate and clean numeric fields
        for field in self.positive_numeric_fields:
            if field in cleaned:
                try:
                    value = float(cleaned[field])
                    if value < 0:
                        cleaned[field] = 0
                    elif np.isnan(value) or np.isinf(value):
                        cleaned[field] = 0
                    else:
                        cleaned[field] = int(value)
                except (ValueError, TypeError):
                    cleaned[field] = 0

        # Validate text fields
        text_fields = ['title', 'description']
        for field in text_fields:
            if field in cleaned:
                if not isinstance(cleaned[field], str):
                    cleaned[field] = str(cleaned[field]) if cleaned[field] is not None else ''
                # Remove null characters and excessive whitespace
                cleaned[field] = cleaned[field].replace('\x00', '').strip()

        # Validate lists
        list_fields = ['tags']
        for field in list_fields:
            if field in cleaned:
                if not isinstance(cleaned[field], list):
                    cleaned[field] = []
                # Clean tag list
                if field == 'tags':
                    cleaned[field] = [tag.strip() for tag in cleaned[field] if tag and isinstance(tag, str)]

        # Validate dates
        if 'published_at' in cleaned:
            try:
                from datetime import datetime
                if isinstance(cleaned['published_at'], str):
                    # Try to parse ISO format
                    datetime.fromisoformat(cleaned['published_at'].replace('Z', '+00:00'))
            except:
                raise ValidationError("Invalid published_at format")

        # Additional business logic validation
        if cleaned.get('view_count', 0) == 0 and cleaned.get('like_count', 0) > 0:
            # Suspicious: likes without views
            self.logger.warning(f"Video {cleaned.get('id', 'unknown')} has likes but no views")

        return cleaned

    def validate_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean a pandas DataFrame"""
        if df.empty:
            raise ValidationError("Empty DataFrame provided")

        # Remove duplicate videos
        if 'video_id' in df.columns:
            initial_count = len(df)
            df = df.drop_duplicates(subset=['video_id'])
            if len(df) < initial_count:
                self.logger.info(f"Removed {initial_count - len(df)} duplicate videos")

        # Handle missing values
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = df[numeric_columns].fillna(0)

        # Handle infinite values
        df.replace([np.inf, -np.inf], 0, inplace=True)

        # Validate feature ranges
        df = self._validate_feature_ranges(df)

        return df

    def _validate_feature_ranges(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate that features are within expected ranges"""

        # Ratio features should be between 0 and 1
        ratio_features = [col for col in df.columns if 'ratio' in col.lower()]
        for feature in ratio_features:
            if feature in df.columns:
                df[feature] = df[feature].clip(0, 1)

        # Percentage features should be between 0 and 1
        percentage_features = [col for col in df.columns if 'percentage' in col.lower() or col.endswith('_pct')]
        for feature in percentage_features:
            if feature in df.columns:
                df[feature] = df[feature].clip(0, 1)

        # Count features should be non-negative
        count_features = [col for col in df.columns if 'count' in col.lower()]
        for feature in count_features:
            if feature in df.columns:
                df[feature] = df[feature].clip(0, None)

        # Score features should typically be between 0 and 1 or 0 and 10
        score_features = [col for col in df.columns if 'score' in col.lower()]
        for feature in score_features:
            if feature in df.columns:
                # Determine if it's a 0-1 or 0-10 scale based on max value
                max_val = df[feature].max()
                if max_val > 5:
                    df[feature] = df[feature].clip(0, 10)
                else:
                    df[feature] = df[feature].clip(0, 1)

        return df