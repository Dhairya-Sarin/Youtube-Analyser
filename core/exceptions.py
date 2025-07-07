class YouTubeAnalyticsError(Exception):
    """Base exception for YouTube Analytics"""
    pass

class DataCollectionError(YouTubeAnalyticsError):
    """Error in data collection process"""
    pass

class FeatureExtractionError(YouTubeAnalyticsError):
    """Error in feature extraction process"""
    pass

class ModelTrainingError(YouTubeAnalyticsError):
    """Error in model training process"""
    pass

class AnalysisError(YouTubeAnalyticsError):
    """Error in analysis process"""
    pass

class APIError(YouTubeAnalyticsError):
    """Error in API communication"""
    pass

class ValidationError(YouTubeAnalyticsError):
    """Error in data validation"""
    pass

class CacheError(YouTubeAnalyticsError):
    """Error in caching operations"""
    pass