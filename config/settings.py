import os
from dataclasses import dataclass
from typing import Optional, List, Dict
from enum import Enum

class ModelType(Enum):
    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOST = "gradient_boost"
    NEURAL_NETWORK = "neural_network"
    ENSEMBLE = "ensemble"

class FeatureSet(Enum):
    BASIC = "basic"
    EXTENDED = "extended"
    FULL = "full"
    CUSTOM = "custom"

@dataclass
class YouTubeConfig:
    api_key: str
    max_videos_per_channel: int = 500
    max_requests_per_minute: int = 100
    timeout_seconds: int = 30
    retry_attempts: int = 3

@dataclass
class FeatureConfig:
    include_thumbnails: bool = True
    include_audio_analysis: bool = False
    include_advanced_nlp: bool = True
    include_vision_models: bool = False
    feature_set: FeatureSet = FeatureSet.EXTENDED
    custom_features: Optional[List[str]] = None

@dataclass
class MLConfig:
    model_type: ModelType = ModelType.RANDOM_FOREST
    test_size: float = 0.2
    random_state: int = 42
    enable_shap: bool = True
    enable_cross_validation: bool = True
    hyperparameter_tuning: bool = False

@dataclass
class ExportConfig:
    export_excel: bool = True
    export_csv: bool = True
    export_json: bool = True
    include_visualizations: bool = True
    compress_exports: bool = False

@dataclass
class CacheConfig:
    enable_caching: bool = True
    cache_dir: str = "./cache"
    cache_ttl_hours: int = 24
    max_cache_size_mb: int = 1000

@dataclass
class AppSettings:
    youtube: YouTubeConfig
    features: FeatureConfig
    ml: MLConfig
    export: ExportConfig
    cache: CacheConfig
    debug: bool = False
    log_level: str = "INFO"

# Default settings
def get_default_settings() -> AppSettings:
    return AppSettings(
        youtube=YouTubeConfig(
            api_key=os.getenv("YOUTUBE_API_KEY", ""),
            max_videos_per_channel=100,
            max_requests_per_minute=100
        ),
        features=FeatureConfig(),
        ml=MLConfig(),
        export=ExportConfig(),
        cache=CacheConfig()
    )
