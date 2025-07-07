from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import pandas as pd


class BaseFeatureExtractor(ABC):
    """Base class for all feature extractors"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.feature_names = []

    @abstractmethod
    def extract(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract features from input data"""
        pass

    @abstractmethod
    def get_feature_names(self) -> List[str]:
        """Return list of feature names this extractor produces"""
        pass

    def validate_input(self, data: Dict[str, Any]) -> bool:
        """Validate input data format"""
        return True