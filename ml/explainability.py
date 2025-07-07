import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional
import tempfile
import io
import base64
import logging


class SHAPExplainer:
    """SHAP-based model explainability"""

    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Try to import SHAP
        try:
            import shap
            self.shap = shap
            self.shap_available = True
        except ImportError:
            self.shap = None
            self.shap_available = False
            self.logger.warning("SHAP not available. Install shap package for model explainability.")

    async def explain_model(self, model, X: pd.DataFrame, sample_size: int = 100) -> Dict[str, Any]:
        """Generate SHAP explanations for model"""
        if not self.shap_available:
            return {}

        try:
            # Limit sample size for performance
            if len(X) > sample_size:
                X_sample = X.sample(n=sample_size, random_state=42)
            else:
                X_sample = X

            # Create explainer
            explainer = self.shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)

            results = {}

            # Generate summary plot
            summary_plot = self._create_summary_plot(shap_values, X_sample)
            if summary_plot:
                results['summary_plot'] = summary_plot

            # Generate waterfall plot for first prediction
            waterfall_plot = self._create_waterfall_plot(explainer, shap_values, X_sample)
            if waterfall_plot:
                results['waterfall_plot'] = waterfall_plot

            # Feature importance based on SHAP values
            feature_importance = self._calculate_shap_importance(shap_values, X_sample.columns)
            results['feature_importance'] = feature_importance

            # Generate textual explanations
            explanations = self._generate_explanations(shap_values, X_sample.columns)
            results['explanations'] = explanations

            return results

        except Exception as e:
            self.logger.error(f"Error generating SHAP explanations: {e}")
            return {}

    def _create_summary_plot(self, shap_values: np.ndarray, X: pd.DataFrame) -> Optional[str]:
        """Create SHAP summary plot"""
        try:
            plt.figure(figsize=(12, 8))
            self.shap.summary_plot(shap_values, X, show=False, max_display=20)
            plt.tight_layout()

            # Convert to base64 string
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()

            return image_base64

        except Exception as e:
            self.logger.error(f"Error creating summary plot: {e}")
            return None

    def _create_waterfall_plot(self, explainer, shap_values: np.ndarray, X: pd.DataFrame) -> Optional[str]:
        """Create SHAP waterfall plot for first prediction"""
        try:
            plt.figure(figsize=(12, 8))

            # Use first sample for waterfall plot
            sample_idx = 0
            expected_value = explainer.expected_value
            shap_values_sample = shap_values[sample_idx]

            # Create simple waterfall-style plot
            features = X.columns[:15]  # Top 15 features
            values = shap_values_sample[:15]

            # Sort by absolute value
            sorted_indices = np.argsort(np.abs(values))[::-1]
            features = features[sorted_indices]
            values = values[sorted_indices]

            # Create horizontal bar plot
            colors = ['red' if v < 0 else 'blue' for v in values]
            plt.barh(range(len(values)), values, color=colors)
            plt.yticks(range(len(values)), features)
            plt.xlabel('SHAP Value')
            plt.title('SHAP Feature Contributions')
            plt.tight_layout()

            # Convert to base64 string
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()

            return image_base64

        except Exception as e:
            self.logger.error(f"Error creating waterfall plot: {e}")
            return None

    def _calculate_shap_importance(self, shap_values: np.ndarray, feature_names: List[str]) -> List[Dict[str, Any]]:
        """Calculate feature importance based on SHAP values"""
        # Calculate mean absolute SHAP values
        importance_scores = np.mean(np.abs(shap_values), axis=0)

        importance_data = []
        for i, (feature, score) in enumerate(zip(feature_names, importance_scores)):
            importance_data.append({
                'feature': feature,
                'importance': float(score),
                'rank': i + 1
            })

        # Sort by importance
        importance_data.sort(key=lambda x: x['importance'], reverse=True)

        # Update ranks
        for i, item in enumerate(importance_data):
            item['rank'] = i + 1

        return importance_data

    def _generate_explanations(self, shap_values: np.ndarray, feature_names: List[str]) -> Dict[
        str, List[Dict[str, str]]]:
        """Generate textual explanations of SHAP results"""
        explanations = {
            'positive_contributors': [],
            'negative_contributors': [],
            'neutral_features': []
        }

        # Calculate average SHAP values for each feature
        avg_shap_values = np.mean(shap_values, axis=0)

        for i, (feature, avg_shap) in enumerate(zip(feature_names, avg_shap_values)):
            explanation = self._explain_feature_impact(feature, avg_shap)

            feature_explanation = {
                'feature': feature,
                'explanation': explanation,
                'impact_score': float(avg_shap)
            }

            if avg_shap > 0.01:
                explanations['positive_contributors'].append(feature_explanation)
            elif avg_shap < -0.01:
                explanations['negative_contributors'].append(feature_explanation)
            else:
                explanations['neutral_features'].append(feature_explanation)

        # Sort by absolute impact
        for category in explanations:
            explanations[category].sort(key=lambda x: abs(x['impact_score']), reverse=True)

        return explanations

    def _explain_feature_impact(self, feature_name: str, impact: float) -> str:
        """Generate human-readable explanation for feature impact"""

        feature_explanations = {
            'title_length': 'longer titles' if impact > 0 else 'shorter titles',
            'thumbnail_brightness': 'brighter thumbnails' if impact > 0 else 'darker thumbnails',
            'publish_hour': 'posting at optimal times' if impact > 0 else 'posting at suboptimal times',
            'clickbait_score': 'more clickbait elements' if impact > 0 else 'fewer clickbait elements',
            'engagement_score': 'higher engagement rates' if impact > 0 else 'lower engagement rates',
            'duration_seconds': 'longer videos' if impact > 0 else 'shorter videos',
            'is_weekend': 'weekend posting' if impact > 0 else 'weekday posting',
            'thumbnail_face_count': 'more faces in thumbnails' if impact > 0 else 'fewer faces in thumbnails'
        }

        base_explanation = feature_explanations.get(feature_name,
                                                    f'higher {feature_name}' if impact > 0 else f'lower {feature_name}')

        direction = "increase" if impact > 0 else "decrease"
        strength = "strongly" if abs(impact) > 0.1 else "moderately" if abs(impact) > 0.05 else "slightly"

        return f"Videos with {base_explanation} tend to {strength} {direction} in views"