import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import logging


class PredictionEngine:
    """Handles model predictions and performance analysis"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def predict_video_performance(self,
                                  video_features: Dict[str, Any],
                                  model_results: Dict[str, Any]) -> Dict[str, Any]:
        """Predict performance for a single video"""
        try:
            model = model_results['model']
            scaler = model_results['scaler']
            feature_columns = model_results['feature_columns']

            # Create feature vector
            feature_vector = []
            for feature in feature_columns:
                value = video_features.get(feature, 0)
                feature_vector.append(value)

            # Scale features
            feature_vector = np.array(feature_vector).reshape(1, -1)
            feature_vector_scaled = scaler.transform(feature_vector)

            # Make prediction
            prediction = model.predict(feature_vector_scaled)[0]

            # Calculate confidence intervals (simplified)
            confidence_interval = self._calculate_confidence_interval(
                model, feature_vector_scaled, model_results
            )

            return {
                'predicted_views': max(0, int(prediction)),
                'confidence_interval': confidence_interval,
                'prediction_quality': self._assess_prediction_quality(model_results),
                'contributing_factors': self._identify_contributing_factors(
                    video_features, model_results
                )
            }

        except Exception as e:
            self.logger.error(f"Error predicting video performance: {e}")
            return {
                'predicted_views': 0,
                'confidence_interval': (0, 0),
                'prediction_quality': 'low',
                'contributing_factors': []
            }

    def batch_predict(self,
                      df: pd.DataFrame,
                      model_results: Dict[str, Any]) -> pd.DataFrame:
        """Make predictions for multiple videos"""
        try:
            model = model_results['model']
            scaler = model_results['scaler']
            feature_columns = model_results['feature_columns']

            # Prepare features
            X = df[feature_columns].fillna(0)
            X_scaled = scaler.transform(X)

            # Make predictions
            predictions = model.predict(X_scaled)

            # Add predictions to dataframe
            result_df = df.copy()
            result_df['predicted_views'] = np.maximum(0, predictions.astype(int))

            # Calculate prediction errors for existing videos
            if 'views' in df.columns:
                result_df['prediction_error'] = np.abs(result_df['views'] - result_df['predicted_views'])
                result_df['prediction_error_pct'] = (
                        result_df['prediction_error'] / np.maximum(result_df['views'], 1) * 100
                )

            return result_df

        except Exception as e:
            self.logger.error(f"Error in batch prediction: {e}")
            return df

    def _calculate_confidence_interval(self,
                                       model,
                                       feature_vector: np.ndarray,
                                       model_results: Dict[str, Any]) -> Tuple[int, int]:
        """Calculate confidence interval for prediction"""
        try:
            # For ensemble models, use prediction variance
            if hasattr(model, 'estimators_'):
                # Get predictions from individual estimators
                individual_predictions = []
                for estimator in model.estimators_:
                    pred = estimator.predict(feature_vector)[0]
                    individual_predictions.append(pred)

                mean_pred = np.mean(individual_predictions)
                std_pred = np.std(individual_predictions)

                # 95% confidence interval
                lower = max(0, int(mean_pred - 1.96 * std_pred))
                upper = int(mean_pred + 1.96 * std_pred)

                return (lower, upper)
            else:
                # For single models, use MSE-based estimate
                mse = model_results.get('mse', 0)
                prediction = model.predict(feature_vector)[0]

                margin = 1.96 * np.sqrt(mse)  # 95% confidence
                lower = max(0, int(prediction - margin))
                upper = int(prediction + margin)

                return (lower, upper)

        except:
            prediction = model.predict(feature_vector)[0]
            return (int(prediction * 0.7), int(prediction * 1.3))

    def _assess_prediction_quality(self, model_results: Dict[str, Any]) -> str:
        """Assess the quality of predictions based on model performance"""
        r2_score = model_results.get('r2_score', 0)

        if r2_score > 0.8:
            return 'high'
        elif r2_score > 0.6:
            return 'medium'
        elif r2_score > 0.3:
            return 'low'
        else:
            return 'very_low'

    def _identify_contributing_factors(self,
                                       video_features: Dict[str, Any],
                                       model_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify the main factors contributing to the prediction"""
        feature_importance = model_results.get('feature_importance', [])

        contributing_factors = []

        # Get top 5 most important features
        top_features = feature_importance[:5]

        for feature_info in top_features:
            feature_name = feature_info['feature']
            importance = feature_info['importance']
            value = video_features.get(feature_name, 0)

            factor = {
                'feature': feature_name,
                'importance': importance,
                'value': value,
                'description': self._describe_feature_contribution(feature_name, value, importance)
            }

            contributing_factors.append(factor)

        return contributing_factors

    def _describe_feature_contribution(self, feature_name: str, value: float, importance: float) -> str:
        """Describe how a feature contributes to the prediction"""
        descriptions = {
            'title_length': f'Title has {int(value)} characters',
            'thumbnail_brightness': f'Thumbnail brightness: {value:.1f}',
            'publish_hour': f'Posted at hour {int(value)}',
            'clickbait_score': f'Clickbait score: {value:.1f}',
            'engagement_score': f'Engagement rate: {value:.3f}',
            'duration_seconds': f'Video duration: {int(value / 60)} minutes',
            'is_weekend': 'Posted on weekend' if value > 0.5 else 'Posted on weekday',
            'thumbnail_face_count': f'Faces in thumbnail: {int(value)}'
        }

        base_desc = descriptions.get(feature_name, f'{feature_name}: {value:.2f}')
        impact = 'high' if importance > 0.1 else 'moderate' if importance > 0.05 else 'low'

        return f"{base_desc} (impact: {impact})"