import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.feature_selection import SelectKBest, f_regression, RFE
import joblib
import tempfile
from config.settings import MLConfig, ModelType
from core.exceptions import ModelTrainingError


class MLModelManager:
    """Manages machine learning model training and prediction"""

    def __init__(self, config: MLConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.models = {}
        self.scalers = {}

    async def train_model(self, df: pd.DataFrame, target_column: str = 'views') -> Optional[Dict[str, Any]]:
        """Train ML model to predict target variable"""
        try:
            if len(df) < 50:  # Minimum samples for training
                self.logger.warning("Insufficient data for model training")
                return None

            # Prepare data
            X, y, feature_columns = self._prepare_training_data(df, target_column)

            if X is None or len(X) == 0:
                return None

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.config.test_size, random_state=self.config.random_state
            )

            # Scale features
            scaler = self._get_scaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Train model
            model = self._get_model()

            if self.config.hyperparameter_tuning:
                model = self._tune_hyperparameters(model, X_train_scaled, y_train)

            model.fit(X_train_scaled, y_train)

            # Evaluate model
            y_pred = model.predict(X_test_scaled)

            # Calculate metrics
            r2 = r2_score(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)

            # Cross-validation if enabled
            cv_scores = None
            if self.config.enable_cross_validation:
                cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')

            # Feature importance
            feature_importance = self._get_feature_importance(model, feature_columns)

            # Feature selection analysis
            feature_selection_results = self._analyze_feature_selection(X_train_scaled, y_train, feature_columns)

            results = {
                'model': model,
                'scaler': scaler,
                'feature_columns': feature_columns,
                'r2_score': r2,
                'mse': mse,
                'mae': mae,
                'cv_scores': cv_scores,
                'cv_mean': np.mean(cv_scores) if cv_scores is not None else None,
                'cv_std': np.std(cv_scores) if cv_scores is not None else None,
                'feature_importance': feature_importance,
                'feature_selection': feature_selection_results,
                'training_samples': len(X_train),
                'test_samples': len(X_test),
                'model_type': self.config.model_type.value
            }

            self.logger.info(f"Model trained successfully. R² score: {r2:.3f}")
            return results

        except Exception as e:
            self.logger.error(f"Error training model: {e}")
            raise ModelTrainingError(f"Model training failed: {e}")

    async def predict(self, df: pd.DataFrame, model_results: Dict[str, Any]) -> np.ndarray:
        """Make predictions using trained model"""
        try:
            model = model_results['model']
            scaler = model_results['scaler']
            feature_columns = model_results['feature_columns']

            # Prepare data
            X = df[feature_columns].fillna(0)
            X_scaled = scaler.transform(X)

            # Make predictions
            predictions = model.predict(X_scaled)

            return predictions

        except Exception as e:
            self.logger.error(f"Error making predictions: {e}")
            return np.array([])

    def _prepare_training_data(self, df: pd.DataFrame, target_column: str) -> Tuple[
        Optional[np.ndarray], Optional[np.ndarray], List[str]]:
        """Prepare data for model training"""
        if target_column not in df.columns:
            self.logger.error(f"Target column '{target_column}' not found in data")
            return None, None, []

        # Remove non-numeric and metadata columns
        exclude_columns = [
            'video_id', 'title', 'description', 'published_at', 'channel_title',
            'thumbnail_url', 'channel_name', target_column, 'video_id_hash', 'channel_id_hash'
        ]

        feature_columns = [col for col in df.columns
                           if col not in exclude_columns and
                           df[col].dtype in ['int64', 'float64', 'int32', 'float32']]

        if not feature_columns:
            self.logger.error("No valid feature columns found")
            return None, None, []

        # Prepare features and target
        X = df[feature_columns].fillna(0)
        y = df[target_column]

        # Remove rows with invalid target values
        valid_mask = (y >= 0) & np.isfinite(y)
        X = X[valid_mask]
        y = y[valid_mask]

        # Feature selection based on correlation and variance
        X_filtered, selected_features = self._basic_feature_selection(X, y, feature_columns)

        self.logger.info(f"Using {len(selected_features)} features for training")
        return X_filtered.values, y.values, selected_features

    def _basic_feature_selection(self, X: pd.DataFrame, y: pd.Series, feature_columns: List[str]) -> Tuple[
        pd.DataFrame, List[str]]:
        """Basic feature selection to remove low-variance and uncorrelated features"""
        # Remove constant features
        constant_features = [col for col in X.columns if X[col].nunique() <= 1]
        X_filtered = X.drop(columns=constant_features)

        # Remove features with very low variance
        low_variance_features = []
        for col in X_filtered.columns:
            if X_filtered[col].std() < 1e-6:
                low_variance_features.append(col)

        X_filtered = X_filtered.drop(columns=low_variance_features)

        # Remove features with very low correlation with target
        correlations = X_filtered.corrwith(y).abs()
        low_correlation_features = correlations[correlations < 0.01].index.tolist()
        X_filtered = X_filtered.drop(columns=low_correlation_features)

        selected_features = X_filtered.columns.tolist()

        removed_count = len(feature_columns) - len(selected_features)
        if removed_count > 0:
            self.logger.info(f"Removed {removed_count} low-quality features")

        return X_filtered, selected_features

    def _get_model(self):
        """Get model based on configuration"""
        if self.config.model_type == ModelType.RANDOM_FOREST:
            return RandomForestRegressor(
                n_estimators=100,
                max_depth=15,
                random_state=self.config.random_state,
                n_jobs=-1
            )
        elif self.config.model_type == ModelType.GRADIENT_BOOST:
            return GradientBoostingRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=self.config.random_state
            )
        elif self.config.model_type == ModelType.NEURAL_NETWORK:
            return MLPRegressor(
                hidden_layer_sizes=(100, 50),
                max_iter=500,
                random_state=self.config.random_state,
                early_stopping=True
            )
        else:  # Ensemble
            return self._create_ensemble_model()

    def _create_ensemble_model(self):
        """Create ensemble model combining multiple algorithms"""
        from sklearn.ensemble import VotingRegressor

        rf = RandomForestRegressor(n_estimators=50, random_state=self.config.random_state, n_jobs=-1)
        gb = GradientBoostingRegressor(n_estimators=50, random_state=self.config.random_state)

        ensemble = VotingRegressor([
            ('rf', rf),
            ('gb', gb)
        ])

        return ensemble

    def _get_scaler(self):
        """Get feature scaler"""
        return RobustScaler()  # More robust to outliers than StandardScaler

    def _tune_hyperparameters(self, model, X: np.ndarray, y: np.ndarray):
        """Tune model hyperparameters using grid search"""
        if self.config.model_type == ModelType.RANDOM_FOREST:
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 15, 20, None],
                'min_samples_split': [2, 5, 10]
            }
        elif self.config.model_type == ModelType.GRADIENT_BOOST:
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2]
            }
        else:
            return model  # No tuning for other models

        grid_search = GridSearchCV(
            model, param_grid, cv=3, scoring='r2', n_jobs=-1
        )
        grid_search.fit(X, y)

        self.logger.info(f"Best parameters: {grid_search.best_params_}")
        return grid_search.best_estimator_

    def _get_feature_importance(self, model, feature_columns: List[str]) -> List[Dict[str, Any]]:
        """Extract feature importance from model"""
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_)
        else:
            return []

        importance_data = []
        for i, importance in enumerate(importances):
            importance_data.append({
                'feature': feature_columns[i],
                'importance': float(importance)
            })

        # Sort by importance
        importance_data.sort(key=lambda x: x['importance'], reverse=True)

        return importance_data

    def _analyze_feature_selection(self, X: np.ndarray, y: np.ndarray, feature_columns: List[str]) -> Dict[str, Any]:
        """Analyze different feature selection methods"""
        results = {}

        try:
            # Univariate feature selection
            selector = SelectKBest(score_func=f_regression, k=min(20, len(feature_columns)))
            X_selected = selector.fit_transform(X, y)

            selected_features = [feature_columns[i] for i in selector.get_support(indices=True)]
            feature_scores = selector.scores_

            results['univariate_selection'] = {
                'selected_features': selected_features,
                'feature_scores': feature_scores.tolist()
            }

            # Recursive feature elimination (if feasible)
            if len(feature_columns) <= 50:  # Only for smaller feature sets
                estimator = RandomForestRegressor(n_estimators=50, random_state=42)
                rfe = RFE(estimator, n_features_to_select=min(15, len(feature_columns)))
                rfe.fit(X, y)

                rfe_selected = [feature_columns[i] for i in range(len(feature_columns)) if rfe.support_[i]]
                results['rfe_selection'] = {
                    'selected_features': rfe_selected,
                    'feature_rankings': rfe.ranking_.tolist()
                }

        except Exception as e:
            self.logger.warning(f"Error in feature selection analysis: {e}")

        return results
