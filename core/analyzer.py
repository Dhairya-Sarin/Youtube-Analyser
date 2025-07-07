import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union
import logging
from datetime import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed

from .data_collector import YouTubeDataCollector
from .exceptions import AnalysisError, DataCollectionError
from features.video_metadata import VideoMetadataExtractor
from features.temporal_features import TemporalFeatureExtractor
from features.engagement_metrics import EngagementMetricsExtractor
from features.title_features import TitleNLPExtractor
from features.thumbnail_features import ThumbnailVisionExtractor
from features.description_features import DescriptionFeatureExtractor
from features.channel_features import ChannelFeatureExtractor
from config.settings import AppSettings
from utils.data_validation import DataValidator


class YouTubeAnalyzer:
    """Main orchestrator for YouTube video analysis"""

    def __init__(self, settings: AppSettings):
        self.settings = settings
        self.logger = self._setup_logging()

        # Initialize components
        self.data_collector = YouTubeDataCollector(settings.youtube)
        self.data_validator = DataValidator()

        # Feature extractors
        self.feature_extractors = self._initialize_extractors()

        # Results storage
        self.analysis_results = {}

    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        logging.basicConfig(
            level=getattr(logging, self.settings.log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)

    def _initialize_extractors(self) -> Dict[str, Any]:
        """Initialize feature extractors based on configuration"""
        extractors = {
            'metadata': VideoMetadataExtractor(),
            'temporal': TemporalFeatureExtractor(),
            'engagement': EngagementMetricsExtractor(),
            'title_nlp': TitleNLPExtractor(),
            'description': DescriptionFeatureExtractor(),
            'channel': ChannelFeatureExtractor()
        }

        # Optional extractors based on settings
        if self.settings.features.include_thumbnails:
            extractors['thumbnail'] = ThumbnailVisionExtractor()

        if self.settings.features.include_audio_analysis:
            from features.audio_features import AudioFeatureExtractor
            extractors['audio'] = AudioFeatureExtractor()

        return extractors

    async def analyze_channel(self,
                              channel_name: str,
                              max_videos: Optional[int] = None,
                              include_predictions: bool = True) -> Dict[str, Any]:
        """Analyze a single YouTube channel"""
        try:
            self.logger.info(f"Starting analysis for channel: {channel_name}")

            # Collect data
            raw_data = await self.data_collector.get_channel_data(
                channel_name,
                max_videos or self.settings.youtube.max_videos_per_channel
            )

            if not raw_data or 'videos' not in raw_data:
                raise DataCollectionError(f"No data collected for channel: {channel_name}")

            # Validate data
            validated_data = self.data_validator.validate_video_data(raw_data['videos'])
            self.logger.info(f"Validated {len(validated_data)} videos for {channel_name}")

            # Extract features
            feature_data = await self._extract_all_features(validated_data, raw_data.get('channel_info', {}))

            # Create DataFrame
            df = pd.DataFrame(feature_data)

            # Train model and get predictions
            model_results = None
            predictions = None
            if include_predictions and len(df) >= 50:  # Minimum samples for training
                try:
                    from ml.models import MLModelManager
                    ml_manager = MLModelManager(self.settings.ml)
                    model_results = await ml_manager.train_model(df)
                    if model_results:
                        from ml.prediction import PredictionEngine
                        prediction_engine = PredictionEngine()
                        predictions = prediction_engine.batch_predict(df, model_results)
                except ImportError:
                    self.logger.warning("ML modules not available")

            # Generate insights
            insights = self._generate_insights(df, model_results)

            # Prepare results
            results = {
                'channel_name': channel_name,
                'analysis_timestamp': datetime.now().isoformat(),
                'data': df,
                'channel_info': raw_data.get('channel_info', {}),
                'feature_summary': self._summarize_features(df),
                'insights': insights,
                'model_results': model_results,
                'predictions': predictions
            }

            # Add SHAP explanations if enabled
            if self.settings.ml.enable_shap and model_results:
                try:
                    from ml.explainability import SHAPExplainer
                    shap_explainer = SHAPExplainer(self.settings.ml)
                    shap_results = await shap_explainer.explain_model(
                        model_results['model'],
                        df[model_results['feature_columns']]
                    )
                    results['shap_explanations'] = shap_results
                except ImportError:
                    self.logger.warning("SHAP not available")

            self.analysis_results[channel_name] = results
            self.logger.info(f"Completed analysis for channel: {channel_name}")

            return results

        except Exception as e:
            self.logger.error(f"Error analyzing channel {channel_name}: {str(e)}")
            raise AnalysisError(f"Failed to analyze channel {channel_name}: {str(e)}")

    async def analyze_multiple_channels(self,
                                        channel_names: List[str],
                                        max_workers: int = 3) -> Dict[str, Any]:
        """Analyze multiple channels concurrently"""
        self.logger.info(f"Starting batch analysis for {len(channel_names)} channels")

        results = {}
        failed_channels = []

        # Use ThreadPoolExecutor for concurrent analysis
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_channel = {
                executor.submit(asyncio.run, self.analyze_channel(channel)): channel
                for channel in channel_names
            }

            # Collect results
            for future in as_completed(future_to_channel):
                channel = future_to_channel[future]
                try:
                    result = future.result()
                    results[channel] = result
                    self.logger.info(f"Completed analysis for {channel}")
                except Exception as e:
                    self.logger.error(f"Failed to analyze {channel}: {str(e)}")
                    failed_channels.append(channel)

        # Generate comparative analysis
        if len(results) > 1:
            comparative_analysis = self._generate_comparative_analysis(results)
            return {
                'individual_results': results,
                'comparative_analysis': comparative_analysis,
                'failed_channels': failed_channels,
                'summary': {
                    'total_channels': len(channel_names),
                    'successful_analyses': len(results),
                    'failed_analyses': len(failed_channels)
                }
            }
        else:
            return {
                'individual_results': results,
                'failed_channels': failed_channels
            }

    async def _extract_all_features(self,
                                    video_data: List[Dict[str, Any]],
                                    channel_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract all features from video data"""
        all_features = []

        for video in video_data:
            try:
                video_features = {'video_id': video.get('id', '')}

                # Extract features from each extractor
                for extractor_name, extractor in self.feature_extractors.items():
                    try:
                        # Combine video data with channel info for context
                        input_data = {**video, **channel_info}
                        features = extractor.extract(input_data)
                        video_features.update(features)
                    except Exception as e:
                        self.logger.warning(f"Error in {extractor_name} for video {video.get('id', 'unknown')}: {e}")
                        # Fill with default values
                        for feature_name in extractor.get_feature_names():
                            video_features[feature_name] = 0

                all_features.append(video_features)

            except Exception as e:
                self.logger.error(f"Error processing video {video.get('id', 'unknown')}: {e}")
                continue

        return all_features

    def _summarize_features(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate feature summary statistics"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        summary = {
            'total_videos': len(df),
            'feature_count': len(df.columns),
            'numeric_features': len(numeric_cols),
            'missing_data': df.isnull().sum().to_dict(),
            'basic_stats': df[numeric_cols].describe().to_dict()
        }

        # Top correlations with views if available
        if 'views' in df.columns:
            correlations = df[numeric_cols].corr()['views'].sort_values(key=abs, ascending=False)
            summary['top_view_correlations'] = correlations.head(10).to_dict()

        return summary

    def _generate_insights(self, df: pd.DataFrame, model_results: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate actionable insights from analysis"""
        insights = {
            'performance_insights': self._analyze_performance_patterns(df),
            'content_insights': self._analyze_content_patterns(df),
            'timing_insights': self._analyze_timing_patterns(df),
            'optimization_recommendations': []
        }

        # Add ML-based insights if model is available
        if model_results and 'feature_importance' in model_results:
            insights['ml_insights'] = self._generate_ml_insights(df, model_results)

        # Generate optimization recommendations
        insights['optimization_recommendations'] = self._generate_recommendations(df, insights)

        return insights

    def _analyze_performance_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze video performance patterns"""
        if 'views' not in df.columns:
            return {}

        # Performance tiers
        top_20_percent = df.nlargest(int(len(df) * 0.2), 'views')
        bottom_20_percent = df.nsmallest(int(len(df) * 0.2), 'views')

        patterns = {
            'avg_views': df['views'].mean(),
            'median_views': df['views'].median(),
            'view_distribution': {
                'top_20_percent_avg': top_20_percent['views'].mean(),
                'bottom_20_percent_avg': bottom_20_percent['views'].mean(),
                'performance_gap': top_20_percent['views'].mean() / max(bottom_20_percent['views'].mean(), 1)
            }
        }

        # Engagement patterns
        if all(col in df.columns for col in ['likes', 'comments']):
            patterns['engagement'] = {
                'avg_engagement_rate': df['engagement_score'].mean() if 'engagement_score' in df.columns else 0,
                'top_performer_engagement': top_20_percent[
                    'engagement_score'].mean() if 'engagement_score' in top_20_percent.columns else 0
            }

        return patterns

    def _analyze_content_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze content-related patterns"""
        patterns = {}

        # Title patterns
        if 'title_length' in df.columns:
            high_performers = df.nlargest(int(len(df) * 0.3), 'views') if 'views' in df.columns else df
            patterns['title_patterns'] = {
                'avg_title_length': df['title_length'].mean(),
                'optimal_title_length': high_performers['title_length'].mean(),
                'title_length_correlation': df['title_length'].corr(df['views']) if 'views' in df.columns else 0
            }

        return patterns

    def _analyze_timing_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze timing-related patterns"""
        patterns = {}

        if 'views' not in df.columns:
            return patterns

        # Best posting times
        if 'publish_hour' in df.columns:
            hourly_performance = df.groupby('publish_hour')['views'].mean().sort_values(ascending=False)
            patterns['optimal_posting_hour'] = int(hourly_performance.index[0])
            patterns['hourly_performance'] = hourly_performance.to_dict()

        # Best posting days
        if 'publish_day_of_week' in df.columns:
            daily_performance = df.groupby('publish_day_of_week')['views'].mean().sort_values(ascending=False)
            day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            patterns['optimal_posting_day'] = day_names[int(daily_performance.index[0])]
            patterns['daily_performance'] = {day_names[int(k)]: v for k, v in daily_performance.items()}

        return patterns

    def _generate_ml_insights(self, df: pd.DataFrame, model_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate insights from ML model results"""
        insights = {
            'model_performance': {
                'r2_score': model_results.get('r2_score', 0),
                'feature_count': len(model_results.get('feature_columns', [])),
                'prediction_accuracy': 'High' if model_results.get('r2_score',
                                                                   0) > 0.7 else 'Medium' if model_results.get(
                    'r2_score', 0) > 0.4 else 'Low'
            }
        }

        # Feature importance insights
        if 'feature_importance' in model_results:
            importance_data = model_results['feature_importance']

            insights['top_success_factors'] = []
            for item in importance_data[:10]:
                feature_name = item['feature']
                importance = item['importance']

                # Generate human-readable explanations
                explanation = self._explain_feature(feature_name, importance, df)
                insights['top_success_factors'].append({
                    'feature': feature_name,
                    'importance': importance,
                    'explanation': explanation
                })

        return insights

    def _explain_feature(self, feature_name: str, importance: float, df: pd.DataFrame) -> str:
        """Generate human-readable explanation for feature importance"""
        explanations = {
            'title_length': 'Title length significantly impacts video performance',
            'thumbnail_brightness': 'Thumbnail brightness affects click-through rates',
            'publish_hour': 'Posting time influences audience reach',
            'clickbait_score': 'Clickbait elements in titles drive engagement',
            'engagement_score': 'Historical engagement predicts future performance',
            'duration_seconds': 'Video length affects viewer retention',
            'is_weekend': 'Weekend posting timing impacts viewership',
            'thumbnail_face_count': 'Human faces in thumbnails increase appeal'
        }

        base_explanation = explanations.get(feature_name, f'{feature_name} affects video performance')

        # Add statistical context if available
        if feature_name in df.columns and 'views' in df.columns:
            correlation = df[feature_name].corr(df['views'])
            if abs(correlation) > 0.3:
                direction = "positively" if correlation > 0 else "negatively"
                base_explanation += f" (strongly {direction} correlated with views)"

        return base_explanation

    def _generate_recommendations(self, df: pd.DataFrame, insights: Dict[str, Any]) -> List[str]:
        """Generate actionable optimization recommendations"""
        recommendations = []

        # Title optimization
        if 'content_insights' in insights and 'title_patterns' in insights['content_insights']:
            optimal_length = insights['content_insights']['title_patterns'].get('optimal_title_length', 0)
            if optimal_length > 0:
                recommendations.append(
                    f"Optimize title length to around {optimal_length:.0f} characters for better performance")

        # Timing optimization
        if 'timing_insights' in insights:
            if 'optimal_posting_hour' in insights['timing_insights']:
                hour = insights['timing_insights']['optimal_posting_hour']
                recommendations.append(f"Post videos around {hour}:00 for maximum reach")

            if 'optimal_posting_day' in insights['timing_insights']:
                day = insights['timing_insights']['optimal_posting_day']
                recommendations.append(f"Consider posting on {day}s for better engagement")

        return recommendations

    def _generate_comparative_analysis(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comparative analysis across multiple channels"""
        if len(results) < 2:
            return {}

        # Combine all data
        all_data = []
        channel_summaries = {}

        for channel_name, result in results.items():
            df = result['data']
            df['channel_name'] = channel_name
            all_data.append(df)

            # Create channel summary
            channel_summaries[channel_name] = {
                'total_videos': len(df),
                'avg_views': df['views'].mean() if 'views' in df.columns else 0,
                'avg_engagement': df['engagement_score'].mean() if 'engagement_score' in df.columns else 0,
                'model_score': result.get('model_results', {}).get('r2_score', 0)
            }

        combined_df = pd.concat(all_data, ignore_index=True)

        # Comparative insights
        comparative_analysis = {
            'channel_rankings': self._rank_channels(channel_summaries),
            'cross_channel_patterns': self._find_cross_channel_patterns(combined_df),
            'best_practices': self._identify_best_practices(results),
            'competitive_insights': self._generate_competitive_insights(channel_summaries)
        }

        return comparative_analysis

    def _rank_channels(self, channel_summaries: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Rank channels by various metrics"""
        rankings = {}

        # Rank by average views
        view_ranking = sorted(channel_summaries.items(),
                              key=lambda x: x[1]['avg_views'], reverse=True)
        rankings['by_views'] = [{'channel': ch, **stats} for ch, stats in view_ranking]

        # Rank by engagement
        engagement_ranking = sorted(channel_summaries.items(),
                                    key=lambda x: x[1]['avg_engagement'], reverse=True)
        rankings['by_engagement'] = [{'channel': ch, **stats} for ch, stats in engagement_ranking]

        # Rank by model accuracy (predictability)
        model_ranking = sorted(channel_summaries.items(),
                               key=lambda x: x[1]['model_score'], reverse=True)
        rankings['by_predictability'] = [{'channel': ch, **stats} for ch, stats in model_ranking]

        return rankings

    def _find_cross_channel_patterns(self, combined_df: pd.DataFrame) -> Dict[str, Any]:
        """Find patterns that work across multiple channels"""
        patterns = {}

        if 'views' in combined_df.columns:
            # Universal timing patterns
            if 'publish_hour' in combined_df.columns:
                universal_timing = combined_df.groupby('publish_hour')['views'].mean().sort_values(ascending=False)
                patterns['universal_best_hours'] = universal_timing.head(3).index.tolist()

            # Universal content patterns
            content_cols = [col for col in combined_df.columns if col.startswith('is_')]
            universal_content_performance = {}

            for col in content_cols:
                type_performance = combined_df[combined_df[col] == 1]['views'].mean()
                overall_performance = combined_df['views'].mean()
                if type_performance > overall_performance * 1.1:  # 10% better
                    universal_content_performance[col] = type_performance / overall_performance

            patterns['universal_content_types'] = universal_content_performance

        return patterns

    def _identify_best_practices(self, results: Dict[str, Any]) -> List[str]:
        """Identify best practices from top-performing channels"""
        best_practices = []

        # Find the best performing channel
        best_channel = max(results.items(),
                           key=lambda x: x[1]['data']['views'].mean() if 'views' in x[1]['data'].columns else 0)

        best_channel_data = best_channel[1]['data']

        # Extract insights from best performer
        if 'views' in best_channel_data.columns:
            top_videos = best_channel_data.nlargest(5, 'views')

            # Title patterns
            if 'title_length' in top_videos.columns:
                avg_title_length = top_videos['title_length'].mean()
                best_practices.append(f"Top videos average {avg_title_length:.0f} characters in titles")

            # Thumbnail patterns
            if 'thumbnail_face_count' in top_videos.columns:
                face_ratio = (top_videos['thumbnail_face_count'] > 0).mean()
                if face_ratio > 0.6:
                    best_practices.append("Include faces in thumbnails (found in 60%+ of top videos)")

        return best_practices

    def _generate_competitive_insights(self, channel_summaries: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Generate competitive insights"""
        values = list(channel_summaries.values())

        insights = {
            'performance_gaps': {
                'max_avg_views': max(ch['avg_views'] for ch in values),
                'min_avg_views': min(ch['avg_views'] for ch in values),
                'performance_ratio': max(ch['avg_views'] for ch in values) / max(min(ch['avg_views'] for ch in values),
                                                                                 1)
            },
            'engagement_leaders': [
                ch_name for ch_name, stats in channel_summaries.items()
                if stats['avg_engagement'] > np.mean([ch['avg_engagement'] for ch in values])
            ],
            'growth_opportunities': []
        }

        # Identify growth opportunities
        avg_views = np.mean([ch['avg_views'] for ch in values])
        for ch_name, stats in channel_summaries.items():
            if stats['avg_views'] < avg_views * 0.8:  # 20% below average
                gap = avg_views - stats['avg_views']
                insights['growth_opportunities'].append({
                    'channel': ch_name,
                    'potential_increase': gap,
                    'percentage_improvement': (gap / stats['avg_views']) * 100
                })

        return insights

    def get_analysis_summary(self) -> Dict[str, Any]:
        """Get summary of all completed analyses"""
        return {
            'total_channels_analyzed': len(self.analysis_results),
            'channels': list(self.analysis_results.keys()),
            'analysis_completion_times': {
                ch: result.get('analysis_timestamp', 'Unknown')
                for ch, result in self.analysis_results.items()
            }
        }