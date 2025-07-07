import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import io
import base64


class ChartGenerator:
    """Generate various charts and visualizations for YouTube analytics"""

    def __init__(self):
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")

    def create_performance_overview(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Create comprehensive performance overview charts"""
        charts = {}

        # Views distribution
        charts['views_distribution'] = self._create_views_distribution(df)

        # Engagement metrics
        charts['engagement_scatter'] = self._create_engagement_scatter(df)

        # Performance over time
        charts['performance_timeline'] = self._create_performance_timeline(df)

        # Top performing videos
        charts['top_videos'] = self._create_top_videos_chart(df)

        return charts

    def create_feature_analysis_charts(self, df: pd.DataFrame, correlations: pd.DataFrame) -> Dict[str, Any]:
        """Create feature analysis and correlation charts"""
        charts = {}

        # Correlation heatmap
        charts['correlation_heatmap'] = self._create_correlation_heatmap(correlations)

        # Feature distributions
        charts['feature_distributions'] = self._create_feature_distributions(df)

        # Feature vs performance
        charts['feature_performance'] = self._create_feature_performance_charts(df)

        return charts

    def create_comparative_charts(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Create charts comparing multiple channels"""
        charts = {}

        # Channel comparison
        charts['channel_comparison'] = self._create_channel_comparison(results)

        # Performance distribution comparison
        charts['performance_comparison'] = self._create_performance_comparison(results)

        # Content type analysis
        charts['content_type_analysis'] = self._create_content_type_analysis(results)

        return charts

    def _create_views_distribution(self, df: pd.DataFrame) -> Optional[str]:
        """Create views distribution chart"""
        if df.empty or 'views' not in df.columns:
            return None

        try:
            fig = px.histogram(
                df,
                x='views',
                nbins=50,
                title='Views Distribution',
                labels={'views': 'Views', 'count': 'Number of Videos'},
                marginal='box'
            )

            fig.update_layout(
                template='plotly_white',
                height=500,
                showlegend=False
            )

            return fig.to_html(include_plotlyjs='cdn')
        except Exception as e:
            print(f"Error creating views distribution chart: {e}")
            return None

    def _create_engagement_scatter(self, df: pd.DataFrame) -> Optional[str]:
        """Create engagement scatter plot"""
        if df.empty or not all(col in df.columns for col in ['views', 'likes', 'comments']):
            return None

        try:
            # Calculate engagement rate
            df_plot = df.copy()
            df_plot['engagement_rate'] = (df_plot['likes'] + df_plot['comments']) / np.maximum(df_plot['views'], 1)

            fig = px.scatter(
                df_plot,
                x='views',
                y='engagement_rate',
                size='likes',
                hover_data=['title', 'comments'] if 'title' in df_plot.columns else ['comments'],
                title='Engagement Rate vs Views',
                labels={
                    'views': 'Views',
                    'engagement_rate': 'Engagement Rate',
                    'likes': 'Likes'
                }
            )

            fig.update_layout(
                template='plotly_white',
                height=500
            )

            return fig.to_html(include_plotlyjs='cdn')
        except Exception as e:
            print(f"Error creating engagement scatter chart: {e}")
            return None

    def _create_performance_timeline(self, df: pd.DataFrame) -> Optional[str]:
        """Create performance timeline chart"""
        if df.empty or 'published_at' not in df.columns or 'views' not in df.columns:
            return None

        try:
            df_plot = df.copy()
            df_plot['published_at'] = pd.to_datetime(df_plot['published_at'], errors='coerce')
            df_plot = df_plot.dropna(subset=['published_at']).sort_values('published_at')

            if df_plot.empty:
                return None

            # Create rolling average
            df_plot['views_rolling'] = df_plot['views'].rolling(window=min(10, len(df_plot)), min_periods=1).mean()

            fig = go.Figure()

            # Individual video views
            fig.add_trace(go.Scatter(
                x=df_plot['published_at'],
                y=df_plot['views'],
                mode='markers',
                name='Individual Videos',
                opacity=0.6,
                hovertemplate='%{text}<br>Views: %{y:,}<br>Date: %{x}<extra></extra>',
                text=df_plot['title'].str[:50] + '...' if 'title' in df_plot.columns else ['Video'] * len(df_plot)
            ))

            # Rolling average
            fig.add_trace(go.Scatter(
                x=df_plot['published_at'],
                y=df_plot['views_rolling'],
                mode='lines',
                name='Rolling Average',
                line=dict(width=3)
            ))

            fig.update_layout(
                title='Performance Timeline',
                xaxis_title='Upload Date',
                yaxis_title='Views',
                template='plotly_white',
                height=500
            )

            return fig.to_html(include_plotlyjs='cdn')
        except Exception as e:
            print(f"Error creating performance timeline chart: {e}")
            return None

    def _create_top_videos_chart(self, df: pd.DataFrame, top_n: int = 10) -> Optional[str]:
        """Create top performing videos chart"""
        if df.empty or 'views' not in df.columns:
            return None

        try:
            top_videos = df.nlargest(min(top_n, len(df)), 'views')

            # Truncate titles for better display
            if 'title' in top_videos.columns:
                top_videos = top_videos.copy()
                top_videos['title_short'] = top_videos['title'].astype(str).str[:40] + '...'
                title_col = 'title_short'
            else:
                top_videos = top_videos.copy()
                top_videos['title_short'] = [f'Video {i + 1}' for i in range(len(top_videos))]
                title_col = 'title_short'

            fig = px.bar(
                top_videos,
                y=title_col,
                x='views',
                orientation='h',
                title=f'Top {len(top_videos)} Performing Videos',
                labels={'views': 'Views', title_col: 'Video Title'}
            )

            fig.update_layout(
                template='plotly_white',
                height=max(400, len(top_videos) * 40),
                yaxis={'categoryorder': 'total ascending'}
            )

            return fig.to_html(include_plotlyjs='cdn')
        except Exception as e:
            print(f"Error creating top videos chart: {e}")
            return None

    def _create_correlation_heatmap(self, correlations: pd.DataFrame) -> Optional[str]:
        """Create correlation heatmap"""
        if correlations.empty:
            return None

        try:
            # Select most relevant correlations (with views if available)
            if 'views' in correlations.columns:
                views_corr = correlations['views'].abs().sort_values(ascending=False)
                top_features = views_corr.head(15).index.tolist()
                corr_subset = correlations.loc[top_features, top_features]
            else:
                # Take first 15x15 subset
                size = min(15, len(correlations))
                corr_subset = correlations.iloc[:size, :size]

            fig = px.imshow(
                corr_subset,
                color_continuous_scale='RdBu',
                aspect='auto',
                title='Feature Correlation Matrix',
                labels={'color': 'Correlation'}
            )

            fig.update_layout(
                template='plotly_white',
                height=600
            )

            return fig.to_html(include_plotlyjs='cdn')
        except Exception as e:
            print(f"Error creating correlation heatmap: {e}")
            return None

    def _create_feature_distributions(self, df: pd.DataFrame) -> Optional[str]:
        """Create feature distribution plots"""
        if df.empty:
            return None

        try:
            # Select interesting numeric features
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            interesting_features = [
                'title_length', 'duration_seconds', 'publish_hour',
                'thumbnail_brightness', 'clickbait_score', 'engagement_score'
            ]

            # Filter to available features
            available_features = [f for f in interesting_features if f in numeric_cols]

            if not available_features:
                # Fallback to any numeric columns
                available_features = list(numeric_cols[:6])

            if not available_features:
                return None

            # Create subplots
            n_features = len(available_features)
            cols = 3
            rows = (n_features + cols - 1) // cols

            fig = make_subplots(
                rows=rows,
                cols=cols,
                subplot_titles=[f.replace('_', ' ').title() for f in available_features],
                vertical_spacing=0.1
            )

            for i, feature in enumerate(available_features):
                row = i // cols + 1
                col = i % cols + 1

                fig.add_trace(
                    go.Histogram(x=df[feature], name=feature, showlegend=False),
                    row=row, col=col
                )

            fig.update_layout(
                title='Feature Distributions',
                template='plotly_white',
                height=300 * rows
            )

            return fig.to_html(include_plotlyjs='cdn')
        except Exception as e:
            print(f"Error creating feature distributions: {e}")
            return None

    def _create_feature_performance_charts(self, df: pd.DataFrame) -> Optional[str]:
        """Create feature vs performance charts"""
        if df.empty or 'views' not in df.columns:
            return None

        try:
            # Key features to analyze
            key_features = ['title_length', 'duration_seconds', 'publish_hour', 'thumbnail_brightness']
            available_features = [f for f in key_features if f in df.columns]

            if not available_features:
                # Fallback to any numeric features
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                available_features = [col for col in numeric_cols if col != 'views'][:4]

            if not available_features:
                return None

            n_features = len(available_features)
            cols = 2
            rows = (n_features + cols - 1) // cols

            fig = make_subplots(
                rows=rows,
                cols=cols,
                subplot_titles=[f'{feature.replace("_", " ").title()} vs Views' for feature in available_features]
            )

            for i, feature in enumerate(available_features):
                row = i // cols + 1
                col = i % cols + 1

                fig.add_trace(
                    go.Scatter(
                        x=df[feature],
                        y=df['views'],
                        mode='markers',
                        name=feature,
                        showlegend=False,
                        opacity=0.6
                    ),
                    row=row, col=col
                )

            fig.update_layout(
                title='Feature vs Performance Analysis',
                template='plotly_white',
                height=400 * rows
            )

            return fig.to_html(include_plotlyjs='cdn')
        except Exception as e:
            print(f"Error creating feature performance charts: {e}")
            return None

    def _create_channel_comparison(self, results: Dict[str, Any]) -> Optional[str]:
        """Create channel comparison chart"""
        if len(results) < 2:
            return None

        try:
            comparison_data = []

            for channel_name, result in results.items():
                df = result.get('data', pd.DataFrame())
                if df.empty:
                    continue

                channel_stats = {
                    'Channel': channel_name,
                    'Avg Views': df['views'].mean() if 'views' in df.columns else 0,
                    'Total Videos': len(df),
                    'Max Views': df['views'].max() if 'views' in df.columns else 0
                }
                comparison_data.append(channel_stats)

            if not comparison_data:
                return None

            comparison_df = pd.DataFrame(comparison_data)

            # Create grouped bar chart
            fig = px.bar(
                comparison_df,
                x='Channel',
                y=['Avg Views', 'Max Views'],
                title='Channel Performance Comparison',
                barmode='group'
            )

            fig.update_layout(
                template='plotly_white',
                height=500
            )

            return fig.to_html(include_plotlyjs='cdn')
        except Exception as e:
            print(f"Error creating channel comparison chart: {e}")
            return None

    def _create_performance_comparison(self, results: Dict[str, Any]) -> Optional[str]:
        """Create performance distribution comparison"""
        if len(results) < 2:
            return None

        try:
            fig = go.Figure()
            has_data = False

            for channel_name, result in results.items():
                df = result.get('data', pd.DataFrame())
                if not df.empty and 'views' in df.columns:
                    fig.add_trace(go.Box(
                        y=df['views'],
                        name=channel_name,
                        boxpoints='outliers'
                    ))
                    has_data = True

            if not has_data:
                return None

            fig.update_layout(
                title='Views Distribution Comparison',
                yaxis_title='Views',
                template='plotly_white',
                height=500
            )

            return fig.to_html(include_plotlyjs='cdn')
        except Exception as e:
            print(f"Error creating performance comparison chart: {e}")
            return None

    def _create_content_type_analysis(self, results: Dict[str, Any]) -> Optional[str]:
        """Create content type performance analysis"""
        try:
            content_data = []

            for channel_name, result in results.items():
                df = result.get('data', pd.DataFrame())
                if df.empty:
                    continue

                # Find content type columns
                content_cols = [col for col in df.columns if
                                col.startswith('is_') and any(suffix in col for suffix in
                                                              ['tutorial', 'review', 'reaction', 'how_to', 'vs',
                                                               'challenge'])]

                for col in content_cols:
                    if col in df.columns and 'views' in df.columns:
                        content_videos = df[df[col] == 1]
                        if len(content_videos) > 0:
                            content_data.append({
                                'Channel': channel_name,
                                'Content Type': col.replace('is_', '').replace('_', ' ').title(),
                                'Count': len(content_videos),
                                'Avg Views': content_videos['views'].mean()
                            })

            if not content_data:
                return None

            content_df = pd.DataFrame(content_data)

            fig = px.scatter(
                content_df,
                x='Count',
                y='Avg Views',
                color='Channel',
                size='Count',
                hover_data=['Content Type'],
                title='Content Type Performance Analysis'
            )

            fig.update_layout(
                template='plotly_white',
                height=500
            )

            return fig.to_html(include_plotlyjs='cdn')
        except Exception as e:
            print(f"Error creating content type analysis chart: {e}")
            return None