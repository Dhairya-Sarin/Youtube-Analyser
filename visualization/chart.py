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

    def _create_views_distribution(self, df: pd.DataFrame) -> str:
        """Create views distribution chart"""
        if 'views' not in df.columns:
            return ""

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

    def _create_engagement_scatter(self, df: pd.DataFrame) -> str:
        """Create engagement scatter plot"""
        if not all(col in df.columns for col in ['views', 'likes', 'comments']):
            return ""

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

    def _create_performance_timeline(self, df: pd.DataFrame) -> str:
        """Create performance timeline chart"""
        if 'published_at' not in df.columns or 'views' not in df.columns:
            return ""

        df_plot = df.copy()
        df_plot['published_at'] = pd.to_datetime(df_plot['published_at'])
        df_plot = df_plot.sort_values('published_at')

        # Create rolling average
        df_plot['views_rolling'] = df_plot['views'].rolling(window=10, min_periods=1).mean()

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
            name='Rolling Average (10 videos)',
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

    def _create_top_videos_chart(self, df: pd.DataFrame, top_n: int = 10) -> str:
        """Create top performing videos chart"""
        if 'views' not in df.columns:
            return ""

        top_videos = df.nlargest(top_n, 'views')

        # Truncate titles for better display
        if 'title' in top_videos.columns:
            top_videos = top_videos.copy()
            top_videos['title_short'] = top_videos['title'].str[:40] + '...'
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
            title=f'Top {top_n} Performing Videos',
            labels={'views': 'Views', title_col: 'Video Title'}
        )

        fig.update_layout(
            template='plotly_white',
            height=600,
            yaxis={'categoryorder': 'total ascending'}
        )

        return fig.to_html(include_plotlyjs='cdn')

    def _create_correlation_heatmap(self, correlations: pd.DataFrame) -> str:
        """Create correlation heatmap"""
        if correlations.empty:
            return ""

        # Select most relevant correlations (with views if available)
        if 'views' in correlations.columns:
            views_corr = correlations['views'].abs().sort_values(ascending=False)
            top_features = views_corr.head(15).index.tolist()
            corr_subset = correlations.loc[top_features, top_features]
        else:
            # Take first 15x15 subset
            corr_subset = correlations.iloc[:15, :15]

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

    def _create_feature_distributions(self, df: pd.DataFrame) -> str:
        """Create feature distribution plots"""
        # Select interesting numeric features
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        interesting_features = [
            'title_length', 'duration_seconds', 'publish_hour',
            'thumbnail_brightness', 'clickbait_score', 'engagement_score'
        ]

        # Filter to available features
        available_features = [f for f in interesting_features if f in numeric_cols]

        if not available_features:
            return ""

        # Create subplots
        n_features = len(available_features)
        cols = 3
        rows = (n_features + cols - 1) // cols

        fig = make_subplots(
            rows=rows,
            cols=cols,
            subplot_titles=available_features,
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

    def _create_feature_performance_charts(self, df: pd.DataFrame) -> str:
        """Create feature vs performance charts"""
        if 'views' not in df.columns:
            return ""

        # Key features to analyze
        key_features = ['title_length', 'duration_seconds', 'publish_hour', 'thumbnail_brightness']
        available_features = [f for f in key_features if f in df.columns]

        if not available_features:
            return ""

        n_features = len(available_features)
        cols = 2
        rows = (n_features + cols - 1) // cols

        fig = make_subplots(
            rows=rows,
            cols=cols,
            subplot_titles=[f'{feature} vs Views' for feature in available_features]
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

    def _create_channel_comparison(self, results: Dict[str, Any]) -> str:
        """Create channel comparison chart"""
        if len(results) < 2:
            return ""

        comparison_data = []

        for channel_name, result in results.items():
            df = result['data']

            channel_stats = {
                'Channel': channel_name,
                'Avg Views': df['views'].mean() if 'views' in df.columns else 0,
                'Total Videos': len(df),
                'Max Views': df['views'].max() if 'views' in df.columns else 0
            }
            comparison_data.append(channel_stats)

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

    def _create_performance_comparison(self, results: Dict[str, Any]) -> str:
        """Create performance distribution comparison"""
        if len(results) < 2:
            return ""

        fig = go.Figure()

        for channel_name, result in results.items():
            df = result['data']
            if 'views' in df.columns:
                fig.add_trace(go.Box(
                    y=df['views'],
                    name=channel_name,
                    boxpoints='outliers'
                ))

        fig.update_layout(
            title='Views Distribution Comparison',
            yaxis_title='Views',
            template='plotly_white',
            height=500
        )

        return fig.to_html(include_plotlyjs='cdn')

    def _create_content_type_analysis(self, results: Dict[str, Any]) -> str:
        """Create content type performance analysis"""
        content_data = []

        for channel_name, result in results.items():
            df = result['data']

            # Find content type columns
            content_cols = [col for col in df.columns if
                            col.startswith('is_') and col.endswith(('_video', '_tutorial', '_review', '_reaction'))]

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
            return ""

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
