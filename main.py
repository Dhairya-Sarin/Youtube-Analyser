# main.py - Fixed Streamlit Application
import streamlit as st
import pandas as pd
import numpy as np
import asyncio
import logging
from datetime import datetime
from typing import Dict, Any, List
import sys
import os
import traceback

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import modular components
from config.settings import get_default_settings, AppSettings, FeatureSet, ModelType
from core.analyzer import YouTubeAnalyzer
from core.exceptions import AnalysisError, DataCollectionError, APIError
from visualization.charts import ChartGenerator
from visualization.export import DataExporter

# Page configuration
st.set_page_config(
    page_title="🎯 YouTube Analytics Pro",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #FF0000;
        font-size: 3rem;
        font-weight: bold;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    .metric-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 15px;
        color: white;
        margin: 10px 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    .success-box {
        background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
        padding: 15px;
        border-radius: 10px;
        color: white;
        margin: 10px 0;
        box-shadow: 0 3px 10px rgba(0,0,0,0.2);
    }
    .feature-card {
        background: white;
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        margin: 15px 0;
        border-left: 5px solid #FF0000;
        transition: transform 0.3s ease;
    }
    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 30px rgba(0,0,0,0.15);
    }
    .status-indicator {
        display: inline-block;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        margin-right: 8px;
    }
    .status-success { background-color: #4CAF50; }
    .status-warning { background-color: #FFC107; }
    .status-error { background-color: #F44336; }
    .progress-container {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 15px;
        padding: 25px;
        margin: 15px 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .insight-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 12px;
        margin: 10px 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    .recommendation-card {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        padding: 15px;
        border-radius: 10px;
        margin: 8px 0;
        box-shadow: 0 3px 12px rgba(0,0,0,0.15);
    }
    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 3px 15px rgba(0,0,0,0.1);
        border-top: 4px solid #FF0000;
    }
</style>
""", unsafe_allow_html=True)


class YouTubeAnalyticsApp:
    """Main Streamlit application class"""

    def __init__(self):
        self.settings = self._load_settings()
        self.analyzer = None
        self.chart_generator = ChartGenerator()
        self.data_exporter = DataExporter()

        # Initialize session state
        self._initialize_session_state()

    def _load_settings(self) -> AppSettings:
        """Load application settings"""
        settings = get_default_settings()

        # Override with session state if available
        if 'settings' in st.session_state:
            return st.session_state.settings

        return settings

    def _initialize_session_state(self):
        """Initialize Streamlit session state"""
        if 'analysis_complete' not in st.session_state:
            st.session_state.analysis_complete = False
        if 'analysis_results' not in st.session_state:
            st.session_state.analysis_results = {}
        if 'current_analysis' not in st.session_state:
            st.session_state.current_analysis = None
        if 'settings' not in st.session_state:
            st.session_state.settings = self.settings
        if 'export_files' not in st.session_state:
            st.session_state.export_files = {}

    def run(self):
        """Main application entry point"""
        try:
            # Header
            st.markdown('<h1 class="main-header">🎯 YouTube Analytics Pro</h1>', unsafe_allow_html=True)
            st.markdown(
                "### 🧠 Advanced AI-Powered YouTube Analytics • 📊 150+ Feature Analysis • 🚀 Predictive ML Models")

            # Sidebar configuration
            self._render_sidebar()

            # Main content
            if st.session_state.analysis_complete and st.session_state.analysis_results:
                self._render_results()
            else:
                self._render_analysis_setup()

        except Exception as e:
            st.error(f"Application Error: {str(e)}")
            st.error("Please refresh the page and try again.")
            if st.checkbox("Show Error Details"):
                st.code(traceback.format_exc())

    def _render_sidebar(self):
        """Render sidebar configuration"""
        st.sidebar.markdown("## 🔧 Configuration")
        st.sidebar.markdown("---")

        # API Configuration
        st.sidebar.markdown("### 🔑 API Settings")
        api_key = st.sidebar.text_input(
            "YouTube Data API Key",
            type="password",
            help="Get your API key from Google Cloud Console",
            value=self.settings.youtube.api_key
        )

        # Update settings
        self.settings.youtube.api_key = api_key

        # Channel Input
        st.sidebar.markdown("### 📺 Channels to Analyze")
        channels_input = st.sidebar.text_area(
            "Enter channel names (one per line)",
            placeholder="MrBeast\nPewDiePie\nT-Series\nMarkiplier\nDude Perfect",
            height=120
        )

        # Analysis Settings
        st.sidebar.markdown("### ⚙️ Analysis Settings")

        max_videos = st.sidebar.slider(
            "Max videos per channel",
            50, 500,
            self.settings.youtube.max_videos_per_channel,
            help="More videos = better analysis but slower processing"
        )
        self.settings.youtube.max_videos_per_channel = max_videos

        # Feature Selection
        feature_set = st.sidebar.selectbox(
            "Feature Complexity",
            options=[fs.value for fs in FeatureSet],
            index=list(FeatureSet).index(self.settings.features.feature_set),
            help="Choose the complexity of features to extract"
        )
        self.settings.features.feature_set = FeatureSet(feature_set)

        # Advanced Features
        st.sidebar.markdown("### 🚀 Advanced Features")

        include_thumbnails = st.sidebar.checkbox(
            "🖼️ Computer Vision Analysis",
            value=self.settings.features.include_thumbnails,
            help="Analyze thumbnails for faces, colors, composition"
        )
        self.settings.features.include_thumbnails = include_thumbnails

        include_nlp = st.sidebar.checkbox(
            "🧠 Advanced NLP Processing",
            value=self.settings.features.include_advanced_nlp,
            help="Deep analysis of titles and descriptions"
        )
        self.settings.features.include_advanced_nlp = include_nlp

        include_shap = st.sidebar.checkbox(
            "📊 AI Explainability (SHAP)",
            value=self.settings.ml.enable_shap,
            help="Generate model explanations with SHAP"
        )
        self.settings.ml.enable_shap = include_shap

        # Model Selection
        st.sidebar.markdown("### 🤖 Machine Learning")
        model_type = st.sidebar.selectbox(
            "ML Algorithm",
            options=[mt.value for mt in ModelType],
            index=list(ModelType).index(self.settings.ml.model_type),
            help="Choose the machine learning algorithm"
        )
        self.settings.ml.model_type = ModelType(model_type)

        enable_tuning = st.sidebar.checkbox(
            "🎯 Hyperparameter Optimization",
            value=self.settings.ml.hyperparameter_tuning,
            help="Automatically optimize model parameters"
        )
        self.settings.ml.hyperparameter_tuning = enable_tuning

        # Export Settings
        st.sidebar.markdown("### 📥 Export Options")
        export_excel = st.sidebar.checkbox("📊 Excel Reports", value=True)
        export_charts = st.sidebar.checkbox("📈 Interactive Charts", value=True)
        export_html = st.sidebar.checkbox("📄 HTML Dashboard", value=True)
        export_csv = st.sidebar.checkbox("📋 CSV Data", value=True)

        # Store channels and export options
        st.session_state.channels = [ch.strip() for ch in channels_input.split('\n') if ch.strip()]
        st.session_state.export_options = {
            'excel': export_excel,
            'charts': export_charts,
            'html': export_html,
            'csv': export_csv
        }

        # Update session state settings
        st.session_state.settings = self.settings

        # System Status
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 📊 System Status")
        self._show_system_status()

        # Help Section
        st.sidebar.markdown("---")
        st.sidebar.markdown("### ❓ Help & Tips")

        with st.sidebar.expander("🔑 Getting API Key"):
            st.markdown("""
            1. Go to [Google Cloud Console](https://console.cloud.google.com/)
            2. Create/select a project
            3. Enable YouTube Data API v3
            4. Create credentials (API key)
            5. Copy the key here
            """)

        with st.sidebar.expander("🎯 Feature Complexity"):
            st.markdown("""
            - **Basic**: Core metrics (~50 features)
            - **Extended**: + Thumbnails + NLP (~100 features)
            - **Full**: All features (~150+ features)
            - **Custom**: User-defined selection
            """)

    def _show_system_status(self):
        """Show system status indicators"""
        # API Key Status
        api_status = "🟢 Valid" if self.settings.youtube.api_key else "🔴 Missing"
        st.sidebar.markdown(f"**API Key:** {api_status}")

        # Channels Status
        channel_count = len(st.session_state.get('channels', []))
        channels_status = f"🟢 {channel_count} channels" if channel_count > 0 else "🔴 No channels"
        st.sidebar.markdown(f"**Channels:** {channels_status}")

        # Feature Status
        feature_count = self._estimate_feature_count()
        st.sidebar.markdown(f"**Features:** ~{feature_count} features")

        # Model Status
        model_status = f"🤖 {self.settings.ml.model_type.value.replace('_', ' ').title()}"
        st.sidebar.markdown(f"**ML Model:** {model_status}")

    def _estimate_feature_count(self) -> int:
        """Estimate number of features based on configuration"""
        base_features = 50

        if self.settings.features.include_thumbnails:
            base_features += 30

        if self.settings.features.include_advanced_nlp:
            base_features += 25

        if self.settings.features.feature_set == FeatureSet.FULL:
            base_features += 50
        elif self.settings.features.feature_set == FeatureSet.EXTENDED:
            base_features += 25

        return base_features

    def _render_analysis_setup(self):
        """Render analysis setup and start button"""
        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown("## 🎯 Advanced YouTube Analytics Platform")

            st.markdown("""
            <div class="feature-card">
                <h4>🔍 <strong>Comprehensive Data Collection</strong></h4>
                <p>Fetches detailed video metadata, performance metrics, and channel information using YouTube Data API v3</p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            <div class="feature-card">
                <h4>🖼️ <strong>Computer Vision Analysis</strong></h4>
                <p>Advanced thumbnail analysis including face detection, color analysis, composition scoring, and clickbait pattern recognition</p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            <div class="feature-card">
                <h4>📝 <strong>Natural Language Processing</strong></h4>
                <p>Deep NLP analysis of titles and descriptions with sentiment analysis, named entity recognition, and psychological feature extraction</p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            <div class="feature-card">
                <h4>🤖 <strong>Machine Learning & AI</strong></h4>
                <p>Predictive models with ensemble algorithms, hyperparameter tuning, and SHAP-powered explainable AI</p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            <div class="feature-card">
                <h4>📊 <strong>Interactive Visualizations</strong></h4>
                <p>Rich dashboards with correlation heatmaps, performance timelines, and comparative analysis charts</p>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown("## 📈 Analysis Output")

            st.markdown("""
            <div class="metric-container">
                <h4>🔥 What You'll Get:</h4>
                <ul>
                    <li>📊 150+ extracted features</li>
                    <li>🧠 AI-powered predictions</li>
                    <li>📈 Performance insights</li>
                    <li>🎯 Optimization recommendations</li>
                    <li>📋 Comprehensive reports</li>
                    <li>🔤 Viral keyword analysis</li>
                    <li>⏰ Optimal timing insights</li>
                    <li>🖼️ Thumbnail effectiveness</li>
                    <li>📥 Multiple export formats</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

            # Feature breakdown
            st.markdown("### 🔧 Feature Categories")

            feature_categories = {
                "📺 Video Metadata": "Duration, category, tags, captions",
                "⏰ Temporal Analysis": "Upload timing, seasonality, trends",
                "📈 Performance Metrics": "Views, engagement, viral potential",
                "📝 Title Analysis": "NLP, sentiment, clickbait scoring",
                "🖼️ Thumbnail Vision": "Faces, colors, composition, aesthetics",
                "📄 Content Analysis": "Description, tags, links, CTAs",
                "🧍‍♂️ Channel Insights": "Age, growth, consistency, branding"
            }

            for category, description in feature_categories.items():
                st.markdown(f"**{category}**: {description}")

        # Analysis Control Section
        st.markdown("---")
        st.markdown("## 🚀 Start Analysis")

        # Validation
        can_analyze, error_message = self._validate_analysis_setup()

        if not can_analyze:
            st.error(f"❌ {error_message}")
        else:
            st.success("✅ Ready to analyze! All requirements met.")

        # Analysis options
        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button(
                    "🚀 Quick Analysis",
                    help="Analyze with basic features for faster results",
                    disabled=not can_analyze
            ):
                # Set to basic features for quick analysis
                self.settings.features.feature_set = FeatureSet.BASIC
                self.settings.features.include_thumbnails = False
                self.settings.ml.enable_shap = False
                self._start_analysis()

        with col2:
            if st.button(
                    "🔬 Deep Analysis",
                    type="primary",
                    help="Full analysis with all features and AI explanations",
                    disabled=not can_analyze
            ):
                # Set to full features
                self.settings.features.feature_set = FeatureSet.FULL
                self.settings.features.include_thumbnails = True
                self.settings.ml.enable_shap = True
                self._start_analysis()

        with col3:
            if st.button(
                    "⚙️ Custom Analysis",
                    help="Use current sidebar settings",
                    disabled=not can_analyze
            ):
                self._start_analysis()

    def _validate_analysis_setup(self) -> tuple[bool, str]:
        """Validate analysis setup"""
        if not self.settings.youtube.api_key:
            return False, "Please enter your YouTube API key in the sidebar"

        if not st.session_state.get('channels'):
            return False, "Please enter at least one channel name in the sidebar"

        if len(st.session_state.get('channels', [])) > 5:
            return False, "Maximum 5 channels allowed for performance reasons"

        return True, ""

    def _start_analysis(self):
        """Start the analysis process"""
        try:
            # Initialize analyzer
            self.analyzer = YouTubeAnalyzer(self.settings)

            # Show progress
            st.markdown("""
            <div class="progress-container">
                <h3>🔄 Analysis in Progress</h3>
                <p>Processing your request with advanced AI algorithms...</p>
            </div>
            """, unsafe_allow_html=True)

            progress_bar = st.progress(0)
            status_text = st.empty()
            detailed_status = st.empty()

            channels = st.session_state.channels

            # Show analysis details
            with detailed_status.container():
                st.info(f"🔍 **Analyzing {len(channels)} channel(s)** with {self._estimate_feature_count()} features")
                st.info(f"🤖 **ML Model**: {self.settings.ml.model_type.value.replace('_', ' ').title()}")
                st.info(f"📊 **Feature Set**: {self.settings.features.feature_set.value.title()}")

            # Run analysis
            if len(channels) == 1:
                # Single channel analysis
                status_text.text(f"🔍 Analyzing {channels[0]}...")
                progress_bar.progress(0.1)

                # Run analysis synchronously to avoid asyncio issues
                result = self._run_single_analysis(channels[0])

                progress_bar.progress(0.8)
                status_text.text("📊 Generating insights...")

                # Process results
                if result:
                    progress_bar.progress(1.0)
                    status_text.text("✅ Analysis Complete!")

                    st.session_state.analysis_results = result
                    st.session_state.current_analysis = 'single'
                    st.session_state.analysis_complete = True

                    # Auto-generate some exports
                    self._generate_auto_exports(result)

                    st.rerun()
                else:
                    st.error("❌ No results obtained. Please check your inputs and try again.")

            else:
                # Multi-channel analysis
                status_text.text(f"🔍 Analyzing {len(channels)} channels...")
                progress_bar.progress(0.1)

                results = self._run_multi_analysis(channels)

                progress_bar.progress(0.8)
                status_text.text("📊 Generating comparative analysis...")

                if results and results.get('individual_results'):
                    progress_bar.progress(1.0)
                    status_text.text("✅ Analysis Complete!")

                    st.session_state.analysis_results = results
                    st.session_state.current_analysis = 'multiple'
                    st.session_state.analysis_complete = True

                    # Auto-generate exports
                    self._generate_auto_exports(results)

                    st.rerun()
                else:
                    st.error("❌ No results obtained. Please check your inputs and try again.")

        except APIError as e:
            st.error(f"❌ **API Error**: {str(e)}")
            st.info("💡 **Solutions**: Check your API key, quota limits, or try again later")

        except DataCollectionError as e:
            st.error(f"❌ **Data Collection Error**: {str(e)}")
            st.info("💡 **Solutions**: Verify channel names are correct and publicly accessible")

        except Exception as e:
            st.error(f"❌ **Unexpected Error**: {str(e)}")
            st.info("💡 **Solutions**: Try refreshing the page or using fewer channels")
            if st.checkbox("Show Technical Details"):
                st.code(traceback.format_exc())

    def _run_single_analysis(self, channel: str) -> Dict[str, Any]:
        """Run single channel analysis synchronously"""
        try:
            # Create new event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            try:
                result = loop.run_until_complete(
                    self.analyzer.analyze_channel(
                        channel,
                        max_videos=self.settings.youtube.max_videos_per_channel,
                        include_predictions=True
                    )
                )
                return result
            finally:
                loop.close()
        except Exception as e:
            st.error(f"Error analyzing channel {channel}: {str(e)}")
            return None

    def _run_multi_analysis(self, channels: List[str]) -> Dict[str, Any]:
        """Run multi-channel analysis synchronously"""
        try:
            # Create new event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            try:
                results = loop.run_until_complete(
                    self.analyzer.analyze_multiple_channels(
                        channels,
                        max_workers=2
                    )
                )
                return results
            finally:
                loop.close()
        except Exception as e:
            st.error(f"Error in multi-channel analysis: {str(e)}")
            return None

    def _generate_auto_exports(self, results: Any):
        """Automatically generate some export files"""
        try:
            if st.session_state.export_options.get('excel', True):
                # Generate Excel export
                if st.session_state.current_analysis == 'single':
                    excel_file = self.data_exporter.export_to_excel(results)
                    st.session_state.export_files['excel'] = excel_file

        except Exception as e:
            st.warning(f"Note: Auto-export failed: {e}")

    def _render_results(self):
        """Render analysis results"""
        # Header with reset option
        col1, col2, col3 = st.columns([2, 1, 1])

        with col1:
            st.markdown("# 📊 Analysis Results")

        with col2:
            if st.button("🔄 New Analysis", type="secondary", use_container_width=True):
                st.session_state.analysis_complete = False
                st.session_state.analysis_results = {}
                st.session_state.current_analysis = None
                st.session_state.export_files = {}
                st.rerun()

        with col3:
            # Quick export button
            if st.button("📥 Quick Export", use_container_width=True):
                self._quick_export()

        # Show analysis summary
        self._show_analysis_summary()

        # Create tabs for different views
        if st.session_state.current_analysis == 'single':
            self._render_single_channel_results()
        else:
            self._render_multi_channel_results()

    def _show_analysis_summary(self):
        """Show analysis summary at the top"""
        try:
            if st.session_state.current_analysis == 'single':
                results = st.session_state.analysis_results
                data = results.get('data', pd.DataFrame())
                channel_name = results.get('channel_name', 'Unknown')

                st.markdown(f"### 📺 Analysis for **{channel_name}**")

                # Key metrics in columns
                col1, col2, col3, col4, col5 = st.columns(5)

                with col1:
                    st.markdown("""
                    <div class="metric-card">
                        <h3>📹</h3>
                        <h2>{:,}</h2>
                        <p>Videos</p>
                    </div>
                    """.format(len(data)), unsafe_allow_html=True)

                with col2:
                    avg_views = data['views'].mean() if 'views' in data.columns and not data.empty else 0
                    st.markdown("""
                    <div class="metric-card">
                        <h3>👀</h3>
                        <h2>{:,.0f}</h2>
                        <p>Avg Views</p>
                    </div>
                    """.format(avg_views), unsafe_allow_html=True)

                with col3:
                    total_views = data['views'].sum() if 'views' in data.columns and not data.empty else 0
                    st.markdown("""
                    <div class="metric-card">
                        <h3>🔥</h3>
                        <h2>{:,.0f}</h2>
                        <p>Total Views</p>
                    </div>
                    """.format(total_views), unsafe_allow_html=True)

                with col4:
                    feature_count = len(data.columns) if not data.empty else 0
                    st.markdown("""
                    <div class="metric-card">
                        <h3>🎯</h3>
                        <h2>{}</h2>
                        <p>Features</p>
                    </div>
                    """.format(feature_count), unsafe_allow_html=True)

                with col5:
                    model_score = results.get('model_results', {}).get('r2_score', 0) if results.get(
                        'model_results') else 0
                    st.markdown("""
                    <div class="metric-card">
                        <h3>🤖</h3>
                        <h2>{:.1%}</h2>
                        <p>AI Accuracy</p>
                    </div>
                    """.format(model_score), unsafe_allow_html=True)

            else:
                # Multi-channel summary
                results = st.session_state.analysis_results
                individual_results = results.get('individual_results', {})

                st.markdown(f"### 🏆 Multi-Channel Analysis ({len(individual_results)} channels)")

                col1, col2, col3 = st.columns(3)

                with col1:
                    total_videos = sum(len(r.get('data', pd.DataFrame())) for r in individual_results.values())
                    st.metric("Total Videos", f"{total_videos:,}")

                with col2:
                    successful = len(individual_results)
                    failed = len(results.get('failed_channels', []))
                    st.metric("Success Rate", f"{successful}/{successful + failed}")

                with col3:
                    model_results = [r.get('model_results', {}) for r in individual_results.values() if
                                     r.get('model_results')]
                    if model_results:
                        avg_accuracy = np.mean([mr.get('r2_score', 0) for mr in model_results])
                        st.metric("Avg AI Accuracy", f"{avg_accuracy:.1%}")
                    else:
                        st.metric("Avg AI Accuracy", "N/A")
        except Exception as e:
            st.error(f"Error displaying summary: {str(e)}")

    def _render_single_channel_results(self):
        """Render results for single channel analysis"""
        results = st.session_state.analysis_results

        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📈 Performance Overview",
            "🔥 Top Content",
            "🧠 AI Insights",
            "🎯 Recommendations",
            "📊 Feature Analysis",
            "📥 Export & Download"
        ])

        with tab1:
            self._render_performance_overview(results)

        with tab2:
            self._render_top_content(results)

        with tab3:
            self._render_ai_insights(results)

        with tab4:
            self._render_recommendations(results)

        with tab5:
            self._render_feature_analysis(results)

        with tab6:
            self._render_export_section(results)

    def _render_multi_channel_results(self):
        """Render results for multi-channel analysis"""
        results = st.session_state.analysis_results

        tab1, tab2, tab3, tab4 = st.tabs([
            "🏆 Channel Rankings",
            "📊 Comparative Analysis",
            "💡 Best Practices",
            "📥 Export & Download"
        ])

        with tab1:
            self._render_channel_rankings(results)

        with tab2:
            self._render_comparative_analysis(results)

        with tab3:
            self._render_best_practices(results)

        with tab4:
            self._render_multi_export_section(results)

    def _render_performance_overview(self, results: Dict[str, Any]):
        """Render performance overview tab"""
        data = results.get('data', pd.DataFrame())

        if data.empty:
            st.error("No data available for analysis")
            return

        # Performance insights
        insights = results.get('insights', {})
        perf_insights = insights.get('performance_insights', {})

        if perf_insights:
            st.markdown("## 📊 Performance Insights")

            col1, col2 = st.columns(2)

            with col1:
                if 'view_distribution' in perf_insights:
                    dist = perf_insights['view_distribution']
                    gap = dist.get('performance_gap', 1)

                    st.markdown(f"""
                    <div class="insight-box">
                        <h4>🎯 Performance Distribution</h4>
                        <p><strong>{gap:.1f}x</strong> performance gap between top and bottom videos</p>
                        <p>Top 20% average: <strong>{dist.get('top_20_percent_avg', 0):,.0f}</strong> views</p>
                        <p>Bottom 20% average: <strong>{dist.get('bottom_20_percent_avg', 0):,.0f}</strong> views</p>
                    </div>
                    """, unsafe_allow_html=True)

            with col2:
                if 'engagement' in perf_insights:
                    eng = perf_insights['engagement']

                    st.markdown(f"""
                    <div class="insight-box">
                        <h4>💝 Engagement Patterns</h4>
                        <p>Average engagement rate: <strong>{eng.get('avg_engagement_rate', 0):.3f}</strong></p>
                        <p>Top performers engagement: <strong>{eng.get('top_performer_engagement', 0):.3f}</strong></p>
                    </div>
                    """, unsafe_allow_html=True)

        # Charts
        if st.session_state.export_options.get('charts', True):
            st.markdown("## 📈 Performance Charts")

            try:
                # Views distribution
                if 'views' in data.columns:
                    import plotly.express as px
                    fig = px.histogram(
                        data,
                        x='views',
                        nbins=30,
                        title='Views Distribution',
                        labels={'views': 'Views', 'count': 'Number of Videos'}
                    )
                    fig.update_layout(template='plotly_white', height=400)
                    st.plotly_chart(fig, use_container_width=True)

                # Performance timeline
                if 'published_at' in data.columns and 'views' in data.columns:
                    data_plot = data.copy()
                    data_plot['published_at'] = pd.to_datetime(data_plot['published_at'], errors='coerce')
                    data_plot = data_plot.dropna(subset=['published_at']).sort_values('published_at')

                    if not data_plot.empty:
                        fig = px.scatter(
                            data_plot,
                            x='published_at',
                            y='views',
                            title='Performance Timeline',
                            labels={'published_at': 'Upload Date', 'views': 'Views'}
                        )
                        fig.update_layout(template='plotly_white', height=400)
                        st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.warning(f"Could not generate charts: {e}")

    def _render_top_content(self, results: Dict[str, Any]):
        """Render top content tab"""
        data = results.get('data', pd.DataFrame())

        if data.empty or 'views' not in data.columns:
            st.error("No video data available")
            return

        st.markdown("## 🔥 Top Performing Content")

        # Top videos table
        top_videos = data.nlargest(15, 'views')

        for idx, (_, video) in enumerate(top_videos.iterrows(), 1):
            with st.expander(
                    f"#{idx} - {str(video.get('title', 'Unknown Title'))[:80]}{'...' if len(str(video.get('title', ''))) > 80 else ''}"):
                col1, col2, col3 = st.columns([2, 1, 1])

                with col1:
                    st.write(f"**📅 Published:** {video.get('published_at', 'Unknown')}")
                    if 'description' in video and video['description']:
                        desc = str(video['description'])[:200] + "..." if len(str(video['description'])) > 200 else str(
                            video['description'])
                        st.write(f"**📝 Description:** {desc}")

                with col2:
                    st.metric("👀 Views", f"{video.get('views', 0):,}")
                    st.metric("👍 Likes", f"{video.get('likes', 0):,}")

                with col3:
                    st.metric("💬 Comments", f"{video.get('comments', 0):,}")
                    eng_score = video.get('engagement_score', 0)
                    st.metric("📊 Engagement", f"{eng_score:.3f}")

        # Performance chart
        try:
            if 'views' in top_videos.columns:
                import plotly.express as px

                # Create titles for chart
                chart_data = top_videos.copy()
                if 'title' in chart_data.columns:
                    chart_data['title_short'] = chart_data['title'].astype(str).str[:40] + '...'
                else:
                    chart_data['title_short'] = [f'Video {i + 1}' for i in range(len(chart_data))]

                fig = px.bar(
                    chart_data,
                    y='title_short',
                    x='views',
                    orientation='h',
                    title=f'Top {len(chart_data)} Performing Videos',
                    labels={'views': 'Views', 'title_short': 'Video Title'}
                )
                fig.update_layout(
                    template='plotly_white',
                    height=max(400, len(chart_data) * 30),
                    yaxis={'categoryorder': 'total ascending'}
                )
                st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not generate top videos chart: {e}")

    def _render_ai_insights(self, results: Dict[str, Any]):
        """Render AI insights tab"""
        model_results = results.get('model_results', {})
        shap_results = results.get('shap_explanations', {})

        st.markdown("## 🤖 AI Model Performance")

        if model_results:
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                r2_score = model_results.get('r2_score', 0)
                st.metric("🎯 R² Score", f"{r2_score:.3f}")

            with col2:
                feature_count = len(model_results.get('feature_columns', []))
                st.metric("📊 Features Used", feature_count)

            with col3:
                training_samples = model_results.get('training_samples', 0)
                st.metric("🔢 Training Samples", training_samples)

            with col4:
                cv_mean = model_results.get('cv_mean', 0)
                st.metric("✅ CV Score", f"{cv_mean:.3f}" if cv_mean else "N/A")

            # Feature importance
            st.markdown("### 🎯 Top Success Factors")

            feature_importance = model_results.get('feature_importance', [])
            if feature_importance:
                try:
                    # Create importance chart
                    importance_df = pd.DataFrame(feature_importance[:15])
                    importance_df['feature_clean'] = importance_df['feature'].str.replace('_', ' ').str.title()

                    col1, col2 = st.columns([2, 1])

                    with col1:
                        # Bar chart
                        import plotly.express as px
                        fig = px.bar(
                            importance_df,
                            x='importance',
                            y='feature_clean',
                            orientation='h',
                            title="Feature Importance",
                            labels={'importance': 'Importance Score', 'feature_clean': 'Feature'}
                        )
                        fig.update_layout(height=600, template='plotly_white')
                        st.plotly_chart(fig, use_container_width=True)

                    with col2:
                        # Top features list
                        st.markdown("**🏆 Most Important:**")
                        for i, feat in enumerate(feature_importance[:8], 1):
                            impact = "🔥" if feat['importance'] > 0.1 else "📈" if feat['importance'] > 0.05 else "📊"
                            clean_name = feat['feature'].replace('_', ' ').title()
                            st.markdown(f"{impact} **{i}.** {clean_name}")
                except Exception as e:
                    st.warning(f"Could not generate feature importance chart: {e}")

                    # Fallback: show as table
                    st.dataframe(pd.DataFrame(feature_importance[:10]), use_container_width=True)

        # SHAP Analysis
        if shap_results and st.session_state.settings.ml.enable_shap:
            st.markdown("## 🧠 AI Explainability (SHAP)")

            if 'summary_plot' in shap_results:
                st.markdown("### 📊 SHAP Summary Plot")
                st.markdown("*Shows how each feature impacts video performance predictions*")

                try:
                    # Display SHAP plot
                    import base64
                    plot_data = shap_results['summary_plot']
                    image_data = base64.b64decode(plot_data)
                    st.image(image_data, caption="SHAP Feature Impact Analysis")
                except Exception as e:
                    st.warning(f"Could not display SHAP plot: {e}")

            if 'explanations' in shap_results:
                explanations = shap_results['explanations']

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("### 🚀 Positive Impact Features")
                    pos_contributors = explanations.get('positive_contributors', [])
                    for contrib in pos_contributors[:5]:
                        feature_name = contrib['feature'].replace('_', ' ').title()
                        st.success(f"**{feature_name}**: {contrib['explanation']}")

                with col2:
                    st.markdown("### ⚠️ Negative Impact Features")
                    neg_contributors = explanations.get('negative_contributors', [])
                    for contrib in neg_contributors[:5]:
                        feature_name = contrib['feature'].replace('_', ' ').title()
                        st.warning(f"**{feature_name}**: {contrib['explanation']}")

        else:
            st.info("🧠 Enable SHAP in sidebar settings for AI explanations")

    def _render_recommendations(self, results: Dict[str, Any]):
        """Render recommendations tab"""
        insights = results.get('insights', {})
        recommendations = insights.get('optimization_recommendations', [])

        st.markdown("## 🎯 Optimization Recommendations")

        if recommendations:
            for i, rec in enumerate(recommendations, 1):
                st.markdown(f"""
                <div class="recommendation-card">
                    <h4>#{i} Recommendation</h4>
                    <p>{rec}</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No specific recommendations available. Try analyzing more videos for better insights.")

        # Additional insights
        col1, col2 = st.columns(2)

        with col1:
            # Timing insights
            timing_insights = insights.get('timing_insights', {})
            if timing_insights:
                st.markdown("### ⏰ Optimal Timing")

                if 'optimal_posting_hour' in timing_insights:
                    hour = timing_insights['optimal_posting_hour']
                    st.success(f"🕐 **Best posting time**: {hour}:00")

                if 'optimal_posting_day' in timing_insights:
                    day = timing_insights['optimal_posting_day']
                    st.success(f"📅 **Best posting day**: {day}")

        with col2:
            # Content insights
            content_insights = insights.get('content_insights', {})
            if content_insights and 'title_patterns' in content_insights:
                st.markdown("### 📝 Content Optimization")

                title_patterns = content_insights['title_patterns']
                optimal_length = title_patterns.get('optimal_title_length', 0)

                if optimal_length > 0:
                    st.success(f"📏 **Optimal title length**: {optimal_length:.0f} characters")

        # ML-based insights
        ml_insights = insights.get('ml_insights', {})
        if ml_insights and 'top_success_factors' in ml_insights:
            st.markdown("### 🤖 AI-Discovered Success Factors")

            for factor in ml_insights['top_success_factors'][:3]:
                importance = factor['importance']
                impact_level = "🔥 High" if importance > 0.1 else "📈 Medium" if importance > 0.05 else "📊 Low"

                st.info(f"**{impact_level} Impact**: {factor['explanation']}")

    def _render_feature_analysis(self, results: Dict[str, Any]):
        """Render feature analysis tab"""
        data = results.get('data', pd.DataFrame())
        feature_summary = results.get('feature_summary', {})

        if data.empty:
            st.error("No data available for feature analysis")
            return

        st.markdown("## 📊 Feature Analysis Dashboard")

        # Feature summary
        if feature_summary:
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("📊 Total Features", feature_summary.get('feature_count', 0))
            with col2:
                st.metric("🔢 Numeric Features", feature_summary.get('numeric_features', 0))
            with col3:
                missing_data = feature_summary.get('missing_data', {})
                missing_pct = sum(missing_data.values()) / max(len(data), 1) * 100 if missing_data else 0
                st.metric("❓ Missing Data", f"{missing_pct:.1f}%")
            with col4:
                st.metric("📹 Total Videos", len(data))

        # Top correlations with views
        top_correlations = feature_summary.get('top_view_correlations', {})
        if top_correlations:
            st.markdown("### 🔗 Features Most Correlated with Views")

            # Filter out 'views' itself
            filtered_correlations = {k: v for k, v in top_correlations.items() if k != 'views'}

            if filtered_correlations:
                try:
                    corr_df = pd.DataFrame([
                        {
                            'Feature': k.replace('_', ' ').title(),
                            'Correlation': v,
                            'Strength': 'Strong' if abs(v) > 0.5 else 'Moderate' if abs(v) > 0.3 else 'Weak'
                        }
                        for k, v in list(filtered_correlations.items())[:10]
                    ])

                    # Create correlation chart
                    import plotly.express as px
                    fig = px.bar(
                        corr_df,
                        x='Correlation',
                        y='Feature',
                        color='Strength',
                        orientation='h',
                        title="Feature Correlation with Views",
                        color_discrete_map={'Strong': '#ff4444', 'Moderate': '#ffaa44', 'Weak': '#44aaff'}
                    )
                    fig.update_layout(height=400, template='plotly_white')
                    st.plotly_chart(fig, use_container_width=True)

                    # Data table
                    st.dataframe(corr_df, use_container_width=True)
                except Exception as e:
                    st.warning(f"Could not generate correlation chart: {e}")
                    st.dataframe(pd.DataFrame(list(filtered_correlations.items())[:10],
                                              columns=['Feature', 'Correlation']), use_container_width=True)

        # Feature distributions
        if st.session_state.export_options.get('charts', True):
            st.markdown("### 📈 Feature Distributions")

            try:
                # Select interesting numeric features
                numeric_cols = data.select_dtypes(include=[np.number]).columns
                interesting_features = [
                    'title_length', 'duration_seconds', 'publish_hour',
                    'thumbnail_brightness', 'clickbait_score', 'engagement_score'
                ]

                # Filter to available features
                available_features = [f for f in interesting_features if f in numeric_cols]

                if not available_features:
                    # Fallback to any numeric columns
                    available_features = list(numeric_cols[:6])

                if available_features:
                    import plotly.graph_objects as go
                    from plotly.subplots import make_subplots

                    n_features = min(len(available_features), 6)  # Limit to 6 for performance
                    cols = 3
                    rows = (n_features + cols - 1) // cols

                    fig = make_subplots(
                        rows=rows,
                        cols=cols,
                        subplot_titles=[f.replace('_', ' ').title() for f in available_features[:n_features]],
                        vertical_spacing=0.1
                    )

                    for i, feature in enumerate(available_features[:n_features]):
                        row = i // cols + 1
                        col = i % cols + 1

                        fig.add_trace(
                            go.Histogram(x=data[feature], name=feature, showlegend=False),
                            row=row, col=col
                        )

                    fig.update_layout(
                        title='Feature Distributions',
                        template='plotly_white',
                        height=300 * rows
                    )

                    st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.warning(f"Could not generate feature distributions: {e}")

    def _render_export_section(self, results: Dict[str, Any]):
        """Render export section for single channel"""
        st.markdown("## 📥 Export & Download")

        channel_name = results.get('channel_name', 'YouTube_Channel')

        # Quick export buttons
        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("📊 Generate Excel Report", use_container_width=True):
                with st.spinner("Generating Excel report..."):
                    try:
                        excel_file = self.data_exporter.export_to_excel(results)
                        with open(excel_file, 'rb') as f:
                            st.download_button(
                                label="📥 Download Excel Report",
                                data=f.read(),
                                file_name=f"{channel_name}_analysis.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                            )
                        st.success("✅ Excel report generated!")
                    except Exception as e:
                        st.error(f"Error generating Excel: {e}")

        with col2:
            if st.button("📋 Export CSV Data", use_container_width=True):
                try:
                    data = results.get('data', pd.DataFrame())
                    if not data.empty:
                        csv_data = data.to_csv(index=False)
                        st.download_button(
                            label="📥 Download CSV Data",
                            data=csv_data,
                            file_name=f"{channel_name}_data.csv",
                            mime="text/csv"
                        )
                    else:
                        st.error("No data available for export")
                except Exception as e:
                    st.error(f"Error generating CSV: {e}")

        with col3:
            if st.button("📄 Export JSON", use_container_width=True):
                try:
                    json_file = self.data_exporter.export_to_json(results)
                    with open(json_file, 'r', encoding='utf-8') as f:
                        st.download_button(
                            label="📥 Download JSON Data",
                            data=f.read(),
                            file_name=f"{channel_name}_analysis.json",
                            mime="application/json"
                        )
                    st.success("✅ JSON export ready!")
                except Exception as e:
                    st.error(f"Error generating JSON: {e}")

        # Export insights separately
        st.markdown("### 💡 Export Insights")

        col1, col2 = st.columns(2)

        with col1:
            insights = results.get('insights', {})
            if insights:
                import json
                insights_json = json.dumps(insights, indent=2, default=str)
                st.download_button(
                    label="🧠 Download Insights (JSON)",
                    data=insights_json,
                    file_name=f"{channel_name}_insights.json",
                    mime="application/json"
                )

        with col2:
            # Export model results
            model_results = results.get('model_results', {})
            if model_results:
                # Create a simplified model summary
                model_summary = {
                    'r2_score': model_results.get('r2_score', 0),
                    'feature_importance': model_results.get('feature_importance', [])[:20],
                    'model_type': model_results.get('model_type', 'unknown'),
                    'training_samples': model_results.get('training_samples', 0)
                }

                model_json = json.dumps(model_summary, indent=2, default=str)
                st.download_button(
                    label="🤖 Download ML Results (JSON)",
                    data=model_json,
                    file_name=f"{channel_name}_ml_results.json",
                    mime="application/json"
                )

        # Export summary
        st.markdown("### 📋 Export Summary")

        export_info = {
            'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'channel_name': channel_name,
            'total_videos': len(results.get('data', pd.DataFrame())),
            'features_extracted': len(results.get('data', pd.DataFrame()).columns),
            'model_accuracy': results.get('model_results', {}).get('r2_score', 0),
            'export_formats': list(st.session_state.export_options.keys())
        }

        st.json(export_info)

    def _render_channel_rankings(self, results: Dict[str, Any]):
        """Render channel rankings for multi-channel analysis"""
        comparative_analysis = results.get('comparative_analysis', {})
        rankings = comparative_analysis.get('channel_rankings', {})

        if not rankings:
            st.warning("No ranking data available")
            return

        st.markdown("## 🏆 Channel Performance Rankings")

        # Create ranking tabs
        rank_tab1, rank_tab2, rank_tab3 = st.tabs(["📈 By Views", "💝 By Engagement", "🎯 By Predictability"])

        with rank_tab1:
            if 'by_views' in rankings:
                st.markdown("### 📈 Ranked by Average Views")

                views_data = rankings['by_views']
                for i, channel_data in enumerate(views_data, 1):
                    medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"#{i}"

                    st.markdown(f"""
                    <div class="metric-container">
                        <h4>{medal} {channel_data['channel']}</h4>
                        <p><strong>Average Views:</strong> {channel_data['avg_views']:,.0f}</p>
                        <p><strong>Total Videos:</strong> {channel_data['total_videos']}</p>
                    </div>
                    """, unsafe_allow_html=True)

        with rank_tab2:
            if 'by_engagement' in rankings:
                st.markdown("### 💝 Ranked by Engagement")

                engagement_data = rankings['by_engagement']
                for i, channel_data in enumerate(engagement_data, 1):
                    medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"#{i}"

                    st.markdown(f"""
                    <div class="metric-container">
                        <h4>{medal} {channel_data['channel']}</h4>
                        <p><strong>Engagement Rate:</strong> {channel_data['avg_engagement']:.3f}</p>
                        <p><strong>Total Videos:</strong> {channel_data['total_videos']}</p>
                    </div>
                    """, unsafe_allow_html=True)

        with rank_tab3:
            if 'by_predictability' in rankings:
                st.markdown("### 🎯 Ranked by AI Model Accuracy")

                predictability_data = rankings['by_predictability']
                for i, channel_data in enumerate(predictability_data, 1):
                    medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"#{i}"

                    st.markdown(f"""
                    <div class="metric-container">
                        <h4>{medal} {channel_data['channel']}</h4>
                        <p><strong>Model Accuracy:</strong> {channel_data['model_score']:.1%}</p>
                        <p><strong>Total Videos:</strong> {channel_data['total_videos']}</p>
                    </div>
                    """, unsafe_allow_html=True)

    def _render_comparative_analysis(self, results: Dict[str, Any]):
        """Render comparative analysis"""
        st.markdown("## 📊 Comparative Performance Analysis")

        # Channel comparison charts
        if st.session_state.export_options.get('charts', True):
            individual_results = results.get('individual_results', {})

            if len(individual_results) > 1:
                try:
                    # Performance comparison
                    comparison_data = []

                    for channel_name, result in individual_results.items():
                        df = result.get('data', pd.DataFrame())
                        if not df.empty and 'views' in df.columns:
                            channel_stats = {
                                'Channel': channel_name,
                                'Avg Views': df['views'].mean(),
                                'Total Videos': len(df),
                                'Max Views': df['views'].max()
                            }
                            comparison_data.append(channel_stats)

                    if comparison_data:
                        comparison_df = pd.DataFrame(comparison_data)

                        # Create grouped bar chart
                        import plotly.express as px
                        fig = px.bar(
                            comparison_df,
                            x='Channel',
                            y=['Avg Views', 'Max Views'],
                            title='Channel Performance Comparison',
                            barmode='group'
                        )
                        fig.update_layout(template='plotly_white', height=500)
                        st.plotly_chart(fig, use_container_width=True)

                        # Distribution comparison
                        import plotly.graph_objects as go
                        fig = go.Figure()

                        for channel_name, result in individual_results.items():
                            df = result.get('data', pd.DataFrame())
                            if not df.empty and 'views' in df.columns:
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
                        st.plotly_chart(fig, use_container_width=True)

                except Exception as e:
                    st.warning(f"Could not generate comparison charts: {e}")

        # Cross-channel patterns
        comparative_analysis = results.get('comparative_analysis', {})
        cross_patterns = comparative_analysis.get('cross_channel_patterns', {})

        if cross_patterns:
            st.markdown("### 🔄 Universal Success Patterns")

            col1, col2 = st.columns(2)

            with col1:
                if 'universal_best_hours' in cross_patterns:
                    hours = cross_patterns['universal_best_hours']
                    st.success(f"🕐 **Universal best posting hours**: {', '.join(map(str, hours))}")

            with col2:
                if 'universal_content_types' in cross_patterns:
                    content_types = cross_patterns['universal_content_types']
                    if content_types:
                        st.markdown("**📺 Best performing content types:**")
                        for content_type, performance_ratio in content_types.items():
                            clean_name = content_type.replace('is_', '').replace('_', ' ').title()
                            st.info(f"**{clean_name}**: {performance_ratio:.1f}x better than average")

    def _render_best_practices(self, results: Dict[str, Any]):
        """Render best practices from analysis"""
        st.markdown("## 💡 Best Practices & Insights")

        comparative_analysis = results.get('comparative_analysis', {})
        best_practices = comparative_analysis.get('best_practices', [])

        if best_practices:
            st.markdown("### 🏆 Insights from Top Performers")

            for practice in best_practices:
                st.markdown(f"""
                <div class="recommendation-card">
                    <p>✅ {practice}</p>
                </div>
                """, unsafe_allow_html=True)

        # Competitive insights
        competitive_insights = comparative_analysis.get('competitive_insights', {})
        if competitive_insights:
            st.markdown("### 🎯 Competitive Intelligence")

            # Performance gaps
            performance_gaps = competitive_insights.get('performance_gaps', {})
            if performance_gaps:
                max_views = performance_gaps.get('max_avg_views', 0)
                min_views = performance_gaps.get('min_avg_views', 1)
                ratio = performance_gaps.get('performance_ratio', 1)

                st.info(
                    f"📊 **Performance Gap**: Top performer averages {max_views:,.0f} views vs {min_views:,.0f} views (ratio: {ratio:.1f}x)")

            # Growth opportunities
            growth_opportunities = competitive_insights.get('growth_opportunities', [])
            if growth_opportunities:
                st.markdown("#### 🚀 Growth Opportunities")
                for opp in growth_opportunities:
                    st.warning(
                        f"**{opp['channel']}**: Potential to increase views by {opp['percentage_improvement']:.0f}% ({opp['potential_increase']:,.0f} views)")

    def _render_multi_export_section(self, results: Dict[str, Any]):
        """Render export section for multi-channel analysis"""
        st.markdown("## 📥 Multi-Channel Export & Download")

        # Generate comprehensive report
        col1, col2 = st.columns(2)

        with col1:
            if st.button("📊 Generate Comparative Report", use_container_width=True):
                st.info("💡 Comprehensive multi-channel reports will be available in the full version")

        with col2:
            if st.button("📄 Export All Data", use_container_width=True):
                st.info("💡 Bulk export functionality will be available in the full version")

        # Individual channel exports
        st.markdown("### 📋 Individual Channel Data")

        individual_results = results.get('individual_results', {})
        for channel_name, result in individual_results.items():
            with st.expander(f"📺 {channel_name} Exports"):
                col1, col2, col3 = st.columns(3)

                with col1:
                    data = result.get('data', pd.DataFrame())
                    if not data.empty:
                        csv_data = data.to_csv(index=False)
                        st.download_button(
                            label="📊 CSV Data",
                            data=csv_data,
                            file_name=f"{channel_name}_data.csv",
                            mime="text/csv",
                            key=f"csv_{channel_name}"
                        )

                with col2:
                    insights = result.get('insights', {})
                    if insights:
                        import json
                        insights_json = json.dumps(insights, indent=2, default=str)
                        st.download_button(
                            label="💡 Insights JSON",
                            data=insights_json,
                            file_name=f"{channel_name}_insights.json",
                            mime="application/json",
                            key=f"insights_{channel_name}"
                        )

                with col3:
                    model_results = result.get('model_results', {})
                    if model_results:
                        model_summary = {
                            'r2_score': model_results.get('r2_score', 0),
                            'feature_importance': model_results.get('feature_importance', [])[:10]
                        }
                        model_json = json.dumps(model_summary, indent=2, default=str)
                        st.download_button(
                            label="🤖 ML Results",
                            data=model_json,
                            file_name=f"{channel_name}_ml.json",
                            mime="application/json",
                            key=f"ml_{channel_name}"
                        )

    def _quick_export(self):
        """Quick export functionality"""
        if st.session_state.current_analysis == 'single':
            results = st.session_state.analysis_results
            channel_name = results.get('channel_name', 'channel')

            # Generate CSV quickly
            data = results.get('data', pd.DataFrame())
            if not data.empty:
                csv_data = data.to_csv(index=False)
                st.download_button(
                    label="📥 Quick CSV Download",
                    data=csv_data,
                    file_name=f"{channel_name}_quick_export.csv",
                    mime="text/csv"
                )
                st.success("✅ Quick export ready!")
            else:
                st.error("No data available for export")
        else:
            st.info("💡 Use the Export tab for multi-channel downloads")


def main():
    """Main application entry point"""
    try:
        app = YouTubeAnalyticsApp()
        app.run()
    except Exception as e:
        st.error(f"Critical Application Error: {str(e)}")
        st.error("Please refresh the page to restart the application.")
        if st.checkbox("Show Technical Details"):
            st.code(traceback.format_exc())


if __name__ == "__main__":
    main()