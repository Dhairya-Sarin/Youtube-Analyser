import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from youtube_backend import YouTubeAnalyzer
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import warnings

warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="🎯 YouTube Analytics Pro",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False
if 'all_results' not in st.session_state:
    st.session_state.all_results = []
if 'analysis_started' not in st.session_state:
    st.session_state.analysis_started = False

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #FF0000;
        font-size: 3rem;
        font-weight: bold;
        margin-bottom: 2rem;
    }
    .metric-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        margin: 10px 0;
    }
    .success-box {
        background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
        padding: 15px;
        border-radius: 8px;
        color: white;
        margin: 10px 0;
    }
    .feature-importance {
        background: linear-gradient(135deg, #FF6B6B 0%, #FF8E8E 100%);
        padding: 15px;
        border-radius: 8px;
        color: white;
        margin: 10px 0;
    }
    .stDownloadButton > button {
        background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
        border: none;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    .stDownloadButton > button:hover {
        background: linear-gradient(135deg, #45a049 0%, #4CAF50 100%);
        transform: translateY(-2px);
    }
</style>
""", unsafe_allow_html=True)

# Main header
st.markdown('<h1 class="main-header">🎯 YouTube Analytics Pro</h1>', unsafe_allow_html=True)
st.markdown("### 🧠 Predict Video Success with AI • 📊 Analyze Top Creators • 🚀 Optimize Your Content")

# Sidebar configuration
st.sidebar.markdown("## 🔧 Configuration")
st.sidebar.markdown("---")

# API Key input - Store in session state
if 'api_key' not in st.session_state:
    st.session_state.api_key = ""

api_key = st.sidebar.text_input(
    "🔑 YouTube Data API Key",
    type="password",
    help="Get your API key from Google Cloud Console",
    value=st.session_state.api_key,
    key="api_key_input"
)

# Store API key in session state
if api_key:
    st.session_state.api_key = api_key

# Channel input - Store in session state
if 'channels_input' not in st.session_state:
    st.session_state.channels_input = ""

channels_input = st.sidebar.text_area(
    "Enter channel names (one per line)",
    placeholder="Dhruv Rathee\nSlayy Point\nFactTechz\nTechnical Guruji",
    height=100,
    value=st.session_state.channels_input,
    key="channels_input_area"
)

# Store channels input in session state
if channels_input:
    st.session_state.channels_input = channels_input

# Analysis parameters
st.sidebar.markdown("### ⚙️ Analysis Settings")
max_videos = st.sidebar.slider("Max videos per channel", 50, 500, 100, key="max_videos_slider")
include_thumbnails = st.sidebar.checkbox("Analyze thumbnails", value=True, key="thumbnails_checkbox")
include_shap = st.sidebar.checkbox("Generate SHAP explanations", value=True, key="shap_checkbox")

# Process channels
channels = [ch.strip() for ch in channels_input.split('\n') if ch.strip()]

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("## 🎯 How It Works")
    st.markdown("""
    1. **🔍 Data Collection**: Fetches video data from YouTube API
    2. **🖼️ Thumbnail Analysis**: Analyzes brightness, faces, colors
    3. **📝 Title Processing**: Extracts keywords, sentiment, clickbait phrases
    4. **🤖 ML Training**: Trains Random Forest to predict views
    5. **🧠 SHAP Insights**: Explains what makes videos go viral
    6. **📊 Visual Reports**: Interactive charts and correlations
    """)

with col2:
    st.markdown("## 📈 What You'll Get")
    st.markdown("""
    - 🔥 Top performing videos
    - 📊 Feature correlation heatmap
    - 🧠 SHAP importance plots
    - 🔤 Viral keywords analysis
    - 📥 Complete Excel report
    - 🎯 Success predictions
    """)

# Reset analysis button
if st.session_state.analysis_complete:
    if st.button("🔄 Reset Analysis", type="secondary", key="reset_analysis"):
        st.session_state.analysis_complete = False
        st.session_state.all_results = []
        st.session_state.analysis_started = False
        st.rerun()

# Analysis button
if not st.session_state.analysis_complete:
    if st.button("🚀 Start Analysis", type="primary", use_container_width=True, key="start_analysis"):
        if not api_key:
            st.error("❌ Please enter your YouTube API key")
        elif not channels:
            st.error("❌ Please enter at least one channel name")
        else:
            st.session_state.analysis_started = True
            # Initialize analyzer
            analyzer = YouTubeAnalyzer(api_key)

            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()

            all_results = []

            for i, channel in enumerate(channels):
                try:
                    status_text.text(f"🔍 Analyzing {channel}...")
                    progress_bar.progress((i + 0.5) / len(channels))

                    # Analyze channel
                    results = analyzer.analyze_channel(
                        channel,
                        max_videos=max_videos,
                        include_thumbnails=include_thumbnails,
                        include_shap=include_shap
                    )

                    if results:
                        all_results.append(results)
                        status_text.text(f"✅ Completed {channel}")
                    else:
                        st.warning(f"⚠️ Could not analyze {channel}")

                    progress_bar.progress((i + 1) / len(channels))

                except Exception as e:
                    st.error(f"❌ Error analyzing {channel}: {str(e)}")

            status_text.text("🎉 Analysis Complete!")
            progress_bar.progress(1.0)

            # Store results in session state
            st.session_state.all_results = all_results
            st.session_state.analysis_complete = True

            # Force rerun to show results
            st.rerun()

# Display results if analysis is complete
if st.session_state.analysis_complete and st.session_state.all_results:
    # Display results
    st.markdown("---")
    st.markdown("# 📊 Analysis Results")

    # Create tabs for different views
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["📈 Overview", "🔥 Top Videos", "🧠 ML Insights", "🎯 SHAP Analysis", "📋 Detailed Data"])

    with tab1:
        st.markdown("## 📊 Channel Performance Overview")

        # Combine all data
        all_data = pd.concat([r['data'] for r in st.session_state.all_results], ignore_index=True)

        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Videos", len(all_data))
        with col2:
            st.metric("Avg Views", f"{all_data['views'].mean():,.0f}")
        with col3:
            st.metric("Avg Likes", f"{all_data['likes'].mean():,.0f}")
        with col4:
            st.metric("Channels", len(st.session_state.all_results))

        # Views distribution
        fig = px.histogram(
            all_data,
            x='views',
            nbins=50,
            title="📈 Views Distribution",
            labels={'views': 'Views', 'count': 'Number of Videos'}
        )
        st.plotly_chart(fig, use_container_width=True)

        # Performance by channel
        if len(st.session_state.all_results) > 1:
            channel_stats = all_data.groupby('channel_name').agg({
                'views': ['mean', 'max'],
                'likes': 'mean',
                'comments': 'mean'
            }).round(0)

            fig = px.bar(
                x=channel_stats.index,
                y=channel_stats[('views', 'mean')],
                title="📊 Average Views by Channel",
                labels={'x': 'Channel', 'y': 'Average Views'}
            )
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.markdown("## 🔥 Top Performing Videos")

        # Show top videos across all channels
        all_data = pd.concat([r['data'] for r in st.session_state.all_results], ignore_index=True)
        top_videos = all_data.nlargest(20, 'views')[
            ['title', 'channel_name', 'views', 'likes', 'comments', 'published_at']
        ]

        for idx, video in top_videos.iterrows():
            with st.expander(
                    f"🎯 {video['title'][:100]}..." if len(video['title']) > 100 else f"🎯 {video['title']}"):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"**Channel:** {video['channel_name']}")
                    st.write(f"**Published:** {video['published_at']}")
                with col2:
                    st.metric("Views", f"{video['views']:,}")
                    st.metric("Likes", f"{video['likes']:,}")
                    st.metric("Comments", f"{video['comments']:,}")

    with tab3:
        st.markdown("## 🧠 Machine Learning Insights")

        for result in st.session_state.all_results:
            st.markdown(f"### 📺 {result['channel_name']}")

            # Model performance
            if 'model_score' in result:
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"""
                    <div class="success-box">
                        <h4>🎯 Model Accuracy</h4>
                        <h2>{result['model_score']:.3f}</h2>
                        <p>R² Score (closer to 1.0 = better)</p>
                    </div>
                    """, unsafe_allow_html=True)

                with col2:
                    # Feature importance
                    if 'feature_importance' in result:
                        importance_df = pd.DataFrame({
                            'feature': result['feature_importance']['feature'],
                            'importance': result['feature_importance']['importance']
                        }).head(10)

                        fig = px.bar(
                            importance_df,
                            x='importance',
                            y='feature',
                            orientation='h',
                            title="🔍 Top Feature Importance"
                        )
                        st.plotly_chart(fig, use_container_width=True)

            # Correlation heatmap
            if 'correlations' in result:
                st.markdown("#### 📊 Feature Correlations")

                # Create heatmap
                corr_matrix = result['correlations']

                # Create interactive heatmap
                fig = px.imshow(
                    corr_matrix,
                    color_continuous_scale='RdBu',
                    aspect='auto',
                    title="Feature Correlation Matrix",
                    text_auto=True,
                    labels={'x': 'Features', 'y': 'Features', 'color': 'Correlation'}
                )
                fig.update_layout(height=600)
                st.plotly_chart(fig, use_container_width=True)

                # Download correlation matrix as CSV
                csv_corr = corr_matrix.to_csv()
                st.download_button(
                    label="📥 Download Correlation Matrix (CSV)",
                    data=csv_corr,
                    file_name=f"{result['channel_name']}_correlations.csv",
                    mime="text/csv",
                    key=f"corr_matrix_{result['channel_name']}"
                )

                # Show top correlations in a table
                st.markdown("**🔍 Top Correlations with Views:**")
                if 'views' in corr_matrix.columns:
                    views_corr = corr_matrix['views'].drop('views').sort_values(key=abs, ascending=False)

                    corr_table = pd.DataFrame({
                        'Feature': views_corr.index,
                        'Correlation': views_corr.values,
                        'Strength': ['Strong' if abs(x) > 0.5 else 'Moderate' if abs(x) > 0.3 else 'Weak' for x
                                     in views_corr.values]
                    })

                    st.dataframe(corr_table, use_container_width=True)

                    # Download correlations table
                    csv_corr_table = corr_table.to_csv(index=False)
                    st.download_button(
                        label="📋 Download Correlations Table (CSV)",
                        data=csv_corr_table,
                        file_name=f"{result['channel_name']}_correlations_table.csv",
                        mime="text/csv",
                        key=f"corr_table_{result['channel_name']}"
                    )

            # Top keywords - FIXED THIS SECTION
            if 'top_keywords' in result and result['top_keywords']:
                st.markdown("#### 🔤 Viral Keywords")

                # Handle the case where top_keywords is a list of dictionaries
                if isinstance(result['top_keywords'], list) and len(result['top_keywords']) > 0:
                    # Create keyword display
                    keywords_display = []
                    for kw in result['top_keywords'][:8]:
                        if isinstance(kw, dict) and 'keyword' in kw and 'frequency' in kw:
                            keywords_display.append(f"{kw['keyword']} ({kw['frequency']})")
                        else:
                            # Handle other possible formats
                            keywords_display.append(str(kw))

                    st.write("**Top Keywords:** " + ", ".join(keywords_display))

                    # Word cloud generation
                    try:
                        wordcloud_data = {}
                        for kw in result['top_keywords'][:20]:
                            if isinstance(kw, dict) and 'keyword' in kw and 'frequency' in kw:
                                wordcloud_data[kw['keyword']] = kw['frequency']

                        if wordcloud_data:
                            wordcloud = WordCloud(
                                width=800,
                                height=400,
                                background_color='white',
                                colormap='viridis'
                            ).generate_from_frequencies(wordcloud_data)

                            fig, ax = plt.subplots(figsize=(10, 5))
                            ax.imshow(wordcloud, interpolation='bilinear')
                            ax.axis('off')
                            st.pyplot(fig)
                    except Exception as e:
                        st.write(f"Could not generate word cloud: {str(e)}")

            st.markdown("---")

    with tab4:
        st.markdown("## 🎯 SHAP Analysis - Explainable AI")
        st.markdown("*Understanding what makes videos go viral through AI explanations*")

        for result in st.session_state.all_results:
            st.markdown(f"### 📺 {result['channel_name']} - SHAP Insights")

            # SHAP Summary Plot
            if 'shap_summary' in result:
                st.markdown("#### 🧠 SHAP Summary Plot")
                st.markdown("*Shows the impact of each feature on video views prediction*")

                # Display SHAP summary plot
                st.image(result['shap_summary'], caption="SHAP Summary Plot - Feature Impact on Views")

                # Download SHAP summary plot
                st.download_button(
                    label="🖼️ Download SHAP Summary Plot",
                    data=result['shap_summary'],
                    file_name=f"{result['channel_name']}_shap_summary.png",
                    mime="image/png",
                    key=f"shap_summary_{result['channel_name']}"
                )

            # SHAP Waterfall Plot
            if 'shap_waterfall' in result:
                st.markdown("#### 🌊 SHAP Waterfall Plot")
                st.markdown("*Shows how features contribute to a specific prediction*")

                st.image(result['shap_waterfall'], caption="SHAP Waterfall Plot - Prediction Breakdown")

                st.download_button(
                    label="🖼️ Download SHAP Waterfall Plot",
                    data=result['shap_waterfall'],
                    file_name=f"{result['channel_name']}_shap_waterfall.png",
                    mime="image/png",
                    key=f"shap_waterfall_{result['channel_name']}"
                )

            # SHAP Feature Importance
            if 'shap_importance' in result:
                st.markdown("#### 📊 SHAP Feature Importance")

                # Display as DataFrame
                shap_df = pd.DataFrame({
                    'Feature': result['shap_importance']['feature'],
                    'SHAP_Value': result['shap_importance']['importance'],
                    'Impact': ['High' if abs(x) > 0.1 else 'Medium' if abs(x) > 0.05 else 'Low'
                               for x in result['shap_importance']['importance']]
                })

                st.dataframe(shap_df, use_container_width=True)

                # Download SHAP importance
                csv_shap = shap_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download SHAP Importance (CSV)",
                    data=csv_shap,
                    file_name=f"{result['channel_name']}_shap_importance.csv",
                    mime="text/csv",
                    key=f"shap_importance_{result['channel_name']}"
                )

            # SHAP Explanations
            if 'shap_explanations' in result:
                st.markdown("#### 🔍 SHAP Explanations")

                # Top positive influences
                st.markdown("**🚀 Features that INCREASE views:**")
                for explanation in result['shap_explanations'].get('positive', [])[:5]:
                    st.write(f"• **{explanation['feature']}**: {explanation['explanation']}")

                # Top negative influences
                st.markdown("**⚠️ Features that DECREASE views:**")
                for explanation in result['shap_explanations'].get('negative', [])[:5]:
                    st.write(f"• **{explanation['feature']}**: {explanation['explanation']}")

            # SHAP Dependence Plots
            if 'shap_dependence' in result:
                st.markdown("#### 📈 SHAP Dependence Plots")
                st.markdown("*Shows how feature values affect predictions*")

                for i, (feature, plot_data) in enumerate(result['shap_dependence'].items()):
                    st.image(plot_data, caption=f"SHAP Dependence Plot - {feature}")

                    st.download_button(
                        label=f"🖼️ Download {feature} Dependence Plot",
                        data=plot_data,
                        file_name=f"{result['channel_name']}_shap_dependence_{feature}.png",
                        mime="image/png",
                        key=f"shap_dep_{result['channel_name']}_{i}"
                    )

            # SHAP Global Feature Importance
            if 'shap_global' in result:
                st.markdown("#### 🌍 Global SHAP Feature Importance")

                fig_shap_global, ax = plt.subplots(figsize=(10, 6))
                features = result['shap_global']['features']
                importance = result['shap_global']['importance']

                bars = ax.barh(features, importance)
                ax.set_xlabel('Mean |SHAP Value|')
                ax.set_title('Global Feature Importance (SHAP)')

                # Color bars based on importance
                for i, bar in enumerate(bars):
                    if importance[i] > 0:
                        bar.set_color('green' if importance[i] > 0.05 else 'orange')
                    else:
                        bar.set_color('red')

                plt.tight_layout()
                st.pyplot(fig_shap_global)

                # Save and download
                import io

                img_buffer = io.BytesIO()
                fig_shap_global.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
                img_buffer.seek(0)

                st.download_button(
                    label="🖼️ Download Global SHAP Importance",
                    data=img_buffer.getvalue(),
                    file_name=f"{result['channel_name']}_shap_global_importance.png",
                    mime="image/png",
                    key=f"shap_global_{result['channel_name']}"
                )

            # SHAP Insights Summary
            if 'shap_insights' in result:
                st.markdown("#### 💡 Key SHAP Insights")

                insights_df = pd.DataFrame({
                    'Insight': result['shap_insights']['insights'],
                    'Impact': result['shap_insights']['impact'],
                    'Recommendation': result['shap_insights']['recommendations']
                })

                st.dataframe(insights_df, use_container_width=True)

                # Download insights
                csv_insights = insights_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download SHAP Insights (CSV)",
                    data=csv_insights,
                    file_name=f"{result['channel_name']}_shap_insights.csv",
                    mime="text/csv",
                    key=f"shap_insights_{result['channel_name']}"
                )

            st.markdown("---")

    with tab5:
        st.markdown("## 📋 Detailed Analysis Data")

        for result in st.session_state.all_results:
            st.markdown(f"### 📺 {result['channel_name']}")

            # Display data
            display_data = result['data'].copy()

            # Format numeric columns
            numeric_cols = ['views', 'likes', 'comments', 'duration']
            for col in numeric_cols:
                if col in display_data.columns:
                    def format_number(x):
                        try:
                            # Check if it's a number (int or float)
                            if pd.notna(x) and isinstance(x, (int, float)):
                                return f"{int(x):,}"
                            elif pd.notna(x) and str(x).replace('.', '').replace('-', '').isdigit():
                                return f"{int(float(x)):,}"
                            else:
                                return x
                        except (ValueError, TypeError):
                            return x


                    display_data[col] = display_data[col].apply(format_number)

            st.dataframe(display_data, use_container_width=True)

            # Download button
            if 'excel_file' in result:
                with open(result['excel_file'], 'rb') as file:
                    st.download_button(
                        label=f"📥 Download {result['channel_name']} Report",
                        data=file.read(),
                        file_name=f"{result['channel_name']}_analysis.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        key=f"excel_{result['channel_name']}"
                    )

            # Additional export options
            st.markdown("**📋 Additional Export Options:**")
            col1, col2, col3 = st.columns(3)

            with col1:
                # Export raw data as CSV
                csv_data = result['data'].to_csv(index=False)
                st.download_button(
                    label="📊 Raw Data (CSV)",
                    data=csv_data,
                    file_name=f"{result['channel_name']}_raw_data.csv",
                    mime="text/csv",
                    key=f"raw_data_{result['channel_name']}"
                )

            with col2:
                # Export feature importance
                if 'feature_importance' in result:
                    importance_df = pd.DataFrame({
                        'feature': result['feature_importance']['feature'],
                        'importance': result['feature_importance']['importance']
                    })
                    csv_importance = importance_df.to_csv(index=False)
                    st.download_button(
                        label="🔍 Feature Importance (CSV)",
                        data=csv_importance,
                        file_name=f"{result['channel_name']}_feature_importance.csv",
                        mime="text/csv",
                        key=f"feature_importance_{result['channel_name']}"
                    )

            with col3:
                # Export keywords
                if 'top_keywords' in result and result['top_keywords']:
                    keywords_df = pd.DataFrame(result['top_keywords'])
                    csv_keywords = keywords_df.to_csv(index=False)
                    st.download_button(
                        label="🔤 Keywords (CSV)",
                        data=csv_keywords,
                        file_name=f"{result['channel_name']}_keywords.csv",
                        mime="text/csv",
                        key=f"keywords_{result['channel_name']}"
                    )

            # Create comprehensive JSON export
            comprehensive_data = {
                'channel_name': result['channel_name'],
                'model_score': result.get('model_score', 'N/A'),
                'total_videos': len(result['data']),
                'avg_views': result['data']['views'].mean(),
                'correlations': result['correlations'].to_dict() if 'correlations' in result else {},
                'feature_importance': result.get('feature_importance', {}),
                'top_keywords': result.get('top_keywords', []),
                'summary_stats': result['data'].describe().to_dict()
            }

            import json

            json_data = json.dumps(comprehensive_data, indent=2, default=str)
            st.download_button(
                label="📋 Comprehensive Analysis (JSON)",
                data=json_data,
                file_name=f"{result['channel_name']}_comprehensive_analysis.json",
                mime="application/json",
                key=f"comprehensive_{result['channel_name']}"
            )

            st.markdown("---")

elif st.session_state.analysis_started and not st.session_state.all_results:
    st.error("❌ No results to display. Please check your API key and channel names.")

# Footer
st.markdown("---")
st.markdown("### 🔗 Resources")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    **🔑 Get YouTube API Key:**
    1. Go to [Google Cloud Console](https://console.cloud.google.com/)
    2. Create a new project
    3. Enable YouTube Data API v3
    4. Create credentials (API key)
    """)

with col2:
    st.markdown("""
    **📊 Features Analyzed:**
    - Title length & sentiment
    - Thumbnail brightness & faces
    - Upload timing patterns
    - Keyword effectiveness
    - Engagement metrics
    """)

with col3:
    st.markdown("""
    **🎯 Use Cases:**
    - Content strategy optimization
    - Competitor analysis
    - Viral content prediction
    - Thumbnail A/B testing
    - Title optimization
    """)

st.markdown("---")
st.markdown("*Built with ❤️ using Streamlit, scikit-learn, and SHAP*")