# 🎯 YouTube Analytics Pro

Advanced AI-powered YouTube analytics platform with comprehensive feature extraction, machine learning predictions, and interactive visualizations.

![YouTube Analytics Pro](https://img.shields.io/badge/YouTube-Analytics-red?style=for-the-badge&logo=youtube)
![Python](https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-green?style=for-the-badge&logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

## 🌟 Features

### 🔍 Comprehensive Data Collection
- YouTube Data API v3 integration
- Channel and video metadata extraction
- Performance metrics and engagement data
- Automatic data validation and cleaning

### 🖼️ Computer Vision Analysis
- **Thumbnail Analysis**: Face detection, color analysis, composition scoring
- **Visual Complexity**: Edge detection, symmetry analysis, rule of thirds
- **Clickbait Detection**: Visual pattern recognition for engagement elements
- **Aesthetic Scoring**: Golden ratio and color harmony analysis

### 📝 Natural Language Processing
- **Title Analysis**: Sentiment analysis, clickbait scoring, psychological features
- **Description Processing**: Link detection, hashtag analysis, call-to-action identification
- **Content Classification**: Tutorial, review, reaction, challenge detection
- **Keyword Extraction**: N-gram analysis and topic modeling

### 🤖 Machine Learning & AI
- **Predictive Models**: Random Forest, Gradient Boosting, Neural Networks, Ensemble methods
- **Feature Engineering**: 150+ automated feature extraction
- **Model Explainability**: SHAP integration for AI interpretability
- **Hyperparameter Tuning**: Automated optimization with GridSearchCV

### 📊 Interactive Visualizations
- Performance timeline charts
- Correlation heatmaps
- Feature distribution plots
- Comparative channel analysis
- Top content rankings

### 📥 Export Capabilities
- Excel reports with multiple sheets
- CSV data exports
- JSON analysis results
- Interactive HTML dashboards

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- YouTube Data API v3 key
- 4GB+ RAM recommended

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/your-username/youtube-analytics-pro.git
cd youtube-analytics-pro
```

2. **Create virtual environment**
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download NLTK data** (first time only)
```python
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
```

### 🔑 Getting YouTube API Key

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select existing one
3. Enable **YouTube Data API v3**
4. Go to "Credentials" → "Create Credentials" → "API Key"
5. Copy your API key

### 🏃‍♂️ Running the Application

```bash
streamlit run main.py
```

The application will open in your browser at `http://localhost:8501`

## 📖 Usage Guide

### Basic Analysis

1. **Enter API Key**: Paste your YouTube API key in the sidebar
2. **Add Channels**: Enter channel names (one per line) in the sidebar
3. **Configure Settings**: Choose feature complexity and ML options
4. **Start Analysis**: Click "Quick Analysis" for fast results or "Deep Analysis" for comprehensive insights

### Advanced Configuration

#### Feature Sets
- **Basic**: ~50 core features (fastest)
- **Extended**: ~100 features with thumbnails and NLP
- **Full**: 150+ features with all advanced analysis
- **Custom**: User-defined feature selection

#### Machine Learning Models
- **Random Forest**: Best for feature importance analysis
- **Gradient Boosting**: High accuracy predictions
- **Neural Network**: Complex pattern recognition
- **Ensemble**: Combines multiple models for best results

### Understanding Results

#### Performance Overview
- View distribution analysis
- Engagement rate calculations
- Performance timeline visualization
- Top performing content identification

#### AI Insights
- Feature importance rankings
- SHAP explainability plots
- Predictive model accuracy metrics
- Success factor identification

#### Recommendations
- Optimal posting times
- Content optimization suggestions
- Title length recommendations
- Engagement improvement strategies

## 🏗️ Architecture

```
youtube-analytics-pro/
├── config/                 # Configuration files
│   ├── constants.py        # Feature definitions and mappings
│   └── settings.py         # Application settings and enums
├── core/                   # Core analysis engine
│   ├── analyzer.py         # Main orchestrator
│   ├── data_collector.py   # YouTube API integration
│   └── exceptions.py       # Custom exception classes
├── features/               # Feature extraction modules
│   ├── video_metadata.py   # Basic video properties
│   ├── temporal_features.py # Time-based analysis
│   ├── engagement_metrics.py # Performance metrics
│   ├── title_features.py    # NLP title analysis
│   ├── thumbnail_features.py # Computer vision
│   ├── description_features.py # Content analysis
│   ├── channel_features.py  # Channel-level metrics
│   └── audio_features.py    # Audio analysis (optional)
├── ml/                     # Machine learning components
│   ├── models.py           # Model training and evaluation
│   ├── prediction.py       # Prediction engine
│   └── explainability.py   # SHAP integration
├── visualization/          # Charts and exports
│   ├── charts.py           # Interactive visualizations
│   └── export.py           # Data export utilities
├── utils/                  # Utility functions
│   ├── data_validation.py  # Data cleaning and validation
│   ├── image_processing.py # Computer vision utilities
│   ├── text_processing.py  # NLP utilities
│   └── cache.py            # Caching system
├── tests/                  # Unit tests
└── main.py                 # Streamlit application
```

## 🔧 Configuration

### Environment Variables
```bash
export YOUTUBE_API_KEY="your_api_key_here"
export LOG_LEVEL="INFO"
export CACHE_ENABLED="true"
```

### Settings File
Modify `config/settings.py` for default configurations:

```python
# Example custom settings
settings = AppSettings(
    youtube=YouTubeConfig(
        api_key="your_key",
        max_videos_per_channel=200,
        max_requests_per_minute=100
    ),
    features=FeatureConfig(
        include_thumbnails=True,
        include_advanced_nlp=True,
        feature_set=FeatureSet.FULL
    ),
    ml=MLConfig(
        model_type=ModelType.ENSEMBLE,
        enable_shap=True,
        hyperparameter_tuning=True
    )
)
```

## 📊 Feature Reference

### Core Features (50+)
- Video metadata (duration, category, tags)
- Engagement metrics (views, likes, comments)
- Temporal features (upload time, seasonality)
- Basic title analysis (length, word count)

### Extended Features (100+)
- Thumbnail computer vision analysis
- Advanced NLP processing
- Content type classification
- Channel-level metrics

### Full Features (150+)
- Audio analysis (optional)
- Advanced visual composition scoring
- Psychological title features
- Predictive engagement modeling

## 🧪 Testing

Run the test suite:
```bash
python -m pytest tests/ -v
```

Run specific test modules:
```bash
python -m pytest tests/test_features/test_extractors.py -v
```

## 📈 Performance

### Optimization Tips
1. **Use Basic feature set** for faster analysis
2. **Limit video count** for quicker results
3. **Disable thumbnails** if computer vision not needed
4. **Enable caching** for repeated analyses

### System Requirements
- **Minimum**: 4GB RAM, 2 CPU cores
- **Recommended**: 8GB RAM, 4 CPU cores
- **Storage**: 1GB free space for caching

## 🛠️ Troubleshooting

### Common Issues

#### API Quota Exceeded
```
Solution: Check your Google Cloud Console quota limits
- Increase daily quota or wait for reset
- Reduce max_videos_per_channel setting
```

#### Memory Errors
```
Solution: Reduce analysis scope
- Use Basic feature set instead of Full
- Analyze fewer videos per channel
- Disable thumbnail analysis
```

#### Import Errors
```
Solution: Install missing dependencies
pip install -r requirements.txt
```

#### Slow Performance
```
Solution: Optimize settings
- Disable SHAP explanations
- Reduce cross-validation folds
- Use Random Forest instead of Neural Network
```

### Debug Mode
Enable debug logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

### Development Setup
```bash
pip install -e .[development]
pre-commit install
```

### Code Style
```bash
black .
isort .
flake8 .
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **YouTube Data API** for comprehensive video data
- **Streamlit** for the interactive web interface
- **SHAP** for explainable AI capabilities
- **OpenCV** for computer vision features
- **scikit-learn** for machine learning models

## 📞 Support

- 🐛 **Bug Reports**: [GitHub Issues](https://github.com/your-username/youtube-analytics-pro/issues)
- 💡 **Feature Requests**: [GitHub Discussions](https://github.com/your-username/youtube-analytics-pro/discussions)
- 📧 **Email**: support@youtube-analytics-pro.com
- 📖 **Documentation**: [Wiki](https://github.com/your-username/youtube-analytics-pro/wiki)

## 🔮 Roadmap

### Version 2.1
- [ ] Real-time analytics dashboard
- [ ] A/B testing for thumbnails
- [ ] Competitor analysis features
- [ ] API for programmatic access

### Version 2.2
- [ ] Video transcript analysis
- [ ] Comment sentiment analysis
- [ ] Revenue prediction models
- [ ] Mobile app companion

---

**Made with ❤️ for YouTube creators and analysts**