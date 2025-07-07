import pandas as pd
import numpy as np
import requests
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import re
import os
import tempfile
from collections import Counter
import warnings

warnings.filterwarnings('ignore')

# Machine Learning imports
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error

# SHAP for model explainability
import shap

# Text processing
from textblob import TextBlob
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Excel export
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.chart import BarChart, Reference

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('vader_lexicon', quiet=True)
except:
    pass


class YouTubeAnalyzer:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://www.googleapis.com/youtube/v3"
        self.clickbait_phrases = [
            'you won\'t believe', 'shocking', 'amazing', 'incredible', 'must watch',
            'gone wrong', 'exposed', 'secret', 'truth', 'revealed', 'hack',
            'trick', 'tips', 'how to', 'tutorial', 'review', 'reaction',
            'vs', 'challenge', 'experiment', 'test', 'comparison', 'ultimate',
            'best', 'worst', 'top', 'epic', 'fail', 'success', 'viral',
            'trending', 'new', 'latest', 'update', 'breaking', 'exclusive'
        ]

    def get_channel_id(self, channel_name):
        """Get channel ID from channel name"""
        try:
            # Try search first
            search_url = f"{self.base_url}/search"
            params = {
                'part': 'snippet',
                'q': channel_name,
                'type': 'channel',
                'key': self.api_key,
                'maxResults': 5
            }

            response = requests.get(search_url, params=params)
            data = response.json()

            if 'items' in data and len(data['items']) > 0:
                # Find exact match or closest match
                for item in data['items']:
                    if channel_name.lower() in item['snippet']['title'].lower():
                        return item['snippet']['channelId']

                # If no exact match, return first result
                return data['items'][0]['snippet']['channelId']

        except Exception as e:
            print(f"Error getting channel ID for {channel_name}: {e}")
            return None

        return None

    def get_channel_videos(self, channel_id, max_videos=100):
        """Fetch videos from a channel"""
        videos = []
        next_page_token = None

        while len(videos) < max_videos:
            try:
                url = f"{self.base_url}/search"
                params = {
                    'part': 'snippet',
                    'channelId': channel_id,
                    'type': 'video',
                    'order': 'date',
                    'maxResults': min(50, max_videos - len(videos)),
                    'key': self.api_key
                }

                if next_page_token:
                    params['pageToken'] = next_page_token

                response = requests.get(url, params=params)
                data = response.json()

                if 'items' in data:
                    video_ids = [item['id']['videoId'] for item in data['items']]
                    video_details = self.get_video_details(video_ids)
                    videos.extend(video_details)

                next_page_token = data.get('nextPageToken')
                if not next_page_token:
                    break

            except Exception as e:
                print(f"Error fetching videos: {e}")
                break

        return videos[:max_videos]

    def get_video_details(self, video_ids):
        """Get detailed stats for videos"""
        videos = []

        # Process in batches of 50
        for i in range(0, len(video_ids), 50):
            batch_ids = video_ids[i:i + 50]

            try:
                url = f"{self.base_url}/videos"
                params = {
                    'part': 'snippet,statistics,contentDetails',
                    'id': ','.join(batch_ids),
                    'key': self.api_key
                }

                response = requests.get(url, params=params)
                data = response.json()

                if 'items' in data:
                    for item in data['items']:
                        video_info = {
                            'video_id': item['id'],
                            'title': item['snippet']['title'],
                            'description': item['snippet'].get('description', ''),
                            'published_at': item['snippet']['publishedAt'],
                            'channel_title': item['snippet']['channelTitle'],
                            'views': int(item['statistics'].get('viewCount', 0)),
                            'likes': int(item['statistics'].get('likeCount', 0)),
                            'comments': int(item['statistics'].get('commentCount', 0)),
                            'duration': item['contentDetails']['duration'],
                            'thumbnail_url': item['snippet']['thumbnails']['high']['url']
                        }
                        videos.append(video_info)

            except Exception as e:
                print(f"Error getting video details: {e}")
                continue

        return videos

    def parse_duration(self, duration_str):
        """Parse ISO 8601 duration to seconds"""
        try:
            # Remove PT prefix
            duration_str = duration_str.replace('PT', '')

            # Extract hours, minutes, seconds
            hours = 0
            minutes = 0
            seconds = 0

            if 'H' in duration_str:
                hours = int(duration_str.split('H')[0])
                duration_str = duration_str.split('H')[1]

            if 'M' in duration_str:
                minutes = int(duration_str.split('M')[0])
                duration_str = duration_str.split('M')[1]

            if 'S' in duration_str:
                seconds = int(duration_str.split('S')[0])

            return hours * 3600 + minutes * 60 + seconds

        except:
            return 0

    def extract_title_features(self, title):
        """Extract features from video title"""
        features = {}

        # Basic features
        features['title_length'] = len(title)
        features['title_word_count'] = len(title.split())
        features['title_uppercase_ratio'] = sum(1 for c in title if c.isupper()) / len(title) if title else 0
        features['title_exclamation_count'] = title.count('!')
        features['title_question_count'] = title.count('?')
        features['title_number_count'] = sum(1 for c in title if c.isdigit())
        features['title_has_numbers'] = 1 if any(c.isdigit() for c in title) else 0

        # Clickbait features
        title_lower = title.lower()
        features['clickbait_score'] = sum(1 for phrase in self.clickbait_phrases if phrase in title_lower)

        # Sentiment analysis
        try:
            blob = TextBlob(title)
            features['title_sentiment_polarity'] = blob.sentiment.polarity
            features['title_sentiment_subjectivity'] = blob.sentiment.subjectivity
        except:
            features['title_sentiment_polarity'] = 0
            features['title_sentiment_subjectivity'] = 0

        # Emotional words
        emotional_words = ['love', 'hate', 'amazing', 'terrible', 'awesome', 'awful', 'great', 'bad']
        features['emotional_word_count'] = sum(1 for word in emotional_words if word in title_lower)

        # Time-related words
        time_words = ['today', 'now', 'new', 'latest', 'recent', 'current', 'live', 'breaking']
        features['time_word_count'] = sum(1 for word in time_words if word in title_lower)

        return features

    def analyze_thumbnail(self, thumbnail_url):
        """Analyze thumbnail image features"""
        features = {}

        try:
            # Download thumbnail
            response = requests.get(thumbnail_url, timeout=10)
            if response.status_code == 200:
                # Convert to OpenCV image
                img_array = np.frombuffer(response.content, np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

                if img is not None:
                    # Basic image features
                    features['thumbnail_brightness'] = np.mean(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
                    features['thumbnail_contrast'] = np.std(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))

                    # Color features
                    b, g, r = cv2.split(img)
                    features['thumbnail_red_mean'] = np.mean(r)
                    features['thumbnail_green_mean'] = np.mean(g)
                    features['thumbnail_blue_mean'] = np.mean(b)

                    # Detect faces using Haar Cascade
                    try:
                        face_cascade = cv2.CascadeClassifier(
                            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
                        features['thumbnail_face_count'] = len(faces)
                    except:
                        features['thumbnail_face_count'] = 0

                    # Edge detection for sharpness
                    edges = cv2.Canny(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 50, 150)
                    features['thumbnail_sharpness'] = np.mean(edges)

                else:
                    # Default values if image processing fails
                    features.update({
                        'thumbnail_brightness': 0,
                        'thumbnail_contrast': 0,
                        'thumbnail_red_mean': 0,
                        'thumbnail_green_mean': 0,
                        'thumbnail_blue_mean': 0,
                        'thumbnail_face_count': 0,
                        'thumbnail_sharpness': 0
                    })

        except Exception as e:
            print(f"Error analyzing thumbnail: {e}")
            # Default values
            features.update({
                'thumbnail_brightness': 0,
                'thumbnail_contrast': 0,
                'thumbnail_red_mean': 0,
                'thumbnail_green_mean': 0,
                'thumbnail_blue_mean': 0,
                'thumbnail_face_count': 0,
                'thumbnail_sharpness': 0
            })

        return features

    def extract_temporal_features(self, published_at):
        """Extract features from publish time"""
        features = {}

        try:
            # Parse datetime
            dt = datetime.fromisoformat(published_at.replace('Z', '+00:00'))

            features['publish_hour'] = dt.hour
            features['publish_day_of_week'] = dt.weekday()  # 0=Monday, 6=Sunday
            features['publish_month'] = dt.month
            features['is_weekend'] = 1 if dt.weekday() >= 5 else 0

            # Peak time features
            features['is_prime_time'] = 1 if 18 <= dt.hour <= 22 else 0
            features['is_morning'] = 1 if 6 <= dt.hour <= 12 else 0
            features['is_afternoon'] = 1 if 12 <= dt.hour <= 18 else 0
            features['is_evening'] = 1 if 18 <= dt.hour <= 24 else 0
            features['is_night'] = 1 if 0 <= dt.hour <= 6 else 0

        except Exception as e:
            print(f"Error extracting temporal features: {e}")
            features.update({
                'publish_hour': 0,
                'publish_day_of_week': 0,
                'publish_month': 1,
                'is_weekend': 0,
                'is_prime_time': 0,
                'is_morning': 0,
                'is_afternoon': 0,
                'is_evening': 0,
                'is_night': 0
            })

        return features

    def train_model(self, df):
        """Train machine learning model to predict views"""
        try:
            # Prepare features
            feature_columns = [col for col in df.columns if col not in
                               ['video_id', 'title', 'description', 'published_at', 'channel_title',
                                'views', 'likes', 'comments', 'thumbnail_url', 'channel_name']]

            X = df[feature_columns].fillna(0)
            y = df['views']

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Train Random Forest
            model = RandomForestRegressor(
                n_estimators=100,
                max_depth=15,
                random_state=42,
                n_jobs=-1
            )
            model.fit(X_train_scaled, y_train)

            # Predictions
            y_pred = model.predict(X_test_scaled)

            # Calculate metrics
            r2 = r2_score(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)

            # Feature importance
            feature_importance = pd.DataFrame({
                'feature': feature_columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)

            return {
                'model': model,
                'scaler': scaler,
                'r2_score': r2,
                'mse': mse,
                'feature_importance': feature_importance,
                'feature_columns': feature_columns
            }

        except Exception as e:
            print(f"Error training model: {e}")
            return None

    def generate_shap_explanations(self, model, X, feature_names):
        """Generate SHAP explanations for model predictions"""
        try:
            # Create explainer
            explainer = shap.TreeExplainer(model)

            # Generate SHAP values
            shap_values = explainer.shap_values(X)

            # Create summary plot
            plt.figure(figsize=(12, 8))
            shap.summary_plot(shap_values, X, feature_names=feature_names, show=False)
            plt.tight_layout()

            # Save plot
            plot_path = tempfile.mktemp(suffix='.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()

            return {
                'shap_values': shap_values,
                'explainer': explainer,
                'plot_path': plot_path
            }

        except Exception as e:
            print(f"Error generating SHAP explanations: {e}")
            return None

    def extract_top_keywords(self, df, top_n=50):
        """Extract most frequent keywords from top-performing videos"""
        try:
            # Get top 20% of videos by views
            top_videos = df.nlargest(int(len(df) * 0.2), 'views')

            # Extract keywords from titles
            all_words = []
            stop_words = set(stopwords.words('english'))

            for title in top_videos['title']:
                # Clean and tokenize
                words = word_tokenize(title.lower())
                words = [word for word in words if word.isalpha() and word not in stop_words and len(word) > 2]
                all_words.extend(words)

            # Count frequency
            word_freq = Counter(all_words)

            # Create DataFrame
            keywords_df = pd.DataFrame({
                'keyword': [word for word, _ in word_freq.most_common(top_n)],
                'frequency': [freq for _, freq in word_freq.most_common(top_n)]
            })

            return keywords_df

        except Exception as e:
            print(f"Error extracting keywords: {e}")
            return pd.DataFrame()

    def create_excel_report(self, df, channel_name, model_results=None, correlations=None, keywords=None):
        """Create comprehensive Excel report"""
        try:
            # Create temporary file
            excel_path = tempfile.mktemp(suffix='.xlsx')

            with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                # Main data sheet
                df.to_excel(writer, sheet_name='Video Data', index=False)

                # Format main sheet
                workbook = writer.book
                worksheet = writer.sheets['Video Data']

                # Header formatting
                header_font = Font(bold=True, color='FFFFFF')
                header_fill = PatternFill(start_color='4472C4', end_color='4472C4', fill_type='solid')

                for col in worksheet[1]:
                    col.font = header_font
                    col.fill = header_fill
                    col.alignment = Alignment(horizontal='center')

                # Auto-adjust column widths
                for column in worksheet.columns:
                    max_length = 0
                    column_letter = column[0].column_letter
                    for cell in column:
                        try:
                            if len(str(cell.value)) > max_length:
                                max_length = len(str(cell.value))
                        except:
                            pass
                    adjusted_width = min(max_length + 2, 50)
                    worksheet.column_dimensions[column_letter].width = adjusted_width

                # Model performance sheet
                if model_results:
                    model_df = pd.DataFrame({
                        'Metric': ['R² Score', 'MSE', 'Feature Count'],
                        'Value': [model_results['r2_score'], model_results['mse'],
                                  len(model_results['feature_columns'])]
                    })
                    model_df.to_excel(writer, sheet_name='Model Performance', index=False)

                    # Feature importance
                    model_results['feature_importance'].to_excel(writer, sheet_name='Feature Importance', index=False)

                # Correlations sheet
                if correlations is not None:
                    correlations.to_excel(writer, sheet_name='Correlations')

                # Keywords sheet
                if keywords is not None and not keywords.empty:
                    keywords.to_excel(writer, sheet_name='Top Keywords', index=False)

                # Summary statistics
                summary_stats = df.describe()
                summary_stats.to_excel(writer, sheet_name='Summary Statistics')

                # Channel overview
                overview_data = {
                    'Metric': [
                        'Channel Name',
                        'Total Videos',
                        'Total Views',
                        'Average Views',
                        'Max Views',
                        'Total Likes',
                        'Average Likes',
                        'Total Comments',
                        'Average Comments'
                    ],
                    'Value': [
                        channel_name,
                        len(df),
                        df['views'].sum(),
                        df['views'].mean(),
                        df['views'].max(),
                        df['likes'].sum(),
                        df['likes'].mean(),
                        df['comments'].sum(),
                        df['comments'].mean()
                    ]
                }
                overview_df = pd.DataFrame(overview_data)
                overview_df.to_excel(writer, sheet_name='Channel Overview', index=False)

            return excel_path

        except Exception as e:
            print(f"Error creating Excel report: {e}")
            return None

    def analyze_channel(self, channel_name, max_videos=100, include_thumbnails=True, include_shap=True):
        """Main function to analyze a YouTube channel"""
        try:
            print(f"Starting analysis for {channel_name}...")

            # Get channel ID
            channel_id = self.get_channel_id(channel_name)
            if not channel_id:
                print(f"Could not find channel: {channel_name}")
                return None

            print(f"Found channel ID: {channel_id}")

            # Get videos
            videos = self.get_channel_videos(channel_id, max_videos)
            if not videos:
                print(f"No videos found for {channel_name}")
                return None

            print(f"Found {len(videos)} videos")

            # Create DataFrame
            df = pd.DataFrame(videos)
            df['channel_name'] = channel_name

            # Extract features
            print("Extracting features...")

            # Title features
            title_features = df['title'].apply(self.extract_title_features)
            title_features_df = pd.DataFrame(title_features.tolist())

            # Duration features
            df['duration_seconds'] = df['duration'].apply(self.parse_duration)
            df['is_short_video'] = (df['duration_seconds'] < 60).astype(int)
            df['is_long_video'] = (df['duration_seconds'] > 1800).astype(int)  # 30 minutes

            # Temporal features
            temporal_features = df['published_at'].apply(self.extract_temporal_features)
            temporal_features_df = pd.DataFrame(temporal_features.tolist())

            # Thumbnail features
            if include_thumbnails:
                print("Analyzing thumbnails...")
                thumbnail_features = df['thumbnail_url'].apply(self.analyze_thumbnail)
                thumbnail_features_df = pd.DataFrame(thumbnail_features.tolist())
            else:
                # Create empty thumbnail features
                thumbnail_features_df = pd.DataFrame({
                    'thumbnail_brightness': [0] * len(df),
                    'thumbnail_contrast': [0] * len(df),
                    'thumbnail_red_mean': [0] * len(df),
                    'thumbnail_green_mean': [0] * len(df),
                    'thumbnail_blue_mean': [0] * len(df),
                    'thumbnail_face_count': [0] * len(df),
                    'thumbnail_sharpness': [0] * len(df)
                })

            # Combine all features
            df_features = pd.concat([
                df.reset_index(drop=True),
                title_features_df.reset_index(drop=True),
                temporal_features_df.reset_index(drop=True),
                thumbnail_features_df.reset_index(drop=True)
            ], axis=1)

            # Engagement features
            df_features['views_per_day'] = df_features['views'] / (
                    (datetime.now() - pd.to_datetime(df_features['published_at']).dt.tz_localize(None)).dt.days + 1
            )
            df_features['like_ratio'] = df_features['likes'] / (df_features['views'] + 1)
            df_features['comment_ratio'] = df_features['comments'] / (df_features['views'] + 1)
            df_features['engagement_score'] = (df_features['likes'] + df_features['comments']) / (
                        df_features['views'] + 1)

            # Train ML model
            print("Training machine learning model...")
            model_results = self.train_model(df_features)

            # Calculate correlations
            numeric_cols = df_features.select_dtypes(include=[np.number]).columns
            correlations = df_features[numeric_cols].corr()

            # Extract top keywords
            print("Extracting viral keywords...")
            keywords = self.extract_top_keywords(df_features)

            # Generate SHAP explanations
            shap_results = None
            if include_shap and model_results:
                print("Generating SHAP explanations...")
                feature_data = df_features[model_results['feature_columns']].fillna(0)
                scaled_data = model_results['scaler'].transform(feature_data)
                shap_results = self.generate_shap_explanations(
                    model_results['model'],
                    scaled_data[:100],  # Use first 100 samples for SHAP
                    model_results['feature_columns']
                )

            # Create Excel report
            print("Creating Excel report...")
            excel_file = self.create_excel_report(
                df_features,
                channel_name,
                model_results,
                correlations,
                keywords
            )

            # Prepare results
            results = {
                'channel_name': channel_name,
                'data': df_features,
                'excel_file': excel_file
            }

            if model_results:
                results.update({
                    'model_score': model_results['r2_score'],
                    'feature_importance': model_results['feature_importance'].to_dict('records')
                })

            if not correlations.empty:
                results['correlations'] = correlations

            if not keywords.empty:
                results['top_keywords'] = keywords.to_dict('records')

            if shap_results:
                results['shap_results'] = shap_results

            print(f"Analysis complete for {channel_name}!")
            return results

        except Exception as e:
            print(f"Error analyzing channel {channel_name}: {e}")
            return None


# Utility functions for additional analysis
def compare_channels(results_list):
    """Compare multiple channels"""
    comparison_data = []

    for result in results_list:
        channel_data = {
            'channel_name': result['channel_name'],
            'total_videos': len(result['data']),
            'avg_views': result['data']['views'].mean(),
            'max_views': result['data']['views'].max(),
            'avg_likes': result['data']['likes'].mean(),
            'avg_comments': result['data']['comments'].mean(),
            'avg_engagement': result['data']['engagement_score'].mean(),
            'model_score': result.get('model_score', 0)
        }
        comparison_data.append(channel_data)

    return pd.DataFrame(comparison_data)


def generate_insights(results):
    """Generate actionable insights from analysis"""
    insights = []

    if 'feature_importance' in results:
        # Top features for success
        top_features = results['feature_importance'][:5]
        insights.append("🎯 Top Success Factors:")
        for feature in top_features:
            insights.append(f"   • {feature['feature']}: {feature['importance']:.3f}")

    # Performance insights
    data = results['data']
    top_videos = data.nlargest(5, 'views')

    insights.append("\n🔥 Top Performing Videos:")
    for _, video in top_videos.iterrows():
        insights.append(f"   • {video['title'][:50]}... - {video['views']:,} views")

    # Timing insights
    best_hour = data.groupby('publish_hour')['views'].mean().idxmax()
    best_day = data.groupby('publish_day_of_week')['views'].mean().idxmax()
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

    insights.append(f"\n⏰ Optimal Posting Time:")
    insights.append(f"   • Best hour: {best_hour}:00")
    insights.append(f"   • Best day: {days[best_day]}")

    # Title insights
    avg_title_length = data['title_length'].mean()
    best_title_length = data.loc[data['views'].idxmax(), 'title_length']

    insights.append(f"\n📝 Title Optimization:")
    insights.append(f"   • Average title length: {avg_title_length:.0f} characters")
    insights.append(f"   • Best performing title length: {best_title_length} characters")

    if 'top_keywords' in results:
        top_keywords = [kw['keyword'] for kw in results['top_keywords'][:5]]
        insights.append(f"\n🔤 Viral Keywords: {', '.join(top_keywords)}")

    return '\n'.join(insights)


# Example usage and testing
if __name__ == "__main__":
    # Example usage
    print("YouTube Analytics Backend - Ready for use!")
    print("Import this module in your Streamlit app:")
    print("from youtube_backend import YouTubeAnalyzer")
    print("\nExample:")
    print("analyzer = YouTubeAnalyzer('YOUR_API_KEY')")
    print("results = analyzer.analyze_channel('Channel Name')")
    print("insights = generate_insights(results)")