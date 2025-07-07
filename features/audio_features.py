import numpy as np
from typing import Dict, Any, List, Optional
import tempfile
import os
from .base_extractor import BaseFeatureExtractor


class AudioFeatureExtractor(BaseFeatureExtractor):
    """Extract audio and speech features from videos (advanced feature)"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.feature_names = [
            'audio_duration_seconds', 'audio_has_speech', 'speech_duration_ratio',
            'speech_word_count', 'speech_words_per_minute', 'speech_pause_count',
            'speech_avg_pause_duration', 'speech_sentiment_polarity', 'speech_emotion_score',
            'speech_volume_mean', 'speech_volume_std', 'speech_pitch_mean', 'speech_pitch_std',
            'has_background_music', 'music_volume_ratio', 'audio_quality_score',
            'speech_clarity_score', 'audio_noise_level', 'speech_energy_level'
        ]

        # Try to import audio processing libraries
        self.audio_available = self._check_audio_dependencies()

    def _check_audio_dependencies(self) -> bool:
        """Check if audio processing dependencies are available"""
        try:
            import librosa
            import speech_recognition as sr
            return True
        except ImportError:
            return False

    def extract(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract audio features from video"""
        if not self.audio_available:
            return {name: 0 for name in self.feature_names}

        video_id = data.get('video_id', '')
        duration = data.get('duration_seconds', 0)

        if not video_id or duration == 0:
            return {name: 0 for name in self.feature_names}

        try:
            # Download audio (in production, you'd need proper YouTube audio extraction)
            audio_file = self._download_audio(video_id)
            if not audio_file:
                return {name: 0 for name in self.feature_names}

            # Extract features
            features = {}
            features.update(self._extract_basic_audio_features(audio_file, duration))
            features.update(self._extract_speech_features(audio_file))
            features.update(self._extract_music_features(audio_file))

            # Cleanup
            if os.path.exists(audio_file):
                os.remove(audio_file)

            return features

        except Exception as e:
            print(f"Error extracting audio features: {e}")
            return {name: 0 for name in self.feature_names}

    def _download_audio(self, video_id: str) -> Optional[str]:
        """Download audio from video (placeholder - requires yt-dlp or similar)"""
        # Note: In production, you'd use yt-dlp or similar tool
        # This is a placeholder implementation

        try:
            import yt_dlp

            # Create temporary file
            temp_file = tempfile.mktemp(suffix='.wav')

            ydl_opts = {
                'format': 'bestaudio/best',
                'outtmpl': temp_file,
                'quiet': True,
                'no_warnings': True,
                'extractaudio': True,
                'audioformat': 'wav',
                'audio_quality': 0,  # Best quality
            }

            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                url = f"https://www.youtube.com/watch?v={video_id}"
                ydl.download([url])

            return temp_file if os.path.exists(temp_file) else None

        except:
            return None

    def _extract_basic_audio_features(self, audio_file: str, duration: int) -> Dict[str, Any]:
        """Extract basic audio properties"""
        try:
            import librosa

            # Load audio
            y, sr = librosa.load(audio_file, duration=min(duration, 300))  # Limit to 5 minutes for processing

            features = {}
            features['audio_duration_seconds'] = len(y) / sr

            # Volume analysis
            rms = librosa.feature.rms(y=y)[0]
            features['speech_volume_mean'] = np.mean(rms)
            features['speech_volume_std'] = np.std(rms)

            # Pitch analysis
            pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
            pitch_values = pitches[magnitudes > np.median(magnitudes)]

            if len(pitch_values) > 0:
                features['speech_pitch_mean'] = np.mean(pitch_values[pitch_values > 0])
                features['speech_pitch_std'] = np.std(pitch_values[pitch_values > 0])
            else:
                features['speech_pitch_mean'] = 0
                features['speech_pitch_std'] = 0

            # Audio quality indicators
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            features['audio_quality_score'] = np.mean(spectral_centroids) / 4000  # Normalize

            # Noise level estimation
            onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
            if len(onset_frames) > 0:
                features['audio_noise_level'] = len(onset_frames) / features['audio_duration_seconds']
            else:
                features['audio_noise_level'] = 0

            # Energy level
            features['speech_energy_level'] = np.mean(np.abs(y))

            return features

        except Exception as e:
            print(f"Error in basic audio analysis: {e}")
            return {name: 0 for name in ['audio_duration_seconds', 'speech_volume_mean',
                                         'speech_volume_std', 'speech_pitch_mean', 'speech_pitch_std',
                                         'audio_quality_score', 'audio_noise_level', 'speech_energy_level']}

    def _extract_speech_features(self, audio_file: str) -> Dict[str, Any]:
        """Extract speech-related features"""
        features = {
            'audio_has_speech': 0,
            'speech_duration_ratio': 0,
            'speech_word_count': 0,
            'speech_words_per_minute': 0,
            'speech_pause_count': 0,
            'speech_avg_pause_duration': 0,
            'speech_sentiment_polarity': 0,
            'speech_emotion_score': 0,
            'speech_clarity_score': 0
        }

        try:
            import speech_recognition as sr
            from textblob import TextBlob

            # Initialize recognizer
            r = sr.Recognizer()

            # Load audio file
            with sr.AudioFile(audio_file) as source:
                # Adjust for ambient noise
                r.adjust_for_ambient_noise(source, duration=1)

                # Record audio data
                audio_data = r.record(source, duration=60)  # Limit to first minute

            # Recognize speech
            try:
                text = r.recognize_google(audio_data)

                if text:
                    features['audio_has_speech'] = 1

                    # Word count and rate
                    words = text.split()
                    features['speech_word_count'] = len(words)
                    features['speech_words_per_minute'] = len(words)  # Approximate for 1 minute sample

                    # Sentiment analysis of speech
                    blob = TextBlob(text)
                    features['speech_sentiment_polarity'] = blob.sentiment.polarity

                    # Simple emotion scoring (based on word choice)
                    emotional_words = ['excited', 'amazing', 'terrible', 'awesome', 'shocked']
                    emotion_count = sum(1 for word in emotional_words if word in text.lower())
                    features['speech_emotion_score'] = min(emotion_count / 10.0, 1.0)

                    # Speech clarity proxy (longer recognized text suggests clearer speech)
                    features['speech_clarity_score'] = min(len(text) / 500.0, 1.0)

            except sr.UnknownValueError:
                # Could not understand audio
                pass
            except sr.RequestError:
                # Could not request results
                pass

        except Exception as e:
            print(f"Error in speech analysis: {e}")

        return features

    def _extract_music_features(self, audio_file: str) -> Dict[str, Any]:
        """Extract music and background audio features"""
        features = {
            'has_background_music': 0,
            'music_volume_ratio': 0
        }

        try:
            import librosa

            # Load audio
            y, sr = librosa.load(audio_file, duration=60)  # First minute

            # Detect harmonic content (indicator of music)
            harmonic, percussive = librosa.effects.hpss(y)

            # Check for sustained harmonic content
            harmonic_strength = np.mean(np.abs(harmonic))
            percussive_strength = np.mean(np.abs(percussive))

            if harmonic_strength > percussive_strength * 0.5:
                features['has_background_music'] = 1
                features['music_volume_ratio'] = harmonic_strength / max(np.mean(np.abs(y)), 0.001)

        except Exception as e:
            print(f"Error in music analysis: {e}")

        return features

    def get_feature_names(self) -> List[str]:
        return self.feature_names
