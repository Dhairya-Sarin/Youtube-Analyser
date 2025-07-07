import asyncio
import aiohttp
import time
from typing import Dict, List, Any, Optional
import logging
from urllib.parse import urlencode
from config.settings import YouTubeConfig
from .exceptions import DataCollectionError, APIError


class YouTubeDataCollector:
    """Handles all YouTube API data collection"""

    def __init__(self, config: YouTubeConfig):
        self.config = config
        self.base_url = "https://www.googleapis.com/youtube/v3"
        self.logger = logging.getLogger(__name__)
        self.request_count = 0
        self.last_request_time = 0

    async def get_channel_data(self, channel_name: str, max_videos: int) -> Dict[str, Any]:
        """Get comprehensive channel data including videos and channel info"""
        try:
            # Get channel ID and info
            channel_info = await self._get_channel_info(channel_name)
            if not channel_info:
                raise DataCollectionError(f"Channel not found: {channel_name}")

            channel_id = channel_info['id']

            # Get videos
            videos = await self._get_channel_videos(channel_id, max_videos)

            # Enhance video data with additional details
            enhanced_videos = await self._enhance_video_data(videos)

            return {
                'channel_info': channel_info,
                'videos': enhanced_videos,
                'collection_timestamp': time.time()
            }

        except Exception as e:
            self.logger.error(f"Error collecting data for {channel_name}: {e}")
            raise DataCollectionError(f"Failed to collect data for {channel_name}: {e}")

    async def _get_channel_info(self, channel_name: str) -> Optional[Dict[str, Any]]:
        """Get channel information by name"""
        # First try to search for the channel
        search_params = {
            'part': 'snippet',
            'q': channel_name,
            'type': 'channel',
            'maxResults': 5,
            'key': self.config.api_key
        }

        search_results = await self._make_api_request('search', search_params)

        if not search_results.get('items'):
            return None

        # Find the best matching channel
        best_match = None
        for item in search_results['items']:
            if channel_name.lower() in item['snippet']['title'].lower():
                best_match = item
                break

        if not best_match:
            best_match = search_results['items'][0]

        channel_id = best_match['snippet']['channelId']

        # Get detailed channel information
        channel_params = {
            'part': 'snippet,statistics,contentDetails,brandingSettings',
            'id': channel_id,
            'key': self.config.api_key
        }

        channel_details = await self._make_api_request('channels', channel_params)

        if channel_details.get('items'):
            channel = channel_details['items'][0]
            return {
                'id': channel['id'],
                'title': channel['snippet']['title'],
                'description': channel['snippet']['description'],
                'published_at': channel['snippet']['publishedAt'],
                'subscriber_count': int(channel['statistics'].get('subscriberCount', 0)),
                'video_count': int(channel['statistics'].get('videoCount', 0)),
                'view_count': int(channel['statistics'].get('viewCount', 0)),
                'thumbnail_url': channel['snippet']['thumbnails']['high']['url'],
                'country': channel['snippet'].get('country', ''),
                'custom_url': channel['snippet'].get('customUrl', ''),
                'branding': channel.get('brandingSettings', {})
            }

        return None

    async def _get_channel_videos(self, channel_id: str, max_videos: int) -> List[Dict[str, Any]]:
        """Get videos from a channel"""
        videos = []
        next_page_token = None

        while len(videos) < max_videos:
            params = {
                'part': 'snippet',
                'channelId': channel_id,
                'type': 'video',
                'order': 'date',
                'maxResults': min(50, max_videos - len(videos)),
                'key': self.config.api_key
            }

            if next_page_token:
                params['pageToken'] = next_page_token

            search_results = await self._make_api_request('search', params)

            if not search_results.get('items'):
                break

            # Extract video IDs
            video_ids = [item['id']['videoId'] for item in search_results['items']]

            # Get detailed video information
            video_details = await self._get_video_details(video_ids)
            videos.extend(video_details)

            next_page_token = search_results.get('nextPageToken')
            if not next_page_token:
                break

        return videos[:max_videos]

    async def _get_video_details(self, video_ids: List[str]) -> List[Dict[str, Any]]:
        """Get detailed information for videos"""
        videos = []

        # Process in batches of 50 (API limit)
        for i in range(0, len(video_ids), 50):
            batch_ids = video_ids[i:i + 50]

            params = {
                'part': 'snippet,statistics,contentDetails,status,topicDetails',
                'id': ','.join(batch_ids),
                'key': self.config.api_key
            }

            video_data = await self._make_api_request('videos', params)

            if video_data.get('items'):
                for item in video_data['items']:
                    video_info = {
                        'id': item['id'],
                        'title': item['snippet']['title'],
                        'description': item['snippet'].get('description', ''),
                        'published_at': item['snippet']['publishedAt'],
                        'channel_id': item['snippet']['channelId'],
                        'channel_title': item['snippet']['channelTitle'],
                        'tags': item['snippet'].get('tags', []),
                        'category_id': int(item['snippet'].get('categoryId', 0)),
                        'default_language': item['snippet'].get('defaultLanguage', ''),
                        'duration': item['contentDetails']['duration'],
                        'dimension': item['contentDetails']['dimension'],
                        'definition': item['contentDetails']['definition'],
                        'caption': item['contentDetails']['caption'],
                        'licensed_content': item['contentDetails']['licensedContent'],
                        'projection': item['contentDetails']['projection'],
                        'upload_status': item['status']['uploadStatus'],
                        'privacy_status': item['status']['privacyStatus'],
                        'license': item['status']['license'],
                        'embeddable': item['status']['embeddable'],
                        'public_stats_viewable': item['status']['publicStatsViewable'],
                        'view_count': int(item['statistics'].get('viewCount', 0)),
                        'like_count': int(item['statistics'].get('likeCount', 0)),
                        'dislike_count': int(item['statistics'].get('dislikeCount', 0)),
                        'comment_count': int(item['statistics'].get('commentCount', 0)),
                        'favorite_count': int(item['statistics'].get('favoriteCount', 0)),
                        'thumbnail_url': item['snippet']['thumbnails']['high']['url'],
                        'topic_categories': item.get('topicDetails', {}).get('topicCategories', [])
                    }
                    videos.append(video_info)

        return videos

    async def _enhance_video_data(self, videos: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Enhance video data with additional computed fields"""
        enhanced_videos = []

        for video in videos:
            # Add computed fields
            video['duration_seconds'] = self._parse_duration(video['duration'])
            video['is_short'] = video['duration_seconds'] < 60
            video['is_long'] = video['duration_seconds'] > 1800  # 30 minutes
            video['days_since_upload'] = self._calculate_days_since_upload(video['published_at'])

            # Engagement metrics
            views = video['view_count']
            likes = video['like_count']
            comments = video['comment_count']

            video['engagement_score'] = (likes + comments) / max(views, 1)
            video['like_ratio'] = likes / max(likes + video['dislike_count'], 1)
            video['comment_ratio'] = comments / max(views, 1)

            enhanced_videos.append(video)

        return enhanced_videos

    def _parse_duration(self, duration_str: str) -> int:
        """Parse ISO 8601 duration to seconds"""
        if not duration_str:
            return 0

        try:
            duration_str = duration_str.replace('PT', '')

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

    def _calculate_days_since_upload(self, published_at: str) -> int:
        """Calculate days since video was uploaded"""
        try:
            from datetime import datetime
            published_date = datetime.fromisoformat(published_at.replace('Z', '+00:00'))
            return (datetime.now() - published_date.replace(tzinfo=None)).days
        except:
            return 0

    async def _make_api_request(self, endpoint: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Make rate-limited API request"""
        await self._enforce_rate_limit()

        url = f"{self.base_url}/{endpoint}?" + urlencode(params)

        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(url, timeout=self.config.timeout_seconds) as response:
                    self.request_count += 1
                    self.last_request_time = time.time()

                    if response.status == 200:
                        return await response.json()
                    elif response.status == 403:
                        error_data = await response.json()
                        if 'quotaExceeded' in str(error_data):
                            raise APIError("YouTube API quota exceeded")
                        else:
                            raise APIError(f"YouTube API error: {error_data}")
                    else:
                        raise APIError(f"HTTP {response.status}: {await response.text()}")

            except asyncio.TimeoutError:
                raise APIError("YouTube API request timeout")

    async def _enforce_rate_limit(self):
        """Enforce API rate limiting"""
        current_time = time.time()

        # Simple rate limiting: max requests per minute
        if self.request_count > 0:
            time_since_last = current_time - self.last_request_time
            min_interval = 60.0 / self.config.max_requests_per_minute

            if time_since_last < min_interval:
                sleep_time = min_interval - time_since_last
                await asyncio.sleep(sleep_time)
