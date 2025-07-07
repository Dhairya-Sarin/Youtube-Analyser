import pickle
import hashlib
import os
from datetime import datetime, timedelta
from typing import Any, Optional
import logging
from config.settings import CacheConfig
from core.exceptions import CacheError


class CacheManager:
    """Manages caching of analysis results"""

    def __init__(self, config: CacheConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Create cache directory
        os.makedirs(config.cache_dir, exist_ok=True)

        # Clean up old cache files on startup
        self._cleanup_cache()

    async def get(self, key: str) -> Optional[Any]:
        """Retrieve item from cache"""
        try:
            cache_file = self._get_cache_file(key)

            if not os.path.exists(cache_file):
                return None

            # Check if cache is expired
            if self._is_expired(cache_file):
                os.remove(cache_file)
                return None

            # Load and return cached data
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)
                self.logger.debug(f"Cache hit for key: {key}")
                return data

        except Exception as e:
            self.logger.warning(f"Error reading from cache: {e}")
            return None

    async def set(self, key: str, value: Any) -> bool:
        """Store item in cache"""
        try:
            cache_file = self._get_cache_file(key)

            # Check cache size limits
            if not self._check_cache_size():
                self._cleanup_cache(force=True)

            # Store data
            with open(cache_file, 'wb') as f:
                pickle.dump(value, f)

            self.logger.debug(f"Cached data for key: {key}")
            return True

        except Exception as e:
            self.logger.warning(f"Error writing to cache: {e}")
            return False

    def _get_cache_file(self, key: str) -> str:
        """Generate cache file path for key"""
        # Create hash of key for filename
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return os.path.join(self.config.cache_dir, f"{key_hash}.cache")

    def _is_expired(self, cache_file: str) -> bool:
        """Check if cache file is expired"""
        try:
            mtime = datetime.fromtimestamp(os.path.getmtime(cache_file))
            expiry_time = mtime + timedelta(hours=self.config.cache_ttl_hours)
            return datetime.now() > expiry_time
        except:
            return True

    def _check_cache_size(self) -> bool:
        """Check if cache size is within limits"""
        try:
            total_size = 0
            for filename in os.listdir(self.config.cache_dir):
                filepath = os.path.join(self.config.cache_dir, filename)
                if os.path.isfile(filepath):
                    total_size += os.path.getsize(filepath)

            size_mb = total_size / (1024 * 1024)
            return size_mb < self.config.max_cache_size_mb

        except:
            return True

    def _cleanup_cache(self, force: bool = False):
        """Clean up expired or excess cache files"""
        try:
            cache_files = []

            for filename in os.listdir(self.config.cache_dir):
                filepath = os.path.join(self.config.cache_dir, filename)
                if os.path.isfile(filepath) and filename.endswith('.cache'):
                    cache_files.append((filepath, os.path.getmtime(filepath)))

            # Remove expired files
            removed_count = 0
            for filepath, mtime in cache_files:
                if force or self._is_expired(filepath):
                    try:
                        os.remove(filepath)
                        removed_count += 1
                    except:
                        pass

            # If force cleanup, remove oldest files if still over size limit
            if force and not self._check_cache_size():
                # Sort by modification time, oldest first
                cache_files.sort(key=lambda x: x[1])

                while not self._check_cache_size() and cache_files:
                    filepath, _ = cache_files.pop(0)
                    try:
                        if os.path.exists(filepath):
                            os.remove(filepath)
                            removed_count += 1
                    except:
                        pass

            if removed_count > 0:
                self.logger.info(f"Cleaned up {removed_count} cache files")

        except Exception as e:
            self.logger.warning(f"Error during cache cleanup: {e}")
