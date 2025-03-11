import os
import hashlib
import pickle
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from pathlib import Path
import functools
import inspect


class Cache:
    """
    Cache manager for storing and retrieving computed results.
    """
    
    def __init__(self, cache_dir: str = "cache", max_age_days: Optional[int] = None):
        """
        Initialize cache manager.
        
        Args:
            cache_dir: Directory to store cache files
            max_age_days: Maximum age of cache files in days (None = no limit)
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_age_days = max_age_days
    
    def _get_cache_key(self, prefix: str, args: Tuple, kwargs: Dict) -> str:
        """
        Generate a cache key from function call arguments.
        
        Args:
            prefix: Prefix for the cache key (usually function name)
            args: Positional arguments
            kwargs: Keyword arguments
            
        Returns:
            Cache key string
        """
        # Convert args and kwargs to a string representation
        args_str = str(args)
        kwargs_str = str(sorted(kwargs.items()))
        
        # Create hash from combined string
        combined = f"{prefix}:{args_str}:{kwargs_str}"
        key = hashlib.md5(combined.encode()).hexdigest()
        
        return f"{prefix}_{key}"
    
    def _get_cache_path(self, key: str) -> Path:
        """
        Get file path for a cache key.
        
        Args:
            key: Cache key
            
        Returns:
            Path object for the cache file
        """
        return self.cache_dir / f"{key}.pkl"
    
    def get(self, key: str) -> Tuple[bool, Any]:
        """
        Get value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Tuple containing:
            - Boolean indicating whether value was found in cache
            - Retrieved value (or None if not found)
        """
        path = self._get_cache_path(key)
        
        if not path.exists():
            return False, None
        
        # Check if cache is too old
        if self.max_age_days is not None:
            file_age_seconds = time.time() - path.stat().st_mtime
            max_age_seconds = self.max_age_days * 24 * 60 * 60
            
            if file_age_seconds > max_age_seconds:
                return False, None
        
        # Load cached value
        try:
            with open(path, "rb") as f:
                value = pickle.load(f)
            return True, value
        except (pickle.PickleError, IOError):
            return False, None
    
    def set(self, key: str, value: Any) -> None:
        """
        Store value in cache.
        
        Args:
            key: Cache key
            value: Value to store
        """
        path = self._get_cache_path(key)
        
        try:
            with open(path, "wb") as f:
                pickle.dump(value, f)
        except (pickle.PickleError, IOError):
            # Ignore cache failures
            pass
    
    def clear(self, prefix: Optional[str] = None) -> int:
        """
        Clear cache files.
        
        Args:
            prefix: Optional prefix to limit which files to clear
            
        Returns:
            Number of files cleared
        """
        pattern = f"{prefix}_*.pkl" if prefix else "*.pkl"
        count = 0
        
        for path in self.cache_dir.glob(pattern):
            try:
                path.unlink()
                count += 1
            except OSError:
                pass
        
        return count


def memoize(cache: Optional[Cache] = None, prefix: Optional[str] = None) -> Callable:
    """
    Decorator for memoizing function results using the cache.
    
    Args:
        cache: Cache object to use (creates a new one if None)
        prefix: Prefix for cache keys (uses function name if None)
        
    Returns:
        Decorated function
    """
    if cache is None:
        cache = Cache()
    
    def decorator(func: Callable) -> Callable:
        func_prefix = prefix or func.__name__
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            key = cache._get_cache_key(func_prefix, args, kwargs)
            
            # Try to get from cache
            found, value = cache.get(key)
            if found:
                return value
            
            # Call function and cache result
            result = func(*args, **kwargs)
            cache.set(key, result)
            
            return result
        
        return wrapper
    
    return decorator


def get_cache_stats(cache_dir: str = "cache") -> Dict[str, Any]:
    """
    Get statistics about the cache.
    
    Args:
        cache_dir: Cache directory
        
    Returns:
        Dictionary with cache statistics
    """
    cache_path = Path(cache_dir)
    
    if not cache_path.exists():
        return {
            "exists": False,
            "file_count": 0,
            "total_size_bytes": 0,
            "oldest_file_age_days": 0,
            "newest_file_age_days": 0
        }
    
    # Get all cache files
    cache_files = list(cache_path.glob("*.pkl"))
    
    if not cache_files:
        return {
            "exists": True,
            "file_count": 0,
            "total_size_bytes": 0,
            "oldest_file_age_days": 0,
            "newest_file_age_days": 0
        }
    
    # Calculate statistics
    total_size = sum(f.stat().st_size for f in cache_files)
    
    # Get file ages
    now = time.time()
    file_ages = [(now - f.stat().st_mtime) / (24 * 60 * 60) for f in cache_files]
    
    return {
        "exists": True,
        "file_count": len(cache_files),
        "total_size_bytes": total_size,
        "total_size_mb": total_size / (1024 * 1024),
        "oldest_file_age_days": max(file_ages) if file_ages else 0,
        "newest_file_age_days": min(file_ages) if file_ages else 0
    } 