"""
YouTube Tools - List videos from YouTube channels.

Uses yt-dlp for extracting video information without API keys.
"""

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Download directory for videos
DOWNLOADS_PATH = Path(__file__).parent.parent / "downloads"

# Track downloaded videos per channel
DOWNLOADED_TRACKER_PATH = Path(__file__).parent.parent / "downloaded_videos.json"

# Track downloaded thumbnails per channel
DOWNLOADED_THUMBNAILS_PATH = Path(__file__).parent.parent / "downloaded_thumbnails.json"


def load_downloaded_videos() -> dict[str, list[str]]:
    """Load the record of downloaded video IDs per channel."""
    if not DOWNLOADED_TRACKER_PATH.exists():
        return {}
    try:
        return json.loads(DOWNLOADED_TRACKER_PATH.read_text(encoding="utf-8"))
    except Exception as e:
        logger.error(f"Error loading downloaded videos tracker: {e}")
        return {}


def save_downloaded_video(channel_id: str, video_id: str) -> None:
    """Record a video as downloaded for a channel."""
    data = load_downloaded_videos()
    if channel_id not in data:
        data[channel_id] = []
    if video_id not in data[channel_id]:
        data[channel_id].append(video_id)
    try:
        DOWNLOADED_TRACKER_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except Exception as e:
        logger.error(f"Error saving downloaded videos tracker: {e}")


def get_downloaded_video_ids(channel_id: str) -> set[str]:
    """Get set of already-downloaded video IDs for a channel."""
    data = load_downloaded_videos()
    return set(data.get(channel_id, []))


def load_downloaded_thumbnails() -> dict[str, list[str]]:
    """Load the record of downloaded thumbnail IDs per channel."""
    if not DOWNLOADED_THUMBNAILS_PATH.exists():
        return {}
    try:
        return json.loads(DOWNLOADED_THUMBNAILS_PATH.read_text(encoding="utf-8"))
    except Exception as e:
        logger.error(f"Error loading downloaded thumbnails tracker: {e}")
        return {}


def save_downloaded_thumbnail(channel_id: str, video_id: str) -> None:
    """Record a thumbnail as downloaded for a channel."""
    data = load_downloaded_thumbnails()
    if channel_id not in data:
        data[channel_id] = []
    if video_id not in data[channel_id]:
        data[channel_id].append(video_id)
    try:
        DOWNLOADED_THUMBNAILS_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except Exception as e:
        logger.error(f"Error saving downloaded thumbnails tracker: {e}")


def get_downloaded_thumbnail_ids(channel_id: str) -> set[str]:
    """Get set of already-downloaded thumbnail IDs for a channel."""
    data = load_downloaded_thumbnails()
    return set(data.get(channel_id, []))


def seed_thumbnails_from_existing_files(channel_id: str, videos: list[dict]) -> int:
    """
    Check existing files in downloads folder and mark matching videos as downloaded.
    This allows retroactively tracking thumbnails that were downloaded before tracking existed.

    Args:
        channel_id: The channel identifier for tracking
        videos: List of video dicts with 'title' and 'id' keys

    Returns:
        Number of thumbnails newly marked as downloaded
    """
    if not DOWNLOADS_PATH.exists():
        return 0

    # Get existing thumbnail files (common image extensions)
    existing_files = set()
    for ext in ["webp", "jpg", "png", "jpeg", "gif"]:
        for f in DOWNLOADS_PATH.glob(f"*.{ext}"):
            # Store filename without extension for matching
            existing_files.add(f.stem)

    if not existing_files:
        return 0

    # Get already-tracked IDs
    already_tracked = get_downloaded_thumbnail_ids(channel_id)

    # Check each video to see if its thumbnail file already exists
    newly_tracked = 0
    for video in videos:
        video_id = video.get("id", "")
        title = video.get("title", "")

        # Skip if already tracked
        if video_id in already_tracked:
            continue

        # Check if a file with this title exists
        if title in existing_files:
            save_downloaded_thumbnail(channel_id, video_id)
            newly_tracked += 1
            logger.info(f"Seeded existing thumbnail: {title}")

    return newly_tracked


def extract_channel_videos(channel_url: str, max_videos: int = 50) -> str:
    """
    List videos from a YouTube channel.

    Args:
        channel_url: YouTube channel URL (e.g., https://www.youtube.com/@ChannelName/videos)
        max_videos: Maximum number of videos to list (default 50)

    Returns:
        Formatted list of videos as a string
    """
    try:
        import yt_dlp
    except ImportError:
        return "Error: yt-dlp not installed. Run: pip install yt-dlp"

    # Normalize URL to videos tab
    if not channel_url.endswith("/videos"):
        channel_url = channel_url.rstrip("/") + "/videos"

    # Configure yt-dlp for fast extraction (no download)
    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "extract_flat": True,  # Don't download, just extract info
        "playlistend": max_videos,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(channel_url, download=False)

            if not info:
                return f"Could not extract info from: {channel_url}"

            channel_name = info.get("channel", info.get("uploader", "Unknown Channel"))
            entries = info.get("entries", [])

            if not entries:
                return f"No videos found for channel: {channel_name}"

            # Format output
            lines = [f"**Videos from {channel_name}** ({len(entries)} videos)\n"]

            for i, video in enumerate(entries, 1):
                if video is None:
                    continue

                title = video.get("title", "Unknown Title")
                video_id = video.get("id", "")
                duration = video.get("duration")

                # Format duration if available
                if duration:
                    mins, secs = divmod(int(duration), 60)
                    hours, mins = divmod(mins, 60)
                    if hours:
                        duration_str = f" ({hours}:{mins:02d}:{secs:02d})"
                    else:
                        duration_str = f" ({mins}:{secs:02d})"
                else:
                    duration_str = ""

                url = f"https://www.youtube.com/watch?v={video_id}" if video_id else ""

                lines.append(f"{i}. {title}{duration_str}")
                if url:
                    lines.append(f"   {url}")

            return "\n".join(lines)

    except Exception as e:
        logger.error(f"YouTube extraction error: {e}")
        return f"Error extracting videos: {e}"


def search_channel_videos(channel_url: str, search_term: str, max_results: int = 20) -> str:
    """
    Search for videos within a YouTube channel.

    Args:
        channel_url: YouTube channel URL
        search_term: Term to search for in video titles
        max_results: Maximum number of results (default 20)

    Returns:
        Formatted list of matching videos
    """
    try:
        import yt_dlp
    except ImportError:
        return "Error: yt-dlp not installed. Run: pip install yt-dlp"

    # Normalize URL to videos tab
    if not channel_url.endswith("/videos"):
        channel_url = channel_url.rstrip("/") + "/videos"

    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "extract_flat": True,
        "playlistend": 200,  # Fetch more to search through
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(channel_url, download=False)

            if not info:
                return f"Could not extract info from: {channel_url}"

            channel_name = info.get("channel", info.get("uploader", "Unknown Channel"))
            entries = info.get("entries", [])

            # Filter by search term (case-insensitive)
            search_lower = search_term.lower()
            matches = []
            for video in entries:
                if video is None:
                    continue
                title = video.get("title", "")
                if search_lower in title.lower():
                    matches.append(video)
                    if len(matches) >= max_results:
                        break

            if not matches:
                return f"No videos matching '{search_term}' found in {channel_name}"

            # Format output
            lines = [
                f"**Videos matching '{search_term}' in {channel_name}** ({len(matches)} found)\n"
            ]

            for i, video in enumerate(matches, 1):
                title = video.get("title", "Unknown Title")
                video_id = video.get("id", "")
                url = f"https://www.youtube.com/watch?v={video_id}" if video_id else ""

                lines.append(f"{i}. {title}")
                if url:
                    lines.append(f"   {url}")

            return "\n".join(lines)

    except Exception as e:
        logger.error(f"YouTube search error: {e}")
        return f"Error searching videos: {e}"


def download_video(url: str) -> str:
    """
    Download a YouTube video using yt-dlp Python library.

    Args:
        url: YouTube video URL

    Returns:
        Status message with file path or error
    """
    try:
        import yt_dlp
    except ImportError:
        return "Error: yt-dlp not installed. Run: pip install yt-dlp"

    # Ensure downloads directory exists
    DOWNLOADS_PATH.mkdir(exist_ok=True)

    # Track downloaded file
    downloaded_file = None

    def progress_hook(d: dict) -> None:
        nonlocal downloaded_file
        if d["status"] == "finished":
            downloaded_file = d.get("filename", d.get("info_dict", {}).get("_filename"))

    ydl_opts = {
        "outtmpl": str(DOWNLOADS_PATH / "%(title)s.%(ext)s"),
        "progress_hooks": [progress_hook],
        "quiet": True,
        "no_warnings": True,
        "sleep_interval": 10,  # Delay between requests to avoid rate limiting
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

        if downloaded_file:
            return f"✅ Downloaded: `{downloaded_file}`"
        return f"✅ Download complete. Check: `{DOWNLOADS_PATH}`"

    except yt_dlp.utils.DownloadError as e:
        error_msg = str(e)
        if "already been downloaded" in error_msg.lower():
            return "ℹ️ Video already downloaded"
        logger.error(f"yt-dlp download error: {e}")
        return f"Error downloading video: {error_msg[:500]}"
    except Exception as e:
        logger.error(f"Download error: {e}")
        return f"Error: {e}"


def get_channel_video_urls(channel_url: str, max_videos: int = 100) -> list[dict]:
    """
    Get video URLs from a YouTube channel.

    Args:
        channel_url: YouTube channel URL
        max_videos: Maximum number of videos to get

    Returns:
        List of dicts with 'title', 'url', 'id' for each video
    """
    logger.info(f"[get_channel_video_urls] START - url={channel_url}, max_videos={max_videos}")

    try:
        import yt_dlp
    except ImportError:
        logger.error("[get_channel_video_urls] yt-dlp not installed!")
        return []

    # Normalize URL to videos tab
    if not channel_url.endswith("/videos"):
        channel_url = channel_url.rstrip("/") + "/videos"
        logger.info(f"[get_channel_video_urls] Normalized URL to: {channel_url}")

    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "extract_flat": True,
        "playlistend": max_videos,
    }
    logger.info(f"[get_channel_video_urls] yt-dlp opts: {ydl_opts}")

    try:
        logger.info("[get_channel_video_urls] Calling yt-dlp extract_info...")
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(channel_url, download=False)

            if not info:
                logger.warning("[get_channel_video_urls] extract_info returned None!")
                return []

            entries = info.get("entries", [])
            logger.info(f"[get_channel_video_urls] Got {len(entries)} entries from channel")

            videos = []
            for video in entries:
                if video is None:
                    continue
                video_id = video.get("id", "")
                if video_id:
                    videos.append(
                        {
                            "title": video.get("title", "Unknown"),
                            "url": f"https://www.youtube.com/watch?v={video_id}",
                            "id": video_id,
                        }
                    )
            logger.info(f"[get_channel_video_urls] END - returning {len(videos)} videos")
            return videos
    except Exception as e:
        logger.error(f"[get_channel_video_urls] ERROR: {e}")
        return []


def extract_channel_id(channel_url: str) -> str:
    """Extract a channel identifier from URL for tracking purposes."""
    # Remove protocol and www
    url = channel_url.replace("https://", "").replace("http://", "").replace("www.", "")
    # Extract channel handle or ID
    if "/@" in url:
        # Handle format: youtube.com/@ChannelName
        return url.split("/@")[1].split("/")[0]
    elif "/channel/" in url:
        # Channel ID format: youtube.com/channel/UCxxxxx
        return url.split("/channel/")[1].split("/")[0]
    elif "/c/" in url:
        # Custom URL format: youtube.com/c/ChannelName
        return url.split("/c/")[1].split("/")[0]
    else:
        # Fallback: use the whole URL as identifier
        return url.split("/")[1] if "/" in url else url


async def download_channel_videos(
    channel_url: str,
    max_videos: int = 50,
    progress_callback=None,
) -> str:
    """
    Download videos from a YouTube channel, skipping already-downloaded ones.

    Args:
        channel_url: YouTube channel URL
        max_videos: Maximum number of NEW videos to download per run
        progress_callback: Async function to call with progress updates

    Returns:
        Summary of downloads
    """
    import asyncio

    channel_id = extract_channel_id(channel_url)

    if progress_callback:
        await progress_callback("📋 Getting video list from channel...")

    # Fetch a larger list to find new videos (up to 500)
    # We'll filter out already-downloaded and take first max_videos new ones
    fetch_count = max(max_videos * 10, 500)
    all_videos = await asyncio.to_thread(get_channel_video_urls, channel_url, fetch_count)

    if not all_videos:
        return "No videos found or error getting video list."

    # Get already-downloaded video IDs
    downloaded_ids = get_downloaded_video_ids(channel_id)

    if progress_callback:
        await progress_callback(
            f"Found {len(all_videos)} videos, {len(downloaded_ids)} already downloaded..."
        )

    # Filter to only videos we haven't downloaded yet
    new_videos = [v for v in all_videos if v["id"] not in downloaded_ids]

    if not new_videos:
        return f"All {len(all_videos)} videos from this channel have already been downloaded!"

    # Take only the requested number of new videos
    videos_to_download = new_videos[:max_videos]
    total = len(videos_to_download)

    if progress_callback:
        await progress_callback(f"Downloading {total} new videos...")

    downloaded = 0
    errors = 0
    failed_videos = []  # Track failed downloads with reasons

    for i, video in enumerate(videos_to_download, 1):
        title = video["title"][:50]  # Truncate long titles
        url = video["url"]
        video_id = video["id"]

        if progress_callback:
            await progress_callback(f"⏳ [{i}/{total}] Downloading: {title}...")

        # Download in thread to not block
        result = await asyncio.to_thread(download_video, url)

        if "✅" in result:
            downloaded += 1
            # Record this video as downloaded
            save_downloaded_video(channel_id, video_id)
        else:
            errors += 1
            # Extract error reason (remove "Error: " prefix if present)
            reason = result.replace("Error: ", "").replace("Error downloading video: ", "")[:100]
            failed_videos.append((title, reason))
            logger.error(f"Failed to download {title}: {result}")

    remaining_new = len(new_videos) - total
    summary = f"""**Channel Download Complete**

✅ Downloaded: {downloaded}
❌ Errors: {errors}
📊 Total tracked: {len(downloaded_ids) + downloaded}
📋 Remaining new: {remaining_new}
📁 Location: `{DOWNLOADS_PATH}`"""

    # Add failed videos section if there were errors
    if failed_videos:
        summary += "\n\n**Failed Downloads:**"
        for title, reason in failed_videos:
            summary += f"\n• {title}: {reason}"

    return summary


def download_thumbnail(url: str) -> str:
    """
    Download the thumbnail/cover art for a YouTube video.

    Args:
        url: YouTube video URL

    Returns:
        Status message with file path or error
    """
    import time

    start_time = time.time()
    logger.info(f"[download_thumbnail] START - url={url}")

    try:
        import yt_dlp
    except ImportError:
        logger.error("[download_thumbnail] yt-dlp not installed!")
        return "Error: yt-dlp not installed. Run: pip install yt-dlp"

    # Ensure downloads directory exists
    DOWNLOADS_PATH.mkdir(exist_ok=True)
    logger.info(f"[download_thumbnail] Downloads path: {DOWNLOADS_PATH}")

    # Track downloaded file
    downloaded_file = None

    ydl_opts = {
        "outtmpl": str(DOWNLOADS_PATH / "%(title)s.%(ext)s"),
        "writethumbnail": True,
        "skip_download": True,  # Don't download the video, just the thumbnail
        "quiet": True,
        "no_warnings": True,
        "sleep_interval": 10,  # Delay between requests to avoid rate limiting
    }
    logger.info(f"[download_thumbnail] yt-dlp opts: {ydl_opts}")

    try:
        logger.info("[download_thumbnail] Creating YoutubeDL instance...")
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            logger.info("[download_thumbnail] Calling extract_info (this triggers the download)...")
            extract_start = time.time()
            info = ydl.extract_info(url, download=True)
            extract_duration = time.time() - extract_start
            logger.info(f"[download_thumbnail] extract_info completed in {extract_duration:.2f}s")

            if info:
                title = info.get("title", "thumbnail")
                video_id = info.get("id", "unknown")
                logger.info(f"[download_thumbnail] Video info - title='{title}', id={video_id}")

                # yt-dlp saves thumbnails with various extensions
                for ext in ["webp", "jpg", "png", "jpeg"]:
                    thumb_path = DOWNLOADS_PATH / f"{title}.{ext}"
                    if thumb_path.exists():
                        downloaded_file = str(thumb_path)
                        logger.info(f"[download_thumbnail] Found thumbnail file: {thumb_path}")
                        break
                else:
                    logger.warning(
                        f"[download_thumbnail] No thumbnail file found for title: {title}"
                    )
            else:
                logger.warning("[download_thumbnail] extract_info returned None!")

        total_duration = time.time() - start_time
        if downloaded_file:
            logger.info(
                f"[download_thumbnail] SUCCESS in {total_duration:.2f}s - file={downloaded_file}"
            )
            return f"✅ Downloaded thumbnail: `{downloaded_file}`"

        logger.info(
            f"[download_thumbnail] PARTIAL SUCCESS in {total_duration:.2f}s - no file path confirmed"
        )
        return f"✅ Thumbnail downloaded. Check: `{DOWNLOADS_PATH}`"

    except Exception as e:
        total_duration = time.time() - start_time
        error_str = str(e)
        logger.error(f"[download_thumbnail] ERROR after {total_duration:.2f}s: {error_str}")

        # Check for rate limiting indicators
        if "rate" in error_str.lower():
            logger.error("[download_thumbnail] !!! RATE LIMIT DETECTED !!!")
        if "unavailable" in error_str.lower():
            logger.error("[download_thumbnail] !!! VIDEO UNAVAILABLE (possibly rate limited) !!!")

        return f"Error downloading thumbnail: {e}"


async def download_channel_thumbnails(
    channel_url: str,
    max_thumbnails: int = 100,
    progress_callback=None,
) -> str:
    """
    Download thumbnails/cover art for all videos in a YouTube channel.
    Tracks already-downloaded thumbnails and skips them on subsequent runs.

    Args:
        channel_url: YouTube channel URL
        max_thumbnails: Maximum number of NEW thumbnails to download
        progress_callback: Async function to call with progress updates

    Returns:
        Summary of downloads
    """
    import asyncio
    import time

    overall_start = time.time()
    logger.info("=" * 60)
    logger.info("[download_channel_thumbnails] START")
    logger.info(f"[download_channel_thumbnails] channel_url={channel_url}")
    logger.info(f"[download_channel_thumbnails] max_thumbnails={max_thumbnails}")
    logger.info("=" * 60)

    channel_id = extract_channel_id(channel_url)
    logger.info(f"[download_channel_thumbnails] Extracted channel_id: {channel_id}")

    if progress_callback:
        await progress_callback("📋 Getting video list from channel...")

    # Fetch all videos from the channel (up to 10000)
    fetch_count = 10000
    logger.info(
        f"[download_channel_thumbnails] Fetching up to {fetch_count} videos from channel..."
    )
    fetch_start = time.time()
    all_videos = await asyncio.to_thread(get_channel_video_urls, channel_url, fetch_count)
    fetch_duration = time.time() - fetch_start
    logger.info(
        f"[download_channel_thumbnails] Fetch completed in {fetch_duration:.2f}s, got {len(all_videos)} videos"
    )

    if not all_videos:
        logger.warning("[download_channel_thumbnails] No videos found! Returning early.")
        return "No videos found or error getting video list."

    # Seed tracker from existing files (retroactive tracking)
    logger.info("[download_channel_thumbnails] Seeding tracker from existing files...")
    seed_start = time.time()
    seeded = await asyncio.to_thread(seed_thumbnails_from_existing_files, channel_id, all_videos)
    seed_duration = time.time() - seed_start
    logger.info(
        f"[download_channel_thumbnails] Seeding completed in {seed_duration:.2f}s, seeded {seeded} thumbnails"
    )

    if seeded > 0 and progress_callback:
        await progress_callback(f"📂 Found {seeded} existing thumbnails, added to tracker...")

    # Get already-downloaded thumbnail IDs (including newly seeded ones)
    downloaded_ids = get_downloaded_thumbnail_ids(channel_id)
    logger.info(f"[download_channel_thumbnails] Already tracked: {len(downloaded_ids)} thumbnails")

    if progress_callback:
        await progress_callback(
            f"Found {len(all_videos)} videos, {len(downloaded_ids)} thumbnails already downloaded..."
        )

    # Filter to only videos we haven't downloaded thumbnails for
    new_videos = [v for v in all_videos if v["id"] not in downloaded_ids]
    logger.info(f"[download_channel_thumbnails] New videos to download: {len(new_videos)}")

    if not new_videos:
        logger.info("[download_channel_thumbnails] All thumbnails already downloaded!")
        return f"All {len(all_videos)} thumbnails from this channel have already been downloaded!"

    # Take only the requested number of new thumbnails
    videos_to_download = new_videos[:max_thumbnails]
    total = len(videos_to_download)
    logger.info(f"[download_channel_thumbnails] Will attempt to download {total} thumbnails")

    if progress_callback:
        await progress_callback(f"Downloading {total} new thumbnails...")

    downloaded = 0
    errors = 0
    skipped = 0
    failed_videos = []

    logger.info("[download_channel_thumbnails] --- STARTING DOWNLOAD LOOP ---")
    loop_start = time.time()

    for i, video in enumerate(videos_to_download, 1):
        title = video["title"][:50]  # Truncate long titles
        url = video["url"]
        video_id = video["id"]

        logger.info(
            f"[download_channel_thumbnails] [{i}/{total}] Processing: {title} (id={video_id})"
        )

        if progress_callback and i % 10 == 1:  # Update every 10 videos to avoid spam
            await progress_callback(f"⏳ [{i}/{total}] Downloading: {title}...")

        # Download thumbnail in thread to not block
        dl_start = time.time()
        logger.info(f"[download_channel_thumbnails] [{i}/{total}] Calling download_thumbnail...")
        result = await asyncio.to_thread(download_thumbnail, url)
        dl_duration = time.time() - dl_start
        logger.info(
            f"[download_channel_thumbnails] [{i}/{total}] download_thumbnail returned in {dl_duration:.2f}s"
        )
        logger.info(f"[download_channel_thumbnails] [{i}/{total}] Result: {result[:100]}...")

        if "✅" in result:
            downloaded += 1
            logger.info(f"[download_channel_thumbnails] [{i}/{total}] SUCCESS - saving to tracker")
            # Record this thumbnail as downloaded
            save_downloaded_thumbnail(channel_id, video_id)
        elif "rate" in result.lower() or "unavailable" in result.lower():
            # Rate limited or unavailable - stop to avoid further issues
            errors += 1
            skipped = total - i
            failed_videos.append((title, "Rate limited - stopping"))
            logger.error(f"[download_channel_thumbnails] !!! RATE LIMITED at {i}/{total} !!!")
            logger.error(f"[download_channel_thumbnails] Full error: {result}")
            logger.error("[download_channel_thumbnails] Stopping download loop!")
            break
        else:
            errors += 1
            reason = result.replace("Error: ", "").replace("Error downloading thumbnail: ", "")[
                :100
            ]
            failed_videos.append((title, reason))
            logger.error(f"[download_channel_thumbnails] [{i}/{total}] FAILED: {reason}")

        # Delay between downloads to avoid rate limiting - 1 per minute
        if i < total:
            delay_seconds = 60  # 1 download per minute to avoid YouTube rate limiting
            logger.info(
                f"[download_channel_thumbnails] Sleeping {delay_seconds}s before next download..."
            )
            await asyncio.sleep(delay_seconds)
            logger.info("[download_channel_thumbnails] Sleep complete, continuing...")

    loop_duration = time.time() - loop_start
    total_duration = time.time() - overall_start
    logger.info("[download_channel_thumbnails] --- DOWNLOAD LOOP COMPLETE ---")
    logger.info(f"[download_channel_thumbnails] Loop took {loop_duration:.2f}s")
    logger.info(f"[download_channel_thumbnails] Total time: {total_duration:.2f}s")
    logger.info(
        f"[download_channel_thumbnails] Downloaded: {downloaded}, Errors: {errors}, Skipped: {skipped}"
    )

    remaining_new = len(new_videos) - total + skipped
    summary = f"""**Channel Thumbnails Download Complete**

✅ Downloaded: {downloaded}
❌ Errors: {errors}
📊 Total tracked: {len(downloaded_ids) + downloaded}
📋 Remaining new: {remaining_new}
📁 Location: `{DOWNLOADS_PATH}`"""

    if skipped > 0:
        summary += "\n\n⚠️ **Stopped early due to rate limiting.** Run again later to continue."

    if failed_videos and len(failed_videos) <= 5:
        summary += "\n\n**Failed Downloads:**"
        for title, reason in failed_videos:
            summary += f"\n• {title}: {reason}"
    elif failed_videos:
        summary += f"\n\n**Failed Downloads:** {len(failed_videos)} (too many to list)"

    logger.info("[download_channel_thumbnails] END - returning summary")
    return summary


def get_youtube_tools() -> list[Any]:
    """Get YouTube tools for the agent."""
    from .core import create_tool

    return [
        create_tool(
            "list_youtube_videos",
            (
                "List videos from a YouTube channel. Provide the channel URL "
                "(e.g., https://www.youtube.com/@ChannelName or https://www.youtube.com/@ChannelName/videos). "
                "Returns video titles, durations, and URLs."
            ),
            {
                "type": "object",
                "properties": {
                    "channel_url": {
                        "type": "string",
                        "description": "YouTube channel URL",
                    },
                    "max_videos": {
                        "type": "integer",
                        "description": "Maximum number of videos to list (default 50)",
                    },
                },
                "required": ["channel_url"],
            },
            lambda channel_url, max_videos=50: extract_channel_videos(channel_url, max_videos),
        ),
        create_tool(
            "search_youtube_channel",
            (
                "Search for videos within a specific YouTube channel. "
                "Useful when looking for a specific topic or video in a channel's library."
            ),
            {
                "type": "object",
                "properties": {
                    "channel_url": {
                        "type": "string",
                        "description": "YouTube channel URL",
                    },
                    "search_term": {
                        "type": "string",
                        "description": "Term to search for in video titles",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results (default 20)",
                    },
                },
                "required": ["channel_url", "search_term"],
            },
            lambda channel_url, search_term, max_results=20: search_channel_videos(
                channel_url, search_term, max_results
            ),
        ),
    ]
