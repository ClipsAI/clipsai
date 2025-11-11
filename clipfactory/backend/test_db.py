"""
Test script for database integration
Run this to verify database connectivity and basic CRUD operations
"""
import asyncio
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

from database import get_db_context, startup_event, shutdown_event
from crud import (
    create_video, get_video_by_id, get_videos,
    create_clip, get_clips_by_video,
    create_music_track, get_all_music_tracks,
    get_video_statistics
)


async def test_database_integration():
    """Test database integration with basic CRUD operations."""

    print("=" * 60)
    print("Database Integration Test")
    print("=" * 60)

    try:
        # Initialize database
        print("\n1. Initializing database...")
        await startup_event()
        print("✓ Database initialized successfully")

        # Test video creation
        print("\n2. Creating test video...")
        async with get_db_context() as db:
            video = await create_video(
                db=db,
                filename="test_video.mp4",
                file_path="/uploads/test_video.mp4",
                file_size=1024000,
                duration=60.5,
                resolution="1920x1080",
                fps=30.0,
                metadata={"test": True}
            )
            print(f"✓ Created video: {video.id}")
            video_id = video.id

        # Test video retrieval
        print("\n3. Retrieving video...")
        async with get_db_context() as db:
            retrieved_video = await get_video_by_id(db, video_id)
            if retrieved_video:
                print(f"✓ Retrieved video: {retrieved_video.filename}")
                print(f"  - Status: {retrieved_video.status}")
                print(f"  - Size: {retrieved_video.file_size} bytes")
            else:
                print("✗ Failed to retrieve video")
                return False

        # Test clip creation
        print("\n4. Creating test clips...")
        async with get_db_context() as db:
            clip1 = await create_clip(
                db=db,
                video_id=video_id,
                start_time=10.0,
                end_time=20.0,
                transcript="This is a test clip",
                hook_score=0.85,
                hook_score_data={"confidence": 0.9}
            )
            clip2 = await create_clip(
                db=db,
                video_id=video_id,
                start_time=30.0,
                end_time=45.0,
                transcript="Another test clip",
                hook_score=0.92,
                hook_score_data={"confidence": 0.95}
            )
            print(f"✓ Created 2 clips: {clip1.id}, {clip2.id}")

        # Test clip retrieval
        print("\n5. Retrieving clips for video...")
        async with get_db_context() as db:
            clips = await get_clips_by_video(db, video_id)
            print(f"✓ Found {len(clips)} clips")
            for clip in clips:
                print(f"  - Clip {clip.id}: {clip.start_time}s - {clip.end_time}s (score: {clip.hook_score})")

        # Test music tracks
        print("\n6. Creating test music tracks...")
        async with get_db_context() as db:
            track1 = await create_music_track(
                db=db,
                name="Energetic Beat 1",
                file_path="/music/energetic_1.mp3",
                vibe="High energy",
                context_description="Use for intense training moments",
                color="#FF5722",
                bpm=140,
                duration=180.0
            )
            track2 = await create_music_track(
                db=db,
                name="Chill Vibes 1",
                file_path="/music/chill_1.mp3",
                vibe="Relaxed",
                context_description="Use for recovery or cool-down",
                color="#4CAF50",
                bpm=90,
                duration=200.0
            )
            print(f"✓ Created 2 music tracks")

        # Test music retrieval
        print("\n7. Retrieving music tracks...")
        async with get_db_context() as db:
            tracks = await get_all_music_tracks(db, is_available=True)
            print(f"✓ Found {len(tracks)} music tracks")
            for track in tracks[:5]:  # Show first 5
                print(f"  - {track.name} ({track.vibe}) - {track.bpm} BPM")

        # Test statistics
        print("\n8. Getting system statistics...")
        async with get_db_context() as db:
            stats = await get_video_statistics(db)
            print(f"✓ Statistics:")
            print(f"  - Total videos: {stats['total_videos']}")
            print(f"  - Total clips: {stats['total_clips']}")
            print(f"  - Total variations: {stats['total_variations']}")

        # Test list videos
        print("\n9. Listing all videos...")
        async with get_db_context() as db:
            all_videos = await get_videos(db, limit=10)
            print(f"✓ Found {len(all_videos)} videos")
            for vid in all_videos:
                print(f"  - {vid.filename} (status: {vid.status})")

        print("\n" + "=" * 60)
        print("✓ All tests passed!")
        print("=" * 60)
        print("\nDatabase integration is working correctly.")
        print("You can now start the API server with: uvicorn backend.main:app --reload")

        return True

    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        # Cleanup
        print("\n10. Shutting down...")
        await shutdown_event()
        print("✓ Database connections closed")


if __name__ == "__main__":
    print("\nStarting database integration test...")
    print("Make sure PostgreSQL is running and accessible!\n")

    result = asyncio.run(test_database_integration())
    sys.exit(0 if result else 1)
