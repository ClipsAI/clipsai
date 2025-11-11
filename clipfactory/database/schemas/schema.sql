-- Clip Factory Database Schema
-- PostgreSQL 15+

-- Videos uploaded for processing
CREATE TABLE videos (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    filename TEXT NOT NULL,
    file_path TEXT NOT NULL,
    file_size BIGINT,
    duration FLOAT,
    resolution TEXT,
    fps FLOAT,
    uploaded_at TIMESTAMP DEFAULT NOW(),
    status TEXT DEFAULT 'uploaded',
    metadata JSONB
);

-- Clips found by council
CREATE TABLE clips (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    video_id UUID REFERENCES videos(id) ON DELETE CASCADE,
    start_time FLOAT NOT NULL,
    end_time FLOAT NOT NULL,
    duration FLOAT GENERATED ALWAYS AS (end_time - start_time) STORED,
    transcript TEXT,
    hook_score FLOAT,
    hook_score_data JSONB,
    created_at TIMESTAMP DEFAULT NOW(),
    status TEXT DEFAULT 'pending', -- pending, approved, rejected
    metadata JSONB
);

-- Variations generated from clips
CREATE TABLE variations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    clip_id UUID REFERENCES clips(id) ON DELETE CASCADE,
    variation_type TEXT NOT NULL, -- temporal type: base, +4s, +35s
    reframe_style TEXT NOT NULL, -- original, flipped, blurry_bg
    title_style TEXT, -- TT3, AdLab
    start_time FLOAT NOT NULL,
    end_time FLOAT NOT NULL,
    duration FLOAT,
    frame_offset FLOAT,
    music_track_id UUID,
    video_path TEXT,
    thumbnail_path TEXT,
    captions_path TEXT,
    created_at TIMESTAMP DEFAULT NOW(),
    metadata JSONB
);

-- Title variants for A/B testing
CREATE TABLE title_variants (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    variation_id UUID REFERENCES variations(id) ON DELETE CASCADE,
    variant_id TEXT, -- A, B, C, D, E
    title_text TEXT NOT NULL,
    hook_style TEXT,
    target_audience TEXT,
    predicted_ctr FLOAT,
    tags TEXT[],
    created_at TIMESTAMP DEFAULT NOW(),
    metadata JSONB
);

-- Music tracks (40 total)
CREATE TABLE music_tracks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT NOT NULL,
    file_path TEXT NOT NULL,
    vibe TEXT,
    context_description TEXT,
    color TEXT, -- for UI
    bpm INTEGER,
    duration FLOAT,
    is_available BOOLEAN DEFAULT true,
    times_used INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT NOW(),
    metadata JSONB
);

-- Accounts for posting
CREATE TABLE accounts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    platform TEXT NOT NULL, -- tiktok, instagram, youtube
    username TEXT NOT NULL,
    account_type TEXT NOT NULL, -- fan, brand, watermark
    posting_strategy TEXT, -- hashtag pattern, posting frequency, etc.
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP DEFAULT NOW(),
    metadata JSONB
);

-- Posted clips tracking
CREATE TABLE posts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    variation_id UUID REFERENCES variations(id),
    account_id UUID REFERENCES accounts(id),
    title_variant_id UUID REFERENCES title_variants(id),
    post_url TEXT,
    posted_at TIMESTAMP,
    scheduled_for TIMESTAMP,
    status TEXT DEFAULT 'scheduled', -- scheduled, posted, failed
    performance_data JSONB, -- views, likes, comments, etc.
    created_at TIMESTAMP DEFAULT NOW(),
    metadata JSONB
);

-- Calendar events for posting schedule
CREATE TABLE calendar_events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    post_id UUID REFERENCES posts(id),
    event_start TIMESTAMP NOT NULL,
    event_end TIMESTAMP,
    title TEXT,
    description TEXT,
    calendar_service TEXT, -- google, icloud
    external_event_id TEXT,
    created_at TIMESTAMP DEFAULT NOW(),
    metadata JSONB
);

-- Database tracking (A/B testing, performance)
CREATE TABLE performance_tracking (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    post_id UUID REFERENCES posts(id),
    tracked_at TIMESTAMP DEFAULT NOW(),
    views INTEGER,
    likes INTEGER,
    comments INTEGER,
    shares INTEGER,
    avg_view_duration FLOAT,
    completion_rate FLOAT,
    engagement_rate FLOAT,
    metadata JSONB
);

-- Indexes for performance
CREATE INDEX idx_videos_status ON videos(status);
CREATE INDEX idx_clips_video_id ON clips(video_id);
CREATE INDEX idx_clips_hook_score ON clips(hook_score DESC);
CREATE INDEX idx_variations_clip_id ON variations(clip_id);
CREATE INDEX idx_posts_account_id ON posts(account_id);
CREATE INDEX idx_posts_scheduled_for ON posts(scheduled_for);
CREATE INDEX idx_performance_post_id ON performance_tracking(post_id);

-- Functions for analytics
CREATE OR REPLACE FUNCTION get_top_performing_variations(limit_count INT DEFAULT 10)
RETURNS TABLE (
    variation_id UUID,
    avg_views FLOAT,
    avg_engagement FLOAT,
    total_posts BIGINT
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        v.id as variation_id,
        AVG(pt.views)::FLOAT as avg_views,
        AVG(pt.engagement_rate)::FLOAT as avg_engagement,
        COUNT(p.id) as total_posts
    FROM variations v
    JOIN posts p ON p.variation_id = v.id
    JOIN performance_tracking pt ON pt.post_id = p.id
    GROUP BY v.id
    ORDER BY avg_engagement DESC
    LIMIT limit_count;
END;
$$ LANGUAGE plpgsql;

-- Function to get account performance
CREATE OR REPLACE FUNCTION get_account_performance(account_uuid UUID)
RETURNS TABLE (
    total_posts BIGINT,
    avg_views FLOAT,
    avg_engagement FLOAT,
    best_posting_time TIME
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        COUNT(p.id) as total_posts,
        AVG(pt.views)::FLOAT as avg_views,
        AVG(pt.engagement_rate)::FLOAT as avg_engagement,
        (
            SELECT EXTRACT(HOUR FROM posted_at)::TIME
            FROM posts p2
            JOIN performance_tracking pt2 ON pt2.post_id = p2.id
            WHERE p2.account_id = account_uuid
            GROUP BY EXTRACT(HOUR FROM posted_at)
            ORDER BY AVG(pt2.engagement_rate) DESC
            LIMIT 1
        ) as best_posting_time
    FROM posts p
    JOIN performance_tracking pt ON pt.post_id = p.id
    WHERE p.account_id = account_uuid;
END;
$$ LANGUAGE plpgsql;

-- Seed data for music tracks (placeholders)
INSERT INTO music_tracks (name, file_path, vibe, context_description, color) VALUES
('Energetic Beat 1', '/music/energetic_1.mp3', 'High energy', 'Use for intense training moments', '#FF5722'),
('Chill Vibes 1', '/music/chill_1.mp3', 'Relaxed', 'Use for recovery or cool-down', '#4CAF50'),
('Motivational 1', '/music/motivational_1.mp3', 'Inspiring', 'Use for personal bests or achievements', '#2196F3'),
('Dramatic 1', '/music/dramatic_1.mp3', 'Tense', 'Use for challenges or competitions', '#9C27B0'),
('Uplifting 1', '/music/uplifting_1.mp3', 'Positive', 'Use for success moments', '#FFC107');
-- Add 35 more to reach 40 total

-- Comments
COMMENT ON TABLE videos IS 'Source videos uploaded for processing';
COMMENT ON TABLE clips IS 'Clips identified by AI council from source videos';
COMMENT ON TABLE variations IS 'All variations generated from base clips (3 temporal × 3 reframe = 9 per clip)';
COMMENT ON TABLE title_variants IS 'A/B testing title variants for each variation';
COMMENT ON TABLE music_tracks IS '40 music tracks for swapping';
COMMENT ON TABLE accounts IS 'Social media accounts for multi-account posting';
COMMENT ON TABLE posts IS 'Posted or scheduled clips';
COMMENT ON TABLE performance_tracking IS 'Performance metrics for posted clips';
