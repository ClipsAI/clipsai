# Council Voting System - Quick Start Guide

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install anthropic openai google-generativeai
```

### 2. Set API Keys

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
export OPENAI_API_KEY="sk-..."
export GOOGLE_API_KEY="..."  # Optional
```

### 3. Use the Council

```python
from adlab.council import create_council_voter
from adlab.config import Config

# Initialize
config = Config()
council = create_council_voter(config)

# Vote on clips
clips = [
    (clip1, "What's the secret to going viral?"),
    (clip2, "This trick changed everything..."),
    # ... more clips
]

top_500 = council.vote_on_clips(clips, top_n=500)

# Results
for clip, vote in top_500:
    print(f"{clip.clip_id}: {vote.consensus_score:.2f}/10")
```

---

## 📊 Using the Hybrid Scorer (Recommended)

```python
from adlab.vvsa import create_hybrid_scorer
from adlab.config import Config

config = Config()
hybrid = create_hybrid_scorer(config)

# Two-stage scoring
selected = hybrid.score_and_vote(
    clips=clips_with_hooks,
    vvsa_threshold=6.0,      # VVSA minimum score
    council_top_n=500         # Final clip count
)

# Results include both scores
for clip, vvsa_score, council_vote in selected:
    print(f"VVSA: {vvsa_score.overall_score:.2f}")
    print(f"Council: {council_vote.consensus_score:.2f}")
```

---

## 🔧 Configuration

Edit `adlab/config.yaml`:

```yaml
council:
  enabled: true
  voting_method: weighted_average
  enable_parallel: true

  weights:
    claude_sonnet: 0.35    # Highest quality
    gpt4: 0.30             # Diverse perspective
    claude_haiku: 0.20     # Fast fallback
    gemini_flash: 0.10     # Additional view
    local_fallback: 0.05   # Always available

processing:
  max_clips: 500           # Council selects this many
```

---

## 🎯 Full Pipeline (Orchestrator)

```python
from clipfactory.processing.orchestrator import ClipFactoryOrchestrator

orchestrator = ClipFactoryOrchestrator(config={
    "anthropic_api_key": "sk-ant-...",
    "openai_api_key": "sk-...",
})

# Complete pipeline: transcribe → find clips → vote → select
clips = await orchestrator.phase1_council_deliberation(
    video_path="/path/to/video.mp4"
)

print(f"Selected {len(clips)} clips with scores 0-10")
```

---

## 🌐 API Usage

### Start the server:

```bash
cd clipfactory/backend
uvicorn main:app --host 0.0.0.0 --port 8000
```

### Upload video:

```bash
curl -X POST "http://localhost:8000/api/phase1/upload" \
  -F "video=@myvideo.mp4"
```

Response:
```json
{
  "video_id": "vid_abc123...",
  "filename": "myvideo.mp4",
  "size": 52428800,
  "status": "uploaded"
}
```

### Check status:

```bash
curl "http://localhost:8000/api/phase1/status/vid_abc123"
```

Response:
```json
{
  "video_id": "vid_abc123",
  "status": "processing",
  "clips_found": 127,
  "avg_hook_score": 7.2,
  "progress": 0.65
}
```

### Get clips:

```bash
curl "http://localhost:8000/api/phase1/clips/vid_abc123"
```

---

## 🔍 Voting Methods

### 1. Weighted Average (Default)

```python
council = CouncilVoter(voting_method="weighted_average")
```

Best for: Production use, leverages model quality differences

### 2. Simple Average

```python
council = CouncilVoter(voting_method="average")
```

Best for: Testing, equal model weight

### 3. Median

```python
council = CouncilVoter(voting_method="median")
```

Best for: Reducing outlier influence

### 4. Majority Vote

```python
council = CouncilVoter(voting_method="majority")
```

Best for: Binary decisions (pass/fail)

---

## 💡 Common Patterns

### Pattern 1: High Quality Only

```python
# Use Claude Sonnet + GPT-4 only
council = CouncilVoter(
    anthropic_api_key="...",
    openai_api_key="...",
    weights={
        "claude_sonnet": 0.55,
        "gpt4": 0.45
    },
    min_models=2
)
```

### Pattern 2: Cost-Optimized

```python
# VVSA filters 80%, council only scores 200 finalists
hybrid = create_hybrid_scorer(config)
selected = hybrid.score_and_vote(
    clips=all_clips,
    vvsa_threshold=7.0,   # Higher threshold = fewer clips
    council_top_n=200     # Fewer council calls
)
```

### Pattern 3: Fast Testing

```python
# Sequential, no cache, local only
council = CouncilVoter(
    enable_parallel=False,
    cache_enabled=False
)
# Only uses LocalFallbackModel - instant results
```

---

## 📈 Monitoring

### Get voting statistics:

```python
stats = council.get_voting_stats()
print(stats)
```

Output:
```python
{
    'num_models': 5,
    'active_models': ['claude_sonnet', 'gpt4', 'claude_haiku',
                     'gemini_flash', 'local_fallback'],
    'voting_method': 'weighted_average',
    'weights': {...},
    'cache_size': 127
}
```

### Clear cache:

```python
council.clear_cache()
```

---

## ⚠️ Troubleshooting

### Issue: "No module named 'anthropic'"

```bash
pip install anthropic
```

### Issue: All models failing

Check API keys:
```python
import os
print(os.getenv("ANTHROPIC_API_KEY"))  # Should not be None
```

### Issue: Slow performance

Enable parallel execution:
```python
council = CouncilVoter(enable_parallel=True)
```

### Issue: High API costs

Use VVSA pre-filtering:
```python
hybrid = create_hybrid_scorer(config)
# Only top VVSA clips go to council
```

---

## 📚 Key Files

- **Council Implementation**: `/home/user/clipsai/adlab/council.py`
- **Hybrid Scorer**: `/home/user/clipsai/adlab/vvsa.py`
- **Orchestrator**: `/home/user/clipsai/clipfactory/processing/orchestrator.py`
- **API Backend**: `/home/user/clipsai/clipfactory/backend/main.py`
- **Configuration**: `/home/user/clipsai/adlab/config.yaml`

---

## 🎓 Best Practices

1. **Use Hybrid Scorer** - Combines efficiency of VVSA with quality of Council
2. **Enable Parallel** - 5x speedup for multi-model voting
3. **Cache Responses** - Avoid re-scoring identical clips
4. **Set Min Models** - Require at least 2 models for valid votes
5. **Monitor Costs** - Track API usage, especially for large batches

---

## ✅ Verification Checklist

- [ ] API keys set in environment
- [ ] Dependencies installed (`anthropic`, `openai`)
- [ ] Config file exists (`adlab/config.yaml`)
- [ ] At least 2 models available (check logs)
- [ ] Test with sample clips before production

---

**Need Help?**

See full documentation in `COUNCIL_IMPLEMENTATION_REPORT.md`
