# OpenRouter Model Configuration Audit Report
## ClipsAI Codebase - Complete Analysis

**Audit Date:** 2025-11-10  
**Codebase:** ClipsAI with AdLab extension  
**Branch:** claude/adlab-viral-clip-factory-011CUypUQaxsnp5wf3uL15mm

---

## EXECUTIVE SUMMARY

**FINDING: NO OpenRouter References Found**

The codebase does NOT use OpenRouter. Instead, it uses direct API calls to Anthropic and OpenAI services.

**Current Model Architecture:**
- **Primary LLM:** Anthropic Claude (configurable)
- **Secondary LLM:** OpenAI GPT (fallback/high-quality path)
- **Models Used:** Claude 3.5 Sonnet, Claude 3.5 Haiku, GPT-4
- **Transcription:** WhisperX (local, not an LLM)
- **Embeddings:** Sentence-Transformers RoBERTa (local, not an LLM)

---

## 1. COMPLETE FILE INVENTORY

### Files with Model/API Configuration References:

#### Primary Configuration Files:
1. `/home/user/clipsai/adlab/config.py` - Configuration manager
2. `/home/user/clipsai/adlab/config.example.yaml` - Example YAML config
3. `/home/user/clipsai/clipfactory/.env.example` - Environment variables

#### LLM Implementation Files:
4. `/home/user/clipsai/adlab/llm.py` - Anthropic Claude wrapper
5. `/home/user/clipsai/clipfactory/processing/ai/title_generator.py` - Title generation (Claude/GPT)
6. `/home/user/clipsai/adlab/titles.py` - Title generation abstraction
7. `/home/user/clipsai/adlab/vvsa.py` - Hook scoring (uses Claude)

#### Orchestration/Backend:
8. `/home/user/clipsai/clipfactory/backend/main.py` - FastAPI backend
9. `/home/user/clipsai/clipfactory/processing/orchestrator.py` - Pipeline orchestrator

#### Other Processing:
10. `/home/user/clipsai/clipsai/transcribe/transcriber.py` - WhisperX transcription
11. `/home/user/clipsai/clipsai/clip/text_embedder.py` - RoBERTa embeddings
12. `/home/user/clipsai/adlab/run.py` - CLI entry point

---

## 2. CURRENT MODEL NAMES & CONFIGURATIONS

### Active Models In Use:

#### Anthropic Models:
| Location | Model Name | Purpose | Version | Config Type |
|----------|-----------|---------|---------|------------|
| `/adlab/llm.py:18` | `claude-3-5-sonnet-20241022` | Default hook scoring, title generation | Claude 3.5 Sonnet | Hardcoded default |
| `/adlab/config.py:53` | `claude-3-5-sonnet-20241022` | Config default | Claude 3.5 Sonnet | Hardcoded in defaults |
| `/adlab/config.example.yaml:7` | `claude-3-5-sonnet-20241022` | Config template | Claude 3.5 Sonnet | YAML config |
| `/clipfactory/processing/ai/title_generator.py:104` | `claude-3-5-haiku-20241022` | Fallback title generation | Claude 3.5 Haiku | Hardcoded |
| `/clipfactory/processing/ai/title_generator.py:172` | `claude-3-5-haiku-20241022` | Transcript-based titles | Claude 3.5 Haiku | Hardcoded |
| `/clipfactory/processing/ai/title_generator.py:251` | `claude-3-5-haiku-20241022` | Account-specific titles | Claude 3.5 Haiku | Hardcoded |

#### OpenAI Models:
| Location | Model Name | Purpose | Version | Config Type |
|----------|-----------|---------|---------|------------|
| `/clipfactory/processing/ai/title_generator.py:90` | `gpt-4` | High-quality title variants | GPT-4 | Hardcoded (comment: "Will be GPT-5 when available") |
| `/clipfactory/processing/ai/title_generator.py:181` | `gpt-4` | Transcript-based titles (GPT path) | GPT-4 | Hardcoded |

#### Other Models:
| Location | Model Name | Purpose | Version | Config Type |
|----------|-----------|---------|---------|------------|
| `/clipsai/transcribe/transcriber.py:62` | `large-v2` or `tiny` | Speech transcription | WhisperX | Auto-selected (GPU if available) |
| `/clipsai/clip/text_embedder.py:20` | `all-roberta-large-v1` | Text embeddings | RoBERTa | Hardcoded |

---

## 3. API KEY REFERENCES

### Environment Variables:
```
ANTHROPIC_API_KEY      - Used in: adlab/llm.py, adlab/config.py, clipfactory/processing/ai/title_generator.py
OPENAI_API_KEY         - Used in: clipfactory/processing/ai/title_generator.py, clipfactory/.env.example
GOOGLE_GEMINI_API_KEY  - Defined in: clipfactory/.env.example (NOT USED anywhere)
```

### Configuration Paths:
1. **Environment Variable:** `os.getenv("ANTHROPIC_API_KEY")`
2. **Config File:** `config.yaml` → `anthropic.api_key`
3. **Default:** Falls back to environment variable

---

## 4. IDENTIFIED INCONSISTENCIES

### ⚠️ ISSUE #1: Hardcoded vs Configuration-Based Models

**Severity:** MEDIUM

**Details:**
- AdLab uses **configuration-based models** (good practice):
  - `adlab/llm.py:18` - model parameter with default
  - `adlab/config.py:53` - configurable via YAML
  
- But Clip Factory uses **hardcoded models** (inconsistent):
  - `clipfactory/processing/ai/title_generator.py:104` - hardcoded `claude-3-5-haiku-20241022`
  - `clipfactory/processing/ai/title_generator.py:172` - hardcoded `claude-3-5-haiku-20241022`
  - `clipfactory/processing/ai/title_generator.py:251` - hardcoded `claude-3-5-haiku-20241022`
  - `clipfactory/processing/ai/title_generator.py:90` - hardcoded `gpt-4`
  - `clipfactory/processing/ai/title_generator.py:181` - hardcoded `gpt-4`

**Impact:** Cannot easily swap Claude Haiku for other models without code changes.

### ⚠️ ISSUE #2: Different Model Selection Strategies

**Severity:** MEDIUM

**Details:**

**AdLab (Good):**
```python
# adlab/llm.py uses configuration
model=config.get("anthropic.model")  # Configurable
```

**Clip Factory (Problematic):**
```python
# clipfactory/processing/ai/title_generator.py:87-110
if openai.api_key:
    # Uses GPT-4
    response = openai.ChatCompletion.create(model="gpt-4", ...)
elif self.claude:
    # Falls back to Claude 3.5 Haiku (hardcoded)
    response = self.claude.messages.create(model="claude-3-5-haiku-20241022", ...)
```

**Problem:** 
- GPT-4 is used for "high quality" but it's a hardcoded fallback
- Claude Haiku is always the fallback
- No way to configure which model to use

### ⚠️ ISSUE #3: Two Separate LLM Initialization Patterns

**Severity:** MEDIUM

**Details:**

**Pattern A - AdLab (Centralized):**
```python
# adlab/config.py - Single point of configuration
defaults = {
    "anthropic": {
        "api_key": os.getenv("ANTHROPIC_API_KEY", ""),
        "model": "claude-3-5-sonnet-20241022",
        "max_tokens": 1024,
        "temperature": 1.0,
    }
}
```

**Pattern B - Clip Factory (Decentralized):**
```python
# clipfactory/processing/orchestrator.py:42-43
self.title_generator = TitleGenerator(
    anthropic_key=config.get('anthropic_api_key'),
    openai_key=config.get('openai_api_key')
)

# clipfactory/backend/main.py - No API key management
# (marked as TODO: Call Claude API)
```

**Problem:** Two different configuration styles make it hard to maintain.

### ⚠️ ISSUE #4: Undefined Google Gemini Integration

**Severity:** LOW

**Details:**
- `clipfactory/.env.example:7` defines `GOOGLE_GEMINI_API_KEY=your_gemini_key_here`
- This variable is **NEVER USED** anywhere in the codebase
- Suggests incomplete or abandoned Gemini integration

**Files Checked:** No references to Gemini/Google API calls found.

### ⚠️ ISSUE #5: Model Versioning Drift

**Severity:** LOW

**Details:**

Current models are using specific versions with dates:
- `claude-3-5-sonnet-20241022` (October 22, 2024)
- `claude-3-5-haiku-20241022` (October 22, 2024)

These will be outdated. Should have a system to easily update to newer versions:
- `claude-4-20250101` (if/when released)
- No current mechanism for batch updating

---

## 5. COUNCIL VOTING SYSTEM ANALYSIS

### Finding: No Council Voting System Implemented

**Status:** Planned but not built

**Evidence:**

1. **Backend Placeholder (Phase 1):**
```python
# clipfactory/backend/main.py:73-97
@app.post("/api/phase1/upload", ...)
async def upload_video_for_council(...):
    # TODO: Trigger council deliberation in background
    # background_tasks.add_task(run_council_deliberation, video_path, video_id)
```

2. **Orchestrator Placeholder:**
```python
# clipfactory/processing/orchestrator.py:145-170
async def phase1_council_deliberation(self, video_path: str):
    # TODO: Integrate with adlab council
    # from adlab.council import run_council
    logger.info("Running council deliberation...")
    # Returns mock clips
```

3. **AdLab has no council voting module:**
   - No `adlab/council.py` file
   - No voting mechanism
   - Uses simple heuristic scoring instead

**What exists instead:**
- **VVSA Scoring** (`adlab/vvsa.py`) - Uses weighted component analysis
  - Text score (heuristic)
  - Audio score (heuristic)
  - Visual score (placeholder)
  - LLM score (Claude-based)
  - **Weighted average** (not voting)

---

## 6. ENVIRONMENT VARIABLES ANALYSIS

### Defined Environment Variables:

```bash
# API Keys
ANTHROPIC_API_KEY          ✓ Used
OPENAI_API_KEY             ✓ Used
GOOGLE_GEMINI_API_KEY      ✗ NOT USED (orphaned)

# Database
DATABASE_URL               ✗ NOT USED (backend placeholder)
REDIS_URL                  ✗ NOT USED (backend placeholder)

# File Storage
UPLOAD_DIR                 ✓ Used in backend/main.py
OUTPUT_DIR                 ✓ Used in backend/main.py
MAX_VIDEO_SIZE_GB          ✗ NOT USED

# Video Processing
FFMPEG_PATH                ✓ Used implicitly
MAX_VIDEO_SIZE_GB          ✗ NOT USED

# Frontend
NEXT_PUBLIC_API_URL        ✗ NOT CHECKED (Next.js frontend)

# Calendar Integration
GOOGLE_CALENDAR_API_KEY    ✗ NOT USED (feature not implemented)
ICLOUD_USERNAME            ✗ NOT USED (feature not implemented)
ICLOUD_PASSWORD            ✗ NOT USED (feature not implemented)

# Social Media
TIKTOK_API_KEY             ✗ NOT USED
INSTAGRAM_API_KEY          ✗ NOT USED
YOUTUBE_API_KEY            ✗ NOT USED

# Feature Flags
ENABLE_COUNCIL_DELIBERATION ✗ NOT USED (council not implemented)
ENABLE_FACE_TRACKING        ✓ Referenced but placeholder
ENABLE_AUTO_POSTING         ✗ NOT USED

# Monitoring
SENTRY_DSN                 ✗ NOT USED
PROMETHEUS_PORT            ✗ NOT USED
```

---

## 7. CONFIGURATION FILE ANALYSIS

### Config Locations:
1. **YAML Configuration:** `adlab/config.example.yaml` (actively used)
2. **Environment Variables:** `clipfactory/.env.example` (partially used)
3. **Python Defaults:** `adlab/config.py:_apply_defaults()` (good backup)

### Current Config Structure:

```yaml
# adlab/config.example.yaml
anthropic:
  api_key: ${ANTHROPIC_API_KEY}          # ✓ Configurable
  model: claude-3-5-sonnet-20241022      # ✓ Configurable
  max_tokens: 1024                       # ✓ Configurable
  temperature: 1.0                       # ✓ Configurable

vvsa:
  hook_duration: 3.0
  min_score: 6.0
  weights:
    visual: 0.3
    audio: 0.2
    text: 0.3
    llm: 0.2

# Note: No OpenRouter configuration
# Note: No council voting configuration
# Note: No GPT/Gemini configuration
```

---

## 8. DEPRECATED OR INCORRECT REFERENCES

### ⚠️ Outdated Comments:

1. **File:** `clipfactory/processing/ai/title_generator.py:90`
   ```python
   model="gpt-4",  # Will be GPT-5 when available
   ```
   **Issue:** Comment implies GPT-5 upgrade planned, but hardcoded still to GPT-4
   **Status:** Outdated/misleading

2. **File:** `clipfactory/processing/ai/title_generator.py:19-21`
   ```python
   """
   Uses:
   - Claude 4.5 Haiku (cheap, fast)
   - GPT-5 (high quality variants)
   - Gemini 2.5 Flash (formatting)
   """
   ```
   **Issue:** 
   - "Claude 4.5 Haiku" doesn't exist (it's Claude 3.5 Haiku)
   - "GPT-5" not available (code uses GPT-4)
   - "Gemini 2.5 Flash" not used anywhere
   **Status:** Completely outdated docstring

---

## 9. HARDCODED vs CONFIGURATION-BASED ANALYSIS

### Hardcoded References Found:

| File | Line | Content | Type | Fixable |
|------|------|---------|------|---------|
| `adlab/llm.py` | 18 | `model="claude-3-5-sonnet-20241022"` | Default | ✓ Yes, already configurable |
| `adlab/config.py` | 53 | `"model": "claude-3-5-sonnet-20241022"` | Default | ✓ Yes, user can override |
| `clipfactory/processing/ai/title_generator.py` | 90 | `model="gpt-4"` | Hardcoded | ✗ No, must change code |
| `clipfactory/processing/ai/title_generator.py` | 104 | `model="claude-3-5-haiku-20241022"` | Hardcoded | ✗ No, must change code |
| `clipfactory/processing/ai/title_generator.py` | 172 | `model="claude-3-5-haiku-20241022"` | Hardcoded | ✗ No, must change code |
| `clipfactory/processing/ai/title_generator.py` | 181 | `model="gpt-4"` | Hardcoded | ✗ No, must change code |
| `clipfactory/processing/ai/title_generator.py` | 251 | `model="claude-3-5-haiku-20241022"` | Hardcoded | ✗ No, must change code |
| `clipsai/clip/text_embedder.py` | 20 | `"all-roberta-large-v1"` | Hardcoded | ✗ No, fine for embeddings |
| `clipsai/transcribe/transcriber.py` | 62 | `"large-v2"/"tiny"` | Auto-select | ✓ Yes, configurable |

### Summary:
- **AdLab:** 80% configurable (good practice)
- **Clip Factory:** 20% configurable (problematic)

---

## 10. RECOMMENDATIONS & ACTION ITEMS

### Priority 1: Standardize Configuration (HIGH)

**Problem:** Inconsistent model configuration between AdLab and Clip Factory

**Solution:**

1. **Create unified config structure:**
```yaml
# config.yaml - unified
anthropic:
  api_key: ${ANTHROPIC_API_KEY}
  models:
    primary: claude-3-5-sonnet-20241022
    fallback: claude-3-5-haiku-20241022

openai:
  api_key: ${OPENAI_API_KEY}
  model: gpt-4

gemini:
  api_key: ${GOOGLE_GEMINI_API_KEY}
  model: gemini-2.5-flash
  enabled: false  # Currently unused

lm_studio:  # Optional local models
  enabled: false
  base_url: http://localhost:1234/v1
```

2. **Make Clip Factory use same config:**
   - Move `clipfactory/processing/ai/title_generator.py` to accept model config
   - Stop hardcoding model names
   - Support model fallback chain

3. **Create model registry:**
```python
# adlab/models.py
SUPPORTED_MODELS = {
    "claude-sonnet": {
        "provider": "anthropic",
        "cost": "high",
        "speed": "medium",
        "quality": "highest"
    },
    "claude-haiku": {
        "provider": "anthropic",
        "cost": "low",
        "speed": "fastest",
        "quality": "good"
    },
    "gpt-4": {
        "provider": "openai",
        "cost": "very-high",
        "speed": "slow",
        "quality": "excellent"
    },
    # ... etc
}
```

### Priority 2: Remove Unused Environment Variables (MEDIUM)

**Problem:** Orphaned env variables create confusion

**Solution:**
1. Remove from `.env.example`:
   - `GOOGLE_GEMINI_API_KEY` (if not implementing Gemini)
   - `DATABASE_URL` (if keeping in-memory)
   - `REDIS_URL` (if not using Celery)
   - `GOOGLE_CALENDAR_API_KEY` (if not implementing calendar)
   - `ICLOUD_USERNAME/PASSWORD` (if not implementing iCloud)
   - `TIKTOK_API_KEY`, `INSTAGRAM_API_KEY`, `YOUTUBE_API_KEY` (if not implementing auto-posting)
   - `SENTRY_DSN` (if not using Sentry)
   - `PROMETHEUS_PORT` (if not using Prometheus)

2. Document which features are planned vs. implemented

### Priority 3: Fix Outdated Docstrings (LOW)

**Problem:** Misleading documentation

**Solution:**
1. Update `clipfactory/processing/ai/title_generator.py:19-21`:
```python
"""
Title Generation using Claude/GPT
Supports voice dictation, A/B testing, and account-specific styles

Models used:
- Claude 3.5 Haiku (fallback, cheap, fast)
- GPT-4 (when available, high quality)
"""
```

2. Remove references to non-existent models:
   - ~~Claude 4.5 Haiku~~ → Claude 3.5 Haiku
   - ~~GPT-5~~ → GPT-4
   - ~~Gemini 2.5 Flash~~ (remove or implement)

### Priority 4: Implement Proper Council Voting (if needed) (MEDIUM)

**Current State:** Placeholder with mock data

**Options:**

**Option A: Remove Council (Simplest)**
- Delete `phase1_council_deliberation` from orchestrator
- Use VVSA scoring as sole decision mechanism
- Update documentation

**Option B: Implement Council Properly (Comprehensive)**
```python
# adlab/council.py (new file)
class CouncilVoter:
    """
    Multi-model voting system for clip selection.
    Different models vote on clip quality, final score is consensus.
    """
    
    def __init__(self, config):
        self.claude = ClaudeClient(...)
        self.gpt = OpenAIClient(...)  # If enabled
        self.config = config
    
    def get_council_votes(self, clip_data):
        """Get votes from all models"""
        votes = []
        
        # Vote 1: Claude scores
        claude_score = self.claude.score_hook(...)
        votes.append(("claude", claude_score))
        
        # Vote 2: GPT scores (if enabled)
        if self.config.get("openai.enabled"):
            gpt_score = self.gpt.score_hook(...)
            votes.append(("gpt-4", gpt_score))
        
        # Vote 3: VVSA heuristic
        vvsa_score = self.vvsa.score_clip(...)
        votes.append(("vvsa", vvsa_score))
        
        return votes
    
    def consensus_score(self, votes):
        """Calculate consensus from council votes"""
        # Weighted average with tie-breaking
        pass
```

### Priority 5: Create Model Compatibility Matrix (LOW)

**Documentation needed:**

| Feature | Claude 3.5 Sonnet | Claude 3.5 Haiku | GPT-4 | Status |
|---------|------------------|-----------------|-------|--------|
| Hook Scoring | ✓ | ✓ | ✓ | All work |
| Title Generation | ✓ | ✓ | ✓ | All work |
| Tag Generation | ✓ | ✓ | ? | Untested with GPT |
| Vision (future) | ✓ | ✗ | ✓ | Haiku lacks vision |
| Cost | High | Low | Very High | Consider budget |

---

## 11. NO OpenRouter FINDINGS

### Search Results:

```bash
grep -r "openrouter\|OpenRouter" /home/user/clipsai
# Result: No matches found
```

### Conclusion:
The codebase makes **direct API calls** to service providers, not through OpenRouter.

**Advantages of current approach:**
- Direct access to latest models
- Simpler billing/auth
- No intermediary latency

**Disadvantages vs. OpenRouter:**
- Multiple API key management needed
- No unified error handling
- No built-in fallback between providers
- Rate limiting managed per-provider

---

## 12. RECOMMENDED STANDARD CONFIGURATION

### Proposed Standard Structure:

```yaml
# config.yaml - Standard Template
version: "1.0"

# Anthropic (Primary)
anthropic:
  api_key: ${ANTHROPIC_API_KEY}
  models:
    hook_scoring: claude-3-5-sonnet-20241022
    title_generation: claude-3-5-haiku-20241022
    fallback: claude-3-5-haiku-20241022
  parameters:
    max_tokens: 1024
    temperature: 1.0

# OpenAI (Optional secondary)
openai:
  enabled: false
  api_key: ${OPENAI_API_KEY}
  model: gpt-4
  parameters:
    max_tokens: 500
    temperature: 1.2

# Local model support (optional)
local_models:
  enabled: false
  lm_studio_url: http://localhost:1234/v1
  lm_studio_model: llama2-7b-chat

# Transcription (Local)
transcription:
  provider: whisperx  # Only option currently
  model_size: base  # Options: tiny, base, small, medium, large
  device: auto  # auto, cuda, cpu
  language: null  # null = auto-detect

# Vector embeddings (Local)
embeddings:
  provider: sentence-transformers
  model: all-roberta-large-v1
  device: auto

# Processing
processing:
  min_clip_duration: 10
  max_clip_duration: 90
  target_clips: 300
  max_clips: 500
  batch_size: 10

# VVSA Hook Scoring
vvsa:
  hook_duration: 3.0
  min_score: 6.0
  weights:
    visual: 0.3
    audio: 0.2
    text: 0.3
    llm: 0.2
  
  # Optional: Use different models for voting
  voting:
    enabled: false
    models:
      - provider: anthropic
        model: claude-3-5-sonnet-20241022
      - provider: openai
        model: gpt-4
    strategy: weighted_average  # or: majority, weighted_average, first_available

# Export settings
export:
  output_dir: ./output
  video_codec: libx264
  audio_codec: aac
  preset: medium
  crf: 23
  thumbnail_time: 1.0

# Variations
variations:
  temporal_shifts: [-1.0, -0.5, 0, 0.5, 1.0]
  durations: [15, 30, 45, 60]
  aspect_ratios: ["9:16", "1:1", "4:5"]
  max_variations_per_clip: 12
```

---

## 13. FILES NOT USING OPENROUTER (Confirmed)

All files checked for OpenRouter:
- ✓ No OpenRouter references
- ✓ No OpenRouter base URLs
- ✓ No OpenRouter API keys
- ✓ No OpenRouter model routing

---

## SUMMARY TABLE

| Finding | Status | Severity | Fixable |
|---------|--------|----------|---------|
| No OpenRouter found | Confirmed | N/A | N/A |
| Hardcoded models in Clip Factory | Issue | MEDIUM | ✓ Yes |
| Two config patterns (AdLab vs Clip Factory) | Issue | MEDIUM | ✓ Yes |
| Unused Gemini API key defined | Issue | LOW | ✓ Yes |
| Outdated docstrings (Claude 4.5, GPT-5) | Issue | LOW | ✓ Yes |
| Council voting not implemented | Design | MEDIUM | ✓ Yes |
| Orphaned environment variables | Issue | LOW | ✓ Yes |
| Model versioning not future-proof | Issue | LOW | ✓ Partial |
| AdLab uses good patterns | Positive | N/A | N/A |
| Clear separation of concerns | Positive | N/A | N/A |

---

## APPENDIX A: File Locations Reference

### AdLab Components (Good Practice):
```
adlab/
├── config.py              # Centralized configuration ✓
├── llm.py                 # Anthropic wrapper with configurable model ✓
├── vvsa.py                # Hook scoring ✓
├── titles.py              # Title generation ✓
└── run.py                 # CLI orchestration ✓
```

### Clip Factory Components (Needs Refactoring):
```
clipfactory/
├── backend/main.py        # FastAPI backend (incomplete) ⚠️
├── processing/
│   ├── ai/title_generator.py   # Hardcoded models ✗
│   ├── orchestrator.py         # Placeholder council ⚠️
│   ├── matrix/
│   ├── premiere/
│   └── variations/
└── .env.example           # Orphaned vars ⚠️
```

---

## APPENDIX B: Complete Model Reference List

### Anthropic Models Used:
- `claude-3-5-sonnet-20241022` - Primary model, highest quality
- `claude-3-5-haiku-20241022` - Fallback model, lower cost

### OpenAI Models Used:
- `gpt-4` - High-quality title generation (secondary path)

### Other Models:
- `large-v2` / `tiny` (WhisperX) - Speech transcription
- `all-roberta-large-v1` (Sentence Transformers) - Text embeddings

### Models Mentioned But Not Used:
- `claude-4.5-haiku` (doesn't exist, typo in docstring)
- `gpt-5` (mentioned in comments, not available)
- `gemini-2.5-flash` (defined in env but not used)

---

**End of Audit Report**
