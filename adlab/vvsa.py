"""
VVSA (Viral Video Success Analysis) Hook Scoring System.

Combines heuristic analysis with LLM-based evaluation to score the first 3 seconds
of video clips for viral potential.
"""
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

from clipsai import Transcription

logger = logging.getLogger(__name__)


@dataclass
class HookScore:
    """Container for hook scoring results."""
    overall_score: float  # 0-10
    visual_score: float  # 0-10
    audio_score: float  # 0-10
    text_score: float  # 0-10
    llm_score: float  # 0-10
    reasoning: str
    strengths: List[str]
    improvements: List[str]
    metadata: Dict[str, Any]


class VVSAScorer:
    """
    VVSA-based hook scoring system.
    Evaluates the first 3 seconds of clips for viral potential.
    """

    def __init__(self, llm_client=None, hook_duration: float = 3.0,
                 weights: Optional[Dict[str, float]] = None):
        """
        Initialize VVSA scorer.

        Parameters
        ----------
        llm_client : ClaudeClient, optional
            LLM client for advanced scoring
        hook_duration : float
            Duration of hook to analyze (seconds)
        weights : dict, optional
            Weights for scoring components: visual, audio, text, llm
        """
        self.llm_client = llm_client
        self.hook_duration = hook_duration

        # Default weights
        self.weights = weights or {
            "visual": 0.3,
            "audio": 0.2,
            "text": 0.3,
            "llm": 0.2,
        }

        # Normalize weights
        total = sum(self.weights.values())
        self.weights = {k: v / total for k, v in self.weights.items()}

    def score_clip(self, transcription: Transcription,
                   video_path: Optional[str] = None,
                   start_time: float = 0.0) -> HookScore:
        """
        Score a clip's hook (first 3 seconds).

        Parameters
        ----------
        transcription : Transcription
            Full transcription of the clip
        video_path : str, optional
            Path to video file for visual analysis
        start_time : float
            Start time of the clip in the source video

        Returns
        -------
        HookScore
            Complete hook scoring results
        """
        # Extract hook transcript (first 3 seconds)
        hook_text = self._extract_hook_text(transcription, start_time)

        # Score components
        text_score = self._score_text_hook(hook_text)
        visual_score = self._score_visual_hook(video_path, start_time) if video_path else 5.0
        audio_score = self._score_audio_hook(hook_text)

        # LLM-based scoring
        llm_score = 5.0
        llm_reasoning = "LLM scoring not available"
        strengths = []
        improvements = []

        if self.llm_client and hook_text:
            try:
                llm_result = self.llm_client.score_hook(hook_text)
                llm_score = llm_result.get("score", 5.0)
                llm_reasoning = llm_result.get("reasoning", "")
                strengths = llm_result.get("strengths", [])
                improvements = llm_result.get("improvements", [])
            except Exception as e:
                logger.warning(f"LLM scoring failed: {e}")

        # Calculate weighted overall score
        overall_score = (
            self.weights["visual"] * visual_score +
            self.weights["audio"] * audio_score +
            self.weights["text"] * text_score +
            self.weights["llm"] * llm_score
        )

        return HookScore(
            overall_score=round(overall_score, 2),
            visual_score=round(visual_score, 2),
            audio_score=round(audio_score, 2),
            text_score=round(text_score, 2),
            llm_score=round(llm_score, 2),
            reasoning=llm_reasoning,
            strengths=strengths,
            improvements=improvements,
            metadata={
                "hook_text": hook_text,
                "hook_duration": self.hook_duration,
                "weights": self.weights
            }
        )

    def _extract_hook_text(self, transcription: Transcription,
                          start_time: float) -> str:
        """Extract transcript text for the hook period."""
        hook_end = start_time + self.hook_duration

        words = []
        char_info = transcription.get_char_info()

        current_word = []
        for char_data in char_info:
            char_start = char_data.get("start_time")
            char_text = char_data.get("char", "")

            # Skip if before our start time or after hook end
            if char_start is None:
                current_word.append(char_text)
                continue

            if char_start < start_time:
                continue
            if char_start >= hook_end:
                break

            # Build words
            if char_text == " ":
                if current_word:
                    words.append("".join(current_word))
                    current_word = []
            else:
                current_word.append(char_text)

        # Add final word
        if current_word:
            words.append("".join(current_word))

        return " ".join(words)

    def _score_text_hook(self, text: str) -> float:
        """
        Score hook text using heuristics.

        Checks for:
        - Curiosity keywords
        - Questions
        - Strong verbs
        - Emotional triggers
        - Length optimization
        """
        if not text:
            return 0.0

        score = 5.0  # Base score

        text_lower = text.lower()
        words = text.split()

        # Curiosity keywords (+1.5)
        curiosity_words = [
            "secret", "hack", "trick", "revealed", "hidden", "shocking",
            "never", "always", "why", "how", "what", "discover", "truth"
        ]
        if any(word in text_lower for word in curiosity_words):
            score += 1.5

        # Question format (+1.0)
        if "?" in text:
            score += 1.0

        # Strong action verbs (+0.5)
        action_verbs = [
            "learn", "discover", "master", "unlock", "reveal", "transform",
            "create", "build", "achieve", "stop", "start", "avoid"
        ]
        if any(verb in text_lower for verb in action_verbs):
            score += 0.5

        # Emotional triggers (+1.0)
        emotions = [
            "amazing", "incredible", "insane", "crazy", "mind-blowing",
            "unbelievable", "shocking", "terrifying", "hilarious", "genius"
        ]
        if any(emo in text_lower for emo in emotions):
            score += 1.0

        # Numbers/specificity (+0.5)
        if any(char.isdigit() for char in text):
            score += 0.5

        # Length penalties/bonuses
        word_count = len(words)
        if word_count < 3:
            score -= 1.5  # Too short
        elif word_count > 20:
            score -= 1.0  # Too long
        elif 5 <= word_count <= 12:
            score += 0.5  # Optimal length

        # Caps lock check (penalty)
        if text.isupper() and len(text) > 10:
            score -= 0.5

        return max(0.0, min(10.0, score))

    def _score_audio_hook(self, text: str) -> float:
        """
        Score audio/pacing based on text.

        Since we don't have audio analysis, use text as proxy:
        - Word density
        - Exclamation marks
        - Question marks
        """
        if not text:
            return 5.0

        score = 5.0
        words = text.split()

        # Word density (words per second)
        words_per_sec = len(words) / self.hook_duration
        if 2.0 <= words_per_sec <= 4.0:
            score += 1.0  # Good pacing
        elif words_per_sec > 5.0:
            score -= 1.0  # Too fast

        # Exclamation marks (energy)
        if "!" in text:
            score += 0.5

        # Questions (engagement)
        if "?" in text:
            score += 0.5

        return max(0.0, min(10.0, score))

    def _score_visual_hook(self, video_path: str, start_time: float) -> float:
        """
        Score visual hook using simple heuristics.

        Note: Full implementation would use computer vision.
        For now, returns neutral score.
        """
        # TODO: Implement visual analysis
        # - Scene changes in first 3s
        # - Face detection
        # - Motion analysis
        # - Color/brightness changes

        # Placeholder: return neutral score
        return 5.0

    def rank_clips(self, scored_clips: List[tuple]) -> List[tuple]:
        """
        Rank clips by hook score.

        Parameters
        ----------
        scored_clips : List[tuple]
            List of (clip_data, hook_score) tuples

        Returns
        -------
        List[tuple]
            Sorted list (highest score first)
        """
        return sorted(
            scored_clips,
            key=lambda x: x[1].overall_score,
            reverse=True
        )

    def filter_by_threshold(self, scored_clips: List[tuple],
                           min_score: float = 6.0) -> List[tuple]:
        """
        Filter clips below score threshold.

        Parameters
        ----------
        scored_clips : List[tuple]
            List of (clip_data, hook_score) tuples
        min_score : float
            Minimum acceptable score

        Returns
        -------
        List[tuple]
            Filtered list
        """
        return [
            (clip, score) for clip, score in scored_clips
            if score.overall_score >= min_score
        ]


class HybridScorer:
    """
    Hybrid scoring system combining VVSA + Council voting.

    This provides the best of both worlds:
    - VVSA for fast heuristic + single-model analysis
    - Council for multi-model consensus on top candidates
    """

    def __init__(self, vvsa_scorer: VVSAScorer, council_voter=None):
        """
        Initialize hybrid scorer.

        Parameters
        ----------
        vvsa_scorer : VVSAScorer
            VVSA scoring instance
        council_voter : CouncilVoter, optional
            Council voting instance for multi-model consensus
        """
        self.vvsa_scorer = vvsa_scorer
        self.council_voter = council_voter
        self.use_council = council_voter is not None

    def score_and_vote(self, clips: List[tuple],
                      transcription=None,
                      vvsa_threshold: float = 6.0,
                      council_top_n: int = 500) -> List[tuple]:
        """
        Two-stage scoring: VVSA filtering + Council voting.

        Stage 1: VVSA scores all clips, filters by threshold
        Stage 2: Council votes on filtered clips, returns top N

        Parameters
        ----------
        clips : List[tuple]
            List of (clip, hook_text) tuples
        transcription : Transcription, optional
            Full transcription
        vvsa_threshold : float
            Minimum VVSA score to pass to council (default: 6.0)
        council_top_n : int
            Number of clips to select via council (default: 500)

        Returns
        -------
        List[tuple]
            Top clips: (clip, vvsa_score, council_vote)
        """
        logger.info(f"Hybrid scoring: {len(clips)} clips")

        # Stage 1: VVSA scoring
        logger.info("Stage 1: VVSA scoring all clips...")
        vvsa_scored = []

        for clip, hook_text in clips:
            hook_score = self.vvsa_scorer.score_clip(
                transcription=transcription,
                video_path=None,
                start_time=getattr(clip, 'start_time', 0.0)
            )
            vvsa_scored.append((clip, hook_text, hook_score))

        # Filter by VVSA threshold
        filtered = [(c, t, s) for c, t, s in vvsa_scored
                   if s.overall_score >= vvsa_threshold]

        logger.info(f"  {len(filtered)}/{len(clips)} clips passed VVSA threshold (>= {vvsa_threshold})")

        if not filtered:
            logger.warning("No clips passed VVSA filtering!")
            return []

        # Stage 2: Council voting (if available)
        if self.use_council:
            logger.info(f"Stage 2: Council voting on {len(filtered)} clips...")

            # Prepare clips for council
            council_clips = [(c, t) for c, t, _ in filtered]

            # Get council votes
            voted = self.council_voter.vote_on_clips(
                council_clips,
                transcription=transcription,
                top_n=council_top_n
            )

            # Combine VVSA + Council scores
            # Find VVSA scores for voted clips
            vvsa_map = {id(c): s for c, _, s in filtered}
            result = []

            for clip, council_vote in voted:
                vvsa_score = vvsa_map.get(id(clip))
                result.append((clip, vvsa_score, council_vote))

            logger.info(f"Final: {len(result)} clips selected by council")
            return result

        else:
            # No council - just return VVSA-filtered clips
            logger.info("Council not available, using VVSA scores only")
            sorted_clips = sorted(filtered, key=lambda x: x[2].overall_score, reverse=True)
            return sorted_clips[:council_top_n]


def create_scorer(config) -> VVSAScorer:
    """
    Factory function to create VVSA scorer from config.

    Parameters
    ----------
    config : Config
        AdLab configuration

    Returns
    -------
    VVSAScorer
        Configured scorer instance
    """
    from .llm import ClaudeClient

    # Initialize LLM client if API key available
    llm_client = None
    try:
        api_key = config.get("anthropic.api_key")
        if api_key:
            llm_client = ClaudeClient(
                api_key=api_key,
                model=config.get("anthropic.model"),
                max_tokens=config.get("anthropic.max_tokens"),
                temperature=config.get("anthropic.temperature")
            )
    except Exception as e:
        logger.warning(f"Could not initialize LLM client: {e}")

    return VVSAScorer(
        llm_client=llm_client,
        hook_duration=config.get("vvsa.hook_duration", 3.0),
        weights=config.get("vvsa.weights")
    )


def create_hybrid_scorer(config) -> HybridScorer:
    """
    Factory function to create hybrid VVSA + Council scorer.

    Parameters
    ----------
    config : Config
        AdLab configuration

    Returns
    -------
    HybridScorer
        Configured hybrid scorer with VVSA + Council
    """
    # Create VVSA scorer
    vvsa_scorer = create_scorer(config)

    # Create council voter if enabled
    council_voter = None
    if config.get("council.enabled", True):
        try:
            from .council import create_council_voter
            council_voter = create_council_voter(config)
            logger.info("Council voter enabled for hybrid scoring")
        except Exception as e:
            logger.warning(f"Could not initialize council voter: {e}")
            logger.info("Falling back to VVSA-only scoring")

    return HybridScorer(vvsa_scorer, council_voter)
