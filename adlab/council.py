"""
Council Voting System for Clip Selection.

Uses multiple AI models to vote on clip quality, combining their scores
through consensus logic to select the best clips for viral potential.
"""
import os
import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import statistics

from clipsai import Transcription

logger = logging.getLogger(__name__)


@dataclass
class CouncilVote:
    """Container for council voting results."""
    consensus_score: float  # 0-10
    individual_scores: Dict[str, float]  # model_name -> score
    voting_method: str  # "average", "weighted", "median", etc.
    agreement_level: float  # 0-1, variance measure
    reasoning: Dict[str, str]  # model_name -> reasoning
    metadata: Dict[str, Any]


class CouncilVoter:
    """
    Multi-model AI council for clip voting.

    Uses 5 AI models to vote on clips:
    1. Claude 3.5 Sonnet (primary - high quality)
    2. Claude 3.5 Haiku (fallback - fast)
    3. GPT-4 (secondary - diverse perspective)
    4. Gemini 1.5 Flash (tertiary - if available)
    5. Local model fallback (heuristic-based)

    Each model scores clips 0-10 for viral potential.
    Consensus is calculated using configurable voting logic.
    """

    def __init__(self,
                 anthropic_api_key: Optional[str] = None,
                 openai_api_key: Optional[str] = None,
                 google_api_key: Optional[str] = None,
                 voting_method: str = "weighted_average",
                 weights: Optional[Dict[str, float]] = None,
                 min_models: int = 2,
                 enable_parallel: bool = True,
                 cache_enabled: bool = True):
        """
        Initialize the council voter.

        Parameters
        ----------
        anthropic_api_key : str, optional
            Anthropic API key for Claude models
        openai_api_key : str, optional
            OpenAI API key for GPT models
        google_api_key : str, optional
            Google API key for Gemini models
        voting_method : str
            Method for consensus: "average", "weighted_average", "median", "majority"
        weights : dict, optional
            Model weights for weighted voting (model_name -> weight)
        min_models : int
            Minimum models required for valid vote (default: 2)
        enable_parallel : bool
            Enable parallel model calls (default: True)
        cache_enabled : bool
            Cache model responses to avoid re-scoring (default: True)
        """
        self.voting_method = voting_method
        self.min_models = min_models
        self.enable_parallel = enable_parallel
        self.cache_enabled = cache_enabled

        # Default weights - Claude Sonnet gets highest weight
        self.weights = weights or {
            "claude_sonnet": 0.35,
            "gpt4": 0.30,
            "claude_haiku": 0.20,
            "gemini_flash": 0.10,
            "local_fallback": 0.05
        }

        # Normalize weights
        total = sum(self.weights.values())
        if total > 0:
            self.weights = {k: v / total for k, v in self.weights.items()}

        # Initialize model clients
        self.models = {}
        self._initialize_models(anthropic_api_key, openai_api_key, google_api_key)

        # Response cache (clip_id -> model_name -> score)
        self.cache = {} if cache_enabled else None

        # Thread pool for parallel execution
        self.executor = ThreadPoolExecutor(max_workers=5) if enable_parallel else None

        logger.info(f"CouncilVoter initialized with {len(self.models)} models")
        logger.info(f"Active models: {list(self.models.keys())}")

    def _initialize_models(self, anthropic_key, openai_key, google_key):
        """Initialize available AI model clients."""
        # Claude 3.5 Sonnet (primary)
        try:
            if anthropic_key or os.getenv("ANTHROPIC_API_KEY"):
                from .llm import ClaudeClient
                self.models["claude_sonnet"] = ClaudeClient(
                    api_key=anthropic_key or os.getenv("ANTHROPIC_API_KEY"),
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=1024,
                    temperature=1.0
                )
                logger.info("✓ Claude 3.5 Sonnet initialized")
        except Exception as e:
            logger.warning(f"✗ Claude Sonnet unavailable: {e}")

        # Claude 3.5 Haiku (fast fallback)
        try:
            if anthropic_key or os.getenv("ANTHROPIC_API_KEY"):
                from .llm import ClaudeClient
                self.models["claude_haiku"] = ClaudeClient(
                    api_key=anthropic_key or os.getenv("ANTHROPIC_API_KEY"),
                    model="claude-3-5-haiku-20241022",
                    max_tokens=512,
                    temperature=0.8
                )
                logger.info("✓ Claude 3.5 Haiku initialized")
        except Exception as e:
            logger.warning(f"✗ Claude Haiku unavailable: {e}")

        # GPT-4 (diverse perspective)
        try:
            if openai_key or os.getenv("OPENAI_API_KEY"):
                self.models["gpt4"] = self._create_gpt4_client(
                    openai_key or os.getenv("OPENAI_API_KEY")
                )
                logger.info("✓ GPT-4 initialized")
        except Exception as e:
            logger.warning(f"✗ GPT-4 unavailable: {e}")

        # Gemini 1.5 Flash (if available)
        try:
            if google_key or os.getenv("GOOGLE_API_KEY"):
                self.models["gemini_flash"] = self._create_gemini_client(
                    google_key or os.getenv("GOOGLE_API_KEY")
                )
                logger.info("✓ Gemini 1.5 Flash initialized")
        except Exception as e:
            logger.warning(f"✗ Gemini Flash unavailable: {e}")

        # Always add local fallback (heuristic-based)
        self.models["local_fallback"] = LocalFallbackModel()
        logger.info("✓ Local fallback model initialized")

    def _create_gpt4_client(self, api_key: str):
        """Create GPT-4 client wrapper."""
        return GPT4Client(api_key)

    def _create_gemini_client(self, api_key: str):
        """Create Gemini client wrapper."""
        return GeminiClient(api_key)

    def vote_on_clip(self,
                     transcript: str,
                     transcription: Optional[Transcription] = None,
                     video_path: Optional[str] = None,
                     start_time: float = 0.0,
                     clip_id: Optional[str] = None) -> CouncilVote:
        """
        Get council vote on a single clip.

        Parameters
        ----------
        transcript : str
            Hook transcript (first 3 seconds)
        transcription : Transcription, optional
            Full transcription object
        video_path : str, optional
            Path to video file
        start_time : float
            Clip start time
        clip_id : str, optional
            Clip identifier for caching

        Returns
        -------
        CouncilVote
            Complete voting results with consensus
        """
        # Check cache
        if self.cache_enabled and clip_id and clip_id in self.cache:
            cached_scores = self.cache[clip_id]
            logger.debug(f"Using cached scores for {clip_id}")
            return self._calculate_consensus(cached_scores, transcript)

        # Collect votes from all models
        scores = {}
        reasoning = {}

        if self.enable_parallel and len(self.models) > 1:
            # Parallel execution
            scores, reasoning = self._vote_parallel(transcript)
        else:
            # Sequential execution
            scores, reasoning = self._vote_sequential(transcript)

        # Cache results
        if self.cache_enabled and clip_id:
            self.cache[clip_id] = scores

        # Calculate consensus
        return self._calculate_consensus(scores, transcript, reasoning)

    def _vote_parallel(self, transcript: str) -> Tuple[Dict[str, float], Dict[str, str]]:
        """Execute voting in parallel across models."""
        from concurrent.futures import as_completed

        scores = {}
        reasoning = {}

        futures = {
            self.executor.submit(self._get_model_score, name, model, transcript): name
            for name, model in self.models.items()
        }

        for future in as_completed(futures):
            model_name = futures[future]
            try:
                score, reason = future.result(timeout=30)
                scores[model_name] = score
                reasoning[model_name] = reason
            except Exception as e:
                logger.warning(f"Model {model_name} failed: {e}")
                # Use neutral score on failure
                scores[model_name] = 5.0
                reasoning[model_name] = f"Error: {str(e)}"

        return scores, reasoning

    def _vote_sequential(self, transcript: str) -> Tuple[Dict[str, float], Dict[str, str]]:
        """Execute voting sequentially across models."""
        scores = {}
        reasoning = {}

        for name, model in self.models.items():
            try:
                score, reason = self._get_model_score(name, model, transcript)
                scores[name] = score
                reasoning[name] = reason
            except Exception as e:
                logger.warning(f"Model {name} failed: {e}")
                scores[name] = 5.0
                reasoning[name] = f"Error: {str(e)}"

        return scores, reasoning

    def _get_model_score(self, model_name: str, model: Any, transcript: str) -> Tuple[float, str]:
        """Get score from a single model."""
        try:
            result = model.score_hook(transcript)
            score = result.get("score", 5.0)
            reason = result.get("reasoning", "")
            return score, reason
        except Exception as e:
            logger.error(f"Error scoring with {model_name}: {e}")
            return 5.0, f"Error: {str(e)}"

    def _calculate_consensus(self,
                           scores: Dict[str, float],
                           transcript: str,
                           reasoning: Optional[Dict[str, str]] = None) -> CouncilVote:
        """
        Calculate consensus from individual model scores.

        Supports multiple voting methods:
        - average: Simple mean
        - weighted_average: Weighted by model quality
        - median: Median score
        - majority: Majority vote on threshold
        """
        if len(scores) < self.min_models:
            logger.warning(f"Only {len(scores)} models available (min: {self.min_models})")

        reasoning = reasoning or {}

        # Calculate consensus based on voting method
        if self.voting_method == "weighted_average":
            consensus = self._weighted_average(scores)
        elif self.voting_method == "median":
            consensus = statistics.median(scores.values())
        elif self.voting_method == "majority":
            consensus = self._majority_vote(scores)
        else:  # default to average
            consensus = statistics.mean(scores.values())

        # Calculate agreement level (1 - normalized variance)
        if len(scores) > 1:
            variance = statistics.variance(scores.values())
            # Normalize variance to 0-1 scale (assuming max variance ~25 for 0-10 scale)
            agreement = max(0, 1 - (variance / 25))
        else:
            agreement = 1.0

        return CouncilVote(
            consensus_score=round(consensus, 2),
            individual_scores=scores,
            voting_method=self.voting_method,
            agreement_level=round(agreement, 3),
            reasoning=reasoning,
            metadata={
                "transcript": transcript,
                "num_voters": len(scores),
                "score_range": (min(scores.values()), max(scores.values())) if scores else (0, 0)
            }
        )

    def _weighted_average(self, scores: Dict[str, float]) -> float:
        """Calculate weighted average of scores."""
        total = 0.0
        weight_sum = 0.0

        for model_name, score in scores.items():
            weight = self.weights.get(model_name, 0.1)
            total += score * weight
            weight_sum += weight

        return total / weight_sum if weight_sum > 0 else statistics.mean(scores.values())

    def _majority_vote(self, scores: Dict[str, float], threshold: float = 6.0) -> float:
        """Calculate majority vote (simple pass/fail, then average)."""
        passing = [s for s in scores.values() if s >= threshold]
        if len(passing) > len(scores) / 2:
            # Majority passed - return average of passing scores
            return statistics.mean(passing)
        else:
            # Majority failed - return average of all scores
            return statistics.mean(scores.values())

    def vote_on_clips(self,
                     clips: List[Tuple[Any, str]],
                     transcription: Optional[Transcription] = None,
                     top_n: int = 500) -> List[Tuple[Any, CouncilVote]]:
        """
        Vote on multiple clips and return top N by consensus.

        Parameters
        ----------
        clips : List[Tuple[clip_data, transcript]]
            List of clips with their hook transcripts
        transcription : Transcription, optional
            Full transcription object
        top_n : int
            Number of top clips to return (default: 500)

        Returns
        -------
        List[Tuple[clip_data, CouncilVote]]
            Top N clips sorted by consensus score
        """
        logger.info(f"Council voting on {len(clips)} clips...")

        voted_clips = []
        for i, (clip, transcript) in enumerate(clips):
            clip_id = getattr(clip, 'clip_id', None) or f"clip_{i}"

            vote = self.vote_on_clip(
                transcript=transcript,
                transcription=transcription,
                clip_id=clip_id
            )

            voted_clips.append((clip, vote))

            if (i + 1) % 50 == 0:
                logger.info(f"  Voted on {i + 1}/{len(clips)} clips")

        # Sort by consensus score
        voted_clips.sort(key=lambda x: x[1].consensus_score, reverse=True)

        # Return top N
        top_clips = voted_clips[:top_n]

        logger.info(f"Selected top {len(top_clips)} clips")
        logger.info(f"  Score range: {top_clips[0][1].consensus_score:.2f} - {top_clips[-1][1].consensus_score:.2f}")

        return top_clips

    def get_voting_stats(self) -> Dict[str, Any]:
        """Get statistics about council voting."""
        return {
            "num_models": len(self.models),
            "active_models": list(self.models.keys()),
            "voting_method": self.voting_method,
            "weights": self.weights,
            "cache_size": len(self.cache) if self.cache else 0,
            "min_models": self.min_models
        }

    def clear_cache(self):
        """Clear the voting cache."""
        if self.cache:
            self.cache.clear()
            logger.info("Council voting cache cleared")

    def __del__(self):
        """Cleanup executor on deletion."""
        if self.executor:
            self.executor.shutdown(wait=False)


class GPT4Client:
    """Wrapper for GPT-4 API."""

    def __init__(self, api_key: str):
        self.api_key = api_key
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key)
        except ImportError:
            logger.warning("openai package not installed")
            self.client = None

    def score_hook(self, transcript: str) -> Dict[str, Any]:
        """Score hook using GPT-4."""
        if not self.client:
            return {"score": 5.0, "reasoning": "OpenAI client unavailable"}

        prompt = f"""Rate this video hook (0-10) for viral potential on TikTok/Reels.

TRANSCRIPT: "{transcript}"

Consider:
- Pattern interrupt (grabs attention)
- Curiosity gap (makes viewer want more)
- Emotional impact
- Clarity of promise

Respond with just:
SCORE: [0-10]
REASONING: [one sentence]"""

        try:
            # OpenAI SDK v2.x
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.8,
                max_tokens=150
            )

            content = response.choices[0].message.content
            return self._parse_response(content)

        except Exception as e:
            logger.error(f"GPT-4 API error: {e}")
            return {"score": 5.0, "reasoning": f"API error: {str(e)}"}

    def _parse_response(self, content: str) -> Dict[str, Any]:
        """Parse GPT-4 response."""
        lines = content.strip().split('\n')
        score = 5.0
        reasoning = ""

        for line in lines:
            if line.startswith("SCORE:"):
                try:
                    score = float(line.replace("SCORE:", "").strip().split()[0])
                except:
                    pass
            elif line.startswith("REASONING:"):
                reasoning = line.replace("REASONING:", "").strip()

        return {"score": score, "reasoning": reasoning}


class GeminiClient:
    """Wrapper for Google Gemini API."""

    def __init__(self, api_key: str):
        self.api_key = api_key
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel('gemini-1.5-flash')
        except ImportError:
            logger.warning("google-generativeai package not installed")
            self.model = None

    def score_hook(self, transcript: str) -> Dict[str, Any]:
        """Score hook using Gemini."""
        if not self.model:
            return {"score": 5.0, "reasoning": "Gemini client unavailable"}

        prompt = f"""Rate this video hook (0-10) for viral potential.

TRANSCRIPT: "{transcript}"

Respond with:
SCORE: [0-10]
REASONING: [brief explanation]"""

        try:
            response = self.model.generate_content(prompt)
            return self._parse_response(response.text)

        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            return {"score": 5.0, "reasoning": f"API error: {str(e)}"}

    def _parse_response(self, content: str) -> Dict[str, Any]:
        """Parse Gemini response."""
        lines = content.strip().split('\n')
        score = 5.0
        reasoning = ""

        for line in lines:
            if "SCORE:" in line:
                try:
                    score = float(line.split("SCORE:")[1].strip().split()[0])
                except:
                    pass
            elif "REASONING:" in line:
                reasoning = line.split("REASONING:")[1].strip()

        return {"score": score, "reasoning": reasoning}


class LocalFallbackModel:
    """Heuristic-based fallback when APIs unavailable."""

    def score_hook(self, transcript: str) -> Dict[str, Any]:
        """Score using simple heuristics."""
        score = 5.0  # Base score
        reasons = []

        text_lower = transcript.lower()

        # Curiosity keywords (+2)
        curiosity_words = ["secret", "hack", "trick", "revealed", "hidden", "shocking",
                          "never", "why", "how", "discover", "truth"]
        if any(word in text_lower for word in curiosity_words):
            score += 2.0
            reasons.append("uses curiosity keywords")

        # Question format (+1)
        if "?" in transcript:
            score += 1.0
            reasons.append("poses question")

        # Emotional triggers (+1.5)
        emotions = ["amazing", "incredible", "insane", "crazy", "mind-blowing",
                   "unbelievable", "genius"]
        if any(emo in text_lower for emo in emotions):
            score += 1.5
            reasons.append("emotional trigger")

        # Numbers/specificity (+0.5)
        if any(char.isdigit() for char in transcript):
            score += 0.5
            reasons.append("includes numbers")

        # Length penalties
        word_count = len(transcript.split())
        if word_count < 3:
            score -= 2.0
            reasons.append("too short")
        elif word_count > 25:
            score -= 1.5
            reasons.append("too long")
        elif 5 <= word_count <= 15:
            score += 1.0
            reasons.append("optimal length")

        score = max(0, min(10, score))
        reasoning = "Local heuristic: " + ", ".join(reasons) if reasons else "neutral"

        return {"score": score, "reasoning": reasoning}


def create_council_voter(config) -> CouncilVoter:
    """
    Factory function to create council voter from config.

    Parameters
    ----------
    config : Config
        AdLab configuration

    Returns
    -------
    CouncilVoter
        Configured council voter instance
    """
    return CouncilVoter(
        anthropic_api_key=config.get("anthropic.api_key"),
        openai_api_key=config.get("openai.api_key"),
        google_api_key=config.get("google.api_key"),
        voting_method=config.get("council.voting_method", "weighted_average"),
        weights=config.get("council.weights"),
        min_models=config.get("council.min_models", 2),
        enable_parallel=config.get("council.enable_parallel", True),
        cache_enabled=config.get("council.cache_enabled", True)
    )
