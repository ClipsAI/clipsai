"""
Anthropic Claude API wrapper for AdLab.
Handles all LLM-based scoring, title generation, and content analysis.
"""
import os
from typing import Optional, Dict, Any, List
import logging

logger = logging.getLogger(__name__)


class ClaudeClient:
    """
    Wrapper for Anthropic Claude API.
    Provides hook scoring, title generation, and content analysis.
    """

    def __init__(self, api_key: Optional[str] = None, model: str = "claude-3-5-sonnet-20241022",
                 max_tokens: int = 1024, temperature: float = 1.0):
        """
        Initialize Claude client.

        Parameters
        ----------
        api_key : str, optional
            Anthropic API key. If None, reads from ANTHROPIC_API_KEY env var.
        model : str
            Claude model to use
        max_tokens : int
            Maximum tokens for responses
        temperature : float
            Sampling temperature
        """
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("Anthropic API key required")

        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

        # Lazy import to avoid requiring anthropic if not used
        try:
            import anthropic
            self.client = anthropic.Anthropic(api_key=self.api_key)
        except ImportError:
            logger.warning("anthropic package not installed. Install with: pip install anthropic")
            self.client = None

    def score_hook(self, transcript: str, context: str = "") -> Dict[str, Any]:
        """
        Score the hook quality of a video clip's first 3 seconds.

        Uses VVSA (Viral Video Success Analysis) methodology to evaluate:
        - Pattern interrupt (unexpected/surprising opening)
        - Curiosity gap (creates intrigue)
        - Emotional resonance
        - Clarity of promise

        Parameters
        ----------
        transcript : str
            Transcription of the first 3 seconds
        context : str, optional
            Additional context about the clip

        Returns
        -------
        dict
            {
                "score": float (0-10),
                "reasoning": str,
                "strengths": List[str],
                "improvements": List[str]
            }
        """
        if not self.client:
            # Fallback heuristic scoring if API unavailable
            return self._heuristic_hook_score(transcript)

        prompt = f"""Analyze this video hook (first 3 seconds) and score it 0-10 for viral potential.

TRANSCRIPT: "{transcript}"
{f'CONTEXT: {context}' if context else ''}

Use VVSA criteria:
1. Pattern Interrupt: Does it grab attention immediately? (0-10)
2. Curiosity Gap: Does it make viewers want to know more? (0-10)
3. Emotional Resonance: Does it trigger an emotion? (0-10)
4. Clarity: Is the promise/topic clear? (0-10)

Provide your response in this exact format:
SCORE: [0-10 score]
REASONING: [one sentence explanation]
STRENGTHS: [comma-separated list]
IMPROVEMENTS: [comma-separated list]

Be harsh but fair. Most hooks score 3-6. Only exceptional hooks score 8+."""

        try:
            message = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                messages=[{"role": "user", "content": prompt}]
            )

            response_text = message.content[0].text
            return self._parse_hook_response(response_text)

        except ImportError as e:
            logger.warning(f"Anthropic library not available: {e}")
            return self._heuristic_hook_score(transcript)
        except (KeyError, IndexError, AttributeError) as e:
            logger.error(f"Failed to parse Claude API response in score_hook: {e}", exc_info=True)
            return self._heuristic_hook_score(transcript)
        except Exception as e:
            logger.error(
                f"Claude API error in score_hook: {type(e).__name__}",
                exc_info=True,
                extra={
                    "error_type": type(e).__name__,
                    "transcript_length": len(transcript),
                    "model": self.model
                }
            )
            return self._heuristic_hook_score(transcript)

    def generate_titles(self, transcript: str, duration: int,
                       num_variants: int = 2) -> List[Dict[str, str]]:
        """
        Generate engaging titles for a clip.

        Parameters
        ----------
        transcript : str
            Full clip transcription
        duration : int
            Clip duration in seconds
        num_variants : int
            Number of title variants to generate (default: 2 for A/B testing)

        Returns
        -------
        List[dict]
            List of title variants with metadata
            [
                {
                    "title": str,
                    "hook_style": str,
                    "target_audience": str,
                    "predicted_ctr": float
                },
                ...
            ]
        """
        if not self.client:
            return self._generate_fallback_titles(transcript, num_variants)

        prompt = f"""Generate {num_variants} engaging titles for this {duration}s video clip.

TRANSCRIPT: "{transcript[:500]}..."

Requirements:
- Under 60 characters
- Strong hook/curiosity gap
- Clear value proposition
- Optimized for short-form platforms (TikTok, Reels, Shorts)

For each title, also specify:
1. Hook style (curiosity/controversy/education/entertainment)
2. Target audience
3. Predicted CTR (1-10)

Format each title as:
TITLE: [title text]
HOOK_STYLE: [style]
AUDIENCE: [audience]
CTR: [1-10]

Generate {num_variants} diverse variants optimized for A/B testing."""

        try:
            message = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                messages=[{"role": "user", "content": prompt}]
            )

            response_text = message.content[0].text
            return self._parse_titles_response(response_text, num_variants)

        except ImportError as e:
            logger.warning(f"Anthropic library not available: {e}")
            return self._generate_fallback_titles(transcript, num_variants)
        except (KeyError, IndexError, AttributeError) as e:
            logger.error(f"Failed to parse Claude API response in generate_titles: {e}", exc_info=True)
            return self._generate_fallback_titles(transcript, num_variants)
        except Exception as e:
            logger.error(
                f"Claude API error in generate_titles: {type(e).__name__}",
                exc_info=True,
                extra={
                    "error_type": type(e).__name__,
                    "transcript_length": len(transcript),
                    "num_variants": num_variants,
                    "model": self.model
                }
            )
            return self._generate_fallback_titles(transcript, num_variants)

    def suggest_tags(self, transcript: str, title: str) -> List[str]:
        """
        Generate relevant hashtags/tags for a clip.

        Parameters
        ----------
        transcript : str
            Clip transcription
        title : str
            Clip title

        Returns
        -------
        List[str]
            List of recommended tags
        """
        if not self.client:
            return self._generate_fallback_tags(transcript)

        prompt = f"""Generate 10 relevant hashtags for this video clip.

TITLE: {title}
TRANSCRIPT: "{transcript[:300]}..."

Requirements:
- Mix of broad and niche tags
- Platform-optimized (TikTok/Instagram style)
- Include trending topics where relevant
- No # prefix needed

Return as comma-separated list."""

        try:
            message = self.client.messages.create(
                model=self.model,
                max_tokens=256,
                temperature=0.8,
                messages=[{"role": "user", "content": prompt}]
            )

            response_text = message.content[0].text
            tags = [tag.strip().lstrip('#') for tag in response_text.split(',')]
            return tags[:10]

        except ImportError as e:
            logger.warning(f"Anthropic library not available: {e}")
            return self._generate_fallback_tags(transcript)
        except (KeyError, IndexError, AttributeError) as e:
            logger.error(f"Failed to parse Claude API response in suggest_tags: {e}", exc_info=True)
            return self._generate_fallback_tags(transcript)
        except Exception as e:
            logger.error(
                f"Claude API error in suggest_tags: {type(e).__name__}",
                exc_info=True,
                extra={
                    "error_type": type(e).__name__,
                    "transcript_length": len(transcript),
                    "title": title,
                    "model": self.model
                }
            )
            return self._generate_fallback_tags(transcript)

    def _parse_hook_response(self, response: str) -> Dict[str, Any]:
        """Parse Claude's hook scoring response."""
        lines = response.strip().split('\n')
        result = {
            "score": 5.0,
            "reasoning": "",
            "strengths": [],
            "improvements": []
        }

        for line in lines:
            line = line.strip()
            if line.startswith("SCORE:"):
                try:
                    score_str = line.replace("SCORE:", "").strip()
                    result["score"] = float(score_str.split()[0])
                except (ValueError, IndexError):
                    pass
            elif line.startswith("REASONING:"):
                result["reasoning"] = line.replace("REASONING:", "").strip()
            elif line.startswith("STRENGTHS:"):
                strengths = line.replace("STRENGTHS:", "").strip()
                result["strengths"] = [s.strip() for s in strengths.split(',')]
            elif line.startswith("IMPROVEMENTS:"):
                improvements = line.replace("IMPROVEMENTS:", "").strip()
                result["improvements"] = [i.strip() for i in improvements.split(',')]

        return result

    def _parse_titles_response(self, response: str, num_variants: int) -> List[Dict[str, str]]:
        """Parse Claude's title generation response."""
        titles = []
        lines = response.strip().split('\n')

        current_title = {}
        for line in lines:
            line = line.strip()
            if line.startswith("TITLE:"):
                if current_title and "title" in current_title:
                    titles.append(current_title)
                current_title = {"title": line.replace("TITLE:", "").strip()}
            elif line.startswith("HOOK_STYLE:"):
                current_title["hook_style"] = line.replace("HOOK_STYLE:", "").strip()
            elif line.startswith("AUDIENCE:"):
                current_title["target_audience"] = line.replace("AUDIENCE:", "").strip()
            elif line.startswith("CTR:"):
                try:
                    ctr_str = line.replace("CTR:", "").strip()
                    current_title["predicted_ctr"] = float(ctr_str.split()[0])
                except (ValueError, IndexError):
                    current_title["predicted_ctr"] = 5.0

        if current_title and "title" in current_title:
            titles.append(current_title)

        return titles[:num_variants]

    def _heuristic_hook_score(self, transcript: str) -> Dict[str, Any]:
        """Fallback heuristic scoring when API unavailable."""
        score = 5.0
        strengths = []
        improvements = []

        # Simple heuristics
        if any(word in transcript.lower() for word in ["secret", "hack", "trick", "revealed"]):
            score += 1.0
            strengths.append("Uses curiosity-driving keywords")

        if transcript.endswith("?"):
            score += 0.5
            strengths.append("Opens with a question")

        if len(transcript.split()) < 5:
            score -= 0.5
            improvements.append("Hook might be too short")

        if len(transcript.split()) > 25:
            score -= 1.0
            improvements.append("Hook too long, may lose attention")

        return {
            "score": max(0, min(10, score)),
            "reasoning": "Heuristic-based scoring (API unavailable)",
            "strengths": strengths or ["Clear opening"],
            "improvements": improvements or ["Consider adding curiosity gap"]
        }

    def _generate_fallback_titles(self, transcript: str, num_variants: int) -> List[Dict[str, str]]:
        """Generate simple fallback titles."""
        words = transcript.split()[:8]
        base_title = " ".join(words) + "..."

        titles = []
        for i in range(num_variants):
            titles.append({
                "title": base_title,
                "hook_style": "direct",
                "target_audience": "general",
                "predicted_ctr": 5.0
            })
        return titles

    def _generate_fallback_tags(self, transcript: str) -> List[str]:
        """Generate simple fallback tags."""
        # Extract potential keywords
        words = transcript.lower().split()
        common_tags = ["viral", "fyp", "trending", "shorts", "reels"]

        # Simple keyword extraction
        keywords = [w for w in words if len(w) > 5][:5]
        return common_tags + keywords
