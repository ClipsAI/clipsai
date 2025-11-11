"""
Title and tag generation for viral clips.
Supports A/B testing with multiple title variants.
"""
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class TitleVariant:
    """Container for a title variant."""
    text: str
    variant_id: str  # "A", "B", etc.
    hook_style: str
    target_audience: str
    predicted_ctr: float
    tags: List[str]
    metadata: Dict[str, Any]


class TitleGenerator:
    """
    Generates engaging titles and tags for clips.
    Supports A/B testing with multiple variants.
    """

    def __init__(self, llm_client=None, num_variants: int = 2):
        """
        Initialize title generator.

        Parameters
        ----------
        llm_client : ClaudeClient, optional
            LLM client for advanced title generation
        num_variants : int
            Number of title variants to generate (default: 2 for A/B testing)
        """
        self.llm_client = llm_client
        self.num_variants = num_variants

    def generate_titles(self,
                       transcript: str,
                       duration: int,
                       hook_score: float = 5.0,
                       context: Optional[str] = None) -> List[TitleVariant]:
        """
        Generate title variants for a clip.

        Parameters
        ----------
        transcript : str
            Full clip transcription
        duration : int
            Clip duration in seconds
        hook_score : float
            VVSA hook score (0-10)
        context : str, optional
            Additional context about the clip

        Returns
        -------
        List[TitleVariant]
            Generated title variants
        """
        variants = []

        # Try LLM generation first
        if self.llm_client:
            try:
                llm_titles = self.llm_client.generate_titles(
                    transcript=transcript,
                    duration=duration,
                    num_variants=self.num_variants
                )

                for i, title_data in enumerate(llm_titles):
                    # Generate tags for each title
                    tags = self.llm_client.suggest_tags(
                        transcript=transcript,
                        title=title_data.get("title", "")
                    )

                    variant = TitleVariant(
                        text=title_data.get("title", ""),
                        variant_id=chr(65 + i),  # A, B, C, ...
                        hook_style=title_data.get("hook_style", "direct"),
                        target_audience=title_data.get("target_audience", "general"),
                        predicted_ctr=title_data.get("predicted_ctr", 5.0),
                        tags=tags,
                        metadata={
                            "source": "llm",
                            "duration": duration,
                            "hook_score": hook_score,
                        }
                    )
                    variants.append(variant)

                if variants:
                    logger.info(f"Generated {len(variants)} LLM title variants")
                    return variants

            except Exception as e:
                logger.warning(f"LLM title generation failed: {e}")

        # Fallback to template-based generation
        variants = self._generate_template_titles(transcript, duration, hook_score)
        logger.info(f"Generated {len(variants)} template-based title variants")
        return variants

    def _generate_template_titles(self,
                                 transcript: str,
                                 duration: int,
                                 hook_score: float) -> List[TitleVariant]:
        """
        Generate titles using templates (fallback method).

        Parameters
        ----------
        transcript : str
            Clip transcription
        duration : int
            Duration in seconds
        hook_score : float
            Hook score

        Returns
        -------
        List[TitleVariant]
            Template-based title variants
        """
        # Extract key phrases from transcript
        words = transcript.split()[:10]
        preview = " ".join(words)

        # Generate variants using different templates
        templates = [
            {
                "template": f"{preview}... 🔥",
                "style": "direct",
                "audience": "general",
                "ctr": 5.0
            },
            {
                "template": f"Watch: {preview}",
                "style": "curiosity",
                "audience": "engaged",
                "ctr": 6.0
            },
            {
                "template": f"You won't believe what happens... {words[0] if words else ''}",
                "style": "curiosity",
                "audience": "casual",
                "ctr": 4.5
            },
        ]

        variants = []
        for i, template_data in enumerate(templates[:self.num_variants]):
            title = template_data["template"][:60]  # Limit to 60 chars

            tags = self._generate_fallback_tags(transcript)

            variant = TitleVariant(
                text=title,
                variant_id=chr(65 + i),  # A, B, C
                hook_style=template_data["style"],
                target_audience=template_data["audience"],
                predicted_ctr=template_data["ctr"],
                tags=tags,
                metadata={
                    "source": "template",
                    "duration": duration,
                    "hook_score": hook_score,
                }
            )
            variants.append(variant)

        return variants

    def _generate_fallback_tags(self, transcript: str) -> List[str]:
        """Generate simple fallback tags from transcript."""
        # Common viral tags
        base_tags = ["fyp", "viral", "foryou", "trending"]

        # Extract potential keywords (words > 5 chars)
        words = transcript.lower().split()
        keywords = [w for w in words if len(w) > 5 and w.isalpha()][:6]

        return base_tags + keywords

    def optimize_title_length(self, title: str, platform: str = "tiktok") -> str:
        """
        Optimize title length for specific platform.

        Parameters
        ----------
        title : str
            Original title
        platform : str
            Target platform: "tiktok", "instagram", "youtube"

        Returns
        -------
        str
            Optimized title
        """
        max_lengths = {
            "tiktok": 100,
            "instagram": 125,
            "youtube": 100,
            "default": 60,
        }

        max_len = max_lengths.get(platform, max_lengths["default"])

        if len(title) <= max_len:
            return title

        # Truncate and add ellipsis
        return title[:max_len - 3] + "..."

    def generate_seo_metadata(self,
                            title: str,
                            transcript: str,
                            tags: List[str]) -> Dict[str, Any]:
        """
        Generate SEO metadata for a clip.

        Parameters
        ----------
        title : str
            Clip title
        transcript : str
            Full transcription
        tags : List[str]
            Generated tags

        Returns
        -------
        Dict[str, Any]
            SEO metadata including description, keywords, etc.
        """
        # Generate description from transcript
        words = transcript.split()
        description = " ".join(words[:50])
        if len(words) > 50:
            description += "..."

        # Extract keywords from tags and transcript
        keywords = list(set(tags + [
            word.lower() for word in transcript.split()
            if len(word) > 5 and word.isalpha()
        ]))[:20]

        return {
            "title": title,
            "description": description,
            "keywords": keywords,
            "tags": tags,
            "og_title": title,
            "og_description": description,
            "twitter_card": "player",
        }

    def score_title_quality(self, title: str) -> float:
        """
        Score title quality using heuristics.

        Parameters
        ----------
        title : str
            Title to score

        Returns
        -------
        float
            Quality score (0-10)
        """
        score = 5.0

        # Length check
        length = len(title)
        if 30 <= length <= 60:
            score += 1.0
        elif length < 20 or length > 80:
            score -= 1.0

        # Curiosity keywords
        curiosity = ["secret", "revealed", "hidden", "shocking", "never", "why", "how"]
        if any(word in title.lower() for word in curiosity):
            score += 1.5

        # Numbers
        if any(char.isdigit() for char in title):
            score += 0.5

        # Emojis
        if any(ord(char) > 127 for char in title):
            score += 0.5

        # Question mark
        if "?" in title:
            score += 0.5

        # All caps (penalty)
        if title.isupper():
            score -= 1.0

        # Clickbait detection (moderate penalty)
        clickbait = ["you won't believe", "shocking", "incredible", "mind-blowing"]
        if any(phrase in title.lower() for phrase in clickbait):
            score -= 0.5

        return max(0.0, min(10.0, score))

    def batch_generate_titles(self,
                            clips_data: List[Dict[str, Any]]) -> Dict[str, List[TitleVariant]]:
        """
        Generate titles for multiple clips in batch.

        Parameters
        ----------
        clips_data : List[Dict[str, Any]]
            List of clip data dictionaries with transcript, duration, etc.

        Returns
        -------
        Dict[str, List[TitleVariant]]
            Mapping of clip_id to title variants
        """
        results = {}

        for clip_data in clips_data:
            clip_id = clip_data.get("clip_id", "unknown")
            transcript = clip_data.get("transcript", "")
            duration = clip_data.get("duration", 30)
            hook_score = clip_data.get("hook_score", 5.0)

            try:
                variants = self.generate_titles(
                    transcript=transcript,
                    duration=duration,
                    hook_score=hook_score
                )
                results[clip_id] = variants
            except Exception as e:
                logger.error(f"Failed to generate titles for clip {clip_id}: {e}")
                results[clip_id] = []

        logger.info(f"Generated titles for {len(results)} clips")
        return results


def create_title_generator(config, llm_client=None) -> TitleGenerator:
    """
    Factory function to create title generator from config.

    Parameters
    ----------
    config : Config
        AdLab configuration
    llm_client : ClaudeClient, optional
        Pre-initialized LLM client

    Returns
    -------
    TitleGenerator
        Configured generator instance
    """
    if llm_client is None and config.get("anthropic.api_key"):
        from .llm import ClaudeClient
        try:
            llm_client = ClaudeClient(
                api_key=config.get("anthropic.api_key"),
                model=config.get("anthropic.model"),
                max_tokens=config.get("anthropic.max_tokens"),
                temperature=config.get("anthropic.temperature")
            )
        except Exception as e:
            logger.warning(f"Could not initialize LLM client: {e}")

    return TitleGenerator(
        llm_client=llm_client,
        num_variants=2  # A/B testing
    )
