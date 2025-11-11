"""
Title Generation using Claude/GPT
Supports voice dictation, A/B testing, and account-specific styles
"""
from typing import List, Dict, Any, Optional
import logging
import os
from anthropic import Anthropic
from openai import OpenAI

logger = logging.getLogger(__name__)


class TitleGenerator:
    """
    Generate title variants for viral clips.

    Uses:
    - Claude 3.5 Haiku (cheap, fast)
    - GPT-4 (high quality variants)
    """

    def __init__(
        self,
        anthropic_key: Optional[str] = None,
        openai_key: Optional[str] = None
    ):
        """Initialize with API keys."""
        self.anthropic_key = anthropic_key or os.getenv("ANTHROPIC_API_KEY")
        self.openai_key = openai_key or os.getenv("OPENAI_API_KEY")

        if self.anthropic_key:
            self.claude = Anthropic(api_key=self.anthropic_key)
        else:
            self.claude = None

        if self.openai_key:
            self.openai_client = OpenAI(api_key=self.openai_key)
        else:
            self.openai_client = None

    def generate_from_voice(
        self,
        voice_transcript: str,
        hook_score: float = 7.0,
        num_variants: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Generate title variants from voice dictation.

        Wes speaks the title idea, GPT-4 generates 5 variants.

        Args:
            voice_transcript: What Wes said
            hook_score: VVSA hook score
            num_variants: Number of variants to generate

        Returns:
            List of title variant dictionaries
        """
        prompt = f"""Generate {num_variants} title variants from this voice input:

Voice input: "{voice_transcript}"
Hook score: {hook_score}/10

Create viral title variants that:
- Are 40-70 characters (mobile-friendly)
- Include relevant emojis
- Create curiosity gap
- Promise value/entertainment
- Vary in style (curiosity, revelation, educational, emotional, clickbait)

Return as JSON array:
[
  {{
    "text": "title with emoji 🔥",
    "variant_id": "A",
    "hook_style": "curiosity",
    "predicted_ctr": 0.085
  }},
  ...
]
"""

        try:
            if self.openai_client:
                # Use GPT-4 for high quality (OpenAI SDK v2.x)
                response = self.openai_client.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": "You are a viral content title expert."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=1.2,
                    max_tokens=500
                )

                content = response.choices[0].message.content

            elif self.claude:
                # Fallback to Claude
                response = self.claude.messages.create(
                    model="claude-3-5-haiku-20241022",
                    max_tokens=512,
                    temperature=1.0,
                    messages=[{"role": "user", "content": prompt}]
                )

                content = response.content[0].text

            else:
                raise ValueError("No API keys configured")

            # Parse JSON response
            import json
            content = content.replace('```json', '').replace('```', '').strip()
            variants = json.loads(content)

            # Add variant IDs if missing
            for i, variant in enumerate(variants):
                if 'variant_id' not in variant:
                    variant['variant_id'] = chr(65 + i)  # A, B, C...

            logger.info(f"Generated {len(variants)} title variants from voice")

            return variants

        except Exception as e:
            logger.error(f"Title generation failed: {e}")
            # Return fallback variants
            return self._fallback_variants(voice_transcript)

    def generate_from_transcript(
        self,
        transcript: str,
        hook_score: float,
        duration: float,
        num_variants: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Generate titles from clip transcript.

        Args:
            transcript: Full clip transcript
            hook_score: VVSA score
            duration: Clip duration
            num_variants: Number of variants

        Returns:
            List of title variants
        """
        prompt = f"""Generate {num_variants} viral titles for this clip:

Transcript: "{transcript[:300]}..."
Hook score: {hook_score}/10
Duration: {duration}s

Requirements:
- 40-70 characters
- Mobile-friendly
- Include emojis
- Create curiosity
- Different styles

Return JSON array of title variants.
"""

        try:
            if self.claude:
                response = self.claude.messages.create(
                    model="claude-3-5-haiku-20241022",
                    max_tokens=512,
                    messages=[{"role": "user", "content": prompt}]
                )

                content = response.content[0].text

            elif self.openai_client:
                # OpenAI SDK v2.x
                response = self.openai_client.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": "Generate viral video titles."},
                        {"role": "user", "content": prompt}
                    ]
                )

                content = response.choices[0].message.content

            else:
                raise ValueError("No API keys configured")

            import json
            content = content.replace('```json', '').replace('```', '').strip()
            variants = json.loads(content)

            return variants

        except Exception as e:
            logger.error(f"Title generation failed: {e}")
            return self._fallback_variants(transcript[:50])

    def generate_for_account_type(
        self,
        screenshot_description: str,
        account_type: str,
        context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate title for specific account type.

        Account types:
        - fan: Casual fan reaction style
        - brand: Professional brand voice
        - watermark: Fan commentary

        Args:
            screenshot_description: Description of the screenshot
            account_type: Type of account
            context: Additional context

        Returns:
            Generated title with metadata
        """
        style_instructions = {
            "fan": "React like an excited fan. Use casual language, emojis, and slang. Example: 'bro was STRUGGLING 💀'",
            "brand": "Professional, inspiring tone. Clear value proposition. Example: 'Elite Training Techniques Revealed'",
            "watermark": "Fan commentary style. Observational, engaging. Example: 'the way he crushed this tho 🔥'"
        }

        instruction = style_instructions.get(account_type, style_instructions["fan"])

        prompt = f"""Generate a post title for this video screenshot:

Screenshot: {screenshot_description}
{f'Context: {context}' if context else ''}

Style: {instruction}

Return as JSON:
{{
  "title": "generated title",
  "tone": "casual/professional/commentary",
  "engagement_score": 0.85
}}
"""

        try:
            if self.claude:
                response = self.claude.messages.create(
                    model="claude-3-5-haiku-20241022",
                    max_tokens=256,
                    messages=[{"role": "user", "content": prompt}]
                )

                content = response.content[0].text

            else:
                raise ValueError("No Claude API key")

            import json
            content = content.replace('```json', '').replace('```', '').strip()
            result = json.loads(content)

            result['account_type'] = account_type

            return result

        except Exception as e:
            logger.error(f"Account-specific title generation failed: {e}")
            return {
                "title": f"{screenshot_description[:50]}...",
                "tone": account_type,
                "engagement_score": 0.5,
                "account_type": account_type
            }

    def _fallback_variants(self, base_text: str) -> List[Dict[str, Any]]:
        """Generate simple fallback variants."""
        return [
            {
                "text": f"🔥 {base_text[:50]}",
                "variant_id": "A",
                "hook_style": "direct",
                "predicted_ctr": 0.05
            },
            {
                "text": f"Watch: {base_text[:45]}...",
                "variant_id": "B",
                "hook_style": "curiosity",
                "predicted_ctr": 0.06
            },
            {
                "text": f"This is crazy 💀 {base_text[:40]}",
                "variant_id": "C",
                "hook_style": "emotional",
                "predicted_ctr": 0.07
            }
        ]


class CaptionFormatter:
    """
    Format captions according to rules:
    - Proxima Nova Sans, size 135
    - Centered
    - Max 11 chars per line
    - ONE line only
    - Remove commas/periods (except numbers)
    - 10px black stroke
    """

    def format_caption(self, text: str) -> str:
        """Format caption text according to rules."""
        # Remove punctuation except in numbers
        formatted = text

        # Remove commas and periods not in numbers
        import re
        formatted = re.sub(r'(?<!\d)[,.](?!\d)', '', formatted)

        # Truncate to 11 characters max per line
        if len(formatted) > 11:
            # Find good break point
            words = formatted.split()
            line = ""
            for word in words:
                if len(line + word) <= 11:
                    line += word + " "
                else:
                    break
            formatted = line.strip()

        # Ensure uppercase for impact
        formatted = formatted.upper()

        return formatted
