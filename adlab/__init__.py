"""
AdLab: Viral Clip Factory
A smart pivot extension for ClipsAI that generates 300-500 short-form clips
from long videos using VVSA-style hook scoring.
"""

__version__ = "0.1.0"

from .config import Config
from .vvsa import VVSAScorer
from .variations import VariationGenerator
from .titles import TitleGenerator
from .captions import CaptionHandler
from .export import ClipExporter
from .manifest import ManifestWriter

__all__ = [
    "Config",
    "VVSAScorer",
    "VariationGenerator",
    "TitleGenerator",
    "CaptionHandler",
    "ClipExporter",
    "ManifestWriter",
]
