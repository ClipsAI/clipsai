"""
Configuration management for AdLab viral clip factory.
"""
import os
from pathlib import Path
from typing import Optional, Dict, Any
import yaml


class Config:
    """
    Configuration manager for AdLab.
    Loads from config.yaml or environment variables.
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration.

        Parameters
        ----------
        config_path : str, optional
            Path to YAML config file. If None, loads from default locations.
        """
        self.config_data = {}

        # Default config path
        if config_path is None:
            # Try current directory first, then adlab directory
            default_paths = [
                "config.yaml",
                "adlab/config.yaml",
                os.path.join(os.path.dirname(__file__), "config.yaml"),
            ]
            for path in default_paths:
                if os.path.exists(path):
                    config_path = path
                    break

        # Load config file if it exists
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config_data = yaml.safe_load(f) or {}

        # Apply defaults
        self._apply_defaults()

    def _apply_defaults(self):
        """Apply default values for missing config keys."""
        defaults = {
            "anthropic": {
                "api_key": os.getenv("ANTHROPIC_API_KEY", ""),
                "model": "claude-3-5-sonnet-20241022",
                "max_tokens": 1024,
                "temperature": 1.0,
            },
            "vvsa": {
                "hook_duration": 3.0,
                "min_score": 6.0,
                "weights": {
                    "visual": 0.3,
                    "audio": 0.2,
                    "text": 0.3,
                    "llm": 0.2,
                }
            },
            "variations": {
                "temporal_shifts": [-1.0, -0.5, 0, 0.5, 1.0],
                "durations": [15, 30, 45, 60],
                "aspect_ratios": ["9:16", "1:1", "4:5"],
                "max_variations_per_clip": 12,
            },
            "transcription": {
                "model_size": "base",
                "device": "auto",
                "language": None,
            },
            "export": {
                "output_dir": "./output",
                "video_codec": "libx264",
                "audio_codec": "aac",
                "preset": "medium",
                "crf": 23,
                "thumbnail_time": 1.0,
            },
            "processing": {
                "min_clip_duration": 10,
                "max_clip_duration": 90,
                "target_clips": 300,
                "max_clips": 500,
                "batch_size": 10,
            }
        }

        # Merge defaults with loaded config
        for key, value in defaults.items():
            if key not in self.config_data:
                self.config_data[key] = value
            elif isinstance(value, dict):
                for subkey, subvalue in value.items():
                    if subkey not in self.config_data[key]:
                        self.config_data[key][subkey] = subvalue

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a config value by dot-notation key.

        Parameters
        ----------
        key : str
            Configuration key in dot notation (e.g., "anthropic.api_key")
        default : Any, optional
            Default value if key not found

        Returns
        -------
        Any
            Configuration value
        """
        keys = key.split('.')
        value = self.config_data

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def set(self, key: str, value: Any):
        """
        Set a config value by dot-notation key.

        Parameters
        ----------
        key : str
            Configuration key in dot notation
        value : Any
            Value to set
        """
        keys = key.split('.')
        config = self.config_data

        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]

        config[keys[-1]] = value

    def validate(self) -> bool:
        """
        Validate configuration.

        Returns
        -------
        bool
            True if config is valid

        Raises
        ------
        ValueError
            If required configuration is missing or invalid
        """
        # Check for required API key
        api_key = self.get("anthropic.api_key")
        if not api_key:
            raise ValueError(
                "Anthropic API key not found. Set ANTHROPIC_API_KEY environment "
                "variable or add it to config.yaml"
            )

        # Validate output directory
        output_dir = self.get("export.output_dir")
        if not output_dir:
            raise ValueError("export.output_dir must be specified")

        # Validate min/max clip durations
        min_dur = self.get("processing.min_clip_duration")
        max_dur = self.get("processing.max_clip_duration")
        if min_dur >= max_dur:
            raise ValueError(
                f"min_clip_duration ({min_dur}) must be less than "
                f"max_clip_duration ({max_dur})"
            )

        return True

    def save(self, path: str):
        """
        Save configuration to YAML file.

        Parameters
        ----------
        path : str
            Output path for config file
        """
        with open(path, 'w') as f:
            yaml.safe_dump(self.config_data, f, default_flow_style=False, indent=2)


def create_example_config(output_path: str = "config.example.yaml"):
    """
    Create an example configuration file.

    Parameters
    ----------
    output_path : str
        Path where example config will be written
    """
    example_config = {
        "anthropic": {
            "api_key": "${ANTHROPIC_API_KEY}",
            "model": "claude-3-5-sonnet-20241022",
            "max_tokens": 1024,
            "temperature": 1.0,
        },
        "vvsa": {
            "hook_duration": 3.0,
            "min_score": 6.0,
            "weights": {
                "visual": 0.3,
                "audio": 0.2,
                "text": 0.3,
                "llm": 0.2,
            }
        },
        "variations": {
            "temporal_shifts": [-1.0, -0.5, 0, 0.5, 1.0],
            "durations": [15, 30, 45, 60],
            "aspect_ratios": ["9:16", "1:1", "4:5"],
            "max_variations_per_clip": 12,
        },
        "transcription": {
            "model_size": "base",
            "device": "auto",
            "language": None,
        },
        "export": {
            "output_dir": "./output",
            "video_codec": "libx264",
            "audio_codec": "aac",
            "preset": "medium",
            "crf": 23,
            "thumbnail_time": 1.0,
        },
        "processing": {
            "min_clip_duration": 10,
            "max_clip_duration": 90,
            "target_clips": 300,
            "max_clips": 500,
            "batch_size": 10,
        }
    }

    with open(output_path, 'w') as f:
        f.write("# AdLab Viral Clip Factory Configuration\n")
        f.write("# Copy this file to config.yaml and customize as needed\n\n")
        yaml.safe_dump(example_config, f, default_flow_style=False, indent=2)

    print(f"Example config created at {output_path}")


if __name__ == "__main__":
    # Generate example config when run directly
    create_example_config("config.example.yaml")
