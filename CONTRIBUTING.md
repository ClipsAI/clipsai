# Contributing to ClipsAI

Thank you for your interest in contributing to ClipsAI! This document provides guidelines and instructions for contributing to the project.

## Table of Contents

- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Code Style Guide](#code-style-guide)
- [Testing Requirements](#testing-requirements)
- [Pull Request Process](#pull-request-process)
- [Code Review Guidelines](#code-review-guidelines)
- [Issue Reporting](#issue-reporting)
- [Community Guidelines](#community-guidelines)

## Getting Started

Before you begin:

1. Check the [existing issues](https://github.com/ClipsAI/clipsai/issues) to see if your problem/feature has been discussed
2. Read the [README.md](README.md) to understand the project
3. Review the [ARCHITECTURE.md](ARCHITECTURE.md) for technical design details
4. Join our discussions to connect with maintainers and other contributors

## Development Setup

### Prerequisites

- Python 3.9 or higher
- Git
- FFmpeg
- libmagic
- Node.js 18+ (for frontend development)

### Installation

1. **Fork and clone the repository:**

```bash
git clone https://github.com/YOUR_USERNAME/clipsai.git
cd clipsai
```

2. **Create a virtual environment:**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**

```bash
# Core library
pip install -e .

# Development dependencies
pip install -e ".[dev]"

# Install WhisperX
pip install whisperx@git+https://github.com/m-bain/whisperx.git

# For Clip Factory development
cd clipfactory/backend
pip install -r requirements.txt

cd ../frontend
npm install
```

4. **Install system dependencies:**

```bash
# Ubuntu/Debian
sudo apt-get install libmagic1 ffmpeg

# macOS
brew install libmagic ffmpeg
```

5. **Set up environment variables:**

```bash
cp clipfactory/.env.example clipfactory/.env
# Edit .env with your API keys
```

6. **Run tests to verify setup:**

```bash
pytest tests/
```

## Code Style Guide

We follow strict code style guidelines to maintain consistency and readability.

### Python Code Style

#### Formatter: Black

All Python code must be formatted with [Black](https://black.readthedocs.io/):

```bash
black .
```

- Line length: 88 characters (Black's default)
- Use double quotes for strings
- Format on save is recommended

#### Linter: flake8

Code must pass flake8 checks:

```bash
flake8 . --max-line-length=88 --extend-ignore=E203,W503
```

Configuration in `.flake8`:

```ini
[flake8]
max-line-length = 88
extend-ignore = E203, W503
exclude = .git,__pycache__,docs,build,dist,venv
```

#### Type Checker: mypy

All new code should include type hints:

```bash
mypy clipsai/ clipfactory/
```

Example:

```python
def find_clips(
    transcription: Transcription,
    min_duration: float = 30.0,
    max_duration: float = 90.0
) -> List[Clip]:
    """
    Find clips from transcription.

    Args:
        transcription: Video transcription with word-level timing
        min_duration: Minimum clip duration in seconds
        max_duration: Maximum clip duration in seconds

    Returns:
        List of Clip objects with start/end times
    """
    pass
```

### Naming Conventions

- **Classes**: `PascalCase` (e.g., `ClipFinder`, `TitleGenerator`)
- **Functions/methods**: `snake_case` (e.g., `find_clips`, `generate_variations`)
- **Constants**: `UPPER_SNAKE_CASE` (e.g., `MAX_DURATION`, `API_VERSION`)
- **Private methods**: Prefix with `_` (e.g., `_validate_input`)
- **Files**: `snake_case` (e.g., `clip_finder.py`, `title_generator.py`)

### Documentation

All public functions, classes, and modules must have docstrings:

```python
def resize(
    video_file_path: str,
    pyannote_auth_token: str,
    aspect_ratio: Tuple[int, int] = (9, 16)
) -> Crops:
    """
    Resize video to target aspect ratio with intelligent reframing.

    Uses Pyannote for speaker diarization and face tracking to ensure
    the speaker remains centered in the reframed video.

    Args:
        video_file_path: Absolute path to video file
        pyannote_auth_token: HuggingFace token for Pyannote access
        aspect_ratio: Target aspect ratio as (width, height) tuple

    Returns:
        Crops object containing segment-by-segment crop coordinates

    Raises:
        FileNotFoundError: If video file doesn't exist
        ValueError: If aspect ratio is invalid

    Example:
        >>> crops = resize("/path/to/video.mp4", "hf_token", (9, 16))
        >>> print(crops.segments[0])
    """
    pass
```

### TypeScript/JavaScript (Frontend)

- Use TypeScript for all new frontend code
- Follow [Airbnb JavaScript Style Guide](https://github.com/airbnb/javascript)
- Use Prettier for formatting
- ESLint for linting

```bash
# Format
npm run format

# Lint
npm run lint
```

## Testing Requirements

All contributions must include appropriate tests.

### Unit Tests

- Write unit tests for all new functions
- Aim for 80%+ code coverage
- Use `pytest` for Python tests
- Place tests in the `tests/` directory

```python
# tests/test_clip_finder.py
import pytest
from clipsai import ClipFinder

def test_find_clips_basic():
    """Test basic clip finding functionality."""
    finder = ClipFinder()
    # Test implementation
    assert True

def test_find_clips_with_invalid_input():
    """Test error handling with invalid input."""
    finder = ClipFinder()
    with pytest.raises(ValueError):
        finder.find_clips(None)
```

### Integration Tests

- Test complete workflows end-to-end
- Include tests for the full pipeline
- Mock external API calls

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_clip.py

# Run with coverage
pytest --cov=clipsai --cov-report=html

# Run frontend tests
cd clipfactory/frontend
npm test
```

### Test Requirements for PRs

- All existing tests must pass
- New features must have corresponding tests
- Bug fixes must include regression tests
- Coverage should not decrease

## Pull Request Process

### Before Submitting

1. **Update from main:**

```bash
git checkout main
git pull upstream main
git checkout your-feature-branch
git rebase main
```

2. **Run all checks:**

```bash
# Format code
black .

# Run linters
flake8 .
mypy clipsai/

# Run tests
pytest --cov=clipsai
```

3. **Update documentation:**
   - Update relevant docstrings
   - Update README.md if needed
   - Add/update tests
   - Update CHANGELOG.md

### PR Title Format

Use conventional commits format:

- `feat: Add voice-based title generation`
- `fix: Correct aspect ratio calculation in resize`
- `docs: Update CONTRIBUTING.md with testing guidelines`
- `refactor: Simplify clip selection logic`
- `test: Add integration tests for phase 3`
- `chore: Update dependencies`

### PR Description Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix (non-breaking change fixing an issue)
- [ ] New feature (non-breaking change adding functionality)
- [ ] Breaking change (fix or feature causing existing functionality to change)
- [ ] Documentation update

## Testing
- [ ] Unit tests pass locally
- [ ] Integration tests pass locally
- [ ] Manual testing completed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Comments added for complex code
- [ ] Documentation updated
- [ ] Tests added/updated
- [ ] No new warnings generated

## Related Issues
Closes #123
```

### Review Process

1. A maintainer will review your PR within 3-5 business days
2. Address any requested changes
3. Once approved, a maintainer will merge your PR

## Code Review Guidelines

### For Contributors

- Be responsive to feedback
- Keep discussions professional and constructive
- Update your PR based on review comments
- Ask for clarification if feedback is unclear

### For Reviewers

Review checklist:

- [ ] Code follows style guidelines
- [ ] Changes are well-documented
- [ ] Tests are comprehensive
- [ ] No unnecessary complexity
- [ ] Performance considerations addressed
- [ ] Security implications reviewed
- [ ] Error handling is robust

Provide constructive feedback:

```markdown
# Good feedback
"Consider using a list comprehension here for better performance and readability:
`clips = [c for c in all_clips if c.duration > min_duration]`"

# Less helpful feedback
"This code is bad"
```

## Issue Reporting

### Bug Reports

Use the bug report template and include:

1. **Description**: Clear, concise description of the bug
2. **Reproduction Steps**: Step-by-step instructions
3. **Expected Behavior**: What should happen
4. **Actual Behavior**: What actually happens
5. **Environment**:
   - OS and version
   - Python version
   - ClipsAI version
   - Relevant package versions
6. **Code Sample**: Minimal code to reproduce the issue
7. **Error Messages**: Full error traceback

### Feature Requests

Include:

1. **Problem Statement**: What problem does this solve?
2. **Proposed Solution**: How should it work?
3. **Alternatives Considered**: Other approaches you've thought about
4. **Use Cases**: Real-world scenarios where this would be useful

### Security Issues

**Do not open public issues for security vulnerabilities.**

See [SECURITY.md](SECURITY.md) for responsible disclosure process.

## Community Guidelines

### Code of Conduct

- Be respectful and inclusive
- Welcome newcomers
- Focus on constructive criticism
- Respect different viewpoints and experiences
- Accept responsibility for mistakes

### Communication Channels

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: Questions and general discussion
- **Discord**: Real-time chat (link in README)
- **Email**: support@clipsai.com for sensitive issues

### Recognition

Contributors are recognized in:

- CONTRIBUTORS.md file
- Release notes
- Project documentation

## Getting Help

- Check the [Documentation](https://docs.clipsai.com)
- Search [existing issues](https://github.com/ClipsAI/clipsai/issues)
- Ask in [GitHub Discussions](https://github.com/ClipsAI/clipsai/discussions)
- Review [ARCHITECTURE.md](ARCHITECTURE.md) for technical details

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to ClipsAI! Your efforts help make video editing accessible to everyone.
