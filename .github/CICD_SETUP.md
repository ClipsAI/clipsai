# CI/CD Pipeline Setup Guide

This document provides instructions for setting up and using the CI/CD pipeline for ClipsAI.

## Overview

The CI/CD pipeline consists of 5 GitHub Actions workflows:

1. **CI (Continuous Integration)** - Runs on every push and PR
2. **Build** - Builds and pushes Docker images
3. **Deploy Staging** - Deploys to staging environment
4. **Security** - Comprehensive security scanning
5. **Tests** - Extended test suite

## Quick Start

### 1. Install Pre-commit Hooks (Recommended)

Pre-commit hooks help catch issues before they reach CI:

```bash
# Install pre-commit
pip install pre-commit

# Install the git hook scripts
pre-commit install

# (Optional) Run against all files
pre-commit run --all-files
```

### 2. Required GitHub Secrets

Navigate to your repository settings → Secrets and variables → Actions, and add the following secrets:

#### Essential Secrets

- `CODECOV_TOKEN` - Token from codecov.io for coverage reports
  - Sign up at https://codecov.io
  - Add your repository
  - Copy the upload token

#### Optional Secrets (for full deployment)

- `STAGING_HOST` - Staging server hostname
- `STAGING_USER` - SSH username for staging
- `STAGING_SSH_KEY` - SSH private key for staging
- `STAGING_BACKEND_URL` - Backend URL for health checks
- `STAGING_FRONTEND_URL` - Frontend URL for health checks
- `STAGING_URL` - Main staging URL
- `STAGING_API_KEY` - API key for staging tests
- `KUBE_CONFIG_STAGING` - Kubernetes config for staging (if using K8s)

#### Notification Secrets (optional)

- `SLACK_WEBHOOK` - Slack webhook URL for notifications
- `DISCORD_WEBHOOK` - Discord webhook URL for notifications

## Workflows Details

### CI Workflow (`.github/workflows/ci.yml`)

**Triggers:** Push and PR to main branch

**Jobs:**
- **Lint** - Black formatter and Flake8 linter (Python 3.9, 3.10, 3.11)
- **Type Check** - MyPy static type checking
- **Security** - Bandit and Safety security scans
- **Test** - Pytest with coverage (Python 3.9, 3.10, 3.11)
- **Build Docker** - Test Docker image builds

**Required for merge:** Yes

### Build Workflow (`.github/workflows/build.yml`)

**Triggers:** Push to main, version tags, manual dispatch

**Jobs:**
- **Build Backend** - Build and push backend Docker image
- **Build Frontend** - Build and push frontend Docker image
- **Scan Images** - Trivy vulnerability scanning

**Image Registry:** GitHub Container Registry (ghcr.io)

**Tags Generated:**
- `latest` - Latest main branch build
- `v1.2.3` - Semantic version tags
- `main-abc1234` - Commit SHA tags

### Deploy Staging Workflow (`.github/workflows/deploy-staging.yml`)

**Triggers:** Push to main, manual dispatch

**Jobs:**
- **Deploy** - Deploy to staging environment
- **Smoke Test** - Run smoke tests
- **Notify** - Send deployment notifications

**Note:** Currently configured with mock deployment. Uncomment and configure actual deployment steps when staging infrastructure is ready.

### Security Workflow (`.github/workflows/security.yml`)

**Triggers:** Daily at 2 AM UTC, PR, push to main

**Jobs:**
- **Dependency Scan** - Safety and pip-audit
- **Secret Scan** - detect-secrets
- **SAST (Semgrep)** - Static analysis with Semgrep
- **SAST (CodeQL)** - GitHub CodeQL analysis
- **Bandit Scan** - Python security scanning
- **Docker Scan** - Trivy container scanning
- **License Scan** - License compliance check

**Security Reports:** Uploaded to GitHub Security tab

### Tests Workflow (`.github/workflows/tests.yml`)

**Triggers:** PR, push to main

**Jobs:**
- **Unit Tests** - Fast unit tests (Python 3.9, 3.10, 3.11)
- **Integration Tests** - Integration tests (placeholder)
- **E2E Tests** - End-to-end tests (placeholder)
- **Performance Tests** - Benchmark tests (placeholder)
- **Compatibility Tests** - Test on Ubuntu and macOS

## Development Workflow

### Making Changes

1. Create a feature branch:
   ```bash
   git checkout -b feature/my-feature
   ```

2. Make your changes and test locally:
   ```bash
   # Run tests
   pytest tests/ -v

   # Run pre-commit checks
   pre-commit run --all-files
   ```

3. Commit your changes:
   ```bash
   git add .
   git commit -m "Add my feature"
   # Pre-commit hooks will run automatically
   ```

4. Push to GitHub:
   ```bash
   git push origin feature/my-feature
   ```

5. Create a Pull Request
   - CI workflow will run automatically
   - All checks must pass before merge

### Merge to Main

When PR is merged to main:
1. CI workflow runs again
2. Build workflow builds and pushes Docker images
3. Deploy workflow deploys to staging
4. Smoke tests verify deployment

### Creating a Release

1. Tag the commit:
   ```bash
   git tag v0.2.2
   git push origin v0.2.2
   ```

2. Build workflow will:
   - Build Docker images with version tag
   - Scan images for vulnerabilities
   - Push to registry

## Configuration Files

### `.coveragerc` / `pyproject.toml`
Coverage configuration for pytest-cov

### `.bandit`
Bandit security scanner configuration

### `codecov.yml`
Codecov coverage reporting configuration

### `.pre-commit-config.yaml`
Pre-commit hooks configuration

## Caching

The workflows use aggressive caching to speed up runs:

- **pip packages** - Cached based on setup.py hash
- **Docker layers** - GitHub Actions cache
- **Pre-commit environments** - Cached by pre-commit

## Troubleshooting

### CI Fails on Black/Flake8

Run locally to fix:
```bash
black .
flake8 clipsai tests --max-line-length=100 --extend-ignore=E203,W503
```

### Tests Fail Locally but Pass in CI (or vice versa)

Check Python version:
```bash
python --version  # Should be 3.9+
```

Ensure all dependencies installed:
```bash
pip install -e .[dev]
```

### Docker Build Fails

Test build locally:
```bash
cd clipfactory/backend
docker build -t test:latest .
```

### Pre-commit Hooks Fail

Update hooks:
```bash
pre-commit autoupdate
pre-commit run --all-files
```

### Coverage Too Low

Check coverage report:
```bash
pytest --cov=clipsai --cov-report=html
# Open htmlcov/index.html
```

## Best Practices

1. **Always run pre-commit hooks** - Catches issues before CI
2. **Write tests for new code** - Maintain coverage above 80%
3. **Keep commits small** - Easier to review and debug
4. **Update documentation** - Keep this guide current
5. **Monitor security alerts** - Review Security tab regularly
6. **Use semantic versioning** - v{major}.{minor}.{patch}

## GitHub Actions Permissions

The workflows require these permissions:

- `contents: read` - Read repository contents
- `packages: write` - Push to GitHub Container Registry
- `security-events: write` - Upload security scan results
- `actions: read` - Read workflow information

These are configured in each workflow file.

## Status Badges

Add these badges to your README.md (already added):

```markdown
[![CI Status](https://github.com/ClipsAI/clipsai/workflows/CI%20-%20Continuous%20Integration/badge.svg)](https://github.com/ClipsAI/clipsai/actions/workflows/ci.yml)
[![Tests](https://github.com/ClipsAI/clipsai/workflows/Tests%20-%20Extended%20Test%20Suite/badge.svg)](https://github.com/ClipsAI/clipsai/actions/workflows/tests.yml)
[![Security](https://github.com/ClipsAI/clipsai/workflows/Security%20-%20Security%20Scanning/badge.svg)](https://github.com/ClipsAI/clipsai/actions/workflows/security.yml)
[![codecov](https://codecov.io/gh/ClipsAI/clipsai/branch/main/graph/badge.svg)](https://codecov.io/gh/ClipsAI/clipsai)
```

## Next Steps

1. **Set up Codecov** - Sign up and add `CODECOV_TOKEN`
2. **Configure staging deployment** - Uncomment and configure deployment steps
3. **Add integration tests** - Implement actual integration tests
4. **Set up notifications** - Add Slack/Discord webhooks
5. **Configure branch protection** - Require CI checks to pass

## Support

For issues with the CI/CD pipeline:
1. Check the Actions tab for detailed logs
2. Review this documentation
3. Check GitHub Actions documentation
4. Open an issue with the `ci/cd` label

## Updating the Pipeline

To modify workflows:
1. Edit files in `.github/workflows/`
2. Test changes in a feature branch first
3. Monitor the Actions tab for results
4. Update this documentation if needed
