# CI/CD Quick Reference Card

Quick reference for common CI/CD tasks and commands.

## 🚀 Quick Start

```bash
# Install pre-commit hooks
pip install pre-commit
pre-commit install

# Run all pre-commit checks
pre-commit run --all-files

# Run tests locally
pytest tests/ -v --cov=clipsai

# Format code
black .

# Lint code
flake8 clipsai tests --max-line-length=100 --extend-ignore=E203,W503
```

## 📋 Workflows at a Glance

| Workflow | Trigger | Runtime | Purpose |
|----------|---------|---------|---------|
| CI | Push, PR | ~10-15min | Lint, test, security |
| Build | Push main, tags | ~15-20min | Docker build & push |
| Deploy | Push main | ~5-10min | Deploy to staging |
| Security | Daily, PR | ~20-30min | Security scanning |
| Tests | PR, push | ~15-25min | Extended tests |

## 🔑 GitHub Secrets Needed

### Required Now:
- `CODECOV_TOKEN` - Get from codecov.io

### For Staging:
- `STAGING_HOST`
- `STAGING_USER`
- `STAGING_SSH_KEY`
- `STAGING_BACKEND_URL`
- `STAGING_FRONTEND_URL`

### Optional:
- `SLACK_WEBHOOK`
- `DISCORD_WEBHOOK`

## ✅ CI Checks

Every push/PR runs:
- ✓ Black formatting
- ✓ Flake8 linting
- ✓ MyPy type checking
- ✓ Bandit security scan
- ✓ Safety dependency scan
- ✓ Unit tests (3 Python versions)
- ✓ Coverage report
- ✓ Docker builds

## 🔒 Security Scans

Daily scans include:
- ✓ Dependency vulnerabilities (Safety, pip-audit)
- ✓ Secret detection (detect-secrets)
- ✓ Code analysis (Semgrep, CodeQL)
- ✓ Python security (Bandit)
- ✓ Container scanning (Trivy)
- ✓ License compliance (pip-licenses)

## 🐳 Docker Images

### Backend:
```
ghcr.io/clipsai/clipsai/backend:latest
ghcr.io/clipsai/clipsai/backend:v0.2.1
```

### Frontend:
```
ghcr.io/clipsai/clipsai/frontend:latest
ghcr.io/clipsai/clipsai/frontend:v0.2.1
```

## 🏷️ Release Process

```bash
# Create version tag
git tag v0.2.2
git push origin v0.2.2

# Build workflow automatically:
# - Builds images
# - Tags with version
# - Pushes to registry
# - Scans for security
```

## 🛠️ Common Commands

### Local Testing:
```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=clipsai --cov-report=html

# Run specific test
pytest tests/test_clip.py -v

# Run in parallel
pytest tests/ -v -n auto
```

### Code Quality:
```bash
# Format code
black .

# Check formatting (no changes)
black --check .

# Sort imports
isort .

# Lint
flake8 clipsai tests

# Type check
mypy clipsai
```

### Security:
```bash
# Security scan
bandit -r clipsai

# Check dependencies
safety check

# Detect secrets
detect-secrets scan --all-files
```

### Docker:
```bash
# Build backend
cd clipfactory/backend
docker build -t clipsai-backend:local .

# Build frontend
cd clipfactory/frontend
docker build -t clipsai-frontend:local .

# Run locally
docker-compose up
```

## 🐛 Troubleshooting

### CI Failing on Black/Flake8
```bash
black .
flake8 clipsai tests --max-line-length=100 --extend-ignore=E203,W503
git add .
git commit -m "Fix formatting"
```

### Tests Failing
```bash
# Run tests with verbose output
pytest tests/ -v -s

# Run specific failing test
pytest tests/test_file.py::test_name -v -s
```

### Pre-commit Failing
```bash
# Update hooks
pre-commit autoupdate

# Skip hooks (not recommended)
git commit --no-verify

# Run specific hook
pre-commit run black --all-files
```

### Docker Build Failing
```bash
# Test build locally
docker build -t test:latest .

# Check logs
docker logs <container_id>

# Clean build cache
docker builder prune -a
```

## 📊 Monitoring

### Check Workflow Status:
1. Go to repository → Actions tab
2. View recent workflow runs
3. Click run for detailed logs

### Check Security:
1. Go to repository → Security tab
2. View alerts and advisories
3. Review scan results

### Check Coverage:
1. Visit codecov.io/gh/ClipsAI/clipsai
2. View coverage trends
3. Identify uncovered code

## 🔄 Pre-commit Hooks

Installed hooks (run on `git commit`):
1. Trim whitespace
2. Fix EOF
3. Check YAML/JSON/TOML
4. Prevent large files
5. Format with Black
6. Sort imports (isort)
7. Lint (Flake8)
8. Security scan (Bandit)
9. Detect secrets
10. Type check (mypy)

### Bypass (emergency only):
```bash
git commit --no-verify
```

## 📝 Configuration Files

| File | Purpose |
|------|---------|
| `.coveragerc` | Coverage settings |
| `pyproject.toml` | Black, isort, mypy, pytest |
| `.bandit` | Security scanner config |
| `codecov.yml` | Coverage reporting |
| `.pre-commit-config.yaml` | Pre-commit hooks |

## 🎯 Coverage Targets

- **Project:** 80%
- **Patch (new code):** 75%
- **Threshold:** 2% drop allowed

## 📚 Documentation

- **Full Guide:** `.github/CICD_SETUP.md`
- **Security:** `.github/SECURITY_SETUP.md`
- **Secrets:** `.github/secrets.template.env`
- **Report:** `CICD_DEPLOYMENT_REPORT.md`

## 🆘 Getting Help

1. Check workflow logs in Actions tab
2. Review documentation files
3. Search existing issues
4. Create new issue with `ci/cd` label

## 💡 Best Practices

- ✓ Run tests locally before pushing
- ✓ Use pre-commit hooks
- ✓ Keep commits small
- ✓ Write meaningful commit messages
- ✓ Monitor coverage trends
- ✓ Review security alerts promptly
- ✓ Update dependencies regularly

## 🚫 Common Mistakes

- ✗ Committing without running tests
- ✗ Bypassing pre-commit hooks
- ✗ Ignoring security warnings
- ✗ Pushing directly to main
- ✗ Not reviewing CI logs
- ✗ Hardcoding secrets

## 📈 Success Metrics

Monitor these:
- Build success rate (>95%)
- Test coverage (>80%)
- Average build time (<15min)
- Critical vulnerabilities (0)

---

**Quick Links:**
- [Actions](../../actions)
- [Security](../../security)
- [Pull Requests](../../pulls)
- [Issues](../../issues)

**Last Updated:** 2025-11-10
