# Security Setup Guide

This guide explains the security features included in the CI/CD pipeline and how to configure them.

## Security Scanning Overview

The pipeline includes multiple layers of security scanning:

1. **Dependency Vulnerability Scanning** - Identifies vulnerable dependencies
2. **Secret Detection** - Prevents secrets from being committed
3. **Static Application Security Testing (SAST)** - Finds code vulnerabilities
4. **Container Security Scanning** - Scans Docker images
5. **License Compliance** - Ensures license compatibility

## Security Workflow Components

### 1. Dependency Vulnerability Scanning

**Tools:** Safety, pip-audit

**What it checks:**
- Known vulnerabilities in Python packages
- Outdated packages with security fixes available
- CVE database matches

**Configuration:**
- Runs automatically on every PR and daily
- Reports uploaded as artifacts
- Configurable severity levels

**Viewing Results:**
- Check workflow artifacts for detailed reports
- GitHub Security tab shows high-severity issues

### 2. Secret Scanning

**Tool:** detect-secrets

**What it checks:**
- API keys
- Private keys
- Passwords
- Access tokens
- AWS credentials
- Database connection strings

**Setup:**

```bash
# Initialize secrets baseline
pip install detect-secrets
detect-secrets scan --all-files > .secrets.baseline

# Audit the baseline (review and mark false positives)
detect-secrets audit .secrets.baseline
```

**Pre-commit Hook:**
The pre-commit hook automatically scans for secrets before commit.

**Handling False Positives:**
Edit `.secrets.baseline` and mark false positives during audit.

### 3. Static Application Security Testing (SAST)

#### Semgrep

**What it checks:**
- OWASP Top 10 vulnerabilities
- Python security anti-patterns
- Injection vulnerabilities
- Authentication issues
- Cryptography misuse

**Configuration:**
Uses multiple rulesets:
- `p/security-audit` - General security audit
- `p/python` - Python-specific rules
- `p/bandit` - Python security rules
- `p/owasp-top-ten` - OWASP Top 10

**Results:** Uploaded to GitHub Security tab as SARIF

#### CodeQL

**What it checks:**
- SQL injection
- Cross-site scripting (XSS)
- Command injection
- Path traversal
- Insecure randomness
- And 100+ other security issues

**Configuration:**
Uses `security-extended` and `security-and-quality` queries.

**Results:** Viewable in GitHub Security tab

#### Bandit

**What it checks:**
- Hard-coded passwords
- Use of eval()
- Shell injection
- SQL injection
- Assert usage
- Weak cryptography

**Configuration:** `.bandit` file

**Skipped Tests:**
- B101 (assert_used) - Common in tests
- B601 (paramiko_calls) - If using paramiko

**Customization:**
Edit `.bandit` to adjust severity levels or skip specific tests.

### 4. Container Security Scanning

**Tool:** Trivy

**What it checks:**
- OS package vulnerabilities
- Application dependency vulnerabilities
- Configuration issues
- Exposed secrets in images

**When it runs:**
- After Docker images are built
- Daily scheduled scans of latest images
- On-demand via workflow dispatch

**Severity Levels:**
- CRITICAL - Must fix immediately
- HIGH - Should fix soon
- MEDIUM - Consider fixing

**Results:** GitHub Security tab (SARIF format)

### 5. License Compliance

**Tool:** pip-licenses

**What it checks:**
- License types of all dependencies
- License compatibility
- Copyleft licenses

**Reports:**
- Markdown format for documentation
- JSON format for automation

**Viewing Reports:**
Download artifacts from workflow runs.

## GitHub Security Features

### Security Advisories

GitHub automatically creates security advisories for:
- Dependabot alerts
- CodeQL findings
- Secret scanning alerts

**Accessing:**
Repository → Security tab → Advisories

### Dependabot

**Recommendation:** Enable Dependabot for automated dependency updates.

**Setup:**
1. Go to Settings → Code security and analysis
2. Enable "Dependabot alerts"
3. Enable "Dependabot security updates"
4. Optionally enable "Dependabot version updates"

**Configuration:** Create `.github/dependabot.yml`

```yaml
version: 2
updates:
  - package-ecosystem: "pip"
    directory: "/"
    schedule:
      interval: "weekly"
    open-pull-requests-limit: 10
```

## Security Best Practices

### 1. Secrets Management

**DO:**
- Use GitHub Secrets for all sensitive data
- Rotate secrets regularly
- Use different secrets for different environments
- Limit secret access to necessary workflows

**DON'T:**
- Hard-code secrets in code
- Commit `.env` files with secrets
- Log secrets (they're automatically masked in Actions)
- Share secrets in plain text

### 2. Dependency Management

**DO:**
- Keep dependencies up to date
- Review dependency changes
- Use pinned versions in production
- Monitor security advisories

**DON'T:**
- Ignore dependency updates
- Use deprecated packages
- Install from untrusted sources

### 3. Code Security

**DO:**
- Follow OWASP guidelines
- Validate all inputs
- Use parameterized queries
- Implement proper authentication
- Use strong cryptography

**DON'T:**
- Trust user input
- Use eval() or exec()
- Store passwords in plain text
- Use weak encryption

### 4. Container Security

**DO:**
- Use official base images
- Keep base images updated
- Use specific version tags
- Run as non-root user
- Minimize image layers

**DON'T:**
- Use `latest` tag in production
- Install unnecessary packages
- Expose unnecessary ports
- Store secrets in images

## Handling Security Issues

### Critical Vulnerabilities

1. Security workflow fails with CRITICAL severity
2. Review the finding in GitHub Security tab
3. Fix the vulnerability immediately
4. Test the fix locally
5. Deploy to staging first
6. Monitor for issues

### False Positives

1. Review the security finding
2. Verify it's actually a false positive
3. Document why it's a false positive
4. Add to appropriate ignore list:
   - Semgrep: `.semgrepignore`
   - Bandit: `.bandit`
   - detect-secrets: `.secrets.baseline`
5. Include justification in PR

### Security Hotfixes

For critical security issues in production:

1. Create hotfix branch from main
2. Apply minimal fix
3. Security scans run automatically
4. Fast-track PR review
5. Deploy immediately after merge
6. Monitor closely

## Security Reporting

### Internal Reporting

Security scan results are available in:
1. GitHub Security tab
2. Workflow run artifacts
3. PR comments (for Semgrep)
4. Codecov (for coverage)

### External Reporting

For responsible disclosure:
1. Check `SECURITY.md` in repository root
2. Follow responsible disclosure process
3. Allow time for fix before public disclosure

## Compliance and Auditing

### Audit Trail

GitHub Actions provides:
- Complete workflow run history
- Detailed logs (retained 90 days)
- Artifact storage (retained 90 days)
- Security event history

### Compliance Reports

Generate compliance reports from:
- Security scan artifacts
- License compliance reports
- Dependency audit reports
- Docker scan reports

### Regular Security Reviews

Schedule regular reviews:
- **Weekly:** Dependabot PRs
- **Monthly:** Security scan results
- **Quarterly:** Full security audit
- **Yearly:** Penetration testing (recommended)

## Advanced Configuration

### Custom Security Rules

**Semgrep:**
Create `.semgrep.yml` for custom rules.

**Bandit:**
Edit `.bandit` to customize scans.

**CodeQL:**
Create `.github/codeql/codeql-config.yml` for custom queries.

### Integrations

Consider integrating:
- **Snyk** - Advanced dependency scanning
- **SonarQube** - Code quality and security
- **WhiteSource** - License compliance
- **Aqua Security** - Container security
- **HashiCorp Vault** - Secrets management

## Troubleshooting

### High False Positive Rate

- Review and tune scanner configurations
- Add appropriate ignores with documentation
- Consider using multiple scanners for validation

### Performance Issues

- Run expensive scans on schedule, not every PR
- Use caching effectively
- Run scans in parallel

### Missing Vulnerabilities

- Use multiple complementary tools
- Keep scanner databases updated
- Supplement with manual security reviews

## Resources

- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [GitHub Security Features](https://docs.github.com/en/code-security)
- [Semgrep Rules](https://semgrep.dev/explore)
- [CodeQL Documentation](https://codeql.github.com/docs/)
- [Trivy Documentation](https://aquasecurity.github.io/trivy/)
- [Python Security Best Practices](https://python.readthedocs.io/en/latest/library/security_warnings.html)

## Support

For security concerns:
1. Review GitHub Security tab
2. Check workflow logs
3. Consult this guide
4. Open security issue (privately if sensitive)
