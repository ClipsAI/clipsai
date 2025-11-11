# Security Policy

## Supported Versions

Security updates are provided for the following versions:

| Version | Supported          |
| ------- | ------------------ |
| 0.2.x   | :white_check_mark: |
| 0.1.x   | :x:                |
| < 0.1   | :x:                |

## Reporting a Vulnerability

**Please do not report security vulnerabilities through public GitHub issues.**

Instead, please report them responsibly via one of the following methods:

### Email

Send details to: **security@clipsai.com**

Include:
- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if any)

### Expected Response Time

- **Initial Response**: Within 48 hours
- **Status Update**: Within 7 days
- **Fix Timeline**: Depends on severity
  - Critical: 7-14 days
  - High: 14-30 days
  - Medium: 30-60 days
  - Low: Next minor release

### Disclosure Policy

We follow responsible disclosure principles:

1. Report is received and acknowledged
2. Issue is confirmed and assessed
3. Fix is developed and tested
4. Security advisory is prepared
5. Fix is released
6. Advisory is published 7 days after release

## Security Best Practices

### Authentication & Authorization

#### Current State (Development)

⚠️ **Warning**: The current development version has **no authentication**. All API endpoints are publicly accessible.

**Do not deploy to production without implementing authentication.**

#### Planned Authentication

**API Key Authentication**:

```python
from fastapi import Security, HTTPException
from fastapi.security.api_key import APIKeyHeader

API_KEY_HEADER = APIKeyHeader(name="X-API-Key")

async def verify_api_key(api_key: str = Security(API_KEY_HEADER)):
    if api_key not in valid_api_keys:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return api_key
```

**JWT Tokens for Session Management**:

```python
from jose import JWTError, jwt
from datetime import datetime, timedelta

def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=15))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm="HS256")
```

**OAuth 2.0 for Third-Party Integrations**:

For social media posting, use OAuth 2.0:
- TikTok API
- Instagram Graph API
- YouTube Data API

### API Key Management

#### Storing API Keys

**Never commit API keys to version control.**

Use environment variables:

```bash
# .env (never commit this file)
ANTHROPIC_API_KEY=sk-ant-xxxxx
OPENAI_API_KEY=sk-xxxxx
PYANNOTE_AUTH_TOKEN=hf_xxxxx
DATABASE_URL=postgresql://user:pass@localhost/db
```

Load in application:

```python
import os
from dotenv import load_dotenv

load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
```

#### Key Rotation

Rotate API keys regularly:

1. Generate new key
2. Update environment variables
3. Deploy with new key
4. Revoke old key after grace period

### File Upload Security

#### Validation

**Always validate uploaded files:**

```python
from fastapi import UploadFile, HTTPException
import magic

ALLOWED_MIME_TYPES = [
    "video/mp4",
    "video/quicktime",
    "video/x-msvideo",
    "image/png",
    "image/jpeg"
]

MAX_FILE_SIZE = 5 * 1024 * 1024 * 1024  # 5GB

async def validate_upload(file: UploadFile):
    # Check file size
    file.file.seek(0, 2)
    size = file.file.tell()
    file.file.seek(0)

    if size > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="File too large")

    # Check MIME type
    mime = magic.from_buffer(file.file.read(1024), mime=True)
    file.file.seek(0)

    if mime not in ALLOWED_MIME_TYPES:
        raise HTTPException(status_code=400, detail="Invalid file type")
```

#### File Storage

**Store uploads securely:**

```python
import secrets
from pathlib import Path

UPLOAD_DIR = Path("/secure/uploads")
UPLOAD_DIR.mkdir(exist_ok=True, mode=0o700)

def save_upload(file: UploadFile) -> Path:
    # Generate random filename
    ext = Path(file.filename).suffix
    random_name = secrets.token_hex(16) + ext
    file_path = UPLOAD_DIR / random_name

    # Save with restricted permissions
    with open(file_path, "wb") as f:
        f.write(file.file.read())

    file_path.chmod(0o600)
    return file_path
```

#### Virus Scanning

For production, integrate virus scanning:

```python
import clamd

def scan_file(file_path: Path) -> bool:
    """Scan file for viruses using ClamAV."""
    cd = clamd.ClamdUnixSocket()
    result = cd.scan(str(file_path))
    return result[str(file_path)][0] == 'OK'
```

### Database Security

#### Connection Security

**Always use SSL/TLS for database connections:**

```python
from sqlalchemy import create_engine

DATABASE_URL = "postgresql://user:pass@localhost/db?sslmode=require"
engine = create_engine(DATABASE_URL)
```

#### SQL Injection Prevention

**Use parameterized queries:**

```python
# GOOD: Parameterized query
cursor.execute("SELECT * FROM clips WHERE video_id = %s", (video_id,))

# BAD: String concatenation
cursor.execute(f"SELECT * FROM clips WHERE video_id = '{video_id}'")
```

#### Password Hashing

**Never store plain text passwords:**

```python
from passlib.context import CryptContext

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def hash_password(password: str) -> str:
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)
```

#### Database Permissions

Principle of least privilege:

```sql
-- Create application user with limited permissions
CREATE USER clipfactory_app WITH PASSWORD 'strong_password';

-- Grant only necessary permissions
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO clipfactory_app;
REVOKE CREATE ON SCHEMA public FROM clipfactory_app;

-- Read-only user for analytics
CREATE USER clipfactory_readonly WITH PASSWORD 'strong_password';
GRANT SELECT ON ALL TABLES IN SCHEMA public TO clipfactory_readonly;
```

### Input Validation

**Validate all user inputs:**

```python
from pydantic import BaseModel, validator, constr
from typing import List

class VariationRequest(BaseModel):
    clip_id: constr(min_length=1, max_length=100)
    temporal_variations: List[str]
    reframe_styles: List[str]

    @validator('temporal_variations')
    def validate_temporal(cls, v):
        allowed = ['base', '+4s', '+35s']
        if not all(item in allowed for item in v):
            raise ValueError('Invalid temporal variation')
        return v

    @validator('reframe_styles')
    def validate_reframe(cls, v):
        allowed = ['original', 'flipped', 'blurry_bg']
        if not all(item in allowed for item in v):
            raise ValueError('Invalid reframe style')
        return v
```

### CORS Configuration

**Configure CORS appropriately:**

```python
from fastapi.middleware.cors import CORSMiddleware

# Development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://clipfactory.io"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["Content-Type", "Authorization"],
)
```

### Rate Limiting

**Implement rate limiting:**

```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.post("/api/phase1/upload")
@limiter.limit("5/hour")  # 5 uploads per hour per IP
async def upload_video(request: Request, video: UploadFile = File(...)):
    # ...
```

### Secrets Management

#### Development

Use `.env` files (excluded from git):

```bash
# .env
ANTHROPIC_API_KEY=sk-ant-xxxxx
OPENAI_API_KEY=sk-xxxxx
DATABASE_PASSWORD=xxxxx
```

#### Production

Use secrets management services:

**AWS Secrets Manager**:

```python
import boto3
from botocore.exceptions import ClientError

def get_secret(secret_name):
    session = boto3.session.Session()
    client = session.client(service_name='secretsmanager')

    try:
        response = client.get_secret_value(SecretId=secret_name)
        return response['SecretString']
    except ClientError as e:
        raise e
```

**HashiCorp Vault**:

```python
import hvac

client = hvac.Client(url='https://vault.example.com')
client.token = os.environ['VAULT_TOKEN']

secret = client.secrets.kv.v2.read_secret_version(path='clipfactory/api-keys')
api_key = secret['data']['data']['anthropic_key']
```

### Logging & Monitoring

**Log security events:**

```python
import logging

logger = logging.getLogger('security')

# Log authentication attempts
logger.info(f"Login attempt: user={username}, ip={ip_address}, success={success}")

# Log authorization failures
logger.warning(f"Unauthorized access attempt: endpoint={endpoint}, ip={ip_address}")

# Log suspicious activity
logger.error(f"Suspicious activity: {details}")
```

**Never log sensitive data:**

```python
# GOOD
logger.info(f"User {user_id} uploaded video")

# BAD
logger.info(f"API key: {api_key}")
logger.info(f"Password: {password}")
```

### HTTPS/TLS

**Always use HTTPS in production:**

```nginx
# Nginx configuration
server {
    listen 443 ssl http2;
    server_name api.clipfactory.io;

    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;

    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}

# Redirect HTTP to HTTPS
server {
    listen 80;
    server_name api.clipfactory.io;
    return 301 https://$server_name$request_uri;
}
```

### Dependency Security

**Regularly update dependencies:**

```bash
# Check for vulnerabilities
pip install safety
safety check

# Update dependencies
pip list --outdated
pip install --upgrade package_name
```

**Use dependency scanning:**

```yaml
# GitHub Actions
- name: Run Safety Check
  run: |
    pip install safety
    safety check --json
```

### Content Security Policy

**Set CSP headers:**

```python
from fastapi.responses import HTMLResponse

@app.get("/")
async def home():
    content = "<html>...</html>"
    headers = {
        "Content-Security-Policy": (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline'; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data: https:; "
        )
    }
    return HTMLResponse(content=content, headers=headers)
```

### Security Headers

**Add security headers:**

```python
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        return response

app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["clipfactory.io", "*.clipfactory.io"])
```

## Security Checklist

### Before Production Deployment

- [ ] API authentication implemented
- [ ] HTTPS/TLS enabled
- [ ] Environment variables configured
- [ ] File upload validation enabled
- [ ] Rate limiting configured
- [ ] CORS properly configured
- [ ] Security headers added
- [ ] Database uses SSL
- [ ] Secrets stored securely
- [ ] Logging configured (no sensitive data)
- [ ] Dependencies updated
- [ ] Vulnerability scan completed
- [ ] Penetration testing completed

### Regular Maintenance

- [ ] Rotate API keys quarterly
- [ ] Update dependencies monthly
- [ ] Review access logs weekly
- [ ] Backup database daily
- [ ] Test disaster recovery quarterly
- [ ] Review security policies annually

## Known Security Considerations

### Current Limitations

1. **No authentication in development**: All endpoints are public
2. **No rate limiting**: Susceptible to abuse
3. **Local file storage**: Not suitable for distributed systems
4. **No virus scanning**: Uploaded files not scanned
5. **No audit logging**: Limited security event tracking

### Future Enhancements

1. Implement OAuth 2.0 authentication
2. Add rate limiting per endpoint
3. Migrate to cloud storage (S3, GCS)
4. Integrate ClamAV virus scanning
5. Implement comprehensive audit logging
6. Add IP whitelisting/blacklisting
7. Implement API key scoping (read/write permissions)
8. Add request signing for API calls

## Security Resources

- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [FastAPI Security Best Practices](https://fastapi.tiangolo.com/tutorial/security/)
- [Python Security Guide](https://python.readthedocs.io/en/stable/library/security.html)
- [PostgreSQL Security](https://www.postgresql.org/docs/current/security.html)

## Contact

For security concerns:
- **Email**: security@clipsai.com
- **PGP Key**: Available on request

---

**Last Updated**: 2025-11-10

We take security seriously. Thank you for helping keep ClipsAI secure.
