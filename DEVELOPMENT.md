# ImpactSense Development Environment Setup

## Prerequisites
- Python 3.9+
- Node.js 16+ (for frontend development)
- PostgreSQL 13+ (for production database)
- Redis (for caching and background tasks)

## Installation & Setup

### 1. Clone and Setup Python Environment
```bash
# Navigate to project directory
cd c:\Users\semar\Desktop\CodingProjects\BPI

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Environment Configuration
Create a `.env` file in the project root:
```env
# Database Configuration
DATABASE_URL=postgresql://username:password@localhost:5432/impactsense
REDIS_URL=redis://localhost:6379

# BPI Integration
BPI_API_BASE_URL=https://api.bpi.com.ph
BPI_CLIENT_ID=your_client_id
BPI_CLIENT_SECRET=your_client_secret

# NGO Integration
NGO_WEBHOOK_SECRET=your_webhook_secret

# Security
JWT_SECRET_KEY=your_jwt_secret
API_RATE_LIMIT=100

# Logging
LOG_LEVEL=INFO
SENTRY_DSN=your_sentry_dsn  # Optional for error tracking
```

### 3. Database Setup
```bash
# Initialize database
python -m alembic upgrade head

# Load initial NGO data
python scripts/load_ngos.py

# Create test users (development only)
python scripts/create_test_data.py
```

### 4. Start Development Server
```bash
# Start API server
uvicorn src.api.main:app --reload --port 8000

# In another terminal, start background worker
celery -A src.tasks.worker worker --loglevel=info

# Start Redis (if not running as service)
redis-server
```

## Development Workflow

### Running Tests
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_agents.py

# Run tests with live logging
pytest -s --log-cli-level=INFO
```

### Code Quality
```bash
# Format code
black src/ tests/

# Check code style
flake8 src/ tests/

# Type checking
mypy src/

# Sort imports
isort src/ tests/
```

### Database Migrations
```bash
# Create new migration
alembic revision --autogenerate -m "Description of changes"

# Apply migrations
alembic upgrade head

# Rollback last migration
alembic downgrade -1
```

## Agent Development & Testing

### Testing Individual Agents
```python
# Example: Test DonationLikelyScoreAgent
from src.agents.core_agents import DonationLikelyScoreAgent, UserContext

agent = DonationLikelyScoreAgent()
context = UserContext(
    user_id="test_user",
    transaction_amount=500.0,
    wallet_balance=2000.0,
    location={"region": "NCR"},
    transaction_category="groceries",
    days_since_last_prompt=7,
    days_since_last_donation=14,
    # ... other required fields
)

result = agent.process(context)
print(f"Likelihood Score: {result['likelihood_score']}")
print(f"Recommendation: {result['recommendation']}")
```

### Debugging Agent Logic
```python
# Enable detailed logging for agents
import logging
logging.getLogger("ImpactSense").setLevel(logging.DEBUG)

# Use the orchestrator with test data
from src.agents.core_agents import ImpactSenseOrchestrator
from constants.ngos import NGOS

orchestrator = ImpactSenseOrchestrator(NGOS)
recommendation = orchestrator.generate_recommendation(context)
```

## API Development & Testing

### Manual API Testing
```bash
# Test health endpoint
curl http://localhost:8000/health

# Test donation prompt evaluation
curl -X POST http://localhost:8000/evaluate-donation-prompt \
  -H "Content-Type: application/json" \
  -d @tests/fixtures/sample_request.json

# View API documentation
# Open http://localhost:8000/docs in browser
```

### Load Testing
```bash
# Install load testing tools
pip install locust

# Run load tests
locust -f tests/load_test.py --host=http://localhost:8000
```

## NGO Integration Development

### Mock NGO Endpoints
```bash
# Start mock NGO server for testing
python tests/mock_ngo_server.py

# Submit test impact proof
curl -X POST http://localhost:8001/webhook/impact-update \
  -H "Content-Type: application/json" \
  -d @tests/fixtures/sample_impact_proof.json
```

### Testing NGO Workflows
```python
# Test proof collection workflow
from src.agents.proof_collector import ProofCollectorAgent

agent = ProofCollectorAgent()
proof_data = {
    "donation_id": "test_donation_123",
    "ngo_id": "NGO001",
    "proof_type": "photo",
    "content": {...},
    "timestamp": "2025-01-15T10:30:00Z",
    "gps_coordinates": {"lat": 14.5995, "lng": 120.9842}
}

validation_result = agent.validate_proof(proof_data)
```

## Frontend Development (BPI VIBE Integration)

### Mock BPI VIBE Integration
```javascript
// tests/frontend/mock_vibe.js
// Simulates BPI VIBE calling ImpactSense API

class MockVibeIntegration {
    async checkForDonationPrompt(transactionData) {
        const response = await fetch('/evaluate-donation-prompt', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                transaction: transactionData,
                user_profile: this.getUserProfile()
            })
        });
        
        return response.json();
    }
    
    async submitDonation(donationData) {
        const response = await fetch('/submit-donation', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(donationData)
        });
        
        return response.json();
    }
}
```

### UI Component Testing
```bash
# Install frontend testing dependencies
npm install --save-dev jest @testing-library/react

# Run frontend tests
npm test

# Run with coverage
npm run test:coverage
```

## Monitoring & Debugging

### Application Monitoring
```python
# Add to your development environment
import sentry_sdk
from sentry_sdk.integrations.fastapi import FastApiIntegration

sentry_sdk.init(
    dsn="your_sentry_dsn",
    integrations=[FastApiIntegration()],
    traces_sample_rate=1.0,
)
```

### Performance Profiling
```bash
# Profile API endpoints
pip install py-spy

# Profile running application
py-spy record -o profile.svg --pid <process_id>

# Memory profiling
pip install memory_profiler
python -m memory_profiler src/api/main.py
```

### Database Query Analysis
```python
# Enable SQL query logging in development
import logging
logging.getLogger('sqlalchemy.engine').setLevel(logging.INFO)

# Use query profiling
from sqlalchemy import event
from sqlalchemy.engine import Engine
import time

@event.listens_for(Engine, "before_cursor_execute")
def before_cursor_execute(conn, cursor, statement, parameters, context, executemany):
    context._query_start_time = time.time()

@event.listens_for(Engine, "after_cursor_execute")  
def after_cursor_execute(conn, cursor, statement, parameters, context, executemany):
    total = time.time() - context._query_start_time
    logger.info(f"Query took {total:.4f}s: {statement}")
```

## Production Deployment

### Docker Setup
```dockerfile
# Dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Docker Compose for Local Development
```yaml
# docker-compose.yml
version: '3.8'
services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://postgres:password@db:5432/impactsense
      - REDIS_URL=redis://redis:6379
    depends_on:
      - db
      - redis
  
  db:
    image: postgres:13
    environment:
      POSTGRES_DB: impactsense
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: password
    volumes:
      - postgres_data:/var/lib/postgresql/data
  
  redis:
    image: redis:6-alpine
    
volumes:
  postgres_data:
```

### Deployment Commands
```bash
# Build and run with Docker Compose
docker-compose up --build

# Run database migrations in container
docker-compose exec api alembic upgrade head

# View logs
docker-compose logs -f api

# Scale services
docker-compose up --scale api=3
```

## Troubleshooting

### Common Issues

1. **Agent Not Generating Recommendations**
   ```python
   # Check user context data
   print(f"Wallet balance: {context.wallet_balance}")
   print(f"Days since last prompt: {context.days_since_last_prompt}")
   
   # Enable debug logging
   logging.getLogger("ImpactSense.DonationLikelyScoreAgent").setLevel(logging.DEBUG)
   ```

2. **NGO API Integration Failures**
   ```python
   # Test NGO connectivity
   import requests
   response = requests.get("https://ngo-api.example.com/health")
   print(f"NGO API Status: {response.status_code}")
   ```

3. **Database Connection Issues**
   ```bash
   # Check database connectivity
   python -c "from src.database import engine; print(engine.execute('SELECT 1').scalar())"
   ```

4. **Redis Connection Problems**
   ```bash
   # Test Redis connection
   redis-cli ping
   
   # Check Redis logs
   redis-cli monitor
   ```

### Debug Mode Configuration
```python
# src/config.py
import os

DEBUG = os.getenv("DEBUG", "false").lower() == "true"

if DEBUG:
    # Enable detailed logging
    import logging
    logging.basicConfig(level=logging.DEBUG)
    
    # Enable SQL query logging
    logging.getLogger('sqlalchemy.engine').setLevel(logging.INFO)
    
    # Disable rate limiting in development
    RATE_LIMITING_ENABLED = False
else:
    RATE_LIMITING_ENABLED = True
```

## Contributing Guidelines

### Code Style
- Use Black for code formatting
- Follow PEP 8 naming conventions
- Add type hints to all functions
- Write docstrings for all public methods
- Keep functions under 50 lines when possible

### Git Workflow
```bash
# Feature development
git checkout -b feature/agent-improvements
git commit -m "feat: improve donation likelihood scoring"
git push origin feature/agent-improvements

# Create pull request with description
# Include tests and documentation updates
```

### Testing Requirements
- All new features must include unit tests
- Integration tests for API endpoints
- End-to-end tests for critical user flows
- Minimum 80% code coverage

---

**Ready to build the future of microdonations! 🚀**

For questions or issues, check the GitHub issues or contact the development team.
