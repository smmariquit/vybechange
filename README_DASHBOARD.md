# 🎯 ImpactSense - AI-Powered Microdonation Platform

![ImpactSense Dashboard](https://img.shields.io/badge/Status-Active-green) ![Python](https://img.shields.io/badge/Python-3.13-blue) ![UV](https://img.shields.io/badge/UV-Package%20Manager-purple)

ImpactSense is an intelligent microdonation platform that integrates with BPI's VIBE payment system to encourage charitable giving through AI-powered recommendations and real-time analytics.

## 🚀 Quick Start

### Prerequisites
- Python 3.13+
- UV package manager
- Git

### One-Command Setup

```bash
# Install dependencies and start the complete platform
uv sync && uv run start.py
```

This automatically:
- ✅ Installs all dependencies
- 📊 Generates demo data (1K users, 10K transactions)
- 🚀 Starts API server (port 8000)
- 📈 Launches dashboard (port 8501)
- 🌐 Opens browser to both services

### Access Points
- **📊 Dashboard**: http://localhost:8501
- **🔗 API**: http://localhost:8000
- **📚 API Docs**: http://localhost:8000/docs

## 📊 Dashboard Features

### 🎯 Overview Tab
![Dashboard Overview](docs/images/overview-tab.png)

- **Real-time Metrics**: Active users, daily donations, conversion rates
- **Performance Trends**: Historical donation patterns and growth
- **Conversion Funnel**: User journey from reach to recurring donations

### 🤖 Agent Performance Tab
- **AI Agent Analytics**: Compare ML vs rule-based agents
- **Conversion Tracking**: Detailed metrics for each decision agent
- **Performance Trends**: Historical analysis of agent effectiveness

### 👥 User Insights Tab
- **User Segmentation**: Distribution across different user types
- **Engagement Analysis**: Behavior patterns and retention metrics
- **Geographic Patterns**: Regional donation and user activity

### 🎯 Cause Analytics Tab
- **Cause Performance**: Success rates by charity category
- **Impact Visualization**: Interactive charts of donation distribution
- **Geographic Impact**: Where donations are making a difference

### 🔧 System Monitoring Tab
- **Service Health**: Real-time status of all components
- **Performance Metrics**: Response times, error rates, system load
- **Recent Alerts**: System notifications and warnings

### 🗄️ Data Management Tab
- **Synthetic Data**: Generate test datasets for development
- **ML Training**: Retrain AI models with new data
- **Data Quality**: Monitor completeness and accuracy metrics

## 🔗 API Capabilities

### Core Donation Logic
```python
# Evaluate donation opportunity
POST /api/v1/donation/evaluate
{
  "user_id": "user_1234",
  "transaction": {
    "amount": 1500.00,
    "category": "groceries",
    "location": {"city": "Manila", "region": "NCR"}
  }
}

# Response
{
  "should_prompt": true,
  "likelihood_score": 78.5,
  "recommended_cause": {
    "cause": "Education",
    "ngo_id": "edu_manila_001",
    "relevance_score": 0.92
  },
  "suggested_amount": 15.00,
  "reasoning": ["High transaction amount", "Strong local match"]
}
```

### Analytics & Monitoring
```python
# Real-time dashboard metrics
GET /api/v1/metrics/dashboard

# Cause performance analytics  
GET /api/v1/analytics/causes

# System health status
GET /api/v1/admin/system-status
```

## 🏗️ Architecture

### AI Agent System
```
User Transaction → Context Analysis → AI Decision → Recommendation → Feedback Loop
```

1. **Likelihood Agent**: Scores donation probability (0-100%)
2. **Cause Recommender**: Matches users with relevant local causes
3. **Amount Optimizer**: Suggests optimal donation amounts
4. **Performance Tracker**: Monitors and improves agent decisions

### Technology Stack
- **Backend**: FastAPI with async processing
- **Frontend**: Streamlit with Plotly visualizations
- **AI/ML**: Scikit-learn (extensible to TensorFlow/PyTorch)
- **Data**: JSON demo data (extensible to PostgreSQL/MongoDB)
- **Package Management**: UV for fast dependency resolution

## 🎛️ Configuration

### Agent Settings (Dashboard Sidebar)
- **Prompt Threshold**: Minimum likelihood score (0-100%)
- **Daily Prompt Limit**: Max prompts per user per day
- **Agent Toggles**: Enable/disable specific AI components

### Data Generation
- **User Count**: 100-10,000 synthetic users
- **Transaction Volume**: 1K-100K transactions
- **Time Range**: Historical data period (1-365 days)

## 📈 Key Metrics

### Performance KPIs
- **Conversion Rate**: 25-35% (prompts → donations)
- **Response Time**: <200ms API response
- **User Engagement**: Daily active users and retention
- **System Health**: 98%+ uptime target

### Agent Effectiveness
- **Likelihood Agent**: Prediction accuracy vs actual donations
- **Cause Recommender**: Relevance score and acceptance rate
- **Amount Optimizer**: Suggested vs actual donation amounts

## 🛠️ Development

### Project Structure
```
impactsense/
├── 🚀 start.py              # One-command platform launcher
├── 📊 dashboard.py          # Streamlit analytics dashboard  
├── 🔗 api_server.py        # FastAPI backend server
├── 📊 generate_demo_data.py # Synthetic data generator
├── ⚙️ pyproject.toml       # UV dependency configuration
├── 📁 src/                 # Core source modules
│   ├── agents/            # AI agent implementations
│   ├── utils/             # Utility functions
│   └── models/            # Data models and schemas
├── 📊 data/               # Generated demo datasets
├── 🧪 tests/              # Comprehensive test suites
└── 📚 docs/               # Documentation and guides
```

### Adding Features

1. **New AI Agent**: Implement in `src/agents/`
2. **API Endpoint**: Add to `api_server.py`
3. **Dashboard Component**: Extend `dashboard.py`
4. **Tests**: Add comprehensive tests in `tests/`

### Running Tests
```bash
# Run all tests
uv run pytest tests/ -v

# Run specific test categories
uv run pytest tests/test_agents.py -v
uv run pytest tests/test_api.py -v
```

## 🎯 Use Cases

### For Developers
- **Algorithm Development**: Test AI recommendation logic
- **Performance Optimization**: Monitor response times and accuracy
- **A/B Testing**: Compare different approaches side-by-side

### For Product Managers
- **Feature Impact**: Measure effectiveness of new capabilities
- **User Behavior**: Understand donation patterns and preferences
- **Business Metrics**: Track conversion rates and user engagement

### For Data Scientists
- **Model Comparison**: ML vs rule-based performance analysis
- **Feature Engineering**: Identify important prediction variables
- **Data Quality**: Monitor synthetic vs real data patterns

## 🛡️ Privacy & Security

- **Demo Data Only**: No real user information stored
- **Local Development**: Runs entirely on your machine
- **API Validation**: Request/response validation and rate limiting
- **Secure Architecture**: Production-ready security patterns

## 📊 Demo Data

The system generates realistic synthetic data:

### Users (1,000 default)
- **Segments**: New, Regular, Major, Inactive donors
- **Profiles**: Wallet balance, donation history, preferences
- **Geography**: Philippines focus with major cities

### Transactions (10,000 default)  
- **Categories**: Groceries, dining, transportation, shopping
- **Amounts**: ₱50-₱5,000 realistic transaction range
- **Timing**: 30-day rolling window with realistic patterns

### Donation History
- **Amounts**: ₱1-₱50 per donation
- **Causes**: Education, Environment, Health, Disaster Relief
- **Success Rates**: 20-40% based on user segments

## 🔧 Troubleshooting

### Common Issues

**Port Conflicts**
```bash
# Find processes using ports 8000/8501
netstat -ano | findstr :8000
netstat -ano | findstr :8501

# Kill conflicting processes
taskkill /PID <process_id> /F
```

**Dependencies**
```bash
# Reinstall all dependencies
uv sync --reload

# Check Python version
python --version  # Should be 3.13+
```

**Demo Data**
```bash
# Regenerate fresh demo data
uv run generate_demo_data.py

# Check data directory
dir data/  # Should contain JSON files
```

### Getting Help
1. **API Issues**: Check http://localhost:8000/docs
2. **Dashboard Problems**: Review console output
3. **Performance**: Monitor system health tab
4. **Data Issues**: Regenerate demo data

## 🚀 Production Deployment

### Docker Setup
```bash
# Build container
docker build -t impactsense:latest .

# Run with compose
docker-compose up -d
```

### Environment Variables
```env
# Production settings
ENVIRONMENT=production
API_HOST=0.0.0.0
API_PORT=8000
DASHBOARD_PORT=8501

# Database (when ready)
DATABASE_URL=postgresql://user:pass@localhost/impactsense
REDIS_URL=redis://localhost:6379

# BPI Integration
BPI_VIBE_API_URL=https://api.bpi.com/vibe
BPI_VIBE_API_KEY=your_production_key
```

## 🤝 Contributing

1. **Fork the repository**
2. **Create feature branch**: `git checkout -b feature/amazing-feature`
3. **Add tests**: Ensure comprehensive test coverage
4. **Commit changes**: `git commit -m 'Add amazing feature'`
5. **Push to branch**: `git push origin feature/amazing-feature`
6. **Open Pull Request**: Describe changes and impact

### Development Guidelines
- Follow PEP 8 style guidelines
- Add type hints for all functions
- Write comprehensive tests
- Update documentation for changes
- Ensure dashboard components are responsive

## 📈 Roadmap

### Phase 1 ✅ (Current)
- Core AI agent implementation
- Interactive dashboard with real-time metrics
- Synthetic data generation system
- FastAPI backend with full documentation

### Phase 2 🔄 (Next)
- Machine learning model training pipeline
- Advanced user segmentation algorithms
- Real BPI VIBE API integration
- Enhanced impact tracking and reporting

### Phase 3 ⏳ (Future)
- Predictive donation modeling
- Mobile app integration
- Corporate partnership features
- International expansion support

## 📜 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🌟 Acknowledgments

- **BPI Innovation Team**: Vision and partnership
- **Streamlit Community**: Amazing dashboard framework
- **FastAPI Team**: High-performance async API framework
- **UV Project**: Next-generation Python package management
- **Open Source Community**: Foundational libraries and tools

---

**Ready to make a difference with data-driven donations? 🎯**

```bash
uv run start.py
```
