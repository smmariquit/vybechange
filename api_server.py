"""
ImpactSense API Server
FastAPI backend for the AI-powered microdonation platform
"""

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import json
import os
from datetime import datetime, timedelta
import uvicorn

# Import our simplified data generator
from generate_demo_data import SimplifiedDataGenerator

app = FastAPI(
    title="ImpactSense API",
    description="AI-powered microdonation platform for BPI VIBE",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Data models
class UserProfile(BaseModel):
    id: str
    name: str
    email: str
    wallet_balance: float
    segment: str
    location: Dict[str, str]
    preferences: Dict[str, Any]

class TransactionContext(BaseModel):
    amount: float
    category: str
    merchant: str
    location: Dict[str, str]

class DonationRequest(BaseModel):
    user_id: str
    transaction: TransactionContext
    cause_preference: Optional[str] = None

class DonationResponse(BaseModel):
    should_prompt: bool
    likelihood_score: float
    recommended_cause: Optional[Dict[str, Any]] = None
    suggested_amount: Optional[float] = None
    reasoning: List[str]

# Global data storage (in production, this would be a database)
demo_data = {
    'users': {},
    'transactions': [],
    'decisions': [],
    'metrics': {}
}

def load_demo_data():
    """Load demo data if available"""
    global demo_data
    
    try:
        if os.path.exists('data/demo_users.json'):
            with open('data/demo_users.json', 'r') as f:
                users_list = json.load(f)
                demo_data['users'] = {user['id']: user for user in users_list}
        
        if os.path.exists('data/demo_transactions.json'):
            with open('data/demo_transactions.json', 'r') as f:
                demo_data['transactions'] = json.load(f)
        
        if os.path.exists('data/demo_decisions.json'):
            with open('data/data/demo_decisions.json', 'r') as f:
                demo_data['decisions'] = json.load(f)
                
        if os.path.exists('data/demo_metrics.json'):
            with open('data/demo_metrics.json', 'r') as f:
                demo_data['metrics'] = json.load(f)
                
    except Exception as e:
        print(f"Warning: Could not load demo data: {e}")

# Simple agent implementations
class SimpleLikelihoodAgent:
    """Simplified likelihood scoring agent"""
    
    def calculate_score(self, user: Dict[str, Any], transaction: Dict[str, Any]) -> float:
        """Calculate donation likelihood score"""
        
        score = 50.0  # Base score
        
        # Wallet balance factor
        balance = user.get('wallet_balance', 1000)
        if balance > 5000:
            score += 20
        elif balance > 2000:
            score += 10
        elif balance < 500:
            score -= 15
        
        # Transaction amount factor
        amount = transaction.get('amount', 500)
        if amount > 2000:
            score += 15
        elif amount > 1000:
            score += 8
        
        # User segment factor
        segment = user.get('segment', 'new')
        if segment == 'major':
            score += 25
        elif segment == 'regular':
            score += 15
        elif segment == 'new':
            score += 5
        else:  # inactive
            score -= 20
        
        # Donation history factor
        history = user.get('donation_history', [])
        if len(history) > 10:
            score += 15
        elif len(history) > 5:
            score += 10
        elif len(history) > 0:
            score += 5
        
        # Recent donation penalty
        if history:
            recent_donations = [
                d for d in history 
                if (datetime.now() - datetime.fromisoformat(d['timestamp'])).days < 7
            ]
            if recent_donations:
                score -= len(recent_donations) * 10
        
        return max(0, min(100, score))

class SimpleCauseRecommender:
    """Simplified cause recommendation agent"""
    
    def __init__(self):
        self.causes = [
            {'name': 'Education', 'keywords': ['school', 'education', 'book'], 'base_score': 0.8},
            {'name': 'Environment', 'keywords': ['green', 'eco', 'nature'], 'base_score': 0.7},
            {'name': 'Health', 'keywords': ['health', 'medical', 'hospital'], 'base_score': 0.9},
            {'name': 'Disaster Relief', 'keywords': ['emergency', 'disaster'], 'base_score': 0.6},
            {'name': 'Poverty', 'keywords': ['poor', 'community'], 'base_score': 0.7}
        ]
    
    def recommend(self, user: Dict[str, Any], transaction: Dict[str, Any]) -> Dict[str, Any]:
        """Recommend cause based on user and transaction context"""
        
        user_prefs = user.get('preferences', {}).get('causes', [])
        
        # If user has preferences, prioritize them
        if user_prefs:
            primary_cause = user_prefs[0]
            relevance_score = 0.9
        else:
            # Default recommendation based on transaction category
            category = transaction.get('category', 'general')
            if category in ['health', 'medical']:
                primary_cause = 'Health'
                relevance_score = 0.85
            elif category in ['education', 'books']:
                primary_cause = 'Education'
                relevance_score = 0.85
            else:
                primary_cause = 'Environment'
                relevance_score = 0.7
        
        return {
            'primary_recommendation': {
                'cause': primary_cause,
                'ngo_id': f'ngo_{primary_cause.lower()}_001',
                'relevance_score': relevance_score,
                'description': f'Support {primary_cause.lower()} initiatives in your area'
            },
            'alternatives': [
                {
                    'cause': cause['name'],
                    'ngo_id': f'ngo_{cause["name"].lower()}_001',
                    'relevance_score': cause['base_score'] * 0.8,
                    'description': f'Support {cause["name"].lower()} initiatives'
                } for cause in self.causes[:3] if cause['name'] != primary_cause
            ]
        }

class SimpleAmountOptimizer:
    """Simplified amount optimization agent"""
    
    def optimize(self, user: Dict[str, Any], transaction: Dict[str, Any], cause: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize donation amount"""
        
        # Base amount calculation
        balance = user.get('wallet_balance', 1000)
        tx_amount = transaction.get('amount', 500)
        
        # Calculate base amount (0.5-2% of transaction)
        base_amount = tx_amount * 0.01
        
        # Adjust based on user segment
        segment = user.get('segment', 'new')
        if segment == 'major':
            multiplier = 2.0
        elif segment == 'regular':
            multiplier = 1.5
        elif segment == 'new':
            multiplier = 0.8
        else:  # inactive
            multiplier = 0.5
        
        primary_amount = base_amount * multiplier
        
        # Respect user's max donation preference
        max_donation = user.get('preferences', {}).get('max_donation', 50)
        primary_amount = min(primary_amount, max_donation)
        
        # Ensure minimum donation
        primary_amount = max(primary_amount, 1.0)
        
        # Generate alternatives
        alternatives = [
            round(primary_amount * 0.5, 2),
            round(primary_amount * 0.75, 2),
            round(primary_amount * 1.25, 2)
        ]
        
        return {
            'primary_amount': round(primary_amount, 2),
            'alternatives': alternatives,
            'optimization_score': 0.85,
            'reasoning': [
                f'Based on {segment} user segment',
                f'Optimized for ₱{tx_amount:.0f} transaction',
                'Respects user preferences'
            ]
        }

# Initialize agents
likelihood_agent = SimpleLikelihoodAgent()
cause_agent = SimpleCauseRecommender()
amount_agent = SimpleAmountOptimizer()

@app.on_event("startup")
async def startup_event():
    """Load data on startup"""
    load_demo_data()
    
    # Generate demo data if none exists
    if not demo_data['users']:
        print("🔄 Generating demo data...")
        generator = SimplifiedDataGenerator()
        generator.save_data()
        load_demo_data()
        print("✅ Demo data loaded")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "ImpactSense API",
        "version": "1.0.0",
        "status": "active",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "data_loaded": {
            "users": len(demo_data['users']),
            "transactions": len(demo_data['transactions']),
            "decisions": len(demo_data['decisions'])
        }
    }

@app.post("/api/v1/donation/evaluate", response_model=DonationResponse)
async def evaluate_donation_opportunity(request: DonationRequest):
    """Evaluate if user should be prompted for donation"""
    
    # Get user profile
    user = demo_data['users'].get(request.user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Calculate likelihood score
    likelihood_score = likelihood_agent.calculate_score(user, request.transaction.dict())
    
    # Determine if should prompt (threshold: 60%)
    should_prompt = likelihood_score >= 60
    
    recommended_cause = None
    suggested_amount = None
    reasoning = [f"Likelihood score: {likelihood_score:.1f}%"]
    
    if should_prompt:
        # Get cause recommendation
        cause_rec = cause_agent.recommend(user, request.transaction.dict())
        recommended_cause = cause_rec['primary_recommendation']
        
        # Get amount suggestion
        amount_rec = amount_agent.optimize(user, request.transaction.dict(), recommended_cause)
        suggested_amount = amount_rec['primary_amount']
        
        reasoning.extend([
            f"Recommended cause: {recommended_cause['cause']}",
            f"Suggested amount: ₱{suggested_amount:.2f}",
            "User profile indicates good donation potential"
        ])
    else:
        reasoning.append("Below threshold for prompting")
    
    return DonationResponse(
        should_prompt=should_prompt,
        likelihood_score=likelihood_score,
        recommended_cause=recommended_cause,
        suggested_amount=suggested_amount,
        reasoning=reasoning
    )

@app.get("/api/v1/users/{user_id}")
async def get_user_profile(user_id: str):
    """Get user profile"""
    user = demo_data['users'].get(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user

@app.get("/api/v1/metrics/dashboard")
async def get_dashboard_metrics():
    """Get dashboard metrics"""
    
    # Calculate real-time metrics from demo data
    total_users = len(demo_data['users'])
    recent_transactions = [
        t for t in demo_data['transactions']
        if (datetime.now() - datetime.fromisoformat(t['timestamp'])).days <= 1
    ]
    
    donations_today = sum(1 for t in recent_transactions if t.get('donation_made', False))
    amount_raised_today = sum(
        5.0 + (hash(t['id']) % 20)  # Simulate donation amounts
        for t in recent_transactions if t.get('donation_made', False)
    )
    
    prompts_shown = sum(1 for t in recent_transactions if t.get('prompt_shown', False))
    conversion_rate = donations_today / max(1, prompts_shown)
    
    return {
        'timestamp': datetime.now().isoformat(),
        'realtime_metrics': {
            'active_users': len(recent_transactions),
            'donations_today': donations_today,
            'amount_raised_today': round(amount_raised_today, 2),
            'conversion_rate_today': round(conversion_rate, 3),
            'total_users': total_users,
            'system_health': 0.98
        },
        'agent_performance': {
            'likelihood_agent': {
                'decisions': len([d for d in demo_data['decisions'] if 'LikelyScore' in d.get('agent_name', '')]),
                'avg_score': 72.5,
                'conversion_rate': 0.28
            },
            'cause_recommender': {
                'recommendations': len([d for d in demo_data['decisions'] if 'Recommender' in d.get('agent_name', '')]),
                'acceptance_rate': 0.65,
                'avg_relevance': 0.82
            }
        }
    }

@app.get("/api/v1/analytics/causes")
async def get_cause_analytics():
    """Get cause performance analytics"""
    
    causes = ['Education', 'Environment', 'Health', 'Disaster Relief', 'Poverty']
    cause_data = []
    
    for cause in causes:
        # Simulate cause performance data
        base_performance = hash(cause) % 100
        cause_data.append({
            'cause': cause,
            'donations_count': 800 + (base_performance * 10),
            'total_amount': 15000 + (base_performance * 200),
            'avg_amount': 12 + (base_performance % 20),
            'success_rate': 0.2 + (base_performance % 30) / 100
        })
    
    return {'causes': cause_data}

@app.post("/api/v1/admin/generate-data")
async def generate_demo_data_endpoint(background_tasks: BackgroundTasks):
    """Generate new demo data (admin endpoint)"""
    
    def generate_data():
        generator = SimplifiedDataGenerator()
        generator.save_data()
        load_demo_data()
    
    background_tasks.add_task(generate_data)
    
    return {
        'message': 'Demo data generation started',
        'status': 'processing'
    }

@app.get("/api/v1/admin/system-status")
async def get_system_status():
    """Get system status (admin endpoint)"""
    
    return {
        'timestamp': datetime.now().isoformat(),
        'system_health': 0.98,
        'services': {
            'api_gateway': 'healthy',
            'ml_models': 'healthy',
            'database': 'healthy',
            'cache': 'warning',
            'bpi_integration': 'healthy'
        },
        'performance': {
            'avg_response_time': 145,
            'requests_per_minute': 250,
            'error_rate': 0.002
        },
        'data_status': {
            'users_loaded': len(demo_data['users']),
            'transactions_loaded': len(demo_data['transactions']),
            'decisions_logged': len(demo_data['decisions'])
        }
    }

if __name__ == "__main__":
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
