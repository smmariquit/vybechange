"""
Simple Web Dashboard for ImpactSense
Uses FastAPI to serve both API and a simple HTML dashboard
"""

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import json
import os
from datetime import datetime
from generate_demo_data import SimplifiedDataGenerator

app = FastAPI(title="ImpactSense Dashboard", description="AI-powered microdonation platform")

# Load demo data
demo_data = {
    'users': {},
    'transactions': [],
    'decisions': [],
    'metrics': {}
}

def load_demo_data():
    """Load demo data from JSON files"""
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
            with open('data/demo_decisions.json', 'r') as f:
                demo_data['decisions'] = json.load(f)
                
        if os.path.exists('data/demo_metrics.json'):
            with open('data/demo_metrics.json', 'r') as f:
                demo_data['metrics'] = json.load(f)
                
    except Exception as e:
        print(f"Warning: Could not load demo data: {e}")

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

@app.get("/", response_class=HTMLResponse)
async def dashboard_home():
    """Serve the main dashboard"""
    
    # Calculate metrics
    total_users = len(demo_data['users'])
    total_transactions = len(demo_data['transactions'])
    total_decisions = len(demo_data['decisions'])
    
    # Recent transactions (last 7 days)
    recent_transactions = [t for t in demo_data['transactions'][:100]]  # Show first 100
    donations_made = sum(1 for t in recent_transactions if t.get('donation_made', False))
    prompts_shown = sum(1 for t in recent_transactions if t.get('prompt_shown', False))
    
    conversion_rate = (donations_made / max(1, prompts_shown)) * 100
    
    # Generate HTML dashboard
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>🎯 ImpactSense Dashboard</title>
        <style>
            body {{
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
                margin: 0;
                padding: 20px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                color: #333;
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
                background: white;
                border-radius: 20px;
                box-shadow: 0 20px 40px rgba(0,0,0,0.1);
                overflow: hidden;
            }}
            .header {{
                background: linear-gradient(90deg, #1f4e79 0%, #2e7d32 100%);
                color: white;
                padding: 2rem;
                text-align: center;
            }}
            .header h1 {{
                margin: 0;
                font-size: 2.5rem;
                font-weight: 700;
            }}
            .header p {{
                margin: 0.5rem 0 0 0;
                opacity: 0.9;
                font-size: 1.1rem;
            }}
            .metrics {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 2rem;
                padding: 2rem;
                background: #f8fafc;
            }}
            .metric-card {{
                background: white;
                padding: 1.5rem;
                border-radius: 12px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.05);
                border-left: 4px solid #2e7d32;
                transition: transform 0.2s ease;
            }}
            .metric-card:hover {{
                transform: translateY(-2px);
                box-shadow: 0 8px 15px rgba(0,0,0,0.1);
            }}
            .metric-value {{
                font-size: 2.5rem;
                font-weight: bold;
                color: #2e7d32;
                margin: 0;
            }}
            .metric-label {{
                font-size: 0.9rem;
                color: #64748b;
                text-transform: uppercase;
                letter-spacing: 0.05em;
                margin: 0.5rem 0 0 0;
            }}
            .content {{
                padding: 2rem;
            }}
            .section {{
                margin-bottom: 3rem;
            }}
            .section h2 {{
                color: #1e293b;
                margin-bottom: 1rem;
                font-size: 1.5rem;
            }}
            .table {{
                width: 100%;
                border-collapse: collapse;
                background: white;
                border-radius: 8px;
                overflow: hidden;
                box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            }}
            .table th, .table td {{
                padding: 1rem;
                text-align: left;
                border-bottom: 1px solid #e2e8f0;
            }}
            .table th {{
                background: #f1f5f9;
                font-weight: 600;
                color: #374151;
            }}
            .table tr:hover {{
                background: #f8fafc;
            }}
            .status-active {{
                color: #059669;
                font-weight: 600;
            }}
            .status-pending {{
                color: #d97706;
                font-weight: 600;
            }}
            .badge {{
                display: inline-block;
                padding: 0.25rem 0.75rem;
                border-radius: 9999px;
                font-size: 0.75rem;
                font-weight: 600;
                text-transform: uppercase;
                letter-spacing: 0.05em;
            }}
            .badge-success {{
                background: #dcfce7;
                color: #166534;
            }}
            .badge-warning {{
                background: #fef3c7;
                color: #92400e;
            }}
            .badge-info {{
                background: #dbeafe;
                color: #1e40af;
            }}
            .api-links {{
                background: #f1f5f9;
                padding: 1.5rem;
                border-radius: 8px;
                margin-top: 2rem;
            }}
            .api-links h3 {{
                margin-top: 0;
                color: #374151;
            }}
            .api-links a {{
                color: #2563eb;
                text-decoration: none;
                margin-right: 1rem;
                font-weight: 500;
            }}
            .api-links a:hover {{
                text-decoration: underline;
            }}
            .refresh-btn {{
                background: #2e7d32;
                color: white;
                border: none;
                padding: 0.75rem 1.5rem;
                border-radius: 8px;
                font-weight: 600;
                cursor: pointer;
                transition: background 0.2s ease;
            }}
            .refresh-btn:hover {{
                background: #1b5e20;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🎯 ImpactSense Dashboard</h1>
                <p>AI-Powered Microdonation Platform Analytics</p>
                <button class="refresh-btn" onclick="window.location.reload()">🔄 Refresh Data</button>
            </div>
            
            <div class="metrics">
                <div class="metric-card">
                    <div class="metric-value">{total_users:,}</div>
                    <div class="metric-label">Total Users</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{total_transactions:,}</div>
                    <div class="metric-label">Transactions</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{donations_made}</div>
                    <div class="metric-label">Donations Made</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{conversion_rate:.1f}%</div>
                    <div class="metric-label">Conversion Rate</div>
                </div>
            </div>
            
            <div class="content">
                <div class="section">
                    <h2>📊 Recent Activity</h2>
                    <table class="table">
                        <thead>
                            <tr>
                                <th>Transaction ID</th>
                                <th>Amount</th>
                                <th>Category</th>
                                <th>Prompt Shown</th>
                                <th>Donation Made</th>
                                <th>Status</th>
                            </tr>
                        </thead>
                        <tbody>
    """
    
    # Add recent transactions to table
    for transaction in recent_transactions[:10]:  # Show only first 10
        prompt_shown = "✅" if transaction.get('prompt_shown', False) else "❌"
        donation_made = "✅" if transaction.get('donation_made', False) else "❌"
        
        if transaction.get('donation_made', False):
            status = '<span class="badge badge-success">Donated</span>'
        elif transaction.get('prompt_shown', False):
            status = '<span class="badge badge-warning">Prompted</span>'
        else:
            status = '<span class="badge badge-info">No Prompt</span>'
            
        html_content += f"""
                            <tr>
                                <td>{transaction['id']}</td>
                                <td>₱{transaction['amount']:,.2f}</td>
                                <td>{transaction['category'].title()}</td>
                                <td>{prompt_shown}</td>
                                <td>{donation_made}</td>
                                <td>{status}</td>
                            </tr>
        """
    
    html_content += f"""
                        </tbody>
                    </table>
                </div>
                
                <div class="section">
                    <h2>🤖 AI Agent Performance</h2>
                    <table class="table">
                        <thead>
                            <tr>
                                <th>Agent</th>
                                <th>Decisions</th>
                                <th>Performance</th>
                                <th>Status</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr>
                                <td>Donation Likelihood Agent</td>
                                <td>{total_decisions:,}</td>
                                <td class="status-active">85% Accuracy</td>
                                <td><span class="badge badge-success">Active</span></td>
                            </tr>
                            <tr>
                                <td>Cause Recommender</td>
                                <td>{total_decisions:,}</td>
                                <td class="status-active">78% Relevance</td>
                                <td><span class="badge badge-success">Active</span></td>
                            </tr>
                            <tr>
                                <td>Amount Optimizer</td>
                                <td>{total_decisions:,}</td>
                                <td class="status-pending">72% Accuracy</td>
                                <td><span class="badge badge-warning">Training</span></td>
                            </tr>
                        </tbody>
                    </table>
                </div>
                
                <div class="api-links">
                    <h3>🔗 API Endpoints</h3>
                    <a href="/docs" target="_blank">📚 API Documentation</a>
                    <a href="/api/v1/metrics/dashboard" target="_blank">📊 Metrics API</a>
                    <a href="/api/v1/analytics/causes" target="_blank">🎯 Cause Analytics</a>
                    <a href="/health" target="_blank">❤️ Health Check</a>
                </div>
                
                <div style="text-align: center; margin-top: 2rem; color: #64748b;">
                    <p>Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                    <p>ImpactSense v1.0.0 • Built with FastAPI</p>
                </div>
            </div>
        </div>
    </body>
    </html>
    """
    
    return html_content

# Include API endpoints from api_server.py
from api_server import *

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
