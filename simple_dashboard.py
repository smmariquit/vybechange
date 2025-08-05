"""
Simple HTTP Server for ImpactSense Dashboard Demo
"""

import http.server
import socketserver
import json
import os
from urllib.parse import urlparse, parse_qs
from datetime import datetime

class ImpactSenseHandler(http.server.SimpleHTTPRequestHandler):
    
    def do_GET(self):
        """Handle GET requests"""
        
        parsed_path = urlparse(self.path)
        path = parsed_path.path
        
        if path == '/':
            self.serve_dashboard()
        elif path == '/api/health':
            self.serve_health()
        elif path == '/api/metrics':
            self.serve_metrics()
        else:
            self.send_error(404, "Not Found")
    
    def serve_dashboard(self):
        """Serve the main dashboard HTML"""
        
        # Load demo data
        try:
            with open('data/demo_users.json', 'r') as f:
                users = json.load(f)
            with open('data/demo_transactions.json', 'r') as f:
                transactions = json.load(f)
        except:
            users = []
            transactions = []
        
        total_users = len(users)
        total_transactions = len(transactions)
        donations_made = sum(1 for t in transactions if t.get('donation_made', False))
        prompts_shown = sum(1 for t in transactions if t.get('prompt_shown', False))
        conversion_rate = (donations_made / max(1, prompts_shown)) * 100
        
        html = f"""<!DOCTYPE html>
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
        }}
        .metric-value {{
            font-size: 2.5rem;
            font-weight: bold;
            color: #2e7d32;
        }}
        .metric-label {{
            color: #64748b;
            text-transform: uppercase;
            font-size: 0.9rem;
        }}
        .content {{
            padding: 2rem;
        }}
        .section {{
            margin-bottom: 2rem;
        }}
        .section h2 {{
            color: #1e293b;
            margin-bottom: 1rem;
        }}
        .badge {{
            padding: 0.25rem 0.75rem;
            border-radius: 9999px;
            font-size: 0.75rem;
            font-weight: 600;
        }}
        .badge-success {{
            background: #dcfce7;
            color: #166534;
        }}
        .feature-list {{
            list-style: none;
            padding: 0;
        }}
        .feature-list li {{
            padding: 0.5rem 0;
            border-bottom: 1px solid #e2e8f0;
        }}
        .feature-list li:last-child {{
            border-bottom: none;
        }}
        .api-links {{
            background: #f1f5f9;
            padding: 1.5rem;
            border-radius: 8px;
            margin-top: 2rem;
        }}
        .api-links a {{
            color: #2563eb;
            text-decoration: none;
            margin-right: 1rem;
        }}
        .refresh-btn {{
            background: #2e7d32;
            color: white;
            border: none;
            padding: 0.75rem 1.5rem;
            border-radius: 8px;
            font-weight: 600;
            cursor: pointer;
            margin-top: 1rem;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎯 ImpactSense Dashboard</h1>
            <p>AI-Powered Microdonation Platform for BPI VIBE</p>
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
                <h2>🤖 AI Agents Status</h2>
                <ul class="feature-list">
                    <li>📊 <strong>Donation Likelihood Agent</strong> - <span class="badge badge-success">Active</span> - Analyzing user behavior patterns</li>
                    <li>🎯 <strong>Cause Recommender Agent</strong> - <span class="badge badge-success">Active</span> - Matching users with local causes</li>
                    <li>💰 <strong>Amount Optimizer Agent</strong> - <span class="badge badge-success">Active</span> - Optimizing donation suggestions</li>
                    <li>📈 <strong>Performance Tracker</strong> - <span class="badge badge-success">Active</span> - Monitoring system effectiveness</li>
                </ul>
            </div>
            
            <div class="section">
                <h2>🎯 Key Features</h2>
                <ul class="feature-list">
                    <li>🧠 <strong>Smart Prompting</strong> - Context-aware donation timing based on transaction patterns</li>
                    <li>📍 <strong>Local Impact</strong> - Prioritizes causes and NGOs in user's immediate area</li>
                    <li>⚡ <strong>Real-time Processing</strong> - Sub-200ms response times for transaction-time decisions</li>
                    <li>📊 <strong>Comprehensive Analytics</strong> - Live monitoring of donations, conversions, and impact</li>
                    <li>🔄 <strong>Continuous Learning</strong> - AI agents improve through user interaction feedback</li>
                </ul>
            </div>
            
            <div class="section">
                <h2>📈 Platform Capabilities</h2>
                <ul class="feature-list">
                    <li>🎨 <strong>Interactive Dashboard</strong> - Real-time metrics and performance monitoring</li>
                    <li>🔌 <strong>API Integration</strong> - RESTful endpoints for BPI VIBE system integration</li>
                    <li>🗄️ <strong>Synthetic Data</strong> - Comprehensive demo datasets for development and testing</li>
                    <li>🧪 <strong>Testing Framework</strong> - Complete test suites for all agent functionality</li>
                    <li>📚 <strong>Documentation</strong> - Comprehensive guides and API documentation</li>
                </ul>
            </div>
            
            <div class="api-links">
                <h3>🔗 Available Endpoints</h3>
                <a href="/api/health">❤️ Health Check</a>
                <a href="/api/metrics">📊 System Metrics</a>
                <p><strong>Note:</strong> Full API server with interactive documentation available when running the complete FastAPI application.</p>
            </div>
            
            <div style="text-align: center; margin-top: 2rem; color: #64748b;">
                <p>Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p><strong>ImpactSense v1.0.0</strong> • AI-Powered Microdonations • Built with Python & UV</p>
                <p>🚀 <strong>Ready for BPI VIBE Integration</strong> 🚀</p>
            </div>
        </div>
    </div>
</body>
</html>"""
        
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(html.encode())
    
    def serve_health(self):
        """Serve health check endpoint"""
        health_data = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "service": "ImpactSense Demo",
            "version": "1.0.0"
        }
        
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(health_data, indent=2).encode())
    
    def serve_metrics(self):
        """Serve metrics endpoint"""
        try:
            with open('data/demo_users.json', 'r') as f:
                users = json.load(f)
            with open('data/demo_transactions.json', 'r') as f:
                transactions = json.load(f)
        except:
            users = []
            transactions = []
        
        metrics_data = {
            "timestamp": datetime.now().isoformat(),
            "total_users": len(users),
            "total_transactions": len(transactions),
            "donations_made": sum(1 for t in transactions if t.get('donation_made', False)),
            "prompts_shown": sum(1 for t in transactions if t.get('prompt_shown', False)),
            "system_health": 0.98,
            "agents_active": 4
        }
        
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(metrics_data, indent=2).encode())

def main():
    """Start the simple HTTP server"""
    PORT = 8000
    
    print(f"🎯 Starting ImpactSense Dashboard Demo...")
    print(f"📊 Dashboard: http://localhost:{PORT}")
    print(f"❤️ Health Check: http://localhost:{PORT}/api/health")
    print(f"📊 Metrics API: http://localhost:{PORT}/api/metrics")
    print(f"⌨️ Press Ctrl+C to stop")
    
    with socketserver.TCPServer(("", PORT), ImpactSenseHandler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print(f"\n🛑 Shutting down dashboard...")
            httpd.shutdown()

if __name__ == "__main__":
    main()
