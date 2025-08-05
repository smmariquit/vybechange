"""
ImpactSense Startup Script
Easy way to launch the complete ImpactSense platform
"""

import subprocess
import sys
import os
import time
import webbrowser
from pathlib import Path


def check_dependencies():
    """Check if required packages are installed"""
    
    required_packages = [
        'streamlit', 'fastapi', 'uvicorn', 'pandas', 'numpy', 'plotly'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing packages: {', '.join(missing_packages)}")
        print(f"📦 Install with: uv add {' '.join(missing_packages)}")
        return False
    
    print("✅ All dependencies are installed")
    return True


def generate_demo_data():
    """Generate demo data if it doesn't exist"""
    
    data_dir = Path('data')
    demo_files = [
        'demo_users.json',
        'demo_transactions.json', 
        'demo_decisions.json',
        'demo_metrics.json'
    ]
    
    if not data_dir.exists() or not all((data_dir / f).exists() for f in demo_files):
        print("🔄 Generating demo data...")
        
        try:
            result = subprocess.run([
                sys.executable, 'generate_demo_data.py'
            ], capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                print("✅ Demo data generated successfully")
                return True
            else:
                print(f"❌ Error generating demo data: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            print("⏱️ Demo data generation timed out")
            return False
        except Exception as e:
            print(f"❌ Error running demo data generator: {e}")
            return False
    else:
        print("✅ Demo data already exists")
        return True


def start_api_server():
    """Start the FastAPI server"""
    
    print("🚀 Starting API server...")
    
    try:
        # Start API server in background
        api_process = subprocess.Popen([
            sys.executable, '-m', 'uvicorn', 
            'api_server:app',
            '--host', '0.0.0.0',
            '--port', '8000',
            '--reload'
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait a moment for server to start
        time.sleep(3)
        
        # Check if server is running
        if api_process.poll() is None:
            print("✅ API server started on http://localhost:8000")
            return api_process
        else:
            stdout, stderr = api_process.communicate()
            print(f"❌ API server failed to start: {stderr.decode()}")
            return None
            
    except Exception as e:
        print(f"❌ Error starting API server: {e}")
        return None


def start_dashboard():
    """Start the Streamlit dashboard"""
    
    print("📊 Starting Streamlit demo dashboard...")
    
    try:
        # Start Streamlit dashboard with Python module syntax
        dashboard_process = subprocess.Popen([
            sys.executable, '-c', 
            'import streamlit.web.cli as stcli; import sys; sys.argv = ["streamlit", "run", "streamlit_demo.py", "--server.port", "8501", "--server.address", "0.0.0.0"]; stcli.main()'
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait a moment for dashboard to start
        time.sleep(8)
        
        # Check if dashboard is running
        if dashboard_process.poll() is None:
            print("✅ Streamlit demo started on http://localhost:8501")
            return dashboard_process
        else:
            print("⚠️ Streamlit not available, starting simple dashboard...")
            return start_simple_dashboard()
            
    except Exception as e:
        print(f"⚠️ Streamlit error: {e}")
        print("🔄 Starting simple dashboard instead...")
        return start_simple_dashboard()

def start_simple_dashboard():
    """Start the simple HTTP dashboard as fallback"""
    
    try:
        dashboard_process = subprocess.Popen([
            sys.executable, 'simple_dashboard.py'
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        time.sleep(3)
        
        if dashboard_process.poll() is None:
            print("✅ Simple dashboard started on http://localhost:8000")
            return dashboard_process
        else:
            print("❌ Simple dashboard failed to start")
            return None
            
    except Exception as e:
        print(f"❌ Error starting simple dashboard: {e}")
        return None


def open_browser():
    """Open browser tabs for the services"""
    
    print("🌐 Opening browser...")
    
    try:
        # Open dashboard
        webbrowser.open('http://localhost:8501')
        time.sleep(1)
        
        # Open API docs
        webbrowser.open('http://localhost:8000/docs')
        
        print("✅ Browser tabs opened")
        
    except Exception as e:
        print(f"⚠️ Could not open browser: {e}")
        print("📱 Manually open:")
        print("   Dashboard: http://localhost:8501")
        print("   API Docs: http://localhost:8000/docs")


def main():
    """Main startup function"""
    
    print("🎯 ImpactSense Platform Startup")
    print("=" * 40)
    
    # Check dependencies
    if not check_dependencies():
        print("\n❌ Cannot start - missing dependencies")
        return
    
    # Generate demo data
    if not generate_demo_data():
        print("\n❌ Cannot start - demo data generation failed")
        return
    
    # Start API server
    api_process = start_api_server()
    if not api_process:
        print("\n❌ Cannot start - API server failed")
        return
    
    # Start dashboard
    dashboard_process = start_dashboard()
    if not dashboard_process:
        print("\n❌ Dashboard failed to start, but API is running")
        print("🔗 API available at: http://localhost:8000")
        api_process.terminate()
        return
    
    # Open browser
    open_browser()
    
    print(f"\n🎉 ImpactSense Platform is running!")
    print(f"📊 Dashboard: http://localhost:8501")
    print(f"🔗 API: http://localhost:8000")
    print(f"📚 API Docs: http://localhost:8000/docs")
    print(f"\n⌨️ Press Ctrl+C to stop all services")
    
    try:
        # Keep running until interrupted
        while True:
            time.sleep(1)
            
            # Check if processes are still running
            if api_process.poll() is not None:
                print("⚠️ API server stopped unexpectedly")
                break
                
            if dashboard_process.poll() is not None:
                print("⚠️ Dashboard stopped unexpectedly")
                break
                
    except KeyboardInterrupt:
        print(f"\n🛑 Shutting down...")
        
        # Terminate processes
        if api_process and api_process.poll() is None:
            api_process.terminate()
            print("✅ API server stopped")
            
        if dashboard_process and dashboard_process.poll() is None:
            dashboard_process.terminate()
            print("✅ Dashboard stopped")
        
        print("👋 ImpactSense Platform stopped")


if __name__ == "__main__":
    main()
