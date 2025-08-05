"""
ImpactSense Dashboard
Comprehensive monitoring and control panel for the AI-powered microdonation platform
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from datetime import datetime, timedelta
import sys
import os
from typing import Dict, List, Any
import asyncio

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Configure Streamlit page
st.set_page_config(
    page_title="ImpactSense Dashboard",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1f4e79 0%, #2e7d32 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #2e7d32;
    }
    .agent-card {
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .success-metric {
        color: #2e7d32;
        font-weight: bold;
    }
    .warning-metric {
        color: #f57c00;
        font-weight: bold;
    }
    .error-metric {
        color: #d32f2f;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


class ImpactSenseDashboard:
    """Main dashboard class"""
    
    def __init__(self):
        self.demo_mode = True  # Set to False when connected to real data
        
    def generate_demo_data(self):
        """Generate demo data for dashboard"""
        
        # Generate synthetic performance data
        agents = ['Likelihood Agent', 'Cause Recommender', 'Amount Optimizer', 'ML Likelihood', 'ML Amount']
        dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='D')
        
        performance_data = []
        for agent in agents:
            for date in dates:
                base_conversion = np.random.uniform(0.15, 0.35)
                if 'ML' in agent:
                    base_conversion += 0.05  # ML agents perform slightly better
                
                performance_data.append({
                    'date': date,
                    'agent': agent,
                    'decisions': np.random.randint(50, 200),
                    'conversions': int(np.random.randint(50, 200) * base_conversion),
                    'conversion_rate': base_conversion,
                    'avg_amount': np.random.uniform(5, 25),
                    'total_raised': np.random.uniform(200, 1500)
                })
        
        self.performance_df = pd.DataFrame(performance_data)
        
        # Generate user engagement data
        self.user_data = pd.DataFrame({
            'user_segment': ['New Users', 'Regular Donors', 'Major Donors', 'Inactive Users'],
            'count': [2500, 1200, 150, 800],
            'avg_donation': [8.50, 15.20, 45.30, 0],
            'engagement_score': [0.25, 0.65, 0.85, 0.05]
        })
        
        # Generate cause performance data
        self.cause_data = pd.DataFrame({
            'cause': ['Education', 'Environment', 'Health', 'Disaster Relief', 'Poverty'],
            'donations_count': [1250, 980, 1100, 450, 820],
            'total_amount': [18750, 14200, 22500, 9800, 12600],
            'avg_amount': [15.0, 14.5, 20.5, 21.8, 15.4],
            'success_rate': [0.32, 0.28, 0.35, 0.42, 0.29]
        })
        
        # Generate real-time metrics
        self.realtime_metrics = {
            'active_users': np.random.randint(150, 300),
            'donations_today': np.random.randint(45, 85),
            'amount_raised_today': np.random.uniform(800, 1500),
            'conversion_rate_today': np.random.uniform(0.25, 0.35),
            'avg_response_time': np.random.uniform(150, 300),
            'system_health': np.random.uniform(0.95, 0.99)
        }
    
    def render_header(self):
        """Render dashboard header"""
        st.markdown("""
        <div class="main-header">
            <h1>🎯 ImpactSense Dashboard</h1>
            <p>AI-Powered Microdonation Platform Analytics & Control Center</p>
        </div>
        """, unsafe_allow_html=True)
    
    def render_sidebar(self):
        """Render sidebar controls"""
        st.sidebar.title("🔧 Dashboard Controls")
        
        # Time range selector
        time_range = st.sidebar.selectbox(
            "📅 Time Range",
            ["Last 24 Hours", "Last 7 Days", "Last 30 Days", "Custom Range"]
        )
        
        if time_range == "Custom Range":
            col1, col2 = st.sidebar.columns(2)
            start_date = col1.date_input("Start Date")
            end_date = col2.date_input("End Date")
        
        st.sidebar.divider()
        
        # Agent controls
        st.sidebar.subheader("🤖 Agent Controls")
        
        likelihood_enabled = st.sidebar.checkbox("Likelihood Agent", value=True)
        ml_likelihood_enabled = st.sidebar.checkbox("ML Likelihood Agent", value=True)
        recommender_enabled = st.sidebar.checkbox("Cause Recommender", value=True)
        amount_optimizer_enabled = st.sidebar.checkbox("Amount Optimizer", value=True)
        
        st.sidebar.divider()
        
        # System settings
        st.sidebar.subheader("⚙️ System Settings")
        
        prompt_threshold = st.sidebar.slider("Prompt Threshold (%)", 0, 100, 65)
        max_daily_prompts = st.sidebar.number_input("Max Daily Prompts per User", 1, 10, 3)
        
        if st.sidebar.button("🔄 Refresh Data"):
            st.rerun()
        
        return {
            'time_range': time_range,
            'agents': {
                'likelihood': likelihood_enabled,
                'ml_likelihood': ml_likelihood_enabled,
                'recommender': recommender_enabled,
                'amount_optimizer': amount_optimizer_enabled
            },
            'settings': {
                'prompt_threshold': prompt_threshold,
                'max_daily_prompts': max_daily_prompts
            }
        }
    
    def render_realtime_metrics(self):
        """Render real-time system metrics"""
        st.subheader("📊 Real-Time Metrics")
        
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        
        with col1:
            st.metric(
                "Active Users",
                f"{self.realtime_metrics['active_users']:,}",
                delta=f"+{np.random.randint(5, 25)}"
            )
        
        with col2:
            st.metric(
                "Donations Today",
                f"{self.realtime_metrics['donations_today']:,}",
                delta=f"+{np.random.randint(2, 8)}"
            )
        
        with col3:
            st.metric(
                "Amount Raised Today",
                f"₱{self.realtime_metrics['amount_raised_today']:,.0f}",
                delta=f"+₱{np.random.uniform(50, 150):.0f}"
            )
        
        with col4:
            conversion_rate = self.realtime_metrics['conversion_rate_today']
            st.metric(
                "Conversion Rate",
                f"{conversion_rate:.1%}",
                delta=f"{np.random.uniform(-0.02, 0.03):+.1%}"
            )
        
        with col5:
            response_time = self.realtime_metrics['avg_response_time']
            st.metric(
                "Avg Response Time",
                f"{response_time:.0f}ms",
                delta=f"{np.random.uniform(-20, 10):+.0f}ms"
            )
        
        with col6:
            health = self.realtime_metrics['system_health']
            st.metric(
                "System Health",
                f"{health:.1%}",
                delta=f"{np.random.uniform(-0.01, 0.01):+.1%}"
            )
    
    def render_agent_performance(self):
        """Render agent performance analytics"""
        st.subheader("🤖 Agent Performance")
        
        # Agent performance over time
        fig = px.line(
            self.performance_df,
            x='date',
            y='conversion_rate',
            color='agent',
            title='Agent Conversion Rates Over Time',
            labels={'conversion_rate': 'Conversion Rate', 'date': 'Date'}
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Agent comparison metrics
        col1, col2 = st.columns(2)
        
        with col1:
            # Average performance by agent
            agent_avg = self.performance_df.groupby('agent').agg({
                'conversion_rate': 'mean',
                'decisions': 'sum',
                'total_raised': 'sum'
            }).reset_index()
            
            fig = px.bar(
                agent_avg,
                x='agent',
                y='conversion_rate',
                title='Average Conversion Rate by Agent',
                color='conversion_rate',
                color_continuous_scale='Greens'
            )
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Total impact by agent
            fig = px.pie(
                agent_avg,
                values='total_raised',
                names='agent',
                title='Total Amount Raised by Agent'
            )
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        # Detailed agent metrics table
        st.subheader("📈 Detailed Agent Metrics")
        
        # Calculate detailed metrics
        detailed_metrics = self.performance_df.groupby('agent').agg({
            'decisions': ['sum', 'mean'],
            'conversions': 'sum',
            'conversion_rate': ['mean', 'std'],
            'avg_amount': 'mean',
            'total_raised': 'sum'
        }).round(3)
        
        detailed_metrics.columns = [
            'Total Decisions', 'Avg Daily Decisions', 'Total Conversions',
            'Avg Conversion Rate', 'Conversion Rate StdDev', 'Avg Donation Amount', 'Total Raised'
        ]
        
        # Style the dataframe
        styled_df = detailed_metrics.style.format({
            'Total Decisions': '{:,.0f}',
            'Avg Daily Decisions': '{:.1f}',
            'Total Conversions': '{:,.0f}',
            'Avg Conversion Rate': '{:.1%}',
            'Conversion Rate StdDev': '{:.3f}',
            'Avg Donation Amount': '₱{:.2f}',
            'Total Raised': '₱{:,.2f}'
        }).background_gradient(subset=['Avg Conversion Rate'], cmap='Greens')
        
        st.dataframe(styled_df, use_container_width=True)
    
    def render_user_insights(self):
        """Render user behavior insights"""
        st.subheader("👥 User Insights")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # User segment distribution
            fig = px.sunburst(
                self.user_data,
                values='count',
                names='user_segment',
                title='User Segment Distribution'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Engagement vs Donation behavior
            fig = px.scatter(
                self.user_data,
                x='engagement_score',
                y='avg_donation',
                size='count',
                hover_name='user_segment',
                title='Engagement vs Donation Behavior',
                labels={
                    'engagement_score': 'Engagement Score',
                    'avg_donation': 'Average Donation (₱)'
                }
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # User behavior trends
        st.subheader("📱 User Behavior Trends")
        
        # Generate trend data
        trend_dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='D')
        trend_data = []
        
        for date in trend_dates:
            trend_data.append({
                'date': date,
                'new_users': np.random.randint(20, 80),
                'returning_users': np.random.randint(100, 300),
                'churn_rate': np.random.uniform(0.02, 0.08),
                'avg_session_length': np.random.uniform(120, 300)
            })
        
        trend_df = pd.DataFrame(trend_data)
        
        # Plot trends
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Daily Active Users', 'User Acquisition', 'Churn Rate', 'Session Length'],
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Daily active users
        fig.add_trace(
            go.Scatter(x=trend_df['date'], y=trend_df['new_users'] + trend_df['returning_users'],
                      name='Total Active Users', line=dict(color='blue')),
            row=1, col=1
        )
        
        # User acquisition
        fig.add_trace(
            go.Scatter(x=trend_df['date'], y=trend_df['new_users'],
                      name='New Users', line=dict(color='green')),
            row=1, col=2
        )
        
        # Churn rate
        fig.add_trace(
            go.Scatter(x=trend_df['date'], y=trend_df['churn_rate'],
                      name='Churn Rate', line=dict(color='red')),
            row=2, col=1
        )
        
        # Session length
        fig.add_trace(
            go.Scatter(x=trend_df['date'], y=trend_df['avg_session_length'],
                      name='Session Length (s)', line=dict(color='orange')),
            row=2, col=2
        )
        
        fig.update_layout(height=500, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    def render_cause_analytics(self):
        """Render cause performance analytics"""
        st.subheader("🎯 Cause Analytics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Cause performance comparison
            fig = px.bar(
                self.cause_data,
                x='cause',
                y='success_rate',
                color='success_rate',
                title='Success Rate by Cause',
                color_continuous_scale='Viridis'
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Total impact by cause
            fig = px.treemap(
                self.cause_data,
                values='total_amount',
                names='cause',
                title='Total Amount Raised by Cause'
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Cause performance table
        st.subheader("📊 Detailed Cause Metrics")
        
        styled_cause_df = self.cause_data.style.format({
            'donations_count': '{:,.0f}',
            'total_amount': '₱{:,.2f}',
            'avg_amount': '₱{:.2f}',
            'success_rate': '{:.1%}'
        }).background_gradient(subset=['success_rate'], cmap='RdYlGn')
        
        st.dataframe(styled_cause_df, use_container_width=True)
    
    def render_system_monitoring(self):
        """Render system health and monitoring"""
        st.subheader("🔧 System Monitoring")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### 🟢 System Status")
            
            status_items = [
                ("API Gateway", "Healthy", "success"),
                ("ML Models", "Healthy", "success"),
                ("Database", "Healthy", "success"),
                ("Cache Layer", "Warning", "warning"),
                ("BPI Integration", "Healthy", "success")
            ]
            
            for component, status, level in status_items:
                if level == "success":
                    st.success(f"✅ {component}: {status}")
                elif level == "warning":
                    st.warning(f"⚠️ {component}: {status}")
                else:
                    st.error(f"❌ {component}: {status}")
        
        with col2:
            st.markdown("### 📈 Performance Metrics")
            
            # System performance gauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=self.realtime_metrics['system_health'] * 100,
                delta={'reference': 95},
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "System Health (%)"},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkgreen"},
                    'steps': [
                        {'range': [0, 70], 'color': "lightgray"},
                        {'range': [70, 90], 'color': "yellow"},
                        {'range': [90, 100], 'color': "lightgreen"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 95
                    }
                }
            ))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        with col3:
            st.markdown("### 🚨 Recent Alerts")
            
            alerts = [
                ("INFO", "ML model retrained successfully", "2 minutes ago"),
                ("WARNING", "High response latency detected", "15 minutes ago"),
                ("SUCCESS", "New agent deployment completed", "1 hour ago"),
                ("INFO", "Daily backup completed", "2 hours ago")
            ]
            
            for level, message, time in alerts:
                if level == "SUCCESS":
                    st.success(f"✅ {message} ({time})")
                elif level == "WARNING":
                    st.warning(f"⚠️ {message} ({time})")
                elif level == "ERROR":
                    st.error(f"❌ {message} ({time})")
                else:
                    st.info(f"ℹ️ {message} ({time})")
        
        # Detailed system logs
        st.subheader("📋 System Logs")
        
        log_data = pd.DataFrame({
            'timestamp': pd.date_range(start='2024-01-31 09:00:00', periods=20, freq='5min'),
            'level': np.random.choice(['INFO', 'WARNING', 'ERROR', 'DEBUG'], 20, p=[0.6, 0.2, 0.1, 0.1]),
            'component': np.random.choice(['API', 'ML Engine', 'Database', 'Cache', 'BPI'], 20),
            'message': [f"System event {i+1}" for i in range(20)]
        })
        
        # Filter logs by level
        log_filter = st.selectbox("Filter by level:", ["All", "INFO", "WARNING", "ERROR", "DEBUG"])
        
        if log_filter != "All":
            filtered_logs = log_data[log_data['level'] == log_filter]
        else:
            filtered_logs = log_data
        
        st.dataframe(filtered_logs.sort_values('timestamp', ascending=False), use_container_width=True)
    
    def render_data_management(self):
        """Render data generation and management tools"""
        st.subheader("🗄️ Data Management")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📊 Synthetic Data Generation")
            
            num_users = st.number_input("Number of Users to Generate", 100, 10000, 1000)
            num_transactions = st.number_input("Number of Transactions", 1000, 100000, 10000)
            
            if st.button("🔄 Generate Synthetic Data"):
                with st.spinner("Generating synthetic data..."):
                    # Simulate data generation
                    progress_bar = st.progress(0)
                    for i in range(101):
                        progress_bar.progress(i)
                        if i == 100:
                            st.success(f"✅ Generated {num_users:,} users and {num_transactions:,} transactions!")
            
            st.markdown("### 🧠 ML Model Training")
            
            if st.button("🎯 Train Likelihood Model"):
                with st.spinner("Training likelihood prediction model..."):
                    progress_bar = st.progress(0)
                    for i in range(101):
                        progress_bar.progress(i)
                        if i == 100:
                            st.success("✅ Likelihood model trained successfully!")
            
            if st.button("💰 Train Amount Optimization Model"):
                with st.spinner("Training amount optimization model..."):
                    progress_bar = st.progress(0)
                    for i in range(101):
                        progress_bar.progress(i)
                        if i == 100:
                            st.success("✅ Amount optimization model trained successfully!")
        
        with col2:
            st.markdown("### 📈 Data Quality Metrics")
            
            quality_metrics = {
                "Data Completeness": 0.96,
                "Data Accuracy": 0.94,
                "Data Consistency": 0.98,
                "Model Performance": 0.87,
                "Feature Quality": 0.91
            }
            
            for metric, value in quality_metrics.items():
                col_metric, col_bar = st.columns([1, 2])
                with col_metric:
                    st.metric(metric, f"{value:.1%}")
                with col_bar:
                    st.progress(value)
            
            st.markdown("### 🔄 Data Pipeline Status")
            
            pipeline_status = [
                ("Data Ingestion", "Running", "🟢"),
                ("Feature Engineering", "Running", "🟢"),
                ("Model Training", "Scheduled", "🟡"),
                ("Data Validation", "Running", "🟢"),
                ("Export Pipeline", "Idle", "⚪")
            ]
            
            for pipeline, status, indicator in pipeline_status:
                st.write(f"{indicator} **{pipeline}**: {status}")
    
    def run(self):
        """Run the dashboard"""
        
        # Generate demo data
        self.generate_demo_data()
        
        # Render header
        self.render_header()
        
        # Render sidebar and get controls
        controls = self.render_sidebar()
        
        # Main dashboard tabs
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📊 Overview",
            "🤖 Agent Performance", 
            "👥 User Insights",
            "🎯 Cause Analytics",
            "🔧 System Monitoring",
            "🗄️ Data Management"
        ])
        
        with tab1:
            self.render_realtime_metrics()
            st.divider()
            
            # Quick overview charts
            col1, col2 = st.columns(2)
            
            with col1:
                # Daily donations trend
                daily_trend = self.performance_df.groupby('date').agg({
                    'conversions': 'sum',
                    'total_raised': 'sum'
                }).reset_index()
                
                fig = px.line(
                    daily_trend,
                    x='date',
                    y='total_raised',
                    title='Daily Amount Raised Trend',
                    labels={'total_raised': 'Amount Raised (₱)'}
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Conversion funnel
                funnel_data = pd.DataFrame({
                    'stage': ['Users Reached', 'Prompts Shown', 'Donations Made', 'Recurring Donors'],
                    'count': [10000, 3500, 875, 234]
                })
                
                fig = px.funnel(
                    funnel_data,
                    x='count',
                    y='stage',
                    title='Donation Conversion Funnel'
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            self.render_agent_performance()
        
        with tab3:
            self.render_user_insights()
        
        with tab4:
            self.render_cause_analytics()
        
        with tab5:
            self.render_system_monitoring()
        
        with tab6:
            self.render_data_management()


def main():
    """Main function to run the dashboard"""
    dashboard = ImpactSenseDashboard()
    dashboard.run()


if __name__ == "__main__":
    main()
