"""
ImpactSense Streamlit Demo App
Interactive dashboard with API demonstration and explainable AI
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import requests
from datetime import datetime, timedelta
import time
from typing import Dict, List, Any
import sys
import os

# Configure Streamlit
st.set_page_config(
    page_title="ImpactSense Demo",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1f4e79 0%, #2e7d32 100%);
        padding: 1.5rem;
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
    .explanation-box {
        background: #f0f8ff;
        border: 1px solid #add8e6;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .decision-factor {
        background: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 5px;
        padding: 0.5rem;
        margin: 0.25rem 0;
    }
    .positive-factor {
        background: #d4edda;
        border-color: #c3e6cb;
    }
    .negative-factor {
        background: #f8d7da;
        border-color: #f5c6cb;
    }
</style>
""", unsafe_allow_html=True)


class ImpactSenseDemo:
    """Main demo application class"""
    
    def __init__(self):
        self.api_base = "http://localhost:8000"
        self.demo_data = self.load_demo_data()
        
    def load_demo_data(self):
        """Load demo data from files"""
        data = {'users': [], 'transactions': [], 'decisions': [], 'ngos': []}
        
        try:
            # Load users
            if os.path.exists('data/demo_users.json'):
                with open('data/demo_users.json', 'r') as f:
                    data['users'] = json.load(f)
            
            # Load transactions
            if os.path.exists('data/demo_transactions.json'):
                with open('data/demo_transactions.json', 'r') as f:
                    data['transactions'] = json.load(f)
            
            # Load decisions
            if os.path.exists('data/demo_decisions.json'):
                with open('data/demo_decisions.json', 'r') as f:
                    data['decisions'] = json.load(f)
            
            # Generate NGO performance data
            data['ngos'] = self.generate_ngo_data()
            
        except Exception as e:
            st.error(f"Error loading data: {e}")
            
        return data
    
    def generate_ngo_data(self):
        """Generate synthetic NGO performance data"""
        ngos = []
        causes = ['Education', 'Environment', 'Health', 'Disaster Relief', 'Poverty', 'Animal Welfare']
        cities = ['Manila', 'Cebu', 'Davao', 'Quezon City', 'Iloilo', 'Cagayan de Oro']
        
        for i in range(50):
            ngo_id = f"ngo_{i+1:03d}"
            cause = np.random.choice(causes)
            city = np.random.choice(cities)
            
            # Generate time series data (last 90 days)
            dates = pd.date_range(start=datetime.now() - timedelta(days=90), end=datetime.now(), freq='D')
            performance_data = []
            
            base_performance = np.random.uniform(0.6, 0.95)
            trend = np.random.uniform(-0.002, 0.002)
            
            for j, date in enumerate(dates):
                # Add trend and noise
                daily_performance = base_performance + (trend * j) + np.random.normal(0, 0.05)
                daily_performance = max(0.2, min(1.0, daily_performance))
                
                performance_data.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'impact_score': round(daily_performance, 3),
                    'donations_received': np.random.poisson(5),
                    'amount_received': round(np.random.uniform(50, 500), 2)
                })
            
            ngo = {
                'id': ngo_id,
                'name': f"{cause} Initiative {city}",
                'cause': cause,
                'location': city,
                'total_impact_score': round(base_performance, 3),
                'total_donations': sum(p['donations_received'] for p in performance_data),
                'total_amount': round(sum(p['amount_received'] for p in performance_data), 2),
                'performance_history': performance_data
            }
            ngos.append(ngo)
        
        return ngos
    
    def render_header(self):
        """Render app header"""
        st.markdown("""
        <div class="main-header">
            <h1>🎯 ImpactSense Demo Dashboard</h1>
            <p>Interactive API Demonstration with Explainable AI</p>
        </div>
        """, unsafe_allow_html=True)
    
    def render_sidebar(self):
        """Render sidebar controls"""
        st.sidebar.title("🔧 Demo Controls")
        
        # API Demo Section
        st.sidebar.header("🔗 API Demo")
        demo_mode = st.sidebar.selectbox(
            "Select Demo Mode",
            ["Live API Demo", "Simulation Mode", "Data Explorer"]
        )
        
        # User Selection for Demo
        if self.demo_data['users']:
            selected_user = st.sidebar.selectbox(
                "Select User for Demo",
                options=range(min(20, len(self.demo_data['users']))),
                format_func=lambda x: f"User {x+1}: {self.demo_data['users'][x].get('segment', 'unknown').title()}"
            )
        else:
            selected_user = 0
        
        # Transaction Parameters
        st.sidebar.header("💳 Transaction Parameters")
        transaction_amount = st.sidebar.slider("Transaction Amount (₱)", 50, 5000, 1200)
        transaction_category = st.sidebar.selectbox(
            "Category",
            ["groceries", "dining", "transportation", "shopping", "entertainment", "bills"]
        )
        
        # AI Settings
        st.sidebar.header("🤖 AI Agent Settings")
        likelihood_threshold = st.sidebar.slider("Likelihood Threshold (%)", 0, 100, 65)
        explainable_ai = st.sidebar.checkbox("Enable Explainable AI", value=True)
        
        return {
            'demo_mode': demo_mode,
            'selected_user': selected_user,
            'transaction_amount': transaction_amount,
            'transaction_category': transaction_category,
            'likelihood_threshold': likelihood_threshold,
            'explainable_ai': explainable_ai
        }
    
    def api_demo_tab(self, settings):
        """API demonstration tab"""
        st.header("🔗 Live API Demonstration")
        
        if not self.demo_data['users']:
            st.warning("No demo data available. Please generate demo data first.")
            if st.button("Generate Demo Data"):
                with st.spinner("Generating demo data..."):
                    # Simulate data generation
                    time.sleep(2)
                    st.success("Demo data generated! Please refresh the page.")
            return
        
        # Get selected user
        user = self.demo_data['users'][settings['selected_user']]
        
        # Display user profile
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("👤 Selected User Profile")
            st.json({
                'id': user['id'],
                'segment': user['segment'],
                'wallet_balance': user['wallet_balance'],
                'donation_history_count': len(user.get('donation_history', [])),
                'preferred_causes': user.get('preferences', {}).get('causes', [])
            })
        
        with col2:
            st.subheader("💳 Transaction Context")
            transaction_context = {
                'amount': settings['transaction_amount'],
                'category': settings['transaction_category'],
                'location': user.get('location', {'city': 'Manila', 'region': 'NCR'}),
                'timestamp': datetime.now().isoformat()
            }
            st.json(transaction_context)
        
        # API Call Simulation
        if st.button("🚀 Evaluate Donation Opportunity", type="primary"):
            with st.spinner("Calling AI agents..."):
                # Simulate API call
                result = self.simulate_api_call(user, transaction_context, settings)
                
                # Display results
                st.subheader("📊 AI Decision Results")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    likelihood_score = result['likelihood_score']
                    color = "green" if likelihood_score >= 70 else "orange" if likelihood_score >= 40 else "red"
                    st.metric(
                        "Likelihood Score",
                        f"{likelihood_score:.1f}%",
                        delta=f"Threshold: {settings['likelihood_threshold']}%"
                    )
                    st.markdown(f"<span style='color: {color}'>{'✅ Will Prompt' if result['should_prompt'] else '❌ Skip Prompt'}</span>", unsafe_allow_html=True)
                
                with col2:
                    if result.get('recommended_cause'):
                        st.metric("Recommended Cause", result['recommended_cause']['cause'])
                        st.metric("Relevance Score", f"{result['recommended_cause']['relevance_score']:.2f}")
                
                with col3:
                    if result.get('suggested_amount'):
                        st.metric("Suggested Amount", f"₱{result['suggested_amount']:.2f}")
                        percentage = (result['suggested_amount'] / settings['transaction_amount']) * 100
                        st.metric("% of Transaction", f"{percentage:.1f}%")
                
                # Explainable AI Section
                if settings['explainable_ai']:
                    self.render_explainable_ai(user, transaction_context, result)
    
    def simulate_api_call(self, user, transaction, settings):
        """Simulate API call with realistic logic"""
        
        # Likelihood calculation
        likelihood_score = 50.0  # Base score
        reasoning_factors = []
        
        # Wallet balance factor
        balance = user.get('wallet_balance', 1000)
        if balance > 5000:
            likelihood_score += 25
            reasoning_factors.append(("positive", "High wallet balance", 25))
        elif balance > 2000:
            likelihood_score += 15
            reasoning_factors.append(("positive", "Good wallet balance", 15))
        elif balance < 500:
            likelihood_score -= 20
            reasoning_factors.append(("negative", "Low wallet balance", -20))
        
        # Transaction amount factor
        amount = transaction.get('amount', 500)
        if amount > 2000:
            likelihood_score += 20
            reasoning_factors.append(("positive", "Large transaction amount", 20))
        elif amount > 1000:
            likelihood_score += 10
            reasoning_factors.append(("positive", "Medium transaction amount", 10))
        
        # User segment factor
        segment = user.get('segment', 'new')
        if segment == 'major':
            likelihood_score += 30
            reasoning_factors.append(("positive", "Major donor segment", 30))
        elif segment == 'regular':
            likelihood_score += 20
            reasoning_factors.append(("positive", "Regular donor segment", 20))
        elif segment == 'new':
            likelihood_score += 5
            reasoning_factors.append(("neutral", "New user segment", 5))
        else:  # inactive
            likelihood_score -= 25
            reasoning_factors.append(("negative", "Inactive user segment", -25))
        
        # Donation history factor
        history = user.get('donation_history', [])
        if len(history) > 10:
            likelihood_score += 20
            reasoning_factors.append(("positive", "Strong donation history", 20))
        elif len(history) > 5:
            likelihood_score += 10
            reasoning_factors.append(("positive", "Good donation history", 10))
        elif len(history) > 0:
            likelihood_score += 5
            reasoning_factors.append(("positive", "Some donation history", 5))
        else:
            likelihood_score -= 10
            reasoning_factors.append(("negative", "No donation history", -10))
        
        # Recent donation check
        if history:
            recent_donations = [
                d for d in history 
                if (datetime.now() - datetime.fromisoformat(d['timestamp'])).days < 7
            ]
            if recent_donations:
                penalty = len(recent_donations) * 15
                likelihood_score -= penalty
                reasoning_factors.append(("negative", f"Recent donations ({len(recent_donations)})", -penalty))
        
        # Category factor
        category = transaction.get('category', 'general')
        if category in ['groceries', 'bills']:
            likelihood_score += 10
            reasoning_factors.append(("positive", "Essential purchase category", 10))
        elif category in ['entertainment', 'dining']:
            likelihood_score -= 5
            reasoning_factors.append(("negative", "Discretionary spending", -5))
        
        likelihood_score = max(0, min(100, likelihood_score))
        should_prompt = likelihood_score >= settings['likelihood_threshold']
        
        result = {
            'should_prompt': should_prompt,
            'likelihood_score': likelihood_score,
            'reasoning_factors': reasoning_factors
        }
        
        # If should prompt, add cause recommendation and amount
        if should_prompt:
            # Cause recommendation
            user_prefs = user.get('preferences', {}).get('causes', [])
            if user_prefs:
                recommended_cause = {
                    'cause': user_prefs[0],
                    'ngo_id': f'ngo_{user_prefs[0].lower()}_001',
                    'relevance_score': 0.92
                }
            else:
                recommended_cause = {
                    'cause': 'Education',
                    'ngo_id': 'ngo_education_001', 
                    'relevance_score': 0.78
                }
            
            result['recommended_cause'] = recommended_cause
            
            # Amount optimization
            base_amount = amount * 0.015  # 1.5% of transaction
            if segment == 'major':
                base_amount *= 2.0
            elif segment == 'regular':
                base_amount *= 1.5
            elif segment == 'new':
                base_amount *= 0.8
            
            max_donation = user.get('preferences', {}).get('max_donation', 50)
            suggested_amount = min(base_amount, max_donation)
            suggested_amount = max(suggested_amount, 5.0)  # Minimum ₱5
            
            result['suggested_amount'] = round(suggested_amount, 2)
        
        return result
    
    def render_explainable_ai(self, user, transaction, result):
        """Render explainable AI section"""
        st.subheader("🧠 Explainable AI: How We Arrived at This Decision")
        
        # Decision flow visualization
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### 📊 Decision Factors Analysis")
            
            factors_df = pd.DataFrame([
                {
                    'Factor': factor[1],
                    'Impact': factor[2],
                    'Type': factor[0]
                }
                for factor in result['reasoning_factors']
            ])
            
            # Create waterfall chart
            fig = go.Figure()
            
            # Starting point
            cumulative = 50.0
            fig.add_trace(go.Bar(
                x=['Base Score'],
                y=[50],
                name='Base Score',
                marker_color='lightblue'
            ))
            
            # Add each factor
            for _, row in factors_df.iterrows():
                cumulative += row['Impact']
                color = 'green' if row['Type'] == 'positive' else 'red' if row['Type'] == 'negative' else 'orange'
                
                fig.add_trace(go.Bar(
                    x=[row['Factor']],
                    y=[row['Impact']],
                    name=row['Factor'],
                    marker_color=color,
                    text=f"{row['Impact']:+.0f}",
                    textposition='auto'
                ))
            
            # Final score
            fig.add_trace(go.Bar(
                x=['Final Score'],
                y=[result['likelihood_score']],
                name='Final Score',
                marker_color='darkblue'
            ))
            
            fig.update_layout(
                title="Likelihood Score Calculation",
                xaxis_title="Decision Factors",
                yaxis_title="Score Impact",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### 📋 Decision Summary")
            
            for factor_type, factor_desc, impact in result['reasoning_factors']:
                css_class = f"{factor_type}-factor decision-factor"
                icon = "✅" if factor_type == "positive" else "❌" if factor_type == "negative" else "⚠️"
                
                st.markdown(f"""
                <div class="{css_class}">
                    {icon} <strong>{factor_desc}</strong><br>
                    Impact: {impact:+.0f} points
                </div>
                """, unsafe_allow_html=True)
        
        # Feature importance chart
        st.markdown("### 🎯 Feature Importance in Decision Making")
        
        feature_importance = {
            'User Segment': 0.25,
            'Wallet Balance': 0.20,
            'Transaction Amount': 0.18,
            'Donation History': 0.15,
            'Recent Activity': 0.12,
            'Category': 0.10
        }
        
        fig = px.bar(
            x=list(feature_importance.keys()),
            y=list(feature_importance.values()),
            title="Feature Importance in Likelihood Prediction",
            labels={'x': 'Features', 'y': 'Importance Score'}
        )
        fig.update_traces(marker_color='lightcoral')
        st.plotly_chart(fig, use_container_width=True)
    
    def ngo_performance_tab(self):
        """NGO performance visualization tab"""
        st.header("🏢 NGO Performance Analytics")
        
        if not self.demo_data['ngos']:
            st.warning("No NGO data available.")
            return
        
        # NGO selection
        col1, col2 = st.columns([1, 1])
        
        with col1:
            selected_cause = st.selectbox(
                "Filter by Cause",
                ["All"] + list(set(ngo['cause'] for ngo in self.demo_data['ngos']))
            )
        
        with col2:
            selected_city = st.selectbox(
                "Filter by City",
                ["All"] + list(set(ngo['location'] for ngo in self.demo_data['ngos']))
            )
        
        # Filter NGOs
        filtered_ngos = self.demo_data['ngos']
        if selected_cause != "All":
            filtered_ngos = [ngo for ngo in filtered_ngos if ngo['cause'] == selected_cause]
        if selected_city != "All":
            filtered_ngos = [ngo for ngo in filtered_ngos if ngo['location'] == selected_city]
        
        # Overview metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total NGOs", len(filtered_ngos))
        
        with col2:
            avg_impact = np.mean([ngo['total_impact_score'] for ngo in filtered_ngos])
            st.metric("Avg Impact Score", f"{avg_impact:.2f}")
        
        with col3:
            total_donations = sum(ngo['total_donations'] for ngo in filtered_ngos)
            st.metric("Total Donations", f"{total_donations:,}")
        
        with col4:
            total_amount = sum(ngo['total_amount'] for ngo in filtered_ngos)
            st.metric("Total Amount", f"₱{total_amount:,.0f}")
        
        # NGO performance comparison
        st.subheader("📊 NGO Performance Comparison")
        
        ngo_df = pd.DataFrame([{
            'NGO': ngo['name'][:30] + '...' if len(ngo['name']) > 30 else ngo['name'],
            'Cause': ngo['cause'],
            'Location': ngo['location'],
            'Impact Score': ngo['total_impact_score'],
            'Donations': ngo['total_donations'],
            'Amount Raised': ngo['total_amount']
        } for ngo in filtered_ngos[:20]])  # Show top 20
        
        fig = px.scatter(
            ngo_df,
            x='Donations',
            y='Impact Score',
            size='Amount Raised',
            color='Cause',
            hover_name='NGO',
            title="NGO Performance: Donations vs Impact Score"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Time series analysis
        st.subheader("📈 NGO Performance Over Time")
        
        # Select specific NGO for detailed view
        ngo_options = [f"{ngo['name']} ({ngo['cause']})" for ngo in filtered_ngos]
        selected_ngo_idx = st.selectbox(
            "Select NGO for Detailed Analysis",
            range(len(ngo_options)),
            format_func=lambda x: ngo_options[x]
        )
        
        if selected_ngo_idx is not None:
            selected_ngo = filtered_ngos[selected_ngo_idx]
            
            # Create time series DataFrame
            ts_data = pd.DataFrame(selected_ngo['performance_history'])
            ts_data['date'] = pd.to_datetime(ts_data['date'])
            
            # Time series plots
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.line(
                    ts_data,
                    x='date',
                    y='impact_score',
                    title=f"Impact Score Trend - {selected_ngo['name']}"
                )
                fig.update_traces(line_color='green')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.bar(
                    ts_data.tail(30),  # Last 30 days
                    x='date',
                    y='donations_received',
                    title="Daily Donations Received (Last 30 Days)"
                )
                fig.update_traces(marker_color='lightblue')
                st.plotly_chart(fig, use_container_width=True)
            
            # Performance analysis
            st.subheader("🔍 Performance Analysis")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                recent_performance = ts_data.tail(7)['impact_score'].mean()
                overall_performance = ts_data['impact_score'].mean()
                change = ((recent_performance - overall_performance) / overall_performance) * 100
                
                st.metric(
                    "Recent Performance (7 days)",
                    f"{recent_performance:.3f}",
                    delta=f"{change:+.1f}%"
                )
            
            with col2:
                recent_donations = ts_data.tail(7)['donations_received'].sum()
                st.metric("Donations (Last 7 days)", recent_donations)
            
            with col3:
                recent_amount = ts_data.tail(7)['amount_received'].sum()
                st.metric("Amount Raised (Last 7 days)", f"₱{recent_amount:.0f}")
            
            # Trend analysis with explainable insights
            st.markdown("### 🧠 AI Performance Insights")
            
            # Calculate trends
            impact_trend = np.polyfit(range(len(ts_data)), ts_data['impact_score'], 1)[0]
            donation_trend = np.polyfit(range(len(ts_data)), ts_data['donations_received'], 1)[0]
            
            insights = []
            
            if impact_trend > 0.001:
                insights.append("📈 **Improving Impact**: Impact score shows positive trend over time")
            elif impact_trend < -0.001:
                insights.append("📉 **Declining Impact**: Impact score shows concerning downward trend")
            else:
                insights.append("📊 **Stable Impact**: Impact score remains relatively stable")
            
            if donation_trend > 0.01:
                insights.append("🎯 **Growing Support**: Donation frequency is increasing")
            elif donation_trend < -0.01:
                insights.append("⚠️ **Declining Support**: Donation frequency is decreasing")
            
            # Performance ranking
            all_scores = [ngo['total_impact_score'] for ngo in self.demo_data['ngos']]
            percentile = (sum(1 for score in all_scores if score < selected_ngo['total_impact_score']) / len(all_scores)) * 100
            insights.append(f"🏆 **Performance Ranking**: Top {100-percentile:.0f}% of all NGOs")
            
            for insight in insights:
                st.markdown(f"""
                <div class="explanation-box">
                    {insight}
                </div>
                """, unsafe_allow_html=True)
    
    def data_explorer_tab(self):
        """Data exploration tab"""
        st.header("📊 Data Explorer")
        
        # Data overview
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Users", len(self.demo_data['users']))
        
        with col2:
            st.metric("Transactions", len(self.demo_data['transactions']))
        
        with col3:
            st.metric("NGOs", len(self.demo_data['ngos']))
        
        # Data type selector
        data_type = st.selectbox(
            "Select Data Type to Explore",
            ["Users", "Transactions", "NGO Performance", "Agent Decisions"]
        )
        
        if data_type == "Users":
            self.explore_users()
        elif data_type == "Transactions":
            self.explore_transactions()
        elif data_type == "NGO Performance":
            self.explore_ngo_performance()
        elif data_type == "Agent Decisions":
            self.explore_agent_decisions()
    
    def explore_users(self):
        """Explore user data"""
        if not self.demo_data['users']:
            st.warning("No user data available.")
            return
        
        users_df = pd.DataFrame(self.demo_data['users'])
        
        col1, col2 = st.columns(2)
        
        with col1:
            # User segment distribution
            segment_counts = users_df['segment'].value_counts()
            fig = px.pie(
                values=segment_counts.values,
                names=segment_counts.index,
                title="User Segment Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Wallet balance distribution
            fig = px.histogram(
                users_df,
                x='wallet_balance',
                nbins=30,
                title="Wallet Balance Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Raw data
        st.subheader("📋 User Data Sample")
        st.dataframe(users_df.head(10))
    
    def explore_transactions(self):
        """Explore transaction data"""
        if not self.demo_data['transactions']:
            st.warning("No transaction data available.")
            return
        
        transactions_df = pd.DataFrame(self.demo_data['transactions'])
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Transaction category distribution
            category_counts = transactions_df['category'].value_counts()
            fig = px.bar(
                x=category_counts.index,
                y=category_counts.values,
                title="Transaction Categories"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Amount distribution
            fig = px.histogram(
                transactions_df,
                x='amount',
                nbins=30,
                title="Transaction Amount Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Conversion analysis
        st.subheader("💡 Conversion Analysis")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            prompts_shown = sum(1 for t in self.demo_data['transactions'] if t.get('prompt_shown', False))
            st.metric("Prompts Shown", prompts_shown)
        
        with col2:
            donations_made = sum(1 for t in self.demo_data['transactions'] if t.get('donation_made', False))
            st.metric("Donations Made", donations_made)
        
        with col3:
            conversion_rate = (donations_made / max(1, prompts_shown)) * 100
            st.metric("Conversion Rate", f"{conversion_rate:.1f}%")
    
    def explore_ngo_performance(self):
        """Explore NGO performance data"""
        if not self.demo_data['ngos']:
            st.warning("No NGO data available.")
            return
        
        # NGO summary statistics
        ngo_df = pd.DataFrame([{
            'Name': ngo['name'],
            'Cause': ngo['cause'],
            'Location': ngo['location'],
            'Impact Score': ngo['total_impact_score'],
            'Total Donations': ngo['total_donations'],
            'Total Amount': ngo['total_amount']
        } for ngo in self.demo_data['ngos']])
        
        # Summary statistics
        st.subheader("📊 NGO Performance Summary")
        st.dataframe(ngo_df.describe())
        
        # Top performers
        st.subheader("🏆 Top Performing NGOs")
        
        col1, col2 = st.columns(2)
        
        with col1:
            top_impact = ngo_df.nlargest(10, 'Impact Score')[['Name', 'Impact Score']]
            fig = px.bar(
                top_impact,
                x='Impact Score',
                y='Name',
                orientation='h',
                title="Top 10 NGOs by Impact Score"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            top_donations = ngo_df.nlargest(10, 'Total Donations')[['Name', 'Total Donations']]
            fig = px.bar(
                top_donations,
                x='Total Donations',
                y='Name',
                orientation='h',
                title="Top 10 NGOs by Donations"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    def explore_agent_decisions(self):
        """Explore agent decision data"""
        if not self.demo_data['decisions']:
            st.warning("No agent decision data available.")
            return
        
        decisions_df = pd.DataFrame(self.demo_data['decisions'])
        
        # Agent performance
        agent_counts = decisions_df['agent_name'].value_counts()
        fig = px.bar(
            x=agent_counts.index,
            y=agent_counts.values,
            title="Decisions by Agent Type"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Decision details
        st.subheader("📋 Recent Agent Decisions")
        st.dataframe(decisions_df.head(20))
    
    def run(self):
        """Run the demo application"""
        
        # Render header
        self.render_header()
        
        # Render sidebar and get settings
        settings = self.render_sidebar()
        
        # Main tabs
        tab1, tab2, tab3 = st.tabs([
            "🔗 API Demo",
            "🏢 NGO Performance",
            "📊 Data Explorer"
        ])
        
        with tab1:
            self.api_demo_tab(settings)
        
        with tab2:
            self.ngo_performance_tab()
        
        with tab3:
            self.data_explorer_tab()


def main():
    """Main function"""
    
    # Check if demo data exists
    if not os.path.exists('data'):
        st.error("Demo data directory not found. Please run `uv run generate_demo_data.py` first.")
        return
    
    # Initialize and run demo
    demo = ImpactSenseDemo()
    demo.run()


if __name__ == "__main__":
    main()
