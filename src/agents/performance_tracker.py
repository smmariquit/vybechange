"""
Real-time Agent Performance Analytics
Monitors agent decisions and provides insights for continuous improvement
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import sqlite3
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)

class AgentPerformanceTracker:
    """Tracks and analyzes agent performance in real-time"""
    
    def __init__(self, db_path: str = 'data/agent_performance.db'):
        self.db_path = db_path
        self.init_database()
        
        # Performance metrics cache
        self.metrics_cache = {}
        self.cache_timestamp = None
        self.cache_duration = 300  # 5 minutes
    
    def init_database(self):
        """Initialize SQLite database for performance tracking"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Agent decisions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS agent_decisions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                user_id TEXT,
                agent_name TEXT,
                decision_type TEXT,
                context_data TEXT,
                decision_data TEXT,
                outcome TEXT,
                success_metric REAL
            )
        ''')
        
        # User responses table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_responses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                user_id TEXT,
                prompt_id TEXT,
                response_type TEXT,
                response_value TEXT,
                conversion_value REAL
            )
        ''')
        
        # Agent performance metrics table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS performance_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                agent_name TEXT,
                metric_name TEXT,
                metric_value REAL,
                time_period TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def log_agent_decision(self, agent_name: str, user_id: str, context: Dict[str, Any], 
                          decision: Dict[str, Any], decision_type: str = 'recommendation'):
        """Log an agent decision for performance tracking"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO agent_decisions 
            (user_id, agent_name, decision_type, context_data, decision_data)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            user_id,
            agent_name,
            decision_type,
            json.dumps(context),
            json.dumps(decision)
        ))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Logged decision for {agent_name}: {decision_type}")
    
    def log_user_response(self, user_id: str, prompt_id: str, response_type: str, 
                         response_value: Any, conversion_value: float = 0.0):
        """Log user response to agent prompts"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO user_responses 
            (user_id, prompt_id, response_type, response_value, conversion_value)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            user_id,
            prompt_id,
            response_type,
            str(response_value),
            conversion_value
        ))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Logged user response: {response_type} = {response_value}")
    
    def calculate_agent_performance(self, agent_name: str, time_period: str = '24h') -> Dict[str, float]:
        """Calculate performance metrics for a specific agent"""
        
        # Check cache first
        cache_key = f"{agent_name}_{time_period}"
        if self._is_cache_valid() and cache_key in self.metrics_cache:
            return self.metrics_cache[cache_key]
        
        conn = sqlite3.connect(self.db_path)
        
        # Define time window
        if time_period == '24h':
            start_time = datetime.now() - timedelta(hours=24)
        elif time_period == '7d':
            start_time = datetime.now() - timedelta(days=7)
        elif time_period == '30d':
            start_time = datetime.now() - timedelta(days=30)
        else:
            start_time = datetime.now() - timedelta(hours=24)
        
        # Get agent decisions in time period
        decisions_df = pd.read_sql_query('''
            SELECT * FROM agent_decisions 
            WHERE agent_name = ? AND timestamp >= ?
            ORDER BY timestamp DESC
        ''', conn, params=(agent_name, start_time))
        
        if len(decisions_df) == 0:
            return {'total_decisions': 0, 'accuracy': 0, 'conversion_rate': 0}
        
        # Get corresponding user responses
        user_responses_df = pd.read_sql_query('''
            SELECT * FROM user_responses 
            WHERE timestamp >= ?
            ORDER BY timestamp DESC
        ''', conn, params=(start_time,))
        
        conn.close()
        
        # Calculate metrics based on agent type
        if agent_name == "DonationLikelyScoreAgent" or agent_name == "MLDonationLikelyScoreAgent":
            metrics = self._calculate_likelihood_metrics(decisions_df, user_responses_df)
        elif agent_name == "LocalCauseRecommenderAgent":
            metrics = self._calculate_recommendation_metrics(decisions_df, user_responses_df)
        elif agent_name == "DonationAmountOptimizerAgent" or agent_name == "MLDonationAmountOptimizerAgent":
            metrics = self._calculate_amount_metrics(decisions_df, user_responses_df)
        else:
            metrics = self._calculate_generic_metrics(decisions_df, user_responses_df)
        
        # Cache results
        self.metrics_cache[cache_key] = metrics
        self.cache_timestamp = datetime.now()
        
        return metrics
    
    def _calculate_likelihood_metrics(self, decisions_df: pd.DataFrame, 
                                    responses_df: pd.DataFrame) -> Dict[str, float]:
        """Calculate metrics for donation likelihood agents"""
        
        metrics = {
            'total_decisions': len(decisions_df),
            'prompts_shown': 0,
            'prompts_accepted': 0,
            'conversion_rate': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'avg_likelihood_score': 0.0
        }
        
        if len(decisions_df) == 0:
            return metrics
        
        # Parse decision data
        prompted_decisions = []
        likelihood_scores = []
        
        for _, row in decisions_df.iterrows():
            try:
                decision_data = json.loads(row['decision_data'])
                
                if decision_data.get('recommendation') in ['prompt_now', 'prompt_with_caution']:
                    prompted_decisions.append(row)
                
                if 'likelihood_score' in decision_data:
                    likelihood_scores.append(decision_data['likelihood_score'])
                    
            except json.JSONDecodeError:
                continue
        
        metrics['prompts_shown'] = len(prompted_decisions)
        metrics['avg_likelihood_score'] = np.mean(likelihood_scores) if likelihood_scores else 0
        
        # Calculate conversion rate
        if metrics['prompts_shown'] > 0:
            # Count actual donations following prompts
            accepted_prompts = len(responses_df[responses_df['response_type'] == 'donation_accepted'])
            metrics['prompts_accepted'] = accepted_prompts
            metrics['conversion_rate'] = accepted_prompts / metrics['prompts_shown']
        
        # Calculate precision and recall
        true_positives = metrics['prompts_accepted']
        false_positives = metrics['prompts_shown'] - metrics['prompts_accepted']
        
        # This is simplified - in reality, you'd need more sophisticated tracking
        metrics['precision'] = true_positives / max(1, true_positives + false_positives)
        metrics['recall'] = 0.8  # Placeholder - would need more complex calculation
        
        return metrics
    
    def _calculate_recommendation_metrics(self, decisions_df: pd.DataFrame,
                                        responses_df: pd.DataFrame) -> Dict[str, float]:
        """Calculate metrics for cause recommendation agents"""
        
        metrics = {
            'total_decisions': len(decisions_df),
            'recommendations_made': 0,
            'recommendations_accepted': 0,
            'acceptance_rate': 0.0,
            'avg_relevance_score': 0.0,
            'geographic_accuracy': 0.0
        }
        
        if len(decisions_df) == 0:
            return metrics
        
        # Parse decision data
        relevance_scores = []
        geographic_matches = []
        
        for _, row in decisions_df.iterrows():
            try:
                decision_data = json.loads(row['decision_data'])
                
                if 'primary_recommendation' in decision_data:
                    metrics['recommendations_made'] += 1
                    
                    # Extract relevance score
                    if isinstance(decision_data['primary_recommendation'], dict):
                        relevance_score = decision_data['primary_recommendation'].get('relevance_score', 0)
                        relevance_scores.append(relevance_score)
                    
                    # Check geographic match
                    geographic_match = decision_data.get('geographic_match', 'unknown')
                    if geographic_match == 'exact':
                        geographic_matches.append(1)
                    elif geographic_match == 'regional':
                        geographic_matches.append(0.5)
                    else:
                        geographic_matches.append(0)
                        
            except (json.JSONDecodeError, TypeError):
                continue
        
        # Calculate acceptance rate
        accepted_recommendations = len(responses_df[
            responses_df['response_type'] == 'cause_accepted'
        ])
        metrics['recommendations_accepted'] = accepted_recommendations
        
        if metrics['recommendations_made'] > 0:
            metrics['acceptance_rate'] = accepted_recommendations / metrics['recommendations_made']
        
        # Calculate average scores
        metrics['avg_relevance_score'] = np.mean(relevance_scores) if relevance_scores else 0
        metrics['geographic_accuracy'] = np.mean(geographic_matches) if geographic_matches else 0
        
        return metrics
    
    def _calculate_amount_metrics(self, decisions_df: pd.DataFrame,
                                responses_df: pd.DataFrame) -> Dict[str, float]:
        """Calculate metrics for amount optimization agents"""
        
        metrics = {
            'total_decisions': len(decisions_df),
            'amount_suggestions': 0,
            'avg_suggested_amount': 0.0,
            'avg_actual_amount': 0.0,
            'prediction_accuracy': 0.0,
            'amount_acceptance_rate': 0.0
        }
        
        if len(decisions_df) == 0:
            return metrics
        
        # Parse decision data
        suggested_amounts = []
        
        for _, row in decisions_df.iterrows():
            try:
                decision_data = json.loads(row['decision_data'])
                
                if 'primary_amount' in decision_data:
                    metrics['amount_suggestions'] += 1
                    suggested_amounts.append(decision_data['primary_amount'])
                    
            except json.JSONDecodeError:
                continue
        
        # Calculate average suggested amount
        if suggested_amounts:
            metrics['avg_suggested_amount'] = np.mean(suggested_amounts)
        
        # Get actual donation amounts
        actual_donations = responses_df[responses_df['response_type'] == 'donation_amount']
        if len(actual_donations) > 0:
            actual_amounts = actual_donations['conversion_value'].astype(float)
            metrics['avg_actual_amount'] = actual_amounts.mean()
            
            # Calculate prediction accuracy (how close suggestions were to actual)
            if metrics['avg_suggested_amount'] > 0:
                accuracy = 1 - abs(metrics['avg_actual_amount'] - metrics['avg_suggested_amount']) / metrics['avg_suggested_amount']
                metrics['prediction_accuracy'] = max(0, accuracy)
        
        # Amount acceptance rate (users who donated the suggested amount)
        exact_matches = len(actual_donations[actual_donations['conversion_value'].isin(suggested_amounts)])
        if metrics['amount_suggestions'] > 0:
            metrics['amount_acceptance_rate'] = exact_matches / metrics['amount_suggestions']
        
        return metrics
    
    def _calculate_generic_metrics(self, decisions_df: pd.DataFrame,
                                 responses_df: pd.DataFrame) -> Dict[str, float]:
        """Calculate generic metrics for any agent"""
        
        return {
            'total_decisions': len(decisions_df),
            'success_rate': 0.5,  # Placeholder
            'response_rate': len(responses_df) / max(1, len(decisions_df))
        }
    
    def generate_performance_report(self, time_period: str = '24h') -> Dict[str, Any]:
        """Generate comprehensive performance report for all agents"""
        
        agents = [
            'DonationLikelyScoreAgent',
            'MLDonationLikelyScoreAgent', 
            'LocalCauseRecommenderAgent',
            'DonationAmountOptimizerAgent',
            'MLDonationAmountOptimizerAgent'
        ]
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'time_period': time_period,
            'agents': {},
            'overall_metrics': {},
            'insights': []
        }
        
        # Get metrics for each agent
        all_metrics = {}
        for agent in agents:
            metrics = self.calculate_agent_performance(agent, time_period)
            all_metrics[agent] = metrics
            report['agents'][agent] = metrics
        
        # Calculate overall metrics
        total_decisions = sum(m.get('total_decisions', 0) for m in all_metrics.values())
        total_conversions = sum(m.get('prompts_accepted', 0) for m in all_metrics.values())
        
        report['overall_metrics'] = {
            'total_decisions': total_decisions,
            'total_conversions': total_conversions,
            'overall_conversion_rate': total_conversions / max(1, total_decisions),
            'active_agents': len([a for a in all_metrics.values() if a.get('total_decisions', 0) > 0])
        }
        
        # Generate insights
        report['insights'] = self._generate_insights(all_metrics)
        
        return report
    
    def _generate_insights(self, metrics: Dict[str, Dict[str, float]]) -> List[str]:
        """Generate insights from performance metrics"""
        
        insights = []
        
        # Check likelihood agent performance
        likelihood_agents = [k for k in metrics.keys() if 'LikelyScore' in k]
        for agent in likelihood_agents:
            agent_metrics = metrics[agent]
            conversion_rate = agent_metrics.get('conversion_rate', 0)
            
            if conversion_rate > 0.3:
                insights.append(f"🎯 {agent} performing well with {conversion_rate:.1%} conversion rate")
            elif conversion_rate < 0.1:
                insights.append(f"⚠️ {agent} conversion rate low at {conversion_rate:.1%} - may need tuning")
        
        # Check ML vs rule-based performance
        if 'DonationLikelyScoreAgent' in metrics and 'MLDonationLikelyScoreAgent' in metrics:
            rule_conversion = metrics['DonationLikelyScoreAgent'].get('conversion_rate', 0)
            ml_conversion = metrics['MLDonationLikelyScoreAgent'].get('conversion_rate', 0)
            
            if ml_conversion > rule_conversion * 1.1:
                insights.append("🤖 ML agent outperforming rule-based agent by {:.1%}".format(ml_conversion - rule_conversion))
            elif rule_conversion > ml_conversion * 1.1:
                insights.append("📏 Rule-based agent outperforming ML agent - check model training")
        
        # Check amount optimization
        amount_agents = [k for k in metrics.keys() if 'Amount' in k]
        for agent in amount_agents:
            agent_metrics = metrics[agent]
            accuracy = agent_metrics.get('prediction_accuracy', 0)
            
            if accuracy > 0.8:
                insights.append(f"💰 {agent} predicting amounts accurately ({accuracy:.1%})")
            elif accuracy < 0.5:
                insights.append(f"💸 {agent} amount predictions need improvement ({accuracy:.1%})")
        
        # Overall system health
        total_decisions = sum(m.get('total_decisions', 0) for m in metrics.values())
        if total_decisions < 10:
            insights.append("📊 Low decision volume - system may need more traffic")
        elif total_decisions > 1000:
            insights.append("🚀 High decision volume - system scaling well")
        
        return insights[:5]  # Limit to top 5 insights
    
    def _is_cache_valid(self) -> bool:
        """Check if metrics cache is still valid"""
        if not self.cache_timestamp:
            return False
        
        return (datetime.now() - self.cache_timestamp).seconds < self.cache_duration
    
    def get_agent_leaderboard(self, metric: str = 'conversion_rate', time_period: str = '24h') -> List[Dict[str, Any]]:
        """Get agent performance leaderboard"""
        
        agents = [
            'DonationLikelyScoreAgent',
            'MLDonationLikelyScoreAgent',
            'LocalCauseRecommenderAgent', 
            'DonationAmountOptimizerAgent',
            'MLDonationAmountOptimizerAgent'
        ]
        
        leaderboard = []
        
        for agent in agents:
            metrics = self.calculate_agent_performance(agent, time_period)
            
            if metrics.get('total_decisions', 0) > 0:  # Only include active agents
                leaderboard.append({
                    'agent': agent,
                    'score': metrics.get(metric, 0),
                    'decisions': metrics.get('total_decisions', 0)
                })
        
        # Sort by score
        leaderboard.sort(key=lambda x: x['score'], reverse=True)
        
        return leaderboard
    
    def export_performance_data(self, output_path: str, time_period: str = '7d'):
        """Export performance data to CSV for analysis"""
        
        conn = sqlite3.connect(self.db_path)
        
        # Define time window
        if time_period == '24h':
            start_time = datetime.now() - timedelta(hours=24)
        elif time_period == '7d':
            start_time = datetime.now() - timedelta(days=7)
        elif time_period == '30d':
            start_time = datetime.now() - timedelta(days=30)
        else:
            start_time = datetime.now() - timedelta(days=7)
        
        # Export decisions
        decisions_df = pd.read_sql_query('''
            SELECT * FROM agent_decisions 
            WHERE timestamp >= ?
            ORDER BY timestamp DESC
        ''', conn, params=(start_time,))
        
        decisions_df.to_csv(f"{output_path}_decisions.csv", index=False)
        
        # Export responses
        responses_df = pd.read_sql_query('''
            SELECT * FROM user_responses 
            WHERE timestamp >= ?
            ORDER BY timestamp DESC
        ''', conn, params=(start_time,))
        
        responses_df.to_csv(f"{output_path}_responses.csv", index=False)
        
        conn.close()
        
        logger.info(f"Performance data exported to {output_path}_*.csv")


def main():
    """Demo of agent performance tracking"""
    
    tracker = AgentPerformanceTracker()
    
    # Generate sample performance data
    print("📊 Generating sample agent performance data...")
    
    # Simulate some agent decisions and user responses
    sample_agents = ['DonationLikelyScoreAgent', 'MLDonationLikelyScoreAgent']
    
    for i in range(100):
        agent = np.random.choice(sample_agents)
        user_id = f"user_{np.random.randint(1, 1000)}"
        
        # Sample context
        context = {
            'transaction_amount': np.random.uniform(100, 2000),
            'wallet_balance': np.random.uniform(500, 10000),
            'user_segment': np.random.choice(['new', 'regular', 'major'])
        }
        
        # Sample decision
        likelihood_score = np.random.uniform(0, 100)
        decision = {
            'likelihood_score': likelihood_score,
            'recommendation': 'prompt_now' if likelihood_score > 60 else 'skip_prompt'
        }
        
        tracker.log_agent_decision(agent, user_id, context, decision)
        
        # Simulate user response
        if decision['recommendation'] == 'prompt_now':
            # 30% chance of donation
            if np.random.random() < 0.3:
                tracker.log_user_response(
                    user_id, f"prompt_{i}", 'donation_accepted', True, np.random.uniform(1, 20)
                )
    
    # Generate performance report
    print("\n📈 Generating performance report...")
    report = tracker.generate_performance_report('24h')
    
    print(f"\n🎯 Performance Report ({report['time_period']}):")
    print(f"Total Decisions: {report['overall_metrics']['total_decisions']}")
    print(f"Total Conversions: {report['overall_metrics']['total_conversions']}")
    print(f"Overall Conversion Rate: {report['overall_metrics']['overall_conversion_rate']:.1%}")
    
    print(f"\n💡 Key Insights:")
    for insight in report['insights']:
        print(f"  {insight}")
    
    # Show leaderboard
    print(f"\n🏆 Agent Leaderboard (Conversion Rate):")
    leaderboard = tracker.get_agent_leaderboard('conversion_rate', '24h')
    for i, entry in enumerate(leaderboard, 1):
        print(f"  {i}. {entry['agent']}: {entry['score']:.1%} ({entry['decisions']} decisions)")


if __name__ == "__main__":
    main()
