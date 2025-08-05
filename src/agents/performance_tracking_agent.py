"""
Performance Tracking Agent
Monitors and analyzes agent performance, learning from decisions and outcomes
Input-Tool Call-Output pattern implementation
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import logging
import json
import statistics
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class AgentDecision:
    """Input: Individual agent decision record"""
    decision_id: str
    agent_name: str
    user_id: str
    timestamp: datetime
    input_context: Dict[str, Any]
    decision_output: Dict[str, Any]
    confidence_score: float
    reasoning: List[str]

@dataclass
class UserFeedback:
    """Input: User response to agent decisions"""
    decision_id: str
    user_id: str
    feedback_type: str  # accepted, rejected, modified, ignored
    actual_amount: Optional[float]  # If modified
    selected_cause: Optional[str]   # If different from recommended
    timing_feedback: Optional[str]  # too_early, too_late, perfect
    satisfaction_score: Optional[int]  # 1-5 scale
    timestamp: datetime

@dataclass
class PerformanceMetrics:
    """Output: Agent performance metrics"""
    agent_name: str
    time_period: str
    total_decisions: int
    accuracy_rate: float
    acceptance_rate: float
    conversion_rate: float
    average_confidence: float
    improvement_trend: str
    key_insights: List[str]
    recommendations: List[str]

@dataclass
class LearningInsights:
    """Output: Learning insights from performance analysis"""
    successful_patterns: List[Dict[str, Any]]
    failure_patterns: List[Dict[str, Any]]
    user_segments: Dict[str, Dict[str, Any]]
    optimization_opportunities: List[str]
    model_drift_indicators: List[str]
    recommended_adjustments: List[str]

@dataclass
class PerformanceReport:
    """Complete output with comprehensive performance analysis"""
    report_id: str
    generated_at: datetime
    time_period: Dict[str, datetime]
    agent_metrics: List[PerformanceMetrics]
    cross_agent_analysis: Dict[str, Any]
    learning_insights: LearningInsights
    business_impact: Dict[str, float]
    recommendations: List[str]
    confidence_score: float

class PerformanceTrackingAgent:
    """
    Agent that tracks and analyzes performance using input-tool call-output pattern
    
    INPUT: AgentDecision records and UserFeedback data
    TOOL CALLS: Statistical analysis, pattern recognition, learning algorithms
    OUTPUT: PerformanceReport with insights and recommendations
    """
    
    def __init__(self):
        self.name = "PerformanceTrackingAgent"
        self.logger = logging.getLogger(f"ImpactSense.{self.name}")
        
        # Performance tracking configuration
        self.tracked_agents = [
            "DonationLikelihoodAgent",
            "CauseRecommendationAgent", 
            "AmountOptimizationAgent"
        ]
        
        # Metric thresholds
        self.performance_thresholds = {
            'good_accuracy': 0.75,
            'good_acceptance': 0.60,
            'good_conversion': 0.25,
            'min_confidence': 0.70
        }
        
        # Data storage (in production, this would be a database)
        self.decision_history: List[AgentDecision] = []
        self.feedback_history: List[UserFeedback] = []
    
    def process(self, analysis_period: Dict[str, datetime], 
                decisions: List[AgentDecision],
                feedback: List[UserFeedback]) -> PerformanceReport:
        """
        Main agent processing method - INPUT -> TOOL CALLS -> OUTPUT
        
        Args:
            analysis_period: Time period for analysis
            decisions: Agent decision records
            feedback: User feedback records
            
        Returns:
            PerformanceReport: Comprehensive performance analysis
        """
        self.logger.info(f"Analyzing performance for period {analysis_period}")
        
        # Update internal data
        self.decision_history.extend(decisions)
        self.feedback_history.extend(feedback)
        
        # TOOL CALLS - Various analysis methods
        agent_metrics = self._analyze_individual_agents(analysis_period)
        cross_agent_analysis = self._analyze_cross_agent_patterns(analysis_period)
        learning_insights = self._generate_learning_insights(analysis_period)
        business_impact = self._calculate_business_impact(analysis_period)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            agent_metrics, cross_agent_analysis, learning_insights
        )
        
        # Calculate overall confidence
        confidence = self._calculate_report_confidence(agent_metrics, feedback)
        
        # Create final report
        report = PerformanceReport(
            report_id=f"perf_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            generated_at=datetime.now(),
            time_period=analysis_period,
            agent_metrics=agent_metrics,
            cross_agent_analysis=cross_agent_analysis,
            learning_insights=learning_insights,
            business_impact=business_impact,
            recommendations=recommendations,
            confidence_score=confidence
        )
        
        # Log performance analysis
        self._log_performance_analysis(report)
        
        return report
    
    def _analyze_individual_agents(self, period: Dict[str, datetime]) -> List[PerformanceMetrics]:
        """TOOL CALL: Analyze performance of individual agents"""
        metrics_list = []
        
        for agent_name in self.tracked_agents:
            metrics = self._calculate_agent_metrics(agent_name, period)
            metrics_list.append(metrics)
        
        return metrics_list
    
    def _calculate_agent_metrics(self, agent_name: str, 
                                period: Dict[str, datetime]) -> PerformanceMetrics:
        """TOOL CALL: Calculate metrics for a specific agent"""
        
        # Filter decisions for this agent and period
        agent_decisions = [
            d for d in self.decision_history
            if (d.agent_name == agent_name and
                period['start'] <= d.timestamp <= period['end'])
        ]
        
        if not agent_decisions:
            return self._create_empty_metrics(agent_name, period)
        
        # Get corresponding feedback
        decision_ids = {d.decision_id for d in agent_decisions}
        agent_feedback = [
            f for f in self.feedback_history
            if f.decision_id in decision_ids
        ]
        
        # Calculate metrics
        total_decisions = len(agent_decisions)
        total_feedback = len(agent_feedback)
        
        # Accuracy rate (based on satisfaction scores)
        accuracy_rate = self._calculate_accuracy_rate(agent_feedback)
        
        # Acceptance rate
        acceptance_rate = self._calculate_acceptance_rate(agent_feedback, total_decisions)
        
        # Conversion rate (accepted and resulted in donation)
        conversion_rate = self._calculate_conversion_rate(agent_feedback, total_decisions)
        
        # Average confidence
        avg_confidence = statistics.mean([d.confidence_score for d in agent_decisions])
        
        # Improvement trend
        improvement_trend = self._calculate_improvement_trend(agent_name, period)
        
        # Generate insights and recommendations
        insights = self._generate_agent_insights(agent_name, agent_decisions, agent_feedback)
        recommendations = self._generate_agent_recommendations(
            agent_name, accuracy_rate, acceptance_rate, conversion_rate
        )
        
        return PerformanceMetrics(
            agent_name=agent_name,
            time_period=f"{period['start'].strftime('%Y-%m-%d')} to {period['end'].strftime('%Y-%m-%d')}",
            total_decisions=total_decisions,
            accuracy_rate=accuracy_rate,
            acceptance_rate=acceptance_rate,
            conversion_rate=conversion_rate,
            average_confidence=avg_confidence,
            improvement_trend=improvement_trend,
            key_insights=insights,
            recommendations=recommendations
        )
    
    def _calculate_accuracy_rate(self, feedback: List[UserFeedback]) -> float:
        """TOOL CALL: Calculate accuracy based on user satisfaction"""
        if not feedback:
            return 0.0
        
        satisfaction_scores = [f.satisfaction_score for f in feedback if f.satisfaction_score]
        if not satisfaction_scores:
            # Fallback to acceptance rate if no satisfaction scores
            accepted = sum(1 for f in feedback if f.feedback_type == 'accepted')
            return accepted / len(feedback)
        
        # Consider scores >= 4 as accurate
        accurate_predictions = sum(1 for score in satisfaction_scores if score >= 4)
        return accurate_predictions / len(satisfaction_scores)
    
    def _calculate_acceptance_rate(self, feedback: List[UserFeedback], 
                                 total_decisions: int) -> float:
        """TOOL CALL: Calculate user acceptance rate"""
        if total_decisions == 0:
            return 0.0
        
        accepted_feedback = [f for f in feedback if f.feedback_type in ['accepted', 'modified']]
        return len(accepted_feedback) / total_decisions
    
    def _calculate_conversion_rate(self, feedback: List[UserFeedback],
                                 total_decisions: int) -> float:
        """TOOL CALL: Calculate actual conversion to donations"""
        if total_decisions == 0:
            return 0.0
        
        # Count feedback that resulted in actual donations
        conversions = sum(1 for f in feedback 
                         if f.feedback_type in ['accepted', 'modified'] and
                         (f.actual_amount is not None and f.actual_amount > 0))
        
        return conversions / total_decisions
    
    def _calculate_improvement_trend(self, agent_name: str, 
                                   period: Dict[str, datetime]) -> str:
        """TOOL CALL: Calculate performance improvement trend"""
        
        # Get historical data for comparison
        period_days = (period['end'] - period['start']).days
        previous_start = period['start'] - timedelta(days=period_days)
        previous_end = period['start']
        
        current_metrics = self._get_period_metrics(agent_name, period['start'], period['end'])
        previous_metrics = self._get_period_metrics(agent_name, previous_start, previous_end)
        
        if not previous_metrics:
            return "insufficient_data"
        
        # Compare key metrics
        current_score = (current_metrics['accuracy'] + current_metrics['acceptance']) / 2
        previous_score = (previous_metrics['accuracy'] + previous_metrics['acceptance']) / 2
        
        improvement = current_score - previous_score
        
        if improvement > 0.05:
            return "improving"
        elif improvement < -0.05:
            return "declining"
        else:
            return "stable"
    
    def _get_period_metrics(self, agent_name: str, start: datetime, 
                          end: datetime) -> Optional[Dict[str, float]]:
        """TOOL CALL: Get basic metrics for a specific period"""
        
        period_decisions = [
            d for d in self.decision_history
            if (d.agent_name == agent_name and start <= d.timestamp <= end)
        ]
        
        if not period_decisions:
            return None
        
        decision_ids = {d.decision_id for d in period_decisions}
        period_feedback = [
            f for f in self.feedback_history
            if f.decision_id in decision_ids
        ]
        
        return {
            'accuracy': self._calculate_accuracy_rate(period_feedback),
            'acceptance': self._calculate_acceptance_rate(period_feedback, len(period_decisions)),
            'conversion': self._calculate_conversion_rate(period_feedback, len(period_decisions))
        }
    
    def _generate_agent_insights(self, agent_name: str, decisions: List[AgentDecision],
                               feedback: List[UserFeedback]) -> List[str]:
        """TOOL CALL: Generate insights specific to an agent"""
        insights = []
        
        if not decisions:
            return ["Insufficient data for analysis"]
        
        # Confidence analysis
        high_confidence_decisions = [d for d in decisions if d.confidence_score > 0.8]
        if high_confidence_decisions:
            high_conf_ids = {d.decision_id for d in high_confidence_decisions}
            high_conf_feedback = [f for f in feedback if f.decision_id in high_conf_ids]
            
            if high_conf_feedback:
                high_conf_accuracy = self._calculate_accuracy_rate(high_conf_feedback)
                insights.append(f"High confidence decisions show {high_conf_accuracy:.1%} accuracy")
        
        # Timing analysis
        timing_feedback = [f for f in feedback if f.timing_feedback]
        if timing_feedback:
            timing_issues = defaultdict(int)
            for f in timing_feedback:
                timing_issues[f.timing_feedback] += 1
            
            if timing_issues['too_early'] > len(timing_feedback) * 0.3:
                insights.append("Users frequently find prompts too early")
            elif timing_issues['too_late'] > len(timing_feedback) * 0.3:
                insights.append("Users frequently find prompts too late")
        
        # Agent-specific insights
        if agent_name == "DonationLikelihoodAgent":
            insights.extend(self._analyze_likelihood_agent_patterns(decisions, feedback))
        elif agent_name == "CauseRecommendationAgent":
            insights.extend(self._analyze_cause_agent_patterns(decisions, feedback))
        elif agent_name == "AmountOptimizationAgent":
            insights.extend(self._analyze_amount_agent_patterns(decisions, feedback))
        
        return insights
    
    def _analyze_likelihood_agent_patterns(self, decisions: List[AgentDecision],
                                         feedback: List[UserFeedback]) -> List[str]:
        """TOOL CALL: Analyze patterns specific to likelihood agent"""
        insights = []
        
        # Analyze score thresholds
        decision_scores = {}
        for d in decisions:
            likelihood_score = d.decision_output.get('likelihood_score', 0)
            decision_scores[d.decision_id] = likelihood_score
        
        # Group by score ranges
        high_score_decisions = {k: v for k, v in decision_scores.items() if v >= 70}
        medium_score_decisions = {k: v for k, v in decision_scores.items() if 50 <= v < 70}
        low_score_decisions = {k: v for k, v in decision_scores.items() if v < 50}
        
        # Analyze feedback for each group
        for score_range, decision_ids in [
            ("high", high_score_decisions.keys()),
            ("medium", medium_score_decisions.keys()),
            ("low", low_score_decisions.keys())
        ]:
            range_feedback = [f for f in feedback if f.decision_id in decision_ids]
            if range_feedback:
                acceptance = len([f for f in range_feedback if f.feedback_type == 'accepted'])
                total = len(range_feedback)
                insights.append(f"{score_range.title()} likelihood scores: {acceptance/total:.1%} acceptance")
        
        return insights
    
    def _analyze_cause_agent_patterns(self, decisions: List[AgentDecision],
                                    feedback: List[UserFeedback]) -> List[str]:
        """TOOL CALL: Analyze patterns specific to cause recommendation agent"""
        insights = []
        
        # Analyze cause category preferences
        cause_performance = defaultdict(list)
        
        for d in decisions:
            primary_cause = d.decision_output.get('primary_recommendation', {}).get('ngo', {}).get('cause_category')
            if primary_cause:
                decision_feedback = [f for f in feedback if f.decision_id == d.decision_id]
                if decision_feedback:
                    was_accepted = decision_feedback[0].feedback_type in ['accepted', 'modified']
                    cause_performance[primary_cause].append(was_accepted)
        
        # Find best and worst performing causes
        cause_rates = {
            cause: sum(results) / len(results)
            for cause, results in cause_performance.items()
            if len(results) >= 3  # At least 3 decisions
        }
        
        if cause_rates:
            best_cause = max(cause_rates, key=cause_rates.get)
            worst_cause = min(cause_rates, key=cause_rates.get)
            
            insights.append(f"Best performing cause: {best_cause} ({cause_rates[best_cause]:.1%})")
            insights.append(f"Needs improvement: {worst_cause} ({cause_rates[worst_cause]:.1%})")
        
        return insights
    
    def _analyze_amount_agent_patterns(self, decisions: List[AgentDecision],
                                     feedback: List[UserFeedback]) -> List[str]:
        """TOOL CALL: Analyze patterns specific to amount optimization agent"""
        insights = []
        
        # Analyze amount accuracy
        amount_predictions = {}
        actual_amounts = {}
        
        for d in decisions:
            suggested_amount = d.decision_output.get('primary_amount', {}).get('amount')
            if suggested_amount:
                amount_predictions[d.decision_id] = suggested_amount
        
        for f in feedback:
            if f.actual_amount and f.decision_id in amount_predictions:
                actual_amounts[f.decision_id] = f.actual_amount
        
        if actual_amounts:
            # Calculate prediction accuracy
            prediction_errors = []
            for decision_id in actual_amounts:
                predicted = amount_predictions[decision_id]
                actual = actual_amounts[decision_id]
                error = abs(predicted - actual) / actual
                prediction_errors.append(error)
            
            avg_error = statistics.mean(prediction_errors)
            insights.append(f"Amount prediction accuracy: {(1-avg_error):.1%}")
            
            # Analyze over/under prediction tendencies
            over_predictions = sum(1 for decision_id in actual_amounts
                                 if amount_predictions[decision_id] > actual_amounts[decision_id])
            total_predictions = len(actual_amounts)
            
            if over_predictions > total_predictions * 0.6:
                insights.append("Tendency to over-estimate amounts")
            elif over_predictions < total_predictions * 0.4:
                insights.append("Tendency to under-estimate amounts")
        
        return insights
    
    def _generate_agent_recommendations(self, agent_name: str, accuracy: float,
                                      acceptance: float, conversion: float) -> List[str]:
        """TOOL CALL: Generate recommendations for agent improvement"""
        recommendations = []
        
        # General performance recommendations
        if accuracy < self.performance_thresholds['good_accuracy']:
            recommendations.append("Improve prediction accuracy through better feature engineering")
        
        if acceptance < self.performance_thresholds['good_acceptance']:
            recommendations.append("Review user feedback to identify rejection patterns")
        
        if conversion < self.performance_thresholds['good_conversion']:
            recommendations.append("Optimize for actual conversion, not just acceptance")
        
        # Agent-specific recommendations
        if agent_name == "DonationLikelihoodAgent":
            if acceptance < 0.5:
                recommendations.append("Recalibrate likelihood score thresholds")
            recommendations.append("Consider user fatigue and timing preferences")
        
        elif agent_name == "CauseRecommendationAgent":
            if acceptance < 0.6:
                recommendations.append("Improve cause-user preference matching")
            recommendations.append("Update NGO database and relevance scores")
        
        elif agent_name == "AmountOptimizationAgent":
            if conversion < 0.3:
                recommendations.append("Review amount optimization strategy")
            recommendations.append("Test different psychological anchoring approaches")
        
        return recommendations
    
    def _analyze_cross_agent_patterns(self, period: Dict[str, datetime]) -> Dict[str, Any]:
        """TOOL CALL: Analyze patterns across multiple agents"""
        
        # Find decisions where multiple agents worked together
        period_decisions = [
            d for d in self.decision_history
            if period['start'] <= d.timestamp <= period['end']
        ]
        
        # Group by user and time to find related decisions
        user_sessions = defaultdict(list)
        for decision in period_decisions:
            session_key = f"{decision.user_id}_{decision.timestamp.strftime('%Y%m%d_%H')}"
            user_sessions[session_key].append(decision)
        
        # Analyze multi-agent sessions
        multi_agent_sessions = [
            session for session in user_sessions.values()
            if len(set(d.agent_name for d in session)) > 1
        ]
        
        cross_agent_analysis = {
            'total_sessions': len(user_sessions),
            'multi_agent_sessions': len(multi_agent_sessions),
            'agent_combinations': defaultdict(int),
            'combination_success_rates': {},
            'sequential_patterns': []
        }
        
        # Analyze agent combinations
        for session in multi_agent_sessions:
            agents_in_session = sorted(set(d.agent_name for d in session))
            combination = " + ".join(agents_in_session)
            cross_agent_analysis['agent_combinations'][combination] += 1
        
        # Calculate success rates for combinations
        for combination, count in cross_agent_analysis['agent_combinations'].items():
            if count >= 3:  # Only analyze combinations with sufficient data
                # This would require more complex analysis in production
                cross_agent_analysis['combination_success_rates'][combination] = 0.65  # Placeholder
        
        return cross_agent_analysis
    
    def _generate_learning_insights(self, period: Dict[str, datetime]) -> LearningInsights:
        """TOOL CALL: Generate learning insights from performance data"""
        
        period_decisions = [
            d for d in self.decision_history
            if period['start'] <= d.timestamp <= period['end']
        ]
        
        decision_ids = {d.decision_id for d in period_decisions}
        period_feedback = [
            f for f in self.feedback_history
            if f.decision_id in decision_ids
        ]
        
        # Identify successful patterns
        successful_patterns = self._identify_successful_patterns(period_decisions, period_feedback)
        
        # Identify failure patterns
        failure_patterns = self._identify_failure_patterns(period_decisions, period_feedback)
        
        # Analyze user segments
        user_segments = self._analyze_user_segments(period_decisions, period_feedback)
        
        # Identify optimization opportunities
        optimization_opportunities = self._identify_optimization_opportunities(
            successful_patterns, failure_patterns
        )
        
        # Check for model drift
        model_drift_indicators = self._check_model_drift(period)
        
        # Generate recommended adjustments
        recommended_adjustments = self._generate_adjustment_recommendations(
            successful_patterns, failure_patterns, model_drift_indicators
        )
        
        return LearningInsights(
            successful_patterns=successful_patterns,
            failure_patterns=failure_patterns,
            user_segments=user_segments,
            optimization_opportunities=optimization_opportunities,
            model_drift_indicators=model_drift_indicators,
            recommended_adjustments=recommended_adjustments
        )
    
    def _identify_successful_patterns(self, decisions: List[AgentDecision],
                                    feedback: List[UserFeedback]) -> List[Dict[str, Any]]:
        """TOOL CALL: Identify patterns in successful decisions"""
        
        successful_decisions = []
        for decision in decisions:
            decision_feedback = [f for f in feedback if f.decision_id == decision.decision_id]
            if decision_feedback and decision_feedback[0].feedback_type in ['accepted', 'modified']:
                successful_decisions.append(decision)
        
        patterns = []
        
        # Pattern 1: High confidence + High accuracy
        high_conf_successful = [d for d in successful_decisions if d.confidence_score > 0.8]
        if len(high_conf_successful) > len(successful_decisions) * 0.3:
            patterns.append({
                'pattern': 'high_confidence_success',
                'description': 'High confidence decisions show higher success rates',
                'frequency': len(high_conf_successful),
                'success_rate': len(high_conf_successful) / len(successful_decisions)
            })
        
        # Pattern 2: Specific reasoning patterns
        reasoning_patterns = defaultdict(int)
        for decision in successful_decisions:
            for reason in decision.reasoning:
                reasoning_patterns[reason] += 1
        
        top_reasons = sorted(reasoning_patterns.items(), key=lambda x: x[1], reverse=True)[:3]
        for reason, count in top_reasons:
            if count > len(successful_decisions) * 0.2:
                patterns.append({
                    'pattern': 'successful_reasoning',
                    'description': f'Decisions with reasoning "{reason}" show high success',
                    'frequency': count,
                    'success_rate': count / len(successful_decisions)
                })
        
        return patterns
    
    def _identify_failure_patterns(self, decisions: List[AgentDecision],
                                 feedback: List[UserFeedback]) -> List[Dict[str, Any]]:
        """TOOL CALL: Identify patterns in failed decisions"""
        
        failed_decisions = []
        for decision in decisions:
            decision_feedback = [f for f in feedback if f.decision_id == decision.decision_id]
            if decision_feedback and decision_feedback[0].feedback_type in ['rejected', 'ignored']:
                failed_decisions.append(decision)
        
        patterns = []
        
        if not failed_decisions:
            return patterns
        
        # Pattern 1: Low confidence failures
        low_conf_failed = [d for d in failed_decisions if d.confidence_score < 0.5]
        if len(low_conf_failed) > len(failed_decisions) * 0.3:
            patterns.append({
                'pattern': 'low_confidence_failure',
                'description': 'Low confidence decisions more likely to fail',
                'frequency': len(low_conf_failed),
                'failure_rate': len(low_conf_failed) / len(failed_decisions)
            })
        
        # Pattern 2: Common failure reasons
        reasoning_patterns = defaultdict(int)
        for decision in failed_decisions:
            for reason in decision.reasoning:
                reasoning_patterns[reason] += 1
        
        top_failure_reasons = sorted(reasoning_patterns.items(), key=lambda x: x[1], reverse=True)[:3]
        for reason, count in top_failure_reasons:
            if count > len(failed_decisions) * 0.2:
                patterns.append({
                    'pattern': 'failure_reasoning',
                    'description': f'Decisions with reasoning "{reason}" often fail',
                    'frequency': count,
                    'failure_rate': count / len(failed_decisions)
                })
        
        return patterns
    
    def _analyze_user_segments(self, decisions: List[AgentDecision],
                             feedback: List[UserFeedback]) -> Dict[str, Dict[str, Any]]:
        """TOOL CALL: Analyze performance across user segments"""
        
        # Group decisions by user characteristics (simplified)
        segments = {
            'high_value_users': [],
            'frequent_users': [],
            'new_users': []
        }
        
        # This is a simplified segmentation - in production, this would use more sophisticated user profiling
        user_decision_counts = defaultdict(int)
        for decision in decisions:
            user_decision_counts[decision.user_id] += 1
        
        for decision in decisions:
            if user_decision_counts[decision.user_id] > 5:
                segments['frequent_users'].append(decision)
            elif user_decision_counts[decision.user_id] == 1:
                segments['new_users'].append(decision)
            else:
                segments['high_value_users'].append(decision)  # Default category
        
        segment_analysis = {}
        for segment_name, segment_decisions in segments.items():
            if segment_decisions:
                segment_feedback = [
                    f for f in feedback 
                    if f.decision_id in {d.decision_id for d in segment_decisions}
                ]
                
                segment_analysis[segment_name] = {
                    'total_decisions': len(segment_decisions),
                    'acceptance_rate': self._calculate_acceptance_rate(segment_feedback, len(segment_decisions)),
                    'conversion_rate': self._calculate_conversion_rate(segment_feedback, len(segment_decisions)),
                    'avg_confidence': statistics.mean([d.confidence_score for d in segment_decisions])
                }
        
        return segment_analysis
    
    def _identify_optimization_opportunities(self, successful_patterns: List[Dict],
                                           failure_patterns: List[Dict]) -> List[str]:
        """TOOL CALL: Identify optimization opportunities"""
        opportunities = []
        
        # Opportunities from successful patterns
        for pattern in successful_patterns:
            if pattern['pattern'] == 'high_confidence_success':
                opportunities.append("Increase confidence thresholds for decision making")
            elif pattern['pattern'] == 'successful_reasoning':
                opportunities.append(f"Emphasize reasoning pattern: {pattern['description']}")
        
        # Opportunities from failure patterns
        for pattern in failure_patterns:
            if pattern['pattern'] == 'low_confidence_failure':
                opportunities.append("Improve low-confidence decision handling")
            elif pattern['pattern'] == 'failure_reasoning':
                opportunities.append(f"Avoid reasoning pattern: {pattern['description']}")
        
        # General opportunities
        opportunities.extend([
            "Implement A/B testing for optimization strategies",
            "Enhance user feedback collection mechanisms",
            "Develop real-time model adjustment capabilities"
        ])
        
        return opportunities
    
    def _check_model_drift(self, period: Dict[str, datetime]) -> List[str]:
        """TOOL CALL: Check for signs of model drift"""
        indicators = []
        
        # Compare recent performance to historical baselines
        recent_start = period['end'] - timedelta(days=7)
        recent_decisions = [
            d for d in self.decision_history
            if recent_start <= d.timestamp <= period['end']
        ]
        
        if len(recent_decisions) > 10:  # Sufficient data
            recent_avg_confidence = statistics.mean([d.confidence_score for d in recent_decisions])
            
            # Historical baseline (simplified)
            historical_decisions = [
                d for d in self.decision_history
                if d.timestamp < recent_start
            ]
            
            if historical_decisions:
                historical_avg_confidence = statistics.mean([d.confidence_score for d in historical_decisions])
                
                confidence_drift = abs(recent_avg_confidence - historical_avg_confidence)
                if confidence_drift > 0.1:
                    indicators.append(f"Confidence score drift detected: {confidence_drift:.2f}")
        
        # Check for unusual patterns (simplified)
        if len(recent_decisions) > 0:
            recent_agent_distribution = defaultdict(int)
            for decision in recent_decisions:
                recent_agent_distribution[decision.agent_name] += 1
            
            # If one agent is dominating decisions unusually
            total_recent = len(recent_decisions)
            for agent, count in recent_agent_distribution.items():
                if count / total_recent > 0.8:
                    indicators.append(f"Unusual agent dominance: {agent} ({count/total_recent:.1%})")
        
        return indicators
    
    def _generate_adjustment_recommendations(self, successful_patterns: List[Dict],
                                           failure_patterns: List[Dict],
                                           drift_indicators: List[str]) -> List[str]:
        """TOOL CALL: Generate specific adjustment recommendations"""
        recommendations = []
        
        # Recommendations based on successful patterns
        for pattern in successful_patterns[:2]:  # Top 2 patterns
            if 'high_confidence' in pattern['pattern']:
                recommendations.append("Increase minimum confidence threshold for prompting")
            else:
                recommendations.append(f"Reinforce successful pattern: {pattern['description']}")
        
        # Recommendations based on failure patterns
        for pattern in failure_patterns[:2]:  # Top 2 patterns
            recommendations.append(f"Address failure pattern: {pattern['description']}")
        
        # Recommendations based on drift
        if drift_indicators:
            recommendations.append("Schedule model retraining due to detected drift")
            recommendations.append("Review and update feature importance weights")
        
        # General recommendations
        recommendations.extend([
            "Implement continuous learning pipeline",
            "Enhance user feedback integration",
            "Develop automated performance monitoring alerts"
        ])
        
        return recommendations[:8]  # Limit to most important recommendations
    
    def _calculate_business_impact(self, period: Dict[str, datetime]) -> Dict[str, float]:
        """TOOL CALL: Calculate business impact metrics"""
        
        period_feedback = [
            f for f in self.feedback_history
            if period['start'] <= f.timestamp <= period['end']
        ]
        
        # Calculate total donations generated
        total_donations = sum(
            f.actual_amount for f in period_feedback
            if f.actual_amount and f.feedback_type in ['accepted', 'modified']
        )
        
        # Calculate number of new donors
        unique_donors = len(set(
            f.user_id for f in period_feedback
            if f.feedback_type in ['accepted', 'modified']
        ))
        
        # Calculate engagement metrics
        total_prompts = len([f for f in period_feedback if f.feedback_type != 'ignored'])
        engagement_rate = len(period_feedback) / max(1, total_prompts)
        
        return {
            'total_donations_generated': total_donations,
            'number_of_donors': unique_donors,
            'average_donation_amount': total_donations / max(1, unique_donors),
            'engagement_rate': engagement_rate,
            'conversion_rate': len([f for f in period_feedback if f.feedback_type in ['accepted', 'modified']]) / max(1, len(period_feedback))
        }
    
    def _generate_recommendations(self, agent_metrics: List[PerformanceMetrics],
                                cross_agent_analysis: Dict[str, Any],
                                learning_insights: LearningInsights) -> List[str]:
        """TOOL CALL: Generate high-level recommendations"""
        recommendations = []
        
        # Agent-specific recommendations
        for metrics in agent_metrics:
            if metrics.accuracy_rate < 0.7:
                recommendations.append(f"Priority: Improve {metrics.agent_name} accuracy")
            if metrics.improvement_trend == "declining":
                recommendations.append(f"Investigate declining performance in {metrics.agent_name}")
        
        # Cross-agent recommendations
        if cross_agent_analysis['multi_agent_sessions'] > 0:
            recommendations.append("Optimize multi-agent collaboration workflows")
        
        # Learning-based recommendations
        if learning_insights.model_drift_indicators:
            recommendations.append("Address detected model drift through retraining")
        
        if learning_insights.optimization_opportunities:
            recommendations.append(f"Implement top optimization: {learning_insights.optimization_opportunities[0]}")
        
        return recommendations[:6]  # Top 6 recommendations
    
    def _calculate_report_confidence(self, agent_metrics: List[PerformanceMetrics],
                                   feedback: List[UserFeedback]) -> float:
        """TOOL CALL: Calculate confidence in the performance report"""
        
        base_confidence = 60.0
        
        # Add confidence based on data volume
        total_decisions = sum(m.total_decisions for m in agent_metrics)
        if total_decisions > 100:
            base_confidence += 20
        elif total_decisions > 50:
            base_confidence += 10
        
        # Add confidence based on feedback coverage
        if len(feedback) > total_decisions * 0.5:  # Good feedback coverage
            base_confidence += 15
        
        # Reduce confidence if too few agents have sufficient data
        agents_with_data = sum(1 for m in agent_metrics if m.total_decisions > 10)
        if agents_with_data < len(agent_metrics):
            base_confidence -= 10
        
        return min(95.0, max(30.0, base_confidence))
    
    def _create_empty_metrics(self, agent_name: str, period: Dict[str, datetime]) -> PerformanceMetrics:
        """TOOL CALL: Create empty metrics for agents with no data"""
        return PerformanceMetrics(
            agent_name=agent_name,
            time_period=f"{period['start'].strftime('%Y-%m-%d')} to {period['end'].strftime('%Y-%m-%d')}",
            total_decisions=0,
            accuracy_rate=0.0,
            acceptance_rate=0.0,
            conversion_rate=0.0,
            average_confidence=0.0,
            improvement_trend="no_data",
            key_insights=["Insufficient data for analysis"],
            recommendations=["Increase decision volume to enable analysis"]
        )
    
    def _log_performance_analysis(self, report: PerformanceReport):
        """TOOL CALL: Log performance analysis for debugging"""
        log_data = {
            'report_id': report.report_id,
            'time_period': {
                'start': report.time_period['start'].isoformat(),
                'end': report.time_period['end'].isoformat()
            },
            'total_agents_analyzed': len(report.agent_metrics),
            'business_impact': report.business_impact,
            'confidence_score': report.confidence_score,
            'top_recommendations': report.recommendations[:3]
        }
        
        self.logger.info(f"Performance analysis completed: {json.dumps(log_data, indent=2)}")

# Example usage and testing
if __name__ == "__main__":
    agent = PerformanceTrackingAgent()
    
    # Test with sample data
    sample_decisions = [
        AgentDecision(
            decision_id="dec_001",
            agent_name="DonationLikelihoodAgent",
            user_id="user_123",
            timestamp=datetime.now() - timedelta(days=1),
            input_context={"transaction_amount": 250.0},
            decision_output={"likelihood_score": 75.0, "should_prompt": True},
            confidence_score=0.8,
            reasoning=["sufficient_cooldown_period", "healthy_wallet_balance"]
        )
    ]
    
    sample_feedback = [
        UserFeedback(
            decision_id="dec_001",
            user_id="user_123",
            feedback_type="accepted",
            actual_amount=25.0,
            selected_cause=None,
            timing_feedback="perfect",
            satisfaction_score=4,
            timestamp=datetime.now()
        )
    ]
    
    analysis_period = {
        'start': datetime.now() - timedelta(days=7),
        'end': datetime.now()
    }
    
    report = agent.process(analysis_period, sample_decisions, sample_feedback)
    print(f"Report ID: {report.report_id}")
    print(f"Confidence: {report.confidence_score}%")
    print(f"Recommendations: {report.recommendations[:3]}")
