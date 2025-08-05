"""
Donation Likelihood Agent
Analyzes user context to determine optimal timing and likelihood for donation prompts
Input-Tool Call-Output pattern implementation
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import logging
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class UserContext:
    """Input data structure for the agent"""
    user_id: str
    transaction_amount: float
    wallet_balance: float
    location: Dict[str, str]
    transaction_category: str
    days_since_last_prompt: int
    days_since_last_donation: int
    average_donation_amount: float
    total_lifetime_donations: float
    preferred_causes: List[str]
    notification_preferences: Dict[str, bool]
    demographic_hints: Dict[str, Any]
    time_of_day: str
    day_of_week: str

@dataclass
class LikelihoodScore:
    """Output data structure from the agent"""
    likelihood_score: float  # 0-100
    should_prompt: bool
    optimal_timing: str
    reasoning: List[str]
    confidence: float
    next_optimal_time: Optional[datetime]
    risk_factors: List[str]
    opportunity_factors: List[str]

class DonationLikelihoodAgent:
    """
    Agent that determines donation likelihood using input-tool call-output pattern
    
    INPUT: UserContext with transaction and behavioral data
    TOOL CALLS: Internal analysis methods (behavioral analysis, timing analysis, etc.)
    OUTPUT: LikelihoodScore with recommendation and reasoning
    """
    
    def __init__(self):
        self.name = "DonationLikelihoodAgent"
        self.logger = logging.getLogger(f"ImpactSense.{self.name}")
        
        # Agent configuration
        self.cooldown_rules = {
            'declined_prompt': 14,  # days
            'accepted_prompt': 7,   # days  
            'low_wallet': 7,        # days
        }
        
        self.scoring_weights = {
            'timing_factor': 0.25,
            'financial_factor': 0.30,
            'behavioral_factor': 0.25,
            'contextual_factor': 0.20
        }
    
    def process(self, context: UserContext) -> LikelihoodScore:
        """
        Main agent processing method - implements INPUT -> TOOL CALLS -> OUTPUT pattern
        
        Args:
            context (UserContext): Input data for analysis
            
        Returns:
            LikelihoodScore: Complete recommendation with reasoning
        """
        self.logger.info(f"Processing donation likelihood for user {context.user_id}")
        
        # TOOL CALLS - Internal analysis methods
        timing_analysis = self._analyze_timing(context)
        financial_analysis = self._analyze_financial_capacity(context)
        behavioral_analysis = self._analyze_behavioral_patterns(context)
        contextual_analysis = self._analyze_contextual_factors(context)
        
        # Combine all analyses
        final_score = self._calculate_final_score(
            timing_analysis, financial_analysis, behavioral_analysis, contextual_analysis
        )
        
        # Make recommendation decision
        recommendation = self._make_recommendation(final_score, context)
        
        # Log decision for learning
        self._log_decision(context, recommendation)
        
        return recommendation
    
    def _analyze_timing(self, context: UserContext) -> Dict[str, Any]:
        """TOOL CALL: Analyze timing factors for donation prompts"""
        score = 0
        factors = []
        risks = []
        opportunities = []
        
        # Recent prompt behavior
        if context.days_since_last_prompt >= 6:
            score += 30
            factors.append("sufficient_cooldown_period")
            opportunities.append("User hasn't been prompted recently")
        elif context.days_since_last_prompt < 3:
            score -= 20
            factors.append("too_recent_prompt")
            risks.append("Recent prompt may cause fatigue")
        
        # Time-based patterns
        if context.time_of_day in ['morning', 'evening']:
            score += 10
            factors.append("optimal_time_of_day")
        
        if context.day_of_week in ['friday', 'saturday', 'sunday']:
            score += 5
            factors.append("weekend_generosity")
        
        # Days since last donation
        if 7 <= context.days_since_last_donation <= 30:
            score += 15
            factors.append("donation_rhythm_optimal")
            opportunities.append("User in good donation rhythm")
        elif context.days_since_last_donation > 60:
            score += 20
            factors.append("reengagement_opportunity")
            opportunities.append("Good opportunity to re-engage user")
        
        return {
            'score': max(0, min(100, score)),
            'factors': factors,
            'risks': risks,
            'opportunities': opportunities
        }
    
    def _analyze_financial_capacity(self, context: UserContext) -> Dict[str, Any]:
        """TOOL CALL: Analyze user's financial capacity for donations"""
        score = 0
        factors = []
        risks = []
        opportunities = []
        
        # Transaction size consideration
        if context.transaction_amount > 500:
            score += 25
            factors.append("large_transaction_generosity")
            opportunities.append("Large transaction suggests financial capacity")
        elif context.transaction_amount < 50:
            score -= 15
            factors.append("small_transaction_constraint")
            risks.append("Small transaction may indicate budget constraints")
        
        # Wallet balance analysis
        wallet_ratio = context.wallet_balance / max(1, context.transaction_amount)
        if wallet_ratio > 10:
            score += 20
            factors.append("healthy_wallet_balance")
            opportunities.append("Strong wallet balance supports donation")
        elif wallet_ratio < 2:
            score -= 25
            factors.append("low_wallet_balance")
            risks.append("Low wallet balance may discourage donation")
        
        # Average donation behavior
        if context.average_donation_amount > 0:
            donation_to_transaction_ratio = context.average_donation_amount / context.transaction_amount
            if 0.01 <= donation_to_transaction_ratio <= 0.1:  # 1-10% is reasonable
                score += 15
                factors.append("reasonable_donation_habit")
            elif donation_to_transaction_ratio > 0.1:
                score += 10
                factors.append("generous_donation_habit")
        
        return {
            'score': max(0, min(100, score)),
            'factors': factors,
            'risks': risks,
            'opportunities': opportunities
        }
    
    def _analyze_behavioral_patterns(self, context: UserContext) -> Dict[str, Any]:
        """TOOL CALL: Analyze user's behavioral patterns and preferences"""
        score = 0
        factors = []
        risks = []
        opportunities = []
        
        # Lifetime donation behavior
        if context.total_lifetime_donations > 1000:
            score += 25
            factors.append("established_donor")
            opportunities.append("User has strong donation history")
        elif context.total_lifetime_donations > 100:
            score += 15
            factors.append("regular_donor")
        elif context.total_lifetime_donations == 0:
            score += 10  # First-time opportunity
            factors.append("first_time_opportunity")
            opportunities.append("Opportunity to convert new donor")
        
        # Preferred causes alignment
        if len(context.preferred_causes) > 0:
            score += 10
            factors.append("has_cause_preferences")
            opportunities.append("User has expressed cause preferences")
        
        # Notification preferences
        if context.notification_preferences.get('donation_prompts', True):
            score += 15
            factors.append("open_to_prompts")
        else:
            score -= 30
            factors.append("prompt_averse")
            risks.append("User has opted out of donation prompts")
        
        # Transaction category relevance
        relevant_categories = ['food', 'healthcare', 'education', 'utilities']
        if context.transaction_category.lower() in relevant_categories:
            score += 10
            factors.append("relevant_transaction_category")
            opportunities.append("Transaction category aligns with charitable mindset")
        
        return {
            'score': max(0, min(100, score)),
            'factors': factors,
            'risks': risks,
            'opportunities': opportunities
        }
    
    def _analyze_contextual_factors(self, context: UserContext) -> Dict[str, Any]:
        """TOOL CALL: Analyze contextual factors like location and demographics"""
        score = 0
        factors = []
        risks = []
        opportunities = []
        
        # Location-based analysis
        if context.location.get('region') == 'NCR':
            score += 5
            factors.append("metro_location")
        
        # Demographic factors
        age_group = context.demographic_hints.get('age_group', 'unknown')
        if age_group in ['millennial', 'gen_z']:
            score += 10
            factors.append("socially_conscious_demographic")
            opportunities.append("Demographic typically interested in social causes")
        elif age_group in ['gen_x', 'boomer']:
            score += 15
            factors.append("established_demographic")
            opportunities.append("Demographic with established giving patterns")
        
        # Income indicators
        income_level = context.demographic_hints.get('income_level', 'unknown')
        if income_level in ['middle', 'upper_middle', 'high']:
            score += 10
            factors.append("sufficient_income_indicators")
        elif income_level == 'low':
            score -= 10
            factors.append("budget_conscious_demographic")
            risks.append("Lower income may limit donation capacity")
        
        return {
            'score': max(0, min(100, score)),
            'factors': factors,
            'risks': risks,
            'opportunities': opportunities
        }
    
    def _calculate_final_score(self, timing: Dict, financial: Dict, 
                             behavioral: Dict, contextual: Dict) -> Dict[str, Any]:
        """TOOL CALL: Calculate weighted final score from all analyses"""
        
        # Apply weights to each component
        weighted_score = (
            timing['score'] * self.scoring_weights['timing_factor'] +
            financial['score'] * self.scoring_weights['financial_factor'] +
            behavioral['score'] * self.scoring_weights['behavioral_factor'] +
            contextual['score'] * self.scoring_weights['contextual_factor']
        )
        
        # Compile all factors
        all_factors = (
            timing['factors'] + financial['factors'] + 
            behavioral['factors'] + contextual['factors']
        )
        
        all_risks = (
            timing['risks'] + financial['risks'] + 
            behavioral['risks'] + contextual['risks']
        )
        
        all_opportunities = (
            timing['opportunities'] + financial['opportunities'] + 
            behavioral['opportunities'] + contextual['opportunities']
        )
        
        # Calculate confidence based on number of positive factors
        confidence = min(95, max(10, len(all_factors) * 8))
        
        return {
            'final_score': round(weighted_score, 2),
            'component_scores': {
                'timing': timing['score'],
                'financial': financial['score'],
                'behavioral': behavioral['score'],
                'contextual': contextual['score']
            },
            'all_factors': all_factors,
            'all_risks': all_risks,
            'all_opportunities': all_opportunities,
            'confidence': confidence
        }
    
    def _make_recommendation(self, analysis: Dict[str, Any], context: UserContext) -> LikelihoodScore:
        """TOOL CALL: Make final recommendation based on analysis"""
        
        score = analysis['final_score']
        
        # Determine if we should prompt
        should_prompt = score >= 60 and len(analysis['all_risks']) <= 2
        
        # Determine optimal timing
        if should_prompt:
            if context.time_of_day in ['evening']:
                optimal_timing = "immediate"
            else:
                optimal_timing = "next_evening"
        else:
            optimal_timing = "wait_for_better_conditions"
        
        # Calculate next optimal time
        next_optimal_time = None
        if not should_prompt:
            days_to_wait = max(3, 14 - context.days_since_last_prompt)
            next_optimal_time = datetime.now() + timedelta(days=days_to_wait)
        
        return LikelihoodScore(
            likelihood_score=score,
            should_prompt=should_prompt,
            optimal_timing=optimal_timing,
            reasoning=analysis['all_factors'],
            confidence=analysis['confidence'],
            next_optimal_time=next_optimal_time,
            risk_factors=analysis['all_risks'],
            opportunity_factors=analysis['all_opportunities']
        )
    
    def _log_decision(self, context: UserContext, decision: LikelihoodScore):
        """TOOL CALL: Log decision for learning and debugging"""
        log_data = {
            'user_id': context.user_id,
            'transaction_amount': context.transaction_amount,
            'likelihood_score': decision.likelihood_score,
            'should_prompt': decision.should_prompt,
            'reasoning': decision.reasoning,
            'risk_factors': decision.risk_factors,
            'opportunity_factors': decision.opportunity_factors,
            'timestamp': datetime.now().isoformat()
        }
        
        self.logger.info(f"Decision logged: {json.dumps(log_data, indent=2)}")
        
        # In a production system, this would save to a database
        # For now, we'll just log it

# Example usage and testing
if __name__ == "__main__":
    # Test the agent with sample data
    agent = DonationLikelihoodAgent()
    
    test_context = UserContext(
        user_id="user_123",
        transaction_amount=250.0,
        wallet_balance=5000.0,
        location={"region": "NCR", "city": "Manila"},
        transaction_category="food",
        days_since_last_prompt=8,
        days_since_last_donation=15,
        average_donation_amount=25.0,
        total_lifetime_donations=500.0,
        preferred_causes=["education", "healthcare"],
        notification_preferences={"donation_prompts": True},
        demographic_hints={"age_group": "millennial", "income_level": "middle"},
        time_of_day="evening",
        day_of_week="friday"
    )
    
    result = agent.process(test_context)
    print(f"Likelihood Score: {result.likelihood_score}")
    print(f"Should Prompt: {result.should_prompt}")
    print(f"Reasoning: {result.reasoning}")
