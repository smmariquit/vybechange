"""
Amount Optimization Agent
Optimizes donation amounts based on user financial capacity and behavioral patterns
Input-Tool Call-Output pattern implementation
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import logging
import json
import math

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class FinancialContext:
    """Input: User's financial context for amount optimization"""
    user_id: str
    transaction_amount: float
    wallet_balance: float
    monthly_income_estimate: Optional[float]
    recent_transactions: List[Dict[str, Any]]
    donation_history: List[Dict[str, Any]]
    spending_patterns: Dict[str, float]
    savings_indicators: Dict[str, Any]
    payment_method: str
    time_since_payday: int  # days

@dataclass
class BehavioralProfile:
    """User's behavioral patterns for amount optimization"""
    donation_frequency: str  # frequent, occasional, rare, first_time
    average_donation_amount: float
    largest_donation: float
    smallest_donation: float
    preferred_amounts: List[float]
    response_to_suggestions: Dict[str, int]  # accepted, rejected, modified
    price_sensitivity: str  # high, medium, low
    generosity_score: float

@dataclass
class AmountSuggestion:
    """Single amount suggestion with reasoning"""
    amount: float
    confidence: float
    reasoning: List[str]
    psychological_anchor: str
    conversion_probability: float
    impact_description: str

@dataclass
class OptimizedAmounts:
    """Complete output with optimized amount suggestions"""
    primary_amount: AmountSuggestion
    alternative_amounts: List[AmountSuggestion]
    micro_amount: AmountSuggestion  # Very small amount option
    stretch_amount: AmountSuggestion  # Higher amount option
    optimization_strategy: str
    personalization_factors: List[str]
    a_b_test_variant: str
    confidence_score: float

class AmountOptimizationAgent:
    """
    Agent that optimizes donation amounts using input-tool call-output pattern
    
    INPUT: FinancialContext and BehavioralProfile
    TOOL CALLS: Financial analysis, behavioral modeling, optimization algorithms
    OUTPUT: OptimizedAmounts with multiple strategic suggestions
    """
    
    def __init__(self):
        self.name = "AmountOptimizationAgent"
        self.logger = logging.getLogger(f"ImpactSense.{self.name}")
        
        # Optimization strategies
        self.strategies = {
            'conservative': {'multiplier': 0.015, 'max_ratio': 0.02},
            'moderate': {'multiplier': 0.025, 'max_ratio': 0.035},
            'aggressive': {'multiplier': 0.04, 'max_ratio': 0.05}
        }
        
        # Psychological pricing anchors
        self.anchor_amounts = [10, 15, 20, 25, 30, 50, 75, 100, 150, 200, 250, 500]
        
        # A/B test variants
        self.ab_test_variants = ['rounded', 'precise', 'psychological']
    
    def process(self, financial_context: FinancialContext, 
                behavioral_profile: BehavioralProfile) -> OptimizedAmounts:
        """
        Main agent processing method - INPUT -> TOOL CALLS -> OUTPUT
        
        Args:
            financial_context (FinancialContext): User's financial situation
            behavioral_profile (BehavioralProfile): User's behavioral patterns
            
        Returns:
            OptimizedAmounts: Optimized amount suggestions with reasoning
        """
        self.logger.info(f"Optimizing amounts for user {financial_context.user_id}")
        
        # TOOL CALLS - Various analysis and optimization methods
        affordability_analysis = self._analyze_affordability(financial_context)
        behavioral_analysis = self._analyze_behavioral_patterns(behavioral_profile)
        psychological_analysis = self._analyze_psychological_factors(
            financial_context, behavioral_profile
        )
        
        # Determine optimization strategy
        strategy = self._select_optimization_strategy(
            affordability_analysis, behavioral_analysis
        )
        
        # Calculate base amount suggestions
        base_amounts = self._calculate_base_amounts(
            financial_context, behavioral_profile, strategy
        )
        
        # Apply psychological optimizations
        optimized_amounts = self._apply_psychological_optimizations(
            base_amounts, psychological_analysis
        )
        
        # Create final suggestions
        final_suggestions = self._create_final_suggestions(
            optimized_amounts, financial_context, behavioral_profile, strategy
        )
        
        # Log optimization decision
        self._log_optimization(financial_context, final_suggestions)
        
        return final_suggestions
    
    def _analyze_affordability(self, context: FinancialContext) -> Dict[str, Any]:
        """TOOL CALL: Analyze user's financial capacity for donations"""
        analysis = {
            'disposable_income_indicators': {},
            'spending_capacity': {},
            'financial_stress_signals': [],
            'affordability_score': 0
        }
        
        # Wallet balance analysis
        transaction_to_balance_ratio = context.transaction_amount / max(1, context.wallet_balance)
        
        if transaction_to_balance_ratio < 0.1:  # Transaction is <10% of balance
            analysis['disposable_income_indicators']['healthy_balance'] = True
            analysis['affordability_score'] += 30
        elif transaction_to_balance_ratio > 0.5:  # Transaction is >50% of balance
            analysis['financial_stress_signals'].append('high_transaction_to_balance_ratio')
            analysis['affordability_score'] -= 20
        
        # Recent spending pattern analysis
        if context.recent_transactions:
            total_recent_spending = sum(t.get('amount', 0) for t in context.recent_transactions)
            avg_transaction = total_recent_spending / len(context.recent_transactions)
            
            analysis['spending_capacity']['average_transaction'] = avg_transaction
            
            if context.transaction_amount > avg_transaction * 1.5:
                analysis['disposable_income_indicators']['above_average_spending'] = True
                analysis['affordability_score'] += 15
            elif context.transaction_amount < avg_transaction * 0.5:
                analysis['financial_stress_signals'].append('below_average_spending')
        
        # Payment method analysis
        if context.payment_method in ['credit_card', 'savings_account']:
            analysis['disposable_income_indicators']['preferred_payment_method'] = True
            analysis['affordability_score'] += 10
        elif context.payment_method == 'basic_account':
            analysis['financial_stress_signals'].append('basic_payment_method')
        
        # Time since payday factor
        if context.time_since_payday <= 7:  # Recent payday
            analysis['disposable_income_indicators']['recent_payday'] = True
            analysis['affordability_score'] += 15
        elif context.time_since_payday > 20:  # Far from payday
            analysis['financial_stress_signals'].append('far_from_payday')
            analysis['affordability_score'] -= 10
        
        # Income estimate factor
        if context.monthly_income_estimate:
            monthly_transaction_rate = (context.transaction_amount * 30) / context.monthly_income_estimate
            if monthly_transaction_rate < 0.3:  # Less than 30% of income on this type of spending
                analysis['disposable_income_indicators']['sustainable_spending'] = True
                analysis['affordability_score'] += 20
        
        return analysis
    
    def _analyze_behavioral_patterns(self, profile: BehavioralProfile) -> Dict[str, Any]:
        """TOOL CALL: Analyze user's behavioral patterns for optimization"""
        analysis = {
            'donation_persona': '',
            'amount_preferences': {},
            'response_patterns': {},
            'optimization_opportunities': []
        }
        
        # Determine donation persona
        if profile.donation_frequency == 'frequent' and profile.average_donation_amount > 50:
            analysis['donation_persona'] = 'generous_regular'
            analysis['optimization_opportunities'].append('can_handle_higher_amounts')
        elif profile.donation_frequency == 'frequent' and profile.average_donation_amount <= 50:
            analysis['donation_persona'] = 'consistent_modest'
            analysis['optimization_opportunities'].append('frequency_over_amount')
        elif profile.donation_frequency == 'occasional' and profile.average_donation_amount > 100:
            analysis['donation_persona'] = 'thoughtful_giver'
            analysis['optimization_opportunities'].append('quality_over_quantity')
        elif profile.donation_frequency == 'rare':
            analysis['donation_persona'] = 'selective_donor'
            analysis['optimization_opportunities'].append('needs_strong_motivation')
        else:
            analysis['donation_persona'] = 'first_time_potential'
            analysis['optimization_opportunities'].append('gentle_introduction')
        
        # Amount preferences analysis
        if profile.preferred_amounts:
            analysis['amount_preferences']['has_patterns'] = True
            analysis['amount_preferences']['common_amounts'] = profile.preferred_amounts
            
            # Check for round number preference
            round_amounts = [amt for amt in profile.preferred_amounts if amt % 10 == 0]
            if len(round_amounts) > len(profile.preferred_amounts) * 0.7:
                analysis['amount_preferences']['prefers_round_numbers'] = True
        
        # Response pattern analysis
        total_responses = sum(profile.response_to_suggestions.values())
        if total_responses > 0:
            acceptance_rate = profile.response_to_suggestions.get('accepted', 0) / total_responses
            modification_rate = profile.response_to_suggestions.get('modified', 0) / total_responses
            
            analysis['response_patterns']['acceptance_rate'] = acceptance_rate
            analysis['response_patterns']['modification_rate'] = modification_rate
            
            if acceptance_rate > 0.7:
                analysis['optimization_opportunities'].append('high_trust_user')
            elif modification_rate > 0.3:
                analysis['optimization_opportunities'].append('likes_to_customize')
        
        # Price sensitivity analysis
        if profile.price_sensitivity == 'high':
            analysis['optimization_opportunities'].append('emphasize_small_amounts')
        elif profile.price_sensitivity == 'low':
            analysis['optimization_opportunities'].append('can_suggest_premium_amounts')
        
        return analysis
    
    def _analyze_psychological_factors(self, financial_context: FinancialContext,
                                     behavioral_profile: BehavioralProfile) -> Dict[str, Any]:
        """TOOL CALL: Analyze psychological factors affecting amount perception"""
        analysis = {
            'anchoring_strategy': '',
            'framing_approach': '',
            'cognitive_biases': [],
            'motivational_triggers': []
        }
        
        # Determine anchoring strategy based on transaction amount
        if financial_context.transaction_amount >= 500:
            analysis['anchoring_strategy'] = 'high_anchor'
            analysis['motivational_triggers'].append('generosity_moment')
        elif financial_context.transaction_amount >= 100:
            analysis['anchoring_strategy'] = 'moderate_anchor'
        else:
            analysis['anchoring_strategy'] = 'low_anchor'
            analysis['cognitive_biases'].append('small_amount_acceptability')
        
        # Determine framing approach
        if behavioral_profile.generosity_score > 0.7:
            analysis['framing_approach'] = 'impact_focused'
        elif behavioral_profile.price_sensitivity == 'high':
            analysis['framing_approach'] = 'value_focused'
        else:
            analysis['framing_approach'] = 'balanced'
        
        # Identify relevant cognitive biases
        if behavioral_profile.donation_frequency == 'first_time':
            analysis['cognitive_biases'].extend(['social_proof', 'authority'])
        
        if financial_context.time_since_payday <= 7:
            analysis['motivational_triggers'].append('fresh_money_effect')
        
        # Decoy effect opportunities
        if behavioral_profile.average_donation_amount > 0:
            analysis['cognitive_biases'].append('decoy_effect')
        
        return analysis
    
    def _select_optimization_strategy(self, affordability: Dict, behavioral: Dict) -> str:
        """TOOL CALL: Select the optimal strategy based on analyses"""
        
        # Conservative strategy conditions
        if (affordability['affordability_score'] < 30 or
            len(affordability['financial_stress_signals']) > 2 or
            behavioral['donation_persona'] == 'first_time_potential'):
            return 'conservative'
        
        # Aggressive strategy conditions
        if (affordability['affordability_score'] > 60 and
            behavioral['donation_persona'] in ['generous_regular', 'thoughtful_giver'] and
            'can_handle_higher_amounts' in behavioral['optimization_opportunities']):
            return 'aggressive'
        
        # Default to moderate strategy
        return 'moderate'
    
    def _calculate_base_amounts(self, financial_context: FinancialContext,
                              behavioral_profile: BehavioralProfile,
                              strategy: str) -> Dict[str, float]:
        """TOOL CALL: Calculate base amounts using different methods"""
        
        strategy_config = self.strategies[strategy]
        
        # Method 1: Transaction-based calculation
        transaction_based = financial_context.transaction_amount * strategy_config['multiplier']
        
        # Method 2: Behavioral average-based calculation
        behavioral_based = behavioral_profile.average_donation_amount
        if behavioral_based == 0:  # First time donor
            behavioral_based = financial_context.transaction_amount * 0.02
        
        # Method 3: Wallet balance-based calculation
        max_ratio = strategy_config['max_ratio']
        balance_based = financial_context.wallet_balance * max_ratio
        
        # Method 4: Historical pattern-based calculation
        pattern_based = transaction_based
        if behavioral_profile.preferred_amounts:
            avg_preferred = sum(behavioral_profile.preferred_amounts) / len(behavioral_profile.preferred_amounts)
            pattern_based = (transaction_based + avg_preferred) / 2
        
        return {
            'transaction_based': transaction_based,
            'behavioral_based': behavioral_based,
            'balance_based': balance_based,
            'pattern_based': pattern_based
        }
    
    def _apply_psychological_optimizations(self, base_amounts: Dict[str, float],
                                         psychological_analysis: Dict) -> Dict[str, float]:
        """TOOL CALL: Apply psychological optimizations to base amounts"""
        
        optimized = {}
        
        for method, amount in base_amounts.items():
            optimized_amount = amount
            
            # Apply anchoring strategy
            if psychological_analysis['anchoring_strategy'] == 'high_anchor':
                optimized_amount *= 1.2
            elif psychological_analysis['anchoring_strategy'] == 'low_anchor':
                optimized_amount *= 0.8
            
            # Apply psychological pricing
            if 'small_amount_acceptability' in psychological_analysis['cognitive_biases']:
                # Keep amounts small and approachable
                optimized_amount = min(optimized_amount, 50)
            
            # Apply anchor point optimization
            optimized_amount = self._find_nearest_anchor(optimized_amount)
            
            optimized[method] = optimized_amount
        
        return optimized
    
    def _find_nearest_anchor(self, amount: float) -> float:
        """TOOL CALL: Find the nearest psychological anchor point"""
        
        # Find the closest anchor point
        closest_anchor = min(self.anchor_amounts, key=lambda x: abs(x - amount))
        
        # If the amount is close to an anchor, use the anchor
        if abs(amount - closest_anchor) / amount < 0.15:  # Within 15%
            return float(closest_anchor)
        
        # Otherwise, round to nearest 5 or 10
        if amount < 50:
            return round(amount / 5) * 5
        else:
            return round(amount / 10) * 10
    
    def _create_final_suggestions(self, optimized_amounts: Dict[str, float],
                                financial_context: FinancialContext,
                                behavioral_profile: BehavioralProfile,
                                strategy: str) -> OptimizedAmounts:
        """TOOL CALL: Create final amount suggestions with reasoning"""
        
        # Select primary amount (use the most appropriate method)
        if behavioral_profile.average_donation_amount > 0:
            primary_amount_value = optimized_amounts['behavioral_based']
            primary_reasoning = ["Based on your donation history", "Aligned with your giving patterns"]
        else:
            primary_amount_value = optimized_amounts['transaction_based']
            primary_reasoning = ["Proportional to your transaction", "Comfortable starter amount"]
        
        # Create primary suggestion
        primary_amount = AmountSuggestion(
            amount=primary_amount_value,
            confidence=85.0,
            reasoning=primary_reasoning,
            psychological_anchor=self._get_anchor_explanation(primary_amount_value),
            conversion_probability=self._estimate_conversion_probability(
                primary_amount_value, financial_context, behavioral_profile
            ),
            impact_description=self._describe_impact(primary_amount_value)
        )
        
        # Create alternative amounts
        alternatives = []
        
        # Conservative alternative (50% of primary)
        conservative_amount = primary_amount_value * 0.5
        conservative_amount = self._find_nearest_anchor(conservative_amount)
        alternatives.append(AmountSuggestion(
            amount=conservative_amount,
            confidence=75.0,
            reasoning=["Smaller, more comfortable amount", "Easy to approve"],
            psychological_anchor=self._get_anchor_explanation(conservative_amount),
            conversion_probability=self._estimate_conversion_probability(
                conservative_amount, financial_context, behavioral_profile
            ) + 0.15,
            impact_description=self._describe_impact(conservative_amount)
        ))
        
        # Stretch alternative (150% of primary)
        stretch_amount = primary_amount_value * 1.5
        stretch_amount = self._find_nearest_anchor(stretch_amount)
        alternatives.append(AmountSuggestion(
            amount=stretch_amount,
            confidence=60.0,
            reasoning=["Maximize your impact", "For when you're feeling generous"],
            psychological_anchor=self._get_anchor_explanation(stretch_amount),
            conversion_probability=self._estimate_conversion_probability(
                stretch_amount, financial_context, behavioral_profile
            ) - 0.20,
            impact_description=self._describe_impact(stretch_amount)
        ))
        
        # Create micro amount (very small, high conversion)
        micro_amount_value = min(20, primary_amount_value * 0.3)
        micro_amount_value = self._find_nearest_anchor(micro_amount_value)
        micro_amount = AmountSuggestion(
            amount=micro_amount_value,
            confidence=90.0,
            reasoning=["Minimal impact on budget", "Every bit helps"],
            psychological_anchor="Micro-donation",
            conversion_probability=0.85,
            impact_description=self._describe_impact(micro_amount_value)
        )
        
        # Create stretch amount
        stretch_final_amount = max(alternatives, key=lambda x: x.amount)
        
        # Determine personalization factors
        personalization_factors = []
        if behavioral_profile.average_donation_amount > 0:
            personalization_factors.append("donation_history")
        if financial_context.monthly_income_estimate:
            personalization_factors.append("income_estimate")
        personalization_factors.extend(["transaction_context", "psychological_anchoring"])
        
        # Select A/B test variant
        ab_variant = self._select_ab_variant(primary_amount_value)
        
        # Calculate overall confidence
        confidence_score = self._calculate_confidence_score(
            financial_context, behavioral_profile, strategy
        )
        
        return OptimizedAmounts(
            primary_amount=primary_amount,
            alternative_amounts=alternatives,
            micro_amount=micro_amount,
            stretch_amount=stretch_final_amount,
            optimization_strategy=strategy,
            personalization_factors=personalization_factors,
            a_b_test_variant=ab_variant,
            confidence_score=confidence_score
        )
    
    def _get_anchor_explanation(self, amount: float) -> str:
        """TOOL CALL: Explain the psychological anchor for an amount"""
        if amount <= 25:
            return "Small, approachable amount"
        elif amount <= 50:
            return "Standard giving amount"
        elif amount <= 100:
            return "Meaningful contribution"
        else:
            return "Significant impact amount"
    
    def _estimate_conversion_probability(self, amount: float,
                                       financial_context: FinancialContext,
                                       behavioral_profile: BehavioralProfile) -> float:
        """TOOL CALL: Estimate probability of user accepting this amount"""
        
        base_probability = 0.3  # 30% base conversion rate
        
        # Adjust based on amount relative to transaction
        amount_ratio = amount / financial_context.transaction_amount
        if amount_ratio < 0.02:  # Less than 2%
            base_probability += 0.2
        elif amount_ratio > 0.05:  # More than 5%
            base_probability -= 0.15
        
        # Adjust based on behavioral profile
        if behavioral_profile.donation_frequency == 'frequent':
            base_probability += 0.25
        elif behavioral_profile.donation_frequency == 'first_time':
            base_probability -= 0.1
        
        # Adjust based on generosity score
        base_probability += (behavioral_profile.generosity_score - 0.5) * 0.3
        
        return max(0.05, min(0.95, base_probability))
    
    def _describe_impact(self, amount: float) -> str:
        """TOOL CALL: Describe the impact of a donation amount"""
        if amount <= 20:
            return f"₱{amount} provides a nutritious meal for a child"
        elif amount <= 50:
            return f"₱{amount} supplies school materials for a student"
        elif amount <= 100:
            return f"₱{amount} funds medical supplies for multiple patients"
        elif amount <= 200:
            return f"₱{amount} supports a family's emergency needs"
        else:
            return f"₱{amount} makes a significant community impact"
    
    def _select_ab_variant(self, amount: float) -> str:
        """TOOL CALL: Select A/B test variant for amount presentation"""
        
        # Use user_id hash for consistent assignment (simplified for demo)
        # In production, this would use proper A/B testing framework
        
        if amount % 10 == 0:
            return 'rounded'  # ₱50, ₱100
        elif amount % 5 == 0:
            return 'psychological'  # ₱45, ₱95
        else:
            return 'precise'  # ₱47, ₱123
    
    def _calculate_confidence_score(self, financial_context: FinancialContext,
                                  behavioral_profile: BehavioralProfile,
                                  strategy: str) -> float:
        """TOOL CALL: Calculate overall confidence in optimization"""
        
        base_confidence = 60.0
        
        # Add confidence based on data availability
        if behavioral_profile.average_donation_amount > 0:
            base_confidence += 15  # Have donation history
        
        if financial_context.monthly_income_estimate:
            base_confidence += 10  # Have income data
        
        if len(financial_context.recent_transactions) > 5:
            base_confidence += 10  # Good transaction history
        
        # Adjust based on strategy certainty
        strategy_confidence = {
            'conservative': 5,   # Safe but might be too low
            'moderate': 10,      # Balanced approach
            'aggressive': -5     # Higher risk
        }
        base_confidence += strategy_confidence[strategy]
        
        return min(95.0, max(30.0, base_confidence))
    
    def _log_optimization(self, context: FinancialContext, suggestions: OptimizedAmounts):
        """TOOL CALL: Log optimization decision for learning"""
        log_data = {
            'user_id': context.user_id,
            'transaction_amount': context.transaction_amount,
            'primary_suggested_amount': suggestions.primary_amount.amount,
            'conversion_probability': suggestions.primary_amount.conversion_probability,
            'strategy': suggestions.optimization_strategy,
            'confidence': suggestions.confidence_score,
            'personalization_factors': suggestions.personalization_factors,
            'timestamp': datetime.now().isoformat()
        }
        
        self.logger.info(f"Amount optimization logged: {json.dumps(log_data, indent=2)}")

# Example usage and testing
if __name__ == "__main__":
    agent = AmountOptimizationAgent()
    
    # Test with sample data
    financial_context = FinancialContext(
        user_id="user_123",
        transaction_amount=250.0,
        wallet_balance=5000.0,
        monthly_income_estimate=45000.0,
        recent_transactions=[
            {"amount": 200, "category": "food"},
            {"amount": 150, "category": "transport"},
            {"amount": 300, "category": "shopping"}
        ],
        donation_history=[
            {"amount": 25, "date": "2024-01-15"},
            {"amount": 50, "date": "2024-02-10"}
        ],
        spending_patterns={"food": 8000, "transport": 3000, "entertainment": 2000},
        savings_indicators={"has_savings": True, "emergency_fund": True},
        payment_method="credit_card",
        time_since_payday=5
    )
    
    behavioral_profile = BehavioralProfile(
        donation_frequency="occasional",
        average_donation_amount=37.5,
        largest_donation=50.0,
        smallest_donation=25.0,
        preferred_amounts=[25, 50],
        response_to_suggestions={"accepted": 3, "rejected": 1, "modified": 1},
        price_sensitivity="medium",
        generosity_score=0.65
    )
    
    result = agent.process(financial_context, behavioral_profile)
    print(f"Primary Amount: ₱{result.primary_amount.amount}")
    print(f"Strategy: {result.optimization_strategy}")
    print(f"Confidence: {result.confidence_score}%")
