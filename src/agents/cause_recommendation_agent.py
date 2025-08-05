"""
Cause Recommendation Agent
Recommends optimal NGO causes based on user preferences and context
Input-Tool Call-Output pattern implementation
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import logging
import json
import math

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class UserProfile:
    """Input: User profile for cause recommendations"""
    user_id: str
    location: Dict[str, str]
    demographic_hints: Dict[str, Any]
    preferred_causes: List[str]
    donation_history: List[Dict[str, Any]]
    transaction_category: str
    transaction_amount: float
    interests: List[str]
    values: List[str]

@dataclass
class NGOData:
    """NGO information for matching"""
    ngo_id: str
    name: str
    cause_category: str
    subcategories: List[str]
    region_focus: str
    impact_metrics: Dict[str, float]
    urgency_level: str  # high, medium, low
    recent_updates: bool
    success_stories: List[str]
    transparency_score: float
    efficiency_rating: float

@dataclass
class CauseMatch:
    """Output: Recommended cause with scoring"""
    ngo: NGOData
    relevance_score: float
    match_reasons: List[str]
    impact_potential: str
    personalized_message: str
    suggested_amount: float
    urgency_indicator: str
    social_proof: Dict[str, Any]

@dataclass
class CauseRecommendations:
    """Complete output with multiple recommendations"""
    primary_recommendation: CauseMatch
    alternative_recommendations: List[CauseMatch]
    reasoning_summary: str
    personalization_factors: List[str]
    diversity_score: float
    confidence: float

class CauseRecommendationAgent:
    """
    Agent that recommends optimal causes using input-tool call-output pattern
    
    INPUT: UserProfile with preferences and context
    TOOL CALLS: Matching algorithms, scoring methods, personalization tools
    OUTPUT: CauseRecommendations with ranked NGO suggestions
    """
    
    def __init__(self):
        self.name = "CauseRecommendationAgent"
        self.logger = logging.getLogger(f"ImpactSense.{self.name}")
        
        # Load NGO database (in production, this would be from a database)
        self.ngo_database = self._load_ngo_database()
        
        # Scoring weights
        self.scoring_weights = {
            'preference_match': 0.30,
            'geographic_relevance': 0.20,
            'demographic_alignment': 0.15,
            'urgency_factor': 0.15,
            'impact_potential': 0.10,
            'transparency': 0.10
        }
    
    def process(self, user_profile: UserProfile) -> CauseRecommendations:
        """
        Main agent processing method - INPUT -> TOOL CALLS -> OUTPUT
        
        Args:
            user_profile (UserProfile): User data for cause matching
            
        Returns:
            CauseRecommendations: Ranked cause recommendations with reasoning
        """
        self.logger.info(f"Processing cause recommendations for user {user_profile.user_id}")
        
        # TOOL CALLS - Various analysis and matching methods
        preference_analysis = self._analyze_user_preferences(user_profile)
        geographic_analysis = self._analyze_geographic_context(user_profile)
        demographic_analysis = self._analyze_demographic_alignment(user_profile)
        urgency_analysis = self._analyze_urgency_factors()
        
        # Score all NGOs
        scored_ngos = self._score_all_ngos(
            user_profile, preference_analysis, geographic_analysis, 
            demographic_analysis, urgency_analysis
        )
        
        # Select top recommendations
        recommendations = self._select_recommendations(scored_ngos, user_profile)
        
        # Add personalization
        personalized_recommendations = self._personalize_recommendations(
            recommendations, user_profile
        )
        
        # Log decision
        self._log_recommendation(user_profile, personalized_recommendations)
        
        return personalized_recommendations
    
    def _load_ngo_database(self) -> List[NGOData]:
        """TOOL CALL: Load NGO database"""
        # In production, this would load from a database
        # For demo, we'll create sample NGOs
        return [
            NGOData(
                ngo_id="ngo_001",
                name="Gawad Kalinga",
                cause_category="housing",
                subcategories=["community_development", "poverty_alleviation"],
                region_focus="NCR",
                impact_metrics={"families_helped": 10000, "communities_built": 50},
                urgency_level="medium",
                recent_updates=True,
                success_stories=["Built 100 homes in Quezon City"],
                transparency_score=0.95,
                efficiency_rating=0.88
            ),
            NGOData(
                ngo_id="ngo_002",
                name="World Vision Philippines",
                cause_category="children",
                subcategories=["education", "healthcare", "nutrition"],
                region_focus="nationwide",
                impact_metrics={"children_helped": 50000, "schools_built": 25},
                urgency_level="high",
                recent_updates=True,
                success_stories=["Provided education to 5000 children in remote areas"],
                transparency_score=0.92,
                efficiency_rating=0.90
            ),
            NGOData(
                ngo_id="ngo_003",
                name="Philippine Red Cross",
                cause_category="disaster_relief",
                subcategories=["emergency_response", "blood_donation", "health"],
                region_focus="nationwide",
                impact_metrics={"lives_saved": 25000, "disasters_responded": 100},
                urgency_level="high",
                recent_updates=True,
                success_stories=["Responded to 10 major disasters this year"],
                transparency_score=0.96,
                efficiency_rating=0.85
            ),
            NGOData(
                ngo_id="ngo_004",
                name="Teach for the Philippines",
                cause_category="education",
                subcategories=["teacher_training", "rural_education", "leadership"],
                region_focus="nationwide",
                impact_metrics={"students_impacted": 15000, "teachers_trained": 500},
                urgency_level="medium",
                recent_updates=False,
                success_stories=["Trained 100 teachers in underserved communities"],
                transparency_score=0.89,
                efficiency_rating=0.92
            ),
            NGOData(
                ngo_id="ngo_005",
                name="Bantay Bata 163",
                cause_category="children",
                subcategories=["child_protection", "healthcare", "counseling"],
                region_focus="NCR",
                impact_metrics={"children_protected": 5000, "cases_handled": 1200},
                urgency_level="high",
                recent_updates=True,
                success_stories=["Rescued 50 children from dangerous situations"],
                transparency_score=0.94,
                efficiency_rating=0.87
            )
        ]
    
    def _analyze_user_preferences(self, profile: UserProfile) -> Dict[str, Any]:
        """TOOL CALL: Analyze user's stated and inferred preferences"""
        analysis = {
            'explicit_preferences': profile.preferred_causes,
            'inferred_from_history': [],
            'transaction_context': [],
            'strength_scores': {}
        }
        
        # Analyze donation history for patterns
        if profile.donation_history:
            cause_frequency = {}
            for donation in profile.donation_history:
                cause = donation.get('cause_category', 'unknown')
                cause_frequency[cause] = cause_frequency.get(cause, 0) + 1
            
            # Sort by frequency
            sorted_causes = sorted(cause_frequency.items(), key=lambda x: x[1], reverse=True)
            analysis['inferred_from_history'] = [cause for cause, freq in sorted_causes[:3]]
        
        # Infer from transaction category
        category_mapping = {
            'food': ['hunger', 'nutrition', 'agriculture'],
            'healthcare': ['health', 'medical', 'children'],
            'education': ['education', 'children', 'literacy'],
            'transportation': ['environment', 'urban_development'],
            'utilities': ['basic_needs', 'poverty_alleviation']
        }
        
        if profile.transaction_category in category_mapping:
            analysis['transaction_context'] = category_mapping[profile.transaction_category]
        
        # Calculate strength scores for all preferences
        all_preferences = set(
            analysis['explicit_preferences'] + 
            analysis['inferred_from_history'] + 
            analysis['transaction_context']
        )
        
        for pref in all_preferences:
            score = 0
            if pref in analysis['explicit_preferences']:
                score += 40
            if pref in analysis['inferred_from_history']:
                score += 30
            if pref in analysis['transaction_context']:
                score += 20
            
            analysis['strength_scores'][pref] = score
        
        return analysis
    
    def _analyze_geographic_context(self, profile: UserProfile) -> Dict[str, Any]:
        """TOOL CALL: Analyze geographic relevance for cause matching"""
        user_region = profile.location.get('region', 'unknown')
        user_city = profile.location.get('city', 'unknown')
        
        # Priority scoring for geographic alignment
        geographic_preferences = {
            'local_priority': 30,  # Same region gets priority
            'national_scope': 20,  # Nationwide NGOs are relevant
            'nearby_regions': 10   # Adjacent regions get some priority
        }
        
        return {
            'user_region': user_region,
            'user_city': user_city,
            'preferences': geographic_preferences,
            'disaster_context': self._check_recent_disasters(user_region)
        }
    
    def _check_recent_disasters(self, region: str) -> Dict[str, Any]:
        """TOOL CALL: Check for recent disasters affecting the region"""
        # In production, this would check real disaster data
        # For demo, we'll simulate some conditions
        simulated_conditions = {
            'NCR': {'recent_floods': True, 'typhoon_season': True},
            'CALABARZON': {'earthquake_risk': True},
            'Central Luzon': {'drought_conditions': True}
        }
        
        return simulated_conditions.get(region, {})
    
    def _analyze_demographic_alignment(self, profile: UserProfile) -> Dict[str, Any]:
        """TOOL CALL: Analyze demographic factors for cause alignment"""
        age_group = profile.demographic_hints.get('age_group', 'unknown')
        income_level = profile.demographic_hints.get('income_level', 'unknown')
        life_stage = profile.demographic_hints.get('life_stage', 'unknown')
        
        # Age-based cause preferences
        age_preferences = {
            'gen_z': ['environment', 'social_justice', 'mental_health'],
            'millennial': ['education', 'healthcare', 'children'],
            'gen_x': ['education', 'community_development', 'healthcare'],
            'boomer': ['healthcare', 'veterans', 'religious']
        }
        
        # Life stage preferences
        life_stage_preferences = {
            'student': ['education', 'scholarships'],
            'young_professional': ['career_development', 'education'],
            'parent': ['children', 'education', 'healthcare'],
            'senior': ['healthcare', 'elderly_care']
        }
        
        return {
            'age_preferences': age_preferences.get(age_group, []),
            'life_stage_preferences': life_stage_preferences.get(life_stage, []),
            'income_considerations': income_level
        }
    
    def _analyze_urgency_factors(self) -> Dict[str, Any]:
        """TOOL CALL: Analyze current urgency factors affecting cause selection"""
        # In production, this would analyze real-time data
        current_urgencies = {
            'disaster_relief': 'high',  # Typhoon season
            'healthcare': 'medium',     # Ongoing health needs
            'education': 'medium',      # School year considerations
            'hunger': 'high',           # Economic challenges
            'children': 'high'          # Vulnerable population
        }
        
        seasonal_factors = {
            'december': ['holiday_giving', 'year_end_giving'],
            'june': ['back_to_school', 'education'],
            'july': ['typhoon_preparation', 'disaster_relief']
        }
        
        current_month = datetime.now().strftime('%B').lower()
        
        return {
            'current_urgencies': current_urgencies,
            'seasonal_factors': seasonal_factors.get(current_month, []),
            'global_context': ['pandemic_recovery', 'economic_challenges']
        }
    
    def _score_all_ngos(self, profile: UserProfile, preference_analysis: Dict,
                       geographic_analysis: Dict, demographic_analysis: Dict,
                       urgency_analysis: Dict) -> List[Tuple[NGOData, float, Dict]]:
        """TOOL CALL: Score all NGOs against user profile"""
        scored_ngos = []
        
        for ngo in self.ngo_database:
            score_breakdown = {}
            total_score = 0
            
            # Preference matching score
            pref_score = self._calculate_preference_score(ngo, preference_analysis)
            score_breakdown['preference_match'] = pref_score
            total_score += pref_score * self.scoring_weights['preference_match']
            
            # Geographic relevance score
            geo_score = self._calculate_geographic_score(ngo, geographic_analysis)
            score_breakdown['geographic_relevance'] = geo_score
            total_score += geo_score * self.scoring_weights['geographic_relevance']
            
            # Demographic alignment score
            demo_score = self._calculate_demographic_score(ngo, demographic_analysis)
            score_breakdown['demographic_alignment'] = demo_score
            total_score += demo_score * self.scoring_weights['demographic_alignment']
            
            # Urgency factor score
            urgency_score = self._calculate_urgency_score(ngo, urgency_analysis)
            score_breakdown['urgency_factor'] = urgency_score
            total_score += urgency_score * self.scoring_weights['urgency_factor']
            
            # Impact potential score
            impact_score = self._calculate_impact_score(ngo, profile)
            score_breakdown['impact_potential'] = impact_score
            total_score += impact_score * self.scoring_weights['impact_potential']
            
            # Transparency score
            transparency_score = ngo.transparency_score * 100
            score_breakdown['transparency'] = transparency_score
            total_score += transparency_score * self.scoring_weights['transparency']
            
            scored_ngos.append((ngo, total_score, score_breakdown))
        
        # Sort by total score (descending)
        scored_ngos.sort(key=lambda x: x[1], reverse=True)
        return scored_ngos
    
    def _calculate_preference_score(self, ngo: NGOData, analysis: Dict) -> float:
        """TOOL CALL: Calculate preference matching score"""
        score = 0
        
        # Check against all user preferences with their strength scores
        for preference, strength in analysis['strength_scores'].items():
            if preference in [ngo.cause_category] + ngo.subcategories:
                score += strength
        
        return min(100, score)
    
    def _calculate_geographic_score(self, ngo: NGOData, analysis: Dict) -> float:
        """TOOL CALL: Calculate geographic relevance score"""
        score = 0
        
        if ngo.region_focus == analysis['user_region']:
            score += analysis['preferences']['local_priority']
        elif ngo.region_focus == 'nationwide':
            score += analysis['preferences']['national_scope']
        
        # Disaster context bonus
        if analysis['disaster_context'] and ngo.cause_category == 'disaster_relief':
            score += 40
        
        return min(100, score)
    
    def _calculate_demographic_score(self, ngo: NGOData, analysis: Dict) -> float:
        """TOOL CALL: Calculate demographic alignment score"""
        score = 0
        
        # Age-based preferences
        for pref in analysis['age_preferences']:
            if pref in [ngo.cause_category] + ngo.subcategories:
                score += 20
        
        # Life stage preferences
        for pref in analysis['life_stage_preferences']:
            if pref in [ngo.cause_category] + ngo.subcategories:
                score += 15
        
        return min(100, score)
    
    def _calculate_urgency_score(self, ngo: NGOData, analysis: Dict) -> float:
        """TOOL CALL: Calculate urgency-based score"""
        score = 0
        
        # Current urgency levels
        cause_urgency = analysis['current_urgencies'].get(ngo.cause_category, 'low')
        urgency_scores = {'high': 30, 'medium': 15, 'low': 5}
        score += urgency_scores[cause_urgency]
        
        # NGO's own urgency level
        score += urgency_scores[ngo.urgency_level]
        
        # Recent updates bonus (shows active engagement)
        if ngo.recent_updates:
            score += 10
        
        return min(100, score)
    
    def _calculate_impact_score(self, ngo: NGOData, profile: UserProfile) -> float:
        """TOOL CALL: Calculate potential impact score"""
        base_score = ngo.efficiency_rating * 50
        
        # Amount-based impact potential
        suggested_amount = self._calculate_suggested_amount(ngo, profile)
        if suggested_amount >= 100:
            base_score += 20
        elif suggested_amount >= 50:
            base_score += 10
        
        return min(100, base_score)
    
    def _calculate_suggested_amount(self, ngo: NGOData, profile: UserProfile) -> float:
        """TOOL CALL: Calculate suggested donation amount"""
        base_amount = profile.transaction_amount * 0.02  # 2% of transaction
        
        # Adjust based on NGO efficiency and impact
        efficiency_multiplier = ngo.efficiency_rating
        amount = base_amount * efficiency_multiplier
        
        # Round to meaningful amounts
        if amount < 20:
            return 20
        elif amount < 50:
            return 25
        elif amount < 100:
            return 50
        else:
            return round(amount / 25) * 25  # Round to nearest 25
    
    def _select_recommendations(self, scored_ngos: List[Tuple], 
                              profile: UserProfile) -> List[CauseMatch]:
        """TOOL CALL: Select top recommendations ensuring diversity"""
        recommendations = []
        used_categories = set()
        
        for ngo, score, breakdown in scored_ngos:
            # Ensure diversity in cause categories
            if len(recommendations) == 0 or ngo.cause_category not in used_categories:
                match_reasons = self._generate_match_reasons(breakdown, ngo)
                
                cause_match = CauseMatch(
                    ngo=ngo,
                    relevance_score=score,
                    match_reasons=match_reasons,
                    impact_potential=self._describe_impact_potential(ngo, profile),
                    personalized_message="",  # Will be filled in personalization step
                    suggested_amount=self._calculate_suggested_amount(ngo, profile),
                    urgency_indicator=ngo.urgency_level,
                    social_proof=self._generate_social_proof(ngo)
                )
                
                recommendations.append(cause_match)
                used_categories.add(ngo.cause_category)
                
                # Limit to top 3 recommendations
                if len(recommendations) >= 3:
                    break
        
        return recommendations
    
    def _generate_match_reasons(self, breakdown: Dict, ngo: NGOData) -> List[str]:
        """TOOL CALL: Generate human-readable match reasons"""
        reasons = []
        
        if breakdown['preference_match'] > 50:
            reasons.append("Matches your stated preferences")
        
        if breakdown['geographic_relevance'] > 30:
            reasons.append("Active in your region")
        
        if breakdown['urgency_factor'] > 40:
            reasons.append("Addressing urgent current needs")
        
        if ngo.transparency_score > 0.9:
            reasons.append("High transparency rating")
        
        if ngo.efficiency_rating > 0.85:
            reasons.append("Excellent efficiency record")
        
        return reasons
    
    def _describe_impact_potential(self, ngo: NGOData, profile: UserProfile) -> str:
        """TOOL CALL: Describe potential impact of donation"""
        amount = self._calculate_suggested_amount(ngo, profile)
        
        impact_descriptions = {
            'education': f"₱{amount} could provide school supplies for 2 children",
            'healthcare': f"₱{amount} could fund medical supplies for 5 patients",
            'children': f"₱{amount} could provide meals for 10 children for a day",
            'disaster_relief': f"₱{amount} could provide emergency kit for 1 family",
            'housing': f"₱{amount} could contribute to building materials for housing"
        }
        
        return impact_descriptions.get(ngo.cause_category, 
                                     f"₱{amount} could make a meaningful impact")
    
    def _generate_social_proof(self, ngo: NGOData) -> Dict[str, Any]:
        """TOOL CALL: Generate social proof elements"""
        return {
            'recent_donors': f"{len(ngo.success_stories) * 100}+ recent donors",
            'impact_metrics': ngo.impact_metrics,
            'success_story': ngo.success_stories[0] if ngo.success_stories else None,
            'transparency_badge': ngo.transparency_score > 0.9
        }
    
    def _personalize_recommendations(self, recommendations: List[CauseMatch],
                                   profile: UserProfile) -> CauseRecommendations:
        """TOOL CALL: Add personalization to recommendations"""
        
        if not recommendations:
            # Fallback if no matches found
            primary_rec = self._create_fallback_recommendation(profile)
            return CauseRecommendations(
                primary_recommendation=primary_rec,
                alternative_recommendations=[],
                reasoning_summary="Using general recommendations based on your profile",
                personalization_factors=["location", "transaction_category"],
                diversity_score=0.0,
                confidence=30.0
            )
        
        # Personalize messages
        for rec in recommendations:
            rec.personalized_message = self._create_personalized_message(rec, profile)
        
        # Calculate diversity score
        categories = {rec.ngo.cause_category for rec in recommendations}
        diversity_score = (len(categories) / len(recommendations)) * 100
        
        # Determine personalization factors
        personalization_factors = []
        if profile.preferred_causes:
            personalization_factors.append("stated_preferences")
        if profile.donation_history:
            personalization_factors.append("donation_history")
        personalization_factors.extend(["location", "transaction_context"])
        
        # Calculate confidence based on top score and number of factors
        top_score = recommendations[0].relevance_score
        confidence = min(95, max(40, top_score * 0.8 + len(personalization_factors) * 5))
        
        return CauseRecommendations(
            primary_recommendation=recommendations[0],
            alternative_recommendations=recommendations[1:],
            reasoning_summary=self._create_reasoning_summary(recommendations[0], profile),
            personalization_factors=personalization_factors,
            diversity_score=diversity_score,
            confidence=confidence
        )
    
    def _create_personalized_message(self, recommendation: CauseMatch, 
                                   profile: UserProfile) -> str:
        """TOOL CALL: Create personalized message for recommendation"""
        ngo = recommendation.ngo
        amount = recommendation.suggested_amount
        
        # Base message templates
        templates = {
            'education': f"Help provide quality education - your ₱{amount} donation to {ngo.name} can make a difference in a child's future.",
            'healthcare': f"Support healthcare access - ₱{amount} to {ngo.name} helps provide essential medical care to those in need.",
            'children': f"Protect vulnerable children - your ₱{amount} donation helps {ngo.name} keep children safe and healthy.",
            'disaster_relief': f"Emergency response support - ₱{amount} helps {ngo.name} provide immediate aid to disaster victims.",
            'housing': f"Build stronger communities - contribute ₱{amount} to {ngo.name}'s housing initiatives."
        }
        
        base_message = templates.get(ngo.cause_category, 
                                   f"Make an impact - donate ₱{amount} to {ngo.name}")
        
        # Add personalization based on user context
        if profile.transaction_category == 'food' and ngo.cause_category in ['children', 'hunger']:
            base_message += " Your recent food purchase shows you understand the importance of nutrition."
        
        if profile.location.get('region') == ngo.region_focus:
            base_message += f" This cause is active right here in {profile.location.get('region')}."
        
        return base_message
    
    def _create_reasoning_summary(self, primary_rec: CauseMatch, 
                                profile: UserProfile) -> str:
        """TOOL CALL: Create summary of recommendation reasoning"""
        ngo = primary_rec.ngo
        
        summary_parts = [
            f"We recommend {ngo.name} because it"
        ]
        
        if "Matches your stated preferences" in primary_rec.match_reasons:
            summary_parts.append("aligns with your preferred causes")
        
        if "Active in your region" in primary_rec.match_reasons:
            summary_parts.append("operates in your area")
        
        if "High transparency rating" in primary_rec.match_reasons:
            summary_parts.append("maintains excellent transparency")
        
        # Join with proper grammar
        if len(summary_parts) == 2:
            return f"{summary_parts[0]} {summary_parts[1]}."
        elif len(summary_parts) > 2:
            return f"{summary_parts[0]} {', '.join(summary_parts[1:-1])}, and {summary_parts[-1]}."
        else:
            return f"{summary_parts[0]} addresses urgent current needs."
    
    def _create_fallback_recommendation(self, profile: UserProfile) -> CauseMatch:
        """TOOL CALL: Create fallback recommendation when no good matches found"""
        # Use most general, high-rated NGO
        fallback_ngo = max(self.ngo_database, key=lambda ngo: ngo.transparency_score)
        
        return CauseMatch(
            ngo=fallback_ngo,
            relevance_score=50.0,
            match_reasons=["General recommendation", "High transparency"],
            impact_potential=self._describe_impact_potential(fallback_ngo, profile),
            personalized_message=f"Consider supporting {fallback_ngo.name} - a trusted organization making real impact.",
            suggested_amount=self._calculate_suggested_amount(fallback_ngo, profile),
            urgency_indicator=fallback_ngo.urgency_level,
            social_proof=self._generate_social_proof(fallback_ngo)
        )
    
    def _log_recommendation(self, profile: UserProfile, 
                          recommendations: CauseRecommendations):
        """TOOL CALL: Log recommendation for learning and debugging"""
        log_data = {
            'user_id': profile.user_id,
            'primary_recommendation': {
                'ngo_name': recommendations.primary_recommendation.ngo.name,
                'cause_category': recommendations.primary_recommendation.ngo.cause_category,
                'relevance_score': recommendations.primary_recommendation.relevance_score,
                'suggested_amount': recommendations.primary_recommendation.suggested_amount
            },
            'personalization_factors': recommendations.personalization_factors,
            'confidence': recommendations.confidence,
            'diversity_score': recommendations.diversity_score,
            'timestamp': datetime.now().isoformat()
        }
        
        self.logger.info(f"Recommendation logged: {json.dumps(log_data, indent=2)}")

# Example usage and testing
if __name__ == "__main__":
    agent = CauseRecommendationAgent()
    
    test_profile = UserProfile(
        user_id="user_123",
        location={"region": "NCR", "city": "Manila"},
        demographic_hints={"age_group": "millennial", "income_level": "middle"},
        preferred_causes=["education", "children"],
        donation_history=[
            {"cause_category": "education", "amount": 50},
            {"cause_category": "children", "amount": 25}
        ],
        transaction_category="food",
        transaction_amount=250.0,
        interests=["technology", "social_causes"],
        values=["equality", "education", "community"]
    )
    
    recommendations = agent.process(test_profile)
    print(f"Primary Recommendation: {recommendations.primary_recommendation.ngo.name}")
    print(f"Relevance Score: {recommendations.primary_recommendation.relevance_score}")
    print(f"Message: {recommendations.primary_recommendation.personalized_message}")
