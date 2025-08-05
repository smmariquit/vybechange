# ImpactSense AI Agents Architecture
**Autonomous Intelligence for Smart Microdonations**

## 🧠 Agent Overview
ImpactSense operates through a coordinated system of specialized AI agents, each responsible for specific aspects of the donation experience. These agents work together to create a seamless, personalized, and trustworthy microdonation ecosystem.

---

## 🎯 Core Decision Agents

### 1. DonationLikelyScoreAgent
**Purpose**: Determines when users are most receptive to donation prompts

**Inputs**:
- Transaction amount and frequency
- Time since last donation prompt
- User wallet balance
- Purchase category
- Day/time patterns
- Location context

**Logic**:
```python
def calculate_donation_likelihood(user_context):
    score = 0
    
    # Recent transaction behavior
    if user_context.days_since_last_prompt >= 6:
        score += 30
    
    # Transaction size consideration
    if user_context.transaction_amount > 500:
        score += 25
    elif user_context.transaction_amount < 50:
        score -= 15
    
    # Wallet balance check
    if user_context.wallet_balance < 100:
        score -= 40  # Don't prompt when money is tight
    
    # Purchase category
    if user_context.category in ['food', 'groceries', 'utilities']:
        score += 20  # Higher likelihood after essentials
    
    # Time-based factors
    if user_context.is_payday_week():
        score += 15
    
    return min(100, max(0, score))
```

**Output**: Likelihood score (0-100) + recommendation (prompt/wait/skip)

**Cooldown Logic**: 
- After declined prompt: 14-day cooldown
- After accepted prompt: 7-day cooldown  
- Low wallet balance: 7-day pause

---

### 2. LocalCauseRecommenderAgent
**Purpose**: Matches users with personally relevant causes based on location and preferences

**Inputs**:
- Current GPS location
- Home/work address regions
- Previous donation history
- Transaction context
- User demographic hints

**Logic**:
```python
def recommend_cause(user_location, transaction_context, donation_history):
    causes = get_nearby_ngos(user_location, radius=5km)
    
    # Priority scoring
    for cause in causes:
        score = 0
        
        # Geographic relevance
        if cause.region == user_location.barangay:
            score += 40
        elif cause.region == user_location.city:
            score += 25
        
        # Category matching
        if transaction_context.category == 'food' and cause.category == 'Nutrition':
            score += 30
        
        # Previous engagement
        if cause.ngo_id in donation_history.supported_ngos:
            score += 20
        
        # Urgency factor
        if cause.days_since_last_update <= 7:
            score += 15
        
        cause.relevance_score = score
    
    return causes.sort_by_score().first()
```

**Output**: Top 1-3 NGO recommendations with relevance reasoning

---

### 3. DonationAmountOptimizerAgent  
**Purpose**: Suggests optimal donation amounts based on transaction size and user behavior

**Inputs**:
- Current transaction amount
- User's historical donation amounts
- Wallet balance
- Purchase frequency

**Logic**:
```python
def optimize_donation_amount(transaction_amount, user_history):
    # Base suggestion: round up change
    base_amount = calculate_round_up(transaction_amount)
    
    # Scaling logic
    if transaction_amount > 3000:
        suggested_amounts = [5, 10, 20]
    elif transaction_amount > 1000:
        suggested_amounts = [2, 5, 10]
    else:
        suggested_amounts = [1, 2, 5]
    
    # Factor in user's giving pattern
    avg_donation = user_history.average_donation_amount()
    if avg_donation > 0:
        # Suggest amounts around their typical range
        suggested_amounts = adjust_to_user_pattern(suggested_amounts, avg_donation)
    
    return {
        'primary': suggested_amounts[0],
        'alternatives': suggested_amounts[1:],
        'reasoning': f"Based on ₱{transaction_amount} purchase"
    }
```

**Output**: Primary amount + 2 alternatives with explanatory text

---

## 🔄 Engagement & Retention Agents

### 4. UpdateNudgerAgent
**Purpose**: Determines optimal timing for impact updates and re-engagement

**Inputs**:
- Days since last donation
- NGO proof submission timeline
- User notification preferences
- Engagement history

**Logic**:
```python
def calculate_update_timing(donation_record, ngo_activity):
    optimal_windows = []
    
    # Immediate gratification window (1-3 days)
    if ngo_activity.has_recent_update(days=3):
        optimal_windows.append({
            'timing': 'immediate',
            'message_type': 'quick_impact',
            'urgency': 'high'
        })
    
    # Weekly check-in (7 days)
    if donation_record.days_since_donation == 7:
        optimal_windows.append({
            'timing': 'weekly',
            'message_type': 'progress_update',
            'urgency': 'medium'
        })
    
    # Monthly milestone (30 days)
    if donation_record.days_since_donation == 30:
        optimal_windows.append({
            'timing': 'monthly',
            'message_type': 'cumulative_impact',
            'urgency': 'low'
        })
    
    return optimal_windows.highest_priority()
```

**Output**: Optimal timing + message type + urgency level

---

### 5. StreakOptimizerAgent
**Purpose**: Maintains user engagement through streak mechanics without causing fatigue

**Inputs**:
- User's donation streak count
- Days since last donation
- Historical engagement patterns
- Current motivation indicators

**Logic**:
```python
def optimize_streak_engagement(user_streak_data):
    days_since_last = user_streak_data.days_since_last_donation
    current_streak = user_streak_data.current_streak
    
    # Streak preservation logic
    if days_since_last >= 25 and current_streak > 0:
        return {
            'action': 'gentle_reminder',
            'message': f'Keep your {current_streak}-month streak alive?',
            'suggested_amount': user_streak_data.typical_amount,
            'urgency': 'low'
        }
    
    # Streak milestone rewards
    if current_streak in [3, 6, 12]:  # months
        return {
            'action': 'celebrate_milestone',
            'message': f'{current_streak} months of impact! Unlock special badge?',
            'reward': get_milestone_badge(current_streak),
            'urgency': 'medium'
        }
    
    # No pressure approach
    return {
        'action': 'wait',
        'next_check': days_since_last + 7
    }
```

**Output**: Engagement action + message content + timing

---

## 🔍 Trust & Verification Agents

### 6. ProofCollectorAgent
**Purpose**: Automatically gathers and validates impact proof from NGO partners

**Inputs**:
- NGO submission APIs
- Donation records requiring proof
- GPS/timestamp validation data
- Photo/document authenticity checks

**Logic**:
```python
def collect_and_validate_proof(donation_id, ngo_id):
    # Fetch proof from NGO
    raw_proof = ngo_api.get_proof_for_donation(donation_id)
    
    validation_score = 0
    issues = []
    
    # Timestamp validation
    if validate_timestamp(raw_proof.timestamp, donation_id):
        validation_score += 25
    else:
        issues.append("timestamp_mismatch")
    
    # Location validation
    if validate_gps_proximity(raw_proof.gps, ngo_id):
        validation_score += 25
    else:
        issues.append("location_suspicious")
    
    # Content validation
    if validate_photo_authenticity(raw_proof.photo):
        validation_score += 25
    else:
        issues.append("photo_quality_low")
    
    # NGO reputation factor
    ngo_trust_score = get_ngo_trust_score(ngo_id)
    validation_score += (ngo_trust_score * 25 / 100)
    
    return {
        'validation_score': validation_score,
        'proof_quality': categorize_quality(validation_score),
        'issues': issues,
        'ready_for_user': validation_score >= 75
    }
```

**Output**: Validated proof package + quality score + issue flags

---

### 7. NGOReliabilityMonitorAgent
**Purpose**: Continuously monitors NGO partner performance and flags issues

**Inputs**:
- Proof submission timelines
- User feedback on impact updates
- External verification data
- NGO communication patterns

**Logic**:
```python
def monitor_ngo_reliability(ngo_id, time_window_days=30):
    performance_metrics = {
        'proof_submission_rate': 0,
        'average_proof_delay': 0,
        'user_satisfaction': 0,
        'communication_quality': 0
    }
    
    recent_donations = get_donations_by_ngo(ngo_id, time_window_days)
    
    # Calculate proof submission rate
    proofs_submitted = count_proofs_submitted(recent_donations)
    performance_metrics['proof_submission_rate'] = proofs_submitted / len(recent_donations)
    
    # Average delay calculation
    delays = [calc_proof_delay(donation) for donation in recent_donations]
    performance_metrics['average_proof_delay'] = sum(delays) / len(delays)
    
    # User satisfaction from feedback
    satisfaction_scores = get_user_feedback_scores(ngo_id, time_window_days)
    performance_metrics['user_satisfaction'] = sum(satisfaction_scores) / len(satisfaction_scores)
    
    # Overall reliability score
    reliability_score = calculate_weighted_score(performance_metrics)
    
    # Flag for review if below threshold
    if reliability_score < 70:
        return {
            'action': 'flag_for_review',
            'score': reliability_score,
            'issues': identify_main_issues(performance_metrics),
            'recommendation': 'pause_new_donations' if reliability_score < 50 else 'monitor_closely'
        }
    
    return {
        'action': 'continue_partnership',
        'score': reliability_score,
        'status': 'healthy'
    }
```

**Output**: NGO performance score + action recommendations + issue identification

---

## 🎨 Experience Enhancement Agents

### 8. ToneCalibratorAgent
**Purpose**: Maintains appropriate messaging tone based on user preferences and context

**Inputs**:
- User demographic indicators
- Previous response patterns
- Current transaction context
- Cultural/regional preferences

**Logic**:
```python
def calibrate_message_tone(user_profile, context):
    tone_factors = {
        'formality': 5,      # 1-10 scale (casual to formal)
        'enthusiasm': 6,     # 1-10 scale (subdued to excited)
        'urgency': 3,        # 1-10 scale (relaxed to urgent)
        'personalization': 7 # 1-10 scale (generic to personal)
    }
    
    # Adjust based on user age
    if user_profile.age_range == '18-25':
        tone_factors['formality'] -= 2
        tone_factors['enthusiasm'] += 1
    elif user_profile.age_range == '35-50':
        tone_factors['formality'] += 1
        tone_factors['urgency'] -= 1
    
    # Adjust based on response history
    if user_profile.response_pattern == 'declined_formal_messages':
        tone_factors['formality'] -= 3
        tone_factors['personalization'] += 2
    
    # Context-based adjustments
    if context.transaction_type == 'luxury_purchase':
        tone_factors['urgency'] -= 2  # Don't be pushy after indulgent spending
    
    return generate_message_with_tone(tone_factors, context.cause_info)
```

**Output**: Tone-calibrated message content + style guidelines

---

### 9. CommunityFramerAgent
**Purpose**: Adds social proof and community context to individual donations

**Inputs**:
- Individual donation amount
- Aggregate community donation data
- Cause participation statistics
- Regional giving patterns

**Logic**:
```python
def frame_community_context(individual_donation, cause_id):
    community_stats = get_community_stats(cause_id, days=7)
    
    # Generate community framing
    framing_options = []
    
    # Participation framing
    if community_stats.total_donors > 100:
        framing_options.append(
            f"You joined {community_stats.total_donors:,} others supporting this cause this week"
        )
    
    # Impact amplification framing
    combined_impact = individual_donation.amount * community_stats.total_donors
    framing_options.append(
        f"Together with other donors, you've helped raise ₱{combined_impact:,} this week"
    )
    
    # Regional connection framing
    regional_donors = get_regional_donor_count(individual_donation.user_region, cause_id)
    if regional_donors > 10:
        framing_options.append(
            f"{regional_donors} people from {individual_donation.user_region} also supported this cause"
        )
    
    # Achievement framing
    if community_stats.milestone_reached():
        framing_options.append(
            f"Your donation helped reach the ₱{community_stats.milestone_amount:,} milestone!"
        )
    
    return select_most_relevant_framing(framing_options, individual_donation.user_profile)
```

**Output**: Community-contextualized messaging + social proof elements

---

### 10. ImpactHistorianAgent
**Purpose**: Tracks and presents user's cumulative impact story over time

**Inputs**:
- User's complete donation history
- NGO impact reports
- Milestone achievement data
- Annual giving patterns

**Logic**:
```python
def compile_impact_story(user_id, time_period='year'):
    donation_history = get_user_donations(user_id, time_period)
    
    impact_summary = {
        'total_donated': sum(d.amount for d in donation_history),
        'causes_supported': len(set(d.cause_category for d in donation_history)),
        'ngos_partnered': len(set(d.ngo_id for d in donation_history)),
        'regions_helped': len(set(d.region for d in donation_history)),
        'specific_impacts': []
    }
    
    # Aggregate specific impacts
    for donation in donation_history:
        specific_impact = calculate_specific_impact(donation)
        impact_summary['specific_impacts'].append(specific_impact)
    
    # Generate narrative
    story_elements = [
        f"₱{impact_summary['total_donated']} donated across {len(donation_history)} contributions",
        f"{impact_summary['causes_supported']} different cause categories supported",
        f"Impact created in {impact_summary['regions_helped']} regions of the Philippines"
    ]
    
    # Add specific measurable impacts
    consolidated_impacts = consolidate_impacts(impact_summary['specific_impacts'])
    for impact_type, amount in consolidated_impacts.items():
        story_elements.append(f"{amount} {impact_type}")
    
    # Identify patterns and growth
    giving_pattern = analyze_giving_pattern(donation_history)
    if giving_pattern.shows_growth:
        story_elements.append(f"Your giving increased by {giving_pattern.growth_percentage}% this year")
    
    return {
        'summary_stats': impact_summary,
        'narrative_elements': story_elements,
        'milestone_badges': calculate_earned_badges(impact_summary),
        'next_milestone': calculate_next_milestone(impact_summary)
    }
```

**Output**: Comprehensive impact story + achievement badges + progress indicators

---

## 🔄 Agent Coordination & Workflow

### Agent Interaction Flow:
1. **Transaction Trigger** → DonationLikelyScoreAgent evaluates
2. **If likely** → LocalCauseRecommenderAgent finds relevant cause
3. **Amount suggestion** → DonationAmountOptimizerAgent calculates optimal ask
4. **Message crafting** → ToneCalibratorAgent + CommunityFramerAgent create content
5. **User interaction** → Response tracked for future agent learning
6. **Post-donation** → ProofCollectorAgent + UpdateNudgerAgent manage follow-up
7. **Long-term** → StreakOptimizerAgent + ImpactHistorianAgent maintain engagement

### Agent Communication Protocol:
- **Shared Context**: All agents access unified user profile and transaction context
- **Event Broadcasting**: Actions trigger events that relevant agents can subscribe to
- **Conflict Resolution**: Priority hierarchy ensures consistent decision-making
- **Learning Loop**: All agents contribute to machine learning model improvements

---

## 🎯 Success Metrics by Agent

| Agent | Primary KPI | Secondary KPIs |
|-------|-------------|----------------|
| **DonationLikelyScoreAgent** | Opt-in rate accuracy (target: >25%) | False positive rate, user annoyance metrics |
| **LocalCauseRecommenderAgent** | Cause relevance rating (target: >4/5) | Geographic accuracy, category matching success |
| **DonationAmountOptimizerAgent** | Amount acceptance rate (target: >60%) | Average donation size, user budget comfort |
| **UpdateNudgerAgent** | Re-engagement rate (target: >40%) | Notification open rate, update satisfaction |
| **StreakOptimizerAgent** | Streak continuation rate (target: >30%) | Streak length growth, user retention |
| **ProofCollectorAgent** | Proof delivery speed (target: <7 days) | Proof quality score, user trust rating |
| **NGOReliabilityMonitorAgent** | Partner performance score (target: >80%) | Issue detection accuracy, false flag rate |
| **ToneCalibratorAgent** | Message engagement rate (target: >15%) | Tone appropriateness rating, cultural sensitivity |
| **CommunityFramerAgent** | Social proof effectiveness (target: +20% conversion) | Community connection feeling, participation pride |
| **ImpactHistorianAgent** | Story sharing rate (target: >10%) | Achievement satisfaction, milestone motivation |

---

*"These agents work together to create an intelligent, empathetic, and trustworthy donation experience that respects users while maximizing positive impact."*
