# ImpactSense User Journey
**The Complete Experience Map for Kai, Our Ideal Customer**

## 🧍 Ideal Persona: Kai, 26, Metro Manila
- **Demographics**: Remote worker, frequent online shopper
- **Behavior**: Uses VIBE daily for food, bills, and impulse purchases
- **Values**: Transparency, local impact, hates "forced charity"
- **Personality**: Curious, not preachy. Wants proof, not PR
- **Goal**: Wants to help—but only when it feels real, optional, and tied to something personal

---

## 🌐 STAGE 1: Discovery + First Encounter

### 🧭 Context:
Kai is paying ₱689.50 for groceries via the BPI VIBE app.

### 💡 What Happens:
An unobtrusive card appears after payment confirmation:
> **"Round up ₱0.50 to help fund clean water in Marikina?"**
> 
> *Cause: Hyperlocal (near Kai's current GPS)*  
> *Story: "Jessa, 9 y.o., walks 2km for water"*

### 🔄 Agents Involved:
- **DonationLikelyScoreAgent**: User has paid 3x this week, no prompt in 6 days → green light
- **LocalCauseRecommenderAgent**: Tags transaction region + category → picks NGO

### ✅ User Action:
**Kai chooses: [Donate ₱0.50]**

### 💭 Emotional State:
- Feels respected (not pressured)
- Intrigued by local relevance
- Trusts the small amount

---

## 🧾 STAGE 2: Receipt + Immediate Feedback

### 📦 What Happens:
Transaction success screen now includes:
> **"You just helped fund 1L of clean water. NGO: SafePH | Proof coming soon."**
> 
> Visual stamp: 🟢 **100% of your ₱0.50 went to SafePH.**
> 
> Toggle: ☑ **Notify me when this project progresses**

### 💭 Emotional State:
- **Intrigued**: "That was easy"
- **In control**: Can toggle notifications
- **Not sold to**: Felt lightweight and legit

---

## 📸 STAGE 3: Impact Proof & Re-engagement

### ⏱ Timeline: 1 week later

### 📱 Trigger:
Push notification: **"Your ₱0.50 helped finish the well. Here's what you funded."**

### 📄 Content:
Tapping shows:
- 📷 **Photo**: Pump installation
- 🌍 **Map pin**: In Marikina
- 🕒 **Date stamp**: Project completion
- 👥 **Community**: "Donated by 3,245 others this week—including you"
- 🎯 **CTA**: "Help the next project? ₱1 = 2 more liters"

### 🔄 Agents Involved:
- **ProofCollectorAgent**: Pulls NGO media, validates GPS/timestamp
- **UpdateNudgerAgent**: Calculates re-engagement window + timing
- **CommunityFramerAgent**: Adds collective framing ("you + 3K others")

### 💭 Emotional State:
- **Surprised**: "That was fast"
- **Validated**: "It's not charity—it's impact I can see"
- **Connected**: Part of a community effort

---

## 🎉 STAGE 4: Gamification & Social Loop

### 📆 Timeline: After 4 months

### 📊 User Stats:
- **Total donated**: ₱34 across 18 transactions
- **Impact created**: Measurable, visible results

### 🎁 What Happens:
In-app card appears:
> **"Your M-PAC so far:"**
> - ₱34 donated
> - 6 students supported  
> - 22kg plastic removed
> - 3 active regions
> 
> **Unlocks**: "Barangay Hero" badge
> 
> **Option**: Share your impact card on Instagram or Twitter

### 💭 Emotional State:
- **Proud**: Visible achievement
- **Grateful**: "Wouldn't have done it without the nudges"
- **Motivated**: Wants to explore monthly donation mode

---

## 🔁 STAGE 5: Repeat Nudges—Now Smarter

### 🧠 What Changes:
- **Frequency**: Donations offered only after large, guilt-free purchases (₱3,000+)
- **Personalization**: New prompts match causes Kai donated to before
- **Gamification**: "Streak" system: "₱5 more this month to keep your Impact Badge active"

### 🔄 Agents at Work:
- **ImpactHistorianAgent**: Keeps score, highlights patterns
- **StreakOptimizerAgent**: Offers consistent re-engagement without fatigue  
- **ToneCalibratorAgent**: Keeps UX casual, never corporate

---

## 🧠 Behind-the-Scenes Logic

| User Behavior | Agentic Trigger | System Response |
|---------------|-----------------|-----------------|
| Location shifts | New NGO match suggestion | LocalCauseRecommenderAgent activates |
| Wallet low | Pauses prompts for 7 days | DonationLikelyScoreAgent goes dormant |
| No update proof in 14 days | Auto-flag NGO for review | ProofCollectorAgent escalates |
| Repeated large orders | Upscales donation suggestion from ₱1 → ₱5 | DonationAmountOptimizerAgent adjusts |
| Shares impact story | Pushes "social donor" badge + unlocks exclusive NGO previews | SocialEngagementAgent rewards |

---

## 🏁 LIFETIME VALUE COMPARISON

### Without ImpactSense:
- Kai might never donate, or distrust where money goes
- Zero emotional connection to causes
- No systematic giving behavior

### With ImpactSense:
- **Annual Impact**: ₱100–300 per year, across 50+ micro-donations
- **Emotional Connection**: Feels connected to 2–3 NGO stories
- **Viral Growth**: Shares 1–2 impact cards = organic marketing
- **Identity Shift**: Becomes a "passive philanthropist" without leaving the BPI app

---

## 🎯 Key Success Metrics

### User Engagement:
- **Opt-in Rate**: % of users who donate on first prompt
- **Retention Rate**: % who donate again within 30 days
- **Frequency**: Average donations per month
- **Share Rate**: % who share impact cards

### Impact Metrics:
- **Total Donated**: Aggregate amount channeled to NGOs
- **Proof Speed**: Average time from donation to impact update
- **NGO Satisfaction**: Partner organization retention rate
- **User Trust Score**: Survey-based trust in platform

### Business Metrics:
- **Transaction Increase**: % increase in VIBE usage
- **Customer Lifetime Value**: Extended engagement with BPI ecosystem
- **Brand Sentiment**: Net Promoter Score improvement
- **Viral Coefficient**: Organic user acquisition through sharing

---

## 🚀 Implementation Roadmap

### Phase 1: MVP (Weeks 1-8)
- Basic donation prompts after transactions
- Simple NGO database integration
- Basic impact tracking

### Phase 2: AI Enhancement (Weeks 9-16)
- Deploy core agents (DonationLikelyScore, LocalCauseRecommender)
- Behavioral targeting implementation
- Proof collection automation

### Phase 3: Gamification (Weeks 17-24)
- Impact badges and streaks
- Social sharing features
- Year-end M-PACSense Wrap

### Phase 4: Scale (Weeks 25-32)
- Advanced personalization
- Expanded NGO network
- Cross-platform integration

---

*"This journey is your product. Let's make it visual and unforgettable."*
