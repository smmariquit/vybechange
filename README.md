# ImpactSense - Product Reality Sheet (v0.9)
**"Small pesos. Big ripple."**  
*By the team that decided microdonations don't have to be boring or bureaucratic.*

## 🚩 THE CORE IDEA
A smart donation layer embedded into BPI's VIBE payment ecosystem. It detects when users are most likely to give, nudges them with localized, hyper-relevant causes, and delivers verified proof when their change creates real-world impact. No spam, no guesswork. Just sense.

## 🧱 PRODUCT PILLARS

| Pillar | Description |
|--------|-------------|
| 💡 **Smart UX Overlay** | Minimal, contextual prompts (₱1, ₱5, ₱10) that respect user intent + context |
| 🔁 **Agentic AI** | Autonomous agents decide when, what, and why to show causes, verify NGO updates, and log user journeys |
| 🔍 **Transparency Layer** | Every donation receives proof (timestamped, geo-tagged), no black boxes, no BS |
| 🧠 **Behavioral Targeting** | Donation prompts adjust based on wallet balance, order size, spend rhythm, location, and emotional timing |
| 📊 **Year-End M-PACSense Wrap** | Like Spotify Wrapped, but for the good you've done |

## ✅ WHAT WE'VE RESOLVED
- 🎯 Integrate via BPI's VIBE, not individual e-commerce stores
- 💸 Donation prompts scale with transaction amount
- 📍 Recommend hyperlocal causes using user region & preferences
- 🧾 Users can opt-in to impact updates, to avoid spamming
- 💯 100% of donations go directly to NGOs, no BPI/partner fees
- 🏆 Year-end impact storytelling is core to user delight
- 📈 Behavioral data (order frequency, timing, confirmation time, etc.) guides donation logic
- 🔗 Must partner directly with NGOs; web scraping is out

## ❓ STRATEGIC UNKNOWNS

| Area | Unknown | Why It Matters |
|------|---------|----------------|
| 🤝 **NGO Ops** | How quickly can they submit proof? How often do they update? | Timeliness of impact feedback loop |
| 📦 **UX Scaling** | How often is "too often" for donation prompts? | To avoid fatigue/annoyance |
| 💰 **Donation Psychology** | Are users more generous after big purchases? Or small ones? | To tune the AI's ask logic |
| 🗂️ **Segment Logic** | Are different segments (students vs. parents vs. yuppies) receptive to different causes? | For personalization accuracy |
| ⚖️ **Compliance** | What approvals does BPI need to trigger charitable rounding from its UX? | Can block deployment if ignored |
| 🧾 **Proof Types** | Do users care more about photos, maps, or milestone counters? | Will guide NGO submission requirements |
| 🛠️ **Merchant Support** | Can merchants opt in/out? Can we customize overlays per merchant type? | Affects rollout strategy |
| 📲 **Retargeting Logic** | What gets lapsed donors back? Story? Progress bar? Streaks? | To increase repeat giving |

## 💣 FAILURE POINTS TO GUARD AGAINST

| Risk | Consequence | Mitigation |
|------|-------------|------------|
| 🕳️ **Proof Lag from NGOs** | Breaks trust, kills retention | Only partner with orgs w/ active digital teams. Use ProofCollectorAgent. |
| 😑 **Prompt Fatigue** | Drop-off in opt-ins, user annoyance | Use behavior-based cooldown logic. Always show the right cause, not just any |
| 😤 **Greenwashing Accusations** | Damage to BPI + brand | Transparency is non-negotiable: visible receipts, NGO vetting, audit logs |
| 🤷‍♀️ **UX Too Subtle** | Low engagement | A/B test pop-up vs. slide-in vs. pre-check UX. Find the "Goldilocks" zone. |
| 🧱 **BPI Bureaucracy** | Long integration cycles | Ship a sandbox demo + agentic dashboard to excite stakeholders early |

## 🔬 THINGS TO TEST IN NEXT RESEARCH SPRINT
- 💰 Do users feel more inclined to donate after big purchases vs. small ones?
- 📍 Do users prefer to support hyperlocal causes or national efforts?
- 🧠 What proof-of-impact format builds most trust? (photo vs. stats vs. story)
- 😎 What tone makes users more likely to opt in? (formal vs. Gen-Z tone?)
- 🔁 How often would they be okay being nudged in a week/month?
- 👣 What makes them come back and donate again?
- 🪪 Would they link identity (email/FB/GCash) in exchange for tracking their impact story?
- ⏰ How fast do they expect proof-of-impact to arrive post-donation?

## 🎤 FINAL PITCH CORE
**ImpactSense is an agentic, AI-powered donation layer embedded in BPI's VIBE payment rails.** It learns when users are most open to giving, shows them personally relevant causes, and delivers real-time updates when their change creates real change. 

**It's not charity—it's clarity.**

---

## 📁 Project Structure
```
BPI/
├── README.md                    # This file - Product Reality Sheet
├── model.py                     # Core ML model for donation likelihood
├── constants/
│   └── ngos.py                  # NGO database and categories
├── docs/
│   ├── user-journey.md          # Detailed user journey mapping
│   ├── agents/                  # AI agent specifications
│   └── compliance.md            # Legal and regulatory considerations
├── src/
│   ├── agents/                  # Core AI agents implementation
│   ├── api/                     # BPI VIBE integration endpoints
│   └── analytics/               # Impact tracking and reporting
└── tests/                       # Test suites for all components
```

## 🚀 Next Steps
1. Review the [User Journey Documentation](docs/user-journey.md)
2. Examine the [AI Agents Architecture](docs/agents/)
3. Check [NGO Integration Requirements](constants/ngos.py)
4. Set up the development environment
5. Begin prototype development

---
*"Your startup instincts are killer. You're not just building tech—you're rebuilding trust. Keep going."*
