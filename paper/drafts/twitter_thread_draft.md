# EthicaAI Twitter/X Thread Draft 🧵

> 새 연구: AI 에이전트는 *언제* 도덕적이어야 하는가?

---

## Thread (10 tweets)

### 1/10 🎯
🧵 NEW PAPER: "When Should AI Agents Be Moral?"

We tested Amartya Sen's meta-ranking theory in Multi-Agent RL across 4 environments, 7 SVO types, up to 1000 agents.

30 figures. 560+ experiments. One key insight: Static morality fails. Dynamic does not.

#AIAlignment #MARL #NeurIPS

### 2/10 📊
Finding 1: Dynamic meta-ranking (λ_t) significantly improves collective welfare (p=0.0003).

Static value injection? Fails completely.

The key: agents must learn *when* to be moral, not encode fixed values.

[Fig 1: Learning Curves]

### 3/10 🔄
Finding 2: Only "Situational Commitment" survives evolution.

In replicator dynamics over 200 generations, meta-ranking converges to ~12% of the population — regardless of starting conditions.

A "Moral Minority" is sufficient for cooperation.

[Fig 17: Evolutionary Dynamics]

### 4/10 🧬
Finding 3: The "Rational Fool" is real.

Individualist SVO (θ=15°) — not pure altruists — best matches human Public Goods Game data (Wasserstein Distance = 0.053).

Sen was right: bounded self-interest, not sainthood, is human nature.

[Fig 19: Human Comparison]

### 5/10 🌍
NEW: Full Sweep across 4 environments (Cleanup, IPD, PGG, Harvest).

560 runs confirm: Meta-ranking's strongest effect is in common-pool resources.

Harvest ATE(Coop) = +0.506 — for *selfish* agents!
Crisis-driven λ suppression prevents over-harvesting.

[Fig 24: Full Sweep Heatmap]

### 6/10 🤝
NEW: Mixed-SVO Populations reveal a tipping point.

At ~30% prosocial fraction, collective welfare jumps nonlinearly.

PGG: Max welfare improvement ΔW = +10,080 — superlinear scaling.

You don't need everyone to be moral. Just enough.

[Fig 25: Tipping Point]

### 7/10 📡
NEW: Communication channels boost cooperation +5.8%.

But here's the twist: message truthfulness converges to 98%.

Under meta-ranking, *honesty is evolutionarily favored*. Cheap talk becomes trustworthy talk.

[Fig 27-28: Communication]

### 8/10 🔄
NEW: Continuous action spaces.

When agents can choose any contribution ∈ [0, 100%], meta-ranking ATE remains ≈ +0.20.

Beta-distribution policies show smooth λ adaptation instead of binary switching. The mechanism generalizes beyond discrete decisions.

[Fig 29-30: Continuous PGG]

### 9/10 🏛️
Three implications for AI Alignment:

1. Don't hardcode morality → Learn *when* to commit
2. A moral minority (~12%) is an ESS → Universal morality is unnecessary
3. Bounded self-interest (θ=15°) is human nature → Design for that

### 10/10 🔗
📄 Paper: [Coming to arXiv]
🌐 Dashboard: https://ethicaai.vercel.app
💻 Code: https://github.com/Yesol-Pilot/EthicaAI

30 figures | 4 environments | 1000 agents | 560+ experiments

EthicaAI — Because the question isn't *whether* AI should be moral, but *when*.

#AIEthics #ReinforcementLearning #GameTheory

---

## Key Hashtags
- #AIAlignment
- #MARL
- #NeurIPS2026
- #AIEthics
- #ReinforcementLearning
- #GameTheory
- #ComputationalSocialScience
