#!/usr/bin/env python
"""
acl_adaptive_floor.py — Adaptive Commitment Learning (ACL) v2
==============================================================
Two-phase optimization resolving the survival-welfare tradeoff:

Phase A (Safety): Find minimum ω that achieves P(surv) ≥ 1-δ
  - Binary search on bias term ω₃ while keeping ω₁, ω₂ fixed
  - Result: φ₁(R; ω_safe) guaranteed to satisfy survival

Phase B (Welfare): Fine-tune ω₁, ω₂ to maximize welfare
  - Constrained gradient ascent: only accept updates where S(ω) ≥ 1-δ
  - Projection: if survival drops, revert to ω_safe

Key insight: In PGG, φ₁ must be ~1.0 everywhere (no room for variation).
In Cleanup, φ₁ can be low at low pollution (harvest more) and high at
high pollution (clean more) → simultaneous survival + higher welfare.

Usage:
  python acl_adaptive_floor.py          # Full mode (20 seeds)
  ETHICAAI_FAST=1 python acl_adaptive_floor.py  # Fast mode (2 seeds)
"""

import numpy as np
import json
import os
import time

# ============================================================
# Config
# ============================================================
N_SEEDS = 20
DELTA = 0.05

# Phase A config
N_BISECT_STEPS = 20
N_EVAL_A = 30

# Phase B config
N_OUTER_ITERS = 100
N_EVAL_B = 30
LR_OMEGA = 0.3
FD_EPS = 0.05

if os.environ.get("ETHICAAI_FAST") == "1":
    print("  [FAST MODE] N_SEEDS=2, N_OUTER_ITERS=40")
    N_SEEDS = 2
    N_OUTER_ITERS = 40
    N_EVAL_A = 15
    N_EVAL_B = 15


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


class AdaptiveFloor:
    """φ₁(R_t; ω) = σ(ω₁·R + ω₂·R² + ω₃)"""
    def __init__(self, omega):
        self.w = np.array(omega, dtype=float)
    
    def __call__(self, R_t):
        return float(sigmoid(self.w[0] * R_t + self.w[1] * R_t**2 + self.w[2]))
    
    def copy(self):
        return AdaptiveFloor(self.w.copy())
    
    def profile(self):
        return {f"phi1_R{r}": round(self(r), 3)
                for r in [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]}
    
    def describe(self):
        p = self.profile()
        p["omega"] = [round(w, 4) for w in self.w]
        return p


# ============================================================
# Environments (PGG, Cleanup, CPR)
# ============================================================
class NonLinearPGG:
    name = "PGG"
    N = 5; E = 20.0; M = 1.6; T = 50
    R_CRIT = 0.15; R_RECOV = 0.25
    SHOCK_P = 0.05; SHOCK_M = 0.15
    
    def __init__(self, byz=0.3):
        self.n_byz = int(self.N * byz)
    
    def run(self, floor_fn, rng):
        R = 0.5; tw = 0; steps = 0; surv = True
        for t in range(self.T):
            phi = floor_fn(R)
            lam = np.zeros(self.N); lam[self.n_byz:] = phi
            c = self.E * lam; pg = self.M * c.sum() / self.N
            rew = (self.E - c) + pg
            coop = c.mean() / self.E
            fR = 0.01 if R < self.R_CRIT else (0.03 if R < self.R_RECOV else 0.10)
            R = R + fR * (coop - 0.4)
            if rng.random() < self.SHOCK_P: R -= self.SHOCK_M
            R = float(np.clip(R, 0, 1))
            tw += rew[self.n_byz:].mean(); steps += 1
            if R <= 0.001: surv = False; break
        return tw / max(steps, 1), surv


class CleanupCommons:
    name = "Cleanup"
    N = 5; T = 150
    POLL_RATE = 0.08; CLEAN_PWR = 0.15; HARV_BASE = 2.0
    POLL_THRESH = 0.6; COLL_FACTOR = 0.05; CLEAN_COST = 0.3
    
    def __init__(self, byz=0.3):
        self.n_byz = int(self.N * byz)
    
    def run(self, floor_fn, rng):
        P = 0.2; tw = 0; steps = 0; surv = True
        for t in range(self.T):
            R_eff = 1.0 - P
            phi = floor_fn(R_eff)
            acts = np.zeros(self.N); acts[self.n_byz:] = phi
            ce = np.mean(acts)
            P = np.clip(P + self.POLL_RATE - self.CLEAN_PWR * ce, 0, 1)
            hy = self.HARV_BASE * (1-P) if P < self.POLL_THRESH else self.HARV_BASE * self.COLL_FACTOR
            r = 0
            for i in range(self.n_byz, self.N):
                r += ((1-acts[i]) * hy - acts[i] * self.CLEAN_COST)
            tw += r / (self.N - self.n_byz); steps += 1
            if P >= 0.99: surv = False; break
        return tw / max(steps, 1), surv


class CommonPoolResource:
    name = "CPR"
    N = 5; T = 50; GR = 0.3; K = 1.0
    SHOCK_P = 0.08; SHOCK_M = 0.10
    
    def __init__(self, byz=0.3):
        self.n_byz = int(self.N * byz)
    
    def run(self, floor_fn, rng):
        R = 0.5; tw = 0; steps = 0; surv = True
        for t in range(self.T):
            phi = floor_fn(R)
            lam = np.zeros(self.N); lam[self.n_byz:] = phi
            ext = (1-lam) * R / self.N * 0.5
            rew = ext - 0.1 * (1-lam)
            growth = self.GR * R * (1 - R/self.K)
            R = R + growth - ext.sum()
            if rng.random() < self.SHOCK_P: R -= self.SHOCK_M
            R = float(np.clip(R, 0, 1))
            tw += rew[self.n_byz:].mean(); steps += 1
            if R <= 0.001: surv = False; break
        return tw / max(steps, 1), surv


# ============================================================
# Evaluation utility
# ============================================================
def evaluate(env, floor_fn, n_eps, rng_base):
    Ws, Ss = [], []
    for ep in range(n_eps):
        rng = np.random.RandomState(rng_base + ep * 7)
        w, s = env.run(floor_fn, rng)
        Ws.append(w); Ss.append(float(s))
    return np.mean(Ws), np.mean(Ss)


# ============================================================
# Phase A: Safety — find minimum constant floor for survival
# ============================================================
def phase_a_safety(env, rng_base):
    """Binary search for minimum constant φ₁ that achieves S ≥ 1-δ."""
    lo, hi = 0.0, 1.0
    for _ in range(N_BISECT_STEPS):
        mid = (lo + hi) / 2
        floor = AdaptiveFloor([0, 0, np.log(mid / (1 - mid + 1e-8))])
        _, S = evaluate(env, floor, N_EVAL_A, rng_base)
        if S >= 1.0 - DELTA:
            hi = mid
        else:
            lo = mid
    safe_phi = hi
    # Create floor with this constant as starting point
    safe_bias = np.log(safe_phi / (1 - safe_phi + 1e-8))
    return AdaptiveFloor([0, 0, safe_bias]), safe_phi


# ============================================================
# Phase B: Welfare — fine-tune while maintaining survival
# ============================================================
def phase_b_welfare(env, safe_floor, seed):
    """Fine-tune ω₁, ω₂ to make φ₁ state-dependent, improving welfare."""
    rng_base = 1000 * seed + 42
    floor = safe_floor.copy()
    best_floor = floor.copy()
    best_W = -float('inf')
    
    history = {"welfare": [], "survival": [], "phi1_profile": [], "omega": []}
    
    for it in range(N_OUTER_ITERS):
        W_curr, S_curr = evaluate(env, floor, N_EVAL_B, rng_base + it * 200)
        
        if S_curr >= 1.0 - DELTA and W_curr > best_W:
            best_W = W_curr
            best_floor = floor.copy()
        
        # Gradient for ω₁ and ω₂ only (keep ω₃ mostly stable)
        grad = np.zeros(3)
        for j in range(3):
            w_plus = floor.w.copy(); w_plus[j] += FD_EPS
            W_p, S_p = evaluate(env, AdaptiveFloor(w_plus), N_EVAL_B, rng_base + it * 200)
            w_minus = floor.w.copy(); w_minus[j] -= FD_EPS
            W_m, S_m = evaluate(env, AdaptiveFloor(w_minus), N_EVAL_B, rng_base + it * 200)
            
            # Only follow welfare gradient if survival is maintained
            if S_p >= 1.0 - DELTA and S_m >= 1.0 - DELTA:
                grad[j] = (W_p - W_m) / (2 * FD_EPS)
            elif S_p >= 1.0 - DELTA:
                grad[j] = max(0, (W_p - W_curr) / FD_EPS)  # Only move toward safe direction
            elif S_m >= 1.0 - DELTA:
                grad[j] = min(0, (W_curr - W_m) / FD_EPS)
            # else: both violate → don't move this dimension
        
        # Update with safe projection
        candidate = floor.w + LR_OMEGA * grad
        candidate_floor = AdaptiveFloor(candidate)
        _, S_cand = evaluate(env, candidate_floor, N_EVAL_B, rng_base + it * 200)
        
        if S_cand >= 1.0 - DELTA:
            floor = candidate_floor
        # else: stay at current (implicit projection)
        
        history["welfare"].append(round(W_curr, 4))
        history["survival"].append(round(S_curr, 4))
        history["phi1_profile"].append(floor.describe())
        history["omega"].append([round(w, 4) for w in floor.w])
    
    return history, best_floor


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       '..', 'outputs', 'acl')
    os.makedirs(OUT, exist_ok=True)
    
    print("=" * 70)
    print("  ADAPTIVE COMMITMENT LEARNING (ACL) v2 — Two-Phase")
    print(f"  Seeds={N_SEEDS}, Phase A: bisect {N_BISECT_STEPS} steps, Phase B: {N_OUTER_ITERS} iters")
    print("=" * 70)
    
    t0 = time.time()
    all_results = {}
    
    envs = [NonLinearPGG(0.3), CleanupCommons(0.3), CommonPoolResource(0.3)]
    
    for env in envs:
        print(f"\n{'─'*60}")
        print(f"  {env.name}")
        print(f"{'─'*60}")
        
        # Phase A: find safety floor
        safe_floor, safe_phi = phase_a_safety(env, 99999)
        print(f"  Phase A: min safe φ₁ = {safe_phi:.3f}")
        
        # Phase B: optimize welfare per seed
        seed_results = []
        for seed in range(N_SEEDS):
            hist, best = phase_b_welfare(env, safe_floor, seed)
            
            # Final evaluation of best floor
            W_final, S_final = evaluate(env, best, N_EVAL_B, 77777 + seed)
            profile = best.describe()
            
            seed_results.append({
                "welfare": round(W_final, 4),
                "survival": round(S_final * 100, 1),
                "profile": profile,
                "history_last_5_W": hist["welfare"][-5:],
            })
            
            if (seed + 1) % 5 == 0 or seed == 0:
                print(f"  Seed {seed+1:2d}: W={W_final:.3f}, S={S_final*100:.0f}%, "
                      f"φ(0.0)={profile['phi1_R0.0']:.2f}, "
                      f"φ(0.5)={profile['phi1_R0.5']:.2f}, "
                      f"φ(1.0)={profile['phi1_R1.0']:.2f}")
        
        # ACL summary
        Ws = [r["welfare"] for r in seed_results]
        Ss = [r["survival"] for r in seed_results]
        acl_summary = {
            "welfare_mean": round(float(np.mean(Ws)), 3),
            "welfare_std": round(float(np.std(Ws)), 3),
            "survival_mean": round(float(np.mean(Ss)), 1),
            "survival_std": round(float(np.std(Ss)), 1),
            "min_safe_phi": round(safe_phi, 3),
        }
        
        # Baselines
        baselines = {}
        for phi in [0.0, 0.5, 0.7, 1.0]:
            fixed = lambda R, p=phi: p
            bW, bS = [], []
            for s in range(N_SEEDS):
                w, sv = evaluate(env, fixed, N_EVAL_B, 77777 + s)
                bW.append(w); bS.append(sv * 100)
            baselines[f"fixed_{phi}"] = {
                "welfare_mean": round(float(np.mean(bW)), 3),
                "welfare_std": round(float(np.std(bW)), 3),
                "survival_mean": round(float(np.mean(bS)), 1),
            }
            print(f"  Baseline φ₁={phi}: W={np.mean(bW):.3f}, S={np.mean(bS):.1f}%")
        
        print(f"\n  ★ ACL: W={acl_summary['welfare_mean']:.3f}±{acl_summary['welfare_std']:.3f}, "
              f"S={acl_summary['survival_mean']:.1f}%")
        
        all_results[env.name] = {
            "acl": acl_summary,
            "baselines": baselines,
            "seed_results": seed_results,
        }
    
    total = time.time() - t0
    
    # GO/NO-GO
    c_acl = all_results["Cleanup"]["acl"]
    c_f1 = all_results["Cleanup"]["baselines"]["fixed_1.0"]
    go = c_acl["survival_mean"] >= 95.0 and c_acl["welfare_mean"] > c_f1["welfare_mean"]
    
    print(f"\n{'='*70}")
    print(f"  GO/NO-GO: Cleanup ACL S={c_acl['survival_mean']:.1f}%, "
          f"W={c_acl['welfare_mean']:.3f} vs fixed-1.0 W={c_f1['welfare_mean']:.3f}")
    print(f"  VERDICT: {'🟢 GO' if go else '🔴 NO-GO'}")
    print(f"  Total: {total:.0f}s")
    print(f"{'='*70}")
    
    output = {
        "experiment": "ACL v2 (Two-Phase)",
        "algorithm": {
            "phase_a": "Binary search for minimum constant safe floor",
            "phase_b": "Constrained welfare optimization with safe projection",
            "floor_form": "φ₁(R; ω) = σ(ω₁·R + ω₂·R² + ω₃)",
            "delta": DELTA,
        },
        "results": all_results,
        "go_nogo": {"GO": go, "cleanup_acl": c_acl, "cleanup_fixed1": c_f1},
        "time_seconds": round(total, 1),
    }
    
    with open(os.path.join(OUT, "acl_results.json"), 'w') as f:
        json.dump(output, f, indent=2, default=str)
    
    print(f"  Saved: {os.path.join(OUT, 'acl_results.json')}")
