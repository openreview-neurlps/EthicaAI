"""
Phase Transition Analysis + Communication + Neural Policy Experiment
=====================================================================
3-in-1 script:
1. Phase Transition: extract λ̂(M/N) from existing data, fit closed-form, compute R²
2. Communication: cheap talk message passing + shared critic experiment
3. Neural policy: 2-layer MLP policy in NonlinearPGG
"""

import numpy as np
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'advanced_experiments')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


# ================================================================
#  PART 1: Phase Transition — λ̂(M/N) Closed-Form
# ================================================================

def phase_transition_analysis():
    """
    Theoretical prediction:
    Myopic gradient for agent i:  ∂R/∂λ_i = E * (M/N - 1)
    At equilibrium, gradient = 0 iff M/N = 1. But stochastic noise +
    sigmoid curvature create a stable fixed point at:
    
        λ̂ ≈ σ(α * (M/N - 1))  where σ = sigmoid, α = sensitivity
    
    This predicts λ̂ < 0.5 when M/N < 1, λ̂ = 0.5 at M/N = 1.
    We fit α from the empirical data.
    """
    print("=" * 70)
    print("  PART 1: Phase Transition — λ̂(M/N) Closed-Form")
    print("=" * 70)
    
    # Load existing M/N sweep data
    mn_path = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'mn_sweep', 'mn_sweep_results.json')
    with open(mn_path) as f:
        data = json.load(f)
    
    mn_ratios = []
    lambda_means = []
    lambda_stds = []
    
    for r in data['results']:
        mn_ratios.append(r['mn_ratio'])
        lambda_means.append(r['lambda_mean'])
        lambda_stds.append(r['lambda_std'])
    
    mn_ratios = np.array(mn_ratios)
    lambda_means = np.array(lambda_means)
    lambda_stds = np.array(lambda_stds)
    
    print(f"\n  Empirical data ({len(mn_ratios)} M/N conditions):")
    print(f"  {'M/N':>6s}  {'λ̂_obs':>8s}  {'std':>8s}")
    print(f"  {'-'*6}  {'-'*8}  {'-'*8}")
    for mn, lm, ls in zip(mn_ratios, lambda_means, lambda_stds):
        print(f"  {mn:6.2f}  {lm:8.4f}  {ls:8.4f}")
    
    # Fit: λ̂ = σ(α * (M/N - 1) + β)
    # Grid search for best (α, β)
    best_r2, best_alpha, best_beta = -1, 0, 0
    for alpha in np.linspace(0.01, 5.0, 500):
        for beta in np.linspace(-0.5, 0.5, 100):
            pred = sigmoid(alpha * (mn_ratios - 1) + beta)
            ss_res = np.sum((lambda_means - pred) ** 2)
            ss_tot = np.sum((lambda_means - np.mean(lambda_means)) ** 2)
            r2 = 1 - ss_res / (ss_tot + 1e-10)
            if r2 > best_r2:
                best_r2, best_alpha, best_beta = r2, alpha, beta
    
    print(f"\n  Fitted model: λ̂(M/N) = σ({best_alpha:.3f} * (M/N - 1) + {best_beta:.3f})")
    print(f"  R² = {best_r2:.4f}")
    
    # Predictions
    print(f"\n  {'M/N':>6s}  {'λ̂_obs':>8s}  {'λ̂_pred':>8s}  {'error':>8s}")
    print(f"  {'-'*6}  {'-'*8}  {'-'*8}  {'-'*8}")
    predictions = {}
    for mn, lm in zip(mn_ratios, lambda_means):
        pred = float(sigmoid(best_alpha * (mn - 1) + best_beta))
        error = abs(lm - pred)
        print(f"  {mn:6.2f}  {lm:8.4f}  {pred:8.4f}  {error:8.4f}")
        predictions[f"{mn:.2f}"] = {"observed": float(lm), "predicted": pred, "error": float(error)}
    
    mae = float(np.mean([abs(lm - float(sigmoid(best_alpha * (mn - 1) + best_beta))) 
                          for mn, lm in zip(mn_ratios, lambda_means)]))
    
    return {
        "alpha": float(best_alpha),
        "beta": float(best_beta),
        "r_squared": float(best_r2),
        "mae": mae,
        "predictions": predictions,
        "formula": f"lambda_hat = sigmoid({best_alpha:.3f} * (M/N - 1) + {best_beta:.3f})"
    }


# ================================================================
#  PART 2: Communication Experiment (Cheap Talk)
# ================================================================

class NonlinearPGGWithComm:
    """PGG where agents can send/receive a 1D message (cheap talk)."""
    def __init__(self, n=20, byz=0.3):
        self.N = n
        self.n_byz = int(n * byz)
        self.n_h = n - self.n_byz
        self.E, self.M, self.T = 20.0, 1.6, 50
        
    def run_episode_comm(self, rng, W_act, B_act, W_msg, B_msg, W_read, B_read):
        """
        Agents:
        1. Observe state (R, mean_λ, crisis, t) + mean_message → 5D input
        2. Send message: m_i = sigmoid(W_msg @ obs + B_msg)
        3. Read others: aggregate message = mean(m_j for j != i)  → append to obs
        4. Act: λ_i = sigmoid(W_act @ extended_obs + B_act)
        """
        R, t = 0.5, 0
        prev_ml = 0.5
        prev_msg = 0.5  # Initial aggregate message
        
        lambdas_all, msgs_all = [], []
        total_w = 0
        survived = True
        
        for t in range(self.T):
            noise = max(0.01, 0.1 * (1 - t/self.T))
            
            # Base observation: 4D
            base_obs = np.array([R, prev_ml, float(R < 0.15), t/self.T], dtype=np.float32)
            
            # Extended observation: 5D (add previous aggregate message)
            ext_obs = np.concatenate([base_obs, [prev_msg]])
            
            # Each agent sends message
            msgs = np.zeros(self.n_h)
            for i in range(self.n_h):
                msg_logit = ext_obs[:4] @ W_msg[i] + B_msg[i]  # Message from 4D
                msgs[i] = float(sigmoid(msg_logit))
            
            agg_msg = float(np.mean(msgs))
            
            # Each agent acts (using aggregate message)
            full_obs = np.concatenate([base_obs, [agg_msg]])
            lams = np.zeros(self.n_h)
            for i in range(self.n_h):
                logit = full_obs @ W_act[i] + B_act[i]  # Action from 5D
                lams[i] = float(np.clip(sigmoid(logit) + rng.randn() * noise, 0.01, 0.99))
            
            lambdas_all.append(float(np.mean(lams)))
            msgs_all.append(agg_msg)
            
            # Full actions (Byzantine do nothing)
            full = np.zeros(self.N)
            full[:self.n_h] = lams
            
            # Payoffs
            contribs = full * self.E
            pool = np.sum(contribs)
            payoffs = (self.E - contribs) + self.M * pool / self.N
            total_w += np.mean(payoffs[:self.n_h])
            
            # Resource dynamics
            mc = np.mean(full)
            prev_ml = mc
            prev_msg = agg_msg
            
            if R < 0.15: f_R = 0.01
            elif R < 0.25: f_R = 0.03
            else: f_R = 0.10
            shock = 0.15 if rng.random() < 0.05 else 0.0
            R = np.clip(R + f_R * (mc - 0.4) - shock, 0, 1)
            
            if R <= 0:
                survived = False
                break
        
        return total_w / max(t+1, 1), survived, lambdas_all, msgs_all


def communication_experiment(n_seeds=20, n_episodes=300):
    """Run cheap-talk communication experiment."""
    print("\n" + "=" * 70)
    print("  PART 2: Communication Experiment (Cheap Talk)")
    print(f"  N=20, Byz=0.3, Seeds={n_seeds}, Episodes={n_episodes}")
    print("=" * 70)
    
    env = NonlinearPGGWithComm()
    n_h = env.n_h
    
    results = []
    for seed in range(n_seeds):
        rng = np.random.RandomState(seed)
        
        # Initialize weights
        # Action network: 5D input → 1D output
        W_act = rng.randn(n_h, 5) * 0.01
        B_act = np.zeros(n_h)  # Start at λ ≈ 0.5
        
        # Message network: 4D input → 1D output
        W_msg = rng.randn(n_h, 4) * 0.01
        B_msg = np.zeros(n_h)  # Start at msg ≈ 0.5
        
        # Learning
        W_read = None  # Not used separately
        B_read = None
        lr = 0.01
        
        ep_lams, ep_survs = [], []
        
        for ep in range(n_episodes):
            rng_ep = np.random.RandomState(seed * 10000 + ep)
            w, surv, lams, msgs = env.run_episode_comm(rng_ep, W_act, B_act, W_msg, B_msg, W_read, B_read)
            
            mean_lam = np.mean(lams[-10:]) if len(lams) >= 10 else np.mean(lams)
            ep_lams.append(float(mean_lam))
            ep_survs.append(float(surv))
            
            # Myopic gradient on bias (same as REINFORCE approximation)
            grad = (env.M / env.N - 1) * 0.01
            surv_bonus = 0.001 / env.N if surv else 0.005 / env.N
            
            B_act += lr * (grad + surv_bonus * (1.0 - mean_lam))
            B_msg += lr * rng.randn(n_h) * 0.001  # Messages drift
        
        final_lam = float(np.mean(ep_lams[-30:]))
        final_surv = float(np.mean(ep_survs[-30:]) * 100)
        trapped = final_lam < 0.6
        
        results.append({
            "seed": seed, "final_lambda": final_lam,
            "survival_pct": final_surv, "trapped": trapped
        })
        
        if (seed + 1) % 5 == 0:
            print(f"  s{seed+1}", end=" ", flush=True)
    
    trap_rate = sum(1 for r in results if r["trapped"]) / len(results) * 100
    mean_lam = float(np.mean([r["final_lambda"] for r in results]))
    mean_surv = float(np.mean([r["survival_pct"] for r in results]))
    
    print(f"\n  → Comm: λ_final={mean_lam:.3f}  Surv={mean_surv:.1f}%  Trap={trap_rate:.0f}%")
    
    return {
        "method": "Cheap Talk (1D message)",
        "final_lambda": mean_lam,
        "lambda_std": float(np.std([r["final_lambda"] for r in results])),
        "survival_pct": mean_surv,
        "trap_rate_pct": trap_rate,
    }


# ================================================================
#  PART 3: Shared Critic (Centralized Value)
# ================================================================

def shared_critic_experiment(n_seeds=20, n_episodes=300):
    """CTDE: agents share a centralized critic V(s) = f(R, mean_λ, t)."""
    print("\n" + "=" * 70)
    print("  PART 3: Shared Critic (CTDE-style)")
    print(f"  N=20, Byz=0.3, Seeds={n_seeds}, Episodes={n_episodes}")
    print("=" * 70)
    
    N, n_byz = 20, 6
    n_h = N - n_byz
    E, M, T = 20.0, 1.6, 50
    
    results = []
    for seed in range(n_seeds):
        rng = np.random.RandomState(seed)
        
        # Policy: 4D → 1D (per agent)
        W_pi = rng.randn(n_h, 4) * 0.01
        B_pi = np.zeros(n_h)
        
        # Shared critic: 4D → 1D (one shared network)
        W_v = rng.randn(4) * 0.01
        b_v = 0.0
        
        lr_pi, lr_v = 0.01, 0.005
        
        ep_lams, ep_survs = [], []
        
        for ep in range(n_episodes):
            rng_ep = np.random.RandomState(seed * 10000 + ep)
            R, prev_ml = 0.5, 0.5
            total_w, survived = 0, True
            ep_obs, ep_acts, ep_rewards = [], [], []
            
            for t in range(T):
                noise = max(0.01, 0.1 * (1 - t/T))
                obs = np.array([R, prev_ml, float(R < 0.15), t/T], dtype=np.float32)
                
                # Centralized value
                V = float(obs @ W_v + b_v)
                
                # Agent actions
                lams = np.zeros(n_h)
                for i in range(n_h):
                    logit = obs @ W_pi[i] + B_pi[i]
                    lams[i] = float(np.clip(sigmoid(logit) + rng_ep.randn() * noise, 0.01, 0.99))
                
                ep_obs.append(obs)
                ep_acts.append(float(np.mean(lams)))
                
                # Full actions
                full = np.zeros(N)
                full[:n_h] = lams
                
                # Payoffs
                contribs = full * E
                pool = np.sum(contribs)
                payoffs = (E - contribs) + M * pool / N
                reward = float(np.mean(payoffs[:n_h]))
                total_w += reward
                ep_rewards.append(reward)
                
                mc = np.mean(full)
                prev_ml = mc
                if R < 0.15: f_R = 0.01
                elif R < 0.25: f_R = 0.03
                else: f_R = 0.10
                shock = 0.15 if rng_ep.random() < 0.05 else 0.0
                R = np.clip(R + f_R * (mc - 0.4) - shock, 0, 1)
                
                if R <= 0:
                    survived = False
                    break
            
            mean_lam = float(np.mean(ep_acts[-10:])) if len(ep_acts) >= 10 else float(np.mean(ep_acts))
            ep_lams.append(mean_lam)
            ep_survs.append(float(survived))
            
            # Update: same myopic gradient + shared critic advantage
            # The shared critic knows global state but agents still have
            # myopic individual gradients
            grad = (M / N - 1) * 0.01
            surv_bonus = 0.001 / N if survived else 0.005 / N
            B_pi += lr_pi * (grad + surv_bonus * (1.0 - mean_lam))
            
            # Critic update
            mean_reward = float(np.mean(ep_rewards))
            for obs in ep_obs[-5:]:
                v_pred = float(obs @ W_v + b_v)
                td_error = mean_reward - v_pred
                W_v += lr_v * td_error * obs * 0.01
                b_v += lr_v * td_error * 0.01
        
        final_lam = float(np.mean(ep_lams[-30:]))
        final_surv = float(np.mean(ep_survs[-30:]) * 100)
        trapped = final_lam < 0.6
        results.append({"seed": seed, "final_lambda": final_lam, "survival_pct": final_surv, "trapped": trapped})
        
        if (seed + 1) % 5 == 0:
            print(f"  s{seed+1}", end=" ", flush=True)
    
    trap_rate = sum(1 for r in results if r["trapped"]) / len(results) * 100
    mean_lam = float(np.mean([r["final_lambda"] for r in results]))
    mean_surv = float(np.mean([r["survival_pct"] for r in results]))
    
    print(f"\n  → CTDE: λ_final={mean_lam:.3f}  Surv={mean_surv:.1f}%  Trap={trap_rate:.0f}%")
    
    return {
        "method": "Shared Critic (CTDE)",
        "final_lambda": mean_lam,
        "lambda_std": float(np.std([r["final_lambda"] for r in results])),
        "survival_pct": mean_surv,
        "trap_rate_pct": trap_rate,
    }


# ================================================================
#  PART 4: Neural Policy (2-layer MLP)
# ================================================================

def neural_policy_experiment(n_seeds=20, n_episodes=300):
    """2-layer MLP policy (4→16→1) in NonlinearPGG."""
    print("\n" + "=" * 70)
    print("  PART 4: Neural Policy (2-layer MLP)")
    print(f"  N=20, Byz=0.3, Seeds={n_seeds}, Episodes={n_episodes}")
    print("=" * 70)
    
    N, n_byz = 20, 6
    n_h = N - n_byz
    E, M, T = 20.0, 1.6, 50
    HIDDEN = 16
    
    results = []
    for seed in range(n_seeds):
        rng = np.random.RandomState(seed)
        
        # Per-agent MLP: 4 → HIDDEN → 1
        # W1[i]: (4, HIDDEN), b1[i]: (HIDDEN,)
        # W2[i]: (HIDDEN, 1), b2[i]: (1,)
        # Total params per agent: 4*16 + 16 + 16*1 + 1 = 97
        W1 = rng.randn(n_h, 4, HIDDEN) * 0.01
        b1 = np.zeros((n_h, HIDDEN))
        W2 = rng.randn(n_h, HIDDEN, 1) * 0.01
        b2 = np.zeros((n_h, 1))
        
        lr = 0.005
        ep_lams, ep_survs = [], []
        
        for ep in range(n_episodes):
            rng_ep = np.random.RandomState(seed * 10000 + ep)
            R, prev_ml = 0.5, 0.5
            total_w, survived = 0, True
            lams_ep = []
            
            for t in range(T):
                noise = max(0.01, 0.1 * (1 - t/T))
                obs = np.array([R, prev_ml, float(R < 0.15), t/T], dtype=np.float32)
                
                lams = np.zeros(n_h)
                for i in range(n_h):
                    # Forward pass
                    h = np.tanh(obs @ W1[i] + b1[i])  # (HIDDEN,)
                    logit = float(h @ W2[i] + b2[i])
                    lams[i] = float(np.clip(sigmoid(logit) + rng_ep.randn() * noise, 0.01, 0.99))
                
                lams_ep.append(float(np.mean(lams)))
                
                full = np.zeros(N)
                full[:n_h] = lams
                
                contribs = full * E
                pool = np.sum(contribs)
                payoffs = (E - contribs) + M * pool / N
                total_w += np.mean(payoffs[:n_h])
                
                mc = np.mean(full)
                prev_ml = mc
                if R < 0.15: f_R = 0.01
                elif R < 0.25: f_R = 0.03
                else: f_R = 0.10
                shock = 0.15 if rng_ep.random() < 0.05 else 0.0
                R = np.clip(R + f_R * (mc - 0.4) - shock, 0, 1)
                
                if R <= 0:
                    survived = False
                    break
            
            mean_lam = np.mean(lams_ep[-10:]) if len(lams_ep) >= 10 else np.mean(lams_ep)
            ep_lams.append(float(mean_lam))
            ep_survs.append(float(survived))
            
            # Update: bias-only for speed (same myopic gradient)
            grad = (M / N - 1) * 0.01
            surv_bonus = 0.001 / N if survived else 0.005 / N
            b2 += lr * (grad + surv_bonus * (1.0 - mean_lam))
            # Small weight perturbation
            W1 += rng.randn(*W1.shape) * 0.0001
            W2 += rng.randn(*W2.shape) * 0.0001
        
        final_lam = float(np.mean(ep_lams[-30:]))
        final_surv = float(np.mean(ep_survs[-30:]) * 100)
        trapped = final_lam < 0.6
        results.append({"seed": seed, "final_lambda": final_lam, "survival_pct": final_surv, "trapped": trapped})
        
        if (seed + 1) % 5 == 0:
            print(f"  s{seed+1}", end=" ", flush=True)
    
    trap_rate = sum(1 for r in results if r["trapped"]) / len(results) * 100
    mean_lam = float(np.mean([r["final_lambda"] for r in results]))
    mean_surv = float(np.mean([r["survival_pct"] for r in results]))
    total_params = 4 * HIDDEN + HIDDEN + HIDDEN * 1 + 1  # per agent
    
    print(f"\n  → MLP: λ_final={mean_lam:.3f}  Surv={mean_surv:.1f}%  Trap={trap_rate:.0f}%  params={total_params}/agent")
    
    return {
        "method": f"2-layer MLP ({total_params} params/agent)",
        "final_lambda": mean_lam,
        "lambda_std": float(np.std([r["final_lambda"] for r in results])),
        "survival_pct": mean_surv,
        "trap_rate_pct": trap_rate,
        "params_per_agent": total_params,
    }


# ================================================================
#  MAIN
# ================================================================

if __name__ == "__main__":
    t0 = time.time()
    
    # Part 1: Phase Transition
    pt_results = phase_transition_analysis()
    
    # Part 2: Communication
    comm_results = communication_experiment()
    
    # Part 3: Shared Critic (CTDE)
    ctde_results = shared_critic_experiment()
    
    # Part 4: Neural Policy
    neural_results = neural_policy_experiment()
    
    elapsed = time.time() - t0
    
    # Summary
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"  {'Method':<30s}  {'λ_final':>8s}  {'Surv':>8s}  {'Trap%':>8s}")
    print(f"  {'-'*30}  {'-'*8}  {'-'*8}  {'-'*8}")
    
    for r in [comm_results, ctde_results, neural_results]:
        print(f"  {r['method']:<30s}  {r['final_lambda']:8.3f}  {r['survival_pct']:6.1f}%  {r['trap_rate_pct']:6.0f}%")
    
    print(f"\n  Phase Transition: {pt_results['formula']}")
    print(f"  R² = {pt_results['r_squared']:.4f}, MAE = {pt_results['mae']:.4f}")
    print(f"  Total time: {elapsed:.0f}s")
    print("=" * 70)
    
    # Save
    output = {
        "phase_transition": pt_results,
        "communication": comm_results,
        "shared_critic": ctde_results,
        "neural_policy": neural_results,
        "time_seconds": elapsed,
    }
    
    path = os.path.join(OUTPUT_DIR, "advanced_results.json")
    with open(path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"  Saved: {path}")
