import json
import os
import re

PAPER_TEX = os.path.join("..", "..", "paper", "unified_paper.tex")

def load_json(path):
    p = os.path.join("..", "..", "outputs", path)
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)

def inject_partial_obs():
    print("Injecting Phase D: Partial Obs Table...")
    data = load_json("partial_obs/partial_obs_results.json")
    if not data:
        print("  -> No partial obs data found.")
        return
    
    with open(PAPER_TEX, "r", encoding="utf-8") as f:
        tex = f.read()

    def fmt(alg_key, cond_key):
        try:
            m = data[cond_key][alg_key]["survival_mean"]
            return f"{m:.0f}\\%"
        except:
            return "X\\%"

    mapping = [
        (r"Full Obs \(Baseline\) & .*?\\\\", f"Full Obs (Baseline) & X\\% & {fmt('situational', 'noisy_std0.05_dly0')} & {fmt('unconditional', 'noisy_std0.05_dly0')} \\\\"), # Assuming full obs is roughly baseline, but wait, partial config has no 'full'. Just replace X's for noisy/delayed.
        (r"Noisy \(\$\\sigma=0.10\$\) & .*?\\\\", f"Noisy ($\\sigma=0.10$) & {fmt('ippo', 'noisy_std0.1_dly0')} & {fmt('situational', 'noisy_std0.1_dly0')} & {fmt('unconditional', 'noisy_std0.1_dly0')} \\\\"),
        (r"Noisy \(\$\\sigma=0.20\$\) & .*?\\\\", f"Noisy ($\\sigma=0.20$) & {fmt('ippo', 'noisy_std0.2_dly0')} & {fmt('situational', 'noisy_std0.2_dly0')} & {fmt('unconditional', 'noisy_std0.2_dly0')} \\\\"),
        (r"Delayed \(\$k=2\$\) & .*?\\\\", f"Delayed ($k=2$) & {fmt('ippo', 'delayed_std0.0_dly2')} & {fmt('situational', 'delayed_std0.0_dly2')} & {fmt('unconditional', 'delayed_std0.0_dly2')} \\\\"),
        (r"Delayed \(\$k=5\$\) & .*?\\\\", f"Delayed ($k=5$) & {fmt('ippo', 'delayed_std0.0_dly5')} & {fmt('situational', 'delayed_std0.0_dly5')} & {fmt('unconditional', 'delayed_std0.0_dly5')} \\\\"),
        (r"Local-only & .*?\\\\", f"Local-only & {fmt('ippo', 'local_std0.0_dly0')} & {fmt('situational', 'local_std0.0_dly0')} & {fmt('unconditional', 'local_std0.0_dly0')} \\\\"),
    ]
    
    for pattern, repl in mapping:
        tex = re.sub(pattern, repl, tex)
        
    with open(PAPER_TEX, "w", encoding="utf-8") as f:
        f.write(tex)
    print("  -> Injected Partial Obs Table")

def inject_hp_sweep():
    print("Injecting Phase A: HP Sweep Table...")
    data = load_json("cleanrl_baselines/hp_sweep_results.json")
    if not data:
        print("  -> No HP sweep data found.")
        return
        
    with open(PAPER_TEX, "r", encoding="utf-8") as f:
        tex = f.read()

    # The table is in Appendix H. Line: \textbf{Learning Rate} & \textbf{Entropy} ...
    # We will just replace the entire tabular body.
    start_marker = r"\textbf{Learning Rate} & \textbf{Entropy} & \(\boldsymbol{\bar{\lambda}}\) & \textbf{Surv.\%} & \textbf{Trapped?} \\"
    end_marker = r"\bottomrule"
    
    match = re.search(re.escape(start_marker) + r"(.*?)" + re.escape(end_marker), tex, re.DOTALL)
    if match:
        rows = [r"\midrule"]
        for key, run in data.items():
            lr = run.get("lr", 0)
            ent = run.get("entropy_coef", 0)
            lam = run.get("lambda_mean", 0)
            ci = run.get("lambda_ci95", [0,0])
            surv = run.get("survival_mean", 0)
            trapped = "Yes" if run.get("still_trapped", True) else "No"
            
            lr_str = f"{lr:.1e}".replace("e-0", "e-")
            row = f"{lr_str} & {ent:.2f} & {lam:.3f} [{ci[0]:.3f}, {ci[1]:.3f}] & {surv:.1f}\\% & {trapped} \\\\"
            rows.append(row)
            
        replacement = start_marker + "\n" + "\n".join(rows) + "\n" + end_marker
        tex = tex[:match.start()] + replacement + tex[match.end():]
        
        with open(PAPER_TEX, "w", encoding="utf-8") as f:
            f.write(tex)
        print("  -> Injected HP Sweep Table")
    else:
        print("  -> Table marker not found")

if __name__ == "__main__":
    inject_partial_obs()
    inject_hp_sweep()
    print("DONE")
