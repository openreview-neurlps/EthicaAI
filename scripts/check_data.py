"""데이터 구조 심층 분석: 에이전트 간 분산 데이터 확인"""
import json
with open("simulation/outputs/run_large_1771029971/sweep_large_1771030421.json") as f:
    data = json.load(f)

for cond_name in data.keys():
    cond = data[cond_name]
    runs = cond["runs"]
    print(f"\n=== {cond_name} (θ={cond['theta']:.3f}) ===")
    for i, run in enumerate(runs):
        metrics = run["metrics"]
        # threshold_clean_std가 있는지 확인
        if "threshold_clean_std" in metrics:
            tc_std = metrics["threshold_clean_std"]
            print(f"  Run {i}: threshold_clean_std last={tc_std[-1]:.4f}, max={max(tc_std):.4f}")
        if "threshold_clean_mean" in metrics:
            tc_mean = metrics["threshold_clean_mean"]
            print(f"          threshold_clean_mean last={tc_mean[-1]:.4f}")
        # cooperation_rate도
        if "cooperation_rate" in metrics:
            coop = metrics["cooperation_rate"]
            print(f"          cooperation_rate last={coop[-1]:.4f}")
        # final_thresholds_mean
        ft = run.get("final_thresholds_mean", {})
        print(f"          final_thresholds_mean: clean={ft.get('clean', 'N/A')}, harvest={ft.get('harvest', 'N/A')}")
        if i >= 1:
            break  # 2개만
