"""
EthicaAI Submission Packager
투고용 패키지(submission/)를 생성하고 정리합니다.
"""
import os
import shutil
import glob

def create_submission_package():
    base_dir = os.getcwd()
    sub_dir = os.path.join(base_dir, "submission")
    
    # 1. Clean & Create Directory
    if os.path.exists(sub_dir):
        shutil.rmtree(sub_dir)
    os.makedirs(sub_dir)
    os.makedirs(os.path.join(sub_dir, "figures"))
    os.makedirs(os.path.join(sub_dir, "src"))
    
    print(f"Created submission directory: {sub_dir}")
    
    # 2. Copy Figures (Latest Run)
    # Find latest large run figures
    run_base = os.path.join(base_dir, "simulation", "outputs")
    runs = [d for d in os.listdir(run_base) if d.startswith("run_large_")]
    if runs:
        latest_run = sorted(runs)[-1]
        fig_src = os.path.join(run_base, latest_run, "figures")
        if os.path.exists(fig_src):
            for img in glob.glob(os.path.join(fig_src, "*.png")):
                shutil.copy(img, os.path.join(sub_dir, "figures"))
            print(f"Copied figures from {latest_run}")
            
    # 3. Copy Key Source Code
    src_files = [
        "simulation/jax/algo/mappo_jax.py",
        "simulation/jax/env/cleanup_jax.py",
        "simulation/jax/meta/meta_ranking.py",
        "simulation/jax/config.py",
        "simulation/jax/reanalyze.py",
    ]
    
    for f in src_files:
        src_path = os.path.join(base_dir, f)
        dest_path = os.path.join(sub_dir, "src", os.path.basename(f))
        if os.path.exists(src_path):
            shutil.copy(src_path, dest_path)
            
    # 4. Copy Documents
    docs = ["paper_draft.md", "README.md", "requirements.txt", "comprehensive_evaluation.md"]
    for d in docs:
        if os.path.exists(d):
            shutil.copy(d, os.path.join(sub_dir, d))
            
    # 5. Execute HTML Conversion
    # (Checking if Markdown package is installed, if not, skip)
    try:
        import markdown
        print("Converting Paper to HTML...")
        os.system("python convert_to_html.py")
    except ImportError:
        print("Markdown package not found. Skipping HTML conversion.")
        
    print("\nPackage Creation Complete!")
    print(f"Location: {sub_dir}")

if __name__ == "__main__":
    create_submission_package()
