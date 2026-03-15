import glob
import os

print("Patching python scripts...")
count = 0
for f in glob.glob('d:/00.test/PAPER/EthicaAI/NeurIPS2026_final_submission/code/scripts/*.py'):
    with open(f, 'r', encoding='utf-8') as file:
        content = file.read()
    
    lines = content.split('\n')
    new_lines = []
    changed = False
    
    for line in lines:
        if line.lstrip().startswith('OUTPUT_DIR =') and '"outputs"' in line:
            line = line.replace('"outputs"', 'os.environ.get("ETHICAAI_OUTDIR", "outputs")')
            changed = True
        new_lines.append(line)
        
    if changed:
        if 'import os' not in content:
            new_lines.insert(0, 'import os')
        with open(f, 'w', encoding='utf-8') as file:
            file.write('\n'.join(new_lines))
        count += 1

print(f"Patched {count} files.")
