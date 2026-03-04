Write-Host "Waiting for hp_sweep_results.json to be completed..."
while (!(Test-Path -Path "..\outputs\cleanrl_baselines\hp_sweep_results.json")) {
    Start-Sleep -Seconds 30
}
Write-Host "HP Sweep completed. Injecting tables..."
python inject_tables.py

Write-Host "Compiling LaTeX..."
cd ..\paper
pdflatex unified_paper.tex
bibtex unified_paper
pdflatex unified_paper.tex
pdflatex unified_paper.tex

Write-Host "Verifying submission..."
cd ..\code\scripts
python verify_submission.py > verification_report.txt

Write-Host "Deploying to GitHub..."
cd ..\..
git add -A
git commit -m "chore: Final NeurIPS 2026 Camera-Ready with all reviewer responses (v3)"
git push origin HEAD

Write-Host "Packaging ZIP..."
if (Test-Path -Path "..\NeurIPS2026_EthicaAI_Final_Submission.zip") {
    Remove-Item -Path "..\NeurIPS2026_EthicaAI_Final_Submission.zip" -Force
}
Compress-Archive -Path * -DestinationPath "..\NeurIPS2026_EthicaAI_Final_Submission.zip" -Force

Write-Host "ALL DONE! Validation report saved in code/scripts/verification_report.txt"
