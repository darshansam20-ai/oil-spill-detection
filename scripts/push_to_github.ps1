# ==============================================================================
# AEGIS-SAR Git Push Automation Script for GitHub
# Repository: https://github.com/darshansam20-ai/oil-spill-detection.git
# ==============================================================================

Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "         PUSHING AEGIS-SAR PRODUCTION REPOSITORY TO GITHUB                      " -ForegroundColor Cyan
Write-Host "================================================================================" -ForegroundColor Cyan

$RepoUrl = "https://github.com/darshansam20-ai/oil-spill-detection.git"

# Check if git is available
if (-not (Get-Command git.exe -ErrorAction SilentlyContinue)) {
    Write-Host "[!] Git command not found in current PATH." -ForegroundColor Yellow
    Write-Host "[*] Please make sure Git is installed or use GitHub Desktop / VS Code Git tab." -ForegroundColor Yellow
    exit 1
}

# Initialize Git repository if not initialized
if (-not (Test-Path ".git")) {
    Write-Host "[*] Initializing local Git repository..." -ForegroundColor Green
    git init
}

# Configure Git LFS for 383MB model weights
Write-Host "[*] Configuring Git LFS for large PyTorch checkpoints..." -ForegroundColor Green
git lfs install
git lfs track "artifacts/checkpoints/*.pt"
git lfs track "*.pth"

# Add files
Write-Host "[*] Staging production code, web application, and Docker configurations..." -ForegroundColor Green
git add .

# Commit
Write-Host "[*] Creating production commit..." -ForegroundColor Green
git commit -m "AEGIS-SAR Production Release: Dual-Model Pipeline, Web App & Render/Vercel Suite"

# Set main branch and remote
git branch -M main
git remote remove origin -ErrorAction SilentlyContinue
git remote add origin $RepoUrl

# Push
Write-Host "[*] Pushing to $RepoUrl (branch: main)..." -ForegroundColor Cyan
git push -u origin main

Write-Host "`n✓ [SUCCESS] Repository pushed to GitHub!" -ForegroundColor Green
Write-Host "  View repository at: https://github.com/darshansam20-ai/oil-spill-detection" -ForegroundColor Green
