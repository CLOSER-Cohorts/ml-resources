param(
    [string]$Branch = "issue-91-create-powershell-script-for-ci-cd"
)

# Go to your repository
cd ..

Write-Host "Pulling latest commit from branch '$Branch'..."

# Fetch the latest from the remote
git fetch origin

# Reset local branch to match remote
git reset --hard origin/$Branch

Write-Host "Repository is now up-to-date with origin/$Branch."