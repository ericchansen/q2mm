param(
    [ValidateSet("fixture", "live", "both")]
    [string]$Mode = "fixture",
    [string[]]$Case = @(),
    [string]$WorktreePath = "",
    [string]$ReportJson = ""
)

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$repoName = Split-Path $repoRoot -Leaf

if (-not $WorktreePath) {
    $WorktreePath = Join-Path (Split-Path $repoRoot -Parent) "$repoName-upstream-worktree"
}

if ($Mode -eq "live" -or $Mode -eq "both") {
    powershell -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot "setup_upstream_worktree.ps1") -WorktreePath $WorktreePath | Out-Host
}

$args = @(
    (Join-Path $PSScriptRoot "validate_against_upstream.py"),
    "--mode",
    $Mode
)

if ($WorktreePath) {
    $args += @("--worktree", $WorktreePath)
}

if ($ReportJson) {
    $args += @("--report-json", $ReportJson)
}

foreach ($caseId in $Case) {
    $args += @("--case", $caseId)
}

Set-Location $repoRoot
python @args
