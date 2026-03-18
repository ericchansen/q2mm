param(
    [string]$WorktreePath = "",
    [string]$Ref = ""
)

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$repoName = Split-Path $repoRoot -Leaf

if (-not $WorktreePath) {
    $WorktreePath = Join-Path (Split-Path $repoRoot -Parent) "$repoName-upstream-worktree"
}

if (-not $Ref) {
    $fixturePath = Join-Path $repoRoot "test\fixtures\seminario_parity\rh_enamide_reference.json"
    if (Test-Path $fixturePath) {
        try {
            $fixture = Get-Content $fixturePath -Raw | ConvertFrom-Json
            if ($fixture.metadata.upstream_commit) {
                $Ref = $fixture.metadata.upstream_commit
            }
        } catch {
            # Fall back to upstream/master below.
        }
    }
}

if (-not $Ref) {
    $Ref = "upstream/master"
}

git -C $repoRoot fetch upstream --quiet

if (Test-Path $WorktreePath) {
    $null = git -C $WorktreePath rev-parse --show-toplevel 2>$null
    if (-not $?) {
        throw "Path exists but is not a git worktree: $WorktreePath"
    }
    git -C $WorktreePath fetch upstream --quiet
    git -C $WorktreePath checkout --detach $Ref | Out-Null
} else {
    git -C $repoRoot worktree add --detach $WorktreePath $Ref | Out-Null
}

$commit = (git -C $WorktreePath rev-parse HEAD).Trim()
Write-Output "Ready: $WorktreePath @ $commit"
