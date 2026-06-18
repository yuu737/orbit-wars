param(
    [string]$Dir = ""
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

if ([string]::IsNullOrWhiteSpace($Dir)) {
    $latest = Get-ChildItem "research_runs" -Directory -Filter "4p_candidate_eval_*" |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1
    if ($null -eq $latest) {
        throw "No 4p_candidate_eval_* directory found under research_runs."
    }
    $Dir = $latest.FullName
}

$rows = @()
foreach ($file in Get-ChildItem $Dir -File -Filter "*.txt" | Sort-Object Name) {
    $text = (Get-Content $file.FullName -Raw) -replace "`0", ""
    $candidate = [regex]::Match($text, "Candidate:\s*(.+)").Groups[1].Value.Trim()
    $mode = [regex]::Match($text, "Mode:\s*(.+)").Groups[1].Value.Trim()
    $wins = [regex]::Match($text, "Wins:\s*(\d+)\s*\(([^)]+)\)").Groups
    $draws = [regex]::Match($text, "Draws:\s*(\d+)\s*\(([^)]+)\)").Groups
    $losses = [regex]::Match($text, "Losses:\s*(\d+)\s*\(([^)]+)\)").Groups
    $diff = [regex]::Match($text, "Average score diff:\s*([-0-9.]+)").Groups[1].Value
    $place = [regex]::Match($text, "Average placement:\s*([-0-9.]+)").Groups[1].Value
    $survival = [regex]::Match($text, "Average survival turn:\s*([-0-9.]+)").Groups[1].Value
    $crash = [regex]::Match($text, "Crash rate:\s*([0-9.]+%)").Groups[1].Value

    $rows += [pscustomobject]@{
        Candidate = $candidate
        Mode = $mode
        Wins = if ($wins.Count -gt 1) { [int]$wins[1].Value } else { $null }
        WinRate = if ($wins.Count -gt 2) { $wins[2].Value } else { "" }
        Draws = if ($draws.Count -gt 1) { [int]$draws[1].Value } else { $null }
        Losses = if ($losses.Count -gt 1) { [int]$losses[1].Value } else { $null }
        AvgDiff = if ($diff) { [double]$diff } else { $null }
        AvgPlace = if ($place) { [double]$place } else { $null }
        AvgSurvival = if ($survival) { [double]$survival } else { $null }
        Crash = $crash
        File = $file.Name
    }
}

$rows |
    Sort-Object @{ Expression = "Wins"; Descending = $true }, @{ Expression = "AvgPlace"; Ascending = $true }, @{ Expression = "AvgDiff"; Descending = $true } |
    Format-Table -AutoSize
