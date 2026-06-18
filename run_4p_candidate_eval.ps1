param(
    [int]$Games = 5,
    [int]$Workers = 5,
    [Int64]$SeedStart = 56000000,
    [string]$Python = "C:\tmp\ow\Scripts\python.exe",
    [string]$Mode = "pool"
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$outDir = Join-Path $root "research_runs\4p_candidate_eval_$timestamp"
New-Item -ItemType Directory -Force -Path $outDir | Out-Null

$candidates = @(
    @{ Name = "sample7"; Path = "sample7\main.py" },
    @{ Name = "sample8"; Path = "sample8\main.py" },
    @{ Name = "hairate5"; Path = "bots\hairate5.py" },
    @{ Name = "sample23_anchor_line_current"; Path = "sample23_4p_anchor_line_planner_s8_2p_s7_4p\main.py" },
    @{ Name = "sample24_prize_targets"; Path = "sample24_4p_s7_prize_targets\main.py" },
    @{ Name = "sample25_prize_candidate_only"; Path = "sample25_4p_s7_prize_targets_candidate_only\main.py" },
    @{ Name = "sample29_stateful_domain"; Path = "sample29_4p_stateful_domain_planner\main.py" }
)

$poolOpponents = @(
    "sample7\main.py",
    "sample8\main.py",
    "bots\hairate5.py"
)

$hairate5Only = @("bots\hairate5.py")

function Invoke-Eval {
    param(
        [hashtable]$Candidate,
        [string[]]$Opponents,
        [string]$Label
    )

    $safeName = $Candidate.Name -replace '[^A-Za-z0-9_\-]', '_'
    $outFile = Join-Path $outDir "$safeName`_$Label.txt"

    Write-Host ""
    Write-Host "===== RUNNING $($Candidate.Name) [$Label] ====="
    Write-Host "Agent: $($Candidate.Path)"
    Write-Host "Output: $outFile"

    $args = @(
        "evaluate.py",
        "--players", "4",
        "--agent", $Candidate.Path,
        "--games", "$Games",
        "--workers", "$Workers",
        "--seed-start", "$SeedStart"
    )

    foreach ($opp in $Opponents) {
        $args += @("--opponent", $opp)
    }

    $header = @(
        "Candidate: $($Candidate.Name)",
        "Agent: $($Candidate.Path)",
        "Mode: $Label",
        "Games: $Games",
        "Workers: $Workers",
        "SeedStart: $SeedStart",
        "Opponents: $($Opponents -join ', ')",
        ""
    )
    $header | Set-Content -Path $outFile -Encoding UTF8

    & $Python @args 2>&1 | ForEach-Object {
        Write-Host $_
        Add-Content -Path $outFile -Value $_ -Encoding UTF8
    }
}

Write-Host "Output directory: $outDir"
Write-Host "Mode: $Mode"

foreach ($candidate in $candidates) {
    if (-not (Test-Path $candidate.Path)) {
        Write-Host "SKIP missing: $($candidate.Name) -> $($candidate.Path)"
        continue
    }

    if ($Mode -eq "pool" -or $Mode -eq "both") {
        Invoke-Eval -Candidate $candidate -Opponents $poolOpponents -Label "pool"
    }
    if ($Mode -eq "hairate5" -or $Mode -eq "both") {
        Invoke-Eval -Candidate $candidate -Opponents $hairate5Only -Label "hairate5"
    }
}

Write-Host ""
Write-Host "Done. Results saved to: $outDir"
