# Times two builds of a program against each other and prints the minimum and median of each.
#
#   scripts/compare.ps1 -Base <old exe> -New <new exe> -ProgramArgs 300,--gpu -Pattern 'training took ([\d.]+) s'
#
# Pattern is a regex whose first group captures seconds; the last match in the output counts. The examples print
# "training took ... s" and the trainer prints the epoch time, so start-up, data loading and sampling stay out.
# One warm-up run per side comes first, then the sides alternate so that drift in clocks or temperature hits both alike.
# Build the baseline in a worktree (git worktree add) and copy the exe elsewhere if it would be rebuilt meanwhile.
param(
    [Parameter(Mandatory)][string]$Base,
    [Parameter(Mandatory)][string]$New,
    [string[]]$ProgramArgs = @(),
    [Parameter(Mandatory)][string]$Pattern,
    [int]$Rounds = 5
)
$ErrorActionPreference = 'Stop'

function Measure-Once([string]$exe) {
    $out = & $exe @ProgramArgs 2>&1 | Out-String
    $m = [regex]::Matches($out, $Pattern)
    if ($m.Count -eq 0) { throw "no timing in output of $exe" }
    return [double]$m[$m.Count - 1].Groups[1].Value
}

$null = Measure-Once $Base
$null = Measure-Once $New

$times = @{ base = @(); new = @() }
for ($r = 0; $r -lt $Rounds; $r++) {
    $order = if ($r % 2 -eq 0) { @('base', 'new') } else { @('new', 'base') }
    foreach ($side in $order) {
        $exe = if ($side -eq 'base') { $Base } else { $New }
        $times[$side] += Measure-Once $exe
    }
}

$middle = [int][math]::Floor($Rounds / 2)
foreach ($side in 'base', 'new') {
    $sorted = $times[$side] | Sort-Object
    "{0,-4}  min {1,8:N3}  median {2,8:N3}  all {3}" -f $side, $sorted[0], $sorted[$middle], (($times[$side] | ForEach-Object { $_.ToString('N3') }) -join ' ')
}
$bm = ($times.base | Sort-Object)[$middle]
$nm = ($times.new | Sort-Object)[$middle]
"median change: {0:N1}%" -f (100 * ($nm - $bm) / $bm)
