param(
    [string] $Root = 'data/cudnn',
    [ValidateSet('all', 'runtime')] [string] $Only = 'all'
)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
Add-Type -AssemblyName System.IO.Compression.FileSystem

$packages = @(
    @{ name = 'nvidia-cuda-runtime-cu12'; version = $null }
)
if ($Only -eq 'all') {
    $packages += @{ name = 'nvidia-cudnn-cu12'; version = $null }
    $packages += @{ name = 'nvidia-cublas-cu12'; version = $null }
}

New-Item -ItemType Directory -Force $Root | Out-Null
$downloads = Join-Path $Root 'downloads'
New-Item -ItemType Directory -Force $downloads | Out-Null

foreach ($package in $packages) {
    $meta = Invoke-RestMethod -Uri "https://pypi.org/pypi/$($package.name)/json"
    $wheel = $null
    foreach ($release in ($meta.releases.PSObject.Properties | Sort-Object { [version]($_.Name -replace '[^0-9.]', '') } -Descending)) {
        $wheel = $release.Value | Where-Object { $_.filename -match 'win_amd64' } | Select-Object -First 1
        if ($wheel) { break }
    }
    if (-not $wheel) { throw "no Windows wheel found for $($package.name)" }

    $archive = Join-Path $downloads ($wheel.filename -replace '\.whl$', '.zip')
    if (-not (Test-Path $archive)) { Invoke-WebRequest -Uri $wheel.url -OutFile $archive }
    $hash = (Get-FileHash $archive -Algorithm SHA256).Hash.ToLower()
    if ($hash -ne $wheel.digests.sha256) { throw "checksum mismatch for $($wheel.filename)" }

    $zip = [IO.Compression.ZipFile]::OpenRead((Resolve-Path $archive))
    try {
        foreach ($entry in $zip.Entries) {
            if ($entry.FullName -match '\.dll$') {
                [IO.Compression.ZipFileExtensions]::ExtractToFile($entry, (Join-Path $Root $entry.Name), $true)
            }
        }
    } finally { $zip.Dispose() }
    Write-Host "$($wheel.filename) extracted into $Root"
}
