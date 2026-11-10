param(
    [ValidateSet('mnist', 'cifar', 'all')] [string] $Dataset = 'all',
    [string] $Root = 'data'
)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

function Expand-Gzip([string] $Source, [string] $Target) {
    $in = [IO.File]::OpenRead($Source)
    try {
        $gzip = New-Object IO.Compression.GZipStream($in, [IO.Compression.CompressionMode]::Decompress)
        $out = [IO.File]::Create($Target)
        try { $gzip.CopyTo($out) } finally { $out.Close(); $gzip.Close() }
    } finally { $in.Close() }
}

if ($Dataset -in 'mnist', 'all') {
    $dir = Join-Path $Root 'mnist'
    New-Item -ItemType Directory -Force $dir | Out-Null
    foreach ($name in 'train-images-idx3-ubyte', 'train-labels-idx1-ubyte', 't10k-images-idx3-ubyte', 't10k-labels-idx1-ubyte') {
        $target = Join-Path $dir $name
        if (Test-Path $target) { continue }
        $archive = "$target.gz"
        Invoke-WebRequest -Uri "https://ossci-datasets.s3.amazonaws.com/mnist/$name.gz" -OutFile $archive
        Expand-Gzip $archive $target
        Remove-Item $archive
    }
    Write-Host "MNIST ready in $dir"
}

if ($Dataset -in 'cifar', 'all') {
    $dir = Join-Path $Root 'cifar'
    New-Item -ItemType Directory -Force $dir | Out-Null
    if (-not (Test-Path (Join-Path $dir 'test_batch.bin'))) {
        $work = Join-Path $Root 'cifar-download'
        New-Item -ItemType Directory -Force $work | Out-Null
        $archive = Join-Path $work 'cifar-10-binary.tar.gz'
        Invoke-WebRequest -Uri 'https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz' -OutFile $archive
        $hash = (Get-FileHash $archive -Algorithm MD5).Hash.ToLower()
        if ($hash -ne 'c32a1d4ab5d03f1284b67883e8d87530') { throw "unexpected CIFAR-10 checksum $hash" }
        tar -xzf $archive -C $work
        Copy-Item (Join-Path $work 'cifar-10-batches-bin\*.bin') $dir
        Remove-Item -Recurse -Force $work
    }
    Write-Host "CIFAR-10 ready in $dir"
}
