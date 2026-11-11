param(
    [ValidateSet('mnist', 'fashion', 'cifar', 'cifar100', 'text', 'all')] [string] $Dataset = 'all',
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

if ($Dataset -in 'fashion', 'all') {
    $dir = Join-Path $Root 'fashion'
    New-Item -ItemType Directory -Force $dir | Out-Null
    $sums = @{
        'train-images-idx3-ubyte' = '8d4fb7e6c68d591d4c3dfef9ec88bf0d'
        'train-labels-idx1-ubyte' = '25c81989df183df01b3e8a0aad5dffbe'
        't10k-images-idx3-ubyte'  = 'bef4ecab320f06d8554ea6380940ec79'
        't10k-labels-idx1-ubyte'  = 'bb300cfdad3c16e7a12a480ee83cd310'
    }
    foreach ($name in $sums.Keys) {
        $target = Join-Path $dir $name
        if (Test-Path $target) { continue }
        $archive = "$target.gz"
        Invoke-WebRequest -Uri "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/$name.gz" -OutFile $archive
        $hash = (Get-FileHash $archive -Algorithm MD5).Hash.ToLower()
        if ($hash -ne $sums[$name]) { throw "unexpected checksum $hash for $name" }
        Expand-Gzip $archive $target
        Remove-Item $archive
    }
    Write-Host "Fashion-MNIST ready in $dir"
}

if ($Dataset -in 'text', 'all') {
    $dir = Join-Path $Root 'text'
    New-Item -ItemType Directory -Force $dir | Out-Null
    $target = Join-Path $dir 'shakespeare.txt'
    if (-not (Test-Path $target)) {
        Invoke-WebRequest -Uri 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt' -OutFile $target
        $hash = (Get-FileHash $target -Algorithm MD5).Hash.ToLower()
        if ($hash -ne '6fb458f1232090904fb40fe944165e91') { throw "unexpected Tiny Shakespeare checksum $hash" }
    }
    Write-Host "Tiny Shakespeare ready in $dir"
}

if ($Dataset -in 'cifar100', 'all') {
    $dir = Join-Path $Root 'cifar100'
    New-Item -ItemType Directory -Force $dir | Out-Null
    if (-not (Test-Path (Join-Path $dir 'test.bin'))) {
        $work = Join-Path $Root 'cifar100-download'
        New-Item -ItemType Directory -Force $work | Out-Null
        $archive = Join-Path $work 'cifar-100-binary.tar.gz'
        Invoke-WebRequest -Uri 'https://www.cs.toronto.edu/~kriz/cifar-100-binary.tar.gz' -OutFile $archive
        $hash = (Get-FileHash $archive -Algorithm MD5).Hash.ToLower()
        if ($hash -ne '03b5dce01913d631647c71ecec9e9cb8') { throw "unexpected CIFAR-100 checksum $hash" }
        tar -xzf $archive -C $work
        Copy-Item (Join-Path $work 'cifar-100-binary\train.bin') $dir
        Copy-Item (Join-Path $work 'cifar-100-binary\test.bin') $dir
        Remove-Item -Recurse -Force $work
    }
    Write-Host "CIFAR-100 ready in $dir"
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
