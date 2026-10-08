$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
foreach ($mode in @('supervised', 'unsupervised')) {
    $appName = if ($mode -eq 'supervised') { 'ICPMS_Supervised' } else { 'ICPMS_Unsupervised' }
    & .\.venv\Scripts\python.exe -m PyInstaller --noconfirm --onefile --windowed --name $appName --exclude-module matplotlib.tests --exclude-module numpy.tests --exclude-module scipy.tests "${mode}_app.py"
    if ($LASTEXITCODE -ne 0) { throw "Build failed: $appName" }
}
