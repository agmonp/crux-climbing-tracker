$ErrorActionPreference = 'Stop'

$appRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$pythonw = Join-Path $appRoot '.venv\Scripts\pythonw.exe'
$launcher = Join-Path $appRoot 'launch_crux.pyw'
$icon = Join-Path $appRoot 'assets\crux-v2.ico'
$desktop = [Environment]::GetFolderPath('Desktop')
$hebrewName = -join (0x05E0, 0x05D9, 0x05EA, 0x05D5, 0x05D7, 0x20, 0x05D8, 0x05D9, 0x05E4, 0x05D5, 0x05E1 | ForEach-Object { [char]$_ })
$shortcutPath = Join-Path $desktop ('CRUX - ' + $hebrewName + '.lnk')

foreach ($required in @($pythonw, $launcher, $icon)) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
        throw "Required CRUX file is missing: $required"
    }
}

$shell = New-Object -ComObject WScript.Shell
$shortcut = $shell.CreateShortcut($shortcutPath)
$shortcut.TargetPath = $pythonw
$shortcut.Arguments = '"' + $launcher + '"'
$shortcut.WorkingDirectory = $appRoot
$shortcut.IconLocation = $icon + ',0'
$shortcut.Description = 'Open CRUX climbing video analysis'
$shortcut.WindowStyle = 7
$shortcut.Save()

Write-Output $shortcutPath
