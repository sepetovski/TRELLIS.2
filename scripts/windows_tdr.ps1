# Run in an elevated Windows PowerShell, then reboot.
# This does not add VRAM. It stops Windows from resetting the GPU when one
# CUDA kernel (a large sparse conv, or a DiT block streaming over PCIe) runs
# longer than the default 2 seconds. TRELLIS.2 on a 4 GB card is slow on
# purpose; the driver must be allowed to wait.
$ErrorActionPreference = "Stop"
$path = "HKLM:\SYSTEM\CurrentControlSet\Control\GraphicsDrivers"
if (-not (Test-Path $path)) {
    New-Item -Path $path -Force | Out-Null
}
New-ItemProperty -Path $path -Name "TdrDelay" -Value 60 -PropertyType DWord -Force | Out-Null
New-ItemProperty -Path $path -Name "TdrDdiDelay" -Value 60 -PropertyType DWord -Force | Out-Null
Write-Host "Set TdrDelay=60 and TdrDdiDelay=60. Reboot Windows before the next TRELLIS run."
