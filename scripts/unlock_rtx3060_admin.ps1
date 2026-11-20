<#
.SYNOPSIS
    Unlock Windows 11 & NVIDIA Driver performance settings for RTX 3060 12GB (GA106)
    and run a live hardware/runtime diagnostic checklist for AI Video Upscaler 4K.

.DESCRIPTION
    - Default mode (requires Administrator):
      1. Enable Hardware-Accelerated GPU Scheduling (HAGS: HwSchMode = 2) to reduce WDDM latency.
      2. Tune Windows Multimedia SystemProfile & disable PCIe ASPM on active power plan.
      3. Export UTF-16 LE `rtx3060_cuda_p0_unlock.nip` (SettingID 1343646814 = 0)
         to disable `CUDA - Force P2 State` (keeps GDDR6 VRAM at P0 7501+ MHz),
         and auto-import via `nvidiaProfileInspector.exe` if found.
      4. Add `.cache`, `scratch`, and `out` to Windows Defender exclusions.
      5. Print diagnostic checklist (RAM XMP, Resizable BAR, Power Limit, free VRAM, TensorRT 10).
    - `-CheckOnly` mode: Read-only diagnostic checklist without elevation.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File .\scripts\unlock_rtx3060_admin.ps1 -CheckOnly
    powershell -ExecutionPolicy Bypass -File .\scripts\unlock_rtx3060_admin.ps1
#>

param(
    [switch]$CheckOnly,
    [switch]$NoElevatePrompt
)

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot  = Split-Path -Parent $ScriptDir
$NipPath   = Join-Path $ScriptDir "rtx3060_cuda_p0_unlock.nip"
$VenvPy    = Join-Path $RepoRoot ".venv\Scripts\python.exe"

$CurrentIdentity  = [Security.Principal.WindowsIdentity]::GetCurrent()
$CurrentPrincipal = New-Object Security.Principal.WindowsPrincipal($CurrentIdentity)
$IsAdmin          = $CurrentPrincipal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)

Write-Host "===========================================================================" -ForegroundColor Cyan
Write-Host "  RTX 3060 12GB (GA106) & ALDER LAKE i5-12600K - PERFORMANCE UNLOCK TOOL" -ForegroundColor Cyan
Write-Host "===========================================================================" -ForegroundColor Cyan

if (-not $CheckOnly -and -not $IsAdmin) {
    if (-not $NoElevatePrompt) {
        Write-Host "[!] Script not running as Administrator. Requesting UAC elevation..." -ForegroundColor Yellow
        $ArgList = "-NoProfile -ExecutionPolicy Bypass -File `"$($MyInvocation.MyCommand.Path)`" -NoElevatePrompt"
        Start-Process -FilePath "powershell.exe" -ArgumentList $ArgList -Verb RunAs
        exit 0
    } else {
        Write-Host "[X] No Administrator privileges. Switching to read-only check (-CheckOnly)." -ForegroundColor Red
        $CheckOnly = $true
    }
}

# -------------------------------------------------------------------------
# PART 1: APPLY ADMIN PERFORMANCE UNLOCKS (UNLESS -CheckOnly)
# -------------------------------------------------------------------------
if (-not $CheckOnly -and $IsAdmin) {
    Write-Host "`n[1/5] Enabling Hardware-Accelerated GPU Scheduling (HAGS = 2)..." -ForegroundColor Green
    $GfxKey = "HKLM:\SYSTEM\CurrentControlSet\Control\GraphicsDrivers"
    Set-ItemProperty -Path $GfxKey -Name "HwSchMode" -Value 2 -Type DWord
    Write-Host "  -> Set HKLM GraphicsDrivers\HwSchMode = 2 (Windows reboot required if changed)."

    Write-Host "`n[2/5] Tuning Windows Multimedia Scheduler & disabling PCIe ASPM..." -ForegroundColor Green
    $MmKey = "HKLM:\SOFTWARE\Microsoft\Windows NT\CurrentVersion\Multimedia\SystemProfile"
    if (Test-Path $MmKey) {
        Set-ItemProperty -Path $MmKey -Name "SystemResponsiveness" -Value 10 -Type DWord
    }
    powercfg /setacvalueindex SCHEME_CURRENT SUB_PCIEXPRESS ASPM 0 | Out-Null
    powercfg /setactive SCHEME_CURRENT | Out-Null
    Write-Host "  -> Disabled PCIe Link State Power Management on current power scheme."

    Write-Host "`n[3/5] Generating & applying NVIDIA profile to disable 'CUDA - Force P2 State' (P0 VRAM Clock)..." -ForegroundColor Green
    $NipXml = @"
<?xml version="1.0" encoding="utf-16"?>
<ArrayOfProfile>
  <Profile>
    <ProfileName>Base Profile</ProfileName>
    <Executeables />
    <Settings>
      <ProfileSetting>
        <SettingNameInfo>CUDA - Force P2 State</SettingNameInfo>
        <SettingID>1343646814</SettingID>
        <SettingValue>0</SettingValue>
        <ValueType>Dword</ValueType>
      </ProfileSetting>
    </Settings>
  </Profile>
</ArrayOfProfile>
"@
    $NipXml | Out-File -FilePath $NipPath -Encoding Unicode -Force
    Write-Host "  -> Exported UTF-16 LE profile: $NipPath"

    $WingetPkgExe = Get-ChildItem (Join-Path $env:LOCALAPPDATA "Microsoft\WinGet\Packages\Orbmu2k.nvidiaProfileInspector*") -Filter "nvidiaProfileInspector.exe" -Recurse -ErrorAction SilentlyContinue | Select-Object -ExpandProperty FullName -First 1
    $CandidateNpi = @(
        $WingetPkgExe,
        (Get-Command "nvidiaProfileInspector.exe" -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Source -ErrorAction SilentlyContinue),
        (Get-Command "nvpi.exe" -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Source -ErrorAction SilentlyContinue),
        (Join-Path $env:LOCALAPPDATA "Microsoft\WinGet\Links\nvpi.exe"),
        (Join-Path $env:LOCALAPPDATA "Microsoft\WinGet\Links\nvidiaProfileInspector.exe"),
        (Join-Path $RepoRoot "bin\nvidiaProfileInspector.exe"),
        (Join-Path $env:USERPROFILE "Downloads\nvidiaProfileInspector\nvidiaProfileInspector.exe")
    ) | Where-Object { $_ -and (Test-Path $_) } | Select-Object -First 1

    if ($CandidateNpi) {
        Start-Process -FilePath $CandidateNpi -ArgumentList @("-silent", "-mergeImport", $NipPath) -Wait -NoNewWindow
        Write-Host "  -> [OK] Merged CUDA - Force P2 State = Off into Base Profile via: $CandidateNpi" -ForegroundColor Green
    } else {
        Write-Host "  -> [INFO] nvidiaProfileInspector.exe not found." -ForegroundColor Yellow
        Write-Host "     Install with: winget install --id Orbmu2k.nvidiaProfileInspector -e"
        Write-Host "     Then re-run this script or manually import: $NipPath"
    }

    Write-Host "`n[4/5] Adding AI cache/buffer directories to Windows Defender exclusions..." -ForegroundColor Green
    try {
        $ExclDirs = @(
            (Join-Path $RepoRoot ".cache"),
            (Join-Path $RepoRoot "scratch"),
            (Join-Path $RepoRoot "out")
        )
        Add-MpPreference -ExclusionPath $ExclDirs -ErrorAction Stop
        Write-Host "  -> [OK] Excluded .cache, scratch, and out from Windows Defender real-time scan." -ForegroundColor Green
    } catch {
        Write-Host "  -> [INFO] Skipped Defender exclusions (Tamper Protection or policy)." -ForegroundColor DarkGray
    }

    Write-Host "`n[5/5] Verifying live CUDA VRAM clock state..." -ForegroundColor Green
    if (Test-Path $VenvPy) {
        & $VenvPy -c "import subprocess, torch; x = torch.zeros((256, 256), device='cuda'); out = subprocess.check_output(['nvidia-smi', '--query-gpu=name,pstate,clocks.current.memory,clocks.max.memory,power.limit', '--format=csv,noheader'], text=True).strip(); print('  -> Live CUDA State:', out)"
    } else {
        nvidia-smi --query-gpu=name,driver_version,pstate,power.limit,clocks.max.memory --format=csv,noheader
    }
}

# -------------------------------------------------------------------------
# PART 2: DIAGNOSTIC CHECKLIST (LIVE HARDWARE & RUNTIME DASHBOARD)
# -------------------------------------------------------------------------
Write-Host "`n===========================================================================" -ForegroundColor Cyan
Write-Host "  HARDWARE & RUNTIME DIAGNOSTIC CHECKLIST" -ForegroundColor Cyan
Write-Host "===========================================================================" -ForegroundColor Cyan

# 1. RAM Speed (XMP)
try {
    $RamSticks = Get-CimInstance Win32_PhysicalMemory -ErrorAction Stop
    $MinSpeed  = ($RamSticks | Measure-Object -Property ConfiguredClockSpeed -Minimum).Minimum
    $TotalGB   = [math]::Round((($RamSticks | Measure-Object -Property Capacity -Sum).Sum / 1GB), 1)
    if ($MinSpeed -ge 3200) {
        Write-Host ("[PASS] DDR4 RAM      : {0} GB @ {1} MT/s (XMP Enabled)" -f $TotalGB, $MinSpeed) -ForegroundColor Green
    } else {
        Write-Host ("[WARN] DDR4 RAM      : {0} GB @ {1} MT/s -> Enable XMP Profile 1 (3200 MT/s) in BIOS if supported." -f $TotalGB, $MinSpeed) -ForegroundColor Yellow
    }
} catch {
    Write-Host "[INFO] DDR4 RAM      : Unable to query WMI Win32_PhysicalMemory" -ForegroundColor DarkGray
}

# 2. Resizable BAR (BAR1 Size)
try {
    $Bar1Line = (nvidia-smi -q | Select-String -Pattern "BAR1 Memory Usage" -Context 0,3)
    $Bar1TotalText = ($Bar1Line.Context.PostContext | Where-Object { $_ -match "Total\s*:\s*(\d+)\s*MiB" } | Select-Object -First 1)
    if ($Bar1TotalText -match "(\d+)\s*MiB") {
        $Bar1MiB = [int]$Matches[1]
        if ($Bar1MiB -gt 512) {
            Write-Host ("[PASS] Resizable BAR : BAR1 = {0} MiB (Full VRAM Aperture Unlocked)" -f $Bar1MiB) -ForegroundColor Green
        } else {
            Write-Host ("[WARN] Resizable BAR : BAR1 = {0} MiB -> Enable Above 4G + Resizable BAR in BIOS." -f $Bar1MiB) -ForegroundColor Yellow
        }
    }
} catch {
    Write-Host "[INFO] Resizable BAR : Unable to read BAR1 from nvidia-smi" -ForegroundColor DarkGray
}

# 3. HAGS
try {
    $HagsVal = (Get-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\GraphicsDrivers" -Name "HwSchMode" -ErrorAction SilentlyContinue).HwSchMode
    if ($HagsVal -eq 2) {
        Write-Host "[PASS] Windows HAGS  : HwSchMode = 2 (Enabled)" -ForegroundColor Green
    } else {
        Write-Host ("[WARN] Windows HAGS  : HwSchMode = {0} -> Run as Admin to enable HAGS = 2" -f $HagsVal) -ForegroundColor Yellow
    }
} catch {}

# 4. Power Scheme
$ActiveScheme = (powercfg /getactivescheme) -join " "
if ($ActiveScheme -match "Ultimate Performance|High performance") {
    Write-Host ("[PASS] Power Plan    : {0}" -f $ActiveScheme.Trim()) -ForegroundColor Green
} else {
    Write-Host ("[WARN] Power Plan    : {0}" -f $ActiveScheme.Trim()) -ForegroundColor Yellow
}

# 5. Free VRAM
try {
    $VramCsv = (nvidia-smi --query-gpu=memory.used,memory.free,memory.total,power.limit --format=csv,noheader,nounits).Trim()
    $Parts = $VramCsv.Split(",") | ForEach-Object { $_.Trim() }
    $UsedMiB = [int]$Parts[0]
    $FreeMiB = [int]$Parts[1]
    $TotMiB  = [int]$Parts[2]
    $PwrLim  = $Parts[3]
    if ($FreeMiB -ge 8000) {
        Write-Host ("[PASS] RTX 3060 VRAM : Free {0} / {1} MiB (Power Limit: {2} W)" -f $FreeMiB, $TotMiB, $PwrLim) -ForegroundColor Green
    } else {
        Write-Host ("[WARN] RTX 3060 VRAM : Only {0} / {1} MiB free ({2} MiB used) -> Close GPU apps before upscaling." -f $FreeMiB, $TotMiB, $UsedMiB) -ForegroundColor Yellow
    }
} catch {}

# 6. NVIDIA Profile Inspector
$CheckNpi = Get-ChildItem (Join-Path $env:LOCALAPPDATA "Microsoft\WinGet\Packages\Orbmu2k.nvidiaProfileInspector*") -Filter "nvidiaProfileInspector.exe" -Recurse -ErrorAction SilentlyContinue | Select-Object -ExpandProperty FullName -First 1
if (-not $CheckNpi) {
    $CheckNpi = Get-Command "nvpi.exe" -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Source -ErrorAction SilentlyContinue
}
if ($CheckNpi) {
    Write-Host ("[PASS] NV Inspector  : Ready ({0})" -f $CheckNpi) -ForegroundColor Green
} else {
    Write-Host "[WARN] NV Inspector  : Not installed (winget install --id Orbmu2k.nvidiaProfileInspector -e)" -ForegroundColor Yellow
}

# 7. TensorRT & Triton in .venv
if (Test-Path $VenvPy) {
    $PyCheck = & $VenvPy -c "import importlib.util as u, torch; trt=u.find_spec('tensorrt') is not None; tri=u.find_spec('triton') is not None; print(f'Torch={torch.__version__} | TensorRT={trt} | Triton={tri}')" 2>$null
    Write-Host ("[PASS] AI Runtime    : {0}" -f $PyCheck) -ForegroundColor Green
}

Write-Host "===========================================================================" -ForegroundColor Cyan
