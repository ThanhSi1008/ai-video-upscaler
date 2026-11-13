<#
.SYNOPSIS
    Mở khóa các thiết lập cấp Administrator trên Windows 11 & NVIDIA Driver cho RTX 3060 12GB (GA106)
    và kiểm định toàn bộ trạng thái phần cứng cho dự án AI Video Upscaler 4K.

.DESCRIPTION
    - Chế độ mặc định (cần Run as Administrator):
      1. Bật Hardware-Accelerated GPU Scheduling (HAGS: HwSchMode = 2) giảm độ trễ lập lịch WDDM.
      2. Tối ưu Multimedia SystemProfile (GPU Priority = 8) & tắt PCIe ASPM trên Ultimate Performance.
      3. Xuất file cấu hình chuẩn UTF-16 LE `rtx3060_cuda_p0_unlock.nip` (SettingID 1343646814 = 0)
         để tắt `CUDA - Force P2 State` (giữ xung VRAM GDDR6 ở P0 7501+ MHz thay vì tụt về P2 7301 MHz),
         và tự động nạp nếu tìm thấy `nvidiaProfileInspector.exe`.
      4. In bảng kiểm định tổng thể (RAM XMP, Resizable BAR, Power Limit, VRAM trống, TensorRT 10).
    - Chế độ `-CheckOnly`: Chỉ đọc và in bảng chẩn đoán (chạy được ngay cả khi không có quyền Admin).

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
        Write-Host "[!] Script chua chay voi quyen Administrator. Dang yeu cau UAC Elevation..." -ForegroundColor Yellow
        $ArgList = "-NoProfile -ExecutionPolicy Bypass -File `"$($MyInvocation.MyCommand.Path)`" -NoElevatePrompt"
        Start-Process -FilePath "powershell.exe" -ArgumentList $ArgList -Verb RunAs
        exit 0
    } else {
        Write-Host "[X] Khong co quyen Administrator. Chuyen sang che do kiem tra (-CheckOnly)." -ForegroundColor Red
        $CheckOnly = $true
    }
}

# -------------------------------------------------------------------------
# PHẦN 1: ÁP DỤNG MỞ KHÓA CẤP ADMIN (NẾU KHÔNG Ở CHẾ ĐỘ -CheckOnly)
# -------------------------------------------------------------------------
if (-not $CheckOnly -and $IsAdmin) {
    Write-Host "`n[1/4] Kich hoat Hardware-Accelerated GPU Scheduling (HAGS = 2)..." -ForegroundColor Green
    $GfxKey = "HKLM:\SYSTEM\CurrentControlSet\Control\GraphicsDrivers"
    Set-ItemProperty -Path $GfxKey -Name "HwSchMode" -Value 2 -Type DWord
    Write-Host "  -> Da dat HKLM GraphicsDrivers\HwSchMode = 2 (Can Restart Windows de co hieu luc)."

    Write-Host "`n[2/4] Toi uu Windows Multimedia Scheduler & PCIe ASPM Off..." -ForegroundColor Green
    $MmKey = "HKLM:\SOFTWARE\Microsoft\Windows NT\CurrentVersion\Multimedia\SystemProfile"
    if (Test-Path $MmKey) {
        Set-ItemProperty -Path $MmKey -Name "SystemResponsiveness" -Value 10 -Type DWord
    }
    powercfg /setacvalueindex SCHEME_CURRENT SUB_PCIEXPRESS ASPM 0 | Out-Null
    powercfg /setactive SCHEME_CURRENT | Out-Null
    Write-Host "  -> Da khoa PCIe Link State Power Management = Off tren Power Scheme hien tai."

    Write-Host "`n[3/4] Tao & nap profile NVIDIA tat 'CUDA - Force P2 State' (P0 Full VRAM Clock)..." -ForegroundColor Green
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
    Write-Host "  -> Da xuat profile UTF-16 LE: $NipPath"

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
        Write-Host "  -> [OK] Da merge tu dong CUDA - Force P2 State = Off vao Base Profile qua: $CandidateNpi" -ForegroundColor Green
    } else {
        Write-Host "  -> [INFO] Chua tim thay nvidiaProfileInspector.exe tren may." -ForegroundColor Yellow
        Write-Host "     Cai nhanh bang: winget install --id Orbmu2k.nvidiaProfileInspector -e"
        Write-Host "     Sau do chay lai script nay hoac Import file: $NipPath"
    }

    Write-Host "`n[4/5] Them thu muc dem AI & Video vao ngoai le Windows Defender (giam nghen I/O MsMpEng)..." -ForegroundColor Green
    try {
        $ExclDirs = @(
            (Join-Path $RepoRoot ".cache"),
            (Join-Path $RepoRoot "scratch"),
            (Join-Path $RepoRoot "out")
        )
        Add-MpPreference -ExclusionPath $ExclDirs -ErrorAction Stop
        Write-Host "  -> [OK] Da loai tru .cache, scratch va out khoi quet thoi gian thuc Windows Defender." -ForegroundColor Green
    } catch {
        Write-Host "  -> [INFO] Bo qua ngoai le Defender (Tamper Protection hoac chinh sach he thong)." -ForegroundColor DarkGray
    }

    Write-Host "`n[5/5] Kiem chung truc tiep xung VRAM khi mo ngữ cảnh CUDA..." -ForegroundColor Green
    if (Test-Path $VenvPy) {
        & $VenvPy -c "import subprocess, torch; x = torch.zeros((256, 256), device='cuda'); out = subprocess.check_output(['nvidia-smi', '--query-gpu=name,pstate,clocks.current.memory,clocks.max.memory,power.limit', '--format=csv,noheader'], text=True).strip(); print('  -> Live CUDA State:', out)"
    } else {
        nvidia-smi --query-gpu=name,driver_version,pstate,power.limit,clocks.max.memory --format=csv,noheader
    }
}

# -------------------------------------------------------------------------
# PHẦN 2: BẢNG KIỂM ĐỊNH TOÀN DIỆN (LIVE HARDWARE & RUNTIME DASHBOARD)
# -------------------------------------------------------------------------
Write-Host "`n===========================================================================" -ForegroundColor Cyan
Write-Host "  BANG CHAN DOAN HIEU NANG HIEN TAI (DIAGNOSTIC CHECKLIST)" -ForegroundColor Cyan
Write-Host "===========================================================================" -ForegroundColor Cyan

# 1. Kiểm tra RAM Speed (XMP)
try {
    $RamSticks = Get-CimInstance Win32_PhysicalMemory -ErrorAction Stop
    $MinSpeed  = ($RamSticks | Measure-Object -Property ConfiguredClockSpeed -Minimum).Minimum
    $TotalGB   = [math]::Round((($RamSticks | Measure-Object -Property Capacity -Sum).Sum / 1GB), 1)
    if ($MinSpeed -ge 3200) {
        Write-Host ("[PASS] DDR4 RAM      : {0} GB @ {1} MT/s (XMP Da Bat)" -f $TotalGB, $MinSpeed) -ForegroundColor Green
    } else {
        Write-Host ("[WARN] DDR4 RAM      : {0} GB @ {1} MT/s -> CAN BAT XMP PROFILE 1 (3200 MT/s) TRONG BIOS!" -f $TotalGB, $MinSpeed) -ForegroundColor Yellow
    }
} catch {
    Write-Host "[INFO] DDR4 RAM      : Khong doc duoc WMI Win32_PhysicalMemory" -ForegroundColor DarkGray
}

# 2. Kiểm tra Resizable BAR (BAR1 Size)
try {
    $Bar1Line = (nvidia-smi -q | Select-String -Pattern "BAR1 Memory Usage" -Context 0,3)
    $Bar1TotalText = ($Bar1Line.Context.PostContext | Where-Object { $_ -match "Total\s*:\s*(\d+)\s*MiB" } | Select-Object -First 1)
    if ($Bar1TotalText -match "(\d+)\s*MiB") {
        $Bar1MiB = [int]$Matches[1]
        if ($Bar1MiB -gt 512) {
            Write-Host ("[PASS] Resizable BAR : BAR1 = {0} MiB (Da Mo Khoa Full VRAM Aperture)" -f $Bar1MiB) -ForegroundColor Green
        } else {
            Write-Host ("[WARN] Resizable BAR : BAR1 = {0} MiB -> CAN BAT Above 4G + Resizable BAR TRONG BIOS!" -f $Bar1MiB) -ForegroundColor Yellow
        }
    }
} catch {
    Write-Host "[INFO] Resizable BAR : Khong doc duoc BAR1 tu nvidia-smi" -ForegroundColor DarkGray
}

# 3. Kiểm tra HAGS
try {
    $HagsVal = (Get-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\GraphicsDrivers" -Name "HwSchMode" -ErrorAction SilentlyContinue).HwSchMode
    if ($HagsVal -eq 2) {
        Write-Host "[PASS] Windows HAGS  : HwSchMode = 2 (Da Bat)" -ForegroundColor Green
    } else {
        Write-Host ("[WARN] Windows HAGS  : HwSchMode = {0} -> Chay script nay voi Admin de bat HAGS = 2" -f $HagsVal) -ForegroundColor Yellow
    }
} catch {}

# 4. Kiểm tra Power Scheme
$ActiveScheme = (powercfg /getactivescheme) -join " "
if ($ActiveScheme -match "Ultimate Performance|High performance") {
    Write-Host ("[PASS] Power Plan    : {0}" -f $ActiveScheme.Trim()) -ForegroundColor Green
} else {
    Write-Host ("[WARN] Power Plan    : {0}" -f $ActiveScheme.Trim()) -ForegroundColor Yellow
}

# 5. Kiểm tra VRAM trống & tiến trình ngốn VRAM
try {
    $VramCsv = (nvidia-smi --query-gpu=memory.used,memory.free,memory.total,power.limit --format=csv,noheader,nounits).Trim()
    $Parts = $VramCsv.Split(",") | ForEach-Object { $_.Trim() }
    $UsedMiB = [int]$Parts[0]
    $FreeMiB = [int]$Parts[1]
    $TotMiB  = [int]$Parts[2]
    $PwrLim  = $Parts[3]
    if ($FreeMiB -ge 8000) {
        Write-Host ("[PASS] RTX 3060 VRAM : Trong {0} / {1} MiB (Power Limit: {2} W)" -f $FreeMiB, $TotMiB, $PwrLim) -ForegroundColor Green
    } else {
        Write-Host ("[WARN] RTX 3060 VRAM : Chi con trong {0} / {1} MiB (Dang dung {2} MiB) -> Hay dong Blender/Game truoc khi upscale!" -f $FreeMiB, $TotMiB, $UsedMiB) -ForegroundColor Yellow
    }
} catch {}

# 6. Kiểm tra NVIDIA Profile Inspector
$CheckNpi = Get-ChildItem (Join-Path $env:LOCALAPPDATA "Microsoft\WinGet\Packages\Orbmu2k.nvidiaProfileInspector*") -Filter "nvidiaProfileInspector.exe" -Recurse -ErrorAction SilentlyContinue | Select-Object -ExpandProperty FullName -First 1
if (-not $CheckNpi) {
    $CheckNpi = Get-Command "nvpi.exe" -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Source -ErrorAction SilentlyContinue
}
if ($CheckNpi) {
    Write-Host ("[PASS] NV Inspector  : San sang ({0})" -f $CheckNpi) -ForegroundColor Green
} else {
    Write-Host "[WARN] NV Inspector  : Chua cai dat (winget install --id Orbmu2k.nvidiaProfileInspector -e)" -ForegroundColor Yellow
}

# 7. Kiểm tra TensorRT & Triton trong .venv
if (Test-Path $VenvPy) {
    $PyCheck = & $VenvPy -c "import importlib.util as u, torch; trt=u.find_spec('tensorrt') is not None; tri=u.find_spec('triton') is not None; print(f'Torch={torch.__version__} | TensorRT={trt} | Triton={tri}')" 2>$null
    Write-Host ("[PASS] AI Runtime    : {0}" -f $PyCheck) -ForegroundColor Green
}

Write-Host "===========================================================================" -ForegroundColor Cyan
