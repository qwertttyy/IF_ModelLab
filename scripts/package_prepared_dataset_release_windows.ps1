param(
  [string]$DatasetRoot = 'D:\FinalProject\IronFlow\datawork\tank_armor_prepared_v20260630',
  [string]$ReleaseDir = 'D:\FinalProject\IronFlow\datarelease\tank_armor_prepared_v20260630',
  [string]$DatasetId = 'tank_armor_prepared_v20260630',
  [Int64]$PartSizeBytes = 4294967296
)
$ErrorActionPreference = 'Stop'
if (-not (Test-Path -LiteralPath $DatasetRoot -PathType Container)) { throw "dataset root does not exist: $DatasetRoot" }
New-Item -ItemType Directory -Force -Path $ReleaseDir | Out-Null
Get-ChildItem -LiteralPath $ReleaseDir -File -Filter 'part-*.bin' -ErrorAction SilentlyContinue | Remove-Item -Force
$tarName = "ironflow_dataset_$DatasetId.tar"
$tarPath = Join-Path $ReleaseDir $tarName
if (-not (Test-Path -LiteralPath $tarPath)) {
  $parent = Split-Path -Parent $DatasetRoot
  $base = Split-Path -Leaf $DatasetRoot
  & tar.exe -cf $tarPath -C $parent $base
  if ($LASTEXITCODE -ne 0) { throw "tar.exe failed with exit code $LASTEXITCODE" }
}
$buffer = New-Object byte[] (16MB)
$inputStream = [System.IO.File]::OpenRead($tarPath)
try {
  $partIndex = 0
  while ($inputStream.Position -lt $inputStream.Length) {
    $partPath = Join-Path $ReleaseDir ('part-{0:D4}.bin' -f $partIndex)
    $out = [System.IO.File]::Create($partPath)
    try {
      [Int64]$remaining = $PartSizeBytes
      while ($remaining -gt 0 -and $inputStream.Position -lt $inputStream.Length) {
        $toRead = [int]([Math]::Min([Int64]$buffer.Length, [Int64]$remaining))
        $read = $inputStream.Read($buffer, 0, $toRead)
        if ($read -le 0) { break }
        $out.Write($buffer, 0, $read)
        $remaining -= $read
      }
    } finally { $out.Dispose() }
    $partIndex++
  }
} finally { $inputStream.Dispose() }
$partFiles = Get-ChildItem -LiteralPath $ReleaseDir -File -Filter 'part-*.bin' | Sort-Object Name
$totalBytes = ($partFiles | Measure-Object -Property Length -Sum).Sum
@"
cd /workspace/ironflow/drive_dataset_parts/$DatasetId
cat part-*.bin > $tarName
mkdir -p /workspace/ironflow/prestaged
tar -xf $tarName -C /workspace/ironflow/prestaged
ls /workspace/ironflow/prestaged/$DatasetId
"@ | Set-Content -LiteralPath (Join-Path $ReleaseDir 'restore_commands.txt') -Encoding UTF8
@{
  schema_version = '0.1'
  dataset_id = $DatasetId
  tar_name = $tarName
  part_size_bytes = $PartSizeBytes
  part_count = $partFiles.Count
  total_bytes = [Int64]$totalBytes
  expected_remote_root = "/workspace/ironflow/prestaged/$DatasetId"
} | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $ReleaseDir 'manifest.json') -Encoding UTF8
Remove-Item -Force -LiteralPath $tarPath
"part_count=$($partFiles.Count)"
"total_bytes=$([Int64]$totalBytes)"
"restore_commands=$(Join-Path $ReleaseDir 'restore_commands.txt')"
