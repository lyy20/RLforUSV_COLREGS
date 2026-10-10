$repo='D:\DSH_USV_Learning\USV_COLREGS'; Set-Location $repo; $env:PYTHONIOENCODING='utf-8'
$c='D:\ANACONDA\Scripts\conda.exe'
$log=Join-Path $repo 'docs\cost_attribution.log'; Remove-Item $log -ErrorAction SilentlyContinue
foreach ($round in 1..2) {
  foreach ($cfg in 'COST_Aoff_Boff','COST_Aon_Boff','COST_Aon_Bon') {
    $o = & $c run --no-capture-output -n RLforUSV_L_T python main.py "$cfg.txt" 2>&1
    $l = ($o | Select-String -Pattern '60/60' | Select-Object -Last 1)
    "r$round $cfg : $l" | Tee-Object -FilePath $log -Append
    Remove-Item -Recurse -Force "D:\USV\logs\$cfg" -ErrorAction SilentlyContinue
  }
}
"COST DONE" | Tee-Object -FilePath $log -Append