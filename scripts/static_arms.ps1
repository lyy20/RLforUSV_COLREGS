$repo='D:\DSH_USV_Learning\USV_COLREGS'; Set-Location $repo; $env:PYTHONIOENCODING='utf-8'
$c='D:\ANACONDA\Scripts\conda.exe'
foreach ($cfg in 'COV_STATIC_600','COV_STATIC_400') {
  $o = & $c run --no-capture-output -n RLforUSV_L_T python main.py "$cfg.txt" 2>&1
  "$cfg : " + (($o | Select-String -Pattern '150/150' | Select-Object -Last 1)) | Tee-Object -FilePath (Join-Path $repo 'docs\static_arms.log') -Append
}
"STATIC DONE" | Tee-Object -FilePath (Join-Path $repo 'docs\static_arms.log') -Append