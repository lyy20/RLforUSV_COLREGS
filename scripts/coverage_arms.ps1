$repo='D:\DSH_USV_Learning\USV_COLREGS'; Set-Location $repo; $env:PYTHONIOENCODING='utf-8'
$c='D:\ANACONDA\Scripts\conda.exe'
foreach ($cfg in 'COV_PLAIN','COV_CPA','COV_CPA_FILTER') {
  $o = & $c run --no-capture-output -n RLforUSV_L_T python main.py "$cfg.txt" 2>&1
  $l = ($o | Select-String -Pattern '150/150' | Select-Object -Last 1)
  "$cfg : $l" | Tee-Object -FilePath (Join-Path $repo 'docs\coverage_arms.log') -Append
}
"COV DONE" | Tee-Object -FilePath (Join-Path $repo 'docs\coverage_arms.log') -Append