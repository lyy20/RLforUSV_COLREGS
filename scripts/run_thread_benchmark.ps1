$repo = 'D:\DSH_USV_Learning\USV_COLREGS'; Set-Location $repo
$env:PYTHONIOENCODING = 'utf-8'
$log = Join-Path $repo 'docs\bench_threads.log'; Remove-Item $log -ErrorAction SilentlyContinue
foreach ($n in 12,16,24,32,36) {
  "=== BENCH parallel_envs=$n ===" | Tee-Object -FilePath $log -Append
  $o = & 'D:\ANACONDA\Scripts\conda.exe' run --no-capture-output -n RLforUSV_L_T python main.py "BENCH_THREADS_$n.txt" 2>&1
  ($o | Select-String -Pattern 'it/s|s/it' | Select-Object -Last 2) | Tee-Object -FilePath $log -Append
  Remove-Item -Recurse -Force "D:\USV\logs\BENCH_THREADS_$n" -ErrorAction SilentlyContinue
}
"BENCH DONE" | Tee-Object -FilePath $log -Append