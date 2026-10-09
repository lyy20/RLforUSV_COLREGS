# 串行实验流水线：同一时刻只跑 1 个进程，逐阶段落日志
$ErrorActionPreference = 'Continue'
$repo = 'D:\DSH_USV_Learning\USV_COLREGS'
Set-Location $repo
$ts = Get-Date -Format 'yyyyMMdd_HHmmss'
$log = Join-Path $repo "docs\pipeline_$ts.log"
$py = 'D:\ANACONDA\Scripts\conda.exe'
$env:PYTHONIOENCODING = 'utf-8'
function Stage([string]$name, [string[]]$cmdArgs) {
  $stamp = (Get-Date -Format 'HH:mm:ss')
  "[$stamp] STAGE START: $name" | Tee-Object -FilePath $log -Append
  $out = & $py run --no-capture-output -n RLforUSV_L_T python @cmdArgs 2>&1
  $code = $LASTEXITCODE
  $out | Out-File -FilePath $log -Append -Encoding utf8
  $stamp = (Get-Date -Format 'HH:mm:ss')
  "[$stamp] STAGE END: $name exit=$code" | Tee-Object -FilePath $log -Append
  if ($code -ne 0) { "[$stamp] ABORT: stage failed" | Tee-Object -FilePath $log -Append; exit 1 }
}
Stage '1_train_C_stage1050' @('main.py','SAC_3DOF_ENTITY_ENCODER_ARM_C_IL_S1050.txt')
Stage '2_eval_A_80'  @('see_trained_custom.py','curriculum\A0_ATTRIB_ARM_A.txt','--variants','sac','--episodes','80','--no-gif')
Stage '3_eval_B_80'  @('see_trained_custom.py','curriculum\A0_ATTRIB_ARM_B.txt','--variants','sac','--episodes','80','--no-gif')
Stage '4_eval_C1050_80' @('see_trained_custom.py','curriculum\A0_ATTRIB_ARM_C1050.txt','--variants','sac','--episodes','80','--no-gif')
Stage '5_train_D_w020' @('main.py','SAC_3DOF_ENTITY_ENCODER_ARM_D_W20.txt')
Stage '6_eval_D_80'  @('see_trained_custom.py','curriculum\A0_ATTRIB_ARM_D.txt','--variants','sac','--episodes','80','--no-gif')
$stamp = (Get-Date -Format 'HH:mm:ss')
"[$stamp] PIPELINE DONE" | Tee-Object -FilePath $log -Append