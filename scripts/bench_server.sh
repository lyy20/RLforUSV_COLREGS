#!/usr/bin/env bash
# 服务器标定：测 1/2/3 进程 × 不同 workers 的真实吞吐与显存
# 用法: bash scripts/bench_server.sh [base_config] [episodes] [peak_workers]
set -euo pipefail

BASE="${1:-V6_A2_S1.txt}"
EPS="${2:-20}"
PEAK="${3:-16}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export USV_LOG_ROOT="${USV_LOG_ROOT:-/root/USV/logs}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

mkdir -p "$USV_LOG_ROOT/bench"
STAMP="$(date +%m%d_%H%M)"
OUT="$USV_LOG_ROOT/bench/bench_$STAMP.csv"
echo "procs,workers,episodes,wall_s,s_per_episode,peak_vram_mib" > "$OUT"

peak_vram() {
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null \
    | sort -n | tail -1 || echo 0
}

run_one() {  # procs workers
  local procs="$1" workers="$2"
  local cfg="BENCH_${procs}x${workers}.txt"
  sed -e "s/^number_of_episodes.*/number_of_episodes = $EPS/" \
      -e "s/^parallel_envs.*/parallel_envs = $workers/" \
      -e "s/^POLICY_SAVE_INTERVAL.*/POLICY_SAVE_INTERVAL = 999999/" \
      -e "s/^TENSORBOARD_LOG_INTERVAL.*/TENSORBOARD_LOG_INTERVAL = 999999/" \
      -e "s/^TRAINING_RUN_NAME.*/TRAINING_RUN_NAME = BENCH_${procs}x${workers}/" \
      "$BASE" > "$cfg"

  echo "==> 标定 $procs 进程 × $workers workers × $EPS 回合"
  local t0=$SECONDS
  for i in $(seq 1 "$procs"); do
    python main.py "$cfg" >"$USV_LOG_ROOT/bench/${procs}x${workers}_p${i}.log" 2>&1 &
    sleep 20   # 错峰启动，避免同时初始化 CUDA
  done
  wait
  local wall=$(( SECONDS - t0 ))
  # 墙钟含错峰启动的 (procs-1)*20s，抵掉它更接近稳态
  local eff=$(( wall - (procs - 1) * 20 ))
  [ "$eff" -lt 1 ] && eff=1
  awk -v a="$procs" -v b="$workers" -v c="$EPS" -v d="$eff" -v e="$(peak_vram)" \
    'BEGIN{printf "%.6f\n", d/(a*c)}' > /tmp/spe.$$
  echo "$procs,$workers,$((" $procs * $EPS ")),$eff,$(cat /tmp/spe.$$),$(peak_vram)" >> "$OUT"
  rm -f /tmp/spe.$$ "$cfg"
  sleep 10   # 等显存释放
}

for w in 8 12 "$PEAK"; do
  run_one 1 "$w"
done
run_one 2 "$PEAK"
run_one 3 "$PEAK"

echo
echo "==> 结果（s_per_episode 为单进程单回合秒数，"总吞吐 = procs / s_per_episode"）"
cat "$OUT"
echo "==> 模型核对：总需求 4.2M 回合 → 天数 = 4.2e6 * s_per_episode / procs / 86400"
