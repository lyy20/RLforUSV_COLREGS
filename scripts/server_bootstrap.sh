#!/usr/bin/env bash
# 服务器一键部署 + 冒烟 + 环境快照
# 用法: bash scripts/server_bootstrap.sh [配置名]
# 例:   bash scripts/server_bootstrap.sh V6_A2_S1.txt
set -euo pipefail

CFG="${1:-V6_A2_S1.txt}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# ---- 1. 省钱的硬开关（见 docs/plan/服务器选型与Linux迁移.md §6.1）----
export USV_LOG_ROOT="${USV_LOG_ROOT:-/root/USV/logs}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
mkdir -p "$USV_LOG_ROOT"

echo "==> USV_LOG_ROOT=$USV_LOG_ROOT"
echo "==> OMP/MKL/OPENBLAS 线程数 = 1（防止 worker 进程互相抢核）"
nvidia-smi || echo "!! nvidia-smi 不可用，GPU 可能未挂载"

# ---- 2. 系统依赖（无头渲染兜底；失败不致命）----
if command -v apt-get >/dev/null 2>&1; then
  apt-get update -qq && apt-get install -y -qq libgl1 libglib2.0-0 || \
    echo "!! apt 安装失败（无 root 可忽略，训练本身不需要渲染）"
fi

# ---- 3. Python 依赖 ----
python -m pip install -q --upgrade pip
python -m pip install -q -r requirements-linux.txt

# ---- 4. 环境快照（可复现凭据）----
python scripts/dump_env.py | tee "$USV_LOG_ROOT/env_snapshot.txt"
python -m pip freeze > "$USV_LOG_ROOT/requirements-lock.txt"
echo "==> 环境快照与锁文件已写入 $USV_LOG_ROOT"

# ---- 5. 冒烟：30 回合（临时配置，不动冻结配置）----
SMOKE="SMOKE_${CFG}"
sed -e 's/^number_of_episodes.*/number_of_episodes = 30/' \
    -e 's/^POLICY_SAVE_INTERVAL.*/POLICY_SAVE_INTERVAL = 1000/' \
    -e 's/^TENSORBOARD_LOG_INTERVAL.*/TENSORBOARD_LOG_INTERVAL = 10/' \
    "$CFG" > "$SMOKE"
echo "==> 冒烟配置 $SMOKE（30 回合）"
/usr/bin/time -v python main.py "$SMOKE" 2>&1 | tail -n 60

echo "==> 冒烟结束。检查：无 CUDA 报错 / 无 import 报错 / 回合数正常推进"
