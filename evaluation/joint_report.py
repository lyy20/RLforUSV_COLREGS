
# -*- coding: utf-8 -*-
"""M13 判据接口化：六项联合报告（P0-4）。

输出（缺数据时显示 N/A，不伪造）：
  1) 任务成功率 / 超时率
  2) 零事件回合占比（会遇事件数 == 0 的回合）——抗博弈关键项
  3) 过滤器步介入率（是否依赖安全层）
  4) 内化成功率（无干预下合规且安全化解）
  5) A0 窗口合规比（撤除安全层后的动作合规程度）
  6) A0 反向动作比（明确朝错误方向动作的比例）
"""
from __future__ import annotations
import csv
import sys
from pathlib import Path


def _read_csv(path: Path):
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def _f(row, key):
    try:
        return float(row.get(key, "") or "nan")
    except (TypeError, ValueError):
        return float("nan")


def _mean(rows, key):
    vals = [_f(r, key) for r in rows]
    vals = [v for v in vals if v == v]
    return sum(vals) / len(vals) if vals else float("nan")


def _pct(x):
    return "N/A" if x != x else "%.2f%%" % (100.0 * x)


def build_joint_report(out_dir, encounter="head_on"):  # noqa: C901
    out = Path(out_dir)
    episode_rows = _read_csv(out / "episode_summary.csv")
    variant_rows = _read_csv(out / "variant_summary.csv")
    audit_rows = _read_csv(out / "colregs_episode_summary.csv")

    success = _mean(variant_rows, "success_rate")
    if success != success:
        success = _mean(episode_rows, "success")
    timeout = _mean(variant_rows, "timeout_rate")
    if timeout != timeout:
        timeout = _mean(episode_rows, "timeout")
    filter_rate = _mean(variant_rows, "mean_filter_active_rate")
    if filter_rate != filter_rate:
        filter_rate = _mean(episode_rows, "filter_active_rate")
    internal = _mean(audit_rows, "internalization_success_rate")
    if internal != internal:
        internal = _mean(episode_rows, "internalization_success_rate")

    zero_ratio = float("nan")
    ev_vals = [_f(r, "encounter_events") for r in audit_rows]
    ev_vals = [v for v in ev_vals if v == v]
    if ev_vals:
        zero_ratio = sum(1 for v in ev_vals if v <= 0.0) / float(len(ev_vals))

    a0_ratio = float("nan"); a0_anti = float("nan"); a0_first = "N/A"
    step_log = out / "colregs_step_log.csv"
    if step_log.exists():
        try:
            sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
            import analyze_designed_encounter as ade
            rows = ade.load(step_log)
            peer = ade.pick_peer(rows)[0]
            summary, _ = ade.analyse(rows, peer, encounter)
            a0_ratio = float(summary.get("designed_compliance_ratio", float("nan")))
            a0_anti = float(summary.get("anti_rule_ratio", float("nan")))
            a0_first = summary.get("designed_first_compliance_step")
        except Exception as exc:  # 分析失败不影响主报告
            a0_first = "ERR(%s)" % type(exc).__name__

    print("== M13 六项联合报告 / joint internalization report ==")
    print("  %-34s %s" % ("1. 任务成功率", _pct(success)))
    print("  %-34s %s" % ("1b. 超时率", _pct(timeout)))
    print("  %-34s %s" % ("2. 零事件回合占比", _pct(zero_ratio)))
    print("  %-34s %s" % ("3. 过滤器步介入率", _pct(filter_rate)))
    print("  %-34s %s" % ("4. 内化成功率", _pct(internal)))
    print("  %-34s %s" % ("5. A0 窗口合规比", "N/A" if a0_ratio != a0_ratio else "%.3f" % a0_ratio))
    print("  %-34s %s" % ("6. A0 反向动作比", "N/A" if a0_anti != a0_anti else "%.3f" % a0_anti))
    print("  %-34s %s" % ("   首次合规步", a0_first))
    return {
        "success_rate": success, "timeout_rate": timeout, "zero_event_ratio": zero_ratio,
        "filter_active_rate": filter_rate, "internalization_rate": internal,
        "a0_compliance_ratio": a0_ratio, "a0_anti_rule_ratio": a0_anti,
        "first_compliance_step": a0_first,
    }


if __name__ == "__main__":
    build_joint_report(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "head_on")
