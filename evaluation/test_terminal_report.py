"""统一的测试终端评估报告。

该模块只负责把测试结果转换成便于快速判断的终端文本，不参与环境推进、
动作选择或规则过滤。它同时兼容：

* ``see_trained_once.py`` / ``see_trained_policy_only.py`` 的回合 result；
* ``see_trained_custom.py`` 的 episode summary；
* 已有 ``colregs_event_log.csv`` 对应的事件字典。

规则内化的判断必须建立在策略原始动作对应的事件字段上。安全化解率只能
说明事件最终没有失败，不能单独证明策略遵守了 COLREGs。
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Iterable, Mapping, Optional, Sequence

import numpy as np


ENCOUNTER_TYPE_LABELS = OrderedDict(
    (
        ("head_on", "对遇"),
        ("crossing_give_way", "交叉会遇-让路船"),
        ("crossing_stand_on", "交叉会遇-直航船"),
        ("overtaking_give_way", "追越-让路船"),
        ("overtaken_stand_on", "被追越-直航船"),
        ("collision_risk_undefined", "碰撞风险-类型未定义"),
    )
)

_COLLISION_REASONS = {"collision", "target_collision"}
_OUT_OF_BOUNDS_REASONS = {"out_of_bounds", "out_of_world"}
_EPS = 1.0e-12


def _nested_value(row: Mapping, key: str, default=None):
    """读取顶层字段，并兼容 ``result['summary']`` 结构。"""
    if key in row:
        return row[key]
    summary = row.get("summary")
    if isinstance(summary, Mapping) and key in summary:
        return summary[key]
    return default


def _float(value, default=float("nan")) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if np.isfinite(number) else default


def _finite_mean(values: Iterable) -> float:
    data = np.asarray([_float(value) for value in values], dtype=float)
    data = data[np.isfinite(data)]
    return float(np.mean(data)) if data.size else float("nan")


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator > 0.0 else float("nan")


def _wilson_interval(successes, total, z=1.96):
    """Return a binomial Wilson interval for a displayed rate."""
    n = float(total)
    if n <= 0.0:
        return float("nan"), float("nan")
    p = float(successes) / n
    denominator = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / denominator
    margin = z * np.sqrt((p * (1.0 - p) + z * z / (4.0 * n)) / n) / denominator
    return max(0.0, centre - margin), min(1.0, centre + margin)


def _reason(row: Mapping) -> str:
    value = _nested_value(row, "done_reason", "none")
    return str(value or "none").strip().lower()


def _trajectory_minimum_clearance(result: Mapping) -> float:
    """从保存的轨迹回算 ownship 对其余实体的最小净间距。

    轨迹位置和半径均按项目约定使用 km；返回值为 m。此计算只作为一次性
    测试报告的后备值。若 episode summary 已提供环境计算的
    ``minimum_clearance_m``，调用方会优先使用那个值。
    """
    trajectory = result.get("trajectory")
    if not isinstance(trajectory, Mapping):
        return float("nan")
    agents = trajectory.get("agents") or []
    if not agents:
        return float("nan")
    ownship = agents[0]
    own_positions = np.asarray(ownship.get("positions", []), dtype=float)
    if own_positions.ndim != 2 or own_positions.shape[1] < 2:
        return float("nan")
    own_positions = own_positions[:, :2]
    own_radius = _float(ownship.get("size"), 0.0)
    other_tracks = []
    for group in ("targets", "dynamic_obstacles", "static_obstacles"):
        tracks = trajectory.get(group) or []
        if isinstance(tracks, Sequence):
            other_tracks.extend(tracks)
    clearances = []
    for track in other_tracks:
        positions = np.asarray(track.get("positions", []), dtype=float)
        if positions.ndim != 2 or positions.shape[1] < 2 or positions.shape[0] == 0:
            continue
        radius = _float(track.get("size"), 0.0)
        frame_count = min(own_positions.shape[0], positions.shape[0])
        if frame_count <= 0:
            continue
        distance_km = np.linalg.norm(
            own_positions[:frame_count] - positions[:frame_count, :2], axis=1
        )
        clearances.extend((distance_km - own_radius - radius) * 1000.0)
    return float(np.min(clearances)) if clearances else float("nan")


def _episode_rows(results: Sequence[Mapping]) -> list[dict]:
    rows = []
    for result in results or []:
        if not isinstance(result, Mapping):
            continue
        row = dict(result)
        reason = _reason(row)
        row["_reason"] = reason
        row["_success"] = int(
            bool(_float(_nested_value(row, "success", float("nan")), float("nan")) >= 0.5)
            if np.isfinite(_float(_nested_value(row, "success", float("nan"))))
            else reason == "success"
        )
        row["_collision"] = int(
            bool(_float(_nested_value(row, "collision", float("nan")), float("nan")) >= 0.5)
            if np.isfinite(_float(_nested_value(row, "collision", float("nan"))))
            else reason in _COLLISION_REASONS
        )
        row["_out_of_bounds"] = int(
            bool(_float(_nested_value(row, "out_of_bounds", float("nan")), float("nan")) >= 0.5)
            if np.isfinite(_float(_nested_value(row, "out_of_bounds", float("nan"))))
            else reason in _OUT_OF_BOUNDS_REASONS
        )
        row["_timeout"] = int(
            bool(_float(_nested_value(row, "timeout", float("nan")), float("nan")) >= 0.5)
            if np.isfinite(_float(_nested_value(row, "timeout", float("nan"))))
            else reason == "timeout"
        )
        minimum_clearance = _float(_nested_value(row, "minimum_clearance_m"))
        if not np.isfinite(minimum_clearance):
            minimum_clearance = _trajectory_minimum_clearance(row)
        row["_minimum_clearance_m"] = minimum_clearance
        rows.append(row)
    return rows


def _event_rows_for_variant(event_rows: Optional[Sequence[Mapping]], variant_name: str) -> list[dict]:
    selected = []
    for event in event_rows or []:
        if not isinstance(event, Mapping):
            continue
        event_variant = str(event.get("evaluation_variant", variant_name))
        if not variant_name or event_variant == variant_name:
            selected.append(dict(event))
    return selected


def _episode_audit_rows_for_variant(
    audit_episode_rows: Optional[Sequence[Mapping]], variant_name: str
) -> list[dict]:
    selected = []
    for row in audit_episode_rows or []:
        if not isinstance(row, Mapping):
            continue
        row_variant = str(row.get("evaluation_variant", variant_name))
        if not variant_name or row_variant == variant_name:
            selected.append(dict(row))
    return selected


def _audit_metrics(
    event_rows: Sequence[Mapping],
    audit_episode_rows: Sequence[Mapping],
) -> dict:
    """按规则事件统计审计指标，优先使用事件级字段。"""
    events = list(event_rows or [])
    episode_audit = list(audit_episode_rows or [])
    if events:
        total = len(events)
        followed = sum(int(_float(row.get("policy_rule_followed"), 0.0) >= 0.5) for row in events)
        safe = sum(int(_float(row.get("safe_resolution"), 0.0) >= 0.5) for row in events)
        compliant_safe = sum(
            int(
                _float(row.get("policy_rule_followed"), 0.0) >= 0.5
                and _float(row.get("safe_resolution"), 0.0) >= 0.5
            )
            for row in events
        )
        filter_events = sum(
            int(_float(row.get("filter_activated_in_event"), 0.0) >= 0.5)
            for row in events
        )
        minimum_clearance = _finite_mean(
            row.get("minimum_clearance_m") for row in events
        )
        risk_action_steps = sum(int(_float(row.get("risk_action_steps"), 0.0)) for row in events)
        rule_violation_steps = sum(int(_float(row.get("rule_violation_steps"), 0.0)) for row in events)
        anti_rule_steps = sum(int(_float(row.get("anti_rule_steps"), 0.0)) for row in events)
        rule_violation_events = sum(int(_float(row.get("rule_violation_event"), 0.0)) for row in events)
        stand_on_emergency_steps = sum(int(_float(row.get("stand_on_emergency_steps"), 0.0)) for row in events)
    else:
        total = sum(int(_float(row.get("encounter_events"), 0.0)) for row in episode_audit)
        followed = sum(int(_float(row.get("rule_followed_events"), 0.0)) for row in episode_audit)
        safe = sum(int(_float(row.get("successful_resolution_events"), 0.0)) for row in episode_audit)
        compliant_safe = sum(
            int(_float(row.get("compliant_and_safe_events"), 0.0))
            for row in episode_audit
        )
        filter_events = sum(
            int(_float(row.get("filter_activated_events"), 0.0))
            for row in episode_audit
        )
        minimum_clearance = float("nan")
        risk_action_steps = sum(int(_float(row.get("risk_action_steps"), 0.0)) for row in episode_audit)
        rule_violation_steps = sum(int(_float(row.get("rule_violation_steps"), 0.0)) for row in episode_audit)
        anti_rule_steps = sum(int(_float(row.get("anti_rule_steps"), 0.0)) for row in episode_audit)
        rule_violation_events = sum(int(_float(row.get("rule_violation_events"), 0.0)) for row in episode_audit)
        stand_on_emergency_steps = sum(int(_float(row.get("stand_on_emergency_steps"), 0.0)) for row in episode_audit)

    if episode_audit:
        episodes_with_encounter = int(
            sum(
                int(_float(row.get("encounter_events"), 0.0) > 0.0)
                for row in episode_audit
            )
        )
    else:
        # Direct callers may provide only an event log. Derive the count from
        # episode_id so the report remains useful outside the three scripts.
        episodes_with_encounter = len(
            {
                str(row.get("episode_id"))
                for row in events
                if row.get("episode_id") not in {None, ""}
            }
        )

    return {
        "risk_event_count": int(total),
        "episodes_with_encounter": episodes_with_encounter,
        "rule_followed_events": int(followed),
        "safe_resolution_events": int(safe),
        "compliant_and_safe_events": int(compliant_safe),
        "policy_rule_follow_rate": _safe_ratio(followed, total),
        "safe_resolution_rate": _safe_ratio(safe, total),
        "conditional_resolution_rate": _safe_ratio(compliant_safe, followed),
        "internalization_success_rate": _safe_ratio(compliant_safe, total),
        "filter_activated_events": int(filter_events),
        "filter_active_event_rate": _safe_ratio(filter_events, total),
        "minimum_clearance_from_events_m": minimum_clearance,
        "risk_action_steps": int(risk_action_steps),
        "rule_violation_steps": int(rule_violation_steps),
        "anti_rule_steps": int(anti_rule_steps),
        "rule_violation_step_rate": _safe_ratio(rule_violation_steps, risk_action_steps),
        "rule_violation_events": int(rule_violation_events),
        "stand_on_emergency_steps": int(stand_on_emergency_steps),
    }


def _rule_type_metrics(event_rows: Sequence[Mapping]) -> list[dict]:
    grouped = OrderedDict()
    for row in event_rows or []:
        encounter_type = str(row.get("encounter_type", "unknown"))
        grouped.setdefault(encounter_type, []).append(row)
    ordered_types = list(ENCOUNTER_TYPE_LABELS.keys()) + [
        key for key in grouped if key not in ENCOUNTER_TYPE_LABELS
    ]
    output = []
    for encounter_type in ordered_types:
        rows = grouped.get(encounter_type, [])
        if not rows:
            continue
        total = len(rows)
        followed = sum(
            int(_float(row.get("policy_rule_followed"), 0.0) >= 0.5)
            for row in rows
        )
        safe = sum(
            int(_float(row.get("safe_resolution"), 0.0) >= 0.5)
            for row in rows
        )
        compliant_safe = sum(
            int(
                _float(row.get("policy_rule_followed"), 0.0) >= 0.5
                and _float(row.get("safe_resolution"), 0.0) >= 0.5
            )
            for row in rows
        )
        risk_action_steps = sum(int(_float(row.get("risk_action_steps"), 0.0)) for row in rows)
        rule_violation_steps = sum(int(_float(row.get("rule_violation_steps"), 0.0)) for row in rows)
        output.append(
            {
                "encounter_type": encounter_type,
                "label": ENCOUNTER_TYPE_LABELS.get(encounter_type, encounter_type),
                "events": total,
                "policy_rule_follow_rate": _safe_ratio(followed, total),
                "safe_resolution_rate": _safe_ratio(safe, total),
                "internalization_success_rate": _safe_ratio(compliant_safe, total),
                "rule_violation_step_rate": _safe_ratio(
                    rule_violation_steps, risk_action_steps
                ),
            }
        )
    return output


def _pct(value) -> str:
    number = _float(value)
    return "N/A" if not np.isfinite(number) else "%.2f%%" % (100.0 * number)


def _number(value, digits: int = 3) -> str:
    number = _float(value)
    return "N/A" if not np.isfinite(number) else ("%%.%df" % digits) % number


def _judgement(metrics: Mapping) -> str:
    total = int(metrics.get("risk_event_count", 0))
    if total == 0:
        return "无法判断：本次测试没有形成有效 COLREGs 风险事件。"
    if total < 5:
        return "样本偏少：已有规则事件，但暂不建议据此下定论。"
    internalized = _float(metrics.get("internalization_success_rate"))
    collision_free = _float(metrics.get("collision_rate"), 1.0) <= _EPS
    if np.isfinite(internalized) and internalized >= 0.80 and collision_free:
        return "初步表现为规则内化较好：规则合规且安全通过比例达到 80% 以上。"
    if np.isfinite(internalized) and internalized >= 0.50:
        return "部分内化：策略已学到一部分规则动作，但仍有明显失配或安全失败。"
    return "尚未充分内化：规则合规且安全通过的事件比例偏低。"


def summarize_terminal_metrics(
    episode_results: Sequence[Mapping],
    event_rows: Optional[Sequence[Mapping]] = None,
    audit_episode_rows: Optional[Sequence[Mapping]] = None,
    variant_name: str = "",
) -> dict:
    """返回机器可读的单变体终端统计。"""
    rows = _episode_rows(episode_results)
    audit_events = _event_rows_for_variant(event_rows, variant_name)
    audit_episodes = _episode_audit_rows_for_variant(audit_episode_rows, variant_name)
    rewards = [_float(_nested_value(row, "reward_sum")) for row in rows]
    steps = [_float(_nested_value(row, "steps")) for row in rows]
    filter_rates = [_float(_nested_value(row, "filter_active_rate")) for row in rows]
    clearances = [row["_minimum_clearance_m"] for row in rows]
    task = {
        "variant": variant_name,
        "episodes": len(rows),
        "successes": sum(row["_success"] for row in rows),
        "collisions": sum(row["_collision"] for row in rows),
        "out_of_bounds": sum(row["_out_of_bounds"] for row in rows),
        "timeouts": sum(row["_timeout"] for row in rows),
        "success_rate": _safe_ratio(sum(row["_success"] for row in rows), len(rows)),
        "collision_rate": _safe_ratio(sum(row["_collision"] for row in rows), len(rows)),
        "out_of_bounds_rate": _safe_ratio(sum(row["_out_of_bounds"] for row in rows), len(rows)),
        "timeout_rate": _safe_ratio(sum(row["_timeout"] for row in rows), len(rows)),
        "mean_reward": _finite_mean(rewards),
        "reward_std": float(np.nanstd(np.asarray(rewards, dtype=float))) if any(np.isfinite(rewards)) else float("nan"),
        "mean_steps": _finite_mean(steps),
        "mean_filter_active_rate": _finite_mean(filter_rates),
        "mean_minimum_clearance_m": _finite_mean(clearances),
        "worst_minimum_clearance_m": (
            float(np.nanmin(np.asarray(clearances, dtype=float)))
            if any(np.isfinite(clearances))
            else float("nan")
        ),
    }
    audit = _audit_metrics(audit_events, audit_episodes)
    metrics = dict(task)
    metrics.update(audit)
    for rate_key, numerator_key, denominator_key in (
        ("success_rate", "successes", "episodes"),
        ("collision_rate", "collisions", "episodes"),
        ("out_of_bounds_rate", "out_of_bounds", "episodes"),
        ("timeout_rate", "timeouts", "episodes"),
        ("policy_rule_follow_rate", "rule_followed_events", "risk_event_count"),
        ("safe_resolution_rate", "safe_resolution_events", "risk_event_count"),
        ("internalization_success_rate", "compliant_and_safe_events", "risk_event_count"),
    ):
        low, high = _wilson_interval(metrics[numerator_key], metrics[denominator_key])
        metrics[rate_key + "_ci95_low"] = low
        metrics[rate_key + "_ci95_high"] = high
    metrics["encounter_type_metrics"] = _rule_type_metrics(audit_events)
    metrics["judgement"] = _judgement(metrics)
    return metrics


def _print_rate_line(key: str, label: str, value, numerator=None, denominator=None):
    suffix = ""
    if numerator is not None and denominator is not None:
        suffix = " (%s/%s)" % (numerator, denominator)
    display_key = key if not label else key + " / " + label
    print("  %-48s %-10s%s" % (display_key, _pct(value), suffix))


def print_terminal_report(
    episode_results: Sequence[Mapping],
    event_rows: Optional[Sequence[Mapping]] = None,
    audit_episode_rows: Optional[Sequence[Mapping]] = None,
    variant_name: str = "",
    title: str = "测试终端报告 / Test terminal report",
) -> dict:
    """打印一份适合快速判断的中英文终端报告，并返回统计字典。"""
    metrics = summarize_terminal_metrics(
        episode_results,
        event_rows=event_rows,
        audit_episode_rows=audit_episode_rows,
        variant_name=variant_name,
    )
    display_name = variant_name or "default"
    print("\n" + "=" * 78)
    print("%s | variant=%s" % (title, display_name))
    print("任务效果 / Task outcome")
    print("  episodes / 回合数                 %d" % metrics["episodes"])
    _print_rate_line("success_rate / 成功率", "", metrics["success_rate"], metrics["successes"], metrics["episodes"])
    _print_rate_line("collision_rate / 碰撞率", "", metrics["collision_rate"], metrics["collisions"], metrics["episodes"])
    _print_rate_line("out_of_bounds_rate / 出界率", "", metrics["out_of_bounds_rate"], metrics["out_of_bounds"], metrics["episodes"])
    _print_rate_line("timeout_rate / 超时率", "", metrics["timeout_rate"], metrics["timeouts"], metrics["episodes"])
    print("  mean_reward / 平均回合奖励        %s" % _number(metrics["mean_reward"], 4))
    print("  reward_std / 奖励标准差           %s" % _number(metrics["reward_std"], 4))
    print("  mean_steps / 平均步数             %s" % _number(metrics["mean_steps"], 2))
    print("  mean_min_clearance_m / 平均最小净距 %s m" % _number(metrics["mean_minimum_clearance_m"], 3))
    print("  worst_min_clearance_m / 最差最小净距 %s m" % _number(metrics["worst_minimum_clearance_m"], 3))

    print("规则内化与安全 / Rule internalization and safety")
    print("  risk_event_count / 规则风险事件数  %d" % metrics["risk_event_count"])
    print("  episodes_with_encounter / 有会遇回合 %d" % metrics["episodes_with_encounter"])
    _print_rate_line(
        "policy_rule_follow_rate / 策略规则跟随率",
        "",
        metrics["policy_rule_follow_rate"],
        metrics["rule_followed_events"],
        metrics["risk_event_count"],
    )
    _print_rate_line(
        "safe_resolution_rate / 安全化解率",
        "",
        metrics["safe_resolution_rate"],
        metrics["safe_resolution_events"],
        metrics["risk_event_count"],
    )
    _print_rate_line(
        "internalization_success_rate / 规则内化成功率",
        "",
        metrics["internalization_success_rate"],
        metrics["compliant_and_safe_events"],
        metrics["risk_event_count"],
    )
    _print_rate_line(
        "conditional_resolution_rate / 合规事件安全率",
        "",
        metrics["conditional_resolution_rate"],
        metrics["compliant_and_safe_events"],
        metrics["rule_followed_events"],
    )
    print("  rule_violation_step_rate / 规则违反步比例 %s (%d/%d)" % (
        _pct(metrics["rule_violation_step_rate"]),
        metrics["rule_violation_steps"],
        metrics["risk_action_steps"],
    ))
    print("  rule_violation_events / 未完成合规会遇数 %d" % metrics["rule_violation_events"])
    print("  stand_on_emergency_steps / 直航紧急动作步数 %d" % metrics["stand_on_emergency_steps"])
    print("  internalization_success_ci95 / 内化成功率95%%区间 [%s, %s]" % (
        _pct(metrics["internalization_success_rate_ci95_low"]),
        _pct(metrics["internalization_success_rate_ci95_high"]),
    ))
    _print_rate_line(
        "filter_active_rate / 过滤器步介入率",
        "",
        metrics["mean_filter_active_rate"],
    )
    _print_rate_line(
        "filter_active_event_rate / 过滤器事件介入率",
        "",
        metrics["filter_active_event_rate"],
        metrics["filter_activated_events"],
        metrics["risk_event_count"],
    )
    print("  judgement / 快速判断             %s" % metrics["judgement"])
    if metrics["risk_event_count"] == 0:
        print("  note / 说明                      规则事件为 0 时不能据此证明已内化。")

    type_metrics = metrics["encounter_type_metrics"]
    if type_metrics:
        print("各类会遇规则 / Encounter-type rule compliance")
        for item in type_metrics:
            print(
                "  %-24s events=%-4d follow=%-9s safe=%-9s internalized=%s"
                % (
                    item["label"],
                    item["events"],
                    _pct(item["policy_rule_follow_rate"]),
                    _pct(item["safe_resolution_rate"]),
                    _pct(item["internalization_success_rate"]),
                )
            )
    print("=" * 78)
    return metrics


def print_multi_variant_terminal_report(
    episode_results_by_variant: Mapping[str, Sequence[Mapping]],
    event_rows: Optional[Sequence[Mapping]] = None,
    audit_episode_rows: Optional[Sequence[Mapping]] = None,
    title: str = "测试终端报告 / Test terminal report",
) -> dict:
    """按变体依次打印报告；返回 ``{variant: metrics}``。"""
    output = OrderedDict()
    for variant_name, rows in episode_results_by_variant.items():
        output[variant_name] = print_terminal_report(
            rows,
            event_rows=event_rows,
            audit_episode_rows=audit_episode_rows,
            variant_name=variant_name,
            title=title,
        )
    return output
