
# -*- coding: utf-8 -*-
"""B 模块原型：覆盖感知的主动规则教师（查询条件 + 置信度加权）。

设计动机：实测训练中 rule_filter_active_ratio 均值仅 3.62%——教师极少发声。
因此不再等“风险判据触发”，而是在“**义务即将生效**”的状态主动向规则教师查询。

公式（与论文/文档一致）：

    1) 会遇几何（相对运动，恒速外推）
       p_rel = p_o - p_s ,  v_rel = v_o - v_s
       TCPA_hat = -(p_rel · v_rel) / (v_rel · v_rel)
       DCPA_hat = || p_rel + v_rel * TCPA_hat ||

    2) 查询判据（closing 为真时）
       closing  = (p_rel · v_rel) < 0
       query    = closing and ( DCPA_hat < k_d * d_safe  or  TCPA_hat < k_t * T_react )

    3) 教师置信度（余量越小越可信）
       conf = clip( 1 - DCPA_hat / (k_d * d_safe), 0, 1 ) * valid

    4) 加权模仿损失
       L_rule = sum_i conf_i * || pi(s_i) - a_teacher(s_i) ||^2 / sum_i conf_i
"""
from __future__ import annotations
import numpy as np

K_DEFAULT = 3.0     # DCPA 余量倍数（乘以安全净距）
T_DEFAULT = 2.0     # TCPA 余量倍数（乘以反应窗）


def encounter_geometry(p_self, v_self, p_other, v_other):
    """返回 (closing, tcpa_hat, dcpa_hat)；closing=False 时 DCPA 无意义。"""
    p_rel = np.asarray(p_other, dtype=float) - np.asarray(p_self, dtype=float)
    v_rel = np.asarray(v_other, dtype=float) - np.asarray(v_self, dtype=float)
    vv = float(np.dot(v_rel, v_rel))
    if vv <= 1.0e-12:
        return False, float("inf"), float(np.linalg.norm(p_rel))
    tcpa = -float(np.dot(p_rel, v_rel)) / vv
    if tcpa <= 0.0:
        return False, float("inf"), float(np.linalg.norm(p_rel))
    dcpa = float(np.linalg.norm(p_rel + v_rel * tcpa))
    return True, tcpa, dcpa


def should_query(dcpa_hat, tcpa_hat, closing, d_safe, t_react, k_d=K_DEFAULT, k_t=T_DEFAULT):
    """义务即将生效 → 主动查询教师。"""
    if not closing:
        return False
    return bool(dcpa_hat < k_d * d_safe or tcpa_hat < k_t * t_react)


def query_confidence(dcpa_hat, d_safe, closing, k_d=K_DEFAULT):
    """教师置信度（余量越小越高）。"""
    if not closing:
        return 0.0
    denom = max(k_d * float(d_safe), 1.0e-9)
    return float(np.clip(1.0 - float(dcpa_hat) / denom, 0.0, 1.0))


def weighted_rule_loss(distances, confidences, eps=1.0e-8):
    """加权模仿损失。distances/confidences 为等长序列。"""
    d = np.asarray(distances, dtype=float)
    c = np.asarray(confidences, dtype=float)
    if d.size == 0:
        return 0.0
    return float(np.sum(c * d * d) / max(float(np.sum(c)), eps))


if __name__ == "__main__":
    # 自检 1：正前方对向驶来 → 必然查询，且置信度应较高
    geom = encounter_geometry([0.0, 0.0], [0.0014, 0.0], [0.180, 0.010], [-0.0014, 0.0])
    closing, tcpa, dcpa = geom
    q = should_query(dcpa, tcpa, closing, d_safe=0.010, t_react=5.0)
    conf = query_confidence(dcpa, d_safe=0.010, closing=closing)
    print("head-on: closing=%s tcpa=%.2f s dcpa=%.2f m query=%s conf=%.3f"
          % (closing, tcpa, dcpa * 1000.0, q, conf))
    assert closing and q and conf > 0.5

    # 自检 2：同向同速（无会遇）→ 不查询
    g2 = encounter_geometry([0.0, 0.0], [0.0014, 0.0], [0.060, 0.005], [0.0014, 0.0])
    q2 = should_query(g2[2], g2[1], g2[0], 0.010, 5.0)
    print("same-course: closing=%s query=%s" % (g2[0], q2))
    assert not q2

    # 自检 3：加权损失——置信度高的样本应主导
    loss = weighted_rule_loss([1.0, 0.1], [0.1, 0.9])
    print("weighted_rule_loss = %.4f" % loss)
    assert abs(loss - (0.1 * 1.0 + 0.9 * 0.01) / 1.0) < 1e-9
    print("teacher_query_selfcheck = PASS")
