from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "paper_figures"


def _setup_fonts() -> None:
    candidates = [
        Path(r"C:\Windows\Fonts\NotoSansSC-VF.ttf"),
        Path(r"C:\Windows\Fonts\simhei.ttf"),
        Path(r"C:\Windows\Fonts\msyh.ttc"),
    ]
    for font_path in candidates:
        if font_path.exists():
            fm.fontManager.addfont(str(font_path))
            font_name = fm.FontProperties(fname=str(font_path)).get_name()
            plt.rcParams["font.sans-serif"] = [font_name, "DejaVu Sans"]
            break
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["mathtext.fontset"] = "dejavusans"


COLORS = {
    "ink": "#1F2937",
    "muted": "#5B6472",
    "line": "#667085",
    "scenario": "#E8F2FF",
    "scenario_edge": "#2F6B9A",
    "sac": "#EAF7F3",
    "sac_edge": "#207567",
    "filter": "#FFF4E5",
    "filter_edge": "#B66321",
    "dyn": "#F2ECFA",
    "dyn_edge": "#6B4C9A",
    "buffer": "#F3F5F7",
    "buffer_edge": "#64748B",
    "loss": "#FDECEC",
    "loss_edge": "#B64545",
    "accent": "#D95F02",
    "blue": "#1F77B4",
    "green": "#2A9D8F",
    "purple": "#6B4C9A",
}


def box(ax, xy, width, height, title, body, face, edge, title_color=None, fontsize=10.8):
    x, y = xy
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.018,rounding_size=0.028",
        linewidth=1.55,
        edgecolor=edge,
        facecolor=face,
    )
    ax.add_patch(patch)
    ax.text(
        x + 0.035,
        y + height - 0.045,
        title,
        ha="left",
        va="top",
        fontsize=13,
        fontweight="bold",
        color=title_color or edge,
    )
    ax.text(
        x + 0.035,
        y + height - 0.115,
        body,
        ha="left",
        va="top",
        fontsize=fontsize,
        color=COLORS["ink"],
        linespacing=1.28,
    )
    return patch


def pill(ax, x, y, text, color, width=0.30):
    patch = FancyBboxPatch(
        (x, y),
        width,
        0.052,
        boxstyle="round,pad=0.012,rounding_size=0.026",
        linewidth=1.0,
        edgecolor=color,
        facecolor="white",
    )
    ax.add_patch(patch)
    ax.text(x + width / 2, y + 0.026, text, ha="center", va="center", fontsize=9.2, color=color)


def arrow(ax, start, end, text=None, rad=0.0, color=None, lw=1.7, ms=14, text_offset=(0, 0)):
    arr = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=ms,
        linewidth=lw,
        color=color or COLORS["line"],
        connectionstyle=f"arc3,rad={rad}",
        shrinkA=4,
        shrinkB=4,
    )
    ax.add_patch(arr)
    if text:
        mx = (start[0] + end[0]) / 2 + text_offset[0]
        my = (start[1] + end[1]) / 2 + text_offset[1]
        ax.text(
            mx,
            my,
            text,
            ha="center",
            va="center",
            fontsize=9.2,
            color=COLORS["muted"],
            bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="none", alpha=0.92),
        )


def mini_ship(ax, x, y, color, label):
    ax.plot([x - 0.035, x + 0.030], [y, y], color=color, lw=1.3, alpha=0.9)
    ax.scatter([x], [y], s=110, color=color, edgecolor="white", linewidth=1.1, zorder=5)
    ax.text(x + 0.055, y, label, va="center", ha="left", fontsize=8.6, color=COLORS["ink"])


def draw() -> None:
    _setup_fonts()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(16.8, 10.2), dpi=220)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.5,
        0.965,
        "COLREGs-Constrained Rule-Imitation SAC for Safe USV Tracking",
        ha="center",
        va="top",
        fontsize=19,
        fontweight="bold",
        color=COLORS["ink"],
    )
    ax.text(
        0.5,
        0.925,
        "安全动作过滤器 + 规则模仿损失 + 3DOF 欠驱动船舶模型",
        ha="center",
        va="top",
        fontsize=12.2,
        color=COLORS["muted"],
    )

    x1, x2, x3 = 0.045, 0.375, 0.705
    w = 0.250
    y_top, h_top = 0.575, 0.300
    y_bot, h_bot = 0.245, 0.260

    scenario_body = (
        "随机海域：4 km × 4 km 方形边界\n"
        "目标船 / 动态障碍船 / 静态障碍物\n\n"
        "观测输入 $o_t$：\n"
        "• 自身状态 + 目标相对位置\n"
        "• 动态槽：$[dx,dy,d,size,rv_x,rv_y,type]$\n"
        "• 静态槽：$[dx,dy,d,size,type]$"
    )
    box(ax, (x1, y_top), w, h_top, "① 场景与观测构造", scenario_body, COLORS["scenario"], COLORS["scenario_edge"], fontsize=10.1)

    sac_body = (
        "Actor 输出连续原始动作：\n"
        "$a^{raw}_t=[T_t,\\delta_t]$\n\n"
        "SAC 基础目标：\n"
        "$\\mathcal{L}_{\\pi}^{SAC}=\\mathbb{E}[\\alpha\\log\\pi_{\\theta}(a|o)-Q_{\\phi}(o,a)]$\n\n"
        "Critic / Target Critic / 自动熵调节\n"
        "保证连续动作探索与利用平衡"
    )
    box(ax, (x2, y_top), w, h_top, "② SAC 连续动作策略", sac_body, COLORS["sac"], COLORS["sac_edge"], fontsize=9.9)

    filter_body = (
        "风险会遇时，不直接执行危险动作，\n"
        "而是投影/替换到规则安全动作集：\n"
        "$a^{app}_t=\\mathcal{F}_{COLREGs}(o_t,a^{raw}_t)$\n\n"
        "覆盖核心规则：\n"
        "Rule 8：及早、明显、平滑避让\n"
        "Rule 13：追越船让路\n"
        "Rule 14：对遇右转\n"
        "Rule 15/16/17：交叉会遇让路/直航"
    )
    box(ax, (x3, y_top), w, h_top, "③ 海事规则安全动作过滤器", filter_body, COLORS["filter"], COLORS["filter_edge"], fontsize=9.8)

    dyn_body = (
        "执行动作进入欠驱动 USV 模型：\n"
        "$\\eta=[x,y,\\psi]^T,\\quad\\nu=[u,v,r]^T$\n"
        "$\\dot{\\eta}=R(\\psi)\\nu+v_c$\n"
        "$M\\dot{\\nu}+C(\\nu)\\nu+D(\\nu)\\nu=\\tau(T,\\delta)$\n\n"
        "约束：$T\\in[T_{min},T_{max}]$, "
        "$\\delta\\in[-\\delta_{max},\\delta_{max}]$\n"
        "阻尼 / 洋流 / 限速 / 限舵角 / 转向平滑"
    )
    box(ax, (x3, y_bot), w, h_bot, "④ 3DOF 船舶动力学环境", dyn_body, COLORS["dyn"], COLORS["dyn_edge"], fontsize=9.5)

    buffer_body = (
        "经验池保存真实环境转移：\n"
        "$(o_t,a^{app}_t,r_t,o_{t+1},done)$\n\n"
        "额外记录训练信号：\n"
        "$a^{raw}_t$, $a^{app}_t$, $I^{filter}_t$\n"
        "并统计修正幅度与过滤器介入率"
    )
    box(ax, (x2, y_bot), w, h_bot, "⑤ 经验池与规则介入记录", buffer_body, COLORS["buffer"], COLORS["buffer_edge"], fontsize=9.9)

    loss_body = (
        "新增 actor 规则模仿项：\n"
        "$\\mathcal{L}_{imit}=\\mathbb{E}[I^{filter}_t\\|\\tanh(\\mu_\\theta(o_t))-a^{app}_t\\|_2^2]$\n\n"
        "最终策略损失：\n"
        "$\\mathcal{L}_{\\pi}=\\mathcal{L}_{\\pi}^{SAC}+\\lambda_{rule}\\mathcal{L}_{imit}$\n\n"
        "目标：让策略逐步主动输出\n"
        "合规、安全、平滑的避让动作"
    )
    box(ax, (x1, y_bot), w, h_bot, "⑥ 规则引导的 SAC 更新", loss_body, COLORS["loss"], COLORS["loss_edge"], fontsize=9.4)

    # Main flow arrows.
    arrow(ax, (x1 + w, 0.730), (x2, 0.730), "$o_t$", color=COLORS["scenario_edge"])
    arrow(ax, (x2 + w, 0.730), (x3, 0.730), "$a^{raw}_t$", color=COLORS["sac_edge"])
    arrow(ax, (x3 + 0.125, y_top), (x3 + 0.125, y_bot + h_bot), "$a^{app}_t$", color=COLORS["filter_edge"])
    arrow(ax, (x3, 0.375), (x2 + w, 0.375), "$(r_t,o_{t+1},done)$", color=COLORS["dyn_edge"])
    arrow(ax, (x2, 0.375), (x1 + w, 0.375), "sample", color=COLORS["buffer_edge"])
    arrow(ax, (x1 + 0.125, y_bot + h_bot), (x2 + 0.125, y_top), "update $\\theta,\\phi,\\alpha$", color=COLORS["loss_edge"], rad=-0.25)

    # Feedback and environment loop.
    ax.plot([x3 + 0.125, x3 + 0.125], [y_bot, 0.178], color=COLORS["line"], lw=1.55)
    ax.plot([x3 + 0.125, 0.026], [0.178, 0.178], color=COLORS["line"], lw=1.55)
    ax.plot([0.026, 0.026], [0.178, 0.725], color=COLORS["line"], lw=1.55)
    arrow(ax, (0.026, 0.725), (x1, 0.725), None, color=COLORS["line"], lw=1.55)
    ax.text(
        0.500,
        0.189,
        "下一步环境状态与观测",
        ha="center",
        va="center",
        fontsize=9.2,
        color=COLORS["muted"],
        bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="none", alpha=0.92),
    )

    # Safety/action-set detail strip.
    ax.text(0.500, 0.118, "创新点凝练", ha="center", va="center", fontsize=12.5, fontweight="bold", color=COLORS["ink"])
    pill(ax, 0.080, 0.070, "COLREGs 安全动作过滤", COLORS["filter_edge"], width=0.210)
    pill(ax, 0.315, 0.070, "规则模仿损失", COLORS["loss_edge"], width=0.165)
    pill(ax, 0.505, 0.070, "3DOF 推力-舵角控制", COLORS["dyn_edge"], width=0.205)
    pill(ax, 0.735, 0.070, "安全/效率奖励与指标接口", COLORS["sac_edge"], width=0.225)

    # Small note for interpretation.
    ax.text(
        0.050,
        0.030,
        "核心思想：训练阶段允许规则过滤器保障安全，同时把过滤器的正确动作反向塑造成策略能力；测试阶段目标是减少依赖过滤器，仍能独立产生合规、安全、平滑的追踪动作。",
        ha="left",
        va="bottom",
        fontsize=10.2,
        color=COLORS["muted"],
    )

    for ext in ("png", "pdf", "svg"):
        path = OUT_DIR / f"usv_safe_rule_imitation_sac_framework.{ext}"
        if ext == "png":
            fig.savefig(path, dpi=600, bbox_inches="tight", pad_inches=0.08)
        else:
            fig.savefig(path, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


if __name__ == "__main__":
    draw()
