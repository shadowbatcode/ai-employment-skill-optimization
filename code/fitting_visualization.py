from __future__ import annotations
import os
import re
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import FuncFormatter
from sklearn.metrics import r2_score
from scipy.optimize import differential_evolution
# =============================================================================
# 0. Configuration
# =============================================================================
FILE_PATH = r"c:\Users\Wu\Desktop\美赛F题数据集\代码\程序所需数据.xlsx"
SHEET_BASE = "base"
SHEET_AI = "influence"
OUTPUT_DIR = r"c:\Users\Wu\Desktop\美赛F题数据集\代码"
FIG_PREFIX = "Fit_Analysis"
# AI events
AI_KEY_THRESHOLD = 0.7
# Regularization strength (keeps parameters close to original)
LAMBDA_REG = 0.3
# Differential Evolution settings
DE_SEED = 42
DE_MAXITER = 500
DE_ATOL = 1e-6
DE_TOL = 1e-6
# LLM/Generative AI highlighting
MAX_LLM_LABELS_PER_AX = 6
MAX_NONLLM_LABELS_PER_AX = 3
# =============================================================================
# 1. Helper functions
# =============================================================================
warnings.filterwarnings("ignore", category=RuntimeWarning)
def smape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-10) -> float:
    """Symmetric Mean Absolute Percentage Error."""
    denom = 0.5 * (np.abs(y_true) + np.abs(y_pred)) + eps
    return float(np.mean(np.abs(y_pred - y_true) / denom))
def sanitize_filename(name: str, max_len: int = 120) -> str:
    """Make filename safe for Windows."""
    unsafe = ['/', '\\', ':', '*', '?', '"', '<', '>', '|']
    for ch in unsafe:
        name = name.replace(ch, "_")
    return name[:max_len].strip()
def to_year_quarter_float(x) -> float:
    """
    Convert various representations to float.
    Example: 2019.25 or 2019.1 etc.
    Keeps behavior flexible for your dataset.
    """
    s = str(x).strip()
    try:
        return float(s)
    except ValueError:
        s = s.replace(",", "")
        return float(s)
def blend_with_white(hex_color: str, blend: float) -> tuple:
    """
    Create same-hue variants by blending base color with white.
    blend in [0, 1]: 0 -> original color, 1 -> white
    """
    rgb = np.array(mcolors.to_rgb(hex_color))
    white = np.array([1.0, 1.0, 1.0])
    out = (1 - blend) * rgb + blend * white
    return tuple(out)
def make_short_label(title: str, max_len: int = 12) -> str:
    """
    把长行业名压缩成可读缩写，用于线尾标注。
    规则：取每个单词首字母(过滤and/of/the等)，若仍太长则截断。
    """
    stop = {"and", "of", "the", "to", "in", "for", "a", "an", "with", "on"}
    words = [w for w in re.split(r"[^A-Za-z0-9]+", str(title)) if w]
    key = "".join([w[0].upper() for w in words if w.lower() not in stop])
    if not key:
        key = str(title)[:max_len]
    return key[:max_len]
def get_adaptive_formatter():
    """
    百分比格式化器。
    """
    def formatter(x, pos):
        return f"{x:.0f}%"

    return FuncFormatter(formatter)
def detect_turning_points(y_values: np.ndarray, min_change_threshold: float = 2.0) -> list:
    """
    检测曲线的转折点（局部极大值和极小值）。

    Args:
        y_values: 曲线的y值数组
        min_change_threshold: 最小变化幅度（%），低于此值的波动不算转折点

    Returns:
        转折点索引列表
    """
    turning_points = []
    n = len(y_values)

    for i in range(1, n - 1):
        # 局部极大值
        if y_values[i] > y_values[i-1] and y_values[i] > y_values[i+1]:
            # 检查变化幅度是否显著
            change = abs(y_values[i] - y_values[i-1]) + abs(y_values[i] - y_values[i+1])
            if change >= min_change_threshold:
                turning_points.append(i)
        # 局部极小值
        elif y_values[i] < y_values[i-1] and y_values[i] < y_values[i+1]:
            change = abs(y_values[i] - y_values[i-1]) + abs(y_values[i] - y_values[i+1])
            if change >= min_change_threshold:
                turning_points.append(i)

    return turning_points

def add_end_labels(ax, end_label_points: dict, x_pad=0.10, min_dy_frac=0.03, fontsize=9):
    """
    Add end labels with vertical avoidance.
    end_label_points: dict of short_lab: (x_last, y_last, c)
    """
    if not end_label_points:
        return
    items = sorted([(lab, *end_label_points[lab]) for lab in end_label_points], key=lambda z: z[2])
    y0, y1 = ax.get_ylim()
    min_dy = min_dy_frac * (y1 - y0)
    placed_y = []
    for lab, x_last, y_last, c in items:
        y = y_last
        for py in placed_y:
            if abs(y - py) < min_dy:
                y = py + min_dy
        y = min(max(y, y0 + 0.02 * (y1 - y0)), y1 - 0.02 * (y1 - y0))
        placed_y.append(y)
        ax.text(
            x_last + x_pad, y, lab,
            color=c, fontsize=fontsize,
            va="center", ha="left",
            alpha=0.95, clip_on=False
        )
# LLM / Generative AI keyword pattern (edit if needed)
LLM_PATTERN = re.compile(
    r"(gpt|chatgpt|llm|large language|generative|foundation model|transformer|bert|"
    r"claude|gemini|llama|mixtral|copilot|stable diffusion|midjourney|dall[\-\s]?e)",
    re.IGNORECASE,
)
def is_llm_event(event_text: str) -> bool:
    return bool(LLM_PATTERN.search(str(event_text)))
def emphasize_events(ax, x_min, x_max, df_ai_key, y_text_level=None):
    """
    Lite 版：只画事件竖线 + 很淡的span，不画文字标签（避免信息过载）
    - LLM: 红色实线更粗
    - non-LLM: 黑色点线更细
    只保留少量关键事件
    """
    within = df_ai_key[(df_ai_key["t"] >= x_min) & (df_ai_key["t"] <= x_max)].copy()
    if within.empty:
        return
    within["is_llm"] = within["event"].apply(is_llm_event)
    within = within.sort_values("influence", ascending=False)
    # 只保留前1-2个事件
    llm_events = within[within["is_llm"]].head(2)
    non_llm_events = within[~within["is_llm"]].head(1)
    for _, ev in llm_events.iterrows():
        x = float(ev["t"])
        infl = float(ev["influence"])
        scale = min(1.0, max(0.0, (infl - AI_KEY_THRESHOLD) / max(1e-12, (1.0 - AI_KEY_THRESHOLD))))
        lw = 1.4 + 1.4 * scale
        ax.axvline(x=x, color="red", linestyle="-", linewidth=lw, alpha=0.35, zorder=1)
        ax.axvspan(x - 0.02, x + 0.02, color="red", alpha=0.05, zorder=0)
    for _, ev in non_llm_events.iterrows():
        x = float(ev["t"])
        infl = float(ev["influence"])
        scale = min(1.0, max(0.0, (infl - AI_KEY_THRESHOLD) / max(1e-12, (1.0 - AI_KEY_THRESHOLD))))
        lw = 0.8 + 0.8 * scale
        ax.axvline(x=x, color="black", linestyle=":", linewidth=lw, alpha=0.15, zorder=1)
# =============================================================================
# 2. Load and prepare data
# =============================================================================
df = pd.read_excel(FILE_PATH, sheet_name=SHEET_BASE).copy()
# Create continuous time and normalized time
df["t"] = df["Year"] + (df["Quarter"] - 1) / 4.0
t_min, t_max = df["t"].min(), df["t"].max()
df["t_norm"] = (df["t"] - t_min) / (t_max - t_min)
df_ai = pd.read_excel(FILE_PATH, sheet_name=SHEET_AI).copy()
df_ai["t"] = df_ai["Time"].apply(to_year_quarter_float)
df_ai["influence"] = pd.to_numeric(df_ai["influence"], errors="coerce")
df_ai_key = df_ai.loc[df_ai["influence"] >= AI_KEY_THRESHOLD].copy()
# =============================================================================
# 3. Industry parameters (initial)
# =============================================================================
params = pd.DataFrame(
    [
        ["Motion Picture and Video Industries", "Art", 0.724, 0.382, 9.412, 0.118],
        ["Motion picture and sound recording industries", "Art", 0.783, 0.368, 9.215, 0.101],
        ["Performing Arts", "Art", 0.912, 0.142, 10.084, 0.149],
        ["Specialized Design Services", "Art", 0.653, 0.119, 10.517, 0.024],
        ["Museums, historical sites, and similar institutions", "Art", 0.818, 0.131, 10.203, 0.042],
        ["Food services and drinking places", "Trade", 0.879, 0.108, 9.614, 0.026],
        ["Repair and Maintenance", "Trade", 0.812, 0.121, 9.692, 0.040],
        ["Construction of buildings", "Trade", 0.747, 0.128, 9.503, 0.057],
        ["Specialty Trade Contractors", "Trade", 0.841, 0.112, 9.796, 0.048],
        ["Computing infrastructure providers, data processing, web hosting, and related services", "STEM", 0.952, 0.092, 9.318, 0.026],
        ["Computer Systems Design and Related Services", "STEM", 0.681, 0.102, 10.012, 0.057],
        ["Computer and Electronic Product Manufacturing", "STEM", 0.418, 0.079, 10.824, 0.072],
        ["Scientific Research and Development Services", "STEM", 0.892, 0.109, 9.905, 0.061],
    ],
    columns=["title", "sector", "theta", "Ac", "k", "smape_init"],
)
df = df.merge(params, on="title", how="inner")
# =============================================================================
# 4. Fit model per industry title
# =============================================================================
df["Employment_Pred"] = np.nan
fit_rows = []
for title, g in df.groupby("title", sort=False):
    idx = g.index
    y_true = g["Third Month Employment"].to_numpy(dtype=float)
    t_norm_local = g["t_norm"].to_numpy(dtype=float)
    y_bar = float(np.mean(y_true))
    theta0 = float(g["theta"].iloc[0])
    Ac0 = float(g["Ac"].iloc[0])
    k0 = float(g["k"].iloc[0])
    def objective(x: np.ndarray) -> float:
        theta, Ac, k = x
        impact = 1.0 / (1.0 + np.exp(-k * (t_norm_local - Ac)))
        y_pred = y_true - theta * impact * (y_true - y_bar)
        err = smape(y_true, y_pred)
        reg = (
            ((theta - theta0) / max(theta0, 1e-12)) ** 2
            + ((Ac - Ac0) / max(Ac0, 1e-12)) ** 2
            + ((k - k0) / max(k0, 1e-12)) ** 2
        ) / 3.0
        return float(err + LAMBDA_REG * reg)
    bounds = [
        (max(0.10, theta0 - 0.30), min(1.00, theta0 + 0.30)),
        (max(0.00, Ac0 - 0.20), min(1.00, Ac0 + 0.20)),
        (max(1.00, k0 - 3.00), min(20.0, k0 + 3.00)),
    ]
    result = differential_evolution(
        objective,
        bounds=bounds,
        seed=DE_SEED,
        maxiter=DE_MAXITER,
        atol=DE_ATOL,
        tol=DE_TOL,
        updating="deferred",
        polish=True,
    )
    theta_opt, Ac_opt, k_opt = result.x
    impact_opt = 1.0 / (1.0 + np.exp(-k_opt * (t_norm_local - Ac_opt)))
    y_pred = y_true - theta_opt * impact_opt * (y_true - y_bar)
    df.loc[idx, "Employment_Pred"] = y_pred
    smape_val = smape(y_true, y_pred)
    mae_val = float(np.mean(np.abs(y_true - y_pred)))
    r2_val = float(r2_score(y_true, y_pred))
    fit_rows.append(
        {
            "title": title,
            "sector": g["sector"].iloc[0],
            "SMAPE": smape_val,
            "MAE": mae_val,
            "R2": r2_val,
            "theta_opt": float(theta_opt),
            "Ac_opt": float(Ac_opt),
            "k_opt": float(k_opt),
        }
    )
results_df = pd.DataFrame(fit_rows)
# =============================================================================
# 5. Grouped visualization: 1x3 (STEM / Art / Trade) [Optimized]
# =============================================================================
CATEGORY_COLORS = {
    "STEM": "#5392CE",
    "Trade": "#CDA3CB",
    "Art": "#75B956",
}
def plot_grouped_figure(df_all: pd.DataFrame, df_ai_key: pd.DataFrame, output_path: str):
    sectors_order = ["STEM", "Art", "Trade"]
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.fontsize": 9,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )
    # --- 2x2 布局 ---
    fig, axes = plt.subplots(
        nrows=2, ncols=2,
        figsize=(18, 14),  # 调整为方形布局
        sharex=False,
        sharey=False,
        constrained_layout=False  # 改为False以便手动调整
    )
    # 手动调整布局，留出顶部空间给标题
    plt.subplots_adjust(top=0.94, bottom=0.08, left=0.08, right=0.96, hspace=0.20, wspace=0.15)

    # 将axes展平以便于索引: [0,0]=STEM, [0,1]=Art, [1,0]=Trade, [1,1]=参数表
    axes_flat = [axes[0,0], axes[0,1], axes[1,0]]

    fig.suptitle(
        "Employment Change Trends Across Sectors: Observed vs Fitted (% Change from Baseline)",
        fontsize=15, fontweight="bold", y=0.97
    )
    # 全局图例（避免每个子图重复）
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    global_handles = [
        Line2D([0], [0], color="black", lw=2, marker="o", linestyle="-", label="Observed"),
        Line2D([0], [0], color="black", lw=2, marker="s", linestyle="--", label="Fitted"),
        Patch(facecolor='gray', alpha=0.15, label='Pandemic Period (2020-2022)'),
        Line2D([0], [0], color="red", lw=3, linestyle="-", label="LLM/GenAI key event"),
        Line2D([0], [0], color="black", lw=2, linestyle=":", label="Non-LLM key event"),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='#FF6B6B',
               markersize=8, linestyle='None', label="Peak (local max)"),
        Line2D([0], [0], marker='v', color='w', markerfacecolor='#4ECDC4',
               markersize=8, linestyle='None', label="Trough (local min)"),
    ]
    # 将全局图例移到c图（Trade图）的右下角
    fig.legend(handles=global_handles, loc="upper right", bbox_to_anchor=(0.52, 0.12),
               ncol=1, frameon=True, fontsize=9)
    global_xmin = df_all["t"].min()
    global_xmax = df_all["t"].max() + 0.38

    # 底标标签
    panel_labels = ['(a)', '(b)', '(c)']

    for idx, (ax, sector) in enumerate(zip(axes_flat, sectors_order)):
        gsec = df_all[df_all["sector"] == sector].copy()
        if gsec.empty:
            ax.set_title(f"{sector} (no data)", fontweight="bold")
            ax.axis("off")
            continue
        titles = sorted(gsec["title"].unique())
        base_color = CATEGORY_COLORS[sector]
        # 同色系浅色梯度（调整范围使颜色更明显）
        n = len(titles)
        blends = np.linspace(0.00, 0.50, max(n, 2))[:n]
        # --- 线尾标注信息 ---
        end_label_points = {}
        # 收集用于职业图例的handles和截断标签
        profession_handles = []
        profession_labels = []
        for i, title in enumerate(titles):
            gt = gsec[gsec["title"] == title].sort_values("t").copy()
            # 直接使用原始就业人数，不做归一化
            y_true = gt["Third Month Employment"].to_numpy()
            y_pred = gt["Employment_Pred"].to_numpy()

            # 计算百分比变化（相对于第一个值）
            y_true_baseline = y_true[0]
            y_pred_baseline = y_pred[0]
            y_true_pct = ((y_true - y_true_baseline) / y_true_baseline) * 100
            y_pred_pct = ((y_pred - y_pred_baseline) / y_pred_baseline) * 100

            c = blend_with_white(base_color, float(blends[i]))
            obs_line, = ax.plot(
                gt["t"], y_true_pct,
                linestyle="-", marker="o",
                linewidth=2.0, markersize=4.0,
                color=c, alpha=0.95, zorder=2
            )
            # 四舍五入预测值以避免显示过多小数
            y_pred_pct_rounded = np.round(y_pred_pct, 1)
            ax.plot(
                gt["t"], y_pred_pct_rounded,
                linestyle="--", marker="s",
                linewidth=1.8, markersize=3.5,
                color=c, alpha=0.75, zorder=2
            )

            # 检测并标记转折点
            turning_points_idx = detect_turning_points(y_true_pct, min_change_threshold=3.0)
            t_values = gt["t"].to_numpy()

            for tp_idx in turning_points_idx:
                tp_t = t_values[tp_idx]
                tp_y = y_true_pct[tp_idx]

                # 判断是极大值还是极小值
                is_peak = (tp_idx > 0 and tp_idx < len(y_true_pct) - 1 and
                          y_true_pct[tp_idx] > y_true_pct[tp_idx-1] and
                          y_true_pct[tp_idx] > y_true_pct[tp_idx+1])

                # 极大值用红色向上三角，极小值用蓝色向下三角
                marker_style = '^' if is_peak else 'v'
                marker_color = '#FF6B6B' if is_peak else '#4ECDC4'

                ax.scatter(tp_t, tp_y,
                          marker=marker_style, s=100,
                          color=marker_color,
                          edgecolors='white', linewidths=1.5,
                          alpha=0.9, zorder=10)

                # 添加转折点年份标注（小字体）
                year_quarter = int(tp_t), int((tp_t % 1) * 4) + 1
                label_text = f"{year_quarter[0]}Q{year_quarter[1]}"
                offset = 8 if is_peak else -8
                ax.annotate(label_text,
                           xy=(tp_t, tp_y),
                           xytext=(0, offset),
                           textcoords='offset points',
                           fontsize=6.5,
                           color=marker_color,
                           weight='bold',
                           ha='center',
                           alpha=0.85,
                           bbox=dict(boxstyle='round,pad=0.3',
                                   facecolor='white',
                                   edgecolor=marker_color,
                                   alpha=0.7, linewidth=0.8))

            # 生成截断的全称用于图例，缩写用于线尾标注
            max_legend_len = 25
            truncated_label = title[:max_legend_len] + "..." if len(title) > max_legend_len else title
            short_lab = make_short_label(title, max_len=12)
            x_last = float(gt["t"].iloc[-1])
            y_last_pct = float(y_true_pct[-1])
            end_label_points[short_lab] = (x_last, y_last_pct, c)
            profession_handles.append(obs_line)
            profession_labels.append(truncated_label)
        # 职业图例（截断名称，自动调整位置）
        ax.legend(
            handles=profession_handles,
            labels=profession_labels,
            loc="best",
            fontsize=8,
            title=f"{sector} Sectors",
            title_fontsize=9,
            ncol=1,
            framealpha=0.9
        )
        # 添加底标（底部中间）
        ax.text(0.5, -0.12, panel_labels[idx], transform=ax.transAxes,
                fontsize=14, fontweight='bold', va='top', ha='center')
        ax.set_title(sector, fontsize=12, fontweight="bold", color="black")
        # 添加疫情时段灰色阴影区域 (2020-2022)
        ax.axvspan(2020.0, 2022.0, color='gray', alpha=0.15, zorder=0, label='Pandemic Period')
        # 左侧子图显示Y轴标签 (STEM和Trade)
        if sector in ["STEM", "Trade"]:
            ax.set_ylabel("Employment Change (%)", fontsize=11, fontweight="bold")
        # 底部子图显示X轴标签 (Art和Trade)
        if sector in ["Art", "Trade"]:
            ax.set_xlabel("Time (Year)", fontsize=11, fontweight="bold")
        # 添加y=0基准线
        ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.8, alpha=0.5, zorder=1)
        ax.grid(True, alpha=0.20, linestyle="--")
        # 事件强调
        x_min, x_max = gsec["t"].min(), gsec["t"].max()
        emphasize_events(ax, x_min, x_max, df_ai_key)
        # 线尾标注（Observed 末端）
        add_end_labels(ax, end_label_points, x_pad=0.12, min_dy_frac=0.035, fontsize=8.2)
        # 统一 x 轴范围
        ax.set_xlim(global_xmin, global_xmax)
        # 百分比格式化y轴
        formatter = get_adaptive_formatter()
        ax.yaxis.set_major_formatter(formatter)
        # 精简方程特性文本框（只放均值，左下角）
        theta_mean = float(gsec["theta"].mean())
        Ac_mean = float(gsec["Ac"].mean())
        k_mean = float(gsec["k"].mean())
        eq_txt = (
            "Logistic impact I(t)\n"
            f"Sector avg: θ≈{theta_mean:.2f}  Ac≈{Ac_mean:.2f}  k≈{k_mean:.1f}"
        )
        ax.text(
            0.02, 0.02, eq_txt,
            transform=ax.transAxes,
            ha="left", va="bottom",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.35", alpha=0.12, edgecolor="gray")
        )
        # 删除缩写映射表（因为有全称legend）

    # ========== 右下角参数表格 ==========
    ax_table = axes[1, 1]
    ax_table.axis('off')

    # 添加底标(d)（底部中间）
    ax_table.text(0.5, -0.12, '(d)', transform=ax_table.transAxes,
                  fontsize=14, fontweight='bold', va='top', ha='center')

    # 参数表格数据
    param_table_data = [
        ['Parameter', 'Value', 'Parameter', 'Value'],
        [r'$r_i^{(0)}$ (base growth)', 'Data-driven', r'$\theta_i$ (AI sensitivity)', 'Optimized [-1,1]'],
        [r'$K_i^{(0)}$ (base capacity)', '95%ile×1.2', r'$A_{c,i}$ (critical maturity)', 'Optimized [0.1,0.9]'],
        [r'$\alpha_{ij}$ (competition)', 'Sheet-Related', r'$k_i$ (penetration steep)', 'Optimized [1,10]'],
        [r'$\sigma^2$ (noise variance)', '2', 'AICompatibility', 'Cosine sim [0,1]'],
        [r'$O_s$ (init occupations)', '45', r'$\gamma_1$ (STEM weight)', '168.05'],
        [r'$D_o$ (optim. constraint)', '8', r'$\gamma_2$ (Arts weight)', '1.50']
    ]

    # 创建表格
    table = ax_table.table(
        cellText=param_table_data,
        cellLoc='left',
        loc='center',
        bbox=[0.0, 0.0, 1.0, 0.95]
    )

    # 设置表格样式
    table.auto_set_font_size(False)
    table.set_fontsize(12)

    # 设置表头样式
    for j in range(4):
        cell = table[(0, j)]
        cell.set_facecolor('#6FA8DC')
        cell.set_text_props(weight='bold', fontsize=13, color='white')
        cell.set_height(0.08)

    # 设置数据行样式
    for i in range(1, len(param_table_data)):
        for j in [0, 2]:  # Parameter列
            cell = table[(i, j)]
            cell.set_text_props(weight='bold', fontsize=11.5)
            cell.set_facecolor('#E8F0FE')
        # Value列
        for j in [1, 3]:
            cell = table[(i, j)]
            cell.set_facecolor('#F5F5F5')
            cell.set_text_props(fontsize=11)

    # 设置行高
    for i in range(len(param_table_data)):
        for j in range(4):
            table[(i, j)].set_height(0.12)

    # 设置列宽
    for j in range(4):
        if j % 2 == 0:  # Parameter列 (0, 2)
            table.auto_set_column_width([j])
        else:  # Value列 (1, 3)
            table.auto_set_column_width([j])

    # 添加表格标题
    ax_table.text(0.5, 0.98, 'Model Parameters Summary',
                  transform=ax_table.transAxes,
                  fontsize=12, fontweight='bold',
                  ha='center', va='top')

    fig.savefig(output_path, dpi=600, bbox_inches="tight")
    plt.show()
    plt.close(fig)
# Ensure output dir exists
os.makedirs(OUTPUT_DIR, exist_ok=True)
group_fig_path = os.path.join(OUTPUT_DIR, f"{FIG_PREFIX}_Grouped_2x2_with_ParamTable.png")
plot_grouped_figure(df, df_ai_key, group_fig_path)
# =============================================================================
# 6. Output summary
# =============================================================================
print("=" * 88)
print("Model fitting completed and grouped figure generated successfully!")
print("=" * 88)
print("\nFit summary (sorted by sector then R2 descending):")
print(results_df.sort_values(["sector", "R2"], ascending=[True, False]).to_string(index=False))
print(f"\nGrouped figure saved to:\n{group_fig_path}")
