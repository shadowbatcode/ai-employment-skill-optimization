"""
Occupational Competition Analysis under Generative AI Impact (English-only)

Left:  Main-occupation × Related-occupation similarity heatmap
Right: Employment vs Wage change scatter (Pre vs Post inflection)
       Pre:  2020–2022
       Post: 2023–2025

Key fixes in this version:
- English-only (titles/labels/prints)
- Heatmap labels clipped/shortened + more margin to prevent overflow
- Right plot view focused to x∈[-30,30], y∈[0,80] WITHOUT extra text like "view focused..."
- Guaranteed saving (creates output directory if needed)
- Manual legends (no "No artists with labels..." warning)
"""

import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# =========================
# Global style
# =========================
plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "Arial", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.family"] = ["Arial"]

TYPE_COLORS = {
    "STEM":  "#5392CE",
    "Trade": "#CDA3CB",
    "Art":   "#75B956",
}

HEATMAP_CMAP = sns.color_palette("YlGnBu", as_cmap=True)


# =========================
# Utilities
# =========================
def load_data(file_path: str):
    print("Loading data...")
    base_df = pd.read_excel(file_path, sheet_name="base")
    related_df = pd.read_excel(file_path, sheet_name="Related")
    print(f"Base shape: {base_df.shape}")
    print(f"Related shape: {related_df.shape}")
    return base_df, related_df


def parse_onet_codes(onet_str):
    if pd.isna(onet_str):
        return []
    codes = str(onet_str).split(",")
    return [c.strip() for c in codes if c.strip()]


def truncate_title(title, max_len=18):
    s = str(title)
    if len(s) <= max_len:
        return s
    words = s.split()
    if len(words) > 1:
        t = " ".join(words[:2])
        if len(t) <= max_len:
            return t
    return s[: max_len - 2] + ".."


def reorder_matrix_by_strength(mat_df: pd.DataFrame) -> pd.DataFrame:
    row_order = mat_df.sum(axis=1).sort_values(ascending=False).index
    col_order = mat_df.sum(axis=0).sort_values(ascending=False).index
    return mat_df.loc[row_order, col_order]


def scale_sizes(series: pd.Series, smin=25, smax=220):
    x = series.to_numpy(dtype=float)
    if x.size == 0:
        return np.array([])
    lo, hi = np.percentile(x, 5), np.percentile(x, 95)
    if hi <= lo:
        return np.full_like(x, (smin + smax) / 2)
    x = np.clip(x, lo, hi)
    x = (x - lo) / (hi - lo + 1e-9)
    return smin + x * (smax - smin)


# =========================
# Similarity matrix (vectorized)
# =========================
def build_similarity_matrix_v2(
    base_df,
    related_df,
    top_n=50,
    main_code_col="O*NET-SOC Code",
    main_title_col=None,
    rel_code_col=None,
    rel_title_col=None,
    score_col=None,
):
    print("\nBuilding similarity matrix (vectorized)...")

    cols = list(related_df.columns)

    # Defaults aligned with your original sheet structure:
    # main title: col[1], related code: col[2], related title: col[3], score: col[9]
    if main_title_col is None:
        main_title_col = cols[1]
    if rel_code_col is None:
        rel_code_col = cols[2]
    if rel_title_col is None:
        rel_title_col = cols[3]
    if score_col is None:
        score_col = cols[9]

    df = related_df[[main_code_col, main_title_col, rel_code_col, rel_title_col, score_col]].copy()
    df[main_code_col] = df[main_code_col].astype(str).str.strip()
    df[rel_code_col] = df[rel_code_col].astype(str).str.strip()
    df[score_col] = pd.to_numeric(df[score_col], errors="coerce").fillna(0)

    # Choose top related occupations by mean score (robust)
    rel_rank = df.groupby(rel_code_col)[score_col].mean().sort_values(ascending=False)
    top_related_codes = rel_rank.head(top_n).index.tolist()
    df = df[df[rel_code_col].isin(top_related_codes)]

    mat_df = df.pivot_table(
        index=main_code_col,
        columns=rel_code_col,
        values=score_col,
        aggfunc="max",
        fill_value=0,
    ).reindex(columns=top_related_codes)

    main_to_title = (
        df[[main_code_col, main_title_col]]
        .drop_duplicates(subset=[main_code_col])
        .set_index(main_code_col)[main_title_col]
        .to_dict()
    )

    rel_to_title = (
        df[[rel_code_col, rel_title_col]]
        .drop_duplicates(subset=[rel_code_col])
        .set_index(rel_code_col)[rel_title_col]
        .to_dict()
    )

    # Base mappings (optional, kept for compatibility)
    naics_to_onet = base_df.set_index("NAICS")["O*NET"].apply(parse_onet_codes).to_dict()
    naics_to_title = base_df.set_index("NAICS")["title"].to_dict()
    naics_to_type = base_df.set_index("NAICS")["type"].to_dict()

    similarity_matrix = mat_df.values
    main_codes = mat_df.index.to_numpy()
    rel_codes = mat_df.columns.to_numpy()

    print(f"Matrix shape: {mat_df.shape}")
    print(f"Non-zero entries: {np.count_nonzero(similarity_matrix)}")

    return (
        similarity_matrix,
        main_codes,
        rel_codes,
        main_to_title,
        rel_to_title,
        naics_to_onet,
        naics_to_title,
        naics_to_type,
        mat_df,
    )


# =========================
# Period changes (Pre vs Post)
# =========================
def calculate_period_changes_v2(base_df: pd.DataFrame) -> pd.DataFrame:
    print("\nComputing period changes (vectorized)...")

    df = base_df.copy()

    required = [
        "NAICS",
        "Year",
        "Quarter",
        "Third Month Employment",
        "Average Weekly Wage",
        "title",
        "type",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in 'base': {missing}")

    # Baseline = 2020 Q1, fallback to first row per NAICS
    baseline = df[(df["Year"] == 2020) & (df["Quarter"] == 1)][
        ["NAICS", "Third Month Employment", "Average Weekly Wage"]
    ].rename(
        columns={
            "Third Month Employment": "baseline_employment",
            "Average Weekly Wage": "baseline_wage",
        }
    )

    all_naics = df["NAICS"].dropna().unique()
    has_baseline = baseline["NAICS"].unique()
    missing_naics = set(all_naics) - set(has_baseline)

    if missing_naics:
        first_rows = (
            df[df["NAICS"].isin(list(missing_naics))]
            .sort_values(["NAICS", "Year", "Quarter"])
            .groupby("NAICS", as_index=False)
            .first()[["NAICS", "Third Month Employment", "Average Weekly Wage"]]
            .rename(
                columns={
                    "Third Month Employment": "baseline_employment",
                    "Average Weekly Wage": "baseline_wage",
                }
            )
        )
        baseline = pd.concat([baseline, first_rows], ignore_index=True)

    out = df.merge(baseline, on="NAICS", how="left")

    out["employment_change"] = np.where(
        out["baseline_employment"] > 0,
        (out["Third Month Employment"] - out["baseline_employment"]) / out["baseline_employment"],
        0,
    )
    out["wage_change"] = np.where(
        out["baseline_wage"] > 0,
        (out["Average Weekly Wage"] - out["baseline_wage"]) / out["baseline_wage"],
        0,
    )

    out = out.rename(
        columns={
            "Third Month Employment": "employment",
            "Average Weekly Wage": "wage",
            "Year": "year",
            "Quarter": "quarter",
        }
    )

    # Inflection split: Pre (2020–2022) vs Post (2023–2025)
    out["period"] = np.where(out["year"] <= 2022, "pre", "post")

    out = out[
        [
            "NAICS",
            "title",
            "type",
            "year",
            "quarter",
            "period",
            "employment",
            "wage",
            "employment_change",
            "wage_change",
        ]
    ]

    print(f"Total points: {len(out)}")
    return out


# =========================
# Plot: Heatmap (prevent label overflow)
# =========================
def plot_similarity_heatmap_v2(ax, mat_df: pd.DataFrame, main_titles: dict, rel_titles: dict):
    print("\nPlotting similarity heatmap...")

    mat_df = reorder_matrix_by_strength(mat_df)

    data = mat_df.values
    positive = data[data > 0]
    vmax = np.percentile(positive, 90) if positive.size else 1

    sns.heatmap(
        mat_df,
        ax=ax,
        cmap=HEATMAP_CMAP,
        vmin=0,
        vmax=vmax,
        linewidths=0.25,
        linecolor="#E8E8E8",
        cbar_kws={"label": "Similarity", "shrink": 0.86, "pad": 0.02},
    )

    ax.set_title("Main vs Related Occupation Similarity", fontsize=14, fontweight="bold", pad=10)
    ax.set_xlabel("Related Occupations", fontsize=11)
    ax.set_ylabel("Main Occupations", fontsize=11)

    # Downsample X ticks
    ncols = mat_df.shape[1]
    step = max(1, ncols // 12)
    xticks = np.arange(0, ncols, step)

    ax.set_xticks(xticks + 0.5)
    ax.set_xticklabels(
        [truncate_title(rel_titles.get(mat_df.columns[i], mat_df.columns[i]), 14) for i in xticks],
        rotation=45,
        ha="right",
        fontsize=7,
    )

    # Y ticks (shorter to avoid overflow)
    ylabels = [truncate_title(main_titles.get(idx, idx), 20) for idx in mat_df.index]
    ax.set_yticklabels(ylabels, rotation=0, fontsize=7)

    # Reduce padding so labels stay inside
    ax.tick_params(axis="x", pad=2)
    ax.tick_params(axis="y", pad=1)

    return ax


# =========================
# Plot: Scatter (focused view + manual legends)
# =========================
def plot_change_scatter_v2(ax, change_df: pd.DataFrame, xlim=(-30, 30), ylim=(0, 80)):
    print("\nPlotting scatter (focused view)...")

    for job_type in ["STEM", "Trade", "Art"]:
        tdf = change_df[change_df["type"] == job_type]
        if tdf.empty:
            continue

        color = TYPE_COLORS[job_type]
        tdf = tdf.copy()
        tdf["size_scaled"] = scale_sizes(tdf["employment"], smin=25, smax=220)

        pre = tdf[tdf["period"] == "pre"]
        post = tdf[tdf["period"] == "post"]

        if not pre.empty:
            ax.scatter(
                pre["employment_change"] * 100,
                pre["wage_change"] * 100,
                s=pre["size_scaled"] * 0.85,
                c=color,
                alpha=0.30,
                edgecolors="white",
                linewidth=0.5,
            )

        if not post.empty:
            ax.scatter(
                post["employment_change"] * 100,
                post["wage_change"] * 100,
                s=post["size_scaled"],
                c=color,
                alpha=0.70,
                edgecolors="black",
                linewidth=0.7,
            )

    ax.axhline(0, color="#666666", linestyle="--", linewidth=1.2, alpha=0.6, zorder=1)
    ax.axvline(0, color="#666666", linestyle="--", linewidth=1.2, alpha=0.6, zorder=1)

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)

    ax.set_title("Competition Survival Space (Quarterly): Pre vs Post Inflection", fontsize=14, fontweight="bold", pad=10)
    ax.set_xlabel("Employment Change (%)", fontsize=11)
    ax.set_ylabel("Wage Change (%)", fontsize=11)

    ax.grid(True, alpha=0.15, linestyle=":", color="#999999", linewidth=0.5, zorder=0)

    # Period note (English-only; keep it clean)
    ax.text(
        0.02,
        0.98,
        "Pre: 2020–2022 (lighter)\nPost: 2023–2025 (darker)",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9.5,
        color="#444444",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#DDDDDD", alpha=0.9),
    )

    # Manual legend: occupation types
    type_handles = []
    for job_type in ["STEM", "Trade", "Art"]:
        if (change_df["type"] == job_type).any():
            type_handles.append(Patch(facecolor=TYPE_COLORS[job_type], edgecolor="none", label=job_type))

    if type_handles:
        leg1 = ax.legend(
            handles=type_handles,
            loc="upper right",
            bbox_to_anchor=(0.98, 0.98),
            fontsize=9.5,
            framealpha=0.9,
            title="Occupation Type",
            title_fontsize=10,
            edgecolor="#CCCCCC",
            fancybox=False,
        )
        ax.add_artist(leg1)

    # Manual legend: size
    size_elems = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="gray", markersize=6, alpha=0.6, label="Smaller"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="gray", markersize=11, alpha=0.6, label="Larger"),
    ]
    ax.legend(
        handles=size_elems,
        loc="lower right",
        bbox_to_anchor=(0.98, 0.02),
        fontsize=8.5,
        framealpha=0.9,
        title="Employment Scale (illustrative)",
        title_fontsize=9,
        edgecolor="#CCCCCC",
        fancybox=False,
    )

    return ax


# =========================
# Combined visualization (fix clipping + ensure saving)
# =========================
def create_combined_visualization(file_path, output_path="occupational_competition.png", top_n=50):
    print("=" * 70)
    print("Generating visualization...")
    print("=" * 70)

    base_df, related_df = load_data(file_path)

    (
        similarity_matrix,
        main_codes,
        rel_codes,
        main_titles,
        rel_titles,
        naics_to_onet,
        naics_to_title,
        naics_to_type,
        mat_df,
    ) = build_similarity_matrix_v2(base_df, related_df, top_n=top_n)

    change_df = calculate_period_changes_v2(base_df)

    print("\nRendering figure...")
    fig = plt.figure(figsize=(22, 9.2), constrained_layout=False)

    # More generous margins to prevent label overflow
    # right leaves room for legends outside axes
    gs = fig.add_gridspec(
        1, 2,
        width_ratios=[1.22, 1.00],
        wspace=0.08,
        left=0.085, right=0.86,
        bottom=0.24, top=0.88
    )

    ax1 = fig.add_subplot(gs[0, 0])
    plot_similarity_heatmap_v2(ax1, mat_df, main_titles, rel_titles)

    ax2 = fig.add_subplot(gs[0, 1])
    plot_change_scatter_v2(ax2, change_df, xlim=(-30, 30), ylim=(0, 80))

    fig.suptitle("Occupational Competition Analysis under Generative AI Impact", fontsize=18, fontweight="bold", y=0.95)
#    fig.text(0.5, 0.915, "Similarity Structure and Pre/Post Inflection Survival Space", ha="center",
#             fontsize=12, color="#666666", style="italic")

    # Ensure output directory exists
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    fig.savefig(output_path, dpi=600, facecolor="white", bbox_inches="tight", pad_inches=0.25)
    print(f"[OK] Saved to: {output_path}")

    plt.show()
    return fig, change_df


def main():
    # Update these paths
    data_file = r"c:\Users\Wu\Desktop\美赛F题数据集\代码\程序所需数据.xlsx"
    output_file = r"c:\Users\Wu\Desktop\美赛F题数据集\代码\occupational_competition.png"

    try:
        fig, change_df = create_combined_visualization(data_file, output_file, top_n=50)

        print("\n" + "=" * 70)
        print("Summary")
        print("=" * 70)
        print(f"Total points: {len(change_df)}")
        print(f"Unique NAICS: {change_df['NAICS'].nunique()}")

        for job_type in ["STEM", "Trade", "Art"]:
            cnt = int((change_df["type"] == job_type).sum())
            print(f"{job_type}: {cnt} points")

        pre_cnt = int((change_df["period"] == "pre").sum())
        post_cnt = int((change_df["period"] == "post").sum())
        print(f"Pre  (2020–2022): {pre_cnt} points")
        print(f"Post (2023–2025): {post_cnt} points")

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
