import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Font settings (cross-platform safe)
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# Global style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.3)

# ─── Data (unchanged) ────────────────────────────────────────────────────────────
data = {}

data["STEM"] = pd.DataFrame({
    "x1": [0.695, 0.895, 0.545, 0.395, 0.345, 0.295, 0.295, 0.245, 0.275],
    "x2": [0.010, 0.010, 0.010, 0.010, 0.010, 0.010, 0.456, 0.256, 0.156],
    "x3": [0.381, 0.531, 0.831, 0.731, 0.631, 0.581, 0.231, 0.281, 0.331],
    "x4": [0.260, 0.410, 0.510, 0.860, 0.810, 0.760, 0.310, 0.210, 0.410],
    "x5": [0.471, 0.521, 0.621, 0.321, 0.271, 0.251, 0.321, 0.271, 0.291],
    "x6": [0.389, 0.439, 0.589, 0.739, 0.689, 0.639, 0.239, 0.289, 0.489],
    "x7": [0.010, 0.010, 0.010, 0.010, 0.010, 0.010, 0.096, 0.546, 0.396],
}, index=[
    "Programming Ability", "AI Tool Usage", "System Architecture",
    "Communication & Collaboration", "Leadership", "Team Management",
    "AI Ethics & Safety", "Environmental Awareness", "Global Perspective"
])

data["Trade"] = pd.DataFrame({
    "x1": [0.510, 1.000, 0.660, 0.460, 0.560, 0.490, 0.410, 0.390, 0.360],
    "x2": [0.091, 0.221, 0.971, 0.121, 0.171, 0.151, 0.191, 0.221, 0.171],
    "x3": [0.011, 0.062, 0.112, 0.762, 0.262, 0.212, 0.612, 0.512, 0.362],
    "x4": [0.220, 0.270, 0.320, 0.370, 0.920, 0.670, 0.270, 0.420, 0.370],
    "x5": [1.000, 0.678, 0.578, 0.728, 0.778, 0.828, 0.678, 0.628, 1.000],
    "x6": [0.279, 0.429, 0.379, 0.479, 0.529, 0.579, 0.329, 0.429, 0.379],
    "x7": [0.010, 0.010, 0.010, 0.010, 0.010, 0.010, 0.319, 0.419, 0.069],
}, index=[
    "Traditional Cooking Skills", "Modern Kitchen Technology", "AI-assisted Menu Planning",
    "Nutrition & Health", "Multi-sensory Experience", "Team Collaboration",
    "Food Safety", "Sustainable Development", "Cultural Understanding"
])

data["Arts"] = pd.DataFrame({
    "x1": [0.010, 0.522, 0.010, 0.010, 0.010, 0.010, 0.472, 0.010, 0.010],
    "x2": [0.108, 0.508, 0.758, 0.608, 0.658, 0.308, 0.358, 0.458, 0.508],
    "x3": [0.307, 0.507, 0.407, 0.807, 0.707, 0.607, 0.357, 0.507, 0.557],
    "x4": [0.990, 0.304, 0.674, 0.724, 0.624, 0.990, 0.424, 0.574, 0.524],
    "x5": [0.105, 0.205, 0.855, 0.255, 0.155, 0.205, 0.655, 0.755, 0.805],
    "x6": [0.377, 0.277, 0.427, 0.877, 0.927, 1.000, 0.527, 0.677, 0.727],
    "x7": [0.230, 0.330, 0.480, 0.530, 0.580, 0.630, 0.430, 0.980, 1.000],
}, index=[
    "Traditional Performance Skills", "AI Collaboration Ability", "Artistic Critical Thinking",
    "Improvisational Creativity", "Team Cooperation", "Emotional Expression",
    "Technical Performance Literacy", "Cultural Awareness", "Social Responsibility"
])

# ─── Color maps per category ─────────────────────────────────────────────────────
cmaps = {
    "STEM":  sns.color_palette("Blues",  as_cmap=True),   # light → deep blue
    "Trade": sns.color_palette("Purples", as_cmap=True),  # light → deep purple
    "Arts":  sns.color_palette("Greens",  as_cmap=True),  # light → deep green
}

# ─── Plotting ────────────────────────────────────────────────────────────────────
for name, df in data.items():
    plt.figure(figsize=(8.0, 7.2), dpi=160)

    ax = sns.heatmap(
        df,
        annot=True,
        fmt=".3f",
        cmap=cmaps[name],              # different color per category
        linewidths=0.5,
        linecolor="white",
        cbar_kws={'label': 'Efficiency Value', 'shrink': 0.65},
        annot_kws={"size": 9.5, "weight": "medium"},
        vmin=0, vmax=1.0,
        square=False,
    )

    # Title and y-label
    plt.title(f"{name} Efficiency Matrix Heatmap", fontsize=17, pad=18, weight='semibold')
    plt.ylabel("Skill Dimensions", fontsize=13, labelpad=12)

    # Hide x-axis
    ax.set_xticklabels([])
    ax.set_xlabel("")

    # y-axis ticks
    plt.yticks(rotation=0, fontsize=10.5, ha='right')

    plt.tight_layout()

    # Save with category-specific name
    filename = f"efficiency_heatmap_{name.lower()}_9x7_colored_no_xlabels.png"
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    print(f"Saved: {filename}")

    plt.close()
