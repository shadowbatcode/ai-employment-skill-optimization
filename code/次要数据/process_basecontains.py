import pandas as pd
from pathlib import Path

FILE = "基础环境承载量.xlsx"

# ===== 你可以在这里改情景参数 =====
ALPHAS = [0.1, 0.2, 0.3, 0.5]          # 可达/可采用比例 alpha
BETAS  = [0.002, 0.005, 0.01]          # 预算比例 beta（0.2%、0.5%、1.0%）
# ==================================

# 默认情景参数
DEFAULT_ALPHA = 0.3
DEFAULT_BETA = 0.005

required_cols = [
    "type", "title", "Year", "Quarter",
    "Quarterly Establishments",
    "Third Month Employment",
    "Average Weekly Wage",
]

def main():
    path = Path(FILE)
    if not path.exists():
        raise FileNotFoundError(
            f"找不到文件：{FILE}\n"
            "请确认该文件与代码在同一目录，或把 FILE 改成完整路径。"
        )

    # 读取原始数据
    df = pd.read_excel("基础环境承载量.xlsx", sheet_name="Sheet1")

    # 基础清洗：确保数值列是数值
    num_cols = ["Quarterly Establishments", "Third Month Employment", "Average Weekly Wage"]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    print(f"[OK] 读取数据：{len(df)} 行，包含 {df['type'].nunique()} 个行业类型")

    # ===== 核心衍生指标（逐行计算）=====
    df["Annual Wage (52*Weekly)"] = df["Average Weekly Wage"] * 52
    df["Annual Payroll (Employment*AnnualWage)"] = df["Third Month Employment"] * df["Annual Wage (52*Weekly)"]

    # ===== 按行业(type) + 时间(Year, Quarter)聚合 =====
    # 聚合逻辑：同一行业、同一时间段内，所有title的就业人数和薪资总额相加
    agg_dict = {
        "Quarterly Establishments": "sum",
        "Third Month Employment": "sum",  # N_j：潜在客户数量
        "Annual Payroll (Employment*AnnualWage)": "sum"  # 用于计算金额口径
    }

    df_agg = df.groupby(["type", "Year", "Quarter"], as_index=False).agg(agg_dict)

    print(f"[OK] 聚合后：{len(df_agg)} 行（按行业+时间聚合）")

    # ===== 计算默认情景的K0（基础承载力）=====
    df_agg["K0_users (default alpha=0.3)"] = df_agg["Third Month Employment"] * DEFAULT_ALPHA
    df_agg["K0_dollars (default alpha=0.3, beta=0.005)"] = (
        df_agg["Annual Payroll (Employment*AnnualWage)"] * DEFAULT_ALPHA * DEFAULT_BETA
    )

    # ===== 多情景分析：用户口径 =====
    users_scn = df_agg[["type", "Year", "Quarter"]].copy()
    for a in ALPHAS:
        users_scn[f"K0_users_alpha={a}"] = df_agg["Third Month Employment"] * a

    # ===== 多情景分析：金额口径 =====
    dollars_scn = df_agg[["type", "Year", "Quarter"]].copy()
    payroll = df_agg["Annual Payroll (Employment*AnnualWage)"]
    for a in ALPHAS:
        for b in BETAS:
            dollars_scn[f"K0_dollars_alpha={a}_beta={b}"] = payroll * a * b

    # ===== 写回 Excel：新增多个sheet =====
    out_file = Path("基础环境承载量_按行业聚合.xlsx")  # 另存为新文件

    with pd.ExcelWriter(out_file, engine="openpyxl") as writer:
        # Sheet1: 原始数据（带衍生指标）
        df.to_excel(writer, sheet_name="原始数据_逐title", index=False)

        # Sheet2: 按行业聚合后的基础数据
        df_agg.to_excel(writer, sheet_name="按行业聚合_基础数据", index=False)

        # Sheet3: 用户口径多情景
        users_scn.to_excel(writer, sheet_name="K0_用户口径情景", index=False)

        # Sheet4: 金额口径多情景
        dollars_scn.to_excel(writer, sheet_name="K0_金额口径情景", index=False)

    print(f"\n[OK] 计算完成！已生成文件：{out_file.name}")
    print(f"   - 原始数据_逐title: {len(df)} 行")
    print(f"   - 按行业聚合_基础数据: {len(df_agg)} 行")
    print(f"   - K0_用户口径情景: {len(ALPHAS)} 种alpha参数")
    print(f"   - K0_金额口径情景: {len(ALPHAS)}x{len(BETAS)} = {len(ALPHAS)*len(BETAS)} 种参数组合")

if __name__ == "__main__":
    main()
