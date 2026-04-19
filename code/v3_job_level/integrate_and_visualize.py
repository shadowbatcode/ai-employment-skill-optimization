"""
数据整合与可视化脚本
- 整合所有预测结果和真实值
- 生成时间序列预测对比图
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from matplotlib import rcParams

# 设置中文字体
rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("数据整合与可视化工具")
print("=" * 60)

# 1. 读取真实值数据
print("\n[1/4] 读取真实值数据...")
data_file = r"../程序所需数据.xlsx"
df_base = pd.read_excel(data_file, sheet_name='base')
print(f"[OK] 读取了 {len(df_base)} 条真实数据记录")

# 2. 读取所有预测文件
print("\n[2/4] 读取预测文件...")
prediction_files = glob.glob("prediction_*_v3.csv")
print(f"[OK] 找到 {len(prediction_files)} 个预测文件")

# 3. 整合数据
print("\n[3/4] 整合数据...")
all_data = []

for pred_file in prediction_files:
    # 从文件名提取职位名称
    job_title = pred_file.replace("prediction_", "").replace("_v3.csv", "").replace("_", " ")

    # 读取预测数据
    df_pred = pd.read_csv(pred_file, encoding='utf-8-sig')

    # 获取该职位的真实数据
    df_real = df_base[df_base['title'] == job_title].copy()

    if len(df_real) == 0:
        print(f"  [警告] 未找到 {job_title} 的真实数据")
        continue

    # 按年份和季度排序
    df_real = df_real.sort_values(['Year', 'Quarter'])

    # 计算时间索引（从2020年开始的季度数）
    df_real['Quarter_Index'] = (df_real['Year'] - 2020) * 4 + (df_real['Quarter'] - 1)

    # 合并预测和真实数据
    for idx, row in df_pred.iterrows():
        quarter = row['Quarter']
        predicted = row['Predicted_Employment']

        # 查找对应的真实值
        real_row = df_real[df_real['Quarter_Index'] == quarter]
        actual = real_row['Third Month Employment'].values[0] if len(real_row) > 0 else None

        all_data.append({
            'Job_Title': job_title,
            'Job_Type': df_real['type'].iloc[0],
            'Quarter': quarter,
            'Year': 2020 + quarter // 4,
            'Quarter_in_Year': (quarter % 4) + 1,
            'Predicted_Employment': predicted,
            'Actual_Employment': actual
        })

# 创建整合的DataFrame
df_integrated = pd.DataFrame(all_data)

# 计算误差指标
df_integrated['Prediction_Error'] = df_integrated['Predicted_Employment'] - df_integrated['Actual_Employment']
df_integrated['Relative_Error_%'] = (df_integrated['Prediction_Error'] / df_integrated['Actual_Employment'] * 100).where(
    df_integrated['Actual_Employment'].notna() & (df_integrated['Actual_Employment'] != 0), None
)

# 保存整合数据
output_file = 'integrated_predictions_with_actual.csv'
df_integrated.to_csv(output_file, index=False, encoding='utf-8-sig')
print(f"[OK] 整合数据已保存至 {output_file}")
print(f"[INFO] 共 {len(df_integrated)} 行数据")
print(f"[INFO] 其中 {df_integrated['Actual_Employment'].notna().sum()} 行包含真实值")

# 4. 生成可视化图表
print("\n[4/4] 生成可视化图表...")

# 获取所有唯一的职位
unique_jobs = df_integrated['Job_Title'].unique()

# 为每个职位类型创建子图
job_types = df_integrated['Job_Type'].unique()
colors = {'STEM': '#1f77b4', 'Trade': '#ff7f0e', 'Art': '#2ca02c'}

# 创建总览图（所有职位）
fig, axes = plt.subplots(len(unique_jobs), 1, figsize=(14, 5 * len(unique_jobs)))
if len(unique_jobs) == 1:
    axes = [axes]

for idx, job_title in enumerate(unique_jobs):
    ax = axes[idx]
    df_job = df_integrated[df_integrated['Job_Title'] == job_title].sort_values('Quarter')

    job_type = df_job['Job_Type'].iloc[0]
    color = colors.get(job_type, '#333333')

    # 绘制预测值
    ax.plot(df_job['Quarter'], df_job['Predicted_Employment'],
            label='Predicted', color=color, linewidth=2, linestyle='-', marker='o', markersize=4)

    # 绘制真实值（如果有）
    df_actual = df_job[df_job['Actual_Employment'].notna()]
    if len(df_actual) > 0:
        ax.plot(df_actual['Quarter'], df_actual['Actual_Employment'],
                label='Actual', color='red', linewidth=2, linestyle='--', marker='s', markersize=4)

        # 填充预测区间（真实值之后的部分）
        last_actual_quarter = df_actual['Quarter'].max()
        df_future = df_job[df_job['Quarter'] > last_actual_quarter]
        if len(df_future) > 0:
            ax.axvspan(last_actual_quarter, df_job['Quarter'].max(),
                      alpha=0.1, color='gray', label='Forecast Period')

    ax.set_title(f'{job_title} ({job_type})', fontsize=12, fontweight='bold')
    ax.set_xlabel('Quarter (from 2020 Q1)', fontsize=10)
    ax.set_ylabel('Employment', fontsize=10)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    # 添加数值标签（仅显示部分点以避免拥挤）
    step = max(1, len(df_job) // 10)
    for i in range(0, len(df_job), step):
        row = df_job.iloc[i]
        ax.annotate(f'{int(row["Predicted_Employment"]):,}',
                   xy=(row['Quarter'], row['Predicted_Employment']),
                   xytext=(0, 10), textcoords='offset points',
                   fontsize=7, ha='center', alpha=0.7)

plt.tight_layout()
plt.savefig('all_jobs_predictions.png', dpi=300, bbox_inches='tight')
print(f"[OK] 总览图已保存至 all_jobs_predictions.png")
plt.close()

# 创建按类型分组的对比图
fig, axes = plt.subplots(1, len(job_types), figsize=(7 * len(job_types), 6))
if len(job_types) == 1:
    axes = [axes]

for idx, job_type in enumerate(job_types):
    ax = axes[idx]
    df_type = df_integrated[df_integrated['Job_Type'] == job_type]

    for job_title in df_type['Job_Title'].unique():
        df_job = df_type[df_type['Job_Title'] == job_title].sort_values('Quarter')

        # 只绘制有真实值的部分
        df_with_actual = df_job[df_job['Actual_Employment'].notna()]
        if len(df_with_actual) > 0:
            ax.plot(df_with_actual['Quarter'], df_with_actual['Predicted_Employment'],
                   label=f'{job_title} (Pred)', linestyle='-', marker='o', markersize=3, alpha=0.7)
            ax.plot(df_with_actual['Quarter'], df_with_actual['Actual_Employment'],
                   label=f'{job_title} (Actual)', linestyle='--', marker='s', markersize=3, alpha=0.7)

    ax.set_title(f'{job_type} Jobs', fontsize=14, fontweight='bold')
    ax.set_xlabel('Quarter (from 2020 Q1)', fontsize=11)
    ax.set_ylabel('Employment', fontsize=11)
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('jobs_by_type_comparison.png', dpi=300, bbox_inches='tight')
print(f"[OK] 分类对比图已保存至 jobs_by_type_comparison.png")
plt.close()

# 创建误差分析图
df_with_error = df_integrated[df_integrated['Actual_Employment'].notna()].copy()
if len(df_with_error) > 0:
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # 绝对误差
    ax1 = axes[0]
    for job_title in df_with_error['Job_Title'].unique():
        df_job = df_with_error[df_with_error['Job_Title'] == job_title].sort_values('Quarter')
        ax1.plot(df_job['Quarter'], df_job['Prediction_Error'],
                label=job_title, marker='o', markersize=4)

    ax1.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax1.set_title('Prediction Error Over Time (Absolute)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Quarter (from 2020 Q1)', fontsize=11)
    ax1.set_ylabel('Prediction Error', fontsize=11)
    ax1.legend(loc='best', fontsize=8)
    ax1.grid(True, alpha=0.3)

    # 相对误差
    ax2 = axes[1]
    for job_title in df_with_error['Job_Title'].unique():
        df_job = df_with_error[df_with_error['Job_Title'] == job_title].sort_values('Quarter')
        ax2.plot(df_job['Quarter'], df_job['Relative_Error_%'],
                label=job_title, marker='o', markersize=4)

    ax2.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax2.set_title('Prediction Error Over Time (Relative %)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Quarter (from 2020 Q1)', fontsize=11)
    ax2.set_ylabel('Relative Error (%)', fontsize=11)
    ax2.legend(loc='best', fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('prediction_errors.png', dpi=300, bbox_inches='tight')
    print(f"[OK] 误差分析图已保存至 prediction_errors.png")
    plt.close()

print("\n" + "=" * 60)
print("数据整合与可视化完成！")
print("生成的文件：")
print("  1. integrated_predictions_with_actual.csv - 整合数据")
print("  2. all_jobs_predictions.png - 所有职位预测图")
print("  3. jobs_by_type_comparison.png - 分类对比图")
print("  4. prediction_errors.png - 误差分析图")
print("=" * 60)