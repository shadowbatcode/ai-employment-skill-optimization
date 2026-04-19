"""
对比v3和v4版本的预测效果
生成对比图表和误差统计
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 设置中文字体
rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("v3 vs v4 预测效果对比工具")
print("=" * 60)

# 1. 读取真实值数据
print("\n[1/4] 读取真实值数据...")
data_file = r"../程序所需数据.xlsx"
df_base = pd.read_excel(data_file, sheet_name='base')
print(f"[OK] 读取了 {len(df_base)} 条真实数据记录")

# 2. 读取v3和v4的预测结果
print("\n[2/4] 读取预测结果...")
try:
    df_v3 = pd.read_csv('all_predictions_v3.csv', encoding='utf-8-sig')
    print(f"[OK] v3预测结果: {len(df_v3)} 行")
except:
    print("[警告] 未找到v3预测结果文件")
    df_v3 = None

try:
    df_v4 = pd.read_csv('all_predictions_v4.csv', encoding='utf-8-sig')
    print(f"[OK] v4预测结果: {len(df_v4)} 行")
except:
    print("[警告] 未找到v4预测结果文件")
    df_v4 = None

if df_v3 is None and df_v4 is None:
    print("[错误] 没有找到任何预测结果文件")
    exit(1)

# 3. 计算误差统计
print("\n[3/4] 计算误差统计...")

def calculate_metrics(df):
    """计算预测指标"""
    df_with_actual = df[df['Actual_Employment'].notna()].copy()

    if len(df_with_actual) == 0:
        return None

    # RMSE
    rmse = np.sqrt(np.mean(df_with_actual['Prediction_Error'] ** 2))

    # MAPE (Mean Absolute Percentage Error)
    mape = np.mean(np.abs(df_with_actual['Relative_Error_%']))

    # 方向一致性
    actual_diff = df_with_actual.groupby('Job_Title')['Actual_Employment'].diff()
    pred_diff = df_with_actual.groupby('Job_Title')['Predicted_Employment'].diff()
    direction_match = np.mean((actual_diff * pred_diff) > 0)

    # 按职位统计
    job_metrics = df_with_actual.groupby('Job_Title').agg({
        'Relative_Error_%': ['mean', 'std', 'min', 'max']
    })

    return {
        'rmse': rmse,
        'mape': mape,
        'direction_match': direction_match,
        'job_metrics': job_metrics
    }

if df_v3 is not None:
    metrics_v3 = calculate_metrics(df_v3)
    if metrics_v3:
        print("\n[v3 指标]")
        print(f"  RMSE: {metrics_v3['rmse']:.2f}")
        print(f"  MAPE: {metrics_v3['mape']:.2f}%")
        print(f"  方向一致性: {metrics_v3['direction_match']*100:.2f}%")

if df_v4 is not None:
    metrics_v4 = calculate_metrics(df_v4)
    if metrics_v4:
        print("\n[v4 指标]")
        print(f"  RMSE: {metrics_v4['rmse']:.2f}")
        print(f"  MAPE: {metrics_v4['mape']:.2f}%")
        print(f"  方向一致性: {metrics_v4['direction_match']*100:.2f}%")

# 4. 生成对比图表
print("\n[4/4] 生成对比图表...")

if df_v3 is not None and df_v4 is not None:
    # 获取共同的职位
    jobs_v3 = set(df_v3['Job_Title'].unique())
    jobs_v4 = set(df_v4['Job_Title'].unique())
    common_jobs = list(jobs_v3.intersection(jobs_v4))

    if len(common_jobs) > 0:
        print(f"[INFO] 找到 {len(common_jobs)} 个共同职位")

        # 为每个职位创建对比图
        for job_title in common_jobs:
            fig, axes = plt.subplots(2, 1, figsize=(14, 10))

            # 获取该职位的数据
            df_job_v3 = df_v3[df_v3['Job_Title'] == job_title].sort_values('Quarter')
            df_job_v4 = df_v4[df_v4['Job_Title'] == job_title].sort_values('Quarter')

            # 获取真实值
            df_actual_v3 = df_job_v3[df_job_v3['Actual_Employment'].notna()]
            df_actual_v4 = df_job_v4[df_job_v4['Actual_Employment'].notna()]

            # 子图1: 预测值对比
            ax1 = axes[0]
            ax1.plot(df_job_v3['Quarter'], df_job_v3['Predicted_Employment'],
                    label='v3 Predicted', color='blue', linewidth=2, linestyle='-', marker='o', markersize=4)
            ax1.plot(df_job_v4['Quarter'], df_job_v4['Predicted_Employment'],
                    label='v4 Predicted', color='green', linewidth=2, linestyle='-', marker='s', markersize=4)

            if len(df_actual_v3) > 0:
                ax1.plot(df_actual_v3['Quarter'], df_actual_v3['Actual_Employment'],
                        label='Actual', color='red', linewidth=2, linestyle='--', marker='^', markersize=4)

            ax1.set_title(f'{job_title} - 预测值对比', fontsize=12, fontweight='bold')
            ax1.set_xlabel('Quarter (from 2020 Q1)', fontsize=10)
            ax1.set_ylabel('Employment', fontsize=10)
            ax1.legend(loc='best')
            ax1.grid(True, alpha=0.3)

            # 子图2: 误差对比
            ax2 = axes[1]
            if len(df_actual_v3) > 0:
                ax2.plot(df_actual_v3['Quarter'], df_actual_v3['Relative_Error_%'],
                        label='v3 Error', color='blue', linewidth=2, marker='o', markersize=4)
            if len(df_actual_v4) > 0:
                ax2.plot(df_actual_v4['Quarter'], df_actual_v4['Relative_Error_%'],
                        label='v4 Error', color='green', linewidth=2, marker='s', markersize=4)

            ax2.axhline(y=0, color='black', linestyle='--', linewidth=1)
            ax2.set_title(f'{job_title} - 相对误差对比', fontsize=12, fontweight='bold')
            ax2.set_xlabel('Quarter (from 2020 Q1)', fontsize=10)
            ax2.set_ylabel('Relative Error (%)', fontsize=10)
            ax2.legend(loc='best')
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            safe_filename = job_title.replace(' ', '_').replace('/', '_')
            plt.savefig(f'comparison_{safe_filename}.png', dpi=300, bbox_inches='tight')
            print(f"[OK] 已保存: comparison_{safe_filename}.png")
            plt.close()

        # 创建总体误差对比图
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # 子图1: MAPE对比
        ax1 = axes[0]
        if metrics_v3 and metrics_v4:
            job_names = []
            mape_v3 = []
            mape_v4 = []

            for job in common_jobs:
                job_names.append(job[:30])  # 截断长名称
                v3_mape = df_v3[df_v3['Job_Title'] == job]['Relative_Error_%'].abs().mean()
                v4_mape = df_v4[df_v4['Job_Title'] == job]['Relative_Error_%'].abs().mean()
                mape_v3.append(v3_mape)
                mape_v4.append(v4_mape)

            x = np.arange(len(job_names))
            width = 0.35

            ax1.bar(x - width/2, mape_v3, width, label='v3', color='blue', alpha=0.7)
            ax1.bar(x + width/2, mape_v4, width, label='v4', color='green', alpha=0.7)

            ax1.set_xlabel('Job Title', fontsize=11)
            ax1.set_ylabel('MAPE (%)', fontsize=11)
            ax1.set_title('各职位MAPE对比 (v3 vs v4)', fontsize=12, fontweight='bold')
            ax1.set_xticks(x)
            ax1.set_xticklabels(job_names, rotation=45, ha='right', fontsize=8)
            ax1.legend()
            ax1.grid(True, alpha=0.3, axis='y')

        # 子图2: 总体指标对比
        ax2 = axes[1]
        if metrics_v3 and metrics_v4:
            metrics_names = ['RMSE', 'MAPE (%)', 'Direction Match (%)']
            v3_values = [metrics_v3['rmse'], metrics_v3['mape'], metrics_v3['direction_match']*100]
            v4_values = [metrics_v4['rmse'], metrics_v4['mape'], metrics_v4['direction_match']*100]

            x = np.arange(len(metrics_names))
            width = 0.35

            ax2.bar(x - width/2, v3_values, width, label='v3', color='blue', alpha=0.7)
            ax2.bar(x + width/2, v4_values, width, label='v4', color='green', alpha=0.7)

            ax2.set_xlabel('Metric', fontsize=11)
            ax2.set_ylabel('Value', fontsize=11)
            ax2.set_title('总体指标对比 (v3 vs v4)', fontsize=12, fontweight='bold')
            ax2.set_xticks(x)
            ax2.set_xticklabels(metrics_names)
            ax2.legend()
            ax2.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig('overall_comparison_v3_v4.png', dpi=300, bbox_inches='tight')
        print(f"[OK] 已保存: overall_comparison_v3_v4.png")
        plt.close()

print("\n" + "=" * 60)
print("对比分析完成！")
print("=" * 60)