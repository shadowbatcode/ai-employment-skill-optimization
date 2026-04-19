"""
测试v4优化器 - 单职位测试
用于快速验证v4优化器的改进效果
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append('..')

from data_processor import DataProcessor
from job_model import JobModel
from parameter_optimizer_v3 import ParameterOptimizerV3
from parameter_optimizer_v4 import ParameterOptimizerV4
from time_series_predictor import TimeSeriesPredictor
sys.path.append('../v2_improved')
from ai_environment_v2 import AIEnvironmentV2

# 设置中文字体
rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("v4优化器单职位测试")
print("=" * 60)

# 1. 加载数据
print("\n[1/5] 加载数据...")
data_file = r"../程序所需数据.xlsx"
processor = DataProcessor(data_file)

# 2. 初始化AI环境
print("\n[2/5] 初始化AI环境...")
df_events = processor.get_ai_events()
ai_env = AIEnvironmentV2(df_events)
print(f"[OK] AI环境已初始化")

# 3. 选择一个职位进行测试
print("\n[3/5] 选择测试职位...")
all_jobs = processor.get_all_jobs()

# 选择第一个有足够数据的职位
test_job = None
for job in all_jobs:
    train_data, test_data = processor.split_train_test(job['data'])
    if len(train_data) >= 10:  # 至少10个训练点
        test_job = job
        break

if test_job is None:
    print("[错误] 没有找到合适的测试职位")
    exit(1)

print(f"[OK] 测试职位: {test_job['title']} ({test_job['type']})")
print(f"[INFO] 训练集大小: {len(train_data)}, 测试集大小: {len(test_data)}")

# 4. 分别使用v3和v4优化器
print("\n[4/5] 开始优化对比...")

# 创建两个独立的模型实例
model_v3 = JobModel(test_job['data'], test_job['type'])
model_v4 = JobModel(test_job['data'], test_job['type'])

train_data, test_data = processor.split_train_test(test_job['data'])

# v3优化
print("\n--- v3优化器 ---")
optimizer_v3 = ParameterOptimizerV3(model_v3, train_data, ai_env)
params_v3, error_v3 = optimizer_v3.optimize(maxiter=30, popsize=8)
print(f"[v3] theta={params_v3[0]:.4f}, Ac={params_v3[1]:.4f}, k={params_v3[2]:.4f}, error={error_v3:.6f}")

# v4优化
print("\n--- v4优化器 ---")
optimizer_v4 = ParameterOptimizerV4(model_v4, train_data, ai_env)
params_v4, error_v4 = optimizer_v4.optimize(maxiter=50, popsize=15)
print(f"[v4] theta={params_v4[0]:.4f}, Ac={params_v4[1]:.4f}, k={params_v4[2]:.4f}, loss={error_v4:.6f}")

# 5. 生成预测并对比
print("\n[5/5] 生成预测对比...")

# v3预测
predictor_v3 = TimeSeriesPredictor(model_v3, ai_env)
last_employment = train_data['Third Month Employment'].iloc[-1]
t_pred_v3, N_pred_v3 = predictor_v3.predict(start_time=0, end_time=40, N0=last_employment)

# v4预测
predictor_v4 = TimeSeriesPredictor(model_v4, ai_env)
t_pred_v4, N_pred_v4 = predictor_v4.predict(start_time=0, end_time=40, N0=last_employment)

# 获取真实值
actual_employment = test_job['data']['Third Month Employment'].values
years = test_job['data']['Year'].values
quarters = test_job['data']['Quarter'].values
time_points = (years - 2020) * 4 + (quarters - 1)

# 计算训练集和测试集的误差
train_size = len(train_data)
train_actual = actual_employment[:train_size]
train_time = time_points[:train_size]

# 从预测结果中提取对应时间点的值
train_pred_v3 = np.interp(train_time, t_pred_v3, N_pred_v3)
train_pred_v4 = np.interp(train_time, t_pred_v4, N_pred_v4)

# 计算训练集误差
train_rmse_v3 = np.sqrt(np.mean(((train_pred_v3 - train_actual) / train_actual) ** 2))
train_rmse_v4 = np.sqrt(np.mean(((train_pred_v4 - train_actual) / train_actual) ** 2))

print(f"\n[训练集RMSE]")
print(f"  v3: {train_rmse_v3:.6f}")
print(f"  v4: {train_rmse_v4:.6f}")
print(f"  改进: {(train_rmse_v3 - train_rmse_v4) / train_rmse_v3 * 100:.2f}%")

# 如果有测试集,计算测试集误差
if len(test_data) > 0:
    test_actual = actual_employment[train_size:]
    test_time = time_points[train_size:]

    test_pred_v3 = np.interp(test_time, t_pred_v3, N_pred_v3)
    test_pred_v4 = np.interp(test_time, t_pred_v4, N_pred_v4)

    test_rmse_v3 = np.sqrt(np.mean(((test_pred_v3 - test_actual) / test_actual) ** 2))
    test_rmse_v4 = np.sqrt(np.mean(((test_pred_v4 - test_actual) / test_actual) ** 2))

    print(f"\n[测试集RMSE]")
    print(f"  v3: {test_rmse_v3:.6f}")
    print(f"  v4: {test_rmse_v4:.6f}")
    print(f"  改进: {(test_rmse_v3 - test_rmse_v4) / test_rmse_v3 * 100:.2f}%")

# 6. 可视化对比
print("\n[6/6] 生成可视化图表...")

fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# 子图1: 预测值对比
ax1 = axes[0]
ax1.plot(time_points, actual_employment, 'ro-', label='Actual', linewidth=2, markersize=6)
ax1.plot(t_pred_v3, N_pred_v3, 'b--', label='v3 Predicted', linewidth=2, alpha=0.7)
ax1.plot(t_pred_v4, N_pred_v4, 'g-', label='v4 Predicted', linewidth=2, alpha=0.7)

# 标记训练集和测试集分界线
if len(test_data) > 0:
    ax1.axvline(x=train_time[-1], color='gray', linestyle=':', linewidth=2, label='Train/Test Split')

ax1.set_title(f'{test_job["title"]} - 预测对比 (v3 vs v4)', fontsize=14, fontweight='bold')
ax1.set_xlabel('Quarter (from 2020 Q1)', fontsize=11)
ax1.set_ylabel('Employment', fontsize=11)
ax1.legend(loc='best')
ax1.grid(True, alpha=0.3)

# 子图2: 相对误差对比
ax2 = axes[1]
train_error_v3 = (train_pred_v3 - train_actual) / train_actual * 100
train_error_v4 = (train_pred_v4 - train_actual) / train_actual * 100

ax2.plot(train_time, train_error_v3, 'bo-', label='v3 Error', linewidth=2, markersize=5)
ax2.plot(train_time, train_error_v4, 'go-', label='v4 Error', linewidth=2, markersize=5)

if len(test_data) > 0:
    test_error_v3 = (test_pred_v3 - test_actual) / test_actual * 100
    test_error_v4 = (test_pred_v4 - test_actual) / test_actual * 100
    ax2.plot(test_time, test_error_v3, 'bs--', label='v3 Test Error', linewidth=2, markersize=5)
    ax2.plot(test_time, test_error_v4, 'gs--', label='v4 Test Error', linewidth=2, markersize=5)
    ax2.axvline(x=train_time[-1], color='gray', linestyle=':', linewidth=2)

ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
ax2.set_title('相对误差对比 (%)', fontsize=14, fontweight='bold')
ax2.set_xlabel('Quarter (from 2020 Q1)', fontsize=11)
ax2.set_ylabel('Relative Error (%)', fontsize=11)
ax2.legend(loc='best')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('test_v4_single_job.png', dpi=300, bbox_inches='tight')
print(f"[OK] 图表已保存: test_v4_single_job.png")
plt.close()

print("\n" + "=" * 60)
print("测试完成！")
print("=" * 60)