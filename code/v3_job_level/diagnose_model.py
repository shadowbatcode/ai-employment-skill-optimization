"""
模型诊断工具 - 判断问题是"优化层"还是"结构层"
基于专业反馈设计的4项核心检查
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
from parameter_optimizer_v5 import ParameterOptimizerV5
from time_series_predictor import TimeSeriesPredictor
sys.path.append('../v2_improved')
from ai_environment_v2 import AIEnvironmentV2

# 设置中文字体
rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
rcParams['axes.unicode_minus'] = False

print("=" * 80)
print("模型结构诊断工具")
print("检查项目:")
print("  1. 滚动回测 - 验证预测稳定性")
print("  2. 残差分析 - 检查系统性偏差")
print("  3. 参数稳定性 - 检查参数识别性")
print("  4. Baseline对比 - 与简单模型对比")
print("=" * 80)


def simple_baseline_forecast(actual_values, forecast_horizon):
    """
    简单baseline: 上一期值 + 平滑趋势
    """
    if len(actual_values) < 3:
        return np.full(forecast_horizon, actual_values[-1])

    # 计算最近3期的平均变化
    recent_changes = np.diff(actual_values[-4:])
    avg_change = np.mean(recent_changes)

    # 预测
    forecast = []
    last_value = actual_values[-1]
    for _ in range(forecast_horizon):
        next_value = last_value + avg_change
        forecast.append(next_value)
        last_value = next_value

    return np.array(forecast)


def diagnose_job(job_info, processor, ai_env):
    """
    对单个职位进行完整诊断
    """
    print("\n" + "=" * 80)
    print(f"诊断职位: {job_info['title']} ({job_info['type']})")
    print("=" * 80)

    # 获取数据
    train_data, test_data = processor.split_train_test(job_info['data'])

    if len(train_data) < 12:
        print("[跳过] 训练数据不足12个季度")
        return None

    print(f"[数据] 训练集: {len(train_data)} 季度, 测试集: {len(test_data)} 季度")

    # 提取完整数据
    actual_employment = job_info['data']['Third Month Employment'].values
    years = job_info['data']['Year'].values
    quarters = job_info['data']['Quarter'].values
    time_points = (years - 2020) * 4 + (quarters - 1)

    # 数据基本统计
    print(f"[统计] 均值: {np.mean(actual_employment):.0f}, "
          f"标准差: {np.std(actual_employment):.0f}, "
          f"变异系数: {np.std(actual_employment)/np.mean(actual_employment):.3f}")

    # 检查是否有接近0的值
    min_value = np.min(actual_employment)
    print(f"[统计] 最小值: {min_value:.0f}, 最大值: {np.max(actual_employment):.0f}")
    if min_value < np.mean(actual_employment) * 0.1:
        print("[警告] 存在低基数值，相对误差可能不稳定")

    # === 检查1: 滚动回测 ===
    print("\n[检查1/4] 滚动回测...")
    model = JobModel(job_info['data'], job_info['type'])
    optimizer = ParameterOptimizerV5(model, train_data, ai_env)

    # 执行滚动回测（窗口12季度，步长4季度）
    backtest_results = optimizer.rolling_backtest(window_size=12, step=4)

    if len(backtest_results) > 0:
        test_errors = [r['smape_test'] for r in backtest_results]
        print(f"  回测窗口数: {len(backtest_results)}")
        print(f"  测试集SMAPE: 均值={np.mean(test_errors):.4f}, "
              f"标准差={np.std(test_errors):.4f}")
        print(f"  SMAPE范围: [{np.min(test_errors):.4f}, {np.max(test_errors):.4f}]")

        if np.std(test_errors) > np.mean(test_errors) * 0.5:
            print("  [结论] ⚠️ 误差波动大，预测不稳定")
        else:
            print("  [结论] ✓ 误差相对稳定")

    # === 检查2: 残差分析 ===
    print("\n[检查2/4] 残差分析...")

    # 使用全部训练数据优化一次
    model_full = JobModel(job_info['data'], job_info['type'])
    optimizer_full = ParameterOptimizerV5(model_full, train_data, ai_env)
    params, loss = optimizer_full.optimize(maxiter=40, popsize=12)

    # 生成训练集预测
    predictor = TimeSeriesPredictor(model_full, ai_env)
    train_time = time_points[:len(train_data)]
    train_actual = actual_employment[:len(train_data)]

    _, train_pred = predictor.predict(
        start_time=train_time[0],
        end_time=train_time[-1] + 0.25,
        N0=train_actual[0]
    )
    train_pred = np.interp(train_time,
                           np.arange(train_time[0], train_time[-1] + 0.25, 0.25),
                           train_pred)

    # 计算残差
    residuals = train_pred - train_actual
    relative_residuals = residuals / (train_actual + np.mean(train_actual) * 0.01)

    print(f"  残差均值: {np.mean(residuals):.2f} (相对: {np.mean(relative_residuals):.4f})")
    print(f"  残差标准差: {np.std(residuals):.2f}")

    # 检查系统性偏差
    if abs(np.mean(relative_residuals)) > 0.05:
        if np.mean(relative_residuals) > 0:
            print("  [结论] ⚠️ 系统性高估（残差均值显著为正）")
        else:
            print("  [结论] ⚠️ 系统性低估（残差均值显著为负）")
    else:
        print("  [结论] ✓ 无明显系统性偏差")

    # 检查残差趋势
    residual_trend = np.polyfit(np.arange(len(residuals)), residuals, 1)[0]
    if abs(residual_trend) > np.std(residuals) * 0.1:
        print(f"  [结论] ⚠️ 残差存在趋势（斜率={residual_trend:.2f}），可能有结构性漂移")
    else:
        print("  [结论] ✓ 残差无明显趋势")

    # === 检查3: 参数稳定性 ===
    print("\n[检查3/4] 参数稳定性...")

    if len(backtest_results) > 1:
        params_history = np.array([r['params'] for r in backtest_results])
        theta_std = np.std(params_history[:, 0])
        Ac_std = np.std(params_history[:, 1])
        k_std = np.std(params_history[:, 2])

        print(f"  theta: 均值={np.mean(params_history[:, 0]):.4f}, "
              f"标准差={theta_std:.4f}")
        print(f"  Ac: 均值={np.mean(params_history[:, 1]):.4f}, "
              f"标准差={Ac_std:.4f}")
        print(f"  k: 均值={np.mean(params_history[:, 2]):.4f}, "
              f"标准差={k_std:.4f}")

        # 判断稳定性
        unstable_params = []
        if theta_std > 0.3:
            unstable_params.append('theta')
        if Ac_std > 0.2:
            unstable_params.append('Ac')
        if k_std > 3.0:
            unstable_params.append('k')

        if len(unstable_params) > 0:
            print(f"  [结论] ⚠️ 参数不稳定: {', '.join(unstable_params)}")
            print("         可能原因: 模型不可识别或数据不足以约束参数")
        else:
            print("  [结论] ✓ 参数相对稳定")

    # === 检查4: Baseline对比 ===
    print("\n[检查4/4] Baseline对比...")

    if len(test_data) > 0:
        test_actual = actual_employment[len(train_data):]
        test_time = time_points[len(train_data):]

        # 模型预测
        _, model_pred = predictor.predict(
            start_time=test_time[0],
            end_time=test_time[-1] + 0.25,
            N0=train_actual[-1]
        )
        model_pred = np.interp(test_time,
                              np.arange(test_time[0], test_time[-1] + 0.25, 0.25),
                              model_pred)

        # Baseline预测
        baseline_pred = simple_baseline_forecast(train_actual, len(test_actual))

        # 计算SMAPE
        epsilon = np.mean(test_actual) * 0.01
        model_smape = np.mean(2 * np.abs(model_pred - test_actual) /
                             (np.abs(model_pred) + np.abs(test_actual) + epsilon))
        baseline_smape = np.mean(2 * np.abs(baseline_pred - test_actual) /
                                (np.abs(baseline_pred) + np.abs(test_actual) + epsilon))

        print(f"  模型SMAPE: {model_smape:.4f}")
        print(f"  Baseline SMAPE: {baseline_smape:.4f}")
        print(f"  改进: {(baseline_smape - model_smape) / baseline_smape * 100:.2f}%")

        if model_smape < baseline_smape * 0.9:
            print("  [结论] ✓ 模型显著优于baseline")
        elif model_smape < baseline_smape:
            print("  [结论] ⚠️ 模型略优于baseline，改进有限")
        else:
            print("  [结论] ❌ 模型不如baseline，结构可能有问题")
    else:
        print("  [跳过] 无测试集数据")

    # === 综合诊断 ===
    print("\n" + "=" * 80)
    print("综合诊断结论:")
    print("=" * 80)

    # 判断问题类型
    issues = []

    if len(backtest_results) > 0 and np.std(test_errors) > np.mean(test_errors) * 0.5:
        issues.append("预测不稳定（滚动回测误差波动大）")

    if abs(np.mean(relative_residuals)) > 0.05 or abs(residual_trend) > np.std(residuals) * 0.1:
        issues.append("系统性偏差或漂移（残差有趋势）")

    if len(backtest_results) > 1 and len(unstable_params) > 0:
        issues.append(f"参数不稳定（{', '.join(unstable_params)}）")

    if len(test_data) > 0 and model_smape >= baseline_smape:
        issues.append("不如简单baseline")

    if len(issues) == 0:
        print("✓ 模型表现良好，优化层改进可能有效")
    else:
        print("⚠️ 发现以下问题（可能是结构层问题）:")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")

        print("\n建议:")
        if "系统性偏差或漂移" in str(issues):
            print("  - 考虑分段建模（断点前后不同参数）")
            print("  - 检查数据口径是否有变化")
        if "参数不稳定" in str(issues):
            print("  - 模型可能过于复杂，参数不可识别")
            print("  - 考虑简化模型或增加约束")
        if "不如简单baseline" in str(issues):
            print("  - 当前模型结构可能不适合该职位")
            print("  - 考虑使用更简单的时间序列模型")

    # 生成诊断图表
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 子图1: 拟合效果
    ax1 = axes[0, 0]
    ax1.plot(train_time, train_actual, 'o-', label='Actual', linewidth=2, markersize=5)
    ax1.plot(train_time, train_pred, 's--', label='Predicted', linewidth=2, markersize=4)
    ax1.set_title('训练集拟合效果', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Quarter', fontsize=10)
    ax1.set_ylabel('Employment', fontsize=10)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 子图2: 残差图
    ax2 = axes[0, 1]
    ax2.plot(train_time, residuals, 'o-', linewidth=2, markersize=5)
    ax2.axhline(y=0, color='red', linestyle='--', linewidth=1)
    ax2.axhline(y=np.mean(residuals), color='green', linestyle=':', linewidth=2,
                label=f'Mean={np.mean(residuals):.0f}')
    ax2.set_title('残差分析', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Quarter', fontsize=10)
    ax2.set_ylabel('Residual', fontsize=10)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 子图3: 参数稳定性
    ax3 = axes[1, 0]
    if len(backtest_results) > 1:
        windows = [f"W{i+1}" for i in range(len(backtest_results))]
        ax3.plot(windows, params_history[:, 0], 'o-', label='theta', linewidth=2, markersize=6)
        ax3_twin = ax3.twinx()
        ax3_twin.plot(windows, params_history[:, 1], 's-', label='Ac',
                     color='orange', linewidth=2, markersize=6)
        ax3.set_title('参数稳定性（滚动窗口）', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Window', fontsize=10)
        ax3.set_ylabel('theta', fontsize=10)
        ax3_twin.set_ylabel('Ac', fontsize=10)
        ax3.legend(loc='upper left')
        ax3_twin.legend(loc='upper right')
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, '数据不足\n无法进行滚动回测',
                ha='center', va='center', fontsize=14)
        ax3.set_xlim(0, 1)
        ax3.set_ylim(0, 1)

    # 子图4: Baseline对比
    ax4 = axes[1, 1]
    if len(test_data) > 0:
        ax4.plot(test_time, test_actual, 'o-', label='Actual', linewidth=2, markersize=6)
        ax4.plot(test_time, model_pred, 's--', label='Model', linewidth=2, markersize=5)
        ax4.plot(test_time, baseline_pred, '^:', label='Baseline', linewidth=2, markersize=5)
        ax4.set_title(f'测试集预测对比\n(Model SMAPE={model_smape:.4f}, Baseline={baseline_smape:.4f})',
                     fontsize=12, fontweight='bold')
        ax4.set_xlabel('Quarter', fontsize=10)
        ax4.set_ylabel('Employment', fontsize=10)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(0.5, 0.5, '无测试集数据', ha='center', va='center', fontsize=14)
        ax4.set_xlim(0, 1)
        ax4.set_ylim(0, 1)

    plt.tight_layout()
    safe_filename = job_info['title'].replace(' ', '_').replace('/', '_')
    plt.savefig(f'diagnosis_{safe_filename}.png', dpi=300, bbox_inches='tight')
    print(f"\n[图表] 已保存: diagnosis_{safe_filename}.png")
    plt.close()

    return {
        'title': job_info['title'],
        'type': job_info['type'],
        'params': params,
        'train_loss': loss,
        'backtest_results': backtest_results,
        'residual_mean': np.mean(residuals),
        'residual_std': np.std(residuals),
        'residual_trend': residual_trend,
        'issues': issues
    }


def main():
    """主函数"""
    print("\n[初始化] 加载数据...")
    data_file = r"../程序所需数据.xlsx"
    processor = DataProcessor(data_file)

    print("[初始化] 初始化AI环境...")
    df_events = processor.get_ai_events()
    ai_env = AIEnvironmentV2(df_events)

    print("[初始化] 获取职位列表...")
    all_jobs = processor.get_all_jobs()
    print(f"[OK] 找到 {len(all_jobs)} 个职位\n")

    # 选择要诊断的职位（可以修改这里选择不同职位）
    print("可用职位:")
    for i, job in enumerate(all_jobs[:10], 1):  # 只显示前10个
        print(f"  {i}. {job['title']} ({job['type']})")

    print("\n[提示] 将诊断第1个职位，如需诊断其他职位请修改代码")

    # 诊断第一个职位
    if len(all_jobs) > 0:
        result = diagnose_job(all_jobs[0], processor, ai_env)

    print("\n" + "=" * 80)
    print("诊断完成！")
    print("=" * 80)


if __name__ == "__main__":
    main()