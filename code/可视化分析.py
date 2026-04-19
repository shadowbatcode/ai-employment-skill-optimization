#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
可视化结果分析：三类职业的最优方案对比
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import sys
import io

# 修复Windows编码问题
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

def load_results():
    """加载优化结果"""
    with open('optimization_results.json', 'r', encoding='utf-8') as f:
        return json.load(f)

def plot_decision_variables():
    """绘制决策变量对比图"""
    results = load_results()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    categories = ['STEM', 'Trade', 'Arts']
    colors = ['#FF6B6B', '#4ECDC4', '#95E1D3']

    for idx, (cat, ax, color) in enumerate(zip(categories, axes, colors)):
        x_star = results[cat]['optimal_solution']['x_star']
        dims = results[cat]['optimal_solution']['decision_dims']

        # 绘制条形图
        bars = ax.barh(range(len(x_star)), x_star, color=color, alpha=0.8, edgecolor='black')

        # 添加数值标签
        for i, (bar, val) in enumerate(zip(bars, x_star)):
            if val > 0.01:
                ax.text(val + 0.02, i, f'{val:.3f}', va='center', fontsize=10)

        ax.set_yticks(range(len(dims)))
        ax.set_yticklabels(dims, fontsize=10)
        ax.set_xlabel('投入比例', fontsize=12)
        ax.set_title(f'{cat} 类职业最优投入方案', fontsize=14, fontweight='bold')
        ax.set_xlim(0, 1.1)
        ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig('决策变量对比图.png', dpi=300, bbox_inches='tight')
    print("[完成] 决策变量对比图已保存")
    plt.close()

def plot_skill_comparison():
    """绘制技能向量对比图"""
    results = load_results()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    categories = ['STEM', 'Trade', 'Arts']
    colors = ['#FF6B6B', '#4ECDC4', '#95E1D3']

    for idx, (cat, ax, color) in enumerate(zip(categories, axes, colors)):
        G2025 = np.array(results[cat]['input']['G2025'])
        Target = np.array(results[cat]['input']['Target'])
        G2026 = np.array(results[cat]['optimal_solution']['G2026'])
        dims = results[cat]['optimal_solution']['decision_dims']

        x = np.arange(len(dims))
        width = 0.25

        # 绘制三组条形
        ax.bar(x - width, G2025, width, label='当前(2025)', color='#B0BEC5', alpha=0.8)
        ax.bar(x, G2026, width, label='预测(2026)', color=color, alpha=0.9)
        ax.bar(x + width, Target, width, label='目标(2026)', color='#FF5722', alpha=0.7, edgecolor='red', linewidth=2)

        ax.set_xticks(x)
        ax.set_xticklabels(dims, rotation=15, ha='right', fontsize=9)
        ax.set_ylabel('技能水平', fontsize=11)
        ax.set_title(f'{cat} - 技能向量对比', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.set_ylim(0.5, 1.0)
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('技能向量对比图.png', dpi=300, bbox_inches='tight')
    print("[完成] 技能向量对比图已保存")
    plt.close()

def plot_optimization_metrics():
    """绘制优化指标对比图"""
    results = load_results()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    categories = ['STEM', 'Trade', 'Arts']
    colors = ['#FF6B6B', '#4ECDC4', '#95E1D3']

    # MSE损失
    mse_values = [results[cat]['metrics']['objective_value'] for cat in categories]
    axes[0, 0].bar(categories, mse_values, color=colors, alpha=0.8, edgecolor='black')
    axes[0, 0].set_ylabel('MSE损失', fontsize=12)
    axes[0, 0].set_title('目标函数值 (MSE)', fontsize=14, fontweight='bold')
    axes[0, 0].grid(axis='y', alpha=0.3)
    for i, v in enumerate(mse_values):
        axes[0, 0].text(i, v + 0.001, f'{v:.4f}', ha='center', fontsize=10)

    # RMSE
    rmse_values = [results[cat]['metrics']['rmse'] for cat in categories]
    axes[0, 1].bar(categories, rmse_values, color=colors, alpha=0.8, edgecolor='black')
    axes[0, 1].set_ylabel('RMSE', fontsize=12)
    axes[0, 1].set_title('均方根误差 (RMSE)', fontsize=14, fontweight='bold')
    axes[0, 1].grid(axis='y', alpha=0.3)
    for i, v in enumerate(rmse_values):
        axes[0, 1].text(i, v + 0.005, f'{v:.4f}', ha='center', fontsize=10)

    # 投入预算
    budget_values = [results[cat]['metrics']['total_allocation'] for cat in categories]
    axes[1, 0].bar(categories, budget_values, color=colors, alpha=0.8, edgecolor='black')
    axes[1, 0].set_ylabel('投入预算', fontsize=12)
    axes[1, 0].set_title('总投入预算对比', fontsize=14, fontweight='bold')
    axes[1, 0].grid(axis='y', alpha=0.3)
    for i, v in enumerate(budget_values):
        axes[1, 0].text(i, v + 0.05, f'{v:.3f}', ha='center', fontsize=10)

    # 投入效率（RMSE/预算）
    efficiency = [rmse_values[i] / budget_values[i] for i in range(3)]
    axes[1, 1].bar(categories, efficiency, color=colors, alpha=0.8, edgecolor='black')
    axes[1, 1].set_ylabel('效率指标', fontsize=12)
    axes[1, 1].set_title('投入效率 (RMSE/预算，越低越好)', fontsize=14, fontweight='bold')
    axes[1, 1].grid(axis='y', alpha=0.3)
    for i, v in enumerate(efficiency):
        axes[1, 1].text(i, v + 0.005, f'{v:.3f}', ha='center', fontsize=10)

    # 标记最优类别
    best_idx = np.argmin(efficiency)
    axes[1, 1].get_children()[best_idx].set_edgecolor('gold')
    axes[1, 1].get_children()[best_idx].set_linewidth(4)

    plt.tight_layout()
    plt.savefig('优化指标对比图.png', dpi=300, bbox_inches='tight')
    print("[完成] 优化指标对比图已保存")
    plt.close()

def plot_improvements():
    """绘制各维度改进量图"""
    results = load_results()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    categories = ['STEM', 'Trade', 'Arts']
    colors_positive = ['#4CAF50', '#8BC34A', '#CDDC39']
    colors_negative = ['#F44336', '#FF5722', '#FF9800']

    for idx, (cat, ax) in enumerate(zip(categories, axes)):
        improvements = np.array(results[cat]['detailed_analysis']['improvements'])
        dims = results[cat]['optimal_solution']['decision_dims']

        # 根据正负设置颜色
        bar_colors = [colors_positive[idx] if imp >= 0 else colors_negative[idx] for imp in improvements]

        bars = ax.barh(range(len(improvements)), improvements, color=bar_colors, alpha=0.8, edgecolor='black')

        # 添加数值标签
        for i, (bar, val) in enumerate(zip(bars, improvements)):
            offset = 0.005 if val >= 0 else -0.005
            align = 'left' if val >= 0 else 'right'
            ax.text(val + offset, i, f'{val:+.4f}', va='center', ha=align, fontsize=9)

        ax.set_yticks(range(len(dims)))
        ax.set_yticklabels(dims, fontsize=10)
        ax.set_xlabel('技能改进度', fontsize=12)
        ax.set_title(f'{cat} - 各维度技能改进', fontsize=14, fontweight='bold')
        ax.axvline(x=0, color='black', linestyle='--', linewidth=1)
        ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig('技能改进度对比图.png', dpi=300, bbox_inches='tight')
    print("[完成] 技能改进度对比图已保存")
    plt.close()

def plot_radar_chart():
    """绘制雷达图（技能轮廓对比）"""
    results = load_results()

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), subplot_kw=dict(projection='polar'))

    categories = ['STEM', 'Trade', 'Arts']

    for idx, (cat, ax) in enumerate(zip(categories, axes)):
        G2025 = np.array(results[cat]['input']['G2025'])
        Target = np.array(results[cat]['input']['Target'])
        G2026 = np.array(results[cat]['optimal_solution']['G2026'])
        dims = results[cat]['optimal_solution']['decision_dims']

        # 角度
        num_vars = len(dims)
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

        # 闭合图形
        G2025 = np.concatenate((G2025, [G2025[0]]))
        G2026 = np.concatenate((G2026, [G2026[0]]))
        Target = np.concatenate((Target, [Target[0]]))
        angles += angles[:1]

        # 绘制三条线
        ax.plot(angles, G2025, 'o-', linewidth=2, label='当前(2025)', color='#B0BEC5')
        ax.fill(angles, G2025, alpha=0.15, color='#B0BEC5')

        ax.plot(angles, G2026, 's-', linewidth=2.5, label='预测(2026)', color='#4ECDC4')
        ax.fill(angles, G2026, alpha=0.25, color='#4ECDC4')

        ax.plot(angles, Target, '^-', linewidth=2, label='目标(2026)', color='#FF5722', linestyle='--')

        # 设置标签
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(dims, fontsize=9)
        ax.set_ylim(0.5, 1.0)
        ax.set_title(f'{cat} 技能轮廓', fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=9)
        ax.grid(True)

    plt.tight_layout()
    plt.savefig('技能雷达图.png', dpi=300, bbox_inches='tight')
    print("[完成] 技能雷达图已保存")
    plt.close()

def main():
    """主函数"""
    print("\n" + "="*60)
    print("开始生成可视化分析图表...")
    print("="*60 + "\n")

    plot_decision_variables()
    plot_skill_comparison()
    plot_optimization_metrics()
    plot_improvements()
    plot_radar_chart()

    print("\n" + "="*60)
    print("所有图表已生成完成！")
    print("="*60)
    print("\n生成的文件列表：")
    print("  1. 决策变量对比图.png")
    print("  2. 技能向量对比图.png")
    print("  3. 优化指标对比图.png")
    print("  4. 技能改进度对比图.png")
    print("  5. 技能雷达图.png")

if __name__ == '__main__':
    main()
