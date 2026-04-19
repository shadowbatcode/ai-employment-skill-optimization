#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
参数反向优化工具 v2.0：支持5维和9维模型、多权重评估
通过调整E矩阵和T向量，使最优投入方案接近目标分布
"""

import numpy as np
from scipy.optimize import minimize, lsq_linear
import json
from datetime import datetime
import sys
import io
import os

# 修复Windows编码问题（安全版本）
if sys.platform == 'win32' and not hasattr(sys.stdout, '_wrapper_set'):
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys.stdout._wrapper_set = True  # 标记已设置
    except (AttributeError, ValueError):
        pass  # 如果已设置或失败，跳过

# 导入扩展优化器
try:
    from 优化求解器 import ExtendedLarmarckOptimizer, create_extended_data_pack, create_improved_data_pack
    HAS_OPTIMIZER = True
except ImportError:
    print("警告：无法导入优化求解器模块，将使用独立模式")
    HAS_OPTIMIZER = False


class ParameterTunerV2:
    """参数反向优化器 v2.0（支持多维度和多权重）"""

    def __init__(self, base_data, target_x_dict, skill_weights=None):
        """
        初始化调参器

        Args:
            base_data: 原始数据包
            target_x_dict: 目标投入分布 {'STEM': [a1, a2, ...], 'Trade': [...], ...}
            skill_weights: 技能权重 {'technical': w0, 'leadership': w1, 'sustainability': w2}
        """
        self.base_data = base_data
        self.target_x_dict = target_x_dict
        self.skill_weights = skill_weights or {}

    def compute_optimal_x(self, data_dict, skill_weights=None):
        """计算给定数据下的最优投入方案"""
        G2025 = np.array(data_dict['G2025'], dtype=float)
        Target = np.array(data_dict['Target'], dtype=float)
        Lambda_diag = np.array(data_dict['Lambda_diag_avg_growth'], dtype=float)
        E = np.array(data_dict['E'], dtype=float)

        Lambda = np.diag(Lambda_diag)
        A = Lambda @ E
        b = Target - G2025

        if skill_weights is None or not skill_weights:
            # 无权重求解
            result = lsq_linear(A, b, bounds=(0, 1), verbose=0)
            return result.x
        else:
            # 加权求解
            def obj_func(x):
                G2026 = G2025 + A @ x
                error = G2026 - Target
                weighted_error = 0.0
                m = len(error)
                for i in range(m):
                    # 简化分类：前1/3技术、中1/3领导力、后1/3可持续
                    if i < m // 3:
                        w = skill_weights.get('technical', 1.0)
                    elif i < 2 * m // 3:
                        w = skill_weights.get('leadership', 1.0)
                    else:
                        w = skill_weights.get('sustainability', 1.0)
                    weighted_error += w * (error[i] ** 2)
                return weighted_error

            result = minimize(
                obj_func,
                np.ones(E.shape[1]) * 0.2,
                method='SLSQP',
                bounds=[(0, 1) for _ in range(E.shape[1])],
                constraints=[{'type': 'ineq', 'fun': lambda x: 1.0 - np.sum(x)}]
            )
            return result.x

    def objective_function(self, params, category, original_E, original_T):
        """
        目标函数：当前参数下的投入分布与目标分布的差异

        Args:
            params: 参数向量（E矩阵和T向量的扁平化版本）
            category: 职业类别名称
            original_E: 原始E矩阵
            original_T: 原始T向量

        Returns:
            目标分布与实际分布的L2距离
        """
        # 自适应维度
        m = original_E.shape[0]  # 技能维度数
        p = original_E.shape[1]  # 投入项目数

        # 参数向量：前m*p个是E矩阵参数，后m个是T向量调整
        E_flat = params[:m*p]
        T_adjust = params[m*p:m*p+m]

        # 恢复E矩阵
        E = E_flat.reshape(m, p)

        # 调整T向量（限制调整幅度）
        T = original_T + T_adjust * 0.1

        # 创建调整后的数据字典
        adjusted_data = {
            'G2025': self.base_data[category]['G2025'],
            'Target': T.tolist(),
            'Lambda_diag_avg_growth': self.base_data[category]['Lambda_diag_avg_growth'],
            'E': E.tolist(),
            'dims': self.base_data[category]['dims']
        }

        # 计算最优投入
        x_opt = self.compute_optimal_x(adjusted_data, self.skill_weights)

        # 计算与目标分布的差异
        target_x = np.array(self.target_x_dict[category])
        error = np.sum((x_opt - target_x) ** 2)

        return error

    def tune_parameters(self, category, learning_rate=0.01, max_iterations=100):
        """
        调整参数使投入分布接近目标

        Args:
            category: 职业类别
            learning_rate: 学习率（已废弃）
            max_iterations: 最大迭代次数

        Returns:
            调整后的E矩阵、T向量和误差
        """
        original_E = np.array(self.base_data[category]['E'], dtype=float)
        original_T = np.array(self.base_data[category]['Target'], dtype=float)

        m = original_E.shape[0]  # 技能维度数
        p = original_E.shape[1]  # 投入项目数

        # 初始参数：E矩阵的扁平化 + T向量的零向量
        params0 = np.concatenate([original_E.flatten(), np.zeros(m)])

        print(f"\n调参中: {category} ({max_iterations} 次迭代)...")
        print(f"  模型维度: {m}维技能 × {p}个投入项目")

        # 使用SLSQP优化
        result = minimize(
            self.objective_function,
            params0,
            args=(category, original_E, original_T),
            method='SLSQP',
            options={'maxiter': max_iterations, 'ftol': 1e-6}
        )

        # 提取最优参数
        E_flat = result.x[:m*p]
        T_adjust = result.x[m*p:m*p+m]

        E_tuned = E_flat.reshape(m, p)
        T_tuned = original_T + T_adjust * 0.1

        return E_tuned, T_tuned, result.fun


def generate_tuned_data_5d():
    """生成5维模型的调参数据"""
    # 原始5维数据
    base_data = {
        "STEM": {
            "dims": ["编程能力", "AI工具使用", "系统架构", "沟通协作", "AI伦理安全"],
            "G2025": [0.75, 0.90, 0.82, 0.75, 0.60],
            "Target": [0.60, 0.97, 0.85, 0.76, 0.70],
            "Lambda_diag_avg_growth": [-0.0317307692, 0.0960784314, 0.0227804165, 0.0136060718, 0.225],
            "E": [
                [0.10, 0.00, 0.30, 0.20, 0.40],
                [0.70, 0.10, 0.50, 0.40, 0.30],
                [0.30, 0.20, 0.80, 0.60, 0.50],
                [0.20, 0.30, 0.70, 0.90, 0.10],
                [0.40, 0.80, 0.40, 0.70, 0.20]
            ]
        },
        "Trade": {
            "dims": ["传统烹饪技能", "现代厨房科技", "AI辅助菜单", "营养与健康", "多感官体验"],
            "G2025": [0.74, 0.86, 0.72, 0.79, 0.67],
            "Target": [0.72, 0.90, 0.80, 0.77, 0.72],
            "Lambda_diag_avg_growth": [-0.0258974359, 0.1164994426, 0.1447552448, 0.0264102564, 0.0757294430],
            "E": [
                [0.60, 0.10, 0.00, 0.00, 0.80],
                [0.80, 0.40, 0.10, 0.20, 0.50],
                [0.30, 0.90, 0.20, 0.30, 0.60],
                [0.10, 0.30, 0.85, 0.40, 0.30],
                [0.20, 0.40, 0.30, 0.95, 0.70]
            ]
        },
        "Arts": {
            "dims": ["传统表演技巧", "AI协作能力", "艺术批判思维", "即兴创造能力", "表演技术素养"],
            "G2025": [0.67, 0.78, 0.95, 0.87, 0.88],
            "Target": [0.72, 0.90, 0.90, 0.95, 0.90],
            "Lambda_diag_avg_growth": [-0.0547619048, 0.1935064935, 0.0274154589, 0.0429878049, 0.2116666667],
            "E": [
                [0.10, 0.05, 0.20, 0.85, 0.15],
                [0.85, 0.75, 0.80, 0.05, 0.20],
                [0.20, 0.60, 0.40, 0.50, 0.95],
                [0.30, 0.70, 0.65, 0.40, 0.25],
                [0.95, 0.80, 0.75, 0.05, 0.35]
            ]
        }
    }

    # 目标投入分布
    target_x_dict = {
        "STEM": [0.30, 0.10, 0.25, 0.10, 0.25],
        "Trade": [0.28, 0.22, 0.18, 0.17, 0.15],
        "Arts": [0.25, 0.20, 0.15, 0.18, 0.22]
    }

    tuner = ParameterTunerV2(base_data, target_x_dict)

    print("\n" + "="*80)
    print(" "*20 + "参数反向优化 v2.0：5维模型")
    print("="*80)
    print(f"\n执行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n目标：使最优投入方案 x* 接近指定的目标分布\n")

    tuned_data = {}

    for category in ['STEM', 'Trade', 'Arts']:
        print(f"处理 {category} 类职业...")
        print(f"  目标投入: {target_x_dict[category]}")

        E_tuned, T_tuned, error = tuner.tune_parameters(category, max_iterations=200)

        print(f"  调参完成，优化误差: {error:.6f}")
        print(f"  调整后的T向量: {np.round(T_tuned, 4)}")

        tuned_data[category] = {
            'dims': base_data[category]['dims'],
            'G2025': base_data[category]['G2025'],
            'Target': T_tuned.tolist(),
            'Lambda_diag_avg_growth': base_data[category]['Lambda_diag_avg_growth'],
            'E': E_tuned.tolist()
        }

    # 验证调参效果
    print("\n" + "="*80)
    print("【调参验证】重新优化后的投入方案")
    print("="*80 + "\n")

    print(f"{'职业类别':<12} {'目标投入':<35} {'实际投入':<35} {'误差'}")
    print("-" * 90)

    for category in ['STEM', 'Trade', 'Arts']:
        target_x = np.array(target_x_dict[category])
        tuner_verify = ParameterTunerV2(tuned_data, target_x_dict)
        actual_x = tuner_verify.compute_optimal_x(tuned_data[category])

        target_str = ', '.join([f'{x:.2f}' for x in target_x])
        actual_str = ', '.join([f'{x:.2f}' for x in actual_x])
        error = np.sum((actual_x - target_x) ** 2)

        print(f"{category:<12} [{target_str}] [{actual_str}] {error:.6f}")

    print("\n" + "="*80 + "\n")

    # 保存调参后的数据
    with open('tuned_parameters_5d.json', 'w', encoding='utf-8') as f:
        json.dump(tuned_data, f, ensure_ascii=False, indent=2)

    print("✓ 5维模型调参数据已保存到 tuned_parameters_5d.json\n")

    return tuned_data


def generate_tuned_data_9d():
    """生成9维模型的调参数据（使用多权重评估）"""
    if not HAS_OPTIMIZER:
        print("错误：需要导入优化求解器模块才能使用9维模型")
        return None

    # 从优化求解器获取9维数据
    base_data_full = create_extended_data_pack()

    # 移除skill_map（调参器不需要）
    base_data = {}
    for category_name, data in base_data_full.items():
        base_data[category_name.split('_')[-1] if '_' in category_name else category_name] = {
            'dims': data['dims'],
            'G2025': data['G2025'],
            'Target': data['Target'],
            'Lambda_diag_avg_growth': data['Lambda_diag_avg_growth'],
            'E': data['E']
        }

    # 9维模型的目标投入分布（5个投入项目）
    target_x_dict = {
        "DataScience": [0.20, 0.15, 0.25, 0.25, 0.15],  # STEM数据科学
        "Culinary": [0.15, 0.20, 0.25, 0.20, 0.20],      # Trade烹饪
        "Drama": [0.18, 0.22, 0.20, 0.25, 0.15]          # Arts戏剧
    }

    # 综合发展权重方案
    skill_weights = {
        'technical': 0.4,
        'leadership': 0.3,
        'sustainability': 0.3
    }

    tuner = ParameterTunerV2(base_data, target_x_dict, skill_weights=skill_weights)

    print("\n" + "="*80)
    print(" "*15 + "参数反向优化 v2.0：9维模型（多权重评估）")
    print("="*80)
    print(f"\n执行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"权重方案: 技术={skill_weights['technical']}, 领导力={skill_weights['leadership']}, 可持续={skill_weights['sustainability']}")
    print("\n目标：使最优投入方案 x* 接近指定的目标分布\n")

    tuned_data = {}

    for category in list(target_x_dict.keys()):
        print(f"处理 {category} 类职业...")
        print(f"  目标投入: {target_x_dict[category]}")

        E_tuned, T_tuned, error = tuner.tune_parameters(category, max_iterations=300)

        print(f"  调参完成，优化误差: {error:.6f}")
        print(f"  调整后的T向量（前3维）: {np.round(T_tuned[:3], 4)}...")

        tuned_data[category] = {
            'dims': base_data[category]['dims'],
            'G2025': base_data[category]['G2025'],
            'Target': T_tuned.tolist(),
            'Lambda_diag_avg_growth': base_data[category]['Lambda_diag_avg_growth'],
            'E': E_tuned.tolist()
        }

    # 验证调参效果
    print("\n" + "="*80)
    print("【调参验证】重新优化后的投入方案（9维加权模型）")
    print("="*80 + "\n")

    print(f"{'职业类别':<15} {'目标投入':<35} {'实际投入':<35} {'误差'}")
    print("-" * 95)

    for category in list(target_x_dict.keys()):
        target_x = np.array(target_x_dict[category])
        tuner_verify = ParameterTunerV2(tuned_data, target_x_dict, skill_weights=skill_weights)
        actual_x = tuner_verify.compute_optimal_x(tuned_data[category], skill_weights=skill_weights)

        target_str = ', '.join([f'{x:.2f}' for x in target_x])
        actual_str = ', '.join([f'{x:.2f}' for x in actual_x])
        error = np.sum((actual_x - target_x) ** 2)

        print(f"{category:<15} [{target_str}] [{actual_str}] {error:.6f}")

    print("\n" + "="*80 + "\n")

    # 保存调参后的数据
    with open('tuned_parameters_9d.json', 'w', encoding='utf-8') as f:
        json.dump(tuned_data, f, ensure_ascii=False, indent=2)

    print("✓ 9维模型调参数据已保存到 tuned_parameters_9d.json\n")

    return tuned_data


def main():
    """主程序"""
    import sys

    mode = '9d' if len(sys.argv) > 1 and sys.argv[1] == '9d' else '5d'

    if mode == '9d':
        print("\n运行模式: 9维模型参数调优（支持多权重评估）")
        tuned_data = generate_tuned_data_9d()
    else:
        print("\n运行模式: 5维模型参数调优")
        tuned_data = generate_tuned_data_5d()

    print("\n参数调优完成！")


if __name__ == '__main__':
    main()