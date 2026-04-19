#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
参数反向优化工具 - 7课程版本
通过调整效率矩阵E和目标向量Target，使最优开课方案接近指定的目标分配
"""

import numpy as np
from scipy.optimize import minimize
import json
import sys
import io

# 修复Windows编码问题
if sys.platform == 'win32' and not hasattr(sys.stdout, '_wrapper_set'):
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys.stdout._wrapper_set = True
    except (AttributeError, ValueError):
        pass

try:
    from 优化求解器_7课程版 import ExtendedLarmarckOptimizer, create_seven_course_data_pack
    HAS_OPTIMIZER = True
except ImportError:
    print("警告：无法导入7课程优化求解器")
    HAS_OPTIMIZER = False


class CourseAllocationTuner:
    """7课程开课方案参数调优器"""

    def __init__(self, base_data, target_allocations, skill_weights=None):
        """
        Args:
            base_data: 基础数据包
            target_allocations: 目标开课分配字典 {category: {scheme: [x1,...,x7]}}
            skill_weights: 权重方案
        """
        self.base_data = base_data
        self.target_allocations = target_allocations
        self.skill_weights = skill_weights or {}

    def compute_optimal_allocation(self, data_dict, scheme_weights):
        """计算给定数据下的最优开课方案"""
        optimizer = ExtendedLarmarckOptimizer(data_dict)

        result = optimizer.solve_constrained(
            bounds=(0, 1),
            budget_constraint=1.0,
            skill_weights=scheme_weights
        )

        return result['x_star']

    def objective_function(self, params, category, scheme, original_E, original_T):
        """
        目标函数：当前参数下的开课方案与目标分配的差异

        Args:
            params: 参数向量（E矩阵和T向量的扁平化）
            category: 专业类别
            scheme: 权重方案名称
            original_E: 原始E矩阵 (9×7)
            original_T: 原始T向量 (9维)
        """
        m, p = original_E.shape  # m=9维技能, p=7个课程

        # 解析参数：前m*p个是E矩阵，后m个是T向量调整
        E_flat = params[:m*p]
        T_adjust = params[m*p:m*p+m]

        # 恢复E矩阵
        E = E_flat.reshape(m, p)

        # 调整T向量（限制调整幅度）
        T = original_T + T_adjust * 0.05

        # 创建调整后的数据
        adjusted_data = {
            'G2025': self.base_data[category]['G2025'],
            'Target': T.tolist(),
            'Lambda_diag_avg_growth': self.base_data[category]['Lambda_diag_avg_growth'],
            'E': E.tolist(),
            'dims': self.base_data[category]['dims']
        }

        # 获取权重方案
        scheme_weights = self.skill_weights.get(scheme, None)

        # 计算最优分配
        x_opt = self.compute_optimal_allocation(adjusted_data, scheme_weights)

        # 计算与目标分配的差异
        target_x = np.array(self.target_allocations[category][scheme])
        error = np.sum((x_opt - target_x) ** 2)

        # 添加正则化项，防止E矩阵值过大或过小
        e_reg = 0.001 * np.sum((E - original_E) ** 2)

        return error + e_reg

    def tune_parameters(self, category, scheme, max_iterations=200):
        """
        调整参数使开课方案接近目标

        Args:
            category: 专业类别
            scheme: 权重方案
            max_iterations: 最大迭代次数
        """
        original_E = np.array(self.base_data[category]['E'], dtype=float)
        original_T = np.array(self.base_data[category]['Target'], dtype=float)

        m, p = original_E.shape

        # 初始参数
        params0 = np.concatenate([original_E.flatten(), np.zeros(m)])

        print(f"\n调参: {category} - {scheme} 方案")
        print(f"  模型: {m}维技能 × {p}个课程")
        print(f"  目标分配: {self.target_allocations[category][scheme]}")

        # 使用SLSQP优化
        result = minimize(
            self.objective_function,
            params0,
            args=(category, scheme, original_E, original_T),
            method='L-BFGS-B',
            bounds=[(0, 1) for _ in range(m*p)] + [(-0.5, 0.5) for _ in range(m)],
            options={'maxiter': max_iterations, 'ftol': 1e-6}
        )

        # 提取结果
        E_flat = result.x[:m*p]
        T_adjust = result.x[m*p:m*p+m]

        E_tuned = E_flat.reshape(m, p)
        T_tuned = original_T + T_adjust * 0.05

        return E_tuned, T_tuned, result.fun


def run_parameter_tuning():
    """运行参数调优"""

    if not HAS_OPTIMIZER:
        print("错误：需要优化求解器模块")
        return

    # 加载基础数据
    base_data = create_seven_course_data_pack()

    # 定义目标开课分配（用户提供）
    target_allocations = {
        "STEM_DataScience": {
            "comprehensive": [0.30, 0.15, 0.20, 0.15, 0.10, 0.05, 0.05],
            "social": [0.15, 0.25, 0.15, 0.15, 0.10, 0.10, 0.10],
            "technical": [0.25, 0.10, 0.30, 0.10, 0.10, 0.10, 0.05]
        },
        "Trade_Culinary": {
            "comprehensive": [0.20, 0.05, 0.25, 0.20, 0.15, 0.10, 0.05],
            "social": [0.15, 0.05, 0.30, 0.20, 0.10, 0.10, 0.10],
            "technical": [0.10, 0.15, 0.25, 0.25, 0.10, 0.10, 0.05]
        },
        "Arts_Drama": {
            "comprehensive": [0.10, 0.10, 0.25, 0.25, 0.15, 0.10, 0.05],
            "social": [0.10, 0.10, 0.30, 0.15, 0.20, 0.05, 0.10],
            "technical": [0.15, 0.15, 0.25, 0.15, 0.15, 0.10, 0.05]
        }
    }

    # 权重方案
    weight_schemes = {
        "comprehensive": {"technical": 0.4, "leadership": 0.3, "sustainability": 0.3},
        "social": {"technical": 0.3, "leadership": 0.1, "sustainability": 0.6},
        "technical": {"technical": 0.6, "leadership": 0.2, "sustainability": 0.2}
    }

    print("\n" + "="*85)
    print("参数反向优化工具 - 7课程版本")
    print("="*85)
    print("\n目标：调整效率矩阵E和目标向量Target，使最优开课方案接近指定分配\n")

    tuner = CourseAllocationTuner(base_data, target_allocations, weight_schemes)

    tuned_data = {}

    for category in ["STEM_DataScience", "Trade_Culinary", "Arts_Drama"]:
        print(f"\n{'='*85}")
        print(f"处理专业: {category}")
        print(f"{'='*85}")

        tuned_data[category] = {
            'dims': base_data[category]['dims'],
            'course_names': base_data[category]['course_names'],
            'G2025': base_data[category]['G2025'],
            'Lambda_diag_avg_growth': base_data[category]['Lambda_diag_avg_growth'],
            'schemes': {}
        }

        for scheme in ["comprehensive", "social", "technical"]:
            E_tuned, T_tuned, error = tuner.tune_parameters(category, scheme, max_iterations=300)

            print(f"  ✓ 优化误差: {error:.6f}")
            print(f"  调整后的Target向量: {np.round(T_tuned, 3)}\n")

            tuned_data[category]['schemes'][scheme] = {
                'E': E_tuned.tolist(),
                'Target': T_tuned.tolist(),
                'optimization_error': float(error)
            }

    # 保存调参结果
    output_file = 'tuned_parameters_7courses.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(tuned_data, f, ensure_ascii=False, indent=2)

    print("\n" + "="*85)
    print(f"✓ 调参完成！结果已保存到: {output_file}")
    print("="*85 + "\n")

    # 验证调参效果
    print("【验证调参效果】\n")

    for category in ["STEM_DataScience", "Trade_Culinary", "Arts_Drama"]:
        print(f"\n{category}:")
        print(f"{'方案':<15} {'目标分配':<50} {'实际分配':<50} {'误差'}")
        print("-" * 120)

        for scheme in ["comprehensive", "social", "technical"]:
            target_x = np.array(target_allocations[category][scheme])

            # 使用调参后的数据计算
            test_data = {
                'G2025': base_data[category]['G2025'],
                'Target': tuned_data[category]['schemes'][scheme]['Target'],
                'Lambda_diag_avg_growth': base_data[category]['Lambda_diag_avg_growth'],
                'E': tuned_data[category]['schemes'][scheme]['E'],
                'dims': base_data[category]['dims']
            }

            optimizer = ExtendedLarmarckOptimizer(test_data)
            result = optimizer.solve_constrained(
                bounds=(0, 1),
                budget_constraint=1.0,
                skill_weights=weight_schemes[scheme]
            )

            actual_x = result['x_star']
            error = np.sum((actual_x - target_x) ** 2)

            target_str = ', '.join([f'{x:.2f}' for x in target_x])
            actual_str = ', '.join([f'{x:.2f}' for x in actual_x])

            print(f"{scheme:<15} [{target_str}] [{actual_str}] {error:.6f}")


if __name__ == '__main__':
    run_parameter_tuning()