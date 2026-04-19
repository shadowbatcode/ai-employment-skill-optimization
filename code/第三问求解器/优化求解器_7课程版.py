#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
扩展拉马克进化模型 - 7课程版本
支持9维技能向量 × 7个开课项目的优化
支持多权重评估方案
"""

import numpy as np
from scipy.optimize import lsq_linear, minimize
import json
from datetime import datetime
import sys
import io

# 修复Windows编码问题
if sys.platform == 'win32' and not hasattr(sys.stdout, '_wrapper_set'):
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys.stdout._wrapper_set = True
    except AttributeError:
        pass

class ExtendedLarmarckOptimizer:
    """
    扩展拉马克进化优化求解器（7课程版本）

    支持9维技能向量 × 7个开课项目
    模型: G(2026) = G(2025) + Λ·(E·x)
    """

    def __init__(self, data, skill_dimension_map=None):
        self.G2025 = np.array(data['G2025'], dtype=float)
        self.Target = np.array(data['Target'], dtype=float)
        self.Lambda_diag = np.array(data['Lambda_diag_avg_growth'], dtype=float)
        self.E = np.array(data['E'], dtype=float)
        self.dims = data.get('dims', [f'Dim_{i}' for i in range(len(self.G2025))])
        self.course_names = data.get('course_names', [f'Course_{i+1}' for i in range(self.E.shape[1])])

        self.m = len(self.G2025)  # 技能维度
        self.p = self.E.shape[1]  # 投入项目数

        self.skill_dimension_map = skill_dimension_map or self._default_skill_map()
        self.Lambda = np.diag(self.Lambda_diag)
        self.A = self.Lambda @ self.E
        self.b = self.Target - self.G2025

    def _default_skill_map(self):
        """生成默认的技能维度映射"""
        return {i: (self.dims[i], 'technical' if i < self.m//3 else ('leadership' if i < 2*self.m//3 else 'sustainability'))
                for i in range(self.m)}

    def objective_function(self, x, weights=None):
        """目标函数：加权版||G(2026) - Target||²"""
        G2026 = self.G2025 + self.A @ x
        error = G2026 - self.Target

        if weights is None:
            return np.sum(error ** 2)
        else:
            weighted_error = 0.0
            for i, (dim_name, category) in self.skill_dimension_map.items():
                weight = weights.get(category, 1.0)
                weighted_error += weight * (error[i] ** 2)
            return weighted_error

    def solve_constrained(self, bounds=(0, 1), budget_constraint=1.0, skill_weights=None):
        """求解约束优化问题"""
        constraints = []
        if budget_constraint is not None:
            constraints.append({'type': 'ineq', 'fun': lambda x: budget_constraint - np.sum(x)})

        x0 = np.ones(self.p) * (1.0 / self.p)

        def obj_func(x):
            return self.objective_function(x, skill_weights)

        bounds_list = [bounds] * self.p
        result_opt = minimize(
            obj_func,
            x0,
            method='SLSQP',
            bounds=bounds_list,
            constraints=constraints,
            options={'ftol': 1e-9, 'maxiter': 2000}
        )

        x_star = result_opt.x
        G2026 = self.G2025 + self.A @ x_star
        loss = self.objective_function(x_star, skill_weights)

        return {
            'x_star': x_star,
            'G2026': G2026,
            'loss': loss,
            'solver': 'SLSQP',
            'success': result_opt.success
        }

    def analyze_results(self, result):
        """分析优化结果"""
        x_star = result['x_star']
        G2026 = result['G2026']
        error = G2026 - self.Target

        return {
            'total_loss': float(np.sum(error ** 2)),
            'rmse': float(np.sqrt(np.mean(error ** 2))),
            'mae': float(np.mean(np.abs(error))),
            'allocation_sum': float(np.sum(x_star)),
            'allocation_feasible': bool(np.sum(x_star) <= 1.01),
            'improvements': G2026 - self.G2025,
            'remaining_gaps': error
        }


def create_seven_course_data_pack():
    """
    创建支持7个开课项目的数据包
    9维技能 × 7个开课项目
    """

    return {
        "STEM_DataScience": {
            "dims": [
                "编程能力", "AI工具使用", "系统架构",  # Technical (0-2)
                "沟通协作", "领导力", "团队管理",        # Leadership (3-5)
                "AI伦理安全", "环境意识", "全球视野"   # Sustainability (6-8)
            ],
            "course_names": [
                "增加AI基础课",
                "开设AI伦理专题",
                "增加项目实践",
                "邀请业界专家",
                "更新实验设备",
                "增加创业项目",
                "碳足迹数据分析竞赛"
            ],
            "G2025": [0.75, 0.90, 0.82, 0.75, 0.60, 0.52, 0.43, 0.71],
            "Target": [0.60, 0.97, 0.85, 0.76, 0.70, 0.75, 0.70, 0.82],
            "Lambda_diag_avg_growth": [-0.0317, 0.0961, 0.0228, 0.0136, 0.0250, 0.2250, 0.1500, 0.1800],
            "E": np.array([
                [0.40, 0.05, 0.30, 0.20, 0.25, 0.15, 0.10],  # 编程能力
                [0.60, 0.03, 0.45, 0.35, 0.30, 0.20, 0.08],  # AI工具使用
                [0.30, 0.05, 0.75, 0.45, 0.40, 0.25, 0.15],  # 系统架构
                [0.15, 0.08, 0.60, 0.80, 0.10, 0.50, 0.30],  # 沟通协作
                [0.10, 0.05, 0.50, 0.75, 0.05, 0.60, 0.40],  # 领导力
                [0.05, 0.80, 0.15, 0.25, 0.10, 0.20, 0.50],  # 环境意识
                [0.08, 0.60, 0.25, 0.35, 0.10, 0.30, 0.70],  # 全球视野
                [0.10, 0.50, 0.35, 0.40, 0.15, 0.40, 0.60],  # 沟通协作（社会维度）
                [0.12, 0.45, 0.30, 0.50, 0.20, 0.45, 0.65],  # 领导力（社会维度）
            ]).tolist(),
            "skill_map": {
                0: ("编程能力", "technical"),
                1: ("AI工具使用", "technical"),
                2: ("系统架构", "technical"),
                3: ("沟通协作", "leadership"),
                4: ("领导力", "leadership"),
                5: ("环境意识", "sustainability"),
                6: ("全球视野", "sustainability"),
                7: ("沟通协作", "leadership"),
                8: ("领导力", "leadership"),
            }
        },
        "Trade_Culinary": {
            "dims": [
                "传统烹饪技能", "现代厨房科技", "AI辅助菜单",  # Technical (0-2)
                "营养与健康", "多感官体验", "团队协作",        # Leadership (3-5)
                "食品安全", "可持续发展", "文化理解"          # Sustainability (6-8)
            ],
            "course_names": [
                "更新厨房设备",
                "引入AI设计软件",
                "增加营养学课程",
                "建立多感官工作坊",
                "开设行业大师班",
                "参加实习项目",
                "厨余垃圾再利用课程"
            ],
            "G2025": [0.74, 0.86, 0.72, 0.79, 0.67, 0.68, 0.80, 0.70, 0.70],
            "Target": [0.72, 0.90, 0.80, 0.77, 0.72, 0.75, 0.85, 0.90, 0.75],
            "Lambda_diag_avg_growth": [-0.0259, 0.1165, 0.1448, 0.0264, 0.0757, 0.1200, 0.1800, 0.1500, 0.1600],
            "E": np.array([
                [0.20, 0.02, 0.10, 0.20, 0.70, 0.15, 0.15],  # 传统烹饪技能
                [0.70, 0.15, 0.20, 0.25, 0.35, 0.30, 0.10],  # 现代厨房科技
                [0.35, 0.85, 0.25, 0.30, 0.25, 0.20, 0.10],  # AI辅助菜单
                [0.15, 0.05, 0.85, 0.35, 0.40, 0.30, 0.20],  # 营养与健康
                [0.20, 0.08, 0.35, 0.90, 0.45, 0.35, 0.25],  # 多感官体验
                [0.10, 0.10, 0.70, 0.25, 0.35, 0.40, 0.60],  # 食品安全
                [0.08, 0.15, 0.60, 0.40, 0.30, 0.35, 0.75],  # 可持续发展
                [0.15, 0.10, 0.40, 0.75, 0.50, 0.45, 0.30],  # 营养与健康（社会维度）
                [0.12, 0.12, 0.50, 0.80, 0.60, 0.50, 0.35],  # 多感官体验（社会维度）
            ]).tolist(),
            "skill_map": {
                0: ("传统烹饪技能", "technical"),
                1: ("现代厨房科技", "technical"),
                2: ("AI辅助菜单", "technical"),
                3: ("营养与健康", "leadership"),
                4: ("多感官体验", "leadership"),
                5: ("食品安全", "sustainability"),
                6: ("可持续发展", "sustainability"),
                7: ("营养与健康", "leadership"),
                8: ("多感官体验", "leadership"),
            }
        },
        "Arts_Drama": {
            "dims": [
                "传统表演技巧", "AI协作能力", "艺术批判思维",  # Technical (0-2)
                "即兴创造能力", "团队合作", "情感表达",        # Leadership (3-5)
                "表演技术素养", "文化意识", "社会责任"        # Sustainability (6-8)
            ],
            "course_names": [
                "AI技术工作坊",
                "跨学科合作项目",
                "行业实习实践",
                "传统表演深化",
                "批判理论课程",
                "开设舞台沟通课程",
                "开设全球文化课程"
            ],
            "G2025": [0.67, 0.78, 0.95, 0.87, 0.88, 0.72, 0.80, 0.65, 0.83],
            "Target": [0.72, 0.90, 0.90, 0.95, 0.90, 0.70, 0.95, 0.70, 0.95],
            "Lambda_diag_avg_growth": [-0.0548, 0.1935, 0.0274, 0.0430, 0.0350, 0.2117, 0.1600, 0.1900, 0.1500],
            "E": np.array([
                [0.08, 0.15, 0.25, 0.85, 0.12, 0.35, 0.10],  # 传统表演技巧
                [0.85, 0.55, 0.45, 0.05, 0.20, 0.30, 0.15],  # AI协作能力
                [0.20, 0.80, 0.35, 0.40, 0.85, 0.50, 0.60],  # 艺术批判思维
                [0.30, 0.65, 0.75, 0.45, 0.25, 0.60, 0.55],  # 即兴创造能力
                [0.25, 0.70, 0.65, 0.35, 0.15, 0.70, 0.50],  # 团队合作
                [0.80, 0.40, 0.30, 0.15, 0.65, 0.45, 0.40],  # 表演技术素养
                [0.15, 0.50, 0.45, 0.30, 0.70, 0.60, 0.85],  # 文化意识
                [0.35, 0.75, 0.80, 0.50, 0.30, 0.65, 0.60],  # 即兴创造能力（社会维度）
                [0.30, 0.70, 0.70, 0.40, 0.25, 0.70, 0.80],  # 团队合作（社会维度）
            ]).tolist(),
            "skill_map": {
                0: ("传统表演技巧", "technical"),
                1: ("AI协作能力", "technical"),
                2: ("艺术批判思维", "technical"),
                3: ("即兴创造能力", "leadership"),
                4: ("团队合作", "leadership"),
                5: ("表演技术素养", "sustainability"),
                6: ("文化意识", "sustainability"),
                7: ("即兴创造能力", "leadership"),
                8: ("团队合作", "leadership"),
            }
        }
    }


def main_seven_courses():
    """运行7课程版本的优化"""

    print(f"执行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("模型类型: 扩展拉马克模型 [G(2026) = G(2025) + Λ·(E·x)]")
    print("特点: 9维技能向量 × 7个开课项目\n")

    data_pack = create_seven_course_data_pack()

    weight_schemes = {
        "comprehensive": {"technical": 0.4, "leadership": 0.3, "sustainability": 0.3},
        "social": {"technical": 0.3, "leadership": 0.1, "sustainability": 0.6},
        "technical": {"technical": 0.6, "leadership": 0.2, "sustainability": 0.2}
    }

    all_results = {}
    categories = list(data_pack.keys())

    for idx, category_name in enumerate(categories, 1):
        print(f"\n{'='*85}")
        print(f"[{idx}/3] 正在求解 {category_name}")
        print(f"{'='*85}")

        category_data = data_pack[category_name]
        skill_map = category_data.pop('skill_map')
        course_names = category_data.pop('course_names')

        optimizer = ExtendedLarmarckOptimizer(category_data, skill_dimension_map=skill_map)
        optimizer.course_names = course_names

        print(f"  技能维度: {optimizer.m} | 开课项目: {optimizer.p}\n")

        scheme_results = {}
        for scheme_name, weights in weight_schemes.items():
            print(f"  求解权重方案: {scheme_name}")

            result = optimizer.solve_constrained(
                bounds=(0, 1),
                budget_constraint=1.0,
                skill_weights=weights
            )

            analysis = optimizer.analyze_results(result)
            scheme_results[scheme_name] = {'result': result, 'analysis': analysis}

            # 显示开课方案
            x_star = result['x_star']
            print(f"    RMSE={analysis['rmse']:.6f}, 总投入={analysis['allocation_sum']:.4f}")
            print(f"    开课方案: ", end='')
            for i, (x_val, course) in enumerate(zip(x_star, course_names)):
                if x_val > 0.01:
                    print(f"{course}({x_val:.2f}) ", end='')
            print()

        all_results[category_name] = {
            'optimizer': optimizer,
            'schemes': scheme_results
        }

    # 保存结果
    print("\n" + "="*85)
    print("【保存结果】")
    print("="*85)

    output_file = 'optimization_results_7courses.json'
    output_data = {}

    for category_name in categories:
        optimizer = all_results[category_name]['optimizer']
        schemes = all_results[category_name]['schemes']

        output_data[category_name] = {
            'metadata': {
                'skill_dimensions': optimizer.m,
                'course_count': optimizer.p,
                'course_names': optimizer.course_names
            },
            'weight_schemes': {}
        }

        for scheme_name, data in schemes.items():
            result = data['result']
            analysis = data['analysis']
            output_data[category_name]['weight_schemes'][scheme_name] = {
                'optimal_allocation': result['x_star'].tolist(),
                'predicted_G2026': result['G2026'].tolist(),
                'metrics': {
                    'RMSE': float(analysis['rmse']),
                    'MAE': float(analysis['mae']),
                    'total_allocation': float(analysis['allocation_sum'])
                }
            }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"✓ 结果已保存到: {output_file}\n")


if __name__ == '__main__':
    main_seven_courses()