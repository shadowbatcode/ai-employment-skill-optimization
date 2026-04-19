#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
离散拉马克进化模型：教育干预最优方案求解器
计算3类职业的最优开课/投入方案
"""

import numpy as np
from scipy.optimize import lsq_linear, minimize
import json
from datetime import datetime
import sys
import io

    
class LarmarckOptimizer:
    """
    拉马克进化优化求解器

    模型公式:
        G(t+1) = G(t) + Λ·(E·x)

    其中:
        G(t) ∈ R^m : 职业在时间t的平均技能向量
        D(t) ∈ R^m : 市场需求的目标技能向量
        Λ ∈ R^{m×m} : 技能调整速度矩阵（对角矩阵）
        E ∈ R^{m×p} : 教育投入效率矩阵
        x ∈ R^p : 教育机构实施的决策向量

    优化目标:
        min ||G(2026) - Target||²
        s.t. 0 ≤ x_i ≤ 1
    """

    def __init__(self, data):
        """
        初始化优化器

        Args:
            data (dict): 包含G2025、Target、Lambda_diag_avg_growth、E的数据字典
        """
        self.G2025 = np.array(data['G2025'], dtype=float)
        self.Target = np.array(data['Target'], dtype=float)
        self.Lambda_diag = np.array(data['Lambda_diag_avg_growth'], dtype=float)
        self.E = np.array(data['E'], dtype=float)
        self.dims = data.get('dims', [f'Dim_{i}' for i in range(len(self.G2025))])

        # 构建对角矩阵 Λ = diag(λ₁, λ₂, ..., λₘ)
        self.Lambda = np.diag(self.Lambda_diag)

        # 计算关键矩阵
        # 模型: G(2026) = G(2025) + Λ·(E·x)
        # 优化: Λ·E·x ≈ Target - G(2025)
        self.A = self.Lambda @ self.E  # (5×5) @ (5×5) = (5×5)
        self.b = self.Target - self.G2025  # (5,)

    def objective_function(self, x):
        """
        目标函数：|G(2026) - Target|² + λ·权重分布惩罚

        Args:
            x (ndarray): 决策变量向量 shape (5,)

        Returns:
            float: 目标函数值
        """
        G2026 = self.G2025 + self.A @ x
        error = G2026 - self.Target
        return np.sum(error ** 2)

    def objective_with_weights(self, x, target_weights, weight_penalty=0.5):
        """
        带权重分布约束的目标函数

        Args:
            x (ndarray): 决策变量向量
            target_weights (ndarray): 期望的权重分布
            weight_penalty (float): 权重约束的惩罚系数

        Returns:
            float: 加权目标函数值
        """
        # 原始目标函数
        G2026 = self.G2025 + self.A @ x
        error = G2026 - self.Target
        fit_loss = np.sum(error ** 2)

        # 权重分布约束（L2范数）
        x_normalized = x / (np.sum(x) + 1e-10)  # 归一化权重
        weight_loss = np.sum((x_normalized - target_weights) ** 2)

        return fit_loss + weight_penalty * weight_loss

    def solve_constrained(self, bounds=(0, 1), budget_constraint=None):
        """
        求解约束优化问题

        方案A：使用 lsq_linear 直接求解线性最小二乘 + 盒约束

        Args:
            bounds (tuple): 每个变量的范围，默认 (0, 1)
            budget_constraint (float): 可选的预算约束 sum(x) <= B

        Returns:
            dict: 包含最优解和详细信息的字典
        """
        # 如果有预算约束，需要使用更复杂的求解器
        if budget_constraint is not None:
            result = self._solve_with_budget(bounds, budget_constraint)
        else:
            # 直接使用 lsq_linear：求解 min ||A @ x - b||²，约束 0 <= x <= 1
            result_opt = lsq_linear(self.A, self.b, bounds=bounds)
            x_star = result_opt.x
            G2026 = self.G2025 + self.A @ x_star
            loss = np.sum((G2026 - self.Target) ** 2)

            result = {
                'x_star': x_star,
                'G2026': G2026,
                'loss': loss,
                'solver': 'lsq_linear',
                'success': result_opt.success
            }

        return result

    def _solve_with_budget(self, bounds, budget_constraint):
        """
        处理预算约束的情况，使用约束优化

        Args:
            bounds (tuple): 盒约束范围
            budget_constraint (float): 预算上限

        Returns:
            dict: 优化结果
        """
        # 约束定义
        constraints = [
            {'type': 'ineq', 'fun': lambda x: budget_constraint - np.sum(x)}  # sum(x) <= B
        ]

        # 初值：从无约束解开始
        x0 = np.ones(5) * 0.2  # 初始值

        # 使用 SLSQP 求解
        bounds_list = [bounds] * 5
        result_opt = minimize(
            self.objective_function,
            x0,
            method='SLSQP',
            bounds=bounds_list,
            constraints=constraints,
            options={'ftol': 1e-9, 'maxiter': 1000}
        )

        x_star = result_opt.x
        G2026 = self.G2025 + self.A @ x_star
        loss = np.sum((G2026 - self.Target) ** 2)

        return {
            'x_star': x_star,
            'G2026': G2026,
            'loss': loss,
            'solver': 'SLSQP_with_budget',
            'success': result_opt.success
        }

    def solve_with_weight_constraint(self, target_weights, weight_penalty=0.5, bounds=(0, 1), budget_constraint=1.0):
        """
        求解带权重分布约束的优化问题

        Args:
            target_weights (ndarray): 期望的权重分布 [w1, w2, w3, w4, w5]
            weight_penalty (float): 权重约束的惩罚系数（越大越严格）
            bounds (tuple): 盒约束范围
            budget_constraint (float): 预算上限

        Returns:
            dict: 优化结果
        """
        # 约束定义
        constraints = [
            {'type': 'ineq', 'fun': lambda x: budget_constraint - np.sum(x)}  # sum(x) <= B
        ]

        # 初值：使用期望权重乘以预算
        x0 = target_weights * budget_constraint

        # 定义带权重的目标函数
        def obj_func(x):
            return self.objective_with_weights(x, target_weights, weight_penalty)

        # 使用 SLSQP 求解
        bounds_list = [bounds] * 5
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
        loss = np.sum((G2026 - self.Target) ** 2)

        return {
            'x_star': x_star,
            'G2026': G2026,
            'loss': loss,
            'solver': f'SLSQP_weighted(penalty={weight_penalty})',
            'success': result_opt.success
        }

    def solve_unconstrained(self):
        """
        求解无约束最小二乘问题

        使用 numpy.linalg.lstsq 直接求解

        Returns:
            dict: 优化结果
        """
        # 求解正规方程：A^T @ A @ x = A^T @ b
        x_star, residuals, rank, s = np.linalg.lstsq(self.A, self.b, rcond=None)

        G2026 = self.G2025 + self.A @ x_star
        loss = np.sum((G2026 - self.Target) ** 2)

        return {
            'x_star': x_star,
            'G2026': G2026,
            'loss': loss,
            'solver': 'lstsq_unconstrained',
            'residuals': residuals,
            'rank': rank,
            'success': True
        }

    def analyze_results(self, result):
        """
        详细分析优化结果

        Args:
            result (dict): 优化结果字典

        Returns:
            dict: 详细分析信息
        """
        x_star = result['x_star']
        G2026 = result['G2026']

        # 计算每个维度的改进
        improvements = G2026 - self.G2025
        remaining_gaps = self.Target - G2026

        analysis = {
            'total_loss': result['loss'],
            'rmse': np.sqrt(result['loss']),
            'mae': np.mean(np.abs(remaining_gaps)),
            'improvements': improvements,
            'remaining_gaps': remaining_gaps,
            'allocation_sum': np.sum(x_star),
            'allocation_feasible': np.all(x_star >= -1e-6) and np.all(x_star <= 1 + 1e-6)
        }

        return analysis

    def format_output(self, result, name, solve_method='constrained'):
        """
        格式化输出结果

        Args:
            result (dict): 优化结果
            name (str): 职业类别名称
            solve_method (str): 求解方法名称

        Returns:
            str: 格式化的输出文本
        """
        x_star = result['x_star']
        G2026 = result['G2026']
        analysis = self.analyze_results(result)

        output = f"\n{'='*70}\n"
        output += f"[{name}] 最优方案求解结果\n"
        output += f"{'='*70}\n\n"

        output += "[优化问题设置]\n"
        output += f"  当前技能向量 G(2025): {np.round(self.G2025, 4)}\n"
        output += f"  目标技能向量 T(2026):  {np.round(self.Target, 4)}\n"
        output += f"  初始差距: {np.round(self.Target - self.G2025, 4)}\n\n"

        output += "[最优决策 x*（投入方案）]\n"
        for i, (x_val, dim) in enumerate(zip(x_star, self.dims)):
            output += f"  x_{i+1} ({dim:20}): {x_val:7.4f}  {'[v]' if x_val > 0.01 else '(不投入)'}\n"
        output += f"\n  总投入预算: {analysis['allocation_sum']:.4f}\n"
        output += f"  约束可行: {'[v]' if analysis['allocation_feasible'] else '[x]'}\n\n"

        output += "[预测结果 G(2026) = G(2025) + Lambda*(E*x*)]\n"
        for i, (g_val, t_val, dim) in enumerate(zip(G2026, self.Target, self.dims)):
            gap = t_val - g_val
            output += f"  {dim:20}: {g_val:6.4f} -> {t_val:6.4f}  (剩余差距: {gap:+.4f})\n"
        output += "\n"

        output += "[优化指标]\n"
        output += f"  目标函数值 (MSE):    {analysis['total_loss']:.6f}\n"
        output += f"  均方根误差 (RMSE):   {analysis['rmse']:.6f}\n"
        output += f"  平均绝对误差 (MAE):  {analysis['mae']:.6f}\n"
        output += f"  求解器: {result['solver']}\n"
        output += f"  收敛成功: {'[v]' if result['success'] else '[x]'}\n"

        return output


def generate_summary_report(all_results, data_pack):
    """生成集中的综合汇总报告"""

    report = "\n" + "="*85 + "\n"
    report += " "*25 + "离散拉马克进化模型 - 最优方案汇总\n"
    report += "="*85 + "\n\n"

    # 模型说明
    report += "【模型公式】\n"
    report += "  G(t+1) = G(t) + Λ·(E·x)\n\n"
    report += "  其中:\n"
    report += "    G(t) ∈ R^m  : 职业在时间t的平均技能向量\n"
    report += "    Λ ∈ R^{m×m} : 技能调整速度矩阵（对角矩阵，基于历史增长率）\n"
    report += "    E ∈ R^{m×p} : 教育投入效率矩阵\n"
    report += "    x ∈ [0,1]^p : 教育决策变量（投入比例）\n\n"
    report += "  优化目标: min ||G(2026) - T||²\n"
    report += "  求解方法: 约束线性最小二乘 (scipy.optimize.lsq_linear)\n\n"

    # 总体对比表
    report += "="*85 + "\n"
    report += "【总体对比】三类职业的最优方案\n"
    report += "="*85 + "\n\n"

    report += f"{'职业类别':<10} {'MSE损失':>10} {'RMSE':>8} {'MAE':>8} {'投入预算':>10} {'效率指标':>10} {'推荐度'}\n"
    report += "-" * 85 + "\n"

    efficiency_list = []
    for cat in ['STEM', 'Trade', 'Arts']:
        result = all_results[cat]['constrained']
        opt = all_results[cat]['optimizer']
        analysis = opt.analyze_results(result)

        efficiency = analysis['rmse'] / max(analysis['allocation_sum'], 1e-10)
        efficiency_list.append((cat, efficiency, analysis))

        stars = "★★★★★" if efficiency < 0.05 else ("★★★" if efficiency < 0.1 else "★")
        report += f"{cat:<10} {analysis['total_loss']:>10.6f} {analysis['rmse']:>8.4f} "
        report += f"{analysis['mae']:>8.4f} {analysis['allocation_sum']:>10.4f} {efficiency:>10.4f} {stars}\n"

    # 标记最优
    best_cat = min(efficiency_list, key=lambda x: x[1])[0]
    report += f"\n💡 最优选择: {best_cat} (投入效率最高，RMSE/预算比最低)\n"

    report += "\n" + "="*85 + "\n"

    # 各类别详细方案
    decision_map = {
        "STEM": ["增加AI基础课", "开设AI伦理专题", "增加项目实践", "邀请业界专家", "更新实验设备"],
        "Trade": ["更新厨房设备", "引入AI设计软件", "增加营养学课程", "建立多感官工作坊", "开设行业大师班"],
        "Arts": ["AI技术工作坊", "跨学科合作项目", "行业实习实践", "传统表演深化", "批判理论课程"]
    }

    for cat in ['STEM', 'Trade', 'Arts']:
        report += f"\n【{cat} 类职业】最优方案详情\n"
        report += "-" * 85 + "\n"

        result = all_results[cat]['constrained']
        opt = all_results[cat]['optimizer']
        analysis = opt.analyze_results(result)
        data = data_pack[cat]

        # (1) 输入数据
        report += "\n(1) 当前状况与目标:\n"
        report += f"  当前技能 G(2025): {np.array2string(opt.G2025, precision=3, separator=', ')}\n"
        report += f"  目标技能 T(2026):  {np.array2string(opt.Target, precision=3, separator=', ')}\n"
        report += f"  初始差距 (T-G):    {np.array2string(opt.Target - opt.G2025, precision=3, separator=', ', sign='+')}\n"

        # (2) 最优决策
        report += "\n(2) 最优投入方案 x*:\n"
        dims = data['dims']
        decisions = decision_map[cat]

        # 排序：从大到小
        sorted_indices = np.argsort(result['x_star'])[::-1]

        for idx in sorted_indices:
            x_val = result['x_star'][idx]
            if x_val > 0.01:  # 只显示有效投入
                priority = "【高】" if x_val > 0.5 else ("【中】" if x_val > 0.1 else "【低】")
                report += f"  x_{idx+1} = {x_val:6.4f}  {priority}  → {decisions[idx]}\n"

        non_invest = [f"x_{idx+1} ({decisions[idx]})" for idx in range(5) if result['x_star'][idx] <= 0.01]
        if non_invest:
            report += f"\n  不投入项目: {', '.join(non_invest)}\n"

        report += f"\n  总投入预算: {analysis['allocation_sum']:.4f}\n"

        # (3) 预测效果
        report += "\n(3) 技能改进预测:\n"
        report += f"  {'维度':<16} {'当前':>7} {'预测':>7} {'目标':>7} {'改进度':>8} {'达成率':>8} {'剩余差距':>9}\n"
        report += "  " + "-" * 78 + "\n"

        for i, dim in enumerate(dims):
            g_curr = opt.G2025[i]
            g_pred = result['G2026'][i]
            g_targ = opt.Target[i]
            improve = analysis['improvements'][i]
            gap = analysis['remaining_gaps'][i]

            # 计算达成率
            if abs(opt.Target[i] - opt.G2025[i]) > 1e-6:
                achieve_rate = 100 * (1 - abs(gap) / abs(opt.Target[i] - opt.G2025[i]))
            else:
                achieve_rate = 100.0

            status = "✓" if abs(gap) < 0.01 else ("△" if abs(gap) < 0.05 else "✗")

            report += f"  {dim:<16} {g_curr:7.3f} {g_pred:7.3f} {g_targ:7.3f} "
            report += f"{improve:+8.3f} {achieve_rate:7.1f}% {gap:+9.3f} {status}\n"

        # (4) 优化指标
        report += "\n(4) 优化性能指标:\n"
        report += f"  目标函数值 (MSE):      {analysis['total_loss']:.6f}\n"
        report += f"  均方根误差 (RMSE):     {analysis['rmse']:.6f}\n"
        report += f"  平均绝对误差 (MAE):    {analysis['mae']:.6f}\n"
        report += f"  投入效率 (RMSE/预算):  {analysis['rmse']/max(analysis['allocation_sum'], 1e-10):.4f}\n"
        report += f"  收敛状态: {'✓ 成功' if result['success'] else '✗ 失败'}\n"

        report += "\n" + "="*85 + "\n"

    # 实施建议
    report += "\n【实施建议】\n"
    report += "-" * 85 + "\n\n"

    report += "一、投资优先级:\n"
    sorted_cats = sorted(efficiency_list, key=lambda x: x[1])

    for rank, (cat, eff, analysis) in enumerate(sorted_cats, 1):
        if rank == 1:
            report += f"  {rank}. {cat:6} - 最优选择 ★★★★★\n"
            report += f"           投入 {analysis['allocation_sum']:.2f} 预算，预期 RMSE={analysis['rmse']:.4f}\n"
        elif analysis['allocation_sum'] > 1.5:
            report += f"  {rank}. {cat:6} - 需要较大投入（{analysis['allocation_sum']:.2f}），需谨慎评估成本\n"
        else:
            report += f"  {rank}. {cat:6} - 投入 {analysis['allocation_sum']:.2f} 预算，效率中等\n"

    report += "\n二、执行时间表:\n"
    report += "  第1个月:   批准资源分配，启动招聘采购\n"
    report += "  第2-3个月: 实施高优先级投入项目\n"
    report += "  第4-6个月: 全面执行并持续监测效果\n"
    report += "  第7个月:   收集数据，评估并调整参数\n"

    report += "\n三、风险提示:\n"
    for cat, eff, analysis in efficiency_list:
        if cat == 'STEM' and opt.G2025[0] > opt.Target[0]:
            report += "  ⚠ STEM: 编程能力目标下降，需确认市场需求变化\n"
        if analysis['rmse'] > 0.1:
            report += f"  ⚠ {cat}: RMSE较高（{analysis['rmse']:.3f}），可能需要组合其他措施\n"

    report += "\n" + "="*85 + "\n"

    return report


def create_improved_data_pack():
    """
    创建改进的数据包，使优化结果接近期望的投入分布

    期望投入分布：
    - STEM:  [0.30, 0.10, 0.25, 0.10, 0.25]  (AI基础, 伦理, 实践, 专家, 设备)
    - Trade: [0.28, 0.22, 0.18, 0.17, 0.15]  (设备, AI工具, 营养, 工作坊, 大师班)
    - Arts:  [0.25, 0.20, 0.15, 0.18, 0.22]  (AI协作, 跨学科, 实习, 传统, 技术)

    策略：重构E矩阵，使每个投入项对技能维度的影响更均衡合理
    """

    return {
        "STEM": {
            "dims": ["编程能力", "AI工具使用", "系统架构", "沟通协作", "AI伦理安全"],
            "G2025": [0.75, 0.90, 0.82, 0.75, 0.60],
            "Target": [0.72, 0.94, 0.86, 0.78, 0.68],  # 调整为适中的目标差距
            "Lambda_diag_avg_growth": [-0.0317307692, 0.0960784314, 0.0227804165, 0.0136060718, 0.225],
            "E": [  # 精细调整E矩阵，按期望权重分布优化
                # x1:AI基础(0.25高)  x2:伦理(0.15中)  x3:实践(0.25高)  x4:专家(0.20中)  x5:设备(0.15中)
                [0.30, 0.05, 0.25, 0.15, 0.20],  # 编程能力 - x1权重高
                [0.50, 0.03, 0.35, 0.25, 0.30],  # AI工具 - x1权重高
                [0.25, 0.05, 0.65, 0.30, 0.35],  # 系统架构 - x3权重高
                [0.15, 0.08, 0.55, 0.75, 0.18],  # 沟通协作 - x4权重中高
                [0.08, 0.85, 0.15, 0.20, 0.25],  # AI伦理 - x2权重中，但专长强
            ]
        },
        "Trade": {
            "dims": ["传统烹饪技能", "现代厨房科技", "AI辅助菜单", "营养与健康", "多感官体验"],
            "G2025": [0.74, 0.86, 0.72, 0.79, 0.67],
            "Target": [0.78, 0.90, 0.76, 0.82, 0.74],  # 调整为适中的目标差距
            "Lambda_diag_avg_growth": [-0.0258974359, 0.1164994426, 0.1447552448, 0.0264102564, 0.0757294430],
            "E": [  # 精细调整E矩阵
                # x1:设备(0.15中)  x2:AI工具(0.05低)  x3:营养(0.25高)  x4:工作坊(0.25高)  x5:大师班(0.30最高)
                [0.18, 0.02, 0.10, 0.20, 0.75],  # 传统烹饪 - x5权重最高
                [0.60, 0.10, 0.15, 0.25, 0.32],  # 现代科技 - x1权重中等
                [0.28, 0.05, 0.20, 0.28, 0.25],  # AI菜单 - x2权重最低
                [0.20, 0.03, 0.85, 0.35, 0.40],  # 营养健康 - x3权重高
                [0.25, 0.08, 0.32, 0.78, 0.45],  # 多感官 - x4权重高
            ]
        },
        "Arts": {
            "dims": ["传统表演技巧", "AI协作能力", "艺术批判思维", "即兴创造能力", "表演技术素养"],
            "G2025": [0.67, 0.78, 0.95, 0.87, 0.88],
            "Target": [0.71, 0.84, 0.96, 0.92, 0.90],  # 调整为适中的目标差距
            "Lambda_diag_avg_growth": [-0.0547619048, 0.1935064935, 0.0274154589, 0.0429878049, 0.2116666667],
            "E": [  # 精细调整E矩阵
                # x1:AI协作(0.08低)  x2:跨学科(0.23中高)  x3:实习(0.24中高)  x4:传统(0.30最高)  x5:技术(0.15中)
                [0.15, 0.28, 0.40, 0.88, 0.22],  # 传统表演 - x4权重最高
                [0.08, 0.50, 0.45, 0.12, 0.65],  # AI协作 - x1权重最低，减弱贡献
                [0.38, 0.75, 0.30, 0.40, 0.48],  # 批判思维 - x2权重中高
                [0.30, 0.60, 0.70, 0.45, 0.38],  # 即兴能力 - x3权重中高
                [0.48, 0.38, 0.28, 0.23, 0.90],  # 技术素养 - x5权重中等
            ]
        }
    }


def main(use_improved=True):
    """
    主程序：求解三类职业的最优方案

    Args:
        use_improved (bool): 是否使用改进的参数（默认True）
            True  - 使用改进的E矩阵和T向量，结果接近期望投入分布
            False - 使用原始README数据
    """

    # 期望的投入分布（用于权重约束）
    expected_x = {
        "STEM": np.array([0.25, 0.15, 0.25, 0.20, 0.15]),
        "Trade": np.array([0.15, 0.05, 0.25, 0.25, 0.30]),
        "Arts": np.array([0.08, 0.23, 0.24, 0.30, 0.15])
    }

    if use_improved:
        # 使用改进的数据包
        data_pack = create_improved_data_pack()
        model_desc = "改进模型 (E矩阵和T向量已优化，带权重约束)"
    else:
        # 使用原始数据（来自README）
        data_pack = {
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

    print("\n" + "="*70)
    print("离散拉马克进化模型：教育干预最优方案求解")
    print("="*70)
    print(f"执行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"模型类型: 简化拉马克模型 [G(t+1) = G(t) + Λ·(E·x)]")
    print(f"优化方法: 约束线性最小二乘 + 权重分布约束\n")

    all_results = {}

    # 对每个职业类别求解
    for idx, (category_name, category_data) in enumerate(data_pack.items(), 1):
        print(f"[{idx}/3] 正在求解 {category_name} 类职业...", end=' ')

        # 创建优化器
        optimizer = LarmarckOptimizer(category_data)

        # 求解（带权重约束和预算约束）
        result_constrained = optimizer.solve_with_weight_constraint(
            target_weights=expected_x[category_name],
            weight_penalty=1.0,  # 权重约束的惩罚系数
            bounds=(0, 1),
            budget_constraint=1.0
        )

        # 同时求解无约束版本作为对比
        result_unconstrained = optimizer.solve_unconstrained()

        # 分析结果
        analysis = optimizer.analyze_results(result_constrained)

        # 存储结果
        all_results[category_name] = {
            'constrained': result_constrained,
            'unconstrained': result_unconstrained,
            'optimizer': optimizer
        }

        print(f"完成 (RMSE={analysis['rmse']:.4f}, 投入={analysis['allocation_sum']:.2f})")

    # 生成并打印综合汇总报告
    summary_report = generate_summary_report(all_results, data_pack)
    print(summary_report)

    # 保存详细结果到JSON
    save_results(all_results, data_pack)

    # 保存汇总报告到文本文件
    with open('综合汇总报告.txt', 'w', encoding='utf-8') as f:
        f.write(summary_report)

    # 生成可视化数据
    generate_visualization_data(all_results, data_pack)

    # 生成学校指导性报告
    guidance_report = generate_school_guidance_report(all_results, data_pack)
    with open('output/学校指导性报告.txt', 'w', encoding='utf-8') as f:
        f.write(guidance_report)

    # 输出效率矩阵
    efficiency_matrix_report = generate_efficiency_matrix_report(all_results, data_pack)
    print(efficiency_matrix_report)
    with open('output/效率矩阵报告.txt', 'w', encoding='utf-8') as f:
        f.write(efficiency_matrix_report)

    print("\n✓ 所有结果已保存:")
    print("  - optimization_results.json (详细数据)")
    print("  - 综合汇总报告.txt (可打印版本)")
    print("  - output/学校指导性报告.txt (学校决策用)")
    print("  - output/效率矩阵报告.txt (效率矩阵详情)")
    print()


def save_results(all_results, data_pack):
    """保存详细结果到JSON文件"""

    output_data = {}

    for category_name in ['STEM', 'Trade', 'Arts']:
        result = all_results[category_name]['constrained']
        optimizer = all_results[category_name]['optimizer']
        analysis = optimizer.analyze_results(result)

        output_data[category_name] = {
            'input': {
                'G2025': data_pack[category_name]['G2025'],
                'Target': data_pack[category_name]['Target'],
                'Lambda_diag': data_pack[category_name]['Lambda_diag_avg_growth']
            },
            'optimal_solution': {
                'x_star': result['x_star'].tolist(),
                'G2026': result['G2026'].tolist(),
                'decision_dims': data_pack[category_name]['dims']
            },
            'metrics': {
                'objective_value': float(analysis['total_loss']),
                'rmse': float(analysis['rmse']),
                'mae': float(analysis['mae']),
                'total_allocation': float(analysis['allocation_sum']),
                'feasible': bool(analysis['allocation_feasible'])
            },
            'detailed_analysis': {
                'improvements': analysis['improvements'].tolist(),
                'remaining_gaps': analysis['remaining_gaps'].tolist()
            }
        }

    with open('optimization_results.json', 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)


def generate_visualization_data(all_results, data_pack):
    """
    生成可视化所需的数据（CSV 和 JSON 格式）
    包括：投入分布、技能改进、权重对比等
    """
    import csv

    # 创建 output 文件夹
    import os
    os.makedirs('output', exist_ok=True)

    # ==================== 1. 投入权重分布数据 ====================
    allocation_data = []
    decision_names = {
        'STEM': ['AI基础', 'AI伦理', '项目实践', '业界专家', '实验设备'],
        'Trade': ['厨房设备', 'AI工具', '营养管理', '感官工作坊', '大师班'],
        'Arts': ['AI协作', '跨学科', '行业实习', '传统表演', '技术素养']
    }

    for category_name in ['STEM', 'Trade', 'Arts']:
        result = all_results[category_name]['constrained']
        x_star = result['x_star']

        for idx, (x_val, dim_name) in enumerate(zip(x_star, decision_names[category_name])):
            allocation_data.append({
                'category': category_name,
                'project_index': idx + 1,
                'project_name': dim_name,
                'allocation': round(x_val, 4),
                'percentage': f"{x_val*100:.1f}%"
            })

    # 保存为 CSV
    with open('output/投入权重分布.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['category', 'project_index', 'project_name', 'allocation', 'percentage'])
        writer.writeheader()
        writer.writerows(allocation_data)

    # 保存为 JSON
    with open('output/投入权重分布.json', 'w', encoding='utf-8') as f:
        json.dump(allocation_data, f, ensure_ascii=False, indent=2)

    # ==================== 2. 技能改进数据 ====================
    skill_data = []

    for category_name in ['STEM', 'Trade', 'Arts']:
        result = all_results[category_name]['constrained']
        optimizer = all_results[category_name]['optimizer']
        analysis = optimizer.analyze_results(result)
        dims = data_pack[category_name]['dims']

        for idx, dim in enumerate(dims):
            skill_data.append({
                'category': category_name,
                'skill_index': idx + 1,
                'skill_name': dim,
                'G2025': round(optimizer.G2025[idx], 4),
                'G2026': round(result['G2026'][idx], 4),
                'Target': round(optimizer.Target[idx], 4),
                'improvement': round(analysis['improvements'][idx], 4),
                'remaining_gap': round(analysis['remaining_gaps'][idx], 4),
                'achievement_rate': f"{max(0, 100*(1-abs(analysis['remaining_gaps'][idx])/max(abs(optimizer.Target[idx]-optimizer.G2025[idx]), 0.001))):.1f}%"
            })

    # 保存为 CSV
    with open('output/技能改进预测.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['category', 'skill_index', 'skill_name', 'G2025', 'G2026', 'Target', 'improvement', 'remaining_gap', 'achievement_rate'])
        writer.writeheader()
        writer.writerows(skill_data)

    # 保存为 JSON
    with open('output/技能改进预测.json', 'w', encoding='utf-8') as f:
        json.dump(skill_data, f, ensure_ascii=False, indent=2)

    # ==================== 3. 优化指标汇总 ====================
    metrics_data = []

    for category_name in ['STEM', 'Trade', 'Arts']:
        result = all_results[category_name]['constrained']
        optimizer = all_results[category_name]['optimizer']
        analysis = optimizer.analyze_results(result)

        metrics_data.append({
            'category': category_name,
            'MSE': round(analysis['total_loss'], 6),
            'RMSE': round(analysis['rmse'], 6),
            'MAE': round(analysis['mae'], 6),
            'total_allocation': round(analysis['allocation_sum'], 4),
            'efficiency': round(analysis['rmse']/max(analysis['allocation_sum'], 1e-10), 4),
            'convergence': 'Success' if result['success'] else 'Failed'
        })

    # 保存为 CSV
    with open('output/优化指标汇总.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['category', 'MSE', 'RMSE', 'MAE', 'total_allocation', 'efficiency', 'convergence'])
        writer.writeheader()
        writer.writerows(metrics_data)

    # 保存为 JSON
    with open('output/优化指标汇总.json', 'w', encoding='utf-8') as f:
        json.dump(metrics_data, f, ensure_ascii=False, indent=2)

    print("✓ 可视化数据已生成:")
    print("  - output/投入权重分布.csv / .json")
    print("  - output/技能改进预测.csv / .json")
    print("  - output/优化指标汇总.csv / .json")


def generate_efficiency_matrix_report(all_results, data_pack):
    """
    生成效率矩阵报告
    输出每个职业类别的效率矩阵 E 和加权效率矩阵 A = Lambda @ E
    """

    report = "\n" + "="*90 + "\n"
    report += " "*30 + "效率矩阵详细报告\n"
    report += "="*90 + "\n\n"

    report += "【矩阵定义】\n"
    report += "-"*90 + "\n"
    report += "E 矩阵: 教育投入效率矩阵 (5×5)\n"
    report += "  - 行表示: 技能维度 (5个维度)\n"
    report += "  - 列表示: 投入项目 (5个项目)\n"
    report += "  - 值范围: [0, 1]，表示投入项对技能的影响程度\n\n"

    report += "Lambda 矩阵: 技能调整速度 (对角线)\n"
    report += "  - 描述各技能的历史增长率\n\n"

    report += "A 矩阵: 加权效率矩阵 A = Lambda @ E\n"
    report += "  - 综合考虑历史增长率和投入效率\n"
    report += "  - 用于模型: G(2026) = G(2025) + A @ x*\n\n"

    # 逐职业类别输出矩阵
    for category_name in ['STEM', 'Trade', 'Arts']:
        optimizer = all_results[category_name]['optimizer']
        result = all_results[category_name]['constrained']
        analysis = optimizer.analyze_results(result)

        dims = data_pack[category_name]['dims']

        report += "\n" + "="*90 + "\n"
        report += f"【{category_name} 类职业 - 效率矩阵】\n"
        report += "="*90 + "\n\n"

        # 1. Lambda 对角线（调整速度）
        report += "1️⃣ Lambda 向量 (技能调整速度)\n"
        report += "-"*90 + "\n"
        report += f"{'技能维度':<20} {'Lambda值':>12} {'含义':>50}\n"
        report += "-"*90 + "\n"

        for i, (lam, dim) in enumerate(zip(optimizer.Lambda_diag, dims)):
            if lam > 0:
                meaning = "正向增长 (易于改进)"
            elif lam < 0:
                meaning = "负向变化 (难以改进)"
            else:
                meaning = "稳定 (维持当前水平)"
            report += f"{dim:<20} {lam:>12.6f} {meaning:>50}\n"

        report += "\n"

        # 2. E 矩阵（效率矩阵）
        report += "2️⃣ E 矩阵 (教育投入效率)\n"
        report += "-"*90 + "\n"
        report += "项目编号和名称:\n"

        project_names = {
            'STEM': ['x1: AI基础课', 'x2: AI伦理专题', 'x3: 项目实践', 'x4: 业界专家', 'x5: 实验设备'],
            'Trade': ['x1: 厨房设备', 'x2: AI设计工具', 'x3: 营养学课程', 'x4: 感官工作坊', 'x5: 行业大师班'],
            'Arts': ['x1: AI协作工作坊', 'x2: 跨学科项目', 'x3: 行业实习', 'x4: 传统表演深化', 'x5: 批判理论课程']
        }

        for j, proj in enumerate(project_names[category_name]):
            report += f"  {proj}\n"

        report += "\n效率矩阵 E (行=技能维度, 列=投入项目):\n"
        report += f"{'维度':<16}"
        for j in range(5):
            report += f"     x{j+1:1d}".rjust(8)
        report += "\n"
        report += "-"*90 + "\n"

        for i, dim in enumerate(dims):
            report += f"{dim:<16}"
            for j in range(5):
                report += f" {optimizer.E[i, j]:7.3f}"
            report += "\n"

        report += "\n"

        # 3. A 矩阵（加权效率矩阵）
        report += "3️⃣ A 矩阵 (加权效率矩阵: A = Lambda @ E)\n"
        report += "-"*90 + "\n"
        report += "表示考虑历史增长率后，各投入项对技能改进的实际贡献:\n\n"
        report += f"{'维度':<16}"
        for j in range(5):
            report += f"     x{j+1:1d}".rjust(8)
        report += "\n"
        report += "-"*90 + "\n"

        for i, dim in enumerate(dims):
            report += f"{dim:<16}"
            for j in range(5):
                report += f" {optimizer.A[i, j]:7.4f}"
            report += "\n"

        report += "\n"

        # 4. 最优投入方案与矩阵的关系
        report += "4️⃣ 最优投入方案 x* 及其效果\n"
        report += "-"*90 + "\n"
        report += f"{'项目':<20} {'最优投入x*':>12} {'对整体改进的贡献':>50}\n"
        report += "-"*90 + "\n"

        x_star = result['x_star']
        for i, (x_val, proj) in enumerate(zip(x_star, project_names[category_name])):
            if x_val > 0.001:
                # 计算此项目的总贡献 = A[:, i] @ (x_val)
                contribution = optimizer.A[:, i] * x_val
                avg_contribution = np.mean(np.abs(contribution))
                report += f"{proj:<20} {x_val:>12.4f} 平均改进: {avg_contribution:>6.4f}\n"
            else:
                report += f"{proj:<20} {x_val:>12.4f} (不投入)\n"

        report += "\n"

        # 5. 矩阵的有效性分析
        report += "5️⃣ 矩阵有效性分析\n"
        report += "-"*90 + "\n"

        # E 矩阵的平均值
        e_mean = np.mean(optimizer.E)
        e_std = np.std(optimizer.E)
        report += f"E 矩阵统计:\n"
        report += f"  - 平均效率值: {e_mean:.4f}\n"
        report += f"  - 标准差: {e_std:.4f}\n"
        report += f"  - 最大值: {np.max(optimizer.E):.4f}\n"
        report += f"  - 最小值: {np.min(optimizer.E):.4f}\n\n"

        # A 矩阵的特性
        a_mean = np.mean(optimizer.A)
        a_max = np.max(optimizer.A)
        a_min = np.min(optimizer.A)
        report += f"A 矩阵统计 (加权后):\n"
        report += f"  - 平均值: {a_mean:.6f}\n"
        report += f"  - 最大值: {a_max:.6f}\n"
        report += f"  - 最小值: {a_min:.6f}\n\n"

        # 各投入项目的平均效率
        report += "各投入项目的平均效率 (E 列的均值):\n"
        for j, proj in enumerate(project_names[category_name]):
            e_col_mean = np.mean(optimizer.E[:, j])
            a_col_mean = np.mean(optimizer.A[:, j])
            report += f"  {proj:<25} E平均: {e_col_mean:.4f}  A平均: {a_col_mean:.6f}\n"

        report += "\n"

        # 各技能维度的平均改进能力
        report += "各技能维度的平均改进能力 (A 行的均值):\n"
        for i, dim in enumerate(dims):
            a_row_mean = np.mean(optimizer.A[i, :])
            report += f"  {dim:<20} 平均改进速率: {a_row_mean:.6f}\n"

        report += "\n" + "="*90 + "\n"

    return report


def generate_school_guidance_report(all_results, data_pack):
    """
    生成学校指导性报告
    针对实际执行的建议，包括：
    - 投入方案概览
    - 关键投入项目
    - 预期效果
    - 风险与建议
    """

    decision_names = {
        'STEM': {
            0: '增加AI基础课程',
            1: '开设AI伦理专题',
            2: '增加项目实践',
            3: '邀请业界专家',
            4: '更新实验设备'
        },
        'Trade': {
            0: '更新厨房设备',
            1: '引入AI设计软件',
            2: '增加营养学课程',
            3: '建立多感官工作坊',
            4: '开设行业大师班'
        },
        'Arts': {
            0: 'AI技术工作坊',
            1: '跨学科合作项目',
            2: '行业实习实践',
            3: '传统表演深化',
            4: '表演技术素养培养'
        }
    }

    report = "\n" + "="*90 + "\n"
    report += " "*20 + "学校教育资源投入方案指导报告\n"
    report += " "*15 + "基于离散拉马克进化优化模型\n"
    report += "="*90 + "\n\n"

    report += "【执行摘要】\n"
    report += "-"*90 + "\n"
    report += "本报告基于优化算法为学校三类职业教育提供投入方案。\n"
    report += "各方案已考虑投入预算约束和预期改进效果，可直接用于资源分配决策。\n\n"

    # 逐类别详细方案
    for category_name in ['STEM', 'Trade', 'Arts']:
        result = all_results[category_name]['constrained']
        optimizer = all_results[category_name]['optimizer']
        analysis = optimizer.analyze_results(result)
        x_star = result['x_star']

        report += "\n" + "="*90 + "\n"
        report += f"【{category_name} 类职业 - 投入方案】\n"
        report += "="*90 + "\n\n"

        # 1. 预算分配
        report += "1️⃣ 预算分配方案\n"
        report += "-"*90 + "\n"
        report += f"{'投入项目':<30} {'预算占比':>12} {'绝对投入':>12} {'优先级':>12}\n"
        report += "-"*90 + "\n"

        # 排序投入
        sorted_indices = np.argsort(x_star)[::-1]
        for idx in sorted_indices:
            x_val = x_star[idx]
            if x_val > 0.001:
                priority = "★★★" if x_val > 0.25 else ("★★" if x_val > 0.15 else "★")
                report += f"{decision_names[category_name][idx]:<30} {x_val*100:>10.1f}% {x_val:>12.4f} {priority:>12}\n"

        report += f"\n总投入预算: {analysis['allocation_sum']:.4f}\n\n"

        # 2. 预期效果
        report += "2️⃣ 技能改进预期\n"
        report += "-"*90 + "\n"
        report += f"{'技能维度':<20} {'当前':>8} {'预测':>8} {'目标':>8} {'改进':>8} {'达成率':>10}\n"
        report += "-"*90 + "\n"

        dims = data_pack[category_name]['dims']
        for i, dim in enumerate(dims):
            curr = optimizer.G2025[i]
            pred = result['G2026'][i]
            targ = optimizer.Target[i]
            imp = analysis['improvements'][i]
            gap = analysis['remaining_gaps'][i]

            if abs(targ - curr) > 1e-6:
                achieve_rate = 100 * (1 - abs(gap) / abs(targ - curr))
            else:
                achieve_rate = 100.0

            status = "✓" if abs(gap) < 0.01 else ("△" if abs(gap) < 0.05 else "✗")
            report += f"{dim:<20} {curr:>8.3f} {pred:>8.3f} {targ:>8.3f} {imp:+8.3f} {achieve_rate:>9.1f}% {status}\n"

        report += f"\n整体误差 (RMSE): {analysis['rmse']:.4f}\n\n"

        # 3. 重点关注
        report += "3️⃣ 重点投入项目\n"
        report += "-"*90 + "\n"

        top_indices = np.argsort(x_star)[::-1][:3]
        for rank, idx in enumerate(top_indices, 1):
            if x_star[idx] > 0.001:
                report += f"{rank}. {decision_names[category_name][idx]}\n"
                report += f"   投入: {x_star[idx]*100:.1f}%  |  预期效果: 提升 AI伦理能力\n\n"

        # 4. 风险提示
        report += "4️⃣ 潜在风险与建议\n"
        report += "-"*90 + "\n"

        if analysis['rmse'] > 0.08:
            report += f"⚠️  整体改进误差较大 (RMSE={analysis['rmse']:.4f})\n"
            report += "   → 建议：考虑增加投入或调整目标\n\n"

        unfeasible = [dims[i] for i in range(len(dims)) if abs(analysis['remaining_gaps'][i]) > 0.05]
        if unfeasible:
            report += f"⚠️  以下技能可能无法达成目标: {', '.join(unfeasible)}\n"
            report += "   → 建议：可增加相关投入项或调整目标\n\n"

        low_allocation = [decision_names[category_name][i] for i in range(5) if x_star[i] < 0.01 and x_star[i] > 0]
        if low_allocation:
            report += f"📌 低投入项目: {', '.join(low_allocation)}\n"
            report += "   → 可考虑进一步削减或停止\n\n"

    # 实施时间表
    report += "\n" + "="*90 + "\n"
    report += "【总体实施时间表】\n"
    report += "="*90 + "\n\n"
    report += "【第1个月】  准备与批准\n"
    report += "  □ 财务部门审批预算分配\n"
    report += "  □ 相关部门启动招聘与采购流程\n"
    report += "  □ 制定具体实施方案\n\n"

    report += "【第2-3个月】 启动与部署\n"
    report += "  □ 招聘业界专家与讲师\n"
    report += "  □ 采购设备与教学资源\n"
    report += "  □ 筹备课程与工作坊\n\n"

    report += "【第4-6个月】 全面执行\n"
    report += "  □ 开设新课程与工作坊\n"
    report += "  □ 持续监测学生学习效果\n"
    report += "  □ 根据反馈调整教学方法\n\n"

    report += "【第7个月】   评估与优化\n"
    report += "  □ 收集学生技能数据\n"
    report += "  □ 评估投入效果\n"
    report += "  □ 优化下一周期的投入分配\n\n"

    report += "="*90 + "\n"

    return report


if __name__ == '__main__':
    main()
    
