#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
扩展拉马克进化模型：教育干预最优方案求解器
支持多权重评估、多维技能向量、多种教育干预方案
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
        sys.stdout._wrapper_set = True  # 标记已设置，避免重复设置
    except AttributeError:
        pass  # 如果已经被设置过，跳过

class ExtendedLarmarckOptimizer:
    """
    扩展拉马克进化优化求解器

    模型公式（扩展版）:
        G^extended(t+1) = G(t) + μ·ΔG(t) + η·R

    简化实现:
        G(2026) = G(2025) + Λ·(E·x)

    其中:
        G(t) ∈ R^m : 职业在时间t的平均技能向量 (m可以是9维或其他)
        Target ∈ R^m : 市场需求的目标技能向量
        Λ ∈ R^{m×m} : 技能调整速度矩阵（对角矩阵）
        E ∈ R^{m×p} : 教育投入效率矩阵
        x ∈ R^p : 教育机构实施的决策向量

    加权评估系统:
        w_0: 技术技能权重
        w_1: 领导力和协作技能权重
        w_2: 可持续性和全球视野权重
        (w_0 + w_1 + w_2 = 1)

    优化目标（加权版）:
        min Σ(w_k * ||G_k(2026) - Target_k||²) for k in [0,1,2]
        s.t. 0 ≤ x_i ≤ 1
    """


    def __init__(self, data, skill_dimension_map=None):
        """
        初始化优化器

        Args:
            data (dict): 包含G2025、Target、Lambda_diag_avg_growth、E的数据字典
            skill_dimension_map (dict): 技能维度映射 {index: (name, category: 'technical'/'leadership'/'sustainability')}
        """
        self.G2025 = np.array(data['G2025'], dtype=float)
        self.Target = np.array(data['Target'], dtype=float)
        self.Lambda_diag = np.array(data['Lambda_diag_avg_growth'], dtype=float)
        self.E = np.array(data['E'], dtype=float)
        self.dims = data.get('dims', [f'Dim_{i}' for i in range(len(self.G2025))])

        # 技能维度数量和投入项目数量
        self.m = len(self.G2025)  # 技能维度
        self.p = self.E.shape[1]  # 投入项目数

        # 技能维度映射（如果提供）
        self.skill_dimension_map = skill_dimension_map or self._default_skill_map()

        # 构建对角矩阵 Λ = diag(λ₁, λ₂, ..., λₘ)
        self.Lambda = np.diag(self.Lambda_diag)

        # 计算关键矩阵
        # 模型: G(2026) = G(2025) + Λ·(E·x)
        # 优化: Λ·E·x ≈ Target - G(2025)
        self.A = self.Lambda @ self.E
        self.b = self.Target - self.G2025

    def _default_skill_map(self):
        """生成默认的技能维度映射"""
        return {i: (self.dims[i], 'technical' if i < self.m//3 else ('leadership' if i < 2*self.m//3 else 'sustainability'))
                for i in range(self.m)}

    def objective_function(self, x, weights=None):
        """
        目标函数：加权版||G(2026) - Target||²

        Args:
            x (ndarray): 决策变量向量 shape (p,)
            weights (dict): 技能类别权重 {'technical': w0, 'leadership': w1, 'sustainability': w2}
                           默认 None 表示均等权重

        Returns:
            float: 加权目标函数值
        """
        G2026 = self.G2025 + self.A @ x
        error = G2026 - self.Target

        if weights is None:
            # 无加权，均等权重
            return np.sum(error ** 2)
        else:
            # 加权评估
            weighted_error = 0.0
            for i, (dim_name, category) in self.skill_dimension_map.items():
                weight = weights.get(category, 1.0)
                weighted_error += weight * (error[i] ** 2)
            return weighted_error

    def objective_with_weights(self, x, target_weights, weight_penalty=0.5, skill_weights=None):
        """
        带权重分布约束的目标函数

        Args:
            x (ndarray): 决策变量向量
            target_weights (ndarray): 期望的投入权重分布
            weight_penalty (float): 权重约束的惩罚系数
            skill_weights (dict): 技能类别权重 {'technical': w0, 'leadership': w1, 'sustainability': w2}

        Returns:
            float: 加权目标函数值
        """
        # 原始目标函数（可选加权）
        G2026 = self.G2025 + self.A @ x
        error = G2026 - self.Target

        if skill_weights is None:
            fit_loss = np.sum(error ** 2)
        else:
            fit_loss = 0.0
            for i, (dim_name, category) in self.skill_dimension_map.items():
                weight = skill_weights.get(category, 1.0)
                fit_loss += weight * (error[i] ** 2)

        # 权重分布约束（L2范数）
        x_normalized = x / (np.sum(x) + 1e-10)  # 归一化权重
        weight_loss = np.sum((x_normalized - target_weights) ** 2)

        return fit_loss + weight_penalty * weight_loss

    def solve_constrained(self, bounds=(0, 1), budget_constraint=None, skill_weights=None, min_allocation=0.05):
        """
        求解约束优化问题

        Args:
            bounds (tuple): 每个变量的范围，默认 (0, 1)
            budget_constraint (float): 预算约束 sum(x) <= B
            skill_weights (dict): 技能类别权重
            min_allocation (float): 最小投入比例 (0-1)，确保每个项目都有投入

        Returns:
            dict: 包含最优解和详细信息的字典
        """
        if budget_constraint is not None or skill_weights is not None or min_allocation > 0:
            result = self._solve_with_constraints(bounds, budget_constraint, skill_weights, min_allocation)
        else:
            result_opt = lsq_linear(self.A, self.b, bounds=bounds)
            x_star = result_opt.x
            G2026 = self.G2025 + self.A @ x_star
            loss = self.objective_function(x_star, skill_weights)

            result = {
                'x_star': x_star,
                'G2026': G2026,
                'loss': loss,
                'solver': 'lsq_linear',
                'success': result_opt.success
            }

        return result

    def _solve_with_constraints(self, bounds, budget_constraint, skill_weights, min_allocation=0.05):
        """处理各种约束的情况"""
        constraints = []
        if budget_constraint is not None:
            constraints.append({'type': 'ineq', 'fun': lambda x: budget_constraint - np.sum(x)})

        x0 = np.ones(self.p) * (1.0 / self.p)

        def obj_func(x):
            return self.objective_function(x, skill_weights)

        # 修改bounds以包含最小投入约束
        bounds_list = []
        for i in range(self.p):
            lower = max(bounds[0], min_allocation)
            upper = bounds[1]
            bounds_list.append((lower, upper))

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

    def solve_with_weight_constraint(self, target_weights, weight_penalty=0.5, bounds=(0, 1), budget_constraint=1.0, skill_weights=None):
        """
        求解带权重分布约束的优化问题

        Args:
            target_weights (ndarray): 期望的权重分布 [w1, w2, ..., wp]
            weight_penalty (float): 权重约束的惩罚系数（越大越严格）
            bounds (tuple): 盒约束范围
            budget_constraint (float): 预算上限
            skill_weights (dict): 技能类别权重 {'technical': w0, 'leadership': w1, 'sustainability': w2}

        Returns:
            dict: 优化结果
        """
        # 约束定义
        constraints = [
            {'type': 'ineq', 'fun': lambda x: budget_constraint - np.sum(x)}
        ]

        # 初值：使用期望权重乘以预算
        x0 = target_weights * budget_constraint

        # 定义带权重的目标函数
        def obj_func(x):
            return self.objective_with_weights(x, target_weights, weight_penalty, skill_weights)

        # 使用 SLSQP 求解
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
            'solver': f'SLSQP_weighted(penalty={weight_penalty})',
            'success': result_opt.success
        }

    def solve_unconstrained(self, skill_weights=None):
        """
        求解无约束最小二乘问题

        使用 numpy.linalg.lstsq 直接求解

        Args:
            skill_weights (dict): 技能类别权重

        Returns:
            dict: 优化结果
        """
        # 求解正规方程：A^T @ A @ x = A^T @ b
        x_star, residuals, rank, s = np.linalg.lstsq(self.A, self.b, rcond=None)

        G2026 = self.G2025 + self.A @ x_star
        loss = self.objective_function(x_star, skill_weights)

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
        output += f"  初始差距: {np.round(self.Target - self.G2025, 4)}\n"
        output += f"  技能维度数: {self.m}\n"
        output += f"  投入项目数: {self.p}\n\n"

        output += "[最优决策 x*（投入方案）]\n"
        for i, x_val in enumerate(zip(x_star)):
            x_val = x_val[0]
            dim_info = self.dims[i] if i < len(self.dims) else f"Dim_{i}"
            output += f"  x_{i+1} ({dim_info:20}): {x_val:7.4f}  {'[v]' if x_val > 0.01 else '(不投入)'}\n"
        output += f"\n  总投入预算: {analysis['allocation_sum']:.4f}\n"
        output += f"  约束可行: {'[v]' if analysis['allocation_feasible'] else '[x]'}\n\n"

        output += "[预测结果 G(2026) = G(2025) + Lambda*(E*x*)]\n"
        for i, (g_val, t_val) in enumerate(zip(G2026, self.Target)):
            dim_info = self.dims[i] if i < len(self.dims) else f"Dim_{i}"
            gap = t_val - g_val
            output += f"  {dim_info:20}: {g_val:6.4f} -> {t_val:6.4f}  (剩余差距: {gap:+.4f})\n"
        output += "\n"

        output += "[优化指标]\n"
        output += f"  目标函数值 (MSE):    {analysis['total_loss']:.6f}\n"
        output += f"  均方根误差 (RMSE):   {analysis['rmse']:.6f}\n"
        output += f"  平均绝对误差 (MAE):  {analysis['mae']:.6f}\n"
        output += f"  求解器: {result['solver']}\n"
        output += f"  收敛成功: {'[v]' if result['success'] else '[x]'}\n"

        return output


# 为了向后兼容性，提供别名
LarmarckOptimizer = ExtendedLarmarckOptimizer


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
        "STEM": [
            "增加AI基础课", "开设AI伦理专题", "增加项目实践",
            "邀请业界专家", "更新实验设备", "增加创业项目",
            "开展校园碳足迹数据分析竞赛"
        ],
        "Trade": [
            "更新厨房设备", "引入AI设计软件", "增加营养学课程",
            "建立多感官工作坊", "开设行业大师班", "参加实习项目",
            "开设厨余垃圾再利用课程"
        ],
        "Arts": [
            "AI技术工作坊", "跨学科合作项目", "行业实习实践",
            "传统表演深化", "批判理论课程", "开设舞台沟通课程",
            "开设全球文化课程"
        ]
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

        non_invest = [f"x_{idx+1} ({decisions[idx]})" for idx in range(7) if result['x_star'][idx] <= 0.01]
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


def create_extended_data_pack():
    """
    创建扩展数据包：7个教育项目版本

    支持9维技能向量和多权重评估方案

    技能维度映射:
    - 维度0-2: 技术技能 (technical)
    - 维度3-5: 领导力和协作技能 (leadership)
    - 维度6-8: 可持续性和全球视野 (sustainability)

    权重方案示例:
    - 综合发展 (Comprehensive): w0=0.4, w1=0.3, w2=0.3
    - 社会贡献 (Social): w0=0.3, w1=0.1, w2=0.6
    - 技术优先 (Technical): w0=0.6, w1=0.2, w2=0.2
    """

    return {
        "STEM_DataScience": {
            "dims": [
                "编程能力", "AI工具使用", "系统架构",  # Technical (0-2)
                "沟通协作", "领导力", "团队管理",      # Leadership (3-5)
                "AI伦理安全", "环境意识", "全球视野"  # Sustainability (6-8)
            ],
            "project_names": [
                "增加AI基础课", "开设AI伦理专题", "增加项目实践",
                "邀请业界专家", "更新实验设备", "增加创业项目",
                "开展校园碳足迹数据分析竞赛"
            ],
            "G2025": [0.75, 0.90, 0.82, 0.75, 0.60, 0.50, 0.52, 0.43, 0.71],
            "Target": [0.60, 0.97, 0.85, 0.76, 0.70, 0.70, 0.75, 0.70, 0.82],
            "gap": [-0.15, 0.07, 0.03, 0.01, 0.10, 0.20, 0.23, 0.27, 0.11],
            "Lambda_diag_avg_growth": [
                -0.0317, 0.0961, 0.0228,  # Technical
                0.0136, 0.0250, 0.0300,   # Leadership
                0.2250, 0.1500, 0.1800    # Sustainability
            ],
            "E": [
                # x1:AI基础  x2:AI伦理  x3:项目实践  x4:业界专家  x5:实验设备  x6:创业项目  x7:碳足迹竞赛
                [0.50, 0.05, 0.35, 0.25, 0.30, 0.40, 0.10],  # 编程能力
                [0.70, 0.03, 0.50, 0.40, 0.35, 0.45, 0.08],  # AI工具使用
                [0.35, 0.05, 0.80, 0.50, 0.45, 0.60, 0.15],  # 系统架构
                [0.20, 0.10, 0.70, 0.85, 0.15, 0.75, 0.20],  # 沟通协作
                [0.15, 0.08, 0.60, 0.80, 0.10, 0.70, 0.12],  # 领导力
                [0.10, 0.12, 0.55, 0.75, 0.08, 0.65, 0.10],  # 团队管理
                [0.10, 0.90, 0.20, 0.30, 0.15, 0.25, 0.40],  # AI伦理安全
                [0.05, 0.70, 0.25, 0.20, 0.10, 0.30, 0.85],  # 环境意识
                [0.08, 0.60, 0.30, 0.40, 0.12, 0.50, 0.70],  # 全球视野
            ],
            "desired_allocations": {
                "comprehensive": [0.30, 0.15, 0.20, 0.15, 0.10, 0.05, 0.05],
                "social": [0.15, 0.25, 0.15, 0.15, 0.10, 0.10, 0.10],
                "technical": [0.25, 0.10, 0.30, 0.10, 0.10, 0.10, 0.05]
            },
            "skill_map": {
                0: ("编程能力", "technical"),
                1: ("AI工具使用", "technical"),
                2: ("系统架构", "technical"),
                3: ("沟通协作", "leadership"),
                4: ("领导力", "leadership"),
                5: ("团队管理", "leadership"),
                6: ("AI伦理安全", "sustainability"),
                7: ("环境意识", "sustainability"),
                8: ("全球视野", "sustainability"),
            }
        },
        "Trade_Culinary": {
            "dims": [
                "传统烹饪技能", "现代厨房科技", "AI辅助菜单",  # Technical (0-2)
                "营养与健康", "多感官体验", "团队协作",        # Leadership (3-5)
                "食品安全", "可持续发展", "文化理解"          # Sustainability (6-8)
            ],
            "project_names": [
                "更新厨房设备", "引入AI设计软件", "增加营养学课程",
                "建立多感官工作坊", "开设行业大师班", "参加实习项目",
                "开设厨余垃圾再利用课程"
            ],
            "G2025": [0.74, 0.86, 0.72, 0.79, 0.67, 0.68, 0.80, 0.70, 0.75],
            "Target": [0.72, 0.90, 0.80, 0.77, 0.72, 0.75, 0.85, 0.90, 0.75],
            "gap": [-0.02, 0.04, 0.08, -0.02, 0.05, 0.07, 0.05, 0.20, 0.00],
            "Lambda_diag_avg_growth":   [
                -0.0259, 0.1165, 0.1448,  # Technical
                -0.0264, 0.0757, 0.0500,  # Leadership
                0.1200, 0.1800, 0.1500    # Sustainability
            ],
            "E": [
                # x1:厨房设备  x2:AI设计软件  x3:营养课程  x4:感官工作坊  x5:大师班  x6:实习项目  x7:垃圾再利用
                [0.20, 0.02, 0.10, 0.25, 0.85, 0.35, 0.15],  # 传统烹饪技能
                [0.75, 0.15, 0.20, 0.30, 0.40, 0.50, 0.10],  # 现代厨房科技
                [0.35, 0.90, 0.25, 0.35, 0.30, 0.45, 0.20],  # AI辅助菜单
                [0.15, 0.05, 0.90, 0.40, 0.45, 0.55, 0.35],  # 营养与健康
                [0.25, 0.10, 0.40, 0.95, 0.50, 0.60, 0.25],  # 多感官体验
                [0.18, 0.08, 0.35, 0.70, 0.55, 0.65, 0.20],  # 团队协作
                [0.10, 0.12, 0.75, 0.30, 0.40, 0.40, 0.80],  # 食品安全
                [0.08, 0.15, 0.65, 0.45, 0.35, 0.50, 0.90],  # 可持续发展
                [0.05, 0.10, 0.50, 0.40, 0.80, 0.45, 0.55],  # 文化理解
            ],
            "desired_allocations": {
                "comprehensive": [0.20, 0.05, 0.25, 0.20, 0.15, 0.10, 0.05],
                "social": [0.15, 0.05, 0.30, 0.20, 0.10, 0.10, 0.10],
                "technical": [0.10, 0.15, 0.25, 0.25, 0.10, 0.10, 0.05]
            },
            "skill_map": {
                0: ("传统烹饪技能", "technical"),
                1: ("现代厨房科技", "technical"),
                2: ("AI辅助菜单", "technical"),
                3: ("营养与健康", "leadership"),
                4: ("多感官体验", "leadership"),
                5: ("团队协作", "leadership"),
                6: ("食品安全", "sustainability"),
                7: ("可持续发展", "sustainability"),
                8: ("文化理解", "sustainability"),
            }
        },
        "Arts_Drama": {
            "dims": [
                "传统表演技巧", "AI协作能力", "艺术批判思维",  # Technical (0-2)
                "即兴创造能力", "团队合作", "情感表达",        # Leadership (3-5)
                "表演技术素养", "文化意识", "社会责任"        # Sustainability (6-8)
            ],
            "project_names": [
                "AI技术工作坊", "跨学科合作项目", "行业实习实践",
                "传统表演深化", "批判理论课程", "开设舞台沟通课程",
                "开设全球文化课程"
            ],
            "G2025": [0.67, 0.78, 0.95, 0.87, 0.88, 0.72, 0.80, 0.65, 0.83],
            "Target": [0.72, 0.90, 0.90, 0.95, 0.90, 0.70, 0.95, 0.70, 0.95],
            "gap": [0.05, 0.12, -0.05, 0.08, 0.02, -0.02, 0.15, 0.05, 0.12],
            "Lambda_diag_avg_growth": [
                0.0548, 0.1935, -0.0274,  # Technical
                0.0430, 0.0350, -0.0200,  # Leadership
                0.2117, 0.1600, 0.1900    # Sustainability
            ],
            "E": [
                # x1:AI协作  x2:跨学科  x3:行业实习  x4:传统表演  x5:批判理论  x6:舞台沟通  x7:全球文化
                [0.10, 0.20, 0.30, 0.95, 0.15, 0.25, 0.10],  # 传统表演技巧
                [0.90, 0.60, 0.50, 0.08, 0.25, 0.15, 0.20],  # AI协作能力
                [0.25, 0.85, 0.40, 0.45, 0.90, 0.30, 0.35],  # 艺术批判思维
                [0.35, 0.70, 0.80, 0.50, 0.30, 0.75, 0.40],  # 即兴创造能力
                [0.30, 0.75, 0.70, 0.40, 0.20, 0.80, 0.45],  # 团队合作
                [0.15, 0.40, 0.60, 0.85, 0.25, 0.90, 0.50],  # 情感表达
                [0.85, 0.45, 0.35, 0.20, 0.70, 0.40, 0.30],  # 表演技术素养
                [0.20, 0.55, 0.50, 0.35, 0.80, 0.55, 0.85],  # 文化意识
                [0.15, 0.60, 0.55, 0.30, 0.85, 0.60, 0.90],  # 社会责任
            ],
            "desired_allocations": {
                "comprehensive": [0.10, 0.10, 0.25, 0.25, 0.15, 0.10, 0.05],
                "social": [0.10, 0.10, 0.30, 0.15, 0.20, 0.05, 0.10],
                "technical": [0.15, 0.15, 0.25, 0.15, 0.15, 0.10, 0.05]
            },
            "skill_map": {
                0: ("传统表演技巧", "technical"),
                1: ("AI协作能力", "technical"),
                2: ("艺术批判思维", "technical"),
                3: ("即兴创造能力", "leadership"),
                4: ("团队合作", "leadership"),
                5: ("情感表达", "leadership"),
                6: ("表演技术素养", "sustainability"),
                7: ("文化意识", "sustainability"),
                8: ("社会责任", "sustainability"),
            }
        }
    }




def adjust_efficiency_matrix_by_desired_allocations(category_data, desired_allocations, iterations=300, learning_rate=0.05):
    """
    根据期望的投入分配反推调整效率矩阵E（增强版）

    使用迭代优化：对于每个权重方案，计算优化结果与期望值的差异，
    然后根据差异调整E矩阵中的对应元素。

    Args:
        category_data: 职业数据
        desired_allocations: 期望分配字典
        iterations: 迭代次数（增加到300）
        learning_rate: 学习率（调整步长）
    """

    # 提取数据
    G2025 = np.array(category_data['G2025'])
    Target = np.array(category_data['Target'])
    Lambda_diag = np.array(category_data['Lambda_diag_avg_growth'])
    E_original = np.array(category_data['E'], dtype=float)
    dims = category_data['dims']
    project_names = category_data.get('project_names', [])

    m, p = E_original.shape  # m: 技能维度, p: 项目数

    # 定义权重方案
    weight_schemes = {
        "comprehensive": {"technical": 0.4, "leadership": 0.3, "sustainability": 0.3},
        "social": {"technical": 0.3, "leadership": 0.1, "sustainability": 0.6},
        "technical": {"technical": 0.6, "leadership": 0.2, "sustainability": 0.2}
    }

    E_adjusted = E_original.copy()
    best_E = E_original.copy()
    best_total_error = float('inf')

    skill_map = category_data.get('skill_map', {})

    # 动态调整学习率
    lr = learning_rate

    for iteration in range(iterations):
        # 计算所有方案的总误差
        total_error = 0
        gradient_accumulator = np.zeros((m, p))  # 累积梯度

        for scheme_name, weights in weight_schemes.items():
            if scheme_name not in desired_allocations:
                continue

            x_desired = np.array(desired_allocations[scheme_name])

            # 创建临时优化器
            temp_data = {
                'G2025': G2025,
                'Target': Target,
                'Lambda_diag_avg_growth': Lambda_diag,
                'E': E_adjusted,
                'dims': dims,
                'project_names': project_names
            }

            try:
                optimizer = ExtendedLarmarckOptimizer(temp_data, skill_dimension_map=skill_map)
                result = optimizer.solve_constrained(
                    bounds=(0, 1),
                    budget_constraint=1.0,
                    skill_weights=weights,
                    min_allocation=0.03  # 最小投入3%，确保所有项目都有投入
                )

                # 计算与期望的误差（L2范数）
                x_actual = result['x_star']
                error = np.linalg.norm(x_actual - x_desired)
                total_error += error

                # 计算调整方向（更激进的负反馈）
                diff = x_actual - x_desired

                # 针对每个项目调整效率矩阵
                for j in range(p):
                    if diff[j] > 0.02:  # 投入过多
                        # 降低该项目对所有技能的效率
                        gradient_accumulator[:, j] -= lr * diff[j]
                    elif diff[j] < -0.02:  # 投入不足
                        # 提高该项目对所有技能的效率
                        gradient_accumulator[:, j] -= lr * diff[j]

            except Exception as e:
                # 如果优化失败，跳过这个方案
                continue

        # 应用梯度更新
        E_adjusted = np.clip(E_adjusted + gradient_accumulator, 0.01, 1.0)

        # 更新最优解
        if total_error < best_total_error:
            best_total_error = total_error
            best_E = E_adjusted.copy()

            # 动态调整学习率（误差越小，学习率越小）
            if total_error < 1.0:
                lr = learning_rate * 0.5
            elif total_error < 0.5:
                lr = learning_rate * 0.3

        # 定期输出进度
        if (iteration + 1) % 50 == 0:
            print(f"    逆优化迭代 {iteration+1}/{iterations}, 总误差: {total_error:.6f}, 最优误差: {best_total_error:.6f}")

        # 早停：如果误差足够小
        if best_total_error < 0.3:
            print(f"    提前收敛！迭代 {iteration+1}, 误差: {best_total_error:.6f}")
            break

    print(f"    最终调整完成，最优总误差: {best_total_error:.6f}")
    return best_E


def main_extended():
    """
    扩展主程序：支持9维技能向量和多权重评估

    演示3个职业的最优方案求解：
    1. STEM - Data Science (数据科学)
    2. Trade - Advanced Culinary Skills (高级烹饪)
    3. Arts - Drama Performance (戏剧表演)

    支持的权重方案：
    - Comprehensive (综合发展): w0=0.4 技术, w1=0.3 领导力, w2=0.3 可持续
    - Social (社会贡献): w0=0.3 技术, w1=0.1 领导力, w2=0.6 可持续
    """

    print(f"执行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("模型类型: 扩展拉马克模型 [G(2026) = G(2025) + Λ·(E·x)]")
    print("特点: 9维技能向量 + 7个教育项目 + 3类权重评估\n")

    # 获取扩展数据包
    data_pack = create_extended_data_pack()

    # 定义多种权重方案
    weight_schemes = {
        "comprehensive": {  # 综合发展
            "technical": 0.4,
            "leadership": 0.3,
            "sustainability": 0.3
        },
        "social": {  # 社会贡献
            "technical": 0.3,
            "leadership": 0.1,
            "sustainability": 0.6
        },
        "technical": {  # 技术优先
            "technical": 0.6,
            "leadership": 0.2,
            "sustainability": 0.2
        }
    }

    all_results = {}

    # 对每个职业类别求解
    categories = list(data_pack.keys())
    num_categories = len(categories)

    for idx, category_name in enumerate(categories, 1):
        print(f"\n{'='*85}")
        print(f"[{idx}/{num_categories}] 正在求解 {category_name}")
        print(f"{'='*85}")

        category_data = data_pack[category_name]
        skill_map = category_data.pop('skill_map')  # 移除skill_map以避免传入优化器
        desired_allocations = category_data.pop('desired_allocations', {})  # 移除期望分配数据
        project_names = category_data.get('project_names', [])  # 获取项目名称

        # 逆优化：根据期望分配调整效率矩阵
        if desired_allocations:
            print("\n  正在根据期望分配调整效率矩阵（增强版）...")
            E_adjusted = adjust_efficiency_matrix_by_desired_allocations(
                category_data, desired_allocations, iterations=300, learning_rate=0.05
            )
            category_data['E'] = E_adjusted

        # 创建扩展优化器
        optimizer = ExtendedLarmarckOptimizer(category_data, skill_dimension_map=skill_map)

        print(f"  • 技能维度: {optimizer.m}")
        print(f"  • 投入项目: {optimizer.p}")
        print(f"  • 当前技能向量 G(2025): {np.round(optimizer.G2025[:3], 3)}...（首3维）")
        print(f"  • 目标技能向量 T(2026):  {np.round(optimizer.Target[:3], 3)}...（首3维）\n")

        # 求解三种权重方案
        scheme_results = {}
        for scheme_name, weights in weight_schemes.items():
            print(f"  求解权重方案: {scheme_name}")
            print(f"    权重配置: 技术={weights['technical']}, 领导力={weights['leadership']}, 可持续={weights['sustainability']}")

            # 求解（加入最小投入约束 5%）
            result = optimizer.solve_constrained(
                bounds=(0, 1),
                budget_constraint=1.0,
                skill_weights=weights,
                min_allocation=0.05  # 最小投入5%，确保所有项目都有投入
            )

            analysis = optimizer.analyze_results(result)
            scheme_results[scheme_name] = {
                'result': result,
                'analysis': analysis
            }

            print(f"    ✓ RMSE={analysis['rmse']:.6f}, 投入预算={analysis['allocation_sum']:.4f}")

            # 如果有期望分配，显示对比
            if scheme_name in desired_allocations:
                expected = np.array(desired_allocations[scheme_name])
                actual = result['x_star']
                diff = np.linalg.norm(actual - expected)
                print(f"    期望分配: {np.round(expected, 3)}")
                print(f"    实际分配: {np.round(actual, 3)}")
                print(f"    差异(L2范数): {diff:.6f}\n")
            else:
                print()

        all_results[category_name] = {
            'optimizer': optimizer,
            'schemes': scheme_results,
            'desired_allocations': desired_allocations,
            'project_names': project_names
        }

    # 生成综合报告
    print("\n" + "="*85)
    print("【综合对比分析】")
    print("="*85)

    for category_name in categories:
        print(f"\n{category_name} - 三种权重方案对比:")
        print(f"{'权重方案':<20} {'MSE':<12} {'RMSE':<12} {'MAE':<12} {'总投入':<12}")
        print("-" * 60)

        schemes = all_results[category_name]['schemes']
        for scheme_name, data in schemes.items():
            analysis = data['analysis']
            print(f"{scheme_name:<20} {analysis['total_loss']:<12.6f} {analysis['rmse']:<12.6f} "
                  f"{analysis['mae']:<12.6f} {analysis['allocation_sum']:<12.4f}")

    # 保存详细结果
    print("\n" + "="*85)
    print("【保存结果】")
    print("="*85)

    output_file = 'extended_optimization_results.json'
    output_data = {}

    for category_name in categories:
        optimizer = all_results[category_name]['optimizer']
        schemes = all_results[category_name]['schemes']

        output_data[category_name] = {
            'metadata': {
                'skill_dimensions': optimizer.m,
                'intervention_projects': optimizer.p,
                'dims': optimizer.dims
            },
            'current_state': {
                'G2025': optimizer.G2025.tolist(),
                'Target': optimizer.Target.tolist(),
                'Lambda_diag': optimizer.Lambda_diag.tolist()
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
                    'MSE': float(analysis['total_loss']),
                    'RMSE': float(analysis['rmse']),
                    'MAE': float(analysis['mae']),
                    'total_allocation': float(analysis['allocation_sum']),
                    'improvements': analysis['improvements'].tolist(),
                    'remaining_gaps': analysis['remaining_gaps'].tolist()
                }
            }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"✓ 详细结果已保存到: {output_file}")

    # 生成效率矩阵报告
    print("\n" + "="*85)
    print("【生成效率矩阵详细报告】")
    print("="*85)
    efficiency_report = generate_efficiency_matrix_report(all_results, data_pack)
    print(efficiency_report)

    with open('efficiency_matrix_report.txt', 'w', encoding='utf-8') as f:
        f.write(efficiency_report)
    print("✓ 效率矩阵报告已保存到: efficiency_matrix_report.txt")

    # 生成学校指导报告
    print("\n" + "="*85)
    print("【生成学校指导性报告】")
    print("="*85)
    guidance_report = generate_school_guidance_report(all_results, data_pack)
    print(guidance_report)

    with open('school_guidance_report.txt', 'w', encoding='utf-8') as f:
        f.write(guidance_report)
    print("✓ 学校指导报告已保存到: school_guidance_report.txt")

    # 生成可视化数据
    print("\n" + "="*85)
    print("【生成可视化数据】")
    print("="*85)
    generate_visualization_data(all_results, data_pack)

    print("\n扩展模型求解完成！")


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
        'STEM_DataScience': [
            '增加AI基础课', '开设AI伦理专题', '增加项目实践',
            '邀请业界专家', '更新实验设备', '增加创业项目',
            '开展校园碳足迹数据分析竞赛'
        ],
        'Trade_Culinary': [
            '更新厨房设备', '引入AI设计软件', '增加营养学课程',
            '建立多感官工作坊', '开设行业大师班', '参加实习项目',
            '开设厨余垃圾再利用课程'
        ],
        'Arts_Drama': [
            'AI技术工作坊', '跨学科合作项目', '行业实习实践',
            '传统表演深化', '批判理论课程', '开设舞台沟通课程',
            '开设全球文化课程'
        ]
    }

    for category_name in all_results.keys():
        # 使用综合方案的结果
        result = all_results[category_name]['schemes']['comprehensive']['result']
        x_star = result['x_star']

        # 获取分类简称
        cat_short = category_name.split('_')[0] if '_' in category_name else category_name

        for idx, (x_val, dim_name) in enumerate(zip(x_star, decision_names.get(category_name, []))):
            allocation_data.append({
                'category': cat_short,
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

    for category_name in all_results.keys():
        result = all_results[category_name]['schemes']['comprehensive']['result']
        optimizer = all_results[category_name]['optimizer']
        analysis = optimizer.analyze_results(result)
        dims = optimizer.dims

        # 获取分类简称
        cat_short = category_name.split('_')[0] if '_' in category_name else category_name

        for idx, dim in enumerate(dims):
            skill_data.append({
                'category': cat_short,
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

    for category_name in all_results.keys():
        result = all_results[category_name]['schemes']['comprehensive']['result']
        optimizer = all_results[category_name]['optimizer']
        analysis = optimizer.analyze_results(result)

        # 获取分类简称
        cat_short = category_name.split('_')[0] if '_' in category_name else category_name

        metrics_data.append({
            'category': cat_short,
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
    report += "E 矩阵: 教育投入效率矩阵 (9×7)\n"
    report += "  - 行表示: 技能维度 (9个维度)\n"
    report += "  - 列表示: 投入项目 (7个项目)\n"
    report += "  - 值范围: [0, 1]，表示投入项对技能的影响程度\n\n"

    report += "Lambda 矩阵: 技能调整速度 (对角线)\n"
    report += "  - 描述各技能的历史增长率\n\n"

    report += "A 矩阵: 加权效率矩阵 A = Lambda @ E\n"
    report += "  - 综合考虑历史增长率和投入效率\n"
    report += "  - 用于模型: G(2026) = G(2025) + A @ x*\n\n"

    # 逐职业类别输出矩阵
    for category_name in all_results.keys():
        optimizer = all_results[category_name]['optimizer']
        result = all_results[category_name]['schemes']['comprehensive']['result']  # 使用综合方案作为示例
        analysis = optimizer.analyze_results(result)

        dims = optimizer.dims

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

        project_names = all_results[category_name].get('project_names', [])
        for j, proj in enumerate(project_names):
            report += f"  x{j+1}: {proj}\n"

        report += "\n效率矩阵 E (行=技能维度, 列=投入项目):\n"
        report += f"{'维度':<16}"
        for j in range(len(project_names)):
            report += f"     x{j+1:1d}".rjust(8)
        report += "\n"
        report += "-"*90 + "\n"

        for i, dim in enumerate(dims):
            report += f"{dim:<16}"
            for j in range(len(project_names)):
                report += f" {optimizer.E[i, j]:7.3f}"
            report += "\n"

        report += "\n"

        # 3. A 矩阵（加权效率矩阵）
        report += "3️⃣ A 矩阵 (加权效率矩阵: A = Lambda @ E)\n"
        report += "-"*90 + "\n"
        report += "表示考虑历史增长率后，各投入项对技能改进的实际贡献:\n\n"
        report += f"{'维度':<16}"
        for j in range(len(project_names)):
            report += f"     x{j+1:1d}".rjust(8)
        report += "\n"
        report += "-"*90 + "\n"

        for i, dim in enumerate(dims):
            report += f"{dim:<16}"
            for j in range(len(project_names)):
                report += f" {optimizer.A[i, j]:7.4f}"
            report += "\n"

        report += "\n"

        # 4. 最优投入方案与矩阵的关系
        report += "4️⃣ 最优投入方案 x* 及其效果\n"
        report += "-"*90 + "\n"
        report += f"{'项目':<20} {'最优投入x*':>12} {'对整体改进的贡献':>50}\n"
        report += "-"*90 + "\n"

        x_star = result['x_star']
        for i, (x_val, proj) in enumerate(zip(x_star, project_names)):
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
        for j, proj in enumerate(project_names):
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
            4: '更新实验设备',
            5: '增加创业项目',
            6: '开展校园碳足迹数据分析竞赛'
        },
        'Trade': {
            0: '更新厨房设备',
            1: '引入AI设计软件',
            2: '增加营养学课程',
            3: '建立多感官工作坊',
            4: '开设行业大师班',
            5: '参加实习项目',
            6: '开设厨余垃圾再利用课程'
        },
        'Arts': {
            0: 'AI技术工作坊',
            1: '跨学科合作项目',
            2: '行业实习实践',
            3: '传统表演深化',
            4: '批判理论课程',
            5: '开设舞台沟通课程',
            6: '开设全球文化课程'
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
    for category_name in all_results.keys():
        schemes = all_results[category_name]['schemes']
        optimizer = all_results[category_name]['optimizer']
        # 使用综合发展方案作为主要输出方案
        result = schemes['comprehensive']['result']
        analysis = optimizer.analyze_results(result)
        x_star = result['x_star']
        dims = optimizer.dims

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
                # 获取项目名称
                project_name = all_results[category_name]['project_names'][idx] if idx < len(all_results[category_name]['project_names']) else f"项目_{idx+1}"
                report += f"{project_name:<30} {x_val*100:>10.1f}% {x_val:>12.4f} {priority:>12}\n"

        report += f"\n总投入预算: {analysis['allocation_sum']:.4f}\n\n"

        # 2. 预期效果
        report += "2️⃣ 技能改进预期\n"
        report += "-"*90 + "\n"
        report += f"{'技能维度':<20} {'当前':>8} {'预测':>8} {'目标':>8} {'改进':>8} {'达成率':>10}\n"
        report += "-"*90 + "\n"

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
        project_names_list = all_results[category_name]['project_names']
        for rank, idx in enumerate(top_indices, 1):
            if x_star[idx] > 0.001:
                project_name = project_names_list[idx] if idx < len(project_names_list) else f"项目_{idx+1}"
                report += f"{rank}. {project_name}\n"
                report += f"   投入: {x_star[idx]*100:.1f}%  |  预期效果: 提升技能能力\n\n"

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

        low_allocation = [project_names_list[i] for i in range(len(project_names_list)) if x_star[i] < 0.01 and x_star[i] > 0]
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
    main_extended()
    