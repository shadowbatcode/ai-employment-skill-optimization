"""
参数优化器 v4 - 改进目标函数和约束条件以提高拟合效果
主要改进:
1. 多目标优化: 同时考虑RMSE、趋势匹配、方向一致性
2. 自适应权重: 根据数据特征动态调整各项权重
3. 增强约束: 添加物理约束和合理性检查
4. 分段拟合: 对训练集前期和后期分别优化
"""
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.integrate import odeint
from tqdm import tqdm


class ParameterOptimizerV4:
    """参数优化器（v4改进版）"""

    def __init__(self, job_model, train_data, ai_env):
        """
        初始化优化器

        参数:
            job_model: 职位模型对象
            train_data: 训练数据DataFrame
            ai_env: AI环境对象
        """
        self.job_model = job_model
        self.train_data = train_data
        self.ai_env = ai_env

        # 提取实际就业数据
        self.actual_employment = train_data['Third Month Employment'].values
        self.time_points = self._compute_time_points()

        # 计算数据特征用于自适应权重
        self.data_trend = self._compute_trend()
        self.data_volatility = np.std(np.diff(self.actual_employment) / self.actual_employment[:-1])

        # 进度追踪
        self.iteration_count = 0
        self.best_error = float('inf')
        self.pbar = None

    def _compute_time_points(self):
        """计算时间点（季度）"""
        years = self.train_data['Year'].values
        quarters = self.train_data['Quarter'].values
        return (years - 2020) * 4 + (quarters - 1)

    def _compute_trend(self):
        """计算数据整体趋势（线性回归斜率）"""
        if len(self.actual_employment) < 2:
            return 0
        x = np.arange(len(self.actual_employment))
        y = self.actual_employment
        # 简单线性回归
        slope = np.polyfit(x, y, 1)[0]
        return slope / np.mean(y)  # 归一化斜率

    def objective_function(self, params):
        """
        改进的多目标函数

        组成部分:
        1. RMSE: 相对均方根误差
        2. Trend Loss: 趋势匹配损失
        3. Direction Loss: 方向一致性损失
        4. Smoothness Penalty: 平滑度惩罚
        5. Physical Constraint Penalty: 物理约束惩罚
        """
        theta, Ac, k = params

        self.job_model.theta = theta
        self.job_model.Ac = Ac
        self.job_model.k = k

        try:
            predicted = self._simulate()

            # 1. 相对均方根误差 (RMSE)
            relative_errors = (predicted - self.actual_employment) / self.actual_employment
            rmse = np.sqrt(np.mean(relative_errors ** 2))

            # 2. 趋势匹配损失
            # 计算预测值的趋势
            pred_trend = np.polyfit(np.arange(len(predicted)), predicted, 1)[0] / np.mean(predicted)
            trend_loss = abs(pred_trend - self.data_trend) * 10  # 放大趋势差异的权重

            # 3. 方向一致性损失（预测变化方向与实际变化方向的一致性）
            actual_diff = np.diff(self.actual_employment)
            pred_diff = np.diff(predicted)
            # 计算方向一致性（同号为1，异号为-1）
            direction_agreement = np.sign(actual_diff) * np.sign(pred_diff)
            direction_loss = np.mean(direction_agreement < 0) * 5  # 方向不一致的比例

            # 4. 分段误差（对后期数据给予更高权重）
            n = len(self.actual_employment)
            weights = np.linspace(0.5, 1.5, n)  # 后期权重更高
            weighted_rmse = np.sqrt(np.mean(weights * relative_errors ** 2))

            # 5. 平滑度惩罚（避免预测曲线过于震荡）
            pred_second_diff = np.diff(pred_diff)
            smoothness_penalty = np.std(pred_second_diff) / np.mean(np.abs(predicted)) * 0.5

            # 6. 物理约束惩罚
            physical_penalty = 0

            # 6a. 承载力合理性检查
            for t in self.time_points:
                A = self.ai_env.get_maturity(t)
                ai_skill = self.ai_env.get_skill_capability(t)
                K_eff = self.job_model.effective_carrying_capacity(A, ai_skill)

                # 承载力不应过小或过大
                K_ratio = K_eff / self.job_model.K0
                if K_ratio < 0.3 or K_ratio > 3.0:
                    physical_penalty += abs(np.log(K_ratio)) * 0.5

            # 6b. 预测值不应出现负数或异常大的值
            if np.any(predicted < 0):
                physical_penalty += 10
            if np.any(predicted > self.actual_employment.max() * 2):
                physical_penalty += 5

            # 6c. theta和Ac的合理性组合检查
            # 如果theta为负（替代效应），Ac应该较小（早期影响）
            # 如果theta为正（增强效应），Ac可以较大（后期影响）
            if theta < 0 and Ac > 0.7:
                physical_penalty += 2
            if theta > 0.5 and Ac < 0.2:
                physical_penalty += 2

            # 综合目标函数（加权组合）
            # 根据数据波动性自适应调整权重
            if self.data_volatility > 0.1:  # 高波动数据
                total_loss = (
                    rmse * 0.4 +           # RMSE权重降低
                    trend_loss * 0.3 +     # 趋势权重提高
                    direction_loss * 0.2 + # 方向一致性权重提高
                    weighted_rmse * 0.3 +  # 加权RMSE
                    smoothness_penalty * 0.1 +
                    physical_penalty
                )
            else:  # 低波动数据
                total_loss = (
                    rmse * 0.5 +           # RMSE权重提高
                    trend_loss * 0.2 +     # 趋势权重降低
                    direction_loss * 0.15 +
                    weighted_rmse * 0.25 +
                    smoothness_penalty * 0.05 +
                    physical_penalty
                )

            # 更新进度
            if total_loss < self.best_error:
                self.best_error = total_loss
                if self.pbar:
                    self.pbar.set_postfix({
                        'best_loss': f'{total_loss:.6f}',
                        'rmse': f'{rmse:.4f}',
                        'trend': f'{trend_loss:.4f}'
                    })

            self.iteration_count += 1
            if self.pbar:
                self.pbar.update(1)

            return total_loss

        except Exception as e:
            # 如果模拟失败，返回极大惩罚
            return 1e10

    def _simulate(self):
        """运行模拟（改进数值稳定性）"""
        N0 = self.actual_employment[0]
        t = self.time_points

        def derivative(N, t):
            try:
                # 确保N为正数
                if N <= 0:
                    return 0

                A = self.ai_env.get_maturity(t)
                ai_skill = self.ai_env.get_skill_capability(t)
                K_eff = self.job_model.effective_carrying_capacity(A, ai_skill)

                # 数值稳定性检查
                if K_eff <= 0 or not np.isfinite(K_eff):
                    return 0

                # Logistic增长
                growth = self.job_model.r0 * N * (1 - N / K_eff)

                # 限制增长率避免数值爆炸（更严格的限制）
                max_growth_rate = 0.3  # 每季度最大30%增长
                return np.clip(growth, -N * max_growth_rate, N * max_growth_rate)

            except:
                return 0

        try:
            # 使用更严格的容差和更小的步长
            N = odeint(derivative, N0, t, rtol=1e-8, atol=1e-10, mxstep=5000)
            result = N.flatten()

            # 后处理：确保结果合理
            result = np.clip(result, 0, self.actual_employment.max() * 2)

            return result
        except:
            # 如果求解失败，返回常数序列
            return np.full_like(t, N0, dtype=float)

    def optimize(self, maxiter=50, popsize=15):
        """
        执行优化（增加迭代次数和种群大小以提高质量）

        参数:
            maxiter: 最大迭代次数（默认50，比v3增加）
            popsize: 种群大小（默认15，比v3增加）
        """
        # 根据职位类型和数据特征调整参数边界
        if self.data_trend > 0:  # 增长趋势
            theta_bounds = (0.0, 1.0)  # 偏向增强效应
        elif self.data_trend < -0.05:  # 明显下降趋势
            theta_bounds = (-1.0, 0.0)  # 偏向替代效应
        else:  # 平稳或轻微变化
            theta_bounds = (-0.5, 0.5)  # 中性范围

        bounds = [
            theta_bounds,      # theta: 根据趋势调整
            (0.2, 0.8),       # Ac: 收窄范围，避免极端值
            (2.0, 12.0)       # k: 扩大范围，允许更陡峭或平缓的响应
        ]

        # 估算总迭代次数
        total_iterations = (maxiter + 1) * popsize * len(bounds)

        # 创建进度条
        self.pbar = tqdm(
            total=total_iterations,
            desc="优化参数(v4)",
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]',
            mininterval=0.1
        )

        # 第一阶段：全局搜索（differential_evolution）
        result = differential_evolution(
            self.objective_function,
            bounds,
            maxiter=maxiter,
            popsize=popsize,
            seed=42,
            workers=1,
            polish=False,
            atol=1e-5,
            tol=1e-4,
            strategy='best1bin',  # 使用更激进的策略
            mutation=(0.5, 1.5),  # 增加变异范围
            recombination=0.9     # 提高重组率
        )

        self.pbar.close()

        # 第二阶段：局部精细化（可选，如果全局搜索结果不够好）
        if result.fun > 0.1:  # 如果误差仍然较大
            print("  [INFO] 执行局部精细化优化...")
            result_local = minimize(
                self.objective_function,
                result.x,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-6}
            )
            if result_local.fun < result.fun:
                result = result_local

        # 设置最优参数
        self.job_model.theta, self.job_model.Ac, self.job_model.k = result.x

        print(f"  [优化完成] theta={result.x[0]:.4f}, Ac={result.x[1]:.4f}, k={result.x[2]:.4f}, loss={result.fun:.6f}")

        return result.x, result.fun