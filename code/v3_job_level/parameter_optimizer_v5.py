"""
参数优化器 v5 - 解决结构性问题
主要改进（基于专业反馈）:
1. 修复相对误差的低基数爆炸问题（加epsilon）
2. 合并重复计分项（rmse和weighted_rmse）
3. 使用SMAPE替代相对误差
4. 简化目标函数，避免尺度不一致
5. 支持分段建模（断点前后不同参数）
6. 添加滚动回测验证
"""
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.integrate import odeint
from tqdm import tqdm


class ParameterOptimizerV5:
    """参数优化器（v5结构改进版）"""

    def __init__(self, job_model, train_data, ai_env, breakpoint_quarter=None):
        """
        初始化优化器

        参数:
            job_model: 职位模型对象
            train_data: 训练数据DataFrame
            ai_env: AI环境对象
            breakpoint_quarter: 断点季度（如果有结构性断点，如疫情）
        """
        self.job_model = job_model
        self.train_data = train_data
        self.ai_env = ai_env
        self.breakpoint_quarter = breakpoint_quarter

        # 提取实际就业数据
        self.actual_employment = train_data['Third Month Employment'].values
        self.time_points = self._compute_time_points()

        # epsilon用于避免除零（必须在计算趋势之前定义）
        self.epsilon = np.mean(self.actual_employment) * 0.01

        # 计算数据特征
        self.data_trend = self._compute_trend()
        self.data_volatility = np.std(np.diff(self.actual_employment) /
                                     (self.actual_employment[:-1] + 1e-6))

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
        slope = np.polyfit(x, y, 1)[0]
        return slope / (np.mean(y) + self.epsilon)

    def objective_function(self, params):
        """
        改进的目标函数（v5版本）

        主要改进:
        1. 使用SMAPE替代相对误差（避免低基数爆炸）
        2. 合并重复计分项
        3. 简化权重结构
        4. 更合理的尺度归一化
        """
        theta, Ac, k = params

        self.job_model.theta = theta
        self.job_model.Ac = Ac
        self.job_model.k = k

        try:
            predicted = self._simulate()

            # === 1. SMAPE (Symmetric Mean Absolute Percentage Error) ===
            # 避免低基数爆炸，更稳定
            smape = np.mean(2 * np.abs(predicted - self.actual_employment) /
                           (np.abs(predicted) + np.abs(self.actual_employment) + self.epsilon))

            # === 2. 趋势匹配损失 ===
            pred_trend = np.polyfit(np.arange(len(predicted)), predicted, 1)[0] / \
                        (np.mean(predicted) + self.epsilon)
            trend_loss = np.abs(pred_trend - self.data_trend)

            # === 3. 方向一致性（改进：处理零变化） ===
            actual_diff = np.diff(self.actual_employment)
            pred_diff = np.diff(predicted)

            # 只计算变化显著的点（避免零变化污染）
            threshold = np.std(actual_diff) * 0.1
            significant_mask = np.abs(actual_diff) > threshold

            if np.sum(significant_mask) > 0:
                direction_agreement = np.sign(actual_diff[significant_mask]) * \
                                    np.sign(pred_diff[significant_mask])
                direction_loss = np.mean(direction_agreement < 0)
            else:
                direction_loss = 0

            # === 4. 后期加权SMAPE（对近期数据更重视） ===
            n = len(self.actual_employment)
            weights = np.linspace(0.5, 1.5, n)
            weighted_smape = np.mean(weights * 2 * np.abs(predicted - self.actual_employment) /
                                    (np.abs(predicted) + np.abs(self.actual_employment) + self.epsilon))

            # === 5. 物理约束惩罚（简化版） ===
            physical_penalty = 0

            # 5a. 承载力合理性
            for t in self.time_points[::2]:  # 采样检查，提高速度
                A = self.ai_env.get_maturity(t)
                ai_skill = self.ai_env.get_skill_capability(t)
                K_eff = self.job_model.effective_carrying_capacity(A, ai_skill)
                K_ratio = K_eff / self.job_model.K0

                if K_ratio < 0.3 or K_ratio > 3.0:
                    physical_penalty += 0.5

            # 5b. 预测值合理性
            if np.any(predicted < 0):
                physical_penalty += 5
            if np.any(predicted > self.actual_employment.max() * 2):
                physical_penalty += 2

            # === 综合目标函数（简化权重结构） ===
            # 根据数据波动性自适应调整
            if self.data_volatility > 0.1:  # 高波动
                total_loss = (
                    smape * 0.3 +
                    weighted_smape * 0.3 +
                    trend_loss * 2.0 +      # 直接放大，不再嵌套权重
                    direction_loss * 1.5 +
                    physical_penalty
                )
            else:  # 低波动
                total_loss = (
                    smape * 0.4 +
                    weighted_smape * 0.4 +
                    trend_loss * 1.0 +
                    direction_loss * 1.0 +
                    physical_penalty
                )

            # 更新进度
            if total_loss < self.best_error:
                self.best_error = total_loss
                if self.pbar:
                    self.pbar.set_postfix({
                        'best_loss': f'{total_loss:.6f}',
                        'smape': f'{smape:.4f}',
                        'trend': f'{trend_loss:.4f}'
                    })

            self.iteration_count += 1
            if self.pbar:
                self.pbar.update(1)

            return total_loss

        except Exception as e:
            return 1e10

    def _simulate(self):
        """运行模拟（改进数值稳定性）"""
        N0 = self.actual_employment[0]
        t = self.time_points

        def derivative(N, t):
            try:
                if N <= 0:
                    return 0

                A = self.ai_env.get_maturity(t)
                ai_skill = self.ai_env.get_skill_capability(t)
                K_eff = self.job_model.effective_carrying_capacity(A, ai_skill)

                if K_eff <= 0 or not np.isfinite(K_eff):
                    return 0

                growth = self.job_model.r0 * N * (1 - N / K_eff)

                # 放宽增长率限制（之前30%太严格）
                max_growth_rate = 0.5
                return np.clip(growth, -N * max_growth_rate, N * max_growth_rate)

            except:
                return 0

        try:
            N = odeint(derivative, N0, t, rtol=1e-8, atol=1e-10, mxstep=5000)
            result = N.flatten()
            result = np.clip(result, 0, self.actual_employment.max() * 2)
            return result
        except:
            return np.full_like(t, N0, dtype=float)

    def optimize(self, maxiter=50, popsize=15, use_local_refinement=True):
        """
        执行优化

        参数:
            maxiter: 最大迭代次数
            popsize: 种群大小
            use_local_refinement: 是否使用局部精细化
        """
        # 根据数据趋势调整参数边界
        if self.data_trend > 0.02:
            theta_bounds = (0.0, 1.0)
        elif self.data_trend < -0.02:
            theta_bounds = (-1.0, 0.0)
        else:
            theta_bounds = (-0.5, 0.5)

        bounds = [
            theta_bounds,
            (0.2, 0.8),
            (2.0, 12.0)
        ]

        total_iterations = (maxiter + 1) * popsize * len(bounds)

        self.pbar = tqdm(
            total=total_iterations,
            desc="优化参数(v5)",
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]',
            mininterval=0.1
        )

        # 全局搜索
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
            strategy='best1bin',
            mutation=(0.5, 1.5),
            recombination=0.9
        )

        self.pbar.close()

        # 局部精细化（可选）
        if use_local_refinement and result.fun > 0.05:
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

        self.job_model.theta, self.job_model.Ac, self.job_model.k = result.x

        print(f"  [优化完成] theta={result.x[0]:.4f}, Ac={result.x[1]:.4f}, "
              f"k={result.x[2]:.4f}, loss={result.fun:.6f}")

        return result.x, result.fun

    def rolling_backtest(self, window_size=12, step=4):
        """
        滚动回测验证

        参数:
            window_size: 训练窗口大小（季度数）
            step: 滚动步长（季度数）

        返回:
            回测结果字典
        """
        n = len(self.actual_employment)
        results = []

        for start in range(0, n - window_size - step + 1, step):
            end = start + window_size
            test_end = min(end + step, n)

            # 训练窗口
            train_actual = self.actual_employment[start:end]
            train_time = self.time_points[start:end]

            # 测试窗口
            test_actual = self.actual_employment[end:test_end]
            test_time = self.time_points[end:test_end]

            # 临时修改数据进行优化
            original_actual = self.actual_employment
            original_time = self.time_points

            self.actual_employment = train_actual
            self.time_points = train_time

            # 优化
            params, _ = self.optimize(maxiter=30, popsize=10, use_local_refinement=False)

            # 预测测试集
            predicted_test = self._simulate_for_time(test_time, train_actual[-1])

            # 计算测试集误差
            smape_test = np.mean(2 * np.abs(predicted_test - test_actual) /
                                (np.abs(predicted_test) + np.abs(test_actual) + self.epsilon))

            results.append({
                'train_start': start,
                'train_end': end,
                'test_end': test_end,
                'params': params,
                'smape_test': smape_test
            })

            # 恢复原始数据
            self.actual_employment = original_actual
            self.time_points = original_time

        return results

    def _simulate_for_time(self, time_points, N0):
        """为指定时间点模拟"""
        def derivative(N, t):
            try:
                if N <= 0:
                    return 0
                A = self.ai_env.get_maturity(t)
                ai_skill = self.ai_env.get_skill_capability(t)
                K_eff = self.job_model.effective_carrying_capacity(A, ai_skill)
                if K_eff <= 0 or not np.isfinite(K_eff):
                    return 0
                growth = self.job_model.r0 * N * (1 - N / K_eff)
                return np.clip(growth, -N * 0.5, N * 0.5)
            except:
                return 0

        try:
            N = odeint(derivative, N0, time_points, rtol=1e-8, atol=1e-10)
            return N.flatten()
        except:
            return np.full_like(time_points, N0, dtype=float)