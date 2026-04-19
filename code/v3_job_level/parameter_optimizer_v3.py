"""
参数优化器 v3 - 优化速度并添加进度可视化
"""
import numpy as np
from scipy.optimize import differential_evolution
from scipy.integrate import odeint
from tqdm import tqdm


class ParameterOptimizerV3:
    """参数优化器（改进版）"""

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

        # 进度追踪
        self.iteration_count = 0
        self.best_error = float('inf')
        self.pbar = None

    def _compute_time_points(self):
        """计算时间点（季度）"""
        years = self.train_data['Year'].values
        quarters = self.train_data['Quarter'].values
        return (years - 2020) * 4 + (quarters - 1)

    def objective_function(self, params):
        """目标函数：最小化预测误差（带进度更新）"""
        theta, Ac, k = params

        self.job_model.theta = theta
        self.job_model.Ac = Ac
        self.job_model.k = k

        try:
            predicted = self._simulate()
            error = np.mean(((predicted - self.actual_employment) / self.actual_employment) ** 2)

            # 更新进度
            if error < self.best_error:
                self.best_error = error
                if self.pbar:
                    self.pbar.set_postfix({'best_error': f'{error:.6f}'})

            self.iteration_count += 1
            if self.pbar:
                self.pbar.update(1)

            return error
        except:
            return 1e10

    def _simulate(self):
        """运行模拟（添加异常处理和数值稳定性检查）"""
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

                growth = self.job_model.r0 * N * (1 - N / K_eff)

                # 限制增长率避免数值爆炸
                return np.clip(growth, -N * 0.5, N * 0.5)
            except:
                return 0

        try:
            N = odeint(derivative, N0, t, rtol=1e-6, atol=1e-8)
            return N.flatten()
        except:
            # 如果求解失败，返回常数序列
            return np.full_like(t, N0, dtype=float)

    def optimize(self, maxiter=30, popsize=8):
        """
        执行优化（减少迭代次数以提速）

        参数:
            maxiter: 最大迭代次数（默认30，原50）
            popsize: 种群大小（默认8，原10）
        """
        bounds = [
            (-1.0, 1.0),   # theta
            (0.1, 0.9),    # Ac
            (1.0, 10.0)    # k
        ]

        # 估算总迭代次数（differential_evolution实际调用次数更多）
        # 通常是 (maxiter + 1) * popsize * len(bounds)
        total_iterations = (maxiter + 1) * popsize * len(bounds)

        # 创建进度条
        self.pbar = tqdm(total=total_iterations, desc="优化参数",
                        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]',
                        mininterval=0.1)  # 更新间隔0.1秒

        result = differential_evolution(
            self.objective_function,
            bounds,
            maxiter=maxiter,
            popsize=popsize,
            seed=42,
            workers=1,  # 单线程以保证进度条正常
            polish=False,  # 禁用最终抛光步骤以加速
            atol=1e-4,  # 放宽收敛容差
            tol=1e-3
        )

        self.pbar.close()

        # 设置最优参数
        self.job_model.theta, self.job_model.Ac, self.job_model.k = result.x
        return result.x, result.fun


