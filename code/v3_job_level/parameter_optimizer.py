"""
参数优化器 - 使用历史数据优化模型参数
"""
import numpy as np
from scipy.optimize import differential_evolution
from scipy.integrate import odeint


class ParameterOptimizer:
    """参数优化器（使用差分进化算法）"""

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

    def _compute_time_points(self):
        """计算时间点（季度）"""
        years = self.train_data['Year'].values
        quarters = self.train_data['Quarter'].values
        # 2020Q1 = 0
        return (years - 2020) * 4 + (quarters - 1)

    def objective_function(self, params):
        """目标函数：最小化预测误差"""
        # 解包参数
        theta, Ac, k = params

        # 设置参数
        self.job_model.theta = theta
        self.job_model.Ac = Ac
        self.job_model.k = k

        # 模拟预测
        try:
            predicted = self._simulate()
            # 计算相对误差
            error = np.mean(((predicted - self.actual_employment) / self.actual_employment) ** 2)
            return error
        except:
            return 1e10  # 返回大值表示失败

    def _simulate(self):
        """运行模拟"""
        N0 = self.actual_employment[0]
        t = self.time_points

        def derivative(N, t):
            A = self.ai_env.get_maturity(t)
            ai_skill = self.ai_env.get_skill_capability(t)
            K_eff = self.job_model.effective_carrying_capacity(A, ai_skill)
            return self.job_model.r0 * N * (1 - N / K_eff) if K_eff > 0 else 0

        N = odeint(derivative, N0, t)
        return N.flatten()

    def optimize(self):
        """执行优化"""
        bounds = [
            (-1.0, 1.0),   # theta
            (0.1, 0.9),    # Ac
            (1.0, 10.0)    # k
        ]

        result = differential_evolution(
            self.objective_function,
            bounds,
            maxiter=50,
            popsize=10,
            seed=42
        )

        # 设置最优参数
        self.job_model.theta, self.job_model.Ac, self.job_model.k = result.x
        return result.x, result.fun

