"""
时间序列预测器 - 使用优化后的参数进行预测
"""
import numpy as np
import pandas as pd
from scipy.integrate import odeint


class TimeSeriesPredictor:
    """时间序列预测器"""

    def __init__(self, job_model, ai_env):
        """
        初始化预测器

        参数:
            job_model: 优化后的职位模型
            ai_env: AI环境对象
        """
        self.job_model = job_model
        self.ai_env = ai_env

    def predict(self, start_time, end_time, N0):
        """
        预测未来就业规模

        参数:
            start_time: 起始时间（季度）
            end_time: 结束时间（季度）
            N0: 初始规模

        返回:
            时间数组, 预测值数组
        """
        t = np.arange(start_time, end_time, 0.25)

        def derivative(N, t):
            A = self.ai_env.get_maturity(t)
            ai_skill = self.ai_env.get_skill_capability(t)
            K_eff = self.job_model.effective_carrying_capacity(A, ai_skill)
            if K_eff > 0:
                return self.job_model.r0 * N * (1 - N / K_eff)
            return 0

        N = odeint(derivative, N0, t)
        return t, N.flatten()
