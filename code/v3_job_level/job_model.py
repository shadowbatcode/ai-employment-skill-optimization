"""
职位级模型类 - 每个具体职位的独立模型
"""
import numpy as np
import pandas as pd


class JobModel:
    """单个职位的生态演化模型"""

    def __init__(self, job_data, job_type):
        """
        初始化职位模型

        参数:
            job_data: 职位的历史数据DataFrame
            job_type: 职位所属大类 (STEM/Trade/Art)
        """
        self.job_data = job_data
        self.job_type = job_type
        self.title = job_data['title'].iloc[0] if 'title' in job_data.columns else 'Unknown'
        self.naics = job_data['NAICS'].iloc[0] if 'NAICS' in job_data.columns else None
        self.onet = job_data['O*NET '].iloc[0] if 'O*NET ' in job_data.columns else None

        # 从数据计算基础参数
        self._compute_base_parameters()

        # 初始化可学习参数（默认值）
        self._initialize_learnable_parameters()

    def _compute_base_parameters(self):
        """从历史数据计算基础参数"""
        employment = self.job_data['Third Month Employment'].values
        growth_rates = self.job_data['Third Month\nEmployment\n% Change\nOver the Year'].values

        # 初始规模（95%分位数）
        self.N0 = np.percentile(employment, 95)

        # 基准增长率（平均值）
        valid_rates = growth_rates[~np.isnan(growth_rates)]
        self.r0 = np.mean(valid_rates) if len(valid_rates) > 0 else 0.02

        # 基础承载力（初始规模的1.2倍）
        self.K0 = self.N0 * 1.2

    def _initialize_learnable_parameters(self):
        """初始化可学习参数（默认值）"""
        # 根据职位类型设置默认值
        if self.job_type == 'STEM':
            self.theta = 0.5  # 增强敏感
            self.Ac = 0.3
            self.k = 7.0
        elif self.job_type == 'Trade':
            self.theta = -0.1  # 轻微替代
            self.Ac = 0.5
            self.k = 4.0
        else:  # Art
            self.theta = 0.1  # 轻微增强
            self.Ac = 0.4
            self.k = 6.0

        # 技能向量（5维）
        self.G = np.array([0.5, 0.5, 0.5, 0.5, 0.5])

        # 经济参数
        self.cost_reduce = 0.3
        self.epsilon = 0.8

    def set_parameters(self, params_dict):
        """设置可学习参数"""
        for key, value in params_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def get_parameters(self):
        """获取当前参数"""
        return {
            'theta': self.theta,
            'Ac': self.Ac,
            'k': self.k,
            'G': self.G.copy(),
            'cost_reduce': self.cost_reduce,
            'epsilon': self.epsilon
        }

    def sigmoid_response(self, A):
        """S型响应函数"""
        return 1.0 / (1.0 + np.exp(-self.k * (A - self.Ac)))

    def S_function(self, A):
        """零点平移的S型函数"""
        return self.sigmoid_response(A) - 0.5

    def ai_compatibility(self, ai_skill_vector):
        """AI技能兼容性函数"""
        dot_product = np.dot(self.G, ai_skill_vector)
        norm_G = np.linalg.norm(self.G)
        norm_A = np.linalg.norm(ai_skill_vector)

        if norm_G == 0 or norm_A == 0:
            return 0.5

        return (dot_product / (norm_G * norm_A) + 1) / 2

    def effective_carrying_capacity(self, A, ai_skill_vector, g0=0.02, eta=0.05):
        """计算有效承载力"""
        # 技术因子
        S = self.S_function(A)
        C = self.ai_compatibility(ai_skill_vector)
        f_t = 1.0 + self.theta * S * C

        # 经济因子
        E_price = self.epsilon * self.cost_reduce * A
        E_income = g0 + eta * A
        f_c = E_price + E_income

        return self.K0 * f_t * f_c
