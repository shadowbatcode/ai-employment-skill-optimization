"""
数据预处理器 - 处理职位级数据
"""
import pandas as pd
import numpy as np
from job_model import JobModel


class DataProcessor:
    """数据预处理器"""

    def __init__(self, file_path):
        """
        初始化数据处理器

        参数:
            file_path: Excel文件路径
        """
        self.file_path = file_path
        self.df_base = None
        self.df_influence = None
        self.load_data()

    def load_data(self):
        """加载数据"""
        print("加载数据...")
        self.df_base = pd.read_excel(self.file_path, sheet_name='base')
        self.df_influence = pd.read_excel(self.file_path, sheet_name='influence')
        print(f"[OK] 加载了 {len(self.df_base)} 条职位记录")
        print(f"[OK] 加载了 {len(self.df_influence)} 个AI事件")

    def get_all_jobs(self):
        """
        获取所有职位的列表

        返回:
            职位字典列表
        """
        jobs = []
        grouped = self.df_base.groupby(['type', 'title'])

        for (job_type, title), group in grouped:
            job_info = {
                'type': job_type,
                'title': title,
                'data': group.sort_values(['Year', 'Quarter']),
                'naics': group['NAICS'].iloc[0],
                'onet': group['O*NET'].iloc[0]
            }
            jobs.append(job_info)

        return jobs

    def split_train_test(self, job_data, train_end_year=2023, train_end_quarter=4):
        """
        分割训练集和测试集

        参数:
            job_data: 职位数据DataFrame
            train_end_year: 训练集结束年份
            train_end_quarter: 训练集结束季度

        返回:
            train_data, test_data
        """
        train_mask = (job_data['Year'] < train_end_year) | \
                     ((job_data['Year'] == train_end_year) &
                      (job_data['Quarter'] <= train_end_quarter))

        train_data = job_data[train_mask].copy()
        test_data = job_data[~train_mask].copy()

        return train_data, test_data

    def get_ai_events(self):
        """获取AI事件数据"""
        return self.df_influence.copy()

