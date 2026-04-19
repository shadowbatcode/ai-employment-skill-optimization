"""
竞争矩阵构建器 - 基于O*NET相似度构建职位间竞争关系
"""
import pandas as pd
import numpy as np


class CompetitionMatrixBuilder:
    """竞争矩阵构建器"""

    def __init__(self, df_related):
        """
        初始化构建器

        参数:
            df_related: Related表的DataFrame
        """
        self.df_related = df_related

    def build_matrix(self, job_list, competition_strength=0.1):
        """
        构建职位间竞争矩阵

        参数:
            job_list: 职位信息列表
            competition_strength: 竞争强度系数（默认0.1）

        返回:
            竞争矩阵 (n×n)
        """
        n = len(job_list)
        alpha_matrix = np.zeros((n, n))

        print(f"构建 {n}×{n} 竞争矩阵...")

        for i in range(n):
            for j in range(n):
                if i != j:
                    similarity = self._get_similarity(
                        job_list[i]['onet'],
                        job_list[j]['onet']
                    )
                    alpha_matrix[i, j] = similarity * competition_strength

        return alpha_matrix

    def _get_similarity(self, onet1, onet2):
        """
        获取两个职位的相似度

        参数:
            onet1: 职位1的O*NET代码
            onet2: 职位2的O*NET代码

        返回:
            相似度 (0~1)
        """
        # 清理O*NET代码
        onet1_clean = str(onet1).strip().split(',')[0] if pd.notna(onet1) else ''
        onet2_clean = str(onet2).strip().split(',')[0] if pd.notna(onet2) else ''

        if not onet1_clean or not onet2_clean:
            return 0.0

        # 在Related表中查找相似度
        mask = (self.df_related['O*NET-SOC Code'] == onet1_clean) & \
               (self.df_related['O*NET Code'] == onet2_clean)

        if mask.any():
            # 使用综合相关性列
            similarity = self.df_related.loc[mask, '综合相关性'].values[0]
            # 归一化到0-1范围
            return min(1.0, similarity / 3.0)

        return 0.0

        