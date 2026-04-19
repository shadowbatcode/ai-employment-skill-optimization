"""
主程序 v4 - 职位级时间序列预测（改进拟合效果版）
主要改进:
1. 使用v4优化器：多目标优化函数
2. 趋势匹配：确保预测趋势与真实趋势一致
3. 方向一致性：确保变化方向正确
4. 自适应参数边界：根据数据特征调整搜索空间
5. 增强约束：物理合理性检查
"""
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append('..')

from data_processor import DataProcessor
from job_model import JobModel
from parameter_optimizer_v4 import ParameterOptimizerV4
from time_series_predictor import TimeSeriesPredictor
from competition_matrix import CompetitionMatrixBuilder
sys.path.append('../v2_improved')
from ai_environment_v2 import AIEnvironmentV2

from multiprocessing import Pool, cpu_count
import os


def process_single_job(args):
    """处理单个职位（用于多进程）"""
    job_type, job_title, data_file = args

    try:
        # 重新加载必要的对象（每个进程独立）
        processor = DataProcessor(data_file)
        df_events = processor.get_ai_events()
        ai_env = AIEnvironmentV2(df_events)

        # 从processor中重新获取该职位的数据
        all_jobs = processor.get_all_jobs()
        job_info = None
        for job in all_jobs:
            if job['title'] == job_title and job['type'] == job_type:
                job_info = job
                break

        if job_info is None:
            print(f"  [错误] 未找到职位: {job_title}")
            return None

        # 创建职位模型
        model = JobModel(job_info['data'], job_info['type'])

        # 分割训练集和测试集
        train_data, test_data = processor.split_train_test(job_info['data'])

        if len(train_data) < 5:
            print(f"  [跳过] {job_info['title']}: 训练数据不足")
            return None

        # 参数优化（使用v4优化器）
        print(f"  [进程] 开始优化: {job_info['title']}")
        optimizer = ParameterOptimizerV4(model, train_data, ai_env)
        best_params, error = optimizer.optimize(maxiter=50, popsize=15)

        # 预测未来
        predictor = TimeSeriesPredictor(model, ai_env)
        last_employment = train_data['Third Month Employment'].iloc[-1]
        t_pred, N_pred = predictor.predict(start_time=0, end_time=40, N0=last_employment)

        # 获取测试集真实值（如果有）
        test_actual = None
        if len(test_data) > 0:
            test_actual = test_data['Third Month Employment'].values

        print(f"  [完成] {job_info['title']}: 综合损失={error:.6f}")

        return {
            'title': job_info['title'],
            'type': job_info['type'],
            'theta': model.theta,
            'Ac': model.Ac,
            'k': model.k,
            'train_error': error,
            'predictions': N_pred,
            'test_actual': test_actual,
            'train_size': len(train_data),
            'test_size': len(test_data)
        }
    except Exception as e:
        import traceback
        print(f"  [错误] {job_title}: {str(e)}")
        print(traceback.format_exc())
        return None


def main():
    """主函数（多进程版本）"""
    print("=" * 60)
    print("职位级生态协同进化模型 v4 - 改进拟合效果版")
    print("改进点：")
    print("  1. 多目标优化：RMSE + 趋势匹配 + 方向一致性")
    print("  2. 自适应权重：根据数据波动性调整")
    print("  3. 增强约束：物理合理性检查")
    print("  4. 分段优化：对不同时期分别优化")
    print("  5. 局部精细化：全局搜索后进行局部优化")
    print("=" * 60)

    # 1. 加载数据
    print("\n[1/6] 加载数据...")
    data_file = r"../程序所需数据.xlsx"
    processor = DataProcessor(data_file)

    # 2. 初始化AI环境
    print("\n[2/6] 初始化AI环境...")
    df_events = processor.get_ai_events()
    ai_env = AIEnvironmentV2(df_events)
    print(f"[OK] AI环境已初始化")

    # 3. 获取职位并构建竞争矩阵
    print("\n[3/6] 处理职位数据...")
    all_jobs = processor.get_all_jobs()
    print(f"[OK] 找到 {len(all_jobs)} 个职位")

    # 处理所有职位
    selected_jobs = all_jobs[:]
    print(f"[INFO] 处理 {len(selected_jobs)} 个职位")

    # 构建竞争矩阵
    print("\n[4/6] 构建竞争矩阵...")
    df_related = pd.read_excel(data_file, sheet_name='Related')
    matrix_builder = CompetitionMatrixBuilder(df_related)
    alpha_matrix = matrix_builder.build_matrix(selected_jobs, competition_strength=0.1)
    print(f"[OK] 竞争矩阵已构建")

    # 5. 多进程处理职位
    print("\n[5/6] 开始多进程建模和预测...")

    # 获取CPU核心数
    num_processes = max(1, cpu_count() - 1)  # 保留一个核心给系统
    print(f"[INFO] 使用 {num_processes} 个进程并行处理")

    # 准备参数（只传递可序列化的简单数据）
    job_args = [(job_info['type'], job_info['title'], data_file) for job_info in selected_jobs]

    # 使用进程池处理
    results = []
    with Pool(processes=num_processes) as pool:
        for result in pool.imap_unordered(process_single_job, job_args):
            if result is not None:
                results.append(result)

    print(f"\n[OK] 完成 {len(results)}/{len(selected_jobs)} 个职位的处理")

    # 6. 保存结果
    print("\n" + "="*60)
    print("[6/6] 保存结果...")
    save_results(results)
    print("\n" + "=" * 60)
    print("程序执行完毕")
    print("=" * 60)


def save_results(results):
    """保存预测结果（追加模式，不覆盖已有数据）"""
    if len(results) == 0:
        print("[警告] 没有结果可保存")
        return

    # 保存参数（追加模式）
    params_df = pd.DataFrame([{
        'title': r['title'],
        'type': r['type'],
        'theta': r['theta'],
        'Ac': r['Ac'],
        'k': r['k'],
        'train_error': r['train_error']
    } for r in results])

    csv_file = 'job_parameters_v4.csv'

    # 检查文件是否存在
    if os.path.exists(csv_file):
        # 读取已有数据
        existing_df = pd.read_csv(csv_file, encoding='utf-8-sig')
        # 合并新旧数据，去除重复的职位（保留最新的）
        combined_df = pd.concat([existing_df, params_df], ignore_index=True)
        # 按title去重，保留最后出现的（即最新的）
        combined_df = combined_df.drop_duplicates(subset=['title'], keep='last')
        combined_df.to_csv(csv_file, index=False, encoding='utf-8-sig')
        print(f"[OK] 参数已追加至 {csv_file} (共{len(combined_df)}条记录)")
    else:
        # 文件不存在，直接保存
        params_df.to_csv(csv_file, index=False, encoding='utf-8-sig')
        print(f"[OK] 参数已保存至 {csv_file}")

    # 保存所有预测结果到一个CSV文件（包含真实值对比）
    all_predictions = []

    for r in results:
        predictions = r['predictions']
        test_actual = r.get('test_actual', None)
        train_size = r.get('train_size', 0)

        # 为每个季度创建一行数据
        for quarter_idx in range(len(predictions)):
            row = {
                'Job_Title': r['title'],
                'Job_Type': r['type'],
                'Quarter': quarter_idx,
                'Predicted_Employment': predictions[quarter_idx]
            }

            # 如果这个季度有真实值（在测试集范围内）
            if test_actual is not None and quarter_idx >= train_size and (quarter_idx - train_size) < len(test_actual):
                actual_idx = quarter_idx - train_size
                row['Actual_Employment'] = test_actual[actual_idx]
                row['Prediction_Error'] = predictions[quarter_idx] - test_actual[actual_idx]
                row['Relative_Error_%'] = ((predictions[quarter_idx] - test_actual[actual_idx]) / test_actual[actual_idx] * 100) if test_actual[actual_idx] != 0 else 0
            else:
                row['Actual_Employment'] = None
                row['Prediction_Error'] = None
                row['Relative_Error_%'] = None

            all_predictions.append(row)

    # 保存合并的预测结果
    predictions_file = 'all_predictions_v4.csv'
    predictions_df = pd.DataFrame(all_predictions)
    predictions_df.to_csv(predictions_file, index=False, encoding='utf-8-sig')
    print(f"[OK] 所有预测结果已保存至 {predictions_file} (共{len(predictions_df)}行)")

    # 统计有真实值对比的数据
    with_actual = predictions_df['Actual_Employment'].notna().sum()
    if with_actual > 0:
        print(f"[INFO] 其中 {with_actual} 行包含真实值对比")


if __name__ == "__main__":
    # Windows平台需要freeze_support以支持多进程
    from multiprocessing import freeze_support
    freeze_support()
    main()