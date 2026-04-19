"""
主程序 - 职位级时间序列预测
"""
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from data_processor import DataProcessor
from job_model import JobModel
from parameter_optimizer import ParameterOptimizer
from time_series_predictor import TimeSeriesPredictor
from ai_environment_v2 import AIEnvironmentV2


def main():
    """主函数"""
    print("=" * 60)
    print("职位级生态协同进化模型 - 时间序列预测")
    print("=" * 60)

    # 1. 加载数据
    print("\n[1/5] 加载数据...")
    data_file = r"c:\Users\Wu\Desktop\美赛F题数据集\代码\程序所需数据.xlsx"
    processor = DataProcessor(data_file)

    # 2. 初始化AI环境
    print("\n[2/5] 初始化AI环境...")
    df_events = processor.get_ai_events()
    ai_env = AIEnvironmentV2(df_events)
    print(f"[OK] AI环境已初始化")

    # 3. 获取所有职位
    print("\n[3/5] 处理职位数据...")
    all_jobs = processor.get_all_jobs()
    print(f"[OK] 找到 {len(all_jobs)} 个职位")

    # 选择前5个职位进行演示
    selected_jobs = all_jobs[:5]
    print(f"[INFO] 演示模式：处理前 {len(selected_jobs)} 个职位")

    results = []

    # 4. 对每个职位进行建模和预测
    print("\n[4/5] 开始建模和预测...")
    for idx, job_info in enumerate(selected_jobs):
        print(f"\n处理职位 {idx+1}/{len(selected_jobs)}: {job_info['title']}")

        # 创建职位模型
        model = JobModel(job_info['data'], job_info['type'])

        # 分割训练集和测试集
        train_data, test_data = processor.split_train_test(job_info['data'])

        if len(train_data) < 5:
            print(f"  [跳过] 训练数据不足")
            continue

        # 参数优化
        print(f"  优化参数...")
        optimizer = ParameterOptimizer(model, train_data, ai_env)
        best_params, error = optimizer.optimize()
        print(f"  [OK] 训练误差: {error:.4f}")

        # 预测未来
        print(f"  预测未来...")
        predictor = TimeSeriesPredictor(model, ai_env)
        last_employment = train_data['Third Month Employment'].iloc[-1]
        t_pred, N_pred = predictor.predict(start_time=0, end_time=40, N0=last_employment)

        # 保存结果
        results.append({
            'title': job_info['title'],
            'type': job_info['type'],
            'theta': model.theta,
            'Ac': model.Ac,
            'k': model.k,
            'train_error': error,
            'predictions': N_pred
        })

    # 5. 保存结果
    print("\n[5/5] 保存结果...")
    save_results(results)
    print("\n" + "=" * 60)
    print("程序执行完毕")
    print("=" * 60)


def save_results(results):
    """保存预测结果"""
    # 保存参数
    params_df = pd.DataFrame([{
        'title': r['title'],
        'type': r['type'],
        'theta': r['theta'],
        'Ac': r['Ac'],
        'k': r['k'],
        'train_error': r['train_error']
    } for r in results])

    params_df.to_csv('job_parameters.csv', index=False, encoding='utf-8-sig')
    print("[OK] 参数已保存至 job_parameters.csv")

    # 保存预测结果
    for r in results:
        filename = f"prediction_{r['title'].replace(' ', '_')}.csv"
        pred_df = pd.DataFrame({
            'Quarter': range(len(r['predictions'])),
            'Predicted_Employment': r['predictions']
        })
        pred_df.to_csv(filename, index=False, encoding='utf-8-sig')

    print(f"[OK] 预测结果已保存")


if __name__ == "__main__":
    main()


