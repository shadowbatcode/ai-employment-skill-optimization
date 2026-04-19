"""
数据分析脚本 - 读取基础环境承载量数据
"""
import pandas as pd
import numpy as np

# 读取Excel文件
file_path = r"c:\Users\Wu\Desktop\美赛F题数据集\代码\基础环境承载量.xlsx"

try:
    # 尝试读取所有sheet
    excel_file = pd.ExcelFile(file_path)
    print("=" * 60)
    print("Excel文件包含的Sheet:")
    print("=" * 60)
    for sheet_name in excel_file.sheet_names:
        print(f"- {sheet_name}")

    print("\n" + "=" * 60)
    print("各Sheet数据预览:")
    print("=" * 60)

    for sheet_name in excel_file.sheet_names:
        print(f"\n### Sheet: {sheet_name}")
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        print(f"形状: {df.shape}")
        print(f"列名: {list(df.columns)}")
        print("\n前5行数据:")
        print(df.head())
        print("\n数据类型:")
        print(df.dtypes)
        print("\n基本统计:")
        print(df.describe())
        print("\n" + "-" * 60)

except Exception as e:
    print(f"读取文件时出错: {e}")
