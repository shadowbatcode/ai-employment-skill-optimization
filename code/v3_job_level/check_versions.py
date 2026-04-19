"""
版本管理工具 - 快速查看和使用不同版本
"""
import os
import sys

def print_header(text):
    print("\n" + "=" * 70)
    print(text.center(70))
    print("=" * 70)

def print_section(title):
    print(f"\n{'-' * 70}")
    print(f"  {title}")
    print(f"{'-' * 70}")

def check_file_exists(filename):
    return "[OK]" if os.path.exists(filename) else "[NO]"

def main():
    print_header("v3_job_level Version Manager")

    # 检查核心文件
    print_section("Core Model Files (Shared by all versions)")
    core_files = [
        "job_model.py",
        "time_series_predictor.py",
        "data_processor.py",
        "competition_matrix.py"
    ]
    for f in core_files:
        print(f"  {check_file_exists(f)} {f}")

    # 检查v3文件
    print_section("v3 Version Files (Original)")
    v3_files = [
        "parameter_optimizer_v3.py",
        "main_v3.py"
    ]
    for f in v3_files:
        print(f"  {check_file_exists(f)} {f}")
    print("  Note: Single RMSE objective, fast but limited accuracy")

    # 检查v4文件
    print_section("v4 Version Files (Not Recommended)")
    v4_files = [
        "parameter_optimizer_v4.py",
        "main_v4.py"
    ]
    for f in v4_files:
        print(f"  {check_file_exists(f)} {f}")
    print("  Note: Multi-objective but has duplicate scoring issues")

    # 检查v5文件
    print_section("v5 Version Files (Recommended)")
    v5_files = [
        "parameter_optimizer_v5.py",
        "diagnose_model.py"
    ]
    for f in v5_files:
        print(f"  {check_file_exists(f)} {f}")
    print("  Note: Fixed structural issues, added diagnostics")

    # 检查工具文件
    print_section("Comparison and Visualization Tools")
    tool_files = [
        "compare_v3_v4.py",
        "integrate_and_visualize.py"
    ]
    for f in tool_files:
        print(f"  {check_file_exists(f)} {f}")

    # 检查文档
    print_section("Documentation Files")
    doc_files = [
        "快速开始.md",
        "README_v4改进说明.md",
        "README_v5改进说明.md",
        "版本整理说明.md"
    ]
    for f in doc_files:
        print(f"  {check_file_exists(f)} {f}")

    # 使用建议
    print_section("Usage Recommendations")
    print("""
  Scenario 1: Quick Test
    -> python main_v3.py

  Scenario 2: High Accuracy Prediction (Recommended)
    -> python diagnose_model.py  # Diagnose first
    -> Follow diagnostic results

  Scenario 3: Version Comparison
    -> python main_v3.py
    -> python main_v4.py
    -> python compare_v3_v4.py
    """)

    # 检查结果文件
    print_section("Generated Result Files")
    result_files = [
        ("job_parameters_v3.csv", "v3 parameters"),
        ("job_parameters_v4.csv", "v4 parameters"),
        ("all_predictions_v3.csv", "v3 predictions"),
        ("all_predictions_v4.csv", "v4 predictions"),
        ("integrated_predictions_with_actual.csv", "integrated predictions")
    ]
    for f, desc in result_files:
        status = check_file_exists(f)
        print(f"  {status} {f:<45} ({desc})")

    print("\n" + "=" * 70)
    print("TIP: Run 'python diagnose_model.py' first for diagnostics")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    main()