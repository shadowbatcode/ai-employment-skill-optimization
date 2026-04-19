# AI Employment Skill Optimization

该项目研究生成式 AI 对就业结构、职业竞争关系与技能培养路径的影响，并尝试通过参数拟合与优化方法，为教育资源投入和技能提升方案提供可计算的决策支持。

## Project Goals

- 分析不同行业和职业在 AI 冲击下的竞争格局
- 建立职业层级或技能层级的动态演化模型
- 优化教育投入与课程资源分配方案
- 通过敏感性分析评估模型稳定性与策略鲁棒性

## Methods

- 行业竞争分析与可视化
- 职位级时间演化建模
- 参数搜索与调优
- 最小二乘与约束优化
- 效率矩阵分析
- 敏感性分析与方案对比

## Repository Structure

- `code/`
  主代码目录
- `code/第二问求解器/`
  围绕优化求解、参数调优和敏感性分析的核心代码
- `code/第三问求解器/`
  课程配置与教育资源投入优化相关代码
- `code/v3_job_level/`
  职位级模型、参数估计和预测整合流程
- `problem/`
  题面文件
- `paper/`
  论文稿
- `docs/`
  流程图、说明图和辅助材料

## Key Scripts

- `code/occupational_competition_analysis.py`
  行业与职业竞争格局分析
- `code/fitting_visualization.py`
  拟合结果与参数可视化
- `code/第二问求解器/优化求解器.py`
  第二问主优化器
- `code/第二问求解器/敏感性分析.py`
  第二问敏感性分析
- `code/第三问求解器/优化求解器.py`
  第三问主优化器
- `code/v3_job_level/main_v3.py`
  职位级建模主流程
- `code/v3_job_level/integrate_and_visualize.py`
  结果整合与可视化

## Data And Outputs

项目包含程序运行所需的 Excel/CSV 数据、参数文件、优化结果、预测结果、热力图与分析报告。输出内容适合直接用于论文图表、结果汇总与方案比较。

## Running

建议从 `code/` 目录内运行脚本。不同模块通常依赖同级数据目录、参数 JSON 文件和输出目录。

## Main Dependencies

- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `scipy`
- `scikit-learn`
- `openpyxl`
