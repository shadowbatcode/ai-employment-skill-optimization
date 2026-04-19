# AI Employment Skill Optimization

来源：
- `2026美赛F/美赛F题数据集/代码`
- `2026美赛F/赛题/2026_ICM_Problem_F.pdf`
- `2026美赛F/论文.docx`

项目内容：
- `code/`
  主代码目录，含第二问求解器、第三问求解器、职位级模型、分析脚本、图表与结果文件
- `problem/`
  F 题题面
- `paper/`
  当前论文稿
- `docs/`
  工作笔记、流程图和辅助说明图

本次整理刻意排除：
- `2026美赛F/优秀论文`
- `2026美赛F/美赛F题数据集/all.data.combined.csv`
- `2026美赛F/美赛F题数据集/最终数据`
- `__pycache__` 和临时 Excel 文件

原因：
- 原始数据中存在单文件约 1.1 GB 的 CSV，不适合直接放进普通 GitHub 仓库

额外说明：
- `code/v3_job_level/` 中有脚本引用 `ai_environment_v2`、`config`、`dynamics`、`profession` 等模块
- 这些模块在原始目录中只留下了编译缓存，没有对应 `.py` 源文件
- 我没有把 `__pycache__` 编译产物一起放进仓库；如果你需要，我后面可以继续帮你恢复这部分源代码

运行提示：
- 先从 `code/` 目录内运行脚本
- 多个脚本默认依赖 `程序所需数据.xlsx` 以及同级 `数据/`、`output/` 目录

主要依赖：
- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `scipy`
- `scikit-learn`
- `tqdm`
- `openpyxl`
