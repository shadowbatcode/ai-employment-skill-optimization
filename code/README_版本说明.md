# 生态协同进化模型 - 版本说明

## 📁 目录结构

```
代码/
├── v1_basic/              # 版本1：基础模型
│   ├── config.py
│   ├── ai_environment.py
│   ├── profession.py
│   ├── economy.py
│   ├── dynamics.py
│   └── main.py
│
├── v2_improved/           # 版本2：改进模型（基于真实数据）
│   ├── config_v2.py
│   ├── ai_environment_v2.py
│   ├── profession_v2.py
│   ├── dynamics_v2.py
│   ├── data_loader.py
│   └── main_v2.py
│
└── v3_job_level/          # 版本3：职位级模型（推荐使用）
    ├── job_model.py
    ├── data_processor.py
    ├── parameter_optimizer_v3.py  ⭐ 优化速度+进度条
    ├── time_series_predictor.py
    ├── competition_matrix.py      ⭐ 基于Related表
    └── main_v3.py                 ⭐ 整合所有改进
```

---

## 🎯 版本对比

| 特性 | v1 基础版 | v2 改进版 | v3 职位级（推荐） |
|-----|----------|----------|-----------------|
| 建模粒度 | 3个大类 | 3个大类 | 286个具体职位 |
| 数据驱动 | ❌ | ✅ | ✅ |
| 参数优化 | ❌ | ❌ | ✅ |
| 进度可视化 | ❌ | ❌ | ✅ |
| 竞争关系 | 简化矩阵 | 简化矩阵 | Related表 |
| 时间序列预测 | ❌ | ❌ | ✅ |
| 运行速度 | 快 | 中 | 优化后快 |

---

## 🚀 快速开始

### 版本3（推荐）

```bash
cd v3_job_level
python main_v3.py
```

**改进点**：
1. ⚡ **速度优化**：迭代次数减少30%（50→30）
2. 📊 **进度可视化**：实时显示优化进度和最佳误差
3. 🔗 **竞争关系**：基于O*NET相似度自动构建

**输出文件**：
- `job_parameters_v3.csv` - 优化后的参数
- `prediction_*.csv` - 每个职位的预测结果

---

## 📊 关键改进说明

### 1. 参数优化速度提升

**v3改进**：
```python
# 原版（v2）
maxiter=50, popsize=10  # 约500次迭代

# 改进版（v3）
maxiter=30, popsize=8   # 约240次迭代，速度提升52%
```

### 2. 进度可视化

**v3新增**：
```
优化参数: 100%|████████| 240/240 [00:45<00:00, best_error=0.003421]
```

### 3. 竞争矩阵构建

**v3改进**：
```python
# 使用Related表的"综合相关性"列
# 自动计算职位间相似度
alpha_ij = similarity / 3.0 * competition_strength
```

---

## 📝 使用示例

### 处理单个职位

```python
# 在main_v3.py中修改
selected_jobs = [all_jobs[0]]  # 只处理第一个职位
```

### 处理所有职位

```python
# 在main_v3.py中修改
selected_jobs = all_jobs  # 处理全部286个职位
```

### 调整优化参数

```python
# 更快速度（降低精度）
optimizer.optimize(maxiter=20, popsize=6)

# 更高精度（增加时间）
optimizer.optimize(maxiter=50, popsize=12)
```

---

## ⚙️ 参数说明

### 优化器参数

| 参数 | 默认值 | 说明 | 建议范围 |
|-----|-------|------|---------|
| maxiter | 30 | 最大迭代次数 | 20-50 |
| popsize | 8 | 种群大小 | 6-12 |
| competition_strength | 0.1 | 竞争强度系数 | 0.05-0.2 |

### 可学习参数

| 参数 | 范围 | 含义 |
|-----|------|------|
| theta | [-1, 1] | AI敏感度 |
| Ac | [0.1, 0.9] | 临界AI水平 |
| k | [1, 10] | 渗透速率 |

---

## 📈 预期运行时间

| 职位数量 | 预计时间 | 说明 |
|---------|---------|------|
| 1个 | ~1分钟 | 快速测试 |
| 3个 | ~3分钟 | 演示模式 |
| 10个 | ~10分钟 | 小规模分析 |
| 50个 | ~50分钟 | 中等规模 |
| 286个 | ~5小时 | 完整分析 |

---

## 🔧 故障排除

### 问题1：进度条不显示
**解决**：安装tqdm
```bash
pip install tqdm
```

### 问题2：优化时间过长
**解决**：减少迭代次数
```python
optimizer.optimize(maxiter=20, popsize=6)
```

### 问题3：内存不足
**解决**：分批处理职位
```python
for batch in range(0, len(all_jobs), 10):
    selected_jobs = all_jobs[batch:batch+10]
```

---

## 📊 输出文件说明

### job_parameters_v3.csv
```csv
title,type,theta,Ac,k,train_error
Specialty Trade Contractors,Trade,-0.352,0.418,6.234,0.003421
```

### prediction_*.csv
```csv
Quarter,Predicted_Employment
0,12000
1,11850
...
40,10500
```

---

## 🎓 技术细节

### 优化算法
- **方法**：差分进化（Differential Evolution）
- **优势**：全局搜索，不需要梯度
- **适用**：非凸优化问题

### 竞争矩阵
- **数据源**：O*NET职业相似度数据库
- **计算**：综合相关性 / 3.0 × 竞争强度
- **范围**：[0, 1]

---

**当前版本**: v3.0
**最后更新**: 2026-01-31
**推荐使用**: v3_job_level

