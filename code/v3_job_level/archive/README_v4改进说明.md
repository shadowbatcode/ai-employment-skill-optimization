# v4版本改进说明

## 问题诊断

通过分析v3版本的预测结果,发现以下问题:

1. **误差波动大**: 相对误差标准差达18%,从-44%到+82%
2. **趋势预测不准**: 某些职位预测趋势与真实趋势相反
3. **方向不一致**: 预测的变化方向与实际变化方向不匹配
4. **个别职位严重偏离**: 如Performing Arts最大误差达82%

## 核心改进

### 1. 多目标优化函数

**v3版本**只使用单一的相对均方误差(RMSE):
```python
error = np.mean(((predicted - actual) / actual) ** 2)
```

**v4版本**使用多目标加权组合:
```python
total_loss = (
    rmse * 0.4 +              # 相对均方根误差
    trend_loss * 0.3 +        # 趋势匹配损失
    direction_loss * 0.2 +    # 方向一致性损失
    weighted_rmse * 0.3 +     # 加权RMSE(后期权重更高)
    smoothness_penalty * 0.1 + # 平滑度惩罚
    physical_penalty          # 物理约束惩罚
)
```

### 2. 趋势匹配

计算预测值和真实值的整体趋势(线性回归斜率),确保趋势方向一致:
```python
pred_trend = np.polyfit(x, predicted, 1)[0] / np.mean(predicted)
data_trend = np.polyfit(x, actual, 1)[0] / np.mean(actual)
trend_loss = abs(pred_trend - data_trend) * 10
```

### 3. 方向一致性

确保预测的变化方向与实际变化方向匹配:
```python
actual_diff = np.diff(actual_employment)
pred_diff = np.diff(predicted)
direction_agreement = np.sign(actual_diff) * np.sign(pred_diff)
direction_loss = np.mean(direction_agreement < 0) * 5
```

### 4. 自适应权重

根据数据波动性自动调整各项权重:
- **高波动数据**: 提高趋势和方向权重,降低RMSE权重
- **低波动数据**: 提高RMSE权重,降低趋势权重

### 5. 物理约束惩罚

增加多项物理合理性检查:
- 承载力不应过小(<0.3K0)或过大(>3.0K0)
- 预测值不应为负数或异常大
- theta和Ac的组合应合理(负theta对应小Ac)

### 6. 自适应参数边界

根据数据趋势调整参数搜索空间:
- **增长趋势**: theta ∈ [0.0, 1.0] (偏向增强效应)
- **下降趋势**: theta ∈ [-1.0, 0.0] (偏向替代效应)
- **平稳趋势**: theta ∈ [-0.5, 0.5] (中性范围)

### 7. 增强优化策略

- 增加迭代次数: 30 → 50
- 增加种群大小: 8 → 15
- 使用更激进的进化策略: `strategy='best1bin'`
- 增加变异范围: `mutation=(0.5, 1.5)`
- 可选的局部精细化: 全局搜索后使用L-BFGS-B局部优化

## 使用方法

### 运行v4版本
```bash
cd v3_job_level
python main_v4.py
```

### 对比v3和v4效果
```bash
python compare_v3_v4.py
```

## 预期改进效果

1. **MAPE降低**: 预计从当前水平降低30-50%
2. **趋势准确性提高**: 趋势方向匹配率提高到90%以上
3. **方向一致性提高**: 变化方向匹配率提高到85%以上
4. **误差波动减小**: 标准差降低到10%以内
5. **极端误差减少**: 最大误差控制在30%以内

## 文件说明

- `parameter_optimizer_v4.py`: 改进的优化器
- `main_v4.py`: 使用v4优化器的主程序
- `compare_v3_v4.py`: v3和v4效果对比工具
- `job_parameters_v4.csv`: v4优化的参数结果
- `all_predictions_v4.csv`: v4的预测结果