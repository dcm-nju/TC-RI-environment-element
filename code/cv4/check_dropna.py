import pandas as pd
import numpy as np

samples = pd.read_csv('analysis_samples.csv', encoding='utf-8-sig')
vars_list = ['VMPI','XOHC','XDST','SHRD','SHGC','RHLO','EPOS']
delta_cols = [v+'_delta24' for v in vars_list]
abs_cols = [v+'_abs' for v in vars_list]
feature_cols = delta_cols + abs_cols

print(f'构造样本总数: {len(samples)}')
print(f'  增强组: {(samples["group"]=="enhance").sum()}')
print(f'  停止组: {(samples["group"]=="stop").sum()}')

# 图3 dropna (14个特征)
df3 = samples.dropna(subset=feature_cols).copy()
print(f'\n图3 dropna后(14特征): {len(df3)}')
print(f'  增强组: {(df3["group"]=="enhance").sum()}')
print(f'  停止组: {(df3["group"]=="stop").sum()}')
print(f'  事件数: {df3["event_id"].nunique()}')

# 图4 dropna (7个delta24)
df4 = samples.dropna(subset=delta_cols).copy()
print(f'\n图4 dropna后(7特征): {len(df4)}')
print(f'  增强组: {(df4["group"]=="enhance").sum()}')
print(f'  停止组: {(df4["group"]=="stop").sum()}')

# 检查图3和图4 dropna结果是否一致
print(f'\n图3和图4 dropna结果是否一致: {len(df3)==len(df4)}')
if len(df3)!=len(df4):
    diff = set(df4.index) - set(df3.index)
    print(f'  图4有但图3没有的样本数: {len(diff)}')
    for idx in list(diff)[:5]:
        row = samples.loc[idx]
        missing_abs = [c for c in abs_cols if pd.isna(row[c])]
        print(f'    样本{idx}: 缺测abs列={missing_abs}')
