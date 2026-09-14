import pandas as pd
import numpy as np

merged = pd.read_csv('../merged_full_data.csv', encoding='utf-8-sig', nrows=5)
print('列名:', list(merged.columns))
print()

merged = pd.read_csv('../merged_full_data.csv', encoding='utf-8-sig')
print(f'总行数: {len(merged)}')
print()

vars_list = ['VMPI','XOHC','XDST','SHRD','SHGC','RHLO','EPOS']
print('=== _abs列缺测 ===')
for v in vars_list:
    col = v + '_abs'
    print(f'  {col}: {merged[col].isna().sum()} ({merged[col].isna().sum()/len(merged)*100:.1f}%)')

print()
print('=== _delta24列缺测 ===')
for v in vars_list:
    col = v + '_delta24'
    print(f'  {col}: {merged[col].isna().sum()} ({merged[col].isna().sum()/len(merged)*100:.1f}%)')

# 检查分组列
print()
print('=== 分组列 ===')
for col in ['SID', 'USA_ATCF_ID', 'usa_atcf_id', 'storm_id']:
    if col in merged.columns:
        print(f'  {col}: 存在, 唯一值={merged[col].nunique()}')
    else:
        print(f'  {col}: 不存在')

# 检查RI时刻的delta24缺测
print()
ri_data = merged[merged['is_ri'] == True]
print(f'RI时刻总数: {len(ri_data)}')
print('RI时刻的delta24缺测:')
for v in vars_list:
    col = v + '_delta24'
    print(f'  {col}: {ri_data[col].isna().sum()}')
