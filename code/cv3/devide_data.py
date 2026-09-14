import pandas as pd
import numpy as np
from datetime import datetime
import warnings
import sys
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore')

# ===================== 路径设置 =====================
script_dir = Path(__file__).resolve().parent.parent
project_root = script_dir.parent

data_dir = project_root / 'data'
ibtracs_dir = data_dir / 'ibtracs'
ships_dir = data_dir / 'ships'
picture_dir = project_root / 'picture' / 'version3'
picture_dir.mkdir(parents=True, exist_ok=True)

ibtracs_file = ibtracs_dir / 'ibtracs.ALL.list.v04r01.csv'
ships_file = ships_dir / 'lsdiaga_1982_2023_sat_ts_7day.txt'

# ===================== 常量配置 =====================
MISSING_VALUES = {9999, 9999.0, 999.9, 999, 999.0, -999, -999.0, -999.9}
RI_THRESHOLD = 30
RI_MAX_HOURS = 24
TIME_STEP_HOURS = 6
MAX_TIME_MATCH_SEC = 3600
SHIPS_VARS = ['RSST', 'SHRD', 'RHLO', 'RHMD', 'VMPI']
SHIPS_NEEDED_VARS = set(SHIPS_VARS)
PLOT_VARS = ['RSST', 'SHRD', 'RHLO', 'RHMD', 'VMPI', 'MOVE_SPEED']
PLOT_LABELS = {
    'RSST': '海表温度 (℃)',
    'SHRD': '垂直风切变 (m/s)',
    'RHLO': '低层相对湿度 (%)',
    'RHMD': '中层相对湿度 (%)',
    'VMPI': '最大潜在强度 (kt)',
    'MOVE_SPEED': '移动速度 (kt)'
}
PLOT_COLORS = ['#1f77b4', '#ff7f0e']
PLOT_GROUP_NAMES = ['RI增强点', 'RI停止点']

# ===================== 辅助函数 =====================
def parse_number(s: str):
    try:
        if '.' in s:
            return float(s)
        return int(s)
    except (ValueError, TypeError):
        return None

def is_missing_value(val) -> bool:
    return val in MISSING_VALUES

def haversine_distance(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return 3440.065 * c

def calculate_movement_speed(storm_data):
    storm_data = storm_data.sort_values('ISO_TIME').reset_index(drop=True)
    speeds = []
    for i in range(len(storm_data)):
        if i == 0:
            if len(storm_data) > 1:
                lat1, lon1 = storm_data.iloc[i]['LAT'], storm_data.iloc[i]['LON']
                lat2, lon2 = storm_data.iloc[i+1]['LAT'], storm_data.iloc[i+1]['LON']
                distance = haversine_distance(lat1, lon1, lat2, lon2)
                speed = distance / TIME_STEP_HOURS
            else:
                speed = np.nan
        else:
            lat1, lon1 = storm_data.iloc[i-1]['LAT'], storm_data.iloc[i-1]['LON']
            lat2, lon2 = storm_data.iloc[i]['LAT'], storm_data.iloc[i]['LON']
            distance = haversine_distance(lat1, lon1, lat2, lon2)
            speed = distance / TIME_STEP_HOURS
        speeds.append(speed)
    return speeds

# ===================== 读取SHIPS数据 =====================
def read_ships_file(filename: Path) -> pd.DataFrame:
    if not filename.exists():
        raise FileNotFoundError(f"SHIPS数据文件不存在: {filename}")
    records = []
    current_record = {}
    with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            descriptor = None
            for part in reversed(parts):
                if parse_number(part) is None:
                    descriptor = part
                    break
            if descriptor is None:
                continue
            if descriptor == 'HEAD':
                if current_record:
                    records.append(current_record)
                current_record = {}
                current_record['name'] = parts[0]
                date_str = parts[1]
                yy = int(date_str[:2])
                current_record['year'] = 1900 + yy if yy > 50 else 2000 + yy
                current_record['month'] = int(date_str[2:4])
                current_record['day'] = int(date_str[4:6])
                current_record['time'] = int(parts[2])
                current_record['vmax'] = int(parts[3])
                current_record['lat'] = float(parts[4])
                current_record['lon'] = float(parts[5])
                current_record['mslp'] = int(parts[6])
                current_record['atcf_id'] = parts[7]
                current_record['ISO_TIME'] = datetime(
                    year=current_record['year'],
                    month=current_record['month'],
                    day=current_record['day'],
                    hour=current_record['time']
                )
            elif descriptor == 'LAST':
                if current_record:
                    records.append(current_record)
                current_record = {}
            elif descriptor in SHIPS_NEEDED_VARS:
                values = []
                for x in parts:
                    val = parse_number(x)
                    if val is not None:
                        values.append(val)
                if values:
                    val = values[0]
                    current_record[descriptor] = np.nan if is_missing_value(val) else val
        if current_record:
            records.append(current_record)
    if not records:
        raise ValueError(f"未能从 {filename} 解析出任何记录")
    return pd.DataFrame(records)

# ===================== RI检测 =====================
def detect_all_ri_periods(
    storm_data: pd.DataFrame,
    wind_col: str = 'USA_WIND',
    time_col: str = 'ISO_TIME',
    threshold: int = RI_THRESHOLD,
    max_hours: int = RI_MAX_HOURS
) -> tuple[bool, list[tuple[datetime, datetime]]]:
    storm_data = storm_data.sort_values(time_col).reset_index(drop=True)
    n = len(storm_data)
    if n < 2:
        return False, []
    times = storm_data[time_col].values.astype('datetime64[ns]')
    winds = storm_data[wind_col].values.astype(float)
    ri_periods = []
    left = 0
    for right in range(n):
        while left < right:
            time_diff_hours = (times[right] - times[left]) / np.timedelta64(1, 'h')
            if time_diff_hours <= max_hours:
                break
            left += 1
        if right > left:
            wind_diffs = winds[right] - winds[left:right]
            max_diff_idx = np.argmax(wind_diffs)
            if wind_diffs[max_diff_idx] >= threshold:
                ri_end_time = pd.Timestamp(times[right])
                ri_start_time = pd.Timestamp(times[left + max_diff_idx])
                ri_periods.append((ri_start_time, ri_end_time))
    if not ri_periods:
        return False, []
    return True, ri_periods

# ===================== ATCF ID标准化 =====================
def standardize_atcf_id(atcf_id) -> str | None:
    if pd.isna(atcf_id):
        return None
    atcf_id = str(atcf_id).strip().upper()
    atcf_id = ''.join(c for c in atcf_id if c.isalnum())
    if not atcf_id:
        return None
    if atcf_id.startswith('AL') and len(atcf_id) == 8:
        num = atcf_id[2:4]
        yr = atcf_id[4:8]
        return f"{num}L{yr}"
    if len(atcf_id) == 7 and atcf_id[2] == 'L':
        return atcf_id
    return atcf_id

# ===================== 数据匹配 =====================
def match_ships_data(analysis_df: pd.DataFrame, ships_df: pd.DataFrame) -> pd.DataFrame:
    ships_df = ships_df.copy()
    ships_df['atcf_id_standard'] = ships_df['atcf_id'].apply(standardize_atcf_id)
    ships_df = ships_df.dropna(subset=['atcf_id_standard'])
    ships_df = ships_df[ships_df['atcf_id_standard'] != '']
    atcf_index = {}
    for aid, grp in ships_df.groupby('atcf_id_standard'):
        grp = grp.sort_values('ISO_TIME').reset_index(drop=True)
        atcf_index[aid] = {'data': grp, 'times': grp['ISO_TIME'].values.astype('datetime64[ns]')}
    matched_rows = []
    for idx, row in analysis_df.iterrows():
        target_time = np.datetime64(row['ISO_TIME'])
        atcf_id = row['USA_ATCF_ID_STANDARD']
        if atcf_id not in atcf_index:
            continue
        ships_info = atcf_index[atcf_id]
        ships_times = ships_info['times']
        ships_grp = ships_info['data']
        time_diffs_sec = np.abs(ships_times - target_time) / np.timedelta64(1, 's')
        min_idx = int(np.argmin(time_diffs_sec))
        if time_diffs_sec[min_idx] > MAX_TIME_MATCH_SEC:
            continue
        best_match = ships_grp.iloc[min_idx]
        combined = row.to_dict()
        for var in SHIPS_VARS:
            if var in best_match and pd.notna(best_match[var]):
                val = best_match[var]
                combined[var] = val / 10.0 if var in ('RSST', 'SHRD', 'VMPI') else float(val)
            else:
                combined[var] = np.nan
        matched_rows.append(combined)
    return pd.DataFrame(matched_rows)

# ===================== 计算变化量（修复索引bug，放宽匹配到±2小时） =====================
def calculate_lag_changes(df: pd.DataFrame, hours: int, tolerance_hours: float = 2.0) -> pd.DataFrame:
    """
    计算每个点相对于过去hours小时的环境变量变化量（当前值 - 过去值）
    允许时间差在 [hours - tolerance, hours + tolerance] 小时内的匹配。
    修复：使用原始index进行赋值，避免reset_index导致的索引错位。
    """
    df = df.copy()
    # 统一排序并重置index，确保后续groupby的index是连续的
    df = df.sort_values(['SID', 'ISO_TIME']).reset_index(drop=True)
    lag_seconds = hours * 3600
    tol_seconds = tolerance_hours * 3600
    lower = lag_seconds - tol_seconds
    upper = lag_seconds + tol_seconds
    for var in PLOT_VARS:
        change_col = f'{var}_lag{hours}h'
        df[change_col] = np.nan

    for sid, group in df.groupby('SID'):
        # 关键修复：不再reset_index，直接使用groupby返回的原始index
        group = group.sort_values('ISO_TIME')
        indices = group.index.tolist()  # 保存原始index列表
        times = group['ISO_TIME'].values
        
        for i, current_time in enumerate(times):
            best_diff = None
            best_j = None
            for j in range(i):  # 只向前查找
                diff_sec = (current_time - times[j]) / np.timedelta64(1, 's')
                if lower <= diff_sec <= upper:
                    diff_abs = abs(diff_sec - lag_seconds)
                    if best_diff is None or diff_abs < best_diff:
                        best_diff = diff_abs
                        best_j = j
            if best_j is not None:
                for var in PLOT_VARS:
                    change_col = f'{var}_lag{hours}h'
                    # 使用原始index进行赋值
                    df.loc[indices[i], change_col] = group.loc[indices[i], var] - group.loc[indices[best_j], var]
    return df

# ===================== 提取增强点和停止点 =====================
def extract_enhancement_and_stop_points():
    print("正在读取原始数据...")
    if not ibtracs_file.exists():
        print(f"错误：IBTrACS文件不存在: {ibtracs_file}")
        sys.exit(1)
    ibtracs_df = pd.read_csv(ibtracs_file, low_memory=False, skiprows=[1])
    ships_df = read_ships_file(ships_file)

    ibtracs_df['ISO_TIME'] = pd.to_datetime(ibtracs_df['ISO_TIME'])
    ibtracs_df['SEASON'] = ibtracs_df['SEASON'].astype(int)
    ibtracs_df['USA_WIND'] = pd.to_numeric(ibtracs_df['USA_WIND'], errors='coerce')
    ibtracs_df = ibtracs_df.dropna(subset=['USA_WIND'])
    ibtracs_df = ibtracs_df[(ibtracs_df['SEASON'] >= 1988) & (ibtracs_df['SEASON'] <= 2021)]

    na_mask = (
        (ibtracs_df['BASIN'] == 'AL') |
        ((ibtracs_df['BASIN'].isna()) & (ibtracs_df['LON'] >= -100) & (ibtracs_df['LON'] <= 0) &
         (ibtracs_df['LAT'] >= 0) & (ibtracs_df['LAT'] <= 45))
    )
    ibtracs_na = ibtracs_df[na_mask].copy()
    ibtracs_na['USA_ATCF_ID'] = ibtracs_na.groupby('SID')['USA_ATCF_ID'].ffill().bfill()
    ibtracs_na['USA_ATCF_ID_STANDARD'] = ibtracs_na['USA_ATCF_ID'].apply(standardize_atcf_id)
    storms_with_atcf = ibtracs_na['SID'][ibtracs_na['USA_ATCF_ID_STANDARD'].notna()].unique()
    ibtracs_na = ibtracs_na[ibtracs_na['SID'].isin(storms_with_atcf)].copy()

    ri_storms = []
    all_ri_periods = {}
    storm_ids = list(ibtracs_na.groupby('SID').groups.keys())
    print(f"正在检测RI事件，共{len(storm_ids)}个台风...")
    for sid in storm_ids:
        storm_data = ibtracs_na[ibtracs_na['SID'] == sid]
        has_ri, ri_periods = detect_all_ri_periods(storm_data)
        if has_ri:
            ri_storms.append(sid)
            all_ri_periods[sid] = ri_periods
    if not ri_storms:
        print("错误：未检测到任何快速增强台风")
        sys.exit(0)

    print(f"检测到{len(ri_storms)}个包含RI事件的台风")
    print("正在提取增强点和停止点...")
    enhancement_points = []
    stop_points = []

    for sid in ri_storms:
        storm_data = ibtracs_na[ibtracs_na['SID'] == sid].sort_values('ISO_TIME').reset_index(drop=True)
        ri_periods = all_ri_periods[sid]
        storm_matched = match_ships_data(storm_data, ships_df)
        if len(storm_matched) == 0:
            continue
        speeds = calculate_movement_speed(storm_matched)
        storm_matched['MOVE_SPEED'] = speeds
        # 过滤
        storm_matched = storm_matched[(storm_matched['RSST'] >= 22) & (storm_matched['RSST'] <= 30)]
        storm_matched = storm_matched[(storm_matched['MOVE_SPEED'] >= 0) & (storm_matched['MOVE_SPEED'] <= 50)]
        storm_matched = storm_matched.dropna(subset=['RSST', 'SHRD', 'RHLO', 'RHMD', 'VMPI', 'MOVE_SPEED'])
        if len(storm_matched) == 0:
            continue

        for start_time, end_time in ri_periods:
            # 增强点
            enhancement_mask = (storm_matched['ISO_TIME'] >= start_time) & (storm_matched['ISO_TIME'] < end_time)
            for _, row in storm_matched[enhancement_mask].iterrows():
                point = row.to_dict()
                point['label'] = 1
                enhancement_points.append(point)
            # 停止点
            stop_row = storm_matched[storm_matched['ISO_TIME'] == end_time]
            if not stop_row.empty:
                point = stop_row.iloc[0].to_dict()
                point['label'] = 0
                stop_points.append(point)

    enhancement_df = pd.DataFrame(enhancement_points)
    stop_df = pd.DataFrame(stop_points)
    print(f"数据提取完成：")
    print(f"  RI增强点 (标签1): {len(enhancement_df)} 个")
    print(f"  RI停止点 (标签0): {len(stop_df)} 个")
    return enhancement_df, stop_df

# ===================== 统计检验 =====================
def mannwhitney_with_effect(group1, group2):
    group1 = group1.dropna()
    group2 = group2.dropna()
    if len(group1) == 0 or len(group2) == 0:
        return np.nan, np.nan
    u, p = mannwhitneyu(group1, group2, alternative='two-sided')
    n1, n2 = len(group1), len(group2)
    n = n1 + n2
    if n1 * n2 * (n + 1) == 0:
        r = np.nan
    else:
        z = (u - n1*n2/2) / np.sqrt(n1*n2*(n+1)/12)
        r = z / np.sqrt(n)
    return p, r

# ===================== 绘图：原始值 =====================
def plot_original_boxplot(enh_df, stop_df, var, save_path):
    data1 = enh_df[var].dropna()
    data2 = stop_df[var].dropna()
    p, r = mannwhitney_with_effect(data1, data2)
    fig, ax = plt.subplots(figsize=(6, 5))
    bp = ax.boxplot([data1, data2], labels=PLOT_GROUP_NAMES, patch_artist=True,
                    medianprops={'color': 'black', 'linewidth': 1.5},
                    meanline=True, showmeans=True,
                    meanprops={'color': 'red', 'ls': '--', 'linewidth': 1.5},
                    flierprops={'marker': 'o', 'markerfacecolor': 'gray', 'markersize': 4, 'alpha': 0.6})
    for patch, color in zip(bp['boxes'], PLOT_COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_ylabel(PLOT_LABELS[var], fontsize=11)
    ax.set_title(f'{PLOT_LABELS[var]} 原始值对比', fontsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    # 样本量标注（调整位置和字号）
    y_min, y_max = ax.get_ylim()
    y_offset = (y_max - y_min) * 0.02
    ax.text(1, y_min + y_offset, f'n={len(data1)}', ha='center', va='bottom', fontsize=12)
    ax.text(2, y_min + y_offset, f'n={len(data2)}', ha='center', va='bottom', fontsize=12)
    # p值和效应量标注
    if not np.isnan(p):
        stat_text = f'Mann-Whitney U: p={p:.4f}'
        if not np.isnan(r):
            stat_text += f', r={r:.3f}'
        ax.annotate(stat_text, xy=(0.98, 0.98), xycoords='axes fraction',
                    ha='right', va='top', fontsize=15,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

# ===================== 绘图：变化量（6h/12h/24h并排） =====================
def plot_lag_boxplot(enh_df, stop_df, var, save_path):
    hours_list = [6, 12, 24]
    fig, ax = plt.subplots(figsize=(10, 6))
    width = 0.35
    for i, h in enumerate(hours_list):
        col = f'{var}_lag{h}h'
        if col not in enh_df.columns or col not in stop_df.columns:
            continue
        enh_vals = enh_df[col].dropna()
        stop_vals = stop_df[col].dropna()
        pos1 = (i+1) - width/2
        pos2 = (i+1) + width/2
        # 增强点
        bp1 = ax.boxplot([enh_vals], positions=[pos1], widths=width*0.8,
                         patch_artist=True, medianprops={'color':'black','linewidth':1.5},
                         meanline=True, showmeans=True,
                         meanprops={'color':'red','ls':'--','linewidth':1.5},
                         flierprops={'marker':'o','markerfacecolor':'gray','markersize':3,'alpha':0.6})
        # 停止点
        bp2 = ax.boxplot([stop_vals], positions=[pos2], widths=width*0.8,
                         patch_artist=True, medianprops={'color':'black','linewidth':1.5},
                         meanline=True, showmeans=True,
                         meanprops={'color':'red','ls':'--','linewidth':1.5},
                         flierprops={'marker':'o','markerfacecolor':'gray','markersize':3,'alpha':0.6})
        bp1['boxes'][0].set_facecolor(PLOT_COLORS[0]); bp1['boxes'][0].set_alpha(0.7)
        bp2['boxes'][0].set_facecolor(PLOT_COLORS[1]); bp2['boxes'][0].set_alpha(0.7)
        # 样本量标注（放在箱体底部外侧）
        y_min, y_max = ax.get_ylim()
        y_offset = (y_max - y_min) * 0.05
        ax.text(pos1, y_min - y_offset, f'n={len(enh_vals)}', ha='center', va='top', fontsize=12)
        ax.text(pos2, y_min - y_offset, f'n={len(stop_vals)}', ha='center', va='top', fontsize=12)
        # 检验结果
        p, r = mannwhitney_with_effect(enh_vals, stop_vals)
        if not np.isnan(p):
            sig_text = f'p={p:.4f}' if p>=0.001 else 'p<0.001'
            if not np.isnan(r):
                sig_text += f', r={r:.3f}'
            # 将标注放在该组箱线图上方
            y_top = max(
                max(enh_vals.quantile(0.75) + 1.5*(enh_vals.quantile(0.75)-enh_vals.quantile(0.25)) if len(enh_vals)>0 else 0,
                    stop_vals.quantile(0.75) + 1.5*(stop_vals.quantile(0.75)-stop_vals.quantile(0.25)) if len(stop_vals)>0 else 0
                ),
                max(enh_vals.max() if len(enh_vals)>0 else 0, stop_vals.max() if len(stop_vals)>0 else 0)
            )
            ax.text((pos1+pos2)/2, y_top + 0.1, sig_text, ha='center', fontsize=15,
                    bbox=dict(boxstyle="round,pad=0.15", facecolor="white", alpha=0.8))
    ax.set_xticks([1,2,3])
    ax.set_xticklabels(['6h', '12h', '24h'])
    ax.set_ylabel(PLOT_LABELS[var], fontsize=11)
    ax.set_title(f'{PLOT_LABELS[var]} 变化量对比（增强点 vs 停止点）', fontsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=PLOT_COLORS[0], alpha=0.7, label='RI增强点'),
                       Patch(facecolor=PLOT_COLORS[1], alpha=0.7, label='RI停止点')]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

# ===================== 主程序 =====================
def main():
    enh_df, stop_df = extract_enhancement_and_stop_points()
    # 检查必要列
    required_cols = ['SID', 'ISO_TIME'] + PLOT_VARS
    for col in required_cols:
        if col not in enh_df.columns or col not in stop_df.columns:
            print(f"错误：缺少列 {col}，无法计算变化量")
            return
    
    # 数据质量检查（变化量计算前）
    print(f"\n变化量计算前样本量：")
    print(f"  增强点: {len(enh_df)}")
    print(f"  停止点: {len(stop_df)}")
    
    # 合并计算变化量（放宽匹配到±2小时）
    combined = pd.concat([enh_df, stop_df], ignore_index=True)
    for h in [6,12,24]:
        combined = calculate_lag_changes(combined, h, tolerance_hours=2.0)
    enh_lag = combined[combined['label'] == 1].copy()
    stop_lag = combined[combined['label'] == 0].copy()

    # 数据质量检查（变化量计算后）
    print(f"\n变化量计算后样本量：")
    for h in [6, 12, 24]:
        col = f'VMPI_lag{h}h'
        enh_count = len(enh_lag[col].dropna())
        stop_count = len(stop_lag[col].dropna())
        print(f"  {h}h变化量: 增强点={enh_count}/{len(enh_lag)}, 停止点={stop_count}/{len(stop_lag)}")

    # 原始值绘图
    print("\n绘制原始值箱线图...")
    for var in PLOT_VARS:
        save_path = picture_dir / f'boxplot_original_{var}.png'
        plot_original_boxplot(enh_df, stop_df, var, save_path)
        print(f"  已保存: {save_path.name}")
    
    # 变化量绘图
    print("\n绘制变化量箱线图...")
    for var in PLOT_VARS:
        save_path = picture_dir / f'boxplot_lag_{var}.png'
        plot_lag_boxplot(enh_lag, stop_lag, var, save_path)
        print(f"  已保存: {save_path.name}")

    # 输出统计结果CSV
    results = []
    for var in PLOT_VARS:
        p, r = mannwhitney_with_effect(enh_df[var], stop_df[var])
        results.append({
            'Variable': var,
            'Type': 'Original',
            'p_value': p,
            'effect_size_r': r,
            'n_enhancement': len(enh_df[var].dropna()),
            'n_stop': len(stop_df[var].dropna())
        })
    for h in [6,12,24]:
        for var in PLOT_VARS:
            col = f'{var}_lag{h}h'
            if col not in enh_lag.columns:
                continue
            p, r = mannwhitney_with_effect(enh_lag[col], stop_lag[col])
            results.append({
                'Variable': var,
                'Type': f'Change_{h}h',
                'p_value': p,
                'effect_size_r': r,
                'n_enhancement': len(enh_lag[col].dropna()),
                'n_stop': len(stop_lag[col].dropna())
            })
    results_df = pd.DataFrame(results)
    results_df.to_csv(picture_dir / 'mannwhitney_results.csv', index=False)
    print("\n统计结果已保存至:", picture_dir / 'mannwhitney_results.csv')
    print("所有图表生成完毕！")

if __name__ == "__main__":
    main()