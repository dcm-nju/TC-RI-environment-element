import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore')

# 路径设置
script_dir = Path(__file__).resolve().parent.parent
project_root = script_dir.parent

data_dir = project_root / 'data'
ibtracs_dir = data_dir / 'ibtracs'
ships_dir = data_dir / 'ships'
picture_dir = project_root / 'picture' / 'version2' / '2.1'
picture_dir.mkdir(parents=True, exist_ok=True)

ibtracs_file = ibtracs_dir / 'ibtracs.ALL.list.v04r01.csv'
ships_file = ships_dir / 'lsdiaga_1982_2023_sat_ts_7day.txt'

# 常量配置
MISSING_VALUES = {9999, 9999.0, 999.9, 999, 999.0, -999, -999.0, -999.9}
RI_THRESHOLD = 30
RI_MAX_HOURS = 24
TIME_STEP_HOURS = 6
MAX_TIME_MATCH_SEC = 3600
RANDOM_STATE = 42

SHIPS_VARS = ['RSST', 'SHRD', 'RHLO', 'RHMD', 'RHHI', 'VMPI']
SHIPS_NEEDED_VARS = set(SHIPS_VARS)

PLOT_VARS_T0 = ['RSST', 'SHRD', 'RHMD', 'VMPI', 'MPI_DIFF']
PLOT_VARS_DELTA = ['RSST', 'SHRD', 'RHMD', 'VMPI']

PLOT_LABELS = {
    'RSST': '海表温度 (℃)',
    'SHRD': '垂直风切变 (m/s)',
    'RHMD': '中层相对湿度 (%)',
    'VMPI': '最大潜在强度 (kt)',
    'MPI_DIFF': 'MPI与当前强度差值 (kt)',
    'delta_RSST': 'Δ 海表温度 (℃)',
    'delta_SHRD': 'Δ 垂直风切变 (m/s)',
    'delta_RHMD': 'Δ 中层相对湿度 (%)',
    'delta_VMPI': 'Δ 最大潜在强度 (kt)'
}

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

def get_ri_status_per_timepoint(
    storm_data: pd.DataFrame,
    wind_col: str = 'USA_WIND',
    time_col: str = 'ISO_TIME',
    threshold: int = RI_THRESHOLD,
    max_hours: int = RI_MAX_HOURS
) -> pd.Series:
    """
    为台风的每个时间点标记是否处于RI状态
    定义：该时间点之后24小时内，是否存在某一时刻风速增长≥30kt
    """
    storm_data = storm_data.sort_values(time_col).reset_index(drop=True)
    n = len(storm_data)
    ri_status = pd.Series([False] * n, index=storm_data.index)
    
    times = storm_data[time_col].values.astype('datetime64[ns]')
    winds = storm_data[wind_col].values.astype(float)
    
    for i in range(n):
        max_time = times[i] + np.timedelta64(max_hours, 'h')
        j = i + 1
        while j < n and times[j] <= max_time:
            if winds[j] - winds[i] >= threshold:
                ri_status[i] = True
                break
            j += 1
    
    return ri_status

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

def match_ships_to_row(row, ships_index: dict):
    target_time = np.datetime64(row['ISO_TIME'])
    atcf_id = row['USA_ATCF_ID_STANDARD']
    if atcf_id not in ships_index:
        return None
    ships_info = ships_index[atcf_id]
    ships_times = ships_info['times']
    ships_grp = ships_info['data']

    time_diffs_sec = np.abs(ships_times - target_time) / np.timedelta64(1, 's')
    min_idx = int(np.argmin(time_diffs_sec))
    if time_diffs_sec[min_idx] > MAX_TIME_MATCH_SEC:
        return None

    best_match = ships_grp.iloc[min_idx]
    result = {}
    for var in SHIPS_VARS:
        if var in best_match and pd.notna(best_match[var]):
            val = best_match[var]
            result[var] = val / 10.0 if var in ('RSST', 'SHRD', 'VMPI') else float(val)
        else:
            result[var] = np.nan
    return result

def group_and_extract():
    print("=" * 70)
    print("开始读取原始数据...")
    
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
        (
            (ibtracs_df['BASIN'].isna()) &
            (ibtracs_df['LON'] >= -100) & (ibtracs_df['LON'] <= 0) &
            (ibtracs_df['LAT'] >= 0) & (ibtracs_df['LAT'] <= 45)
        )
    )
    ibtracs_na = ibtracs_df[na_mask].copy()

    ibtracs_na['USA_ATCF_ID'] = ibtracs_na.groupby('SID')['USA_ATCF_ID'].ffill().bfill()
    ibtracs_na['USA_ATCF_ID_STANDARD'] = ibtracs_na['USA_ATCF_ID'].apply(standardize_atcf_id)
    storms_with_atcf = ibtracs_na['SID'][ibtracs_na['USA_ATCF_ID_STANDARD'].notna()].unique()
    ibtracs_na = ibtracs_na[ibtracs_na['SID'].isin(storms_with_atcf)].copy()

    ships_df['atcf_id_standard'] = ships_df['atcf_id'].apply(standardize_atcf_id)
    ships_df = ships_df.dropna(subset=['atcf_id_standard'])
    ships_df = ships_df[ships_df['atcf_id_standard'] != '']
    ships_index = {}
    for aid, grp in ships_df.groupby('atcf_id_standard'):
        grp = grp.sort_values('ISO_TIME').reset_index(drop=True)
        ships_index[aid] = {'data': grp, 'times': grp['ISO_TIME'].values.astype('datetime64[ns]')}

    storm_groups = ibtracs_na.groupby('SID')
    all_storm_ids = list(storm_groups.groups.keys())
    print(f"共 {len(all_storm_ids)} 个台风待分析")

    group_labels = {}
    ri_start_times = {}
    storm_data_dict = {}
    storm_matched_dict = {}

    for sid in all_storm_ids:
        storm_data = storm_groups.get_group(sid).sort_values('ISO_TIME').reset_index(drop=True)
        
        ri_status = get_ri_status_per_timepoint(storm_data)
        if not ri_status.any():
            continue

        first_ri_idx = ri_status.idxmax()
        t0 = storm_data.loc[first_ri_idx, 'ISO_TIME']

        def find_closest_timepoint(target_time):
            diff = np.abs((storm_data['ISO_TIME'] - target_time).dt.total_seconds())
            if diff.min() <= 3 * 3600:
                return diff.idxmin()
            return None

        idx_6 = find_closest_timepoint(t0 + pd.Timedelta(hours=6))
        idx_12 = find_closest_timepoint(t0 + pd.Timedelta(hours=12))
        idx_18 = find_closest_timepoint(t0 + pd.Timedelta(hours=18))

        if idx_6 is None or idx_12 is None or idx_18 is None:
            group_labels[sid] = 0
        else:
            if ri_status[idx_6] and ri_status[idx_12] and ri_status[idx_18]:
                group_labels[sid] = 1
            else:
                group_labels[sid] = 0

        ri_start_times[sid] = t0
        storm_data_dict[sid] = storm_data

        storm_matched = storm_data.copy()
        for _, row in storm_matched.iterrows():
            mv = match_ships_to_row(row, ships_index)
            if mv is not None:
                for k, v in mv.items():
                    storm_matched.loc[_, k] = v
            else:
                for k in SHIPS_VARS:
                    storm_matched.loc[_, k] = np.nan

        speeds = calculate_movement_speed(storm_matched)
        storm_matched['MOVE_SPEED'] = speeds

        storm_matched = storm_matched[(storm_matched['RSST'] >= 22) & (storm_matched['RSST'] <= 30)]
        storm_matched = storm_matched[(storm_matched['MOVE_SPEED'] >= 0) & (storm_matched['MOVE_SPEED'] <= 50)]
        storm_matched = storm_matched.dropna(subset=['RSST', 'SHRD', 'RHLO', 'RHMD', 'RHHI', 'VMPI', 'MOVE_SPEED'])
        storm_matched_dict[sid] = storm_matched

    print(f"\n分组完成:")
    n_cont = sum(1 for v in group_labels.values() if v == 1)
    n_term = sum(1 for v in group_labels.values() if v == 0)
    print(f"  持续组台风数: {n_cont}")
    print(f"  终止组台风数: {n_term}")

    if n_cont == 0 or n_term == 0:
        print("警告：其中一组样本数为0")

    return group_labels, ri_start_times, storm_data_dict, storm_matched_dict

def plot_intensity_evolution(group_labels, ri_start_times, storm_data_dict):
    print("\n绘制强度演变合成图...")
    time_range = np.arange(-48, 49, 6)
    groups = {0: [], 1: []}
    
    for sid, label in group_labels.items():
        t0 = ri_start_times[sid]
        storm = storm_data_dict[sid].copy()
        storm = storm.sort_values('ISO_TIME')
        
        intensities = []
        for offset in time_range:
            target_time = t0 + pd.Timedelta(hours=offset)
            diff = np.abs((storm['ISO_TIME'] - target_time).dt.total_seconds())
            if diff.min() <= 3 * 3600:
                idx = diff.idxmin()
                intensities.append(storm.loc[idx, 'USA_WIND'])
            else:
                intensities.append(np.nan)
        groups[label].append(intensities)

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = {0: '#ff7f0e', 1: '#1f77b4'}
    labels = {0: 'RI终止组', 1: 'RI持续组'}

    for label, all_intens in groups.items():
        if len(all_intens) == 0:
            continue
        arr = np.array(all_intens)
        mean_vals = np.nanmean(arr, axis=0)
        sem_vals = np.nanstd(arr, axis=0) / np.sqrt(np.sum(~np.isnan(arr), axis=0))
        ci = 1.96 * sem_vals

        ax.plot(time_range, mean_vals, label=labels[label], color=colors[label], linewidth=2)
        ax.fill_between(time_range, mean_vals - ci, mean_vals + ci, color=colors[label], alpha=0.2)

    all_t0_intens = []
    for label, all_intens in groups.items():
        for intens_list in all_intens:
            t0_idx = list(time_range).index(0)
            if not np.isnan(intens_list[t0_idx]):
                all_t0_intens.append(intens_list[t0_idx])
    if all_t0_intens:
        baseline = np.mean(all_t0_intens)
        slope = 30.0 / 24.0
        ri_line_x = np.array([0, 24])
        ri_line_y = baseline + slope * ri_line_x
        ax.plot(ri_line_x, ri_line_y, 'k--', linewidth=1.5, label='RI阈值 (30kt/24h)')

    ax.axvline(x=0, color='gray', linestyle='--', linewidth=1)
    ax.set_xlabel('相对于RI启动时刻 t0 的时间 (小时)', fontsize=13)
    ax.set_ylabel('平均强度 (kt)', fontsize=13)
    ax.set_title('RI持续组与终止组强度演变合成图', fontsize=15, pad=15)
    ax.legend(fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_xlim(-48, 48)
    plt.tight_layout()
    output_path = picture_dir / 'intensity_evolution.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  强度演变图已保存: {output_path}")

def add_significance_annotation(ax, data1, data2, x_pos, y_pos=None):
    """在指定x位置添加显著性标记"""
    try:
        u_stat, p_val = stats.mannwhitneyu(data1, data2, alternative='two-sided')
    except:
        return
    if p_val < 0.01:
        sig_text = '**'
    elif p_val < 0.05:
        sig_text = '*'
    else:
        return

    if y_pos is None:
        y_pos = max(np.percentile(data1, 95), np.percentile(data2, 95)) * 1.03
    ax.text(x_pos, y_pos, sig_text, ha='center', va='bottom', fontsize=14, color='k', fontweight='bold')

def plot_t0_boxplots(group_labels, ri_start_times, storm_matched_dict):
    print("\n绘制t0时刻环境变量箱线图...")
    data_by_group = {0: {var: [] for var in PLOT_VARS_T0}, 1: {var: [] for var in PLOT_VARS_T0}}
    
    for sid, label in group_labels.items():
        t0 = ri_start_times[sid]
        matched = storm_matched_dict.get(sid)
        if matched is None or matched.empty:
            continue
        diff = np.abs((matched['ISO_TIME'] - t0).dt.total_seconds())
        if diff.min() > 3*3600:
            continue
        row = matched.loc[diff.idxmin()]
        
        mpi_diff = row['VMPI'] - row['USA_WIND']
        
        data_by_group[label]['RSST'].append(row['RSST'])
        data_by_group[label]['SHRD'].append(row['SHRD'])
        data_by_group[label]['RHMD'].append(row['RHMD'])
        data_by_group[label]['VMPI'].append(row['VMPI'])
        data_by_group[label]['MPI_DIFF'].append(mpi_diff)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    for i, var in enumerate(PLOT_VARS_T0):
        ax = axes[i]
        data_term = data_by_group[0][var]
        data_cont = data_by_group[1][var]
        
        bp = ax.boxplot([data_term, data_cont], positions=[1, 2], patch_artist=True,
                         medianprops={'color': 'black', 'linewidth': 1.5},
                         flierprops={"marker": "o", "markerfacecolor": "gray", "markersize": 4, "alpha": 0.6},
                         widths=0.6)
        
        for patch, color in zip(bp['boxes'], ['#ff7f0e', '#1f77b4']):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        if len(data_term) > 0 and len(data_cont) > 0:
            add_significance_annotation(ax, data_term, data_cont, 1.5)

        ax.set_title(PLOT_LABELS[var], fontsize=12)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(['RI终止组', 'RI持续组'], fontsize=10)
        ax.grid(axis='y', linestyle='--', alpha=0.5)
    
    axes[5].set_visible(False)
    
    plt.suptitle('RI启动时刻(t0)环境变量对比', fontsize=15, y=1.01)
    plt.tight_layout()
    output_path = picture_dir / 'boxplot_t0.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  t0箱线图已保存: {output_path}")

def plot_delta_boxplots(group_labels, ri_start_times, storm_matched_dict):
    print("\n绘制t0→t0+24h环境变化量箱线图...")
    data_by_group = {0: {var: [] for var in PLOT_VARS_DELTA}, 1: {var: [] for var in PLOT_VARS_DELTA}}
    
    for sid, label in group_labels.items():
        t0 = ri_start_times[sid]
        t24 = t0 + pd.Timedelta(hours=24)
        matched = storm_matched_dict.get(sid)
        if matched is None or matched.empty:
            continue
        
        diff0 = np.abs((matched['ISO_TIME'] - t0).dt.total_seconds())
        diff24 = np.abs((matched['ISO_TIME'] - t24).dt.total_seconds())
        if diff0.min() > 3*3600 or diff24.min() > 3*3600:
            continue
        
        row0 = matched.loc[diff0.idxmin()]
        row24 = matched.loc[diff24.idxmin()]
        
        for var in PLOT_VARS_DELTA:
            if pd.notna(row0[var]) and pd.notna(row24[var]):
                delta = row24[var] - row0[var]
                data_by_group[label][var].append(delta)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for i, var in enumerate(PLOT_VARS_DELTA):
        ax = axes[i]
        data_term = data_by_group[0][var]
        data_cont = data_by_group[1][var]
        
        bp = ax.boxplot([data_term, data_cont], positions=[1, 2], patch_artist=True,
                         medianprops={'color': 'black', 'linewidth': 1.5},
                         flierprops={"marker": "o", "markerfacecolor": "gray", "markersize": 4, "alpha": 0.6},
                         widths=0.6)
        
        for patch, color in zip(bp['boxes'], ['#ff7f0e', '#1f77b4']):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        if len(data_term) > 0 and len(data_cont) > 0:
            add_significance_annotation(ax, data_term, data_cont, 1.5)

        ax.set_title(f'Δ {PLOT_LABELS[var]}', fontsize=12)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(['RI终止组', 'RI持续组'], fontsize=10)
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.8)
        ax.grid(axis='y', linestyle='--', alpha=0.5)
    
    plt.suptitle('RI启动后24小时环境变化量对比', fontsize=15, y=1.01)
    plt.tight_layout()
    output_path = picture_dir / 'boxplot_delta.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  变化量箱线图已保存: {output_path}")

# -------------------------- 修改：添加显著性标记的环境变量演变图 --------------------------
def plot_environment_evolution(group_labels, ri_start_times, storm_matched_dict):
    """绘制风切变和中层湿度从t=-24h到t=+48h的合成演变图（带显著性标记）"""
    print("\n绘制关键环境变量演变合成图...")
    time_range = np.arange(-24, 49, 6)
    
    env_data = {
        0: {'SHRD': [], 'RHMD': []},
        1: {'SHRD': [], 'RHMD': []}
    }

    for sid, label in group_labels.items():
        t0 = ri_start_times[sid]
        matched = storm_matched_dict.get(sid)
        if matched is None or matched.empty:
            continue
        
        shrd_vals = []
        rhmd_vals = []
        for offset in time_range:
            target_time = t0 + pd.Timedelta(hours=offset)
            diff = np.abs((matched['ISO_TIME'] - target_time).dt.total_seconds())
            if diff.min() <= 3 * 3600:
                idx = diff.idxmin()
                shrd_vals.append(matched.loc[idx, 'SHRD'])
                rhmd_vals.append(matched.loc[idx, 'RHMD'])
            else:
                shrd_vals.append(np.nan)
                rhmd_vals.append(np.nan)
        
        env_data[label]['SHRD'].append(shrd_vals)
        env_data[label]['RHMD'].append(rhmd_vals)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    colors = {0: '#ff7f0e', 1: '#1f77b4'}
    labels = {0: 'RI终止组', 1: 'RI持续组'}

    # 绘制垂直风切变
    ax = ax1
    shrd_term = np.array(env_data[0]['SHRD'])
    shrd_cont = np.array(env_data[1]['SHRD'])
    
    mean_term = np.nanmean(shrd_term, axis=0)
    sem_term = np.nanstd(shrd_term, axis=0) / np.sqrt(np.sum(~np.isnan(shrd_term), axis=0))
    ci_term = 1.96 * sem_term
    
    mean_cont = np.nanmean(shrd_cont, axis=0)
    sem_cont = np.nanstd(shrd_cont, axis=0) / np.sqrt(np.sum(~np.isnan(shrd_cont), axis=0))
    ci_cont = 1.96 * sem_cont

    ax.plot(time_range, mean_term, label=labels[0], color=colors[0], linewidth=2.5)
    ax.fill_between(time_range, mean_term - ci_term, mean_term + ci_term, color=colors[0], alpha=0.2)
    ax.plot(time_range, mean_cont, label=labels[1], color=colors[1], linewidth=2.5)
    ax.fill_between(time_range, mean_cont - ci_cont, mean_cont + ci_cont, color=colors[1], alpha=0.2)

    # 添加显著性标记
    for i, t in enumerate(time_range):
        data_term = shrd_term[:, i][~np.isnan(shrd_term[:, i])]
        data_cont = shrd_cont[:, i][~np.isnan(shrd_cont[:, i])]
        if len(data_term) > 0 and len(data_cont) > 0:
            y_pos = max(mean_term[i] + ci_term[i], mean_cont[i] + ci_cont[i]) * 1.03
            add_significance_annotation(ax, data_term, data_cont, t, y_pos)

    ax.axvline(x=0, color='gray', linestyle='--', linewidth=1.5)
    ax.set_xlabel('相对于RI启动时刻 t0 的时间 (小时)', fontsize=12)
    ax.set_ylabel('垂直风切变 (m/s)', fontsize=12)
    ax.set_title('垂直风切变演变对比', fontsize=14, pad=15)
    ax.legend(fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_xlim(-24, 48)

    # 绘制中层相对湿度
    ax = ax2
    rhmd_term = np.array(env_data[0]['RHMD'])
    rhmd_cont = np.array(env_data[1]['RHMD'])
    
    mean_term = np.nanmean(rhmd_term, axis=0)
    sem_term = np.nanstd(rhmd_term, axis=0) / np.sqrt(np.sum(~np.isnan(rhmd_term), axis=0))
    ci_term = 1.96 * sem_term
    
    mean_cont = np.nanmean(rhmd_cont, axis=0)
    sem_cont = np.nanstd(rhmd_cont, axis=0) / np.sqrt(np.sum(~np.isnan(rhmd_cont), axis=0))
    ci_cont = 1.96 * sem_cont

    ax.plot(time_range, mean_term, label=labels[0], color=colors[0], linewidth=2.5)
    ax.fill_between(time_range, mean_term - ci_term, mean_term + ci_term, color=colors[0], alpha=0.2)
    ax.plot(time_range, mean_cont, label=labels[1], color=colors[1], linewidth=2.5)
    ax.fill_between(time_range, mean_cont - ci_cont, mean_cont + ci_cont, color=colors[1], alpha=0.2)

    # 添加显著性标记
    for i, t in enumerate(time_range):
        data_term = rhmd_term[:, i][~np.isnan(rhmd_term[:, i])]
        data_cont = rhmd_cont[:, i][~np.isnan(rhmd_cont[:, i])]
        if len(data_term) > 0 and len(data_cont) > 0:
            y_pos = max(mean_term[i] + ci_term[i], mean_cont[i] + ci_cont[i]) * 1.03
            add_significance_annotation(ax, data_term, data_cont, t, y_pos)

    ax.axvline(x=0, color='gray', linestyle='--', linewidth=1.5)
    ax.set_xlabel('相对于RI启动时刻 t0 的时间 (小时)', fontsize=12)
    ax.set_ylabel('中层相对湿度 (%)', fontsize=12)
    ax.set_title('中层相对湿度演变对比', fontsize=14, pad=15)
    ax.legend(fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_xlim(-24, 48)

    plt.suptitle('RI启动前后关键环境变量演变合成图', fontsize=16, y=1.02)
    plt.tight_layout()
    output_path = picture_dir / 'environment_evolution.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  环境变量演变图已保存: {output_path}")

def get_ri_periods(storm_data, wind_col='USA_WIND', time_col='ISO_TIME', 
                   threshold_24h=30, max_gap_hours=6):
    storm_data = storm_data.sort_values(time_col).reset_index(drop=True)
    n = len(storm_data)
    if n < 2:
        return np.array([False] * n)
    
    time_diff = storm_data[time_col].diff().dt.total_seconds() / 3600
    wind_diff = storm_data[wind_col].diff()
    delta_24h = wind_diff * (24.0 / time_diff)
    delta_24h = delta_24h.fillna(0)
    
    is_ri_start = delta_24h >= threshold_24h
    
    ri_event = np.zeros(n, dtype=bool)
    i = 0
    while i < n:
        if is_ri_start.iloc[i]:
            start = i
            j = i
            while j < n - 1:
                gap = time_diff.iloc[j+1]
                if gap <= max_gap_hours and (is_ri_start.iloc[j+1] or j - start < 3):
                    j += 1
                else:
                    break
            end = j
            ri_event[start:end+1] = True
            i = j + 1
        else:
            i += 1
    return ri_event

def align_and_interpolate(storm_df, env_df, time_col='ISO_TIME'):
    merged = pd.merge(
        storm_df[[time_col, 'USA_WIND']],
        env_df[[time_col, 'SHRD']],
        on=time_col,
        how='left'
    )
    merged['SHRD'] = merged['SHRD'].interpolate(method='linear', limit_direction='both')
    merged['SHRD'] = merged['SHRD'].bfill().ffill()
    return merged

def detect_shear_surge(data, shear_col='SHRD', window=2, min_increase=5):
    """检测风切变突增的点"""
    diff_roll = data[shear_col].diff().rolling(window, min_periods=1).sum()
    valid = diff_roll.notna()
    if not valid.any():
        return None, None
    max_idx = diff_roll[valid].idxmax()
    max_increase = diff_roll[max_idx]
    if max_increase >= min_increase:
        return max_idx, max_increase
    else:
        return None, None

def get_ri_status_forward_looking(storm_data, wind_col='USA_WIND', time_col='ISO_TIME', 
                                  threshold=30, max_hours=24):
    return get_ri_periods(storm_data, wind_col, time_col, threshold_24h=threshold)

def select_typical_cases(group_labels, ri_start_times, storm_data_dict, storm_matched_dict):
    perfect_case = None
    sudden_stop_case = None

    for sid, label in group_labels.items():
        if label != 1:
            continue
        storm_data = storm_data_dict[sid]
        matched_data = storm_matched_dict.get(sid)
        if matched_data is None or matched_data.empty:
            continue
        
        ri_status = get_ri_status_per_timepoint(storm_data)
        storm_data['RI_STATUS'] = ri_status
        
        ri_times = storm_data[storm_data['RI_STATUS']]['ISO_TIME']
        if len(ri_times) < 8:
            continue
        
        ri_mask = matched_data['ISO_TIME'].isin(ri_times)
        shrd_ri = matched_data[ri_mask]['SHRD']
        if shrd_ri.max() < 15:
            storm_name = storm_data['NAME'].iloc[0] if 'NAME' in storm_data.columns else sid
            perfect_case = (sid, storm_data, matched_data, ri_status, storm_name)
            break

    for sid, label in group_labels.items():
        if label != 0:
            continue
        storm_data = storm_data_dict[sid]
        matched_data = storm_matched_dict.get(sid)
        if matched_data is None or matched_data.empty:
            continue
        
        t0 = ri_start_times[sid]
        ri_status = get_ri_status_per_timepoint(storm_data)
        storm_data['RI_STATUS'] = ri_status
        if not ri_status.any():
            continue
        
        time_mask = (matched_data['ISO_TIME'] >= t0) & (matched_data['ISO_TIME'] <= t0 + pd.Timedelta(hours=24))
        shrd_24h = matched_data[time_mask]['SHRD']
        if len(shrd_24h) < 2:
            continue
        
        shrd_initial = shrd_24h.iloc[0]
        shrd_max = shrd_24h.max()
        shrd_increase = shrd_max - shrd_initial
        if shrd_increase > 10 and shrd_max > 20:
            ri_times = storm_data[storm_data['RI_STATUS']]['ISO_TIME']
            ri_end_time = ri_times.max()
            if ri_end_time < t0 + pd.Timedelta(hours=24):
                storm_name = storm_data['NAME'].iloc[0] if 'NAME' in storm_data.columns else sid
                sudden_stop_case = (sid, storm_data, matched_data, ri_status, storm_name)
                break

    return perfect_case, sudden_stop_case

# -------------------------- 修改：左图添加RI中止点，优化文字位置 --------------------------
def plot_case_comparison(perfect_case, sudden_stop_case, ri_start_times):
    print("\n绘制典型案例对比图...")
    if perfect_case is None or sudden_stop_case is None:
        print("  未找到符合条件的典型案例，跳过绘图")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 7))
    colors = {'perfect': '#1f77b4', 'sudden': '#ff7f0e'}

    # ---------------- 处理完美持续RI风暴（添加RI中止点） ----------------
    sid1, storm1, matched1, ri_status1, name1 = perfect_case
    t0_1 = ri_start_times[sid1]
    data1 = pd.merge_asof(
        storm1[['ISO_TIME', 'USA_WIND', 'RI_STATUS']].sort_values('ISO_TIME'),
        matched1[['ISO_TIME', 'SHRD']].sort_values('ISO_TIME'),
        on='ISO_TIME', direction='nearest'
    )
    data1['time_rel'] = (data1['ISO_TIME'] - t0_1).dt.total_seconds() / 3600
    ri_mask1 = data1['RI_STATUS']
    ri_times_rel1 = data1.loc[ri_mask1, 'time_rel']
    ri_start_time_rel1 = ri_times_rel1.min()
    ri_end_time_rel1 = ri_times_rel1.max()  # 持续组的RI中止点

    ax1.plot(data1['time_rel'], data1['USA_WIND'], color=colors['perfect'], linewidth=2.5, label='台风风速 (kt)')
    ax1.set_xlabel('相对于RI启动时刻 t0 的时间 (小时)', fontsize=12)
    ax1.set_ylabel('台风风速 (kt)', fontsize=12, color=colors['perfect'])
    ax1.tick_params(axis='y', labelcolor=colors['perfect'])
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.set_title(f'完美持续RI风暴：{name1}\n(风切变全程稳定，RI持续>48h)', fontsize=13, pad=15)

    ax1_twin = ax1.twinx()
    ax1_twin.plot(data1['time_rel'], data1['SHRD'], color='gray', linestyle='--', linewidth=2, label='垂直风切变 (m/s)')
    ax1_twin.set_ylabel('垂直风切变 (m/s)', fontsize=12, color='gray')
    ax1_twin.tick_params(axis='y', labelcolor='gray')

    # 标注RI启动点（文字移到左下方，避免遮挡）
    t0_wind1 = data1.loc[data1['time_rel'] == ri_start_time_rel1, 'USA_WIND'].iloc[0]
    ax1.annotate('RI启动 (t0)', 
                 xy=(ri_start_time_rel1, t0_wind1), 
                 xytext=(-40, t0_wind1 - 15),  # 向下移动更多
                 arrowprops=dict(arrowstyle='->', color='red', lw=1.5), 
                 fontsize=10, color='red', fontweight='bold')

    # 标注RI中止点（文字移到右上方，避免遮挡）
    ri_end_wind1 = data1.loc[data1['time_rel'] == ri_end_time_rel1, 'USA_WIND'].iloc[0]
    ax1.annotate('RI终止', 
                 xy=(ri_end_time_rel1, ri_end_wind1), 
                 xytext=(ri_end_time_rel1 + 15, ri_end_wind1 + 10),  # 向右上方移动
                 arrowprops=dict(arrowstyle='->', color='black', lw=1.5), 
                 fontsize=10, color='black', fontweight='bold')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines1_twin, labels1_twin = ax1_twin.get_legend_handles_labels()
    keep_labels = ['台风风速 (kt)', '垂直风切变 (m/s)']
    filtered_lines = []
    filtered_labels = []
    for line, label in zip(lines1 + lines1_twin, labels1 + labels1_twin):
        if label in keep_labels:
            filtered_lines.append(line)
            filtered_labels.append(label)
    ax1.legend(filtered_lines, filtered_labels, loc='upper left', fontsize=10)

    ax1.set_xlim(-50, data1['time_rel'].max() + 20)

    # ---------------- 处理突然终止RI风暴（优化文字位置） ----------------
    sid2, storm2, matched2, ri_status2, name2 = sudden_stop_case
    t0_2 = ri_start_times[sid2]
    data2 = pd.merge_asof(
        storm2[['ISO_TIME', 'USA_WIND', 'RI_STATUS']].sort_values('ISO_TIME'),
        matched2[['ISO_TIME', 'SHRD']].sort_values('ISO_TIME'),
        on='ISO_TIME', direction='nearest'
    )
    data2['time_rel'] = (data2['ISO_TIME'] - t0_2).dt.total_seconds() / 3600
    ri_mask2 = data2['RI_STATUS']
    ri_times_rel2 = data2.loc[ri_mask2, 'time_rel']
    ri_start_time_rel2 = ri_times_rel2.min()

    # 找到风切变突增的点作为RI终止点
    post_t0_data = data2[data2['ISO_TIME'] >= t0_2].reset_index(drop=True)
    shear_surge_idx, _ = detect_shear_surge(post_t0_data, shear_col='SHRD', window=2, min_increase=5)
    
    if shear_surge_idx is not None:
        shear_surge_rel_time = post_t0_data.loc[shear_surge_idx, 'time_rel']
        shear_surge_wind = post_t0_data.loc[shear_surge_idx, 'USA_WIND']
        ri_end_time_rel2 = shear_surge_rel_time
        ri_end_wind2 = shear_surge_wind
    else:
        ri_end_time_rel2 = ri_times_rel2.max()
        ri_end_wind2 = data2.loc[data2['time_rel'] == ri_end_time_rel2, 'USA_WIND'].iloc[0]

    ax2.plot(data2['time_rel'], data2['USA_WIND'], color=colors['sudden'], linewidth=2.5, label='台风风速 (kt)')
    ax2.set_xlabel('相对于RI启动时刻 t0 的时间 (小时)', fontsize=12)
    ax2.set_ylabel('台风风速 (kt)', fontsize=12, color=colors['sudden'])
    ax2.tick_params(axis='y', labelcolor=colors['sudden'])
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.set_title(f'突然终止RI风暴：{name2}\n(风切变突增后，RI立刻终止)', fontsize=13, pad=15)

    ax2_twin = ax2.twinx()
    ax2_twin.plot(data2['time_rel'], data2['SHRD'], color='gray', linestyle='--', linewidth=2, label='垂直风切变 (m/s)')
    ax2_twin.set_ylabel('垂直风切变 (m/s)', fontsize=12, color='gray')
    ax2_twin.tick_params(axis='y', labelcolor='gray')

    # 标注RI启动点（文字移到左下方）
    t0_wind2 = data2.loc[data2['time_rel'] == ri_start_time_rel2, 'USA_WIND'].iloc[0]
    ax2.annotate('RI启动 (t0)', 
                 xy=(ri_start_time_rel2, t0_wind2), 
                 xytext=(-40, t0_wind2 - 12),
                 arrowprops=dict(arrowstyle='->', color='red', lw=1.5), 
                 fontsize=10, color='red', fontweight='bold')

    # 标注RI终止点（文字移到右上方）
    ax2.annotate('RI终止', 
                 xy=(ri_end_time_rel2, ri_end_wind2), 
                 xytext=(ri_end_time_rel2 + 15, ri_end_wind2 + 8),
                 arrowprops=dict(arrowstyle='->', color='black', lw=1.5), 
                 fontsize=10, color='black', fontweight='bold')

    lines2, labels2 = ax2.get_legend_handles_labels()
    lines2_twin, labels2_twin = ax2_twin.get_legend_handles_labels()
    keep_labels = ['台风风速 (kt)', '垂直风切变 (m/s)']
    filtered_lines2 = []
    filtered_labels2 = []
    for line, label in zip(lines2 + lines2_twin, labels2 + labels2_twin):
        if label in keep_labels:
            filtered_lines2.append(line)
            filtered_labels2.append(label)
    ax2.legend(filtered_lines2, filtered_labels2, loc='upper left', fontsize=10)

    ax2.set_xlim(data2['time_rel'].min() - 20, data2['time_rel'].max() + 20)

    plt.tight_layout()
    output_path = picture_dir / 'case_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  典型案例对比图已保存: {output_path}")

# -------------------------- 新增：构建预测数据集 --------------------------
def build_prediction_dataset(group_labels, ri_start_times, storm_matched_dict):
    """构建用于预测RI是否终止的数据集"""
    print("\n构建预测数据集...")
    features = []
    targets = []
    sids = []

    for sid, label in group_labels.items():
        t0 = ri_start_times[sid]
        t24 = t0 + pd.Timedelta(hours=24)
        matched = storm_matched_dict.get(sid)
        if matched is None or matched.empty:
            continue
        
        # 获取t0时刻数据
        diff0 = np.abs((matched['ISO_TIME'] - t0).dt.total_seconds())
        if diff0.min() > 3*3600:
            continue
        row0 = matched.loc[diff0.idxmin()]
        
        # 获取t0+24h时刻数据
        diff24 = np.abs((matched['ISO_TIME'] - t24).dt.total_seconds())
        if diff24.min() > 3*3600:
            continue
        row24 = matched.loc[diff24.idxmin()]
        
        # 计算特征
        feature_dict = {
            'RSST': row0['RSST'],
            'SHRD': row0['SHRD'],
            'RHMD': row0['RHMD'],
            'VMPI': row0['VMPI'],
            'MPI_DIFF': row0['VMPI'] - row0['USA_WIND'],
            'delta_RSST': row24['RSST'] - row0['RSST'],
            'delta_SHRD': row24['SHRD'] - row0['SHRD'],
            'delta_RHMD': row24['RHMD'] - row0['RHMD'],
            'delta_VMPI': row24['VMPI'] - row0['VMPI']
        }
        
        features.append(feature_dict)
        targets.append(label)
        sids.append(sid)

    df = pd.DataFrame(features)
    df['target'] = targets
    df['sid'] = sids
    
    print(f"  数据集构建完成，共 {len(df)} 个样本")
    print(f"  终止组样本数: {sum(df['target'] == 0)}")
    print(f"  持续组样本数: {sum(df['target'] == 1)}")
    
    return df

# -------------------------- 新增：相关性分析 --------------------------
def plot_correlation_heatmap(df):
    """绘制特征相关性热力图"""
    print("\n绘制特征相关性热力图...")
    corr = df.drop('sid', axis=1).corr()
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', square=True, linewidths=0.5)
    plt.title('特征相关性热力图', fontsize=15, pad=15)
    plt.tight_layout()
    output_path = picture_dir / 'correlation_heatmap.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  相关性热力图已保存: {output_path}")

# -------------------------- 新增：模型训练与评估 --------------------------
def train_and_evaluate_models(df):
    """训练逻辑回归和随机森林模型，输出评估结果和特征重要性"""
    print("\n训练预测模型...")
    X = df.drop(['target', 'sid'], axis=1)
    y = df['target']
    feature_names = X.columns
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y
    )
    
    # 标准化（用于逻辑回归）
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 训练逻辑回归
    lr = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)
    lr.fit(X_train_scaled, y_train)
    y_pred_lr = lr.predict(X_test_scaled)
    y_prob_lr = lr.predict_proba(X_test_scaled)[:, 1]
    
    print("\n" + "="*50)
    print("逻辑回归模型评估结果")
    print("="*50)
    print(f"准确率: {accuracy_score(y_test, y_pred_lr):.4f}")
    print(f"AUC: {roc_auc_score(y_test, y_prob_lr):.4f}")
    print("\n混淆矩阵:")
    print(confusion_matrix(y_test, y_pred_lr))
    print("\n分类报告:")
    print(classification_report(y_test, y_pred_lr, target_names=['终止组', '持续组']))
    
    # 训练随机森林
    rf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    y_prob_rf = rf.predict_proba(X_test)[:, 1]
    
    print("\n" + "="*50)
    print("随机森林模型评估结果")
    print("="*50)
    print(f"准确率: {accuracy_score(y_test, y_pred_rf):.4f}")
    print(f"AUC: {roc_auc_score(y_test, y_prob_rf):.4f}")
    print("\n混淆矩阵:")
    print(confusion_matrix(y_test, y_pred_rf))
    print("\n分类报告:")
    print(classification_report(y_test, y_pred_rf, target_names=['终止组', '持续组']))
    
    # 绘制逻辑回归系数
    plt.figure(figsize=(12, 6))
    coefs = lr.coef_[0]
    sorted_idx = np.argsort(coefs)
    plt.barh(feature_names[sorted_idx], coefs[sorted_idx], color='#1f77b4')
    plt.axvline(x=0, color='gray', linestyle='--', linewidth=1)
    plt.xlabel('标准化系数 (正=有利于RI持续)', fontsize=12)
    plt.title('逻辑回归标准化系数', fontsize=14, pad=15)
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    plt.tight_layout()
    output_path = picture_dir / 'logistic_regression_coefficients.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  逻辑回归系数图已保存: {output_path}")
    
    # 绘制随机森林特征重要性
    plt.figure(figsize=(12, 6))
    importances = rf.feature_importances_
    sorted_idx = np.argsort(importances)
    plt.barh(feature_names[sorted_idx], importances[sorted_idx], color='#ff7f0e')
    plt.xlabel('特征重要性', fontsize=12)
    plt.title('随机森林特征重要性', fontsize=14, pad=15)
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    plt.tight_layout()
    output_path = picture_dir / 'random_forest_feature_importance.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  随机森林特征重要性图已保存: {output_path}")

def main():
    print("=" * 70)
    print("基于台风个体的RI持续组与终止组环境因素对比分析")
    print("=" * 70)

    group_labels, ri_start_times, storm_data_dict, storm_matched_dict = group_and_extract()

    if len(group_labels) == 0:
        print("没有符合条件的台风，程序退出。")
        return

    plot_intensity_evolution(group_labels, ri_start_times, storm_data_dict)
    plot_t0_boxplots(group_labels, ri_start_times, storm_matched_dict)
    plot_delta_boxplots(group_labels, ri_start_times, storm_matched_dict)
    plot_environment_evolution(group_labels, ri_start_times, storm_matched_dict)

    perfect_case, sudden_stop_case = select_typical_cases(group_labels, ri_start_times, storm_data_dict, storm_matched_dict)
    plot_case_comparison(perfect_case, sudden_stop_case, ri_start_times)

    # 新增：预测分析部分
    pred_df = build_prediction_dataset(group_labels, ri_start_times, storm_matched_dict)
    if len(pred_df) > 0:
        plot_correlation_heatmap(pred_df)
        train_and_evaluate_models(pred_df)

    print("\n所有分析完成，图表已保存于:", picture_dir)

if __name__ == "__main__":
    main()
