import pandas as pd
import numpy as np
from datetime import datetime
import warnings
import sys
from pathlib import Path
import matplotlib.pyplot as plt

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
picture_dir = project_root / 'picture' / 'version1' / '1.1'
picture_dir.mkdir(parents=True, exist_ok=True)

ibtracs_file = ibtracs_dir / 'ibtracs.ALL.list.v04r01.csv'
ships_file = ships_dir / 'lsdiaga_1982_2023_sat_ts_7day.txt'

# ===================== 常量配置 =====================
MISSING_VALUES = {9999, 9999.0, 999.9, 999, 999.0, -999, -999.0, -999.9}

RI_THRESHOLD = 30
RI_MAX_HOURS = 24
TIME_STEP_HOURS = 6
MAX_TIME_MATCH_SEC = 3600

SHIPS_VARS = ['RSST', 'SHRD', 'RHLO', 'RHMD', 'RHHI', 'VMPI']
SHIPS_NEEDED_VARS = set(SHIPS_VARS)

# 绘图变量配置
PLOT_VARS = ['RSST', 'SHRD', 'RHLO', 'RHMD', 'RHHI', 'VMPI', 'MOVE_SPEED']
PLOT_LABELS = {
    'RSST': '海表温度 (℃)',
    'SHRD': '垂直风切变 (m/s)',
    'RHLO': '低层相对湿度 (%)',
    'RHMD': '中层相对湿度 (%)',
    'RHHI': '高层相对湿度 (%)',
    'VMPI': '最大潜在强度 (kt)',
    'MOVE_SPEED': '移动速度 (kt)'
}
# 三组颜色：持续点(蓝)、中途暂停点(橙)、最后终止点(绿)
PLOT_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c']
PLOT_GROUP_NAMES = ['RI持续点', 'RI中途暂停点', 'RI最后终止点']

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

# ===================== 扩展RI检测：返回所有RI时间段 =====================
def detect_all_ri_periods(
    storm_data: pd.DataFrame,
    wind_col: str = 'USA_WIND',
    time_col: str = 'ISO_TIME',
    threshold: int = RI_THRESHOLD,
    max_hours: int = RI_MAX_HOURS
) -> tuple[bool, datetime | None, list[tuple[datetime, datetime]]]:
    """
    检测所有RI时间段
    返回：(是否有RI, 最后一个RI结束时间, 所有RI时间段列表[(开始时间, 结束时间)])
    """
    storm_data = storm_data.sort_values(time_col).reset_index(drop=True)
    n = len(storm_data)
    if n < 2:
        return False, None, []

    times = storm_data[time_col].values.astype('datetime64[ns]')
    winds = storm_data[wind_col].values.astype(float)

    ri_end_times = set()
    ri_periods = []  # 存储所有RI时间段 (start_time, end_time)
    
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
                ri_end_times.add(ri_end_time)
                ri_periods.append((ri_start_time, ri_end_time))

    if not ri_end_times:
        return False, None, []
    
    last_time = max(ri_end_times)
    return True, last_time, ri_periods

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

    atcf_index: dict[str, dict] = {}
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

# ===================== 核心：提取三组数据 =====================
def extract_three_groups():
    """提取三组数据：持续点、中途暂停点、最后终止点"""
    print("正在读取原始数据...")
    
    # 1. 读取数据
    if not ibtracs_file.exists():
        print(f"错误：IBTrACS文件不存在: {ibtracs_file}")
        sys.exit(1)
    ibtracs_df = pd.read_csv(ibtracs_file, low_memory=False, skiprows=[1])
    ships_df = read_ships_file(ships_file)

    # 2. IBTrACS预处理
    ibtracs_df['ISO_TIME'] = pd.to_datetime(ibtracs_df['ISO_TIME'])
    ibtracs_df['SEASON'] = ibtracs_df['SEASON'].astype(int)
    ibtracs_df['USA_WIND'] = pd.to_numeric(ibtracs_df['USA_WIND'], errors='coerce')
    ibtracs_df = ibtracs_df.dropna(subset=['USA_WIND'])
    ibtracs_df = ibtracs_df[(ibtracs_df['SEASON'] >= 1988) & (ibtracs_df['SEASON'] <= 2021)]

    # 筛选北大西洋
    na_mask = (
        (ibtracs_df['BASIN'] == 'AL') |
        (
            (ibtracs_df['BASIN'].isna()) &
            (ibtracs_df['LON'] >= -100) & (ibtracs_df['LON'] <= 0) &
            (ibtracs_df['LAT'] >= 0) & (ibtracs_df['LAT'] <= 45)
        )
    )
    ibtracs_na = ibtracs_df[na_mask].copy()

    # 3. 筛选离岸距离>100km
    valid_storms = []
    for sid, storm_data in ibtracs_na.groupby('SID'):
        if (storm_data['DIST2LAND'] > 100).all():
            valid_storms.append(sid)
    ibtracs_valid = ibtracs_na[ibtracs_na['SID'].isin(valid_storms)].copy()

    # 4. 处理ATCF ID
    ibtracs_valid['USA_ATCF_ID'] = ibtracs_valid.groupby('SID')['USA_ATCF_ID'].ffill().bfill()
    ibtracs_valid['USA_ATCF_ID_STANDARD'] = ibtracs_valid['USA_ATCF_ID'].apply(standardize_atcf_id)
    storms_with_atcf = ibtracs_valid['SID'][ibtracs_valid['USA_ATCF_ID_STANDARD'].notna()].unique()
    ibtracs_valid = ibtracs_valid[ibtracs_valid['SID'].isin(storms_with_atcf)].copy()

    # 5. 检测所有RI时间段
    ri_storms = []
    all_ri_periods = {}  # 存储每个台风的所有RI时间段
    storm_ids = list(ibtracs_valid.groupby('SID').groups.keys())
    
    print(f"正在检测RI事件，共{len(storm_ids)}个台风...")
    for sid in storm_ids:
        storm_data = ibtracs_valid[ibtracs_valid['SID'] == sid]
        has_ri, _, ri_periods = detect_all_ri_periods(storm_data)
        if has_ri:
            ri_storms.append(sid)
            all_ri_periods[sid] = ri_periods
    
    if not ri_storms:
        print("错误：未检测到任何快速增强台风")
        sys.exit(0)
    
    # 统计RI次数分布
    ri_count_dist = {}
    for sid in ri_storms:
        cnt = len(all_ri_periods[sid])
        ri_count_dist[cnt] = ri_count_dist.get(cnt, 0) + 1
    
    print(f"\n检测到{len(ri_storms)}个包含RI事件的台风")
    print("RI次数分布：")
    for cnt in sorted(ri_count_dist.keys()):
        print(f"  {cnt}次RI: {ri_count_dist[cnt]}个台风")
    total_ri_events = sum(cnt * ri_count_dist[cnt] for cnt in ri_count_dist)
    print(f"总RI事件数: {total_ri_events}")
    print(f"平均每个台风RI次数: {total_ri_events/len(ri_storms):.2f}")

    # 6. 提取三组数据
    print("\n正在提取三组数据...")
    ongoing_points = []    # RI持续点
    pause_points = []      # RI中途暂停点
    final_termination_points = []  # RI最后终止点
    
    for sid in ri_storms:
        storm_data = ibtracs_valid[ibtracs_valid['SID'] == sid].sort_values('ISO_TIME').reset_index(drop=True)
        ri_periods = all_ri_periods[sid]
        num_ri = len(ri_periods)
        
        # 匹配SHIPS数据和计算移动速度
        storm_matched = match_ships_data(storm_data, ships_df)
        if len(storm_matched) == 0:
            continue
            
        # 计算移动速度
        speeds = calculate_movement_speed(storm_matched)
        storm_matched['MOVE_SPEED'] = speeds
        
        # 基础清洗（与原代码完全一致）
        storm_matched = storm_matched[(storm_matched['RSST'] >= 22) & (storm_matched['RSST'] <= 30)]
        storm_matched = storm_matched[(storm_matched['MOVE_SPEED'] >= 0) & (storm_matched['MOVE_SPEED'] <= 50)]
        storm_matched = storm_matched.dropna(subset=['RSST', 'SHRD', 'RHLO', 'RHMD', 'RHHI', 'VMPI', 'MOVE_SPEED'])
        
        if len(storm_matched) == 0:
            continue
        
        # 遍历每个RI事件
        for i, (start_time, end_time) in enumerate(ri_periods):
            # 提取持续点：[开始时间, 结束时间)
            ongoing_mask = (
                (storm_matched['ISO_TIME'] >= start_time) & 
                (storm_matched['ISO_TIME'] < end_time)
            )
            ongoing_rows = storm_matched[ongoing_mask]
            for _, row in ongoing_rows.iterrows():
                ongoing_points.append(row.to_dict())
            
            # 提取终止点
            termination_row = storm_matched[storm_matched['ISO_TIME'] == end_time]
            if not termination_row.empty:
                row_dict = termination_row.iloc[0].to_dict()
                if i == num_ri - 1:
                    # 最后一次RI的终止点
                    final_termination_points.append(row_dict)
                else:
                    # 中途RI的终止点（暂停点）
                    pause_points.append(row_dict)
    
    # 转换为DataFrame
    ongoing_df = pd.DataFrame(ongoing_points)
    pause_df = pd.DataFrame(pause_points)
    final_termination_df = pd.DataFrame(final_termination_points)
    
    print(f"\n数据提取完成：")
    print(f"  RI持续点: {len(ongoing_df)} 个")
    print(f"  RI中途暂停点: {len(pause_df)} 个")
    print(f"  RI最后终止点: {len(final_termination_df)} 个")
    
    return ongoing_df, pause_df, final_termination_df

# ===================== 绘制三组对比箱线图 =====================
def plot_three_groups_boxplots(ongoing_df, pause_df, final_termination_df):
    """绘制三组对比箱线图：持续点、中途暂停点、最后终止点"""
    print("\n开始绘制三组对比箱线图...")
    
    data_groups = [ongoing_df, pause_df, final_termination_df]
    
    for var in PLOT_VARS:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # 准备数据
        plot_data = []
        for df in data_groups:
            plot_data.append(df[var].dropna().values)
        
        # 绘制箱线图
        box = ax.boxplot(
            plot_data,
            labels=PLOT_GROUP_NAMES,
            patch_artist=True,
            medianprops={'color': 'black', 'linewidth': 1.5},
            meanline=True,
            showmeans=True,
            meanprops={'color': 'red', 'ls': '--', 'linewidth': 1.5},
            flierprops={"marker": "o", "markerfacecolor": "gray", "markersize": 4, "alpha": 0.6}
        )
        
        # 设置颜色
        for patch, color in zip(box['boxes'], PLOT_COLORS):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        # 设置标题和标签
        ax.set_title(f'{PLOT_LABELS[var]}对比：RI持续点 vs 中途暂停点 vs 最后终止点', fontsize=14, pad=15)
        ax.set_ylabel(PLOT_LABELS[var], fontsize=12)
        ax.tick_params(axis='both', labelsize=11)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        # 添加样本数量标注
        y_min, y_max = ax.get_ylim()
        y_offset = (y_max - y_min) * 0.02
        for i, data in enumerate(plot_data):
            ax.text(i+1, y_min + y_offset, f'n={len(data)}', ha='center', va='bottom', fontsize=10)
        
        # 保存图片
        output_file = picture_dir / f'boxplot_{var}_three_groups.png'
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  已保存: {output_file.name}")
    
    print("\n所有三组对比箱线图绘制完成！")
    
    # 输出统计汇总表
    print("\n" + "="*100)
    print("三组环境因素统计量汇总表")
    print("="*100)
    print(f"{'变量':<12} {'持续点均值':<12} {'暂停点均值':<12} {'最后终止点均值':<14} {'持续点中位数':<14} {'暂停点中位数':<14} {'最后终止点中位数':<16}")
    print("-"*100)
    
    for var in PLOT_VARS:
        ong_mean = np.mean(ongoing_df[var].dropna())
        pau_mean = np.mean(pause_df[var].dropna())
        fin_mean = np.mean(final_termination_df[var].dropna())
        
        ong_med = np.median(ongoing_df[var].dropna())
        pau_med = np.median(pause_df[var].dropna())
        fin_med = np.median(final_termination_df[var].dropna())
        
        print(f"{PLOT_LABELS[var].split(' ')[0]:<12} {ong_mean:<12.2f} {pau_mean:<12.2f} {fin_mean:<14.2f} "
              f"{ong_med:<14.2f} {pau_med:<14.2f} {fin_med:<16.2f}")
    
    print("="*100)

# ===================== 主程序 =====================
def main():
    print("="*70)
    print("RI持续点、中途暂停点与最后终止点环境因素对比分析")
    print("="*70)
    
    # 提取三组数据
    ongoing_df, pause_df, final_termination_df = extract_three_groups()
    
    # 检查数据有效性
    if len(ongoing_df) == 0 or len(pause_df) == 0 or len(final_termination_df) == 0:
        print("错误：某组数据为空，无法绘制箱线图")
        sys.exit(1)
    
    # 绘制箱线图
    plot_three_groups_boxplots(ongoing_df, pause_df, final_termination_df)
    
    print("\n程序执行完成！所有图片已保存到:", picture_dir)

if __name__ == "__main__":
    main()
