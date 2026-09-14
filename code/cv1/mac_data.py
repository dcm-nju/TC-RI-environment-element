import pandas as pd
import numpy as np
from datetime import datetime
import warnings
import sys
from pathlib import Path

warnings.filterwarnings('ignore')

# ===================== 路径设置（与你完全一致） =====================
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent  # 项目根目录（e:\python\envioronment）

data_dir = project_root / 'data'
ibtracs_dir = data_dir / 'ibtracs'
ships_dir = data_dir / 'ships'
picture_dir = project_root / 'picture' / 'version1' / '1.4'
picture_dir.mkdir(parents=True, exist_ok=True)

ibtracs_file = ibtracs_dir / 'ibtracs.ALL.list.v04r01.csv'
ships_file = ships_dir / 'lsdiaga_1982_2023_sat_ts_7day.txt'
output_csv = picture_dir / 'typhoon_ri_dataset.csv'

# ===================== 常量配置 =====================
MISSING_VALUES = {9999, 9999.0, 999.9, 999, 999.0, -999, -999.0, -999.9}
RI_THRESHOLD = 30
RI_MAX_HOURS = 24
TIME_STEP_HOURS = 6
MAX_TIME_MATCH_SEC = 3600
MIN_DIST2LAND = 50

SHIPS_VARS = ['RSST', 'SHRD', 'RHLO', 'RHMD', 'RHHI', 'VMPI']
SHIPS_NEEDED_VARS = set(SHIPS_VARS)

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
def detect_ri_period(
    storm_data: pd.DataFrame,
    wind_col: str = 'USA_WIND',
    time_col: str = 'ISO_TIME',
    threshold: int = RI_THRESHOLD,
    max_hours: int = RI_MAX_HOURS
) -> tuple[bool, datetime | None, datetime | None]:
    storm_data = storm_data.sort_values(time_col).reset_index(drop=True)
    n = len(storm_data)
    if n < 2:
        return False, None, None

    times = storm_data[time_col].values.astype('datetime64[ns]')
    winds = storm_data[wind_col].values.astype(float)

    ri_moments = set()
    left = 0

    for right in range(n):
        while left < right:
            time_diff_hours = (times[right] - times[left]) / np.timedelta64(1, 'h')
            if time_diff_hours <= max_hours:
                break
            left += 1

        if right > left:
            wind_diffs = winds[right] - winds[left:right]
            if np.any(wind_diffs >= threshold):
                ri_moments.add(times[right])

    if not ri_moments:
        return False, None, None

    ri_start_time = min(ri_moments)
    last_ri_end_time = max(ri_moments)
    
    if isinstance(ri_start_time, np.datetime64):
        ri_start_time = pd.Timestamp(ri_start_time)
    if isinstance(last_ri_end_time, np.datetime64):
        last_ri_end_time = pd.Timestamp(last_ri_end_time)
    
    return True, ri_start_time, last_ri_end_time

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

# ===================== 主程序 =====================
def main():
    print("正在生成最终版数据集（保留24h变化量+允许空缺）...")

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
    print(f"步骤1：初始北大西洋台风数：{ibtracs_na['SID'].nunique()} 个")

    # 3. 筛选离岸距离
    valid_storms = []
    for sid, storm_data in ibtracs_na.groupby('SID'):
        if (storm_data['DIST2LAND'] > MIN_DIST2LAND).all():
            valid_storms.append(sid)
    ibtracs_valid = ibtracs_na[ibtracs_na['SID'].isin(valid_storms)].copy()
    print(f"步骤2：离岸距离>50km后剩余：{ibtracs_valid['SID'].nunique()} 个")

    # 4. 处理ATCF ID
    ibtracs_valid['USA_ATCF_ID'] = ibtracs_valid.groupby('SID')['USA_ATCF_ID'].ffill().bfill()
    ibtracs_valid['USA_ATCF_ID_STANDARD'] = ibtracs_valid['USA_ATCF_ID'].apply(standardize_atcf_id)
    storms_with_atcf = ibtracs_valid['SID'][ibtracs_valid['USA_ATCF_ID_STANDARD'].notna()].unique()
    ibtracs_valid = ibtracs_valid[ibtracs_valid['SID'].isin(storms_with_atcf)].copy()
    print(f"步骤3：有有效ATCF ID后剩余：{len(storms_with_atcf)} 个")

    # 5. 检测RI
    ri_storms = []
    ri_periods = {}
    storm_ids = list(ibtracs_valid.groupby('SID').groups.keys())
    
    for sid in storm_ids:
        storm_data = ibtracs_valid[ibtracs_valid['SID'] == sid]
        has_ri, ri_start, ri_end = detect_ri_period(storm_data)
        if has_ri:
            ri_storms.append(sid)
            ri_periods[sid] = (ri_start, ri_end)
    print(f"步骤4：检测到RI的台风数：{len(ri_storms)} 个")

    # 6. 提取所有RI期间的时间点（不要求连续，不要求台风有完整正样本）
    analysis_data = []
    for sid in ri_storms:
        storm_data = (
            ibtracs_valid[ibtracs_valid['SID'] == sid]
            .sort_values('ISO_TIME')
            .reset_index(drop=True)
        )
        ri_start_ts, ri_end_ts = ri_periods[sid]
        
        # 提取RI期间的所有时间点
        ri_window = storm_data[
            (storm_data['ISO_TIME'] >= ri_start_ts) & 
            (storm_data['ISO_TIME'] <= ri_end_ts)
        ].copy()
        
        if len(ri_window) < 1:  # 只要有至少1个时间点就保留
            continue
            
        # 计算每个时间点距离最后终止点的小时数
        ri_window['relative_hours'] = (
            (ri_window['ISO_TIME'] - ri_end_ts).dt.total_seconds() / 3600
        ).round().astype(int)
        
        ri_window['SID'] = sid
        ri_window['ri_start_time'] = ri_start_ts
        ri_window['last_ri_end_time'] = ri_end_ts
        analysis_data.append(ri_window)

    analysis_df = pd.concat(analysis_data, ignore_index=True)
    print(f"步骤5：提取RI期间数据后：{analysis_df['SID'].nunique()} 个台风，共 {len(analysis_df)} 条记录")

    # 7. 匹配SHIPS数据（按单个时间点匹配）
    matched_df = match_ships_data(analysis_df, ships_df)
    print(f"步骤6：匹配SHIPS数据后：{matched_df['SID'].nunique()} 个台风，共 {len(matched_df)} 条记录")

    # 8. 计算移动速度
    matched_df['MOVE_SPEED'] = np.nan
    for sid, grp in matched_df.groupby('SID'):
        speeds = calculate_movement_speed(grp)
        matched_df.loc[grp.index, 'MOVE_SPEED'] = speeds

    # 9. 基础数据清洗（仅过滤原始特征缺失/异常的行）
    matched_df = matched_df[(matched_df['RSST'] >= 20) & (matched_df['RSST'] <= 32)]
    matched_df = matched_df[(matched_df['MOVE_SPEED'] >= 0) & (matched_df['MOVE_SPEED'] <= 60)]
    # 仅删除原始7个环境要素和移动速度缺失的行
    matched_df = matched_df.dropna(subset=['RSST', 'SHRD', 'RHLO', 'RHMD', 'RHHI', 'VMPI', 'MOVE_SPEED'])
    print(f"步骤7：基础数据清洗后：{matched_df['SID'].nunique()} 个台风，共 {len(matched_df)} 条记录")

    # 10. 按单个时间点分配标签（不检查台风整体正样本完整性）
    print("正在按单个时间点分配标签...")
    positive_hours = [0, -6]
    matched_df['label'] = 0
    matched_df.loc[matched_df['relative_hours'].isin(positive_hours), 'label'] = 1

    # 11. 构造特征（完整保留6h和24h变化量，无法计算的留空）
    matched_df = matched_df.sort_values(['SID', 'ISO_TIME']).reset_index(drop=True)
    feature_base = SHIPS_VARS + ['MOVE_SPEED']

    # 计算6小时变化量
    for var in feature_base:
        matched_df[f'{var}_6h_change'] = matched_df.groupby('SID')[var].diff()

    # 计算24小时变化量（完整保留，无法计算的留NaN）
    for var in feature_base:
        temp = matched_df[['SID', 'ISO_TIME', var]].copy()
        temp['ISO_SHIFT_24'] = temp['ISO_TIME'] + pd.Timedelta(hours=24)
        temp = temp.rename(columns={var: f'{var}_ref_24h'})
        matched_df = matched_df.merge(
            temp[['SID', 'ISO_SHIFT_24', f'{var}_ref_24h']],
            left_on=['SID', 'ISO_TIME'],
            right_on=['SID', 'ISO_SHIFT_24'],
            how='left'
        )
        matched_df[f'{var}_24h_change'] = matched_df[var] - matched_df[f'{var}_ref_24h']
        matched_df.drop(columns=['ISO_SHIFT_24', f'{var}_ref_24h'], inplace=True)

    # 12. 最终整理（不删除变化量缺失的行）
    feature_cols = (
        feature_base
        + [f'{v}_6h_change' for v in feature_base]
        + [f'{v}_24h_change' for v in feature_base]
    )
    final_df = matched_df[['SID', 'ISO_TIME', 'relative_hours', 'ri_start_time', 'last_ri_end_time'] + feature_cols + ['label']]

    # 13. 保存并输出详细统计
    final_df.to_csv(output_csv, index=False, na_rep='NaN')
    pos_cnt = len(final_df[final_df['label'] == 1])
    neg_cnt = len(final_df[final_df['label'] == 0])
    unique_storms = final_df['SID'].nunique()
    
    # 统计缺失值情况
    total_cells = final_df[feature_cols].size
    missing_cells = final_df[feature_cols].isna().sum().sum()
    missing_rate = missing_cells / total_cells * 100

    print("\n" + "="*75)
    print("✅ 最终版数据集生成完毕（保留24h变化量+允许空缺）")
    print("="*75)
    print("📌 标签规则：")
    print("  label=1 正样本：距离最后RI终止点 0h、-6h 的所有有效时间点")
    print("  label=0 负样本：RI持续期间的所有其他有效时间点")
    print(f"\n📊 数据集统计：")
    print(f"  包含台风数：{unique_storms} 个")
    print(f"  总样本数：{len(final_df)}")
    print(f"  正样本数量：{pos_cnt}")
    print(f"  负样本数量：{neg_cnt}")
    print(f"  正负样本比例：1 : {neg_cnt/pos_cnt:.2f}")
    print(f"  特征总数：{len(feature_cols)}")
    print(f"  特征缺失率：{missing_rate:.2f}%")
    print(f"  文件输出路径：{output_csv}")
    print("="*75)

if __name__ == "__main__":
    main()
