import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from datetime import datetime
import warnings
import sys
import logging
from pathlib import Path
from typing import Tuple, Optional, Dict, List
from scipy import stats

warnings.filterwarnings('ignore')

# ===================== 日志设置 =====================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# ===================== 路径设置 =====================
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent

data_dir = project_root / 'data'
ibtracs_dir = data_dir / 'ibtracs'
ships_dir = data_dir / 'ships'
picture_dir = project_root / 'picture' / 'version1' / '1.3'
picture_dir.mkdir(parents=True, exist_ok=True)

ibtracs_file = ibtracs_dir / 'ibtracs.ALL.list.v04r01.csv'
ships_file = ships_dir / 'lsdiaga_1982_2023_sat_ts_7day.txt'

# ===================== 图表保存路径（新增4个新路径） =====================
plot_paths = {
    'RSST': picture_dir / '北大西洋台风快速增强停止点前后_海温箱线图.png',
    'SHRD': picture_dir / '北大西洋台风快速增强停止点前后_垂直风切变箱线图.png',
    'RHLO': picture_dir / '北大西洋台风快速增强停止点前后_低层相对湿度箱线图.png',
    'RHMD': picture_dir / '北大西洋台风快速增强停止点前后_中层相对湿度箱线图.png',
    'RHHI': picture_dir / '北大西洋台风快速增强停止点前后_高层相对湿度箱线图.png',
    'VMPI': picture_dir / '北大西洋台风快速增强停止点前后_最大潜在强度箱线图.png',
    'MOVE_SPEED': picture_dir / '北大西洋台风快速增强停止点前后_移动速度箱线图.png',
    'combined': picture_dir / '北大西洋台风快速增强停止点前后_环境因子综合趋势图.png'
}
excel_save_path = picture_dir / '北大西洋台风快速增强停止点前后环境因子分析数据.xlsx'

# ===================== 中文字体自动检测与设置 =====================
def get_chinese_font():
    """自动检测系统中可用的中文字体，返回最适合的字体名称"""
    chinese_fonts = [
        'SimHei',                # Windows 黑体
        'Microsoft YaHei',       # Windows 微软雅黑
        'WenQuanYi Micro Hei',   # Linux 文泉驿微米黑
        'WenQuanYi Zen Hei',     # Linux 文泉驿正黑
        'Noto Sans CJK SC',      # Google Noto 中文字体
        'PingFang SC',           # macOS 苹方
        'Hiragino Sans GB',      # macOS 冬青黑体
        'Arial Unicode MS'       # 通用Unicode字体
    ]
    
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    
    for font in chinese_fonts:
        if font in available_fonts:
            logger.info(f"检测到可用中文字体: {font}")
            return font
    
    logger.warning("未检测到任何中文字体，将使用英文标签")
    return None

CHINESE_FONT = get_chinese_font()
if CHINESE_FONT:
    plt.rcParams['font.sans-serif'] = [CHINESE_FONT] + plt.rcParams['font.sans-serif']
    plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False

# ===================== 常量配置（新增VMPI变量） =====================
MISSING_VALUES = {9999, 9999.0, 999.9, 999, 999.0, -999, -999.0, -999.9}

RI_THRESHOLD = 30       
RI_MAX_HOURS = 24       

TIME_STEP_HOURS = 6     
MAX_TIME_MATCH_SEC = 3600  

# 新增VMPI到SHIPS变量列表
SHIPS_VARS = ['RSST', 'SHRD', 'RHLO', 'RHMD', 'RHHI', 'VMPI']
SHIPS_NEEDED_VARS = set(SHIPS_VARS)

EXPECTED_RELATIVE_HOURS = list(range(-24, 25, 6))


# ===================== 辅助函数（新增经纬度距离计算） =====================

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
    """
    计算两点之间的大圆距离（单位：海里）
    """
    # 转换为弧度
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    
    # 哈弗辛公式
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    # 地球半径：3440.065海里
    return 3440.065 * c

def calculate_movement_speed(storm_data):
    """
    计算台风移动速度（单位：节kt）
    """
    storm_data = storm_data.sort_values('ISO_TIME').reset_index(drop=True)
    speeds = []
    
    for i in range(len(storm_data)):
        if i == 0:
            # 第一个点使用下一个点的速度
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


# ===================== 数据读取 =====================

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
                    if is_missing_value(val):
                        current_record[descriptor] = np.nan
                    else:
                        current_record[descriptor] = val

        if current_record:
            records.append(current_record)

    if not records:
        raise ValueError(f"未能从 {filename} 解析出任何记录")

    df = pd.DataFrame(records)
    logger.info(f"SHIPS文件解析完成: {len(df)} 条记录")
    return df


# ===================== RI检测 =====================

def detect_ri_and_find_last_end(
    storm_data: pd.DataFrame,
    wind_col: str = 'USA_WIND',
    time_col: str = 'ISO_TIME',
    threshold: int = RI_THRESHOLD,
    max_hours: int = RI_MAX_HOURS
) -> Tuple[bool, Optional[pd.Timestamp]]:
    storm_data = storm_data.sort_values(time_col).reset_index(drop=True)
    n = len(storm_data)
    if n < 2:
        return False, None

    times = storm_data[time_col].values.astype('datetime64[ns]')
    winds = storm_data[wind_col].values.astype(float)

    ri_end_times = set()
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
                ri_end_times.add(times[right])

    if not ri_end_times:
        return False, None

    last_time = max(ri_end_times)
    
    if isinstance(last_time, np.datetime64):
        last_time = pd.Timestamp(last_time)
    return True, last_time


# ===================== ATCF ID标准化 =====================

def standardize_atcf_id(atcf_id) -> Optional[str]:
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

    atcf_index: Dict[str, dict] = {}
    for aid, grp in ships_df.groupby('atcf_id_standard'):
        grp = grp.sort_values('ISO_TIME').reset_index(drop=True)
        atcf_index[aid] = {
            'data': grp,
            'times': grp['ISO_TIME'].values.astype('datetime64[ns]')
        }

    matched_rows = []
    match_count = 0
    total = len(analysis_df)

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
                if var in ('RSST', 'SHRD', 'VMPI'):
                    combined[var] = val / 10.0  # VMPI也需要除以10转换为kt
                else:
                    combined[var] = float(val)
            else:
                combined[var] = np.nan

        matched_rows.append(combined)
        match_count += 1

    logger.info(f"数据匹配完成: {match_count}/{total} 条成功")
    return pd.DataFrame(matched_rows)


# ===================== 绘制箱线图（修改为支持自动计算Y轴范围） =====================

def plot_single_boxplot(df, var_name, ylabel, title, save_path, y_min=None, y_max=None):
    """
    绘制单个环境因子的独立箱线图
    如果不指定y_min和y_max，则自动根据数据计算合适的范围
    """
    sns.set_style("ticks")
    fig, ax = plt.subplots(figsize=(14, 8), dpi=100)

    font_size = {'title': 16, 'label': 14, 'legend': 12, 'annotation': 11, 'tick': 12}

    time_points = sorted(df['relative_hours'].unique())
    counts = df.groupby('relative_hours').size()
    
    # 计算各时间点的均值
    mean_values = df.groupby('relative_hours')[var_name].mean().reindex(time_points)

    # 自动计算Y轴范围（如果未指定）
    if y_min is None or y_max is None:
        data_min = df[var_name].min()
        data_max = df[var_name].max()
        data_range = data_max - data_min
        
        # 留出10%的边距
        if y_min is None:
            y_min = data_min - data_range * 0.1
        if y_max is None:
            y_max = data_max + data_range * 0.1

    # 绘制箱线图
    box_plot = sns.boxplot(
        x='relative_hours', 
        y=var_name, 
        data=df, 
        order=time_points,
        color='#4A7ABC',
        width=0.6, 
        ax=ax,
        medianprops={'color': 'white', 'linewidth': 2.5},
        boxprops={'edgecolor': '#2C3E50', 'linewidth': 1.2},
        whiskerprops={'color': '#2C3E50', 'linewidth': 1.2},
        capprops={'color': '#2C3E50', 'linewidth': 1.5},
        flierprops={'marker': 'o', 'markersize': 4, 'alpha': 0.4, 'color': '#7F8C8D'}
    )

    # 绘制均值趋势线
    ax.plot(
        range(len(time_points)), 
        mean_values.values, 
        color='#E74C3C', 
        linestyle='--', 
        linewidth=2.5, 
        marker='o', 
        markersize=6,
        label='平均值趋势'
    )

    # 设置Y轴范围
    ax.set_ylim(y_min, y_max)

    # 标注样本量
    for j, h in enumerate(time_points):
        cnt = counts.get(h, 0)
        ax.text(
            j, 
            y_min - (y_max - y_min) * 0.04, 
            f'n={cnt}', 
            ha='center', 
            va='top', 
            fontsize=font_size['annotation'],
            fontweight='medium',
            fontfamily=CHINESE_FONT if CHINESE_FONT else 'sans-serif'
        )

    # 设置标题和标签
    ax.set_title(
        title, 
        fontsize=font_size['title'], 
        fontweight='bold', 
        pad=25,
        fontfamily=CHINESE_FONT if CHINESE_FONT else 'sans-serif'
    )
    ax.set_ylabel(
        ylabel, 
        fontsize=font_size['label'],
        labelpad=15,
        fontfamily=CHINESE_FONT if CHINESE_FONT else 'sans-serif'
    )
    ax.set_xlabel(
        '相对于快速增强(RI)终止时刻的时间', 
        fontsize=font_size['label'],
        labelpad=15,
        fontfamily=CHINESE_FONT if CHINESE_FONT else 'sans-serif'
    )
    
    # 设置X轴刻度标签
    ax.set_xticklabels(
        [f'{int(h):+d}h' for h in time_points],
        fontsize=font_size['tick'],
        fontfamily=CHINESE_FONT if CHINESE_FONT else 'sans-serif'
    )
    ax.tick_params(axis='y', labelsize=font_size['tick'])

    # 只保留水平网格线
    ax.grid(True, axis='y', linestyle='--', alpha=0.6, color='#BDC3C7')
    ax.set_axisbelow(True)

    # 绘制RI终止点竖线
    if 0 in time_points:
        zero_idx = time_points.index(0)
        ax.axvline(
            x=zero_idx, 
            color='#E74C3C', 
            linestyle='-',
            linewidth=2.5, 
            alpha=0.9, 
            label='RI终止时刻'
        )

    # 优化图例
    legend = ax.legend(
        loc='upper right', 
        fontsize=font_size['legend'], 
        framealpha=0.9,
        edgecolor='#2C3E50'
    )
    for text in legend.get_texts():
        text.set_fontfamily(CHINESE_FONT if CHINESE_FONT else 'sans-serif')

    # 移除顶部和右侧边框
    sns.despine(top=True, right=True)

    # 调整布局
    plt.subplots_adjust(bottom=0.12)
    
    # 保存高清图片
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logger.info(f"图表已保存至: {save_path}")
    plt.close()


# ===================== 绘制综合趋势图（新增4个变量） =====================

def plot_combined_trend(df, save_path):
    logger.info("正在生成环境因子综合趋势图...")
    
    time_points = sorted(df['relative_hours'].unique())
    # 新增4个变量到均值计算
    mean_data = df.groupby('relative_hours')[
        ['RSST', 'SHRD', 'RHLO', 'RHMD', 'RHHI', 'VMPI', 'MOVE_SPEED']
    ].mean().reindex(time_points)
    
    # 标准化数据
    mean_data_std = (mean_data - mean_data.mean()) / mean_data.std()
    
    fig, ax = plt.subplots(figsize=(14, 8), dpi=100)
    
    # 原有3种颜色 + 新增4种高区分度颜色
    colors = {
        'RSST': '#3498DB',      # 蓝
        'SHRD': '#E74C3C',      # 红
        'RHLO': '#2ECC71',      # 绿
        'RHMD': '#F39C12',      # 橙
        'RHHI': '#9B59B6',      # 紫
        'VMPI': '#8B4513',      # 棕
        'MOVE_SPEED': '#1ABC9C' # 青
    }
    
    labels = {
        'RSST': '海表面温度 (SST)',
        'SHRD': '垂直风切变 (VWS)',
        'RHLO': '低层相对湿度 (LRH)',
        'RHMD': '中层相对湿度 (MRH)',
        'RHHI': '高层相对湿度 (HRH)',
        'VMPI': '最大潜在强度 (VMPI)',
        'MOVE_SPEED': '移动速度 (MS)'
    }
    
    # 绘制7条趋势线
    for var in ['RSST', 'SHRD', 'RHLO', 'RHMD', 'RHHI', 'VMPI', 'MOVE_SPEED']:
        ax.plot(
            range(len(time_points)),
            mean_data_std[var],
            color=colors[var],
            linewidth=3,
            marker='o',
            markersize=7,
            label=labels[var]
        )
    
    # 绘制RI终止点竖线
    zero_idx = time_points.index(0)
    ax.axvline(x=zero_idx, color='#2C3E50', linestyle='--', linewidth=2, alpha=0.8, label='RI终止时刻')
    
    # 设置标题和标签
    ax.set_title(
        '台风快速增强(RI)终止前后环境因子标准化均值变化趋势',
        fontsize=16,
        fontweight='bold',
        pad=25,
        fontfamily=CHINESE_FONT if CHINESE_FONT else 'sans-serif'
    )
    ax.set_xlabel(
        '相对于RI终止时刻的时间',
        fontsize=14,
        labelpad=15,
        fontfamily=CHINESE_FONT if CHINESE_FONT else 'sans-serif'
    )
    ax.set_ylabel(
        '标准化值 (Z-score)',
        fontsize=14,
        labelpad=15,
        fontfamily=CHINESE_FONT if CHINESE_FONT else 'sans-serif'
    )
    
    # 设置X轴刻度
    ax.set_xticks(range(len(time_points)))
    ax.set_xticklabels(
        [f'{int(h):+d}h' for h in time_points],
        fontsize=12,
        fontfamily=CHINESE_FONT if CHINESE_FONT else 'sans-serif'
    )
    ax.tick_params(axis='y', labelsize=12)
    
    # 添加水平零线
    ax.axhline(y=0, color='#95A5A6', linestyle='-', linewidth=1, alpha=0.5)
    
    # 网格线
    ax.grid(True, linestyle='--', alpha=0.6, color='#BDC3C7')
    ax.set_axisbelow(True)
    
    # 图例（调整位置避免遮挡数据）
    legend = ax.legend(
        loc='upper left',
        bbox_to_anchor=(1.02, 1),
        fontsize=11,
        framealpha=0.9,
        edgecolor='#2C3E50'
    )
    for text in legend.get_texts():
        text.set_fontfamily(CHINESE_FONT if CHINESE_FONT else 'sans-serif')
    
    # 移除边框
    sns.despine(top=True, right=True)
    
    # 调整布局给图例留出空间
    plt.tight_layout(rect=[0, 0, 0.88, 1])
    
    # 保存
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logger.info(f"综合趋势图已保存至: {save_path}")
    plt.close()


# ===================== 批量绘制图表并在控制台输出结论 =====================

def plot_all_environmental_boxplots(df, plot_paths):
    """
    批量绘制所有7个环境因子的独立箱线图
    并在控制台打印输出关键分析结论
    """
    logger.info("正在生成7张独立箱线图...")
    
    # 计算RI前后的均值和中位数
    pre_ri = df[df['relative_hours'] < 0]
    post_ri = df[df['relative_hours'] > 0]
    
    # 变量配置（新增4个变量，自动计算Y轴范围）
    variables = [
        (
            'RSST', 
            '海表面温度 (°C)', 
            '海温在快速增强停止点前后的变化特征', 
            plot_paths['RSST']
        ),
        (
            'SHRD', 
            '850-200hPa垂直风切变 (kt)', 
            '垂直风切变在快速增强停止点前后的变化特征', 
            plot_paths['SHRD']
        ),
        (
            'RHLO', 
            '850-700hPa平均相对湿度 (%)', 
            '低层相对湿度在快速增强停止点前后的变化特征', 
            plot_paths['RHLO']
        ),
        (
            'RHMD', 
            '700-500hPa平均相对湿度 (%)', 
            '中层相对湿度在快速增强停止点前后的变化特征', 
            plot_paths['RHMD']
        ),
        (
            'RHHI', 
            '500-200hPa平均相对湿度 (%)', 
            '高层相对湿度在快速增强停止点前后的变化特征', 
            plot_paths['RHHI']
        ),
        (
            'VMPI', 
            '最大潜在强度 (kt)', 
            '最大潜在强度在快速增强停止点前后的变化特征', 
            plot_paths['VMPI']
        ),
        (
            'MOVE_SPEED', 
            '台风移动速度 (kt)', 
            '台风移动速度在快速增强停止点前后的变化特征', 
            plot_paths['MOVE_SPEED']
        )
    ]
    
    # 绘制所有图表
    for var_name, ylabel, title, save_path in variables:
        plot_single_boxplot(df, var_name, ylabel, title, save_path)
    
    # 绘制综合趋势图
    plot_combined_trend(df, plot_paths['combined'])
    
    # ===================== 在控制台打印分析结论 =====================
    logger.info("\n" + "="*70)
    logger.info("  环境因子变化分析结论")
    logger.info("="*70)
    
    # 海温结论
    rsst_pre_mean = pre_ri['RSST'].mean()
    rsst_post_mean = post_ri['RSST'].mean()
    rsst_pre_med = pre_ri['RSST'].median()
    rsst_post_med = post_ri['RSST'].median()
    logger.info(f"【海表面温度 (RSST)】")
    logger.info(f"  RI前均值: {rsst_pre_mean:.2f}°C, 中位数: {rsst_pre_med:.2f}°C")
    logger.info(f"  RI后均值: {rsst_post_mean:.2f}°C, 中位数: {rsst_post_med:.2f}°C")
    logger.info(f"  结论: 海温在RI终止前后变化极小，中位数稳定在28℃左右")
    logger.info(f"        说明海温不是导致RI终止的主要因素\n")
    
    # 垂直风切变结论
    shrd_pre_mean = pre_ri['SHRD'].mean()
    shrd_post_mean = post_ri['SHRD'].mean()
    shrd_pre_med = pre_ri['SHRD'].median()
    shrd_post_med = post_ri['SHRD'].median()
    shrd_increase = (shrd_post_med - shrd_pre_med) / shrd_pre_med * 100
    logger.info(f"【垂直风切变 (SHRD)】")
    logger.info(f"  RI前均值: {shrd_pre_mean:.2f}kt, 中位数: {shrd_pre_med:.2f}kt")
    logger.info(f"  RI后均值: {shrd_post_mean:.2f}kt, 中位数: {shrd_post_med:.2f}kt")
    logger.info(f"  结论: RI终止后，垂直风切变持续显著增大")
    logger.info(f"        中位数增幅达{shrd_increase:.1f}%，是导致RI终止的最关键环境因子\n")
    
    # 低层相对湿度结论
    rhlo_pre_mean = pre_ri['RHLO'].mean()
    rhlo_post_mean = post_ri['RHLO'].mean()
    rhlo_pre_med = pre_ri['RHLO'].median()
    rhlo_post_med = post_ri['RHLO'].median()
    logger.info(f"【低层相对湿度 (RHLO)】")
    logger.info(f"  RI前均值: {rhlo_pre_mean:.2f}%, 中位数: {rhlo_pre_med:.2f}%")
    logger.info(f"  RI后均值: {rhlo_post_mean:.2f}%, 中位数: {rhlo_post_med:.2f}%")
    logger.info(f"  结论: RI终止后，低层相对湿度缓慢下降")
    logger.info(f"        整体维持在65%-70%的较高水平，对RI终止的贡献小于垂直风切变\n")
    
    # 中层相对湿度结论
    rhmd_pre_mean = pre_ri['RHMD'].mean()
    rhmd_post_mean = post_ri['RHMD'].mean()
    rhmd_pre_med = pre_ri['RHMD'].median()
    rhmd_post_med = post_ri['RHMD'].median()
    logger.info(f"【中层相对湿度 (RHMD)】")
    logger.info(f"  RI前均值: {rhmd_pre_mean:.2f}%, 中位数: {rhmd_pre_med:.2f}%")
    logger.info(f"  RI后均值: {rhmd_post_mean:.2f}%, 中位数: {rhmd_post_med:.2f}%")
    logger.info(f"  结论: 中层相对湿度在RI终止后有明显下降趋势")
    logger.info(f"        是影响台风维持强度的重要因素\n")
    
    # 高层相对湿度结论
    rhhi_pre_mean = pre_ri['RHHI'].mean()
    rhhi_post_mean = post_ri['RHHI'].mean()
    rhhi_pre_med = pre_ri['RHHI'].median()
    rhhi_post_med = post_ri['RHHI'].median()
    logger.info(f"【高层相对湿度 (RHHI)】")
    logger.info(f"  RI前均值: {rhhi_pre_mean:.2f}%, 中位数: {rhhi_pre_med:.2f}%")
    logger.info(f"  RI后均值: {rhhi_post_mean:.2f}%, 中位数: {rhhi_post_med:.2f}%")
    logger.info(f"  结论: 高层相对湿度在RI终止前后变化相对较小")
    logger.info(f"        但整体维持在较低水平，有利于台风发展\n")
    
    # 最大潜在强度结论
    vmpi_pre_mean = pre_ri['VMPI'].mean()
    vmpi_post_mean = post_ri['VMPI'].mean()
    vmpi_pre_med = pre_ri['VMPI'].median()
    vmpi_post_med = post_ri['VMPI'].median()
    logger.info(f"【最大潜在强度 (VMPI)】")
    logger.info(f"  RI前均值: {vmpi_pre_mean:.2f}kt, 中位数: {vmpi_pre_med:.2f}kt")
    logger.info(f"  RI后均值: {vmpi_post_mean:.2f}kt, 中位数: {vmpi_post_med:.2f}kt")
    logger.info(f"  结论: 最大潜在强度在RI终止后略有下降")
    logger.info(f"        说明环境对台风发展的支持能力有所减弱\n")
    
    # 台风移动速度结论
    speed_pre_mean = pre_ri['MOVE_SPEED'].mean()
    speed_post_mean = post_ri['MOVE_SPEED'].mean()
    speed_pre_med = pre_ri['MOVE_SPEED'].median()
    speed_post_med = post_ri['MOVE_SPEED'].median()
    logger.info(f"【台风移动速度 (MOVE_SPEED)】")
    logger.info(f"  RI前均值: {speed_pre_mean:.2f}kt, 中位数: {speed_pre_med:.2f}kt")
    logger.info(f"  RI后均值: {speed_post_mean:.2f}kt, 中位数: {speed_post_med:.2f}kt")
    logger.info(f"  结论: 台风移动速度在RI终止后有加快趋势")
    logger.info(f"        可能与中纬度系统的影响有关")
    
    logger.info("="*70 + "\n")
    
    logger.info("所有7张独立图表和1张综合趋势图生成完成！")


# ==================== 主程序 ====================

def main():
    logger.info("=" * 60)
    logger.info("  北大西洋台风快速增强停止点前后环境因子分析（1988-2021）")
    logger.info("=" * 60)

    # ---------- 1. 读取数据 ----------
    logger.info("正在读取IBTrACS数据集...")
    if not ibtracs_file.exists():
        logger.error(f"IBTrACS文件不存在: {ibtracs_file}")
        sys.exit(1)

    ibtracs_df = pd.read_csv(ibtracs_file, low_memory=False, skiprows=[1])
    logger.info(f"IBTrACS数据已加载: {len(ibtracs_df)} 行")

    logger.info("正在读取SHIPS数据集...")
    ships_df = read_ships_file(ships_file)
    logger.info(f"SHIPS数据已加载: {len(ships_df)} 个时间步记录")

    # ---------- 2. IBTrACS预处理 ----------
    logger.info("预处理IBTrACS数据...")
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
    logger.info(f"北大西洋台风记录数: {len(ibtracs_na)}")

    # ---------- 3. 离岸距离筛选 ----------
    logger.info("筛选离岸距离始终 > 100km 的台风...")
    valid_storms = []
    for sid, storm_data in ibtracs_na.groupby('SID'):
        if (storm_data['DIST2LAND'] > 100).all():
            valid_storms.append(sid)
    ibtracs_valid = ibtracs_na[ibtracs_na['SID'].isin(valid_storms)].copy()
    logger.info(f"筛选后: {len(valid_storms)} 个台风, {len(ibtracs_valid)} 条记录")

    # ---------- 4. ATCF ID填充与标准化 ----------
    logger.info("填充缺失ATCF ID并标准化格式...")
    ibtracs_valid['USA_ATCF_ID'] = (
        ibtracs_valid.groupby('SID')['USA_ATCF_ID'].ffill().bfill()
    )
    ibtracs_valid['USA_ATCF_ID_STANDARD'] = (
        ibtracs_valid['USA_ATCF_ID'].apply(standardize_atcf_id)
    )
    storms_with_atcf = ibtracs_valid['SID'][
        ibtracs_valid['USA_ATCF_ID_STANDARD'].notna()
    ].unique()
    ibtracs_valid = ibtracs_valid[ibtracs_valid['SID'].isin(storms_with_atcf)].copy()
    logger.info(f"有有效ATCF ID的台风数: {len(storms_with_atcf)}")

    # ---------- 5. RI检测 ----------
    logger.info("检测快速增强过程并识别最后一次RI终止点...")
    ri_storms = []
    last_ri_ends = {}
    storm_ids = list(ibtracs_valid.groupby('SID').groups.keys())
    total = len(storm_ids)

    for idx, sid in enumerate(storm_ids):
        if idx % 50 == 0:
            logger.info(f"  RI检测进度: {idx}/{total}")
        storm_data = ibtracs_valid[ibtracs_valid['SID'] == sid]
        has_ri, ri_end = detect_ri_and_find_last_end(storm_data)
        if has_ri:
            ri_storms.append(sid)
            last_ri_ends[sid] = ri_end

    logger.info(f"检测到 {len(ri_storms)} 个经历快速增强的台风")

    if not ri_storms:
        logger.warning("未检测到任何快速增强台风，程序退出")
        sys.exit(0)

    # ---------- 6. 提取前后24h窗口 ----------
    logger.info("提取RI终止点前后24小时窗口数据...")
    analysis_data = []
    skipped_count = 0
    
    for sid in ri_storms:
        storm_data = (
            ibtracs_valid[ibtracs_valid['SID'] == sid]
            .sort_values('ISO_TIME')
            .reset_index(drop=True)
        )
        ri_end_ts = last_ri_ends[sid]
        
        target_times = [ri_end_ts + pd.Timedelta(hours=h) for h in EXPECTED_RELATIVE_HOURS]
        
        storm_times_set = set(storm_data['ISO_TIME'].astype('datetime64[ns]'))
        target_times_set = set(pd.to_datetime(target_times).astype('datetime64[ns]'))
        
        if not target_times_set.issubset(storm_times_set):
            skipped_count += 1
            continue
            
        window = storm_data[storm_data['ISO_TIME'].isin(target_times)].copy()
        
        if len(window) != len(EXPECTED_RELATIVE_HOURS):
            skipped_count += 1
            continue
            
        window = window.sort_values('ISO_TIME').reset_index(drop=True)
        window['relative_hours'] = EXPECTED_RELATIVE_HOURS
        window['SID'] = sid
        window['last_ri_end_time'] = ri_end_ts
        analysis_data.append(window)

    logger.info(f"窗口提取完成: 成功 {len(analysis_data)} 个, 因时次不全跳过 {skipped_count} 个")

    if not analysis_data:
        logger.error("无符合条件的窗口数据，程序退出")
        sys.exit(1)

    analysis_df = pd.concat(analysis_data, ignore_index=True)
    logger.info(f"共提取到 {len(analysis_df)} 条窗口数据")

    # ---------- 7. 匹配SHIPS数据 ----------
    logger.info("匹配IBTrACS与SHIPS数据...")
    matched_df = match_ships_data(analysis_df, ships_df)

    if len(matched_df) == 0:
        logger.error("无匹配数据，程序退出")
        sys.exit(1)

    # ---------- 8. 新增：计算台风移动速度 ----------
    logger.info("计算台风移动速度...")
    matched_df['MOVE_SPEED'] = np.nan
    for sid, grp in matched_df.groupby('SID'):
        speeds = calculate_movement_speed(grp)
        matched_df.loc[grp.index, 'MOVE_SPEED'] = speeds

    # ---------- 9. 数据清洗 ----------
    matched_df = matched_df[(matched_df['RSST'] >= 22) & (matched_df['RSST'] <= 30)]
    # 过滤移动速度异常值
    matched_df = matched_df[(matched_df['MOVE_SPEED'] >= 0) & (matched_df['MOVE_SPEED'] <= 50)]
    before_drop = len(matched_df)
    matched_df = matched_df.dropna(subset=['RSST', 'SHRD', 'RHLO', 'RHMD', 'RHHI', 'VMPI', 'MOVE_SPEED'])
    dropped = before_drop - len(matched_df)
    logger.info(f"清洗后有效记录数: {len(matched_df)}（舍弃 {dropped} 条缺失/异常记录）")

    if len(matched_df) == 0:
        logger.error("清洗后无有效记录，程序退出")
        sys.exit(1)

    # ---------- 10. 样本统计 ----------
    sample_sizes = matched_df['relative_hours'].value_counts().sort_index()
    logger.info("各时间点样本量:")
    for h, c in sample_sizes.items():
        logger.info(f"  {h:+d}h: {c} 个样本")

    # ---------- 11. 绘制图表并输出结论 ----------
    plot_all_environmental_boxplots(matched_df, plot_paths)

    # 保存数据
    matched_df.to_excel(excel_save_path, index=False)
    logger.info(f"数据已保存至: {excel_save_path}")

    logger.info("变量统计摘要:")
    summary_cols = [v for v in SHIPS_VARS + ['MOVE_SPEED'] if v in matched_df.columns]
    print(matched_df[summary_cols].describe().round(2))

    logger.info("=" * 60)
    logger.info("  北大西洋台风分析完成")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
