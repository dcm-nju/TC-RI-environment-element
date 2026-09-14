import os
import re
import sys
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use('Agg')  # 非交互式后端，避免多线程下tkinter清理报错
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

#  配置
BASE_DIR = r'E:\中尺度科研\python\envioronment'
DATA_DIR = os.path.join(BASE_DIR, 'data')
CODE_DIR = os.path.join(BASE_DIR, 'code', 'cv4', 'add3')
PIC_DIR  = os.path.join(BASE_DIR, 'picture', 'version4', 'add3')

SHIPS_FILE   = os.path.join(DATA_DIR, 'ships',   'lsdiaga_1982_2023_sat_ts_7day.txt')
IBTRACS_FILE = os.path.join(DATA_DIR, 'ibtracs', 'ibtracs.ALL.list.v04r01.csv')

# 七个环境变量: SHIPS变量名 -> (中文显示名, 缩放因子, 向好方向)
# 向好方向: 'up'=升高有利, 'down'=降低有利(风切类)
ENV_VARS = {
    'VMPI': ('最大潜在强度',     1.0, 'up'),
    'XOHC': ('海洋热含量',       1.0, 'up'),
    'XDST': ('海温距平',         0.1, 'up'),
    'SHRD': ('深层风切变',       0.1, 'down'),
    'SHGC': ('风暴相对风切变',   0.1, 'down'),
    'RHLO': ('低层相对湿度',     0.1, 'up'),
    'EPOS': ('相当位温',         0.1, 'up'),
}
VAR_LIST = list(ENV_VARS.keys())

# 筛选参数
YEAR_START   = 1982
YEAR_END     = 2023
VMAX_MIN     = 34    # kt, 热带风暴及以上
DTL_MIN      = 100   # km, 距岸距离
RI_THRESHOLD = 30    # kt/24h, RI判定阈值

# 缺测值
MISSING = 9999

# 创建输出目录
os.makedirs(CODE_DIR, exist_ok=True)
os.makedirs(PIC_DIR,  exist_ok=True)

# 全局随机种子，确保可复现
np.random.seed(42)

def safe_savefig(out_path, **kwargs):
    """保存图片前先删除旧文件，确保新图一定能覆盖旧图（避免文件锁定导致残留）"""
    if os.path.exists(out_path):
        try:
            os.remove(out_path)
        except Exception:
            pass
    plt.savefig(out_path, **kwargs)

# 中文字体（显式注册字体文件 + rcParams全局设置）
import matplotlib.font_manager as _fm
_ZH_FONT_PATHS = [
    r'C:\Windows\Fonts\simhei.ttf',
    r'C:\Windows\Fonts\msyh.ttc',
]
_zh_font_name = 'SimHei'
for _fp in _ZH_FONT_PATHS:
    if os.path.exists(_fp):
        try:
            _fm.fontManager.addfont(_fp)
            _zh_font_name = _fm.FontProperties(fname=_fp).get_name()
            break
        except Exception:
            pass
plt.rcParams['font.sans-serif'] = [_zh_font_name, 'DejaVu Sans']
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False
sns.set_style('whitegrid')
plt.rcParams['font.sans-serif'] = [_zh_font_name, 'DejaVu Sans']
plt.rcParams['font.family'] = 'sans-serif'


#  模块1: SHIPS 解析
def parse_ships(filepath, vars_to_extract):
    """
    逐行解析 SHIPS 固定宽度文本文件，提取每个记录 0h 分析场的指定变量。

    每个台风记录块:
      HEAD 行  -> 台风ID / 日期 / 时间
      TIME 行  -> 时间偏移 (-12,-6,0,6,...,168h)，0h 为第3个值(索引2)
      变量行   -> 固定宽度数值 + 行末变量名
    """
    records = []
    current = None
    head_re = re.compile(r'^\s*[A-Z]{4}\s+\d{6}\s+\d{2}')
    sid_re  = re.compile(r'(AL\d{6})')
    var_re  = re.compile(r'\s+([A-Z]{3,5})(?:\s+\d+)?\s*$')

    print(f'  读取文件: {filepath}')
    line_count = 0
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line_count += 1
            if line_count % 500000 == 0:
                print(f'    已处理 {line_count} 行, 当前记录数 {len(records)}')

            # ---- HEAD 行: 开启新记录 ----
            if head_re.match(line):
                if current is not None:
                    records.append(current)

                parts = line.split()
                # 台风ID
                m_sid = sid_re.search(line)
                storm_id = m_sid.group(1) if m_sid else (parts[-2] if len(parts) >= 2 else 'UNKNOWN')
                # 日期时间: YYMMDD HH
                date_str = parts[1]
                hour_str = parts[2]
                yy = int(date_str[:2])
                year = 1900 + yy if yy >= 80 else 2000 + yy
                month = int(date_str[2:4])
                day   = int(date_str[4:6])
                hour  = int(hour_str)
                try:
                    dt = pd.Timestamp(year=year, month=month, day=day, hour=hour)
                except ValueError:
                    current = None
                    continue

                current = {'storm_id': storm_id, 'datetime': dt}
                continue

            if current is None:
                continue

            # ---- 变量行 ----
            m_var = var_re.search(line)
            if not m_var:
                continue
            var_name = m_var.group(1)
            if var_name not in vars_to_extract:
                continue

            nums = re.findall(r'-?\d+', line)
            if len(nums) >= 3:
                val = int(nums[2])  # 索引2 = 0h 列
                if val == MISSING:
                    val = np.nan
                current[var_name] = val

    if current is not None:
        records.append(current)

    df = pd.DataFrame(records)
    print(f'  解析完成: {len(df)} 条记录, {df["storm_id"].nunique()} 个台风')
    return df


# ============================================================
#  模块1: IBTrACS 读取
# ============================================================
def load_ibtracs(filepath):
    """读取 IBTrACS CSV，筛选北大西洋 1990-2023、热带风暴及以上、6小时间隔。
    注意: 本文件 SID 列格式异常，改用 USA_ATCF_ID 作为台风ID（AL开头=北大西洋）。
    """
    cols = ['USA_ATCF_ID', 'SEASON', 'ISO_TIME', 'LAT', 'LON',
            'USA_WIND', 'USA_PRES', 'DIST2LAND']
    print(f'  读取文件: {filepath}')
    # skiprows=[1] 跳过第二行单位行; on_bad_lines跳过字段数异常行
    df = pd.read_csv(filepath, usecols=cols,
                     dtype={'USA_WIND': str, 'SEASON': str, 'USA_ATCF_ID': str},
                     skiprows=[1], on_bad_lines='skip', low_memory=False)

    # 北大西洋: USA_ATCF_ID 以 AL 开头
    df = df[df['USA_ATCF_ID'].str.startswith('AL', na=False)].copy()
    df.rename(columns={'USA_ATCF_ID': 'SID'}, inplace=True)
    print(f'  北大西洋筛选后: {len(df)} 条记录, {df["SID"].nunique()} 个台风')
    # 年份
    df['SEASON'] = pd.to_numeric(df['SEASON'], errors='coerce')
    df = df[(df['SEASON'] >= YEAR_START) & (df['SEASON'] <= YEAR_END)]
    print(f'  {YEAR_START}-{YEAR_END}年筛选后: {len(df)} 条记录, {df["SID"].nunique()} 个台风')
    # 时间
    df['datetime'] = pd.to_datetime(df['ISO_TIME'], errors='coerce')
    df = df.dropna(subset=['datetime'])
    # 只保留 00/06/12/18 时 (6小时间隔)
    df = df[df['datetime'].dt.hour.isin([0, 6, 12, 18])]
    print(f'  仅保留00/06/12/18时后: {len(df)} 条记录')
    # 风速
    df['USA_WIND'] = pd.to_numeric(df['USA_WIND'], errors='coerce')
    df = df.dropna(subset=['USA_WIND'])
    df = df[df['USA_WIND'] >= VMAX_MIN]
    print(f'  Vmax≥{VMAX_MIN}kt筛选后: {len(df)} 条记录, {df["SID"].nunique()} 个台风')

    df = df.sort_values(['SID', 'datetime']).reset_index(drop=True)
    print(f'  最终IBTrACS: {len(df)} 条记录, {df["SID"].nunique()} 个台风')
    return df



#  模块1: 数据匹配
def match_data(ibtracs_df, ships_df):
    """以 台风ID + datetime 为键内连接 IBTrACS 与 SHIPS。"""
    merged = pd.merge(
        ibtracs_df, ships_df,
        left_on=['SID', 'datetime'],
        right_on=['storm_id', 'datetime'],
        how='inner'
    )
    merged = merged.sort_values(['SID', 'datetime']).reset_index(drop=True)
    print(f'  匹配后: {len(merged)} 条记录')
    return merged


#  模块2: 缺测插值
def interpolate_missing(df, vars_list):
    """
    对每个台风逐变量做线性插值。
    单帧缺失(前后6h有值) -> 插值补全; 连续缺测>=12h -> 保持NaN。
    同时标记哪些值是插值得到的。
    """
    df = df.sort_values(['SID', 'datetime']).copy()
    for var in vars_list:
        df[var + '_was_interp'] = df[var].isna()
        df[var] = df.groupby('SID')[var].transform(
            lambda x: x.interpolate(method='linear', limit=1)
        )
        df[var + '_was_interp'] = df[var + '_was_interp'] & df[var].notna()
    return df


#  模块3: RI事件识别与节点定义
def identify_ri_events(df):
    """
    1. 计算未来24h风速增幅 ΔV24 = V(t+24h) - V(t)
    2. ΔV24 >= 30kt -> RI时刻
    3. 合并连续RI时刻为独立事件(允许1个时次间断, 连续2个时次非RI则事件结束)
    4. 事件内: 除最后一个RI时刻外=持续点, 最后一个=终止点
    5. 仅1个RI时刻的超短事件 -> 剔除
    """
    df = df.sort_values(['SID', 'datetime']).copy()

    # 未来24h增幅 (4个时次 × 6h = 24h)
    df['V_future24'] = df.groupby('SID')['USA_WIND'].shift(-4)
    df['deltaV24'] = df['V_future24'] - df['USA_WIND']
    df['is_ri'] = df['deltaV24'] >= RI_THRESHOLD
    df['ri_event_id'] = 0  # 0=无事件, 正整数=事件ID

    event_counter = 0
    for sid, group in df.groupby('SID'):
        idx = group.index.tolist()
        is_ri = group['is_ri'].values.tolist()
        n = len(is_ri)
        i = 0
        while i < n:
            if is_ri[i]:
                event_counter += 1
                eid = event_counter
                j = i
                last_ri_pos = i
                while j < n:
                    if is_ri[j]:
                        last_ri_pos = j
                        df.loc[idx[j], 'ri_event_id'] = eid
                        j += 1
                    else:
                        # 连续2个非RI -> 事件结束; 仅1个非RI -> 跳过继续
                        if j + 1 < n and not is_ri[j + 1]:
                            break
                        j += 1
                i = last_ri_pos + 1
            else:
                i += 1

    # 节点类型
    df['node_type'] = 'non_ri'
    valid_events = []
    for eid, group in df.groupby('ri_event_id'):
        if eid == 0:
            continue
        ri_pts = group[group['is_ri']].sort_values('datetime')
        if len(ri_pts) >= 2:
            df.loc[ri_pts.index[:-1], 'node_type'] = 'continue'
            df.loc[ri_pts.index[-1],  'node_type'] = 'terminate'
            valid_events.append(eid)
        # len==1 的超短事件不标记(剔除)

    return df, valid_events


#  模块4: 环境变化量计算
def compute_env_changes(df, vars_list):
    """计算每个变量的瞬时绝对值和24h变化量。"""
    df = df.sort_values(['SID', 'datetime']).copy()
    for var in vars_list:
        df[var + '_abs'] = df[var]
        df[var + '_past24'] = df.groupby('SID')[var].shift(4)
        df[var + '_delta24'] = df[var] - df[var + '_past24']
    return df


#  模块3: 停止组/增强组样本构造
def construct_samples(df, vars_list, valid_events):
    """
    每个有效RI事件贡献:
      增强组(持续组, label=0): 除最后两个RI时刻外的其余RI时刻环境特征取平均
                              （排除终止点和最后一个持续点，保留更早的持续增强阶段）
      停止组(前兆组, label=1): 终止点及其前6h时刻的环境特征取平均
    """
    enhance_rows = []
    stop_rows = []

    for eid in valid_events:
        group = df[df['ri_event_id'] == eid].sort_values('datetime')
        ri_pts = group[group['is_ri']].sort_values('datetime')

        # ---- 增强组: 除最后两个RI时刻外的其余RI时刻平均 ----
        # ri_pts按时间排序，最后一个=终止点，倒数第二个=最后一个持续点
        if len(ri_pts) <= 2:
            continue  # RI时刻不足3个，无法构造增强组
        enhance_pts = ri_pts.iloc[:-2]
        row = {'event_id': eid, 'group': 'enhance'}
        for var in vars_list:
            row[var + '_abs']     = enhance_pts[var + '_abs'].mean()
            row[var + '_delta24'] = enhance_pts[var + '_delta24'].mean()
        enhance_rows.append(row)

        # ---- 停止组: 终止点 + 前6h 平均 ----
        term_pts = ri_pts[ri_pts['node_type'] == 'terminate']
        if len(term_pts) == 0:
            continue
        term_idx = term_pts.index[0]
        pos = group.index.get_loc(term_idx)
        if pos >= 1:
            stop_pts = group.iloc[pos - 1:pos + 1]
        else:
            stop_pts = term_pts

        row = {'event_id': eid, 'group': 'stop'}
        for var in vars_list:
            row[var + '_abs']     = stop_pts[var + '_abs'].mean()
            row[var + '_delta24'] = stop_pts[var + '_delta24'].mean()
        stop_rows.append(row)

    samples = pd.DataFrame(enhance_rows + stop_rows)
    return samples


#  图1: RI过程合成演变图
def plot_composite_evolution(df, valid_events, vars_list):
    """
    以持续点/终止点为0h，提取 -48h~+12h 环境演变，
    全体Z-score标准化后取中位数合成，2行×7列子图。
    """
    time_offsets = list(range(-48, 13, 6))  # -48,-42,...,0,...,12
    n_times = len(time_offsets)

    cont_data = {v: [] for v in vars_list}
    term_data = {v: [] for v in vars_list}

    df_sorted = df.sort_values(['SID', 'datetime'])

    print('    收集持续点演变序列...')
    for eid in valid_events:
        group = df_sorted[df_sorted['ri_event_id'] == eid]
        ri_pts = group[group['is_ri']].sort_values('datetime')
        sid = group['SID'].iloc[0]
        storm = df_sorted[df_sorted['SID'] == sid].set_index('datetime')

        # 持续点
        for _, row in ri_pts[ri_pts['node_type'] == 'continue'].iterrows():
            center = row['datetime']
            for var in vars_list:
                seq = []
                for off in time_offsets:
                    tt = center + pd.Timedelta(hours=off)
                    seq.append(storm.loc[tt, var] if tt in storm.index else np.nan)
                cont_data[var].append(seq)

        # 终止点
        term = ri_pts[ri_pts['node_type'] == 'terminate']
        if len(term) > 0:
            center = term.iloc[0]['datetime']
            for var in vars_list:
                seq = []
                for off in time_offsets:
                    tt = center + pd.Timedelta(hours=off)
                    seq.append(storm.loc[tt, var] if tt in storm.index else np.nan)
                term_data[var].append(seq)

    # 全体Z-score参数：取自全部筛选后的有效台风样本全集（1990-2023, Vmax≥34kt, DTL>100km），不去极值
    means, stds = {}, {}
    for var in vars_list:
        all_vals = df[var].dropna().values.astype(float)
        if len(all_vals) > 0:
            means[var] = np.mean(all_vals)
            stds[var]  = np.std(all_vals)
        else:
            means[var] = 0.0
            stds[var]  = 1.0

    # 绘图: a/b两张独立图，每张7条不同颜色线
    time_labels = [f'{t:+d}h' if t != 0 else '0h' for t in time_offsets]
    colors = plt.cm.tab10(np.linspace(0, 1, len(vars_list)))
    markers = ['o', 's', '^', 'D', 'v', 'p', '*']

    for tag, data, suffix in [('(a) RI持续点', cont_data, 'continue'),
                                ('(b) RI终止点', term_data, 'terminate')]:
        fig, ax = plt.subplots(figsize=(11, 7))
        n_total = len(data[vars_list[0]])  # 该类型点的总数量

        # 计算每个时刻的有效样本量，标注在x轴标签中
        arr_ref = np.array(data[vars_list[0]], dtype=float)
        n_by_time = [int(np.sum(~np.isnan(arr_ref[:, i]))) for i in range(len(time_offsets))]
        xtick_labels = [f'{time_labels[i]}\nn={n_by_time[i]}' for i in range(len(time_offsets))]

        for i, var in enumerate(vars_list):
            arr = np.array(data[var], dtype=float)
            arr_z = (arr - means[var]) / (stds[var] if stds[var] > 0 else 1)
            med = np.nanmedian(arr_z, axis=0)
            ax.plot(time_offsets, med, color=colors[i], linewidth=2.2,
                    marker=markers[i], markersize=6, label=ENV_VARS[var][0])

        ax.axvline(x=0, color='r', linestyle='--', alpha=0.6, linewidth=1.2)
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax.set_title(f'图1 {tag}  环境场合成演变（总样本N={n_total}）',
                     fontsize=14, fontweight='bold')
        ax.set_xlabel('相对于节点时间（小时）', fontsize=12)
        ax.set_ylabel('Z-score', fontsize=12)
        ax.set_xticks(time_offsets)
        ax.set_xticklabels(xtick_labels, rotation=45, fontsize=9)
        ax.legend(loc='best', fontsize=9, ncol=2, framealpha=0.9)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        out = os.path.join(PIC_DIR, f'fig1_{suffix}.png')
        safe_savefig(out, dpi=300, bbox_inches='tight')
        plt.close()
        print(f'    已保存: {out}')


#  图2: 分组箱线图
def plot_boxplot(samples, vars_list):
    """图2: 7个子图，每个双y轴——左y轴Δ24h变化量(蓝)，右y轴绝对值(橙)"""
    n_enhance = len(samples[samples['group'] == 'enhance'])
    n_stop = len(samples[samples['group'] == 'stop'])
    fig, axes = plt.subplots(2, 4, figsize=(22, 12))
    axes = axes.flatten()

    for idx, var in enumerate(vars_list):
        ax = axes[idx]
        ax2 = ax.twinx()

        # 左y轴: Δ24h变化量 (持续、终止)
        delta_data = [
            samples[samples['group'] == 'enhance'][var + '_delta24'].dropna().values,
            samples[samples['group'] == 'stop'][var + '_delta24'].dropna().values,
        ]
        bp1 = ax.boxplot(delta_data, positions=[1, 2], widths=0.35,
                          patch_artist=True, showfliers=False, medianprops={'color': 'black'})
        for patch in bp1['boxes']:
            patch.set_facecolor('#4C72B0')
            patch.set_alpha(0.8)

        # 右y轴: 绝对值 (持续、终止)
        abs_data = [
            samples[samples['group'] == 'enhance'][var + '_abs'].dropna().values,
            samples[samples['group'] == 'stop'][var + '_abs'].dropna().values,
        ]
        bp2 = ax2.boxplot(abs_data, positions=[3.8, 4.8], widths=0.35,
                           patch_artist=True, showfliers=False, medianprops={'color': 'black'})
        for patch in bp2['boxes']:
            patch.set_facecolor('#DD8452')
            patch.set_alpha(0.8)

        # x轴
        ax.set_xlim(0.3, 5.5)
        ax.set_xticks([1.5, 4.3])
        ax.set_xticklabels(['Δ24h变化量', '瞬时绝对值'], fontsize=10)
        ax.text(1, -0.12, '持续', transform=ax.get_xaxis_transform(), ha='center', fontsize=8, color='#4C72B0')
        ax.text(2, -0.12, '终止', transform=ax.get_xaxis_transform(), ha='center', fontsize=8, color='#4C72B0')
        ax.text(3.8, -0.12, '持续', transform=ax.get_xaxis_transform(), ha='center', fontsize=8, color='#DD8452')
        ax.text(4.8, -0.12, '终止', transform=ax.get_xaxis_transform(), ha='center', fontsize=8, color='#DD8452')

        ax.set_ylabel('Δ24h变化量', color='#4C72B0', fontsize=10)
        ax2.set_ylabel('瞬时绝对值', color='#DD8452', fontsize=10)
        ax.tick_params(axis='y', labelcolor='#4C72B0')
        ax2.tick_params(axis='y', labelcolor='#DD8452')

        # 曼-惠特尼U检验 (变化量)
        e_d = samples[samples['group'] == 'enhance'][var + '_delta24'].dropna()
        s_d = samples[samples['group'] == 'stop'][var + '_delta24'].dropna()
        if len(e_d) > 1 and len(s_d) > 1:
            _, p_delta = stats.mannwhitneyu(e_d, s_d, alternative='two-sided')
            sig_delta = '***' if p_delta < 0.001 else '**' if p_delta < 0.01 else '*' if p_delta < 0.05 else 'ns'
            ax.text(0.5, 0.97, f'Δ24h检验: p={p_delta:.4f} {sig_delta}', transform=ax.transAxes,
                    ha='center', fontsize=8.5, color='#4C72B0', fontweight='bold')

        # 曼-惠特尼U检验 (绝对值)
        e_a = samples[samples['group'] == 'enhance'][var + '_abs'].dropna()
        s_a = samples[samples['group'] == 'stop'][var + '_abs'].dropna()
        if len(e_a) > 1 and len(s_a) > 1:
            _, p_abs = stats.mannwhitneyu(e_a, s_a, alternative='two-sided')
            sig_abs = '***' if p_abs < 0.001 else '**' if p_abs < 0.01 else '*' if p_abs < 0.05 else 'ns'
            ax.text(0.5, 0.92, f'绝对值检验: p={p_abs:.4f} {sig_abs}', transform=ax.transAxes,
                    ha='center', fontsize=8.5, color='#DD8452', fontweight='bold')

        ax.set_title(ENV_VARS[var][0], fontsize=13, fontweight='bold')

    axes[7].set_visible(False)
    plt.suptitle(f'图2  增强组与停止组环境变量分组箱线图（增强组N={n_enhance}，停止组N={n_stop}；双y轴：左=Δ24h变化量(蓝)，右=瞬时绝对值(橙)；曼-惠特尼U检验）',
                 fontsize=14, y=1.02)
    plt.tight_layout()
    out = os.path.join(PIC_DIR, 'fig2_boxplot.png')
    safe_savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'    已保存: {out}')


#  图3: 随机森林特征重要性
def plot_feature_importance(samples, vars_list):
    """
    随机森林特征重要性。为提高排名稳定性：
      1. 数据按event_id排序，确保GroupKFold划分确定
      2. n_estimators=1000（更多树，更稳定）
      3. 3个不同随机种子(42,123,456)各跑5折，全部取平均
    """
    delta_cols = [v + '_delta24' for v in vars_list]
    abs_cols   = [v + '_abs'     for v in vars_list]
    feature_cols = delta_cols + abs_cols
    feature_names = [f'Δ{ENV_VARS[v][0]}' for v in vars_list] + \
                    [f'{ENV_VARS[v][0]}(绝对值)' for v in vars_list]

    df_ml = samples.dropna(subset=feature_cols + ['group']).copy()
    # 关键：按event_id排序，确保GroupKFold划分每次都一致
    df_ml = df_ml.sort_values(['event_id', 'group']).reset_index(drop=True)
    X = df_ml[feature_cols].values
    y = (df_ml['group'] == 'stop').astype(int).values
    groups = df_ml['event_id'].values
    unique_groups = np.unique(groups)
    n_splits = min(5, len(unique_groups))

    print(f'    样本: {len(df_ml)}, 特征: {len(feature_cols)}, RI事件: {len(unique_groups)}')
    print(f'    {n_splits}折GroupKFold × 3种随机种子 × 1000棵树，取平均...')

    gkf = GroupKFold(n_splits=n_splits)
    all_importances, all_aucs = [], []
    seeds = [42, 123, 456]

    for seed in seeds:
        for fold, (tr_idx, te_idx) in enumerate(gkf.split(X, y, groups)):
            rf = RandomForestClassifier(
                n_estimators=1000, max_depth=None, min_samples_leaf=2,
                class_weight='balanced', random_state=seed + fold, n_jobs=-1
            )
            rf.fit(X[tr_idx], y[tr_idx])
            all_importances.append(rf.feature_importances_)
            y_prob = rf.predict_proba(X[te_idx])[:, 1]
            if len(np.unique(y[te_idx])) > 1:
                all_aucs.append(roc_auc_score(y[te_idx], y_prob))

    mean_imp = np.mean(all_importances, axis=0)
    std_imp  = np.std(all_importances, axis=0)
    mean_auc = np.mean(all_aucs) if all_aucs else np.nan
    print(f'    交叉验证平均AUC = {mean_auc:.4f}（{len(all_importances)}次训练平均）')

    # 排序
    order = np.argsort(mean_imp)
    s_names = [feature_names[i] for i in order]
    s_imp   = mean_imp[order]
    s_std   = std_imp[order]
    s_type  = ['变化量' if i < len(vars_list) else '绝对值' for i in order]

    fig, ax = plt.subplots(figsize=(13, 9))
    colors = ['#4C72B0' if t == '变化量' else '#DD8452' for t in s_type]
    y_pos = np.arange(len(s_names))
    ax.barh(y_pos, s_imp, xerr=s_std, color=colors, alpha=0.85,
            error_kw={'linewidth': 0.8, 'capsize': 3, 'ecolor': 'gray'})

    ax.set_yticks(y_pos)
    ax.set_yticklabels(s_names, fontsize=11)
    ax.set_xlabel('特征重要性得分（归一化）', fontsize=12)
    ax.set_title(f'图3  随机森林特征重要性排序（{n_splits}折×3种子，平均AUC={mean_auc:.3f}）',
                 fontsize=15, fontweight='bold')

    from matplotlib.patches import Patch
    legend = [Patch(facecolor='#4C72B0', label='24h变化量类'),
              Patch(facecolor='#DD8452', label='瞬时绝对值类')]
    ax.legend(handles=legend, loc='lower right', fontsize=11)
    ax.set_xlim(0, max(s_imp + s_std) * 1.15)

    plt.tight_layout()
    out = os.path.join(PIC_DIR, 'fig3_feature_importance.png')
    safe_savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'    已保存: {out}')
    return mean_imp, feature_cols


#  图4: 加权综合环境趋势堆叠柱状图
def plot_composite_trend(samples, vars_list, feature_imp, feature_cols):
    """
    用图3变化量类特征重要性作权重，7个Δ24h经Z-score+方向统一后
    加权合成「综合环境趋势指数」，三分位分为 变差/稳定/向好，
    百分比堆叠柱状图展示持续/终止占比，附卡方检验。
    Z-score基准μ、σ取自RI事件集(samples)，不去极值。
    """
    delta_cols = [v + '_delta24' for v in vars_list]
    delta_idx  = [feature_cols.index(c) for c in delta_cols]
    weights = feature_imp[delta_idx]
    weights = weights / weights.sum()

    df = samples.dropna(subset=delta_cols).copy()
    print(f'    图4 dropna后有效样本: {len(df)} (增强组={(df["group"]=="enhance").sum()}, 停止组={(df["group"]=="stop").sum()})')

    # Z-score + 方向统一 (风切类乘-1, 使 正值=环境向好)
    # 基准μ、σ：取自RI事件集(df)，不去极值
    z_mat = np.zeros((len(df), len(vars_list)))
    for i, var in enumerate(vars_list):
        col = var + '_delta24'
        vals = df[col].dropna().values
        mu = np.mean(vals) if len(vals) > 0 else 0.0
        sigma = np.std(vals) if len(vals) > 0 else 1.0
        z = (df[col].values - mu) / (sigma if sigma > 0 else 1)
        if ENV_VARS[var][2] == 'down':
            z = -z
        z_mat[:, i] = z

    df['composite_score'] = np.dot(z_mat, weights)

    # 三分位
    t1, t2 = df['composite_score'].quantile([1/3, 2/3]).values
    df['trend_cat'] = pd.cut(df['composite_score'],
                              bins=[-np.inf, t1, t2, np.inf],
                              labels=['变差', '稳定', '向好'])

    # 列联表
    ct = pd.crosstab(df['trend_cat'], df['group'])
    for c in ['enhance', 'stop']:
        if c not in ct.columns:
            ct[c] = 0
    ct = ct[['enhance', 'stop']]
    ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100

    # 卡方检验
    chi2, p, dof, _ = stats.chi2_contingency(ct)
    # 优势比 (向好 vs 变差 的终止odds比)
    if ct.loc['向好', 'enhance'] > 0 and ct.loc['变差', 'stop'] > 0:
        or_val = (ct.loc['变差', 'stop'] / ct.loc['变差', 'enhance']) / \
                 (ct.loc['向好', 'stop'] / ct.loc['向好', 'enhance'])
    else:
        or_val = np.nan

    # 绘图
    fig, ax = plt.subplots(figsize=(11, 7.5))
    x = np.arange(len(ct_pct.index))
    w = 0.6

    b1 = ax.bar(x, ct_pct['enhance'], w, label='RI持续', color='#4C72B0', alpha=0.85)
    b2 = ax.bar(x, ct_pct['stop'],    w, bottom=ct_pct['enhance'],
                label='RI终止', color='#DD8452', alpha=0.85)

    for i, cat in enumerate(ct_pct.index):
        n = ct.loc[cat].sum()
        ax.text(i, -6, f'N={int(n)}', ha='center', fontsize=12, fontweight='bold')

    for bars in [b1, b2]:
        for bar in bars:
            h = bar.get_height()
            if h > 4:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_y() + h / 2,
                        f'{h:.1f}%', ha='center', va='center',
                        fontsize=12, color='white', fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(ct_pct.index, fontsize=14)
    ax.set_ylabel('组内百分比 (%)', fontsize=13)
    ax.set_ylim(-12, 108)
    ax.set_title(f'图4  综合环境趋势与RI持续性关联\n'
                 f'（卡方检验 卡方值={chi2:.2f}, p={p:.4f}, 优势比OR={or_val:.2f}）',
                 fontsize=15, fontweight='bold')
    ax.legend(loc='upper right', fontsize=12)

    plt.tight_layout()
    out = os.path.join(PIC_DIR, 'fig4_composite_trend.png')
    safe_savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'    已保存: {out}')


def main():
    print('=' * 65)
    print('  台风快速增强持续与终止的环境场前兆研究 — 完整分析')
    print('=' * 65)

    all_ships_vars = VAR_LIST + ['DTL']  # 额外提取距岸距离

    # ---- 模块1: 数据读取与匹配 ----
    print('\n[1/5] 解析 SHIPS 数据...')
    ships_df = parse_ships(SHIPS_FILE, all_ships_vars)

    # 应用缩放因子
    for var in VAR_LIST:
        ships_df[var] = ships_df[var] * ENV_VARS[var][1]
    print('  变量缩放完成 (XOHC/VMPI=真实值, 其余÷10)')

    print('\n[2/5] 读取 IBTrACS 数据...')
    ibtracs_df = load_ibtracs(IBTRACS_FILE)

    print('\n[3/5] 时空匹配 + 距岸筛选...')
    merged = match_data(ibtracs_df, ships_df)
    merged = merged[merged['DTL'] > DTL_MIN].copy()
    print(f'  距岸>{DTL_MIN}km后: {len(merged)} 条记录')

    # ---- 模块2: 缺测插值 ----
    print('\n[4/5] 缺测处理与插值...')
    merged = interpolate_missing(merged, VAR_LIST)
    before = len(merged)
    merged = merged.dropna(subset=VAR_LIST, how='all')
    print(f'  剔除七变量全缺: {before} -> {len(merged)} 条')

    # ---- 模块3+4: RI识别 + 变化量 + 样本构造 ----
    print('\n[5/5] RI事件识别、节点定义与样本构造...')
    merged, valid_events = identify_ri_events(merged)
    n_cont = (merged['node_type'] == 'continue').sum()
    n_term = (merged['node_type'] == 'terminate').sum()
    n_ri_total = (merged['is_ri']).sum()
    print(f'  有效RI事件: {len(valid_events)} 个')
    print(f'  RI时刻总数: {n_ri_total} (持续点={n_cont}, 终止点={n_term})')

    merged = compute_env_changes(merged, VAR_LIST)
    samples = construct_samples(merged, VAR_LIST, valid_events)
    n_enh = (samples['group'] == 'enhance').sum()
    n_stp = (samples['group'] == 'stop').sum()
    print(f'  分析样本: {len(samples)} (增强组={n_enh}, 停止组={n_stp})')

    # 保存中间结果
    samples_path = os.path.join(CODE_DIR, 'analysis_samples_1982.csv')
    merged_path  = os.path.join(CODE_DIR, 'merged_full_data_1982.csv')
    samples.to_csv(samples_path, index=False, encoding='utf-8-sig')
    merged.to_csv(merged_path,   index=False, encoding='utf-8-sig')
    print(f'  样本已保存: {samples_path}')
    print(f'  全量数据已保存: {merged_path}')

    # ---- 模块5: 绘图 ----
    print('\n' + '=' * 65)
    print('  开始绘图...')
    print('=' * 65)

    print('\n>>> 图1: 合成演变图')
    plot_composite_evolution(merged, valid_events, VAR_LIST)

    print('\n>>> 图2: 分组箱线图')
    plot_boxplot(samples, VAR_LIST)

    print('\n>>> 图3: 随机森林特征重要性')
    feature_imp, feature_cols = plot_feature_importance(samples, VAR_LIST)

    print('\n>>> 图4: 综合环境趋势堆叠图')
    # 图4的Z-score基准μ、σ取自RI事件集(samples)，不去极值
    plot_composite_trend(samples, VAR_LIST, feature_imp, feature_cols)

    # 保存特征重要性
    imp_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': feature_imp
    }).sort_values('importance', ascending=False)
    imp_path = os.path.join(CODE_DIR, 'feature_importance_1982.csv')
    imp_df.to_csv(imp_path, index=False, encoding='utf-8-sig')

    print('\n' + '=' * 65)
    print('  全部分析完成！')
    print(f'  图片目录: {PIC_DIR}')
    print(f'  代码目录: {CODE_DIR}')
    print('=' * 65)


if __name__ == '__main__':
    main()
