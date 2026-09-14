#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
对比脚本：插值补齐缺测环境数据（原方案构造规则）
结果保存到 add2 文件夹，不覆盖原有结果。
"""
import os
import sys
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# 路径
CODE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CODE_DIR)  # cv4
PIC_DIR = os.path.normpath(os.path.join(BASE_DIR, '..', '..', 'picture', 'version4', 'add2'))
os.makedirs(PIC_DIR, exist_ok=True)

MERGED_CSV = os.path.join(BASE_DIR, 'merged_full_data.csv')
ORIG_SAMPLES_CSV = os.path.join(BASE_DIR, 'analysis_samples.csv')

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
ABS_COLS = [v + '_abs' for v in VAR_LIST]
DELTA_COLS = [v + '_delta24' for v in VAR_LIST]

# 中文字体
_ZH_FONT_PATHS = [r'C:\Windows\Fonts\simhei.ttf', r'C:\Windows\Fonts\msyh.ttc']
_zh_font_name = 'SimHei'
for _fp in _ZH_FONT_PATHS:
    if os.path.exists(_fp):
        try:
            fm.fontManager.addfont(_fp)
            _zh_font_name = fm.FontProperties(fname=_fp).get_name()
            break
        except Exception:
            pass
plt.rcParams['font.sans-serif'] = [_zh_font_name, 'DejaVu Sans']
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False
sns.set_style('whitegrid')
plt.rcParams['font.sans-serif'] = [_zh_font_name, 'DejaVu Sans']
plt.rcParams['font.family'] = 'sans-serif'

def safe_savefig(out_path, **kwargs):
    if os.path.exists(out_path):
        try:
            os.remove(out_path)
        except Exception:
            pass
    plt.savefig(out_path, **kwargs)


def interpolate_missing(df, vars_list):
    """
    按每个台风事件独立进行线性插值，补齐delta24列缺测。
    _abs列已无缺测，直接对_delta24列插值。
    """
    df = df.copy()
    df['datetime'] = pd.to_datetime(df['datetime'])

    # 统计插值前缺测（delta24列）
    print('  插值前缺测统计（delta24列）:')
    total_cells = len(df) * len(vars_list)
    missing_before = 0
    for var in vars_list:
        col = var + '_delta24'
        n_miss = df[col].isna().sum()
        missing_before += n_miss
        print(f'    {col}: {n_miss} ({n_miss/len(df)*100:.1f}%)')
    print(f'    合计缺测: {missing_before}/{total_cells} ({missing_before/total_cells*100:.1f}%)')

    # 按SID分组插值delta24列
    def _interp_group(g):
        g = g.sort_values('datetime').reset_index(drop=True)
        for var in vars_list:
            col = var + '_delta24'
            # 线性插值（边界用最近值填充）
            g[col] = g[col].interpolate(method='linear', limit_direction='both')
        return g

    group_col = 'SID'
    print(f'  按 {group_col} 分组对delta24列进行线性插值...')

    df_interp = df.groupby(group_col, group_keys=False).apply(_interp_group)
    df_interp = df_interp.sort_values(['datetime']).reset_index(drop=True)

    # 统计插值后缺测
    print('  插值后缺测统计（delta24列）:')
    missing_after = 0
    for var in vars_list:
        col = var + '_delta24'
        n_miss = df_interp[col].isna().sum()
        missing_after += n_miss
        print(f'    {col}: {n_miss} ({n_miss/len(df_interp)*100:.1f}%)')
    print(f'    合计缺测: {missing_after}/{total_cells} ({missing_after/total_cells*100:.1f}%)')
    print(f'    补齐了 {missing_before - missing_after} 个缺测值')

    return df_interp


def construct_samples_original(df, vars_list, valid_events):
    """原方案：≥3个RI时刻排除最后2个，2个RI时刻跳过"""
    enhance_rows = []
    stop_rows = []
    n_used = 0
    n_skipped_2pt = 0

    for eid in valid_events:
        group = df[df['ri_event_id'] == eid].sort_values('datetime')
        ri_pts = group[group['is_ri']].sort_values('datetime')

        if len(ri_pts) >= 3:
            # ≥3个RI时刻：排除最后2个，取更早的持续点平均
            enhance_pts = ri_pts.iloc[:-2]
            n_used += 1
        elif len(ri_pts) == 2:
            # 2个RI时刻：跳过（原方案）
            n_skipped_2pt += 1
            continue
        else:
            continue

        # 增强组
        row = {'event_id': eid, 'group': 'enhance'}
        for var in vars_list:
            row[var + '_abs'] = enhance_pts[var + '_abs'].mean()
            row[var + '_delta24'] = enhance_pts[var + '_delta24'].mean()
        enhance_rows.append(row)

        # 停止组：终止点+前6h平均
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
            row[var + '_abs'] = stop_pts[var + '_abs'].mean()
            row[var + '_delta24'] = stop_pts[var + '_delta24'].mean()
        stop_rows.append(row)

    samples = pd.DataFrame(enhance_rows + stop_rows)
    print(f'  原方案构造：使用事件={n_used}个, 跳过2点事件={n_skipped_2pt}个')
    return samples


def plot_boxplot(samples, vars_list, pic_dir):
    n_enhance = len(samples[samples['group'] == 'enhance'])
    n_stop = len(samples[samples['group'] == 'stop'])
    fig, axes = plt.subplots(2, 4, figsize=(22, 12))
    axes = axes.flatten()

    for idx, var in enumerate(vars_list):
        ax = axes[idx]
        ax2 = ax.twinx()
        delta_data = [
            samples[samples['group'] == 'enhance'][var + '_delta24'].dropna().values,
            samples[samples['group'] == 'stop'][var + '_delta24'].dropna().values,
        ]
        bp1 = ax.boxplot(delta_data, positions=[1, 2], widths=0.35,
                          patch_artist=True, showfliers=False, medianprops={'color': 'black'})
        for patch in bp1['boxes']:
            patch.set_facecolor('#4C72B0')
            patch.set_alpha(0.8)

        abs_data = [
            samples[samples['group'] == 'enhance'][var + '_abs'].dropna().values,
            samples[samples['group'] == 'stop'][var + '_abs'].dropna().values,
        ]
        bp2 = ax2.boxplot(abs_data, positions=[3.8, 4.8], widths=0.35,
                           patch_artist=True, showfliers=False, medianprops={'color': 'black'})
        for patch in bp2['boxes']:
            patch.set_facecolor('#DD8452')
            patch.set_alpha(0.8)

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

        e_d = samples[samples['group'] == 'enhance'][var + '_delta24'].dropna()
        s_d = samples[samples['group'] == 'stop'][var + '_delta24'].dropna()
        if len(e_d) > 1 and len(s_d) > 1:
            _, p = stats.mannwhitneyu(e_d, s_d, alternative='two-sided')
            sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
            ax.text(0.5, 0.97, f'Δ24h检验: p={p:.4f} {sig}', transform=ax.transAxes,
                    ha='center', fontsize=8.5, color='darkred', fontweight='bold')
        ax.set_title(ENV_VARS[var][0], fontsize=13, fontweight='bold')

    axes[7].set_visible(False)
    plt.suptitle(f'图2  增强组与停止组环境变量分组箱线图（插值补齐，增强组N={n_enhance}，停止组N={n_stop}）',
                 fontsize=14, y=1.02)
    plt.tight_layout()
    out = os.path.join(pic_dir, 'fig2_boxplot.png')
    safe_savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'    已保存: {out}')


def plot_feature_importance(samples, vars_list, pic_dir):
    delta_cols = [v + '_delta24' for v in vars_list]
    abs_cols = [v + '_abs' for v in vars_list]
    feature_cols = delta_cols + abs_cols
    feature_names = [f'Δ{ENV_VARS[v][0]}' for v in vars_list] + \
                    [f'{ENV_VARS[v][0]}(绝对值)' for v in vars_list]

    df_ml = samples.dropna(subset=feature_cols + ['group']).copy()
    df_ml = df_ml.sort_values(['event_id', 'group']).reset_index(drop=True)
    X = df_ml[feature_cols].values
    y = (df_ml['group'] == 'stop').astype(int).values
    groups = df_ml['event_id'].values
    unique_groups = np.unique(groups)
    n_splits = min(5, len(unique_groups))

    print(f'    样本: {len(df_ml)}, 特征: {len(feature_cols)}, RI事件: {len(unique_groups)}')
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
    std_imp = np.std(all_importances, axis=0)
    mean_auc = np.mean(all_aucs) if all_aucs else np.nan
    print(f'    交叉验证平均AUC = {mean_auc:.4f}（{len(all_importances)}次训练平均）')

    order = np.argsort(mean_imp)
    s_names = [feature_names[i] for i in order]
    s_imp = mean_imp[order]
    s_std = std_imp[order]
    s_type = ['变化量' if i < len(vars_list) else '绝对值' for i in order]

    fig, ax = plt.subplots(figsize=(13, 9))
    colors = ['#4C72B0' if t == '变化量' else '#DD8452' for t in s_type]
    y_pos = np.arange(len(s_names))
    ax.barh(y_pos, s_imp, xerr=s_std, color=colors, alpha=0.85,
            error_kw={'linewidth': 0.8, 'capsize': 3, 'ecolor': 'gray'})
    ax.set_yticks(y_pos)
    ax.set_yticklabels(s_names, fontsize=11)
    ax.set_xlabel('特征重要性得分（归一化）', fontsize=12)
    ax.set_title(f'图3  随机森林特征重要性排序（插值补齐，{n_splits}折×3种子，平均AUC={mean_auc:.3f}）',
                 fontsize=15, fontweight='bold')
    from matplotlib.patches import Patch
    legend = [Patch(facecolor='#4C72B0', label='24h变化量类'),
              Patch(facecolor='#DD8452', label='瞬时绝对值类')]
    ax.legend(handles=legend, loc='lower right', fontsize=11)
    ax.set_xlim(0, max(s_imp + s_std) * 1.15)
    plt.tight_layout()
    out = os.path.join(pic_dir, 'fig3_feature_importance.png')
    safe_savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'    已保存: {out}')
    return mean_imp, feature_cols, mean_auc


def plot_composite_trend(samples, vars_list, feature_imp, feature_cols, pic_dir):
    delta_cols = [v + '_delta24' for v in vars_list]
    delta_idx = [feature_cols.index(c) for c in delta_cols]
    weights = feature_imp[delta_idx]
    weights = weights / weights.sum()

    df = samples.dropna(subset=delta_cols).copy()
    print(f'    图4 dropna后有效样本: {len(df)} (增强组={(df["group"]=="enhance").sum()}, 停止组={(df["group"]=="stop").sum()})')

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
    t1, t2 = df['composite_score'].quantile([1/3, 2/3]).values
    df['trend_cat'] = pd.cut(df['composite_score'],
                              bins=[-np.inf, t1, t2, np.inf],
                              labels=['变差', '稳定', '向好'])

    ct = pd.crosstab(df['trend_cat'], df['group'])
    for c in ['enhance', 'stop']:
        if c not in ct.columns:
            ct[c] = 0
    ct = ct[['enhance', 'stop']]
    ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100

    chi2, p, dof, _ = stats.chi2_contingency(ct)
    if ct.loc['向好', 'enhance'] > 0 and ct.loc['变差', 'stop'] > 0:
        or_val = (ct.loc['变差', 'stop'] / ct.loc['变差', 'enhance']) / \
                 (ct.loc['向好', 'stop'] / ct.loc['向好', 'enhance'])
    else:
        or_val = np.nan

    fig, ax = plt.subplots(figsize=(11, 7.5))
    x = np.arange(len(ct_pct.index))
    w = 0.6
    b1 = ax.bar(x, ct_pct['enhance'], w, label='RI持续', color='#4C72B0', alpha=0.85)
    b2 = ax.bar(x, ct_pct['stop'], w, bottom=ct_pct['enhance'],
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
    ax.set_ylabel('组内百分比（%）', fontsize=13)
    ax.set_ylim(-10, 110)
    ax.set_title(f'图4  综合环境趋势与RI持续性关联（插值补齐）\n（卡方检验 卡方值={chi2:.2f}, p={p:.4f}, 优势比OR={or_val:.2f}）',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    out = os.path.join(pic_dir, 'fig4_composite_trend.png')
    safe_savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'    已保存: {out}')
    return chi2, p, or_val


def main():
    print('=' * 65)
    print('  插值补齐对比：线性插值补齐缺测环境数据（原方案构造规则）')
    print('=' * 65)

    # 读取全量数据
    print('\n[1/4] 读取全量数据...')
    merged = pd.read_csv(MERGED_CSV, encoding='utf-8-sig')
    merged['datetime'] = pd.to_datetime(merged['datetime'])
    merged['is_ri'] = merged['is_ri'].astype(bool)
    print(f'  全量数据: {len(merged)} 条')

    # 插值补齐
    print('\n[2/4] 线性插值补齐缺测...')
    merged_interp = interpolate_missing(merged, VAR_LIST)

    # 保存插值后的数据
    interp_path = os.path.join(CODE_DIR, 'merged_full_data_interp.csv')
    merged_interp.to_csv(interp_path, index=False, encoding='utf-8-sig')
    print(f'  插值后数据已保存: {interp_path}')

    # 构造样本（原方案）
    print('\n[3/4] 原方案构造样本...')
    valid_events = sorted(merged_interp[merged_interp['ri_event_id'] > 0]['ri_event_id'].unique().tolist())
    print(f'  有效RI事件: {len(valid_events)} 个')

    samples_interp = construct_samples_original(merged_interp, VAR_LIST, valid_events)
    n_enh_interp = (samples_interp['group'] == 'enhance').sum()
    n_stop_interp = (samples_interp['group'] == 'stop').sum()
    print(f'  插值后分析样本: {len(samples_interp)} (增强组={n_enh_interp}, 停止组={n_stop_interp})')

    # dropna统计
    feature_cols = DELTA_COLS + ABS_COLS
    df_dropna = samples_interp.dropna(subset=feature_cols).copy()
    print(f'  图3/图4 dropna后: {len(df_dropna)} (增强组={(df_dropna["group"]=="enhance").sum()}, 停止组={(df_dropna["group"]=="stop").sum()})')

    # 保存样本
    samples_path = os.path.join(CODE_DIR, 'analysis_samples_interp.csv')
    samples_interp.to_csv(samples_path, index=False, encoding='utf-8-sig')
    print(f'  样本已保存: {samples_path}')

    # 读取原方案样本用于对比
    samples_orig = pd.read_csv(ORIG_SAMPLES_CSV, encoding='utf-8-sig')
    n_enh_orig = (samples_orig['group'] == 'enhance').sum()
    n_stop_orig = (samples_orig['group'] == 'stop').sum()
    df_orig_dropna = samples_orig.dropna(subset=feature_cols).copy()

    # 绘图
    print('\n[4/4] 绘图...')
    print('\n>>> 图2: 分组箱线图')
    plot_boxplot(samples_interp, VAR_LIST, PIC_DIR)

    print('\n>>> 图3: 随机森林特征重要性')
    feature_imp, feature_cols_out, auc_interp = plot_feature_importance(samples_interp, VAR_LIST, PIC_DIR)

    print('\n>>> 图4: 综合环境趋势堆叠图')
    chi2_interp, p_interp, or_interp = plot_composite_trend(samples_interp, VAR_LIST, feature_imp, feature_cols_out, PIC_DIR)

    # 对比总结
    print('\n' + '=' * 65)
    print('  两种方案对比总结')
    print('=' * 65)
    print(f'{"指标":<20} {"原方案(不插值)":<20} {"插值补齐":<20}')
    print('-' * 60)
    print(f'{"增强组样本量":<20} {n_enh_orig:<20} {n_enh_interp:<20}')
    print(f'{"停止组样本量":<20} {n_stop_orig:<20} {n_stop_interp:<20}')
    print(f'{"总样本量":<20} {len(samples_orig):<20} {len(samples_interp):<20}')
    print(f'{"dropna后样本量":<20} {len(df_orig_dropna):<20} {len(df_dropna):<20}')
    print(f'{"图3 AUC":<20} {"0.6274":<20} {auc_interp:<20.4f}')
    print(f'{"图4 卡方值":<20} {"7.67":<20} {chi2_interp:<20.2f}')
    print(f'{"图4 p值":<20} {"0.0216":<20} {p_interp:<20.4f}')
    print(f'{"图4 OR值":<20} {"3.76":<20} {or_interp:<20.2f}')
    print('=' * 65)
    print(f'\n图片保存目录: {PIC_DIR}')
    print('=' * 65)


if __name__ == '__main__':
    main()
