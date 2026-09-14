#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
对比脚本：新方案（2个RI时刻事件：持续点=增强组，终止点=停止组）
结果保存到 add 文件夹，不覆盖原有结果。
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
PIC_DIR = os.path.normpath(os.path.join(BASE_DIR, '..', '..', 'picture', 'version4', 'add'))
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


def construct_samples_new(df, vars_list, valid_events):
    """
    新方案：
      - ≥3个RI时刻的事件：增强组=排除最后2个RI时刻后的平均（原规则）
      - =2个RI时刻的事件：增强组=持续点（单个点），停止组=终止点+前6h
    """
    enhance_rows = []
    stop_rows = []
    n_2pt_events = 0
    n_3pt_events = 0

    for eid in valid_events:
        group = df[df['ri_event_id'] == eid].sort_values('datetime')
        ri_pts = group[group['is_ri']].sort_values('datetime')

        if len(ri_pts) == 2:
            # 只有2个RI时刻：持续点作为增强组
            enhance_pts = ri_pts[ri_pts['node_type'] == 'continue']
            if len(enhance_pts) == 0:
                continue
            n_2pt_events += 1
        elif len(ri_pts) >= 3:
            # ≥3个RI时刻：排除最后2个，取更早的持续点平均
            enhance_pts = ri_pts.iloc[:-2]
            n_3pt_events += 1
        else:
            continue  # <2个RI时刻，跳过

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
    print(f'  新方案：2个RI时刻事件={n_2pt_events}个, ≥3个RI时刻事件={n_3pt_events}个')
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
    plt.suptitle(f'图2  增强组与停止组环境变量分组箱线图（新方案，增强组N={n_enhance}，停止组N={n_stop}）',
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
    ax.set_title(f'图3  随机森林特征重要性排序（新方案，{n_splits}折×3种子，平均AUC={mean_auc:.3f}）',
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
    ax.set_title(f'图4  综合环境趋势与RI持续性关联（新方案）\n（卡方检验 卡方值={chi2:.2f}, p={p:.4f}, 优势比OR={or_val:.2f}）',
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
    print('  新方案对比：2个RI时刻事件 持续点=增强组，终止点=停止组')
    print('=' * 65)

    # 读取全量数据
    print('\n[1/3] 读取全量数据...')
    merged = pd.read_csv(MERGED_CSV, encoding='utf-8-sig')
    merged['datetime'] = pd.to_datetime(merged['datetime'])
    merged['is_ri'] = merged['is_ri'].astype(bool)
    print(f'  全量数据: {len(merged)} 条')

    valid_events = sorted(merged[merged['ri_event_id'] > 0]['ri_event_id'].unique().tolist())
    print(f'  有效RI事件: {len(valid_events)} 个')

    # 统计RI时刻分布
    ri_counts = []
    for eid in valid_events:
        group = merged[merged['ri_event_id'] == eid]
        n_ri = group['is_ri'].sum()
        ri_counts.append(n_ri)
    print(f'  RI时刻分布: 2个时刻={ri_counts.count(2)}个, 3个时刻={ri_counts.count(3)}个, '
          f'4个时刻={ri_counts.count(4)}个, ≥5个时刻={sum(1 for c in ri_counts if c>=5)}个')

    # 新方案构造样本
    print('\n[2/3] 新方案构造样本...')
    samples_new = construct_samples_new(merged, VAR_LIST, valid_events)
    n_enh_new = (samples_new['group'] == 'enhance').sum()
    n_stop_new = (samples_new['group'] == 'stop').sum()
    print(f'  新方案分析样本: {len(samples_new)} (增强组={n_enh_new}, 停止组={n_stop_new})')

    # 保存新方案样本
    samples_path = os.path.join(CODE_DIR, 'analysis_samples_new.csv')
    samples_new.to_csv(samples_path, index=False, encoding='utf-8-sig')
    print(f'  样本已保存: {samples_path}')

    # 读取原方案样本用于对比
    samples_orig = pd.read_csv(ORIG_SAMPLES_CSV, encoding='utf-8-sig')
    n_enh_orig = (samples_orig['group'] == 'enhance').sum()
    n_stop_orig = (samples_orig['group'] == 'stop').sum()

    # 绘图
    print('\n[3/3] 绘图...')
    print('\n>>> 图2: 分组箱线图')
    plot_boxplot(samples_new, VAR_LIST, PIC_DIR)

    print('\n>>> 图3: 随机森林特征重要性')
    feature_imp, feature_cols, auc_new = plot_feature_importance(samples_new, VAR_LIST, PIC_DIR)

    print('\n>>> 图4: 综合环境趋势堆叠图')
    chi2_new, p_new, or_new = plot_composite_trend(samples_new, VAR_LIST, feature_imp, feature_cols, PIC_DIR)

    # 对比总结
    print('\n' + '=' * 65)
    print('  两种方案对比总结')
    print('=' * 65)
    print(f'{"指标":<20} {"原方案(79事件)":<20} {"新方案(111事件)":<20}')
    print('-' * 60)
    print(f'{"增强组样本量":<20} {n_enh_orig:<20} {n_enh_new:<20}')
    print(f'{"停止组样本量":<20} {n_stop_orig:<20} {n_stop_new:<20}')
    print(f'{"总样本量":<20} {len(samples_orig):<20} {len(samples_new):<20}')
    print(f'{"图3 AUC":<20} {"0.6274":<20} {auc_new:<20.4f}')
    print(f'{"图4 卡方值":<20} {"7.67":<20} {chi2_new:<20.2f}')
    print(f'{"图4 p值":<20} {"0.0216":<20} {p_new:<20.4f}')
    print(f'{"图4 OR值":<20} {"3.76":<20} {or_new:<20.2f}')
    print('=' * 65)
    print(f'\n图片保存目录: {PIC_DIR}')
    print('=' * 65)


if __name__ == '__main__':
    main()
