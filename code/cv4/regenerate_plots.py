import os
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use('Agg')  # 非交互式后端，避免多线程下tkinter清理报错
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

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

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
PIC_DIR  = os.path.join(os.path.dirname(CODE_DIR), '..', 'picture', 'version4')
PIC_DIR  = os.path.normpath(PIC_DIR)
os.makedirs(PIC_DIR, exist_ok=True)

MERGED_CSV  = os.path.join(CODE_DIR, 'merged_full_data.csv')
SAMPLES_CSV = os.path.join(CODE_DIR, 'analysis_samples.csv')

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

_ZH_FONT_PATHS = [
    r'C:\Windows\Fonts\simhei.ttf',
    r'C:\Windows\Fonts\msyh.ttc',
]
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
# sns.set_style 可能覆盖字体，重新设置一次
plt.rcParams['font.sans-serif'] = [_zh_font_name, 'DejaVu Sans']
plt.rcParams['font.family'] = 'sans-serif'

#  图1: 合成演变图
def plot_composite_evolution(df, valid_events, vars_list):
    """图1: a/b两张独立图，每张7条不同颜色线（中位数合成，无宽置信区间）"""
    time_offsets = list(range(-48, 13, 6))
    cont_data = {v: [] for v in vars_list}
    term_data = {v: [] for v in vars_list}
    df_sorted = df.sort_values(['SID', 'datetime'])

    print('    收集持续点演变序列...')
    for eid in valid_events:
        group = df_sorted[df_sorted['ri_event_id'] == eid]
        ri_pts = group[group['is_ri']].sort_values('datetime')
        sid = group['SID'].iloc[0]
        storm = df_sorted[df_sorted['SID'] == sid].set_index('datetime')

        for _, row in ri_pts[ri_pts['node_type'] == 'continue'].iterrows():
            center = row['datetime']
            for var in vars_list:
                seq = []
                for off in time_offsets:
                    tt = center + pd.Timedelta(hours=off)
                    seq.append(storm.loc[tt, var] if tt in storm.index else np.nan)
                cont_data[var].append(seq)

        term = ri_pts[ri_pts['node_type'] == 'terminate']
        if len(term) > 0:
            center = term.iloc[0]['datetime']
            for var in vars_list:
                seq = []
                for off in time_offsets:
                    tt = center + pd.Timedelta(hours=off)
                    seq.append(storm.loc[tt, var] if tt in storm.index else np.nan)
                term_data[var].append(seq)

    # 全体Z-score参数：取自全部筛选后的有效台风样本全集，不去极值
    means, stds = {}, {}
    for var in vars_list:
        all_vals = df[var].dropna().values.astype(float)
        if len(all_vals) > 0:
            means[var] = np.mean(all_vals)
            stds[var]  = np.std(all_vals)
        else:
            means[var] = 0.0
            stds[var]  = 1.0

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
        # 在每组下方标注持续/终止
        ax.text(1, -0.12, '持续', transform=ax.get_xaxis_transform(), ha='center', fontsize=8, color='#4C72B0')
        ax.text(2, -0.12, '终止', transform=ax.get_xaxis_transform(), ha='center', fontsize=8, color='#4C72B0')
        ax.text(3.8, -0.12, '持续', transform=ax.get_xaxis_transform(), ha='center', fontsize=8, color='#DD8452')
        ax.text(4.8, -0.12, '终止', transform=ax.get_xaxis_transform(), ha='center', fontsize=8, color='#DD8452')

        # y轴标签颜色
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
    delta_cols = [v + '_delta24' for v in vars_list]
    delta_idx  = [feature_cols.index(c) for c in delta_cols]
    weights = feature_imp[delta_idx]
    weights = weights / weights.sum()

    df = samples.dropna(subset=delta_cols).copy()
    # Z-score基准μ、σ：取自RI事件集(df)，不去极值
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
    print('=' * 60)
    print('  从中间数据重新生成四张图')
    print('=' * 60)

    print('\n[1/2] 读取中间数据...')
    merged = pd.read_csv(MERGED_CSV, encoding='utf-8-sig')
    merged['datetime'] = pd.to_datetime(merged['datetime'])
    merged['is_ri'] = merged['is_ri'].astype(bool)
    print(f'  全量数据: {len(merged)} 条')

    samples = pd.read_csv(SAMPLES_CSV, encoding='utf-8-sig')
    print(f'  分析样本: {len(samples)} 条')

    # 提取有效RI事件
    valid_events = sorted(merged[merged['ri_event_id'] > 0]['ri_event_id'].unique().tolist())
    print(f'  有效RI事件: {len(valid_events)} 个')

    print('\n[2/2] 重新绘图...')
    print('\n>>> 图1: 合成演变图')
    plot_composite_evolution(merged, valid_events, VAR_LIST)

    print('\n>>> 图2: 分组箱线图')
    plot_boxplot(samples, VAR_LIST)

    print('\n>>> 图3: 随机森林特征重要性')
    feature_imp, feature_cols = plot_feature_importance(samples, VAR_LIST)

    print('\n>>> 图4: 综合环境趋势堆叠图')
    # 图4的Z-score基准μ、σ取自RI事件集(samples)，不去极值
    plot_composite_trend(samples, VAR_LIST, feature_imp, feature_cols)

    print('\n' + '=' * 60)
    print(f'  全部图片已重新生成，保存至: {PIC_DIR}')
    print('=' * 60)


if __name__ == '__main__':
    main()
