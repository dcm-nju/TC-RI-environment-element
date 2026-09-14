import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
import warnings
from pathlib import Path
warnings.filterwarnings("ignore")

# ===================== 路径设置 =====================
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent  # 项目根目录
picture_dir = project_root / 'picture' / 'version1' / '1.4'
picture_dir.mkdir(parents=True, exist_ok=True)

# 数据集路径（从picture/version1/1.4读取）  
DATA_PATH = picture_dir / 'typhoon_ri_dataset.csv'
# ====================================================

# 1. 读取数据集
df = pd.read_csv(DATA_PATH, encoding="utf-8")
print("=" * 70)
print("数据集基本信息预览")
print("=" * 70)
print(f"总样本量：{len(df)}")
# 统计正负样本
label_count = df["label"].value_counts()
print(f"正样本(即将/已终止RI, label=1)：{label_count[1]} 个")
print(f"负样本(短期不会终止RI, label=0)：{label_count[0]} 个")
print(f"正负样本比例 1 : {label_count[0]/label_count[1]:.2f}")
print("\n前5行数据预览：")
print(df.head())

# 2. 定义特征列 & 标签列（完全不变）
feature_cols = [
    # 原始环境要素（绝对值）
    "RSST", "SHRD", "RHLO", "RHMD", "RHHI", "VMPI", "MOVE_SPEED",
    # 6小时变化幅度
    "RSST_6h_change", "SHRD_6h_change", "RHLO_6h_change", "RHMD_6h_change",
    "RHHI_6h_change", "VMPI_6h_change", "MOVE_SPEED_6h_change",
    # 24小时变化幅度
    "RSST_24h_change", "SHRD_24h_change", "RHLO_24h_change", "RHMD_24h_change",
    "RHHI_24h_change", "VMPI_24h_change", "MOVE_SPEED_24h_change"
]
label_col = "label"

# 划分特征X、标签y
X = df[feature_cols]
y = df[label_col]

# 兜底：删除残留缺失值
X = X.dropna()
y = y.loc[X.index]

# 3. 划分训练集(80%) + 测试集(20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\n训练集样本：{len(X_train)} | 测试集样本：{len(X_test)}")

# 4. XGBoost模型
xgb_model = XGBClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=2.3,
    random_state=42,
    n_jobs=-1,
    eval_metric='logloss'
)
xgb_model.fit(X_train, y_train)

# 5. 模型效果评估
y_pred = xgb_model.predict(X_test)
print("\n" + "=" * 70)
print("XGBoost模型分类效果评估")
print("=" * 70)
print(f"整体准确率：{accuracy_score(y_test, y_pred):.4f}")
print("\n分类详细报告：")
print(classification_report(y_test, y_pred, target_names=["短期不终止RI", "即将/已终止RI"]))

# 6. 计算特征重要性
feature_import = pd.DataFrame({
    "环境因子": feature_cols,
    "重要性得分": xgb_model.feature_importances_
})
# 按重要性从高到低排序
feature_import = feature_import.sort_values("重要性得分", ascending=False).reset_index(drop=True)

print("\n" + "=" * 70)
print("各环境因子重要性排名（得分越高，影响越大）")
print("=" * 70)
print(feature_import)

# 7. 导出特征重要性结果到Excel
output_excel = picture_dir / "XGBoost特征重要性排名表.xlsx"
feature_import.to_excel(output_excel, index=False)
print(f"\n特征重要性表格已导出：{output_excel}")

# 8. 绘制特征重要性横向柱状图
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

plt.figure(figsize=(14, 10))
sns.barplot(x="重要性得分", y="环境因子", data=feature_import, palette="viridis")
plt.title("各环境要素及变化幅度对台风RI终止的特征重要性排名（XGBoost）", fontsize=16, pad=20)
plt.xlabel("特征重要性得分", fontsize=12)
plt.ylabel("环境因子", fontsize=12)
plt.tight_layout()

# 保存图片到picture/version1/1.4文件夹
output_img = picture_dir / "XGBoost特征重要性柱状图.png"
plt.savefig(output_img, dpi=300, bbox_inches="tight")
print(f"特征重要性柱状图已导出：{output_img}")
plt.show()
