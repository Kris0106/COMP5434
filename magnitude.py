import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, classification_report
from sklearn.base import clone

# 尝试导入XGBoost（如果可用）
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("提示: XGBoost未安装，将只使用RandomForest。要安装请运行: pip install xgboost")

# --- 配置开关 ---
# 模式：'baseline' 使用原始5特征 + RF(100)；'max_optimize' 使用消融最优组合
MODE = 'baseline'  # 可改为 'max_optimize'

# 细粒度特征开关（仅当 MODE 不是 baseline 时有意义）
FEATURE_SIGABS = False
FEATURE_INTERACTIONS = False
FEATURE_BINNING = False

# 网格搜索（默认关闭）
USE_GRID_SEARCH = False

# --- 数据加载 ---
# 请将下方路径替换为你的文件实际路径
train_file_path = '/Users/kris/Desktop/COMP5434/Project/comp-5434-20251-project-task-1/train.csv' #<-- 训练集路径
test_file_path = '/Users/kris/Desktop/COMP5434/Project/comp-5434-20251-project-task-1/test.csv'   #<-- 测试集路径
submission_file_path = '/Users/kris/Desktop/COMP5434/Project/comp-5434-20251-project-task-1/submission.csv'  # 预测输出

try:
    df = pd.read_csv(train_file_path)
    print(f"成功从 '{train_file_path}' 加载训练数据。")
    print("数据形状 (行, 列):", df.shape)
except FileNotFoundError:
    print(f"错误: 文件未找到，请检查路径 '{train_file_path}' 是否正确。")
    # 如果文件未找到，则退出程序
    exit()


# --- 1. 数据清洗与探查 (Data Cleaning & Inspection) ---
print("\n--- 1. 数据清洗与探查 ---")

# 1a. 查看数据概览和类型
print("\n[INFO] 数据概览和类型:")
df.info()

# 1b. 检查缺失值
print("\n[CHECK] 检查缺失值:")
print(df.isnull().sum())
print("结论: 数据集中没有任何缺失值，每一列都是完整的。")

# 1c. 检查重复值
# 我们检查除'id'外的其他列，因为id本身就是唯一的
is_duplicated = df.duplicated(subset=df.columns.difference(['id'])).sum()
print(f"\n[CHECK] 检查重复的地震事件记录: {is_duplicated}")
if is_duplicated > 0:
    # 如果有重复项，则移除
    df.drop_duplicates(subset=df.columns.difference(['id']), keep='first', inplace=True)
    print(f"移除了 {is_duplicated} 条重复记录。")
else:
    print("结论: 数据中没有重复的事件记录。")

# 1d. 查看统计摘要以发现异常值
print("\n[INFO] 各特征的统计摘要:")
print(df.describe())
print("\n结论分析:")
print("- magnitude (震级): 范围在 6.5 到 8.45 之间，看起来是合理的强震数据。")
print("- depth (深度): 最小值为 5km，最大值为 664km，在地球物理学上是可能的范围。")
print("- cdi/mmi (烈度): 范围在 0-9 之间，符合烈度量表的常规范围。")
print("- sig (显著性): 包含负值，这可能是某个计算指标的特性，范围较大但没有出现极端到需要处理的异常。")
print("整体来看，数据质量很高，没有明显的异常或错误需要清洗。")


#! --- 2. 特征工程 (Feature Engineering) ---
print("\n--- 2. 特征准备/工程 ---")

# 定义基础特征与目标
base_features = ['magnitude', 'depth', 'cdi', 'mmi', 'sig']
target = 'label'

if MODE == 'max_optimize':
    # 使用与最终验证集相同的划分，直接在验证集上测试所有候选配置的Macro-F1
    print("\n[max_optimize] 正在通过验证集 Macro-F1 选择最优配置（确保≥baseline）...")
    
    # 先做划分（与后续训练使用相同的划分）
    X_temp = df[base_features].copy()
    y_temp = df[target]
    X_opt_train, X_opt_val, y_opt_train, y_opt_val = train_test_split(
        X_temp, y_temp, test_size=0.2, random_state=42, stratify=y_temp
    )
    
    # 候选配置：包括baseline和其他组合
    feature_options = [
        {'sigabs': False, 'binning': False, 'name': 'baseline'},
        {'sigabs': True, 'binning': True, 'name': 'sigabs+binning'},
        {'sigabs': True, 'binning': False, 'name': 'sigabs_only'},
        {'sigabs': False, 'binning': True, 'name': 'binning_only'}
    ]
    n_est_options = [50, 100, 200, 300]
    
    def build_X_for_flags(dfin, flags):
        Xtmp = dfin[base_features].copy()
        if flags['sigabs']:
            Xtmp['sig_abs'] = np.abs(Xtmp['sig'])
        if flags['binning']:
            Xtmp['depth_shallow'] = (Xtmp['depth'] < 50).astype(int)
            Xtmp['depth_medium'] = ((Xtmp['depth'] >= 50) & (Xtmp['depth'] < 200)).astype(int)
            Xtmp['depth_deep'] = (Xtmp['depth'] >= 200).astype(int)
            Xtmp['magnitude_low'] = (Xtmp['magnitude'] < 7.0).astype(int)
            Xtmp['magnitude_high'] = (Xtmp['magnitude'] >= 7.5).astype(int)
        return Xtmp
    
    # 在验证集上测试所有候选配置
    val_results = []
    baseline_val_f1 = None
    
    print("\n[验证集测试] 评估所有候选配置...")
    for fflags in feature_options:
        # 构建训练和验证特征
        X_train_feat = build_X_for_flags(pd.DataFrame(X_opt_train, columns=base_features), fflags)
        X_val_feat = build_X_for_flags(pd.DataFrame(X_opt_val, columns=base_features), fflags)
        
        for n_est in n_est_options:
            model_test = RandomForestClassifier(
                n_estimators=n_est, random_state=42, class_weight='balanced'
            )
            model_test.fit(X_train_feat, y_opt_train)
            val_pred = model_test.predict(X_val_feat)
            val_f1 = f1_score(y_opt_val, val_pred, average='macro')
            
            # 记录baseline的验证集分数
            if fflags['name'] == 'baseline' and n_est == 100:
                baseline_val_f1 = val_f1
            
            val_results.append({
                'flags': fflags, 'n_estimators': n_est, 'val_f1': val_f1
            })
            print(f"  {fflags['name']}, n_est={n_est} -> Val_Macro-F1={val_f1:.4f}")
    
    # 选择验证集 Macro-F1 最高的配置
    val_results.sort(key=lambda x: x['val_f1'], reverse=True)
    best_config = val_results[0]
    
    print(f"\n[验证集结果] Top配置: {best_config['flags']['name']}, n_estimators={best_config['n_estimators']}, Val_Macro-F1={best_config['val_f1']:.4f}")
    if baseline_val_f1 is not None:
        print(f"[参考] baseline(100树) Val_Macro-F1={baseline_val_f1:.4f}")
    
    # 确保最优配置的验证集 Macro-F1 至少≥baseline，否则使用baseline
    if baseline_val_f1 is not None and baseline_val_f1 > best_config['val_f1']:
        print(f"\n[最终选择] baseline验证集表现更好({baseline_val_f1:.4f} > {best_config['val_f1']:.4f})，使用baseline配置")
        FEATURE_SIGABS = False
        FEATURE_BINNING = False
        SELECTED_N_ESTIMATORS = 100
    else:
        print(f"\n[最终选择] 使用最优配置: {best_config['flags']['name']}, n_estimators={best_config['n_estimators']}, Val_Macro-F1={best_config['val_f1']:.4f}")
        if baseline_val_f1 is not None:
            improvement = best_config['val_f1'] - baseline_val_f1
            print(f"[提升] 相比baseline提升: {improvement:+.4f}")
        FEATURE_SIGABS = best_config['flags']['sigabs']
        FEATURE_BINNING = best_config['flags']['binning']
        SELECTED_N_ESTIMATORS = best_config['n_estimators']
    
    FEATURE_INTERACTIONS = False
else:
    SELECTED_N_ESTIMATORS = 100

# 根据开关构建训练特征
if FEATURE_SIGABS or FEATURE_INTERACTIONS or FEATURE_BINNING:
    # 创建特征副本用于工程
    X = df[base_features].copy()
    print("\n创建新特征...")
    if FEATURE_SIGABS:
        X['sig_abs'] = np.abs(X['sig'])
    if FEATURE_INTERACTIONS:
        X['magnitude_depth_ratio'] = X['magnitude'] / (X['depth'] + 1)
        X['magnitude_depth_product'] = X['magnitude'] * X['depth']
        X['cdi_mmi_diff'] = np.abs(X['cdi'] - X['mmi'])
        X['cdi_mmi_sum'] = X['cdi'] + X['mmi']
        X['cdi_mmi_product'] = X['cdi'] * X['mmi']
        X['magnitude_cdi'] = X['magnitude'] * X['cdi']
        X['magnitude_mmi'] = X['magnitude'] * X['mmi']
        X['sig_magnitude'] = X['sig'] * X['magnitude']
    if FEATURE_BINNING:
        X['depth_shallow'] = (X['depth'] < 50).astype(int)
        X['depth_medium'] = ((X['depth'] >= 50) & (X['depth'] < 200)).astype(int)
        X['depth_deep'] = (X['depth'] >= 200).astype(int)
        X['magnitude_low'] = (X['magnitude'] < 7.0).astype(int)
        X['magnitude_high'] = (X['magnitude'] >= 7.5).astype(int)
    features = list(X.columns)
    print(f"特征工程开启: 原始特征数 {len(base_features)} -> 当前 {len(features)}")
else:
    # 使用原始特征作为稳健基线
    X = df[base_features].copy()
    features = base_features
    print("使用基线特征:", features)

y = df[target]

print("\n标签分布:")
print(y.value_counts())

# 分割数据为训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42, 
    stratify=y  # 保持训练集和验证集的标签分布与原始数据一致
)
print("\n数据分割完成: 80% 训练, 20% 验证。")


#! --- 3. 模型训练与优化 ---
print("\n--- 3. 模型训练与优化 ---")

if USE_GRID_SEARCH:
    print("启用网格搜索优化...")
    base_rf = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight='balanced'
    )
    cv_scores = cross_val_score(base_rf, X_train, y_train, cv=5, scoring='f1_macro')
    print(f"基础模型 5折交叉验证 Macro-F1: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'max_features': ['sqrt']
    }

    grid_rf = RandomForestClassifier(
        random_state=42,
        class_weight='balanced',
        n_jobs=-1
    )
    grid_search = GridSearchCV(
        grid_rf,
        param_grid,
        cv=5,
        scoring='f1_macro',
        n_jobs=-1,
        verbose=1
    )
    grid_search.fit(X_train, y_train)
    print(f"最佳参数: {grid_search.best_params_}")
    print(f"最佳交叉验证 Macro-F1: {grid_search.best_score_:.4f}")
    model = grid_search.best_estimator_
else:
    # 根据模式选择模型与超参
    if MODE == 'max_optimize':
        model = RandomForestClassifier(
            n_estimators=SELECTED_N_ESTIMATORS,
            random_state=42,
            class_weight='balanced'
        )
    else:
        # 基线配置
        model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight='balanced'
        )

print("训练模型...")
model.fit(X_train, y_train)
print("模型训练完成。")


# --- 4. 模型评估 ---
print("\n--- 4. 模型评估 ---")

# 在验证集上进行预测
y_pred = model.predict(X_val)

# 计算 Macro-F1 分数
macro_f1 = f1_score(y_val, y_pred, average='macro')

print(f"\n验证集 Macro-F1 Score: {macro_f1:.4f}")

# 打印详细的分类报告
print("\n详细分类报告:")
print(classification_report(y_val, y_pred))

# 特征重要性分析
if hasattr(model, 'feature_importances_'):
    print("\n特征重要性 (Top 10):")
    feature_importance = pd.DataFrame({
        'feature': features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    print(feature_importance.head(10).to_string(index=False))


# --- 5. 用全量数据重新训练最终模型 ---
print("\n--- 5. 用全量数据训练最终模型 ---")
print("使用所有训练数据重新训练模型以最大化性能...")
# 克隆模型并重新训练
final_model = clone(model)
final_model.fit(X, y)
print("最终模型训练完成。")

# --- 6. 在测试集上推理并生成提交文件 ---
print("\n--- 6. 测试集推理与提交文件生成 ---")

try:
    df_test = pd.read_csv(test_file_path)
    print(f"成功从 '{test_file_path}' 加载测试数据。形状:", df_test.shape)
except FileNotFoundError:
    print(f"错误: 文件未找到，请检查路径 '{test_file_path}' 是否正确。")
    exit()

"""
对测试集进行与训练集一致的特征构建
"""
X_test = df_test[base_features].copy()
if FEATURE_SIGABS:
    X_test['sig_abs'] = np.abs(X_test['sig'])
if FEATURE_INTERACTIONS:
    X_test['magnitude_depth_ratio'] = X_test['magnitude'] / (X_test['depth'] + 1)
    X_test['magnitude_depth_product'] = X_test['magnitude'] * X_test['depth']
    X_test['cdi_mmi_diff'] = np.abs(X_test['cdi'] - X_test['mmi'])
    X_test['cdi_mmi_sum'] = X_test['cdi'] + X_test['mmi']
    X_test['cdi_mmi_product'] = X_test['cdi'] * X_test['mmi']
    X_test['magnitude_cdi'] = X_test['magnitude'] * X_test['cdi']
    X_test['magnitude_mmi'] = X_test['magnitude'] * X_test['mmi']
    X_test['sig_magnitude'] = X_test['sig'] * X_test['magnitude']
if FEATURE_BINNING:
    X_test['depth_shallow'] = (X_test['depth'] < 50).astype(int)
    X_test['depth_medium'] = ((X_test['depth'] >= 50) & (X_test['depth'] < 200)).astype(int)
    X_test['depth_deep'] = (X_test['depth'] >= 200).astype(int)
    X_test['magnitude_low'] = (X_test['magnitude'] < 7.0).astype(int)
    X_test['magnitude_high'] = (X_test['magnitude'] >= 7.5).astype(int)

# 确保特征顺序与训练时一致
X_test = X_test[features]

# 使用最终模型预测
test_pred = final_model.predict(X_test)

# 构建提交文件 (id, label)
if 'id' not in df_test.columns:
    print("错误: 测试集中缺少 'id' 列，无法生成提交文件。")
    exit()

submission_df = pd.DataFrame({
    'id': df_test['id'],
    'label': test_pred
})

submission_df.to_csv(submission_file_path, index=False)
print(f"提交文件已保存到: {submission_file_path}")
print("预览前5行:")
print(submission_df.head())

# 如果测试集包含真实标签，则计算并打印测试集 Macro-F1
if 'label' in df_test.columns:
    test_macro_f1 = f1_score(df_test['label'], test_pred, average='macro')
    print(f"\n测试集 Macro-F1 Score: {test_macro_f1:.4f}")
    print("\n测试集详细分类报告:")
    print(classification_report(df_test['label'], test_pred))
else:
    print("\n提示: 测试集中不包含真实标签 'label'，无法计算 F1 分数。")