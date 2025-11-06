import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report

# --- 第 1 部分: 决策树节点类 ---
class Node:
    """
    该类用于表示决策树中的一个节点。
    - 如果是决策节点, 它会存储用于分裂的特征索引(feature_index)和阈值(threshold)。
    - 如果是叶节点, 它会存储最终的预测类别(value)。
    """
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, *, value=None):
        self.feature_index = feature_index  # 用于分裂的特征的索引
        self.threshold = threshold          # 分裂的阈值
        self.left = left                    # 左子树 (小于等于阈值的样本)
        self.right = right                  # 右子树 (大于阈值的样本)
        self.value = value                  # 如果是叶节点, 存储预测的类别

    def is_leaf_node(self):
        """判断当前节点是否为叶节点"""
        return self.value is not None

# --- 第 2 部分: 自定义决策树分类器 ---
class MyDecisionTreeClassifier:
    """
    我们自己从零开始实现的决策树分类器。
    """
    def __init__(self, min_samples_split=2, max_depth=100, n_features=None):
        self.min_samples_split = min_samples_split  # 节点分裂所需的最小样本数
        self.max_depth = max_depth                  # 树的最大深度
        self.n_features = n_features                # 每次分裂时考虑的特征数量
        self.root = None                            # 树的根节点

    def _gini(self, y):
        """计算一组标签的基尼不纯度 (Gini Impurity)。"""
        # 统计每个类别的出现次数
        _, counts = np.unique(y, return_counts=True)
        # 计算每个类别的概率
        probabilities = counts / len(y)
        # 基尼不纯度公式: 1 - sum(p^2)
        gini = 1 - np.sum(probabilities**2)
        return gini

    def _best_split(self, X, y, feat_idxs):
        """寻找最佳分裂点 (特征和阈值), 使得基尼不纯度下降最多 (即信息增益最大)。"""
        best_gain = -1  # 记录最大的信息增益
        split_idx, split_thresh = None, None # 记录最佳分裂的特征索引和阈值

        # 遍历指定的特征子集
        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)
            # 遍历该特征所有可能的阈值
            for thr in thresholds:
                # 1. 计算父节点的基尼不纯度
                parent_gini = self._gini(y)
                
                # 2. 根据阈值将数据分裂成左右两部分
                left_idxs = np.argwhere(X_column <= thr).flatten()
                right_idxs = np.argwhere(X_column > thr).flatten()

                # 如果分裂后有一边为空, 则此次分裂无效, 跳过
                if len(left_idxs) == 0 or len(right_idxs) == 0:
                    continue

                # 3. 计算子节点的加权基尼不纯度
                n, n_l, n_r = len(y), len(left_idxs), len(right_idxs)
                gini_l, gini_r = self._gini(y[left_idxs]), self._gini(y[right_idxs])
                child_gini = (n_l / n) * gini_l + (n_r / n) * gini_r
                
                # 4. 信息增益 = 父节点不纯度 - 子节点加权不纯度
                info_gain = parent_gini - child_gini

                # 如果找到了更好的分裂方式, 则更新记录
                if info_gain > best_gain:
                    best_gain = info_gain
                    split_idx = feat_idx
                    split_thresh = thr
                    
        return split_idx, split_thresh

    def _grow_tree(self, X, y, depth=0):
        """使用递归函数来构建决策树。"""
        n_samples, n_feats = X.shape
        n_labels = len(np.unique(y))

        # 停止条件 1: 达到最大深度, 或所有样本都属于同一类别, 或样本数量过少
        if (depth >= self.max_depth or 
            n_labels == 1 or 
            n_samples < self.min_samples_split):
            # 创建叶节点, 预测值为当前节点中数量最多的类别
            leaf_value = Counter(y).most_common(1)[0][0]
            return Node(value=leaf_value)

        # 为当前分裂随机选择一个特征子集
        feat_idxs = np.random.choice(n_feats, self.n_features, replace=False)

        # 寻找最佳分裂点
        best_feat, best_thresh = self._best_split(X, y, feat_idxs)

        # 停止条件 2: 如果找不到能带来信息增益的分裂, 也创建一个叶节点
        if best_feat is None:
            leaf_value = Counter(y).most_common(1)[0][0]
            return Node(value=leaf_value)
            
        # 根据最佳分裂点, 将数据分为左右子集
        left_idxs = np.argwhere(X[:, best_feat] <= best_thresh).flatten()
        right_idxs = np.argwhere(X[:, best_feat] > best_thresh).flatten()
        
        # 递归地为左右子集构建子树
        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth + 1)
        
        # 返回一个决策节点, 包含分裂信息和左右子树
        return Node(best_feat, best_thresh, left, right)

    def fit(self, X, y):
        """训练决策树的入口函数。"""
        # 如果未指定n_features, 则默认为总特征数的平方根 (分类问题的常用做法)
        self.n_features = X.shape[1] if self.n_features is None else min(X.shape[1], self.n_features)
        self.root = self._grow_tree(X, y)

    def _traverse_tree(self, x, node):
        """辅助函数, 用于对单个样本点, 从根节点开始遍历树以获得预测结果。"""
        if node.is_leaf_node():
            return node.value
        
        if x[node.feature_index] <= node.threshold:
            return self._traverse_tree(x, node.left)
        else:
            return self._traverse_tree(x, node.right)

    def predict(self, X):
        """对一组样本进行预测。"""
        return np.array([self._traverse_tree(x, self.root) for x in X])

# --- 第 3 部分: 自定义随机森林分类器 ---
class MyRandomForestClassifier:
    """
    我们自己从零开始实现的随机森林分类器。
    它是由多个我们自定义的决策树组成的集成模型。
    """
    def __init__(self, n_estimators=100, min_samples_split=2, max_depth=10, n_features=None):
        self.n_estimators = n_estimators      # 森林中决策树的数量
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_features = n_features
        self.trees = []                     # 用于存储森林中所有的树

    def _bootstrap_sample(self, X, y):
        """
        创建数据的自助采样 (bootstrap sample)。
        自助采样法: 从原始数据中有放回地随机抽取样本, 形成一个新的数据集。
        """
        n_samples = X.shape[0]
        # 有放回地抽取与原始数据集同样大小的样本索引
        idxs = np.random.choice(n_samples, n_samples, replace=True)
        return X[idxs], y[idxs]
        
    def fit(self, X, y):
        """训练随机森林模型。"""
        self.trees = []
        # 循环创建并训练指定数量的决策树
        for i in range(self.n_estimators):
            print(f"正在训练第 {i+1}/{self.n_estimators} 棵树...")
            tree = MyDecisionTreeClassifier(
                min_samples_split=self.min_samples_split,
                max_depth=self.max_depth,
                n_features=self.n_features
            )
            # 每棵树都在一个不同的自助采样数据集上训练
            X_sample, y_sample = self._bootstrap_sample(X, y)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)
        print("模型训练完成。")

    def predict(self, X):
        """使用多数投票法 (majority voting) 进行预测。"""
        # 1. 从森林中的每棵树获取预测结果
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        # tree_preds 的形状是 (n_estimators, n_samples)
        
        # 2. 对每个样本, 统计所有树的预测结果, 并选出票数最多的类别
        # 先将矩阵转置为 (n_samples, n_estimators), 方便对每一行 (每个样本) 进行投票
        tree_preds = tree_preds.T
        
        # 使用列表推导式和 Counter 快速找到每个样本的多数票
        y_pred = [Counter(preds).most_common(1)[0][0] for preds in tree_preds]
        return np.array(y_pred)


# --- 主程序执行部分 ---
if __name__ == "__main__":
    # --- 数据加载 ---
    file_path = '/Users/kris/Desktop/COMP5434/Project/comp-5434-20251-project-task-1/train.csv'
    try:
        df = pd.read_csv(file_path)
        print(f"成功从 '{file_path}' 加载数据。 数据形状: {df.shape}\n")
    except FileNotFoundError:
        print(f"错误: 文件未找到，请检查路径 '{file_path}'。")
        exit()

    # --- 准备建模数据 ---
    features = ['magnitude', 'depth', 'cdi', 'mmi', 'sig']
    target = 'label'

    # 将DataFrame转换为NumPy数组, 因为我们的实现是基于NumPy的
    X = df[features].values
    y = df[target].values

    # 将数据分割为训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # --- 实例化并训练我们自定义的模型 ---
    # 为了演示, 我们使用较少的树和较小的深度, 以便更快地运行
    # n_features 的一个好的默认值是总特征数的平方根
    n_features = int(np.sqrt(X.shape[1]))
    
    custom_model = MyRandomForestClassifier(
        n_estimators=25,  # 增加这个值可以提升性能, 25是为了演示时能更快跑完
        max_depth=10,     # 限制树的深度可以防止过拟合
        n_features=n_features
    )
    
    custom_model.fit(X_train, y_train)

    # --- 评估模型 ---
    print("\n--- 模型评估 ---")
    y_pred = custom_model.predict(X_val)
    
    # 计算宏平均 F1 分数
    macro_f1 = f1_score(y_val, y_pred, average='macro')
    print(f"\n验证集 Macro-F1 Score: {macro_f1:.4f}")
    
    # 打印详细的分类报告
    print("\n详细分类报告:")
    print(classification_report(y_val, y_pred))