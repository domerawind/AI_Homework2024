import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 读取数据
file_path = './TNSRE2014.xlsx'
data = pd.read_excel(file_path)

# 显示数据集的前几行以了解其结构
data_head = data.head()

# 检查是否有缺失值
missing_values = data.isnull().sum()

# 填补缺失值（使用均值填补）
data.fillna(data.mean(numeric_only=True), inplace=True)

# 分离特征和标签
X = data.iloc[:, :-1]  # 假设最后一列是标签
y = data.iloc[:, -1]   # 最后一列是分类标签

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 划分训练集和测试集，比例为80:20
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 返回处理后的数据集
print((X_train.shape, X_test.shape), (X_train[:5], y_train[:5]), (X_test[:5], y_test[:5]))