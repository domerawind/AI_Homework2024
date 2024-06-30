import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def dataPrep( file_path ) :
    # 读取数据
    data = pd.read_excel(file_path)

    # 显示数据集的前几行以了解其结构
    data_head = data.head()

    # 检查是否有缺失值
    missing_values = data.isnull().sum()

    # 填补缺失值（使用均值填补）
    data.fillna(data.mean(numeric_only=True), inplace=True)

    # 分离特征和标签
    X = data.iloc[:, :-1]  
    y = data.iloc[:, -1]   # 最后一列是分类标签

    # 数据标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 划分训练集和测试集，比例为80:20
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test
