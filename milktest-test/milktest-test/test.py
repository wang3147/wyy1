import time
import numpy as np
from sklearn import metrics
import pandas as pd
import xgboost as xgb
import joblib
# 忽略警告
import warnings
warnings.filterwarnings("ignore")
# Read”按钮读取待测溶液的光强，并将获取的数据送入到预测模型中。
# “Measure”按钮将送入到预测模型的数据进行预测，显示出牛奶蛋白质和脂肪的含量；
# “Save”按钮保存测量信息及预测信息。
# 显示所有行
pd.set_option('display.max_rows', None)

# 获取CSV文件1
pre_train = pd.read_csv('data_test.csv', encoding="unicode_escape")
target1 = 'Protein'  # 定义目标值
target2 = 'Fat'

# 数据清理，删除包含缺失值的行
train = pre_train.dropna()
'''
sample()进行随机抽样，DataFrame.sample(n=None, frac=None, replace=False, weights=None, random_state=None, axis=None)
n 随机抽取的行数，frac 抽取的比例，replace 有无放回抽样默认false无放回，weights 等概率抽样，
random_state这个参数可以复现抽样结果，
axis 控制对行列的抽样，默认0对行
'''
# 拆分成训练集和测试集，测试集大小为原始数据集大小的 1/10
train_data = train.sample(frac=0.9, random_state=10)
test_data = train.drop(train_data.index)
# 列表解析（[ x for x in list]）
X_columns = [x for x in train.columns if x not in [target1, target2]]
# 分类数据
X_train = train_data[X_columns]
y_train = train_data[target1]
X_test = test_data[X_columns]
y_test = test_data[target1]


# 模型评价标准指标mape, smape, mse, rmse, mae
# mape-平均绝对百分比误差。分母为0，该公式不能用，越小越好
def mape(y_true, y_pred):
    return np.mean(np.abs((y_pred - y_true) / y_true)) * 100


# smape-对称平均绝对百分比误差。分母为0，该公式不能用，越小越好
def smape(y_true, y_pred):
    return 2.0 * np.mean(np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true))) * 100


# # mse-均方误差。误差越小，该值越小
#     mse = metrics.mean_squared_error(y_true, y_pred)
# # rmse-均方根误差。误差越小，该值越小。RMSE=10，可以认为回归效果相比真实值平均相差10
#     rmse = np.sqrt(metrics.mean_squared_error(y_true, y_pred))
# # mae-平均绝对误差。误差越小，该值越小
#     mae = metrics.mean_absolute_error(y_true, y_pred)
#
# # 集成学习梯度提升决策树GradientBoostingRegressor回归模型
def xgboost_parameter_tuning(*data):
    X_train, X_test, y_train, y_test = data
    #regr = xgb.XGBRegressor(n_estimators=235, max_depth=6, subsample=0.6,learning_rate=0.14, gamma=0, reg_alpha=0, reg_lambda=0)
    regr = xgb.XGBRegressor(n_estimators=300, max_depth=3, subsample=0.5,learning_rate=0.1, gamma=0, reg_alpha=0, reg_lambda=1)
    regr.fit(X_train, y_train)
    joblib.dump(regr, 'train_model_protein.m')  # 保存模型

    pred_train = regr.predict(X_train)  # 输出预测值
    pred_test = regr.predict(X_test)  # 输出预测值


    print("训练集-R²:%f" % regr.score(X_train, y_train))  # 在训练集上做验证，r^2  通过score调用
    print("训练集-mae:{}".format(metrics.mean_absolute_error(y_train, pred_train)))  # mae平均绝对误差metrics.mean_absolute_error
    print("训练集-mse:{}".format(metrics.mean_squared_error(y_train, pred_train)))  # mse均方误差metrics.mean_squared_error
    print("训练集-mape:{}".format(mape(y_train, pred_train)))  # 平均绝对百分比误差
    print("训练集-smape:{}".format(smape(y_train, pred_train)))  # 对称平均绝对百分比误差
    print("测试集-R²:%f" % regr.score(X_test, y_test))  # 在测试集上做验证
    print("测试集-mae:{}".format(metrics.mean_absolute_error(y_test, pred_test)))
    print("测试集-mse:{}".format(metrics.mean_squared_error(y_test, pred_test)))
    print("测试集-mape:{}".format(mape(y_test, pred_test)))
    print("测试集-smape:{}".format(smape(y_test, pred_test)))

xgboost_parameter_tuning(X_train, X_test, y_train, y_test)

