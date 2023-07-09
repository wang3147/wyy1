import pandas as pd
import numpy as np  # 引用numpy库,主要用来做科学计算
from sklearn import metrics
import time
import matplotlib.pyplot as plt  # 引用matplotlib库,主要用来画
from sklearn import datasets, ensemble
import operator
import sklearn.model_selection as sk_model_selection
from sklearn.model_selection import cross_val_score
import joblib

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
def test_GradientBoostingRegressor(*data):
    X_train, X_test, y_train, y_test = data
    '''
    通常我们用步长和迭代最大次数一起来决定算法的拟合效果，要一起调参
    n_estimators:迭代次数；默认100
    learning_rate:学习率，即步长
    max_depth：基回归估计器的最大深度，调整好参数可以达到更好的效果
    loss:表示损失函数，
    '''
    regr = ensemble.GradientBoostingRegressor(n_estimators=99, max_depth=13, learning_rate=0.09, loss='ls')
    start_time = time.time()
    regr.fit(X_train, y_train)  # 训练模型

    # joblib.dump(regr, 'train_model.m')  # 保存模型    后加的

    end_time = time.time()
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
    print("时间：", end_time - start_time)
    # pp = regr.get_params()
    # print(pp)
    print("测试集-标签：", y_test)
    print("测试集-预测标签：", pred_test)
    # # 交叉验证-1
    # # 将得分*（-1），因为scikit learn计算的是负的MAE
    # # 'neg_log_loss',
    # # 'neg_mean_absolute_error',
    # # 'neg_mean_squared_error',
    # # 'neg_mean_squared_log_error',
    # # 'neg_median_absolute_error',
    scores = -1 * cross_val_score(regr, X_train, y=y_train, cv=5, scoring='neg_mean_absolute_error')
    print("MAE scores:\n", scores)
    scores1 = -1 * cross_val_score(regr, X_train, y=y_train, cv=5, scoring='neg_mean_squared_error')
    print("MSE scores:\n", scores1)
    scores2 = cross_val_score(regr, X_train, y=y_train, cv=5, scoring='r2')
    print("R² scores:\n", scores2)
    accs = sk_model_selection.cross_val_score(regr, X_train, y=y_train, cv=5)
    print("scores: ", accs)
# 调用 test_GradientBoostingRegressor
test_GradientBoostingRegressor(X_train, X_test, y_train, y_test)


# 调参
# n_estimators-迭代次数调参
#GBR使用的学习算法的数量，设备性能好，设置更大，效果更好
# def test_GradientBoostingRegressor_num(*data):
#     X_train, X_test, y_train, y_test = data
#     # 测试 GradientBoostingRegressor 的预测性能随 n_estimators 参数的影响
#     # 250个数据
#     nums = np.arange(1, 251, step=1)
#     testing_scores = []
#     training_scores = []
#     train_mape =[]
#     for num in nums:
#         regr = ensemble.GradientBoostingRegressor(n_estimators=num,max_depth=11)
#         regr.fit(X_train, y_train)
#         training_scores.append(regr.score(X_train, y_train))
#         testing_scores.append(regr.score(X_test, y_test))
#         pred_train = regr.predict(X_train)
#         pred_test = regr.predict(X_test)
#         train_mape.append(mape(y_train, pred_train))
#         print("\n**{}**".format(num))
#         print("训练集-R²:%f" % regr.score(X_train, y_train))
#         print("训练集-mae:{}".format(metrics.mean_absolute_error(y_train, pred_train)))
#         print("训练集-mse:{}".format(metrics.mean_squared_error(y_train, pred_train)))
#         print("训练集-mape:{}".format(mape(y_train, pred_train)))
#         print("训练集-smape:{}".format(smape(y_train, pred_train)))
#         print("测试集-R²:%f" % regr.score(X_test, y_test))
#         print("测试集-mae:{}".format(metrics.mean_absolute_error(y_test, pred_test)))
#         print("测试集-mse:{}".format(metrics.mean_squared_error(y_test, pred_test)))
#         print("测试集-mape:{}".format(mape(y_test, pred_test)))
#         print("测试集-smape:{}".format(smape(y_test, pred_test)))
#
#     # # # 创建一个绘图窗口
#     # fig, ax1=plt.subplots()
#     # # plot x轴数据nums，y轴数据training_scores
#     # ax1.plot(nums, training_scores, label='$\mathregular{R^2}$')
#     # ax2 = ax1.twinx()  # 创建第二个坐标轴
#     # ax2.plot(nums, train_mape, label='MAPE')
#     # ax1.set_xlabel('n_estimators')
#     # ax1.set_ylabel('$\mathregular{R^2}$')
#     # ax1.set_ylim(0, 1.05)
#     # ax1.legend(loc="upper right")
#     # ax2.set_ylabel('MAPE %')
#     # ax2.legend(loc="lower right")
#     # plt.savefig("迭代调整.png", bbox_inches="tight")
#     # plt.show()
#
#     train_max_index, train_max_number = max(enumerate(training_scores), key=operator.itemgetter(1))
#     print('\ntrain_max_index(迭代次数):{}, train_max_number(R²):{}'.format(nums[train_max_index], train_max_number))
#     test_max_index, test_max_number = max(enumerate(testing_scores), key=operator.itemgetter(1))
#     print('\ntest_max_index(迭代次数):{}, test_max_number(R²):{}'.format(nums[test_max_index], test_max_number))
#
# #调用 test_GradientBoostingRegressor_num
# test_GradientBoostingRegressor_num(X_train, X_test, y_train, y_test)

# max_depth-最大深度调参
# 优化其他参数前先调整max_depth参数
# def test_GradientBoostingRegressor_maxdepth(*data):
#     X_train, X_test, y_train, y_test = data
#     # 测试 GradientBoostingRegressor 的预测性能随 max_depth 参数的影响
#     maxdepths = np.arange(1, 51)
#     testing_scores = []
#     training_scores = []
#     train_mape = []
#     for maxdepth in maxdepths:
#         regr = ensemble.GradientBoostingRegressor(max_depth=maxdepth, max_leaf_nodes=None,n_estimators=99,learning_rate=0.1)
#         regr.fit(X_train, y_train)
#         training_scores.append(regr.score(X_train, y_train))
#         testing_scores.append(regr.score(X_test, y_test))
#         pred_test = regr.predict(X_test)
#         pred_train = regr.predict(X_train)
#         print("\n**{}**".format(maxdepth))
#         print("训练集-R²:%f" % regr.score(X_train, y_train))
#         print("训练集-mae:{}".format(metrics.mean_absolute_error(y_train, pred_train)))
#         print("训练集-mse:{}".format(metrics.mean_squared_error(y_train, pred_train)))
#         print("训练集-mape:{}".format(mape(y_train, pred_train)))
#         print("训练集-smape:{}".format(smape(y_train, pred_train)))
#         print("测试集-R²:%f" % regr.score(X_test, y_test))
#         print("测试集-mae:{}".format(metrics.mean_absolute_error(y_test, pred_test)))
#         print("测试集-mse:{}".format(metrics.mean_squared_error(y_test, pred_test)))
#         print("测试集-mape:{}".format(mape(y_test, pred_test)))
#         print("测试集-smape:{}".format(smape(y_test, pred_test)))
#
#     fig = plt.figure()
#     ax = fig.add_subplot(1, 1, 1)
#     ax.plot(maxdepths, training_scores, label="Training Score")
#     ax.plot(maxdepths, testing_scores, label="Testing Score")
#     ax.set_xlabel("max_depth")
#     ax.set_ylabel("score")
#     ax.legend(loc="lower right")
#     ax.set_ylim(-1, 1.05)
#     plt.suptitle("GradientBoostingRegressor")
#     plt.show()
#     train_max_index, train_max_number = max(enumerate(training_scores), key=operator.itemgetter(1))
#     print('\ntrain_max_index(树深):{}, train_max_number(R²):{}'.format(maxdepths[train_max_index], train_max_number))
#     test_max_index, test_max_number = max(enumerate(testing_scores), key=operator.itemgetter(1))
#     print('\ntest_max_index(树深):{}, test_max_numbe(R²)r:{}'.format(maxdepths[test_max_index], test_max_number))
#
#     # # 创建一个绘图窗口
#     # fig, ax1=plt.subplots()
#     # ax1.plot(maxdepths, training_scores, label='$\mathregular{R^2}$')
#     # ax2 = ax1.twinx()  # 创建第二个坐标轴
#     # ax2.plot(maxdepths, train_mape, label='MAPE')
#     # ax1.set_xlabel('n_estimators')
#     # ax1.set_ylabel('$\mathregular{R^2}$')
#     # ax1.set_ylim(-1, 1.05)
#     # ax1.legend(loc="upper right")
#     # ax2.set_ylabel('MAPE %')
#     # ax2.legend(loc="lower right")
#     # plt.savefig("迭代调整.png", bbox_inches="tight")
#     # plt.show()
#
#     train_max_index, train_max_number = max(enumerate(training_scores), key=operator.itemgetter(1))
#     print('\ntrain_max_index(迭代次数):{}, train_max_number(R²):{}'.format(maxdepths[train_max_index], train_max_number))
#     test_max_index, test_max_number = max(enumerate(testing_scores), key=operator.itemgetter(1))
#     print('\ntest_max_index(迭代次数):{}, test_max_number(R²):{}'.format(maxdepths[test_max_index], test_max_number))
#
# #调用 test_GradientBoostingRegressor_maxdepth
# test_GradientBoostingRegressor_maxdepth(X_train, X_test, y_train, y_test)

# # learning_rate学习率调参
# def test_GradientBoostingRegressor_learning(*data):
#     X_train, X_test, y_train, y_test = data
#     # 测试 GradientBoostingRegressor 的预测性能随 learning_rate 参数的影响
#     learnings = np.linspace(0.01, 1, num=100)
#     fig = plt.figure()
#     ax = fig.add_subplot(1, 1, 1)
#     testing_scores = []
#     training_scores = []
#     for learning in learnings:
#         regr = ensemble.GradientBoostingRegressor(learning_rate=learning, max_depth=13, n_estimators=99)
#         regr.fit(X_train, y_train)
#         training_scores.append(regr.score(X_train, y_train))
#         testing_scores.append(regr.score(X_test, y_test))
#         pred_test = regr.predict(X_test)
#         pred_train = regr.predict(X_train)
#         print("\n**{}**".format(learning))
#         print("训练集-R²:%f" % regr.score(X_train, y_train))
#         print("训练集-mae:{}".format(metrics.mean_absolute_error(y_train, pred_train)))
#         print("训练集-mse:{}".format(metrics.mean_squared_error(y_train, pred_train)))
#         print("训练集-mape:{}".format(mape(y_train, pred_train)))
#         print("训练集-smape:{}".format(smape(y_train, pred_train)))
#         print("测试集-R²:%f" % regr.score(X_test, y_test))
#         print("测试集-mae:{}".format(metrics.mean_absolute_error(y_test, pred_test)))
#         print("测试集-mse:{}".format(metrics.mean_squared_error(y_test, pred_test)))
#         print("测试集-mape:{}".format(mape(y_test, pred_test)))
#         print("测试集-smape:{}".format(smape(y_test, pred_test)))
#
#     ax.plot(learnings, training_scores, label="Training Score")
#     ax.plot(learnings, testing_scores, label="Testing Score")
#     ax.set_xlabel("learning_rate")
#     ax.set_ylabel("score")
#     ax.legend(loc="lower right")
#     ax.set_ylim(-1, 1.05)
#     plt.suptitle("GradientBoostingRegressor")
#     plt.show()
#     train_max_index, train_max_number = max(enumerate(training_scores), key=operator.itemgetter(1))
#     print('\ntrain_max_index(学习率):{}, train_max_number(R²):{}'.format(learnings[train_max_index], train_max_number))
#     test_max_index, test_max_number = max(enumerate(testing_scores), key=operator.itemgetter(1))
#     print('\ntest_max_index(学习率):{}, test_max_number(R²):{}'.format(learnings[test_max_index], test_max_number))
#
# # 调用 test_GradientBoostingRegressor_learning
# test_GradientBoostingRegressor_learning(X_train, X_test, y_train, y_test)


# # max_features最大特征数调参
# def test_GradientBoostingRegressor_max_features(*data):
#     '''
#     测试 GradientBoostingRegressor 的预测性能随 max_features 参数的影响
#     '''
#     X_train, X_test, y_train, y_test = data
#     fig = plt.figure()
#     ax = fig.add_subplot(1, 1, 1)
#     max_features = np.linspace(0.1, 0.6)
#     testing_scores = []
#     training_scores = []
#     for features in max_features:
#         regr = ensemble.GradientBoostingRegressor(max_features=features,max_depth=15)
#         regr.fit(X_train, y_train)
#         training_scores.append(regr.score(X_train, y_train))
#         testing_scores.append(regr.score(X_test, y_test))
#
#         print("\n**{}**".format(features))
#         pred_test = regr.predict(X_test)
#         pred_train = regr.predict(X_train)
#         print("Train:{}, Test:{}".format(mape(y_train,pred_train),mape(y_test, pred_test)))
#
#     ax.plot(max_features, training_scores, label="Training Score")
#     ax.plot(max_features, testing_scores, label="Training Score")
#     ax.set_xlabel("max_features")
#     ax.set_ylabel("score")
#     ax.legend(loc="lower right")
#     ax.set_ylim(0, 1.05)
#     plt.suptitle("GradientBoostingRegressor")
#     plt.show()
#     train_max_index, train_max_number = max(enumerate(training_scores), key=operator.itemgetter(1))
#     print('\ntrain_max_index:{}, train_max_number:{}'.format(max_features[train_max_index], train_max_number))
#     test_max_index, test_max_number = max(enumerate(testing_scores), key=operator.itemgetter(1))
#     print('\ntest_max_index:{}, test_max_number:{}'.format(max_features[test_max_index], test_max_number))
#
# # 调用 test_GradientBoostingRegressor_max_features
# test_GradientBoostingRegressor_max_features(X_train, X_test, y_train, y_test)
# #
# #
# subsample子采样概率调参
# def test_GradientBoostingRegressor_subsample(*data):
#     '''
#     测试 GradientBoostingRegressor 的预测性能随 subsample 参数的影响
#     '''
#     X_train, X_test, y_train, y_test = data
#     fig = plt.figure()
#     ax = fig.add_subplot(1, 1, 1)
#     subsamples = np.linspace(0.001, 1.0, num=100)
#     testing_scores = []
#     training_scores = []
#     for subsample in subsamples:
#         regr = ensemble.GradientBoostingRegressor(subsample=subsample,max_depth=14,n_estimators=90,learning_rate=0.11)
#         regr.fit(X_train, y_train)
#         training_scores.append(regr.score(X_train, y_train))
#         testing_scores.append(regr.score(X_test, y_test))
#
#         print("\n**{}**".format(subsample))
#         pred_test = regr.predict(X_test)
#         pred_train = regr.predict(X_train)
#         print("Train:{}, Test:{}".format(mape(y_train,pred_train),mape(y_test, pred_test)))
#
#     ax.plot(subsamples, training_scores, label="Training Score")
#     ax.plot(subsamples, testing_scores, label="Testing Score")
#     ax.set_xlabel("subsample")
#     ax.set_ylabel("score")
#     ax.legend(loc="lower right")
#     ax.set_ylim(-1, 1.05)
#     plt.suptitle("GradientBoostingRegressor")
#     plt.show()
#     train_max_index, train_max_number = max(enumerate(training_scores), key=operator.itemgetter(1))
#     print('\ntrain_max_index:{}, train_max_number:{}'.format(subsamples[train_max_index], train_max_number))
#     test_max_index, test_max_number = max(enumerate(testing_scores), key=operator.itemgetter(1))
#     print('\ntest_max_index:{}, test_max_number:{}'.format(subsamples[test_max_index], test_max_number))
#
#
# # 调用 test_GradientBoostingRegressor_subsample
# test_GradientBoostingRegressor_subsample(X_train, X_test, y_train, y_test)
#

# # loss损失函数和alpha分位数调参
# # def test_GradientBoostingRegressor_loss(*data):
#     '''
#     测试 GradientBoostingRegressor 的预测性能随不同的损失函数和 alpha 参数的影响
#     '''
#     X_train, X_test, y_train, y_test = data
#     fig = plt.figure()
#     nums = np.arange(1, 200, step=2)
#     ########## 绘制 huber ######
#     ax = fig.add_subplot(2, 1, 1)
#     alphas = np.linspace(0.01, 1.0, endpoint=False, num=5)
#     for alpha in alphas:
#         testing_scores = []
#         training_scores = []
#         for num in nums:
#             regr = ensemble.GradientBoostingRegressor(n_estimators=num, loss='huber', alpha=alpha)
#             regr.fit(X_train, y_train)
#             training_scores.append(regr.score(X_train, y_train))
#             testing_scores.append(regr.score(X_test, y_test))
#         ax.plot(nums, training_scores, label="Training Score:alpha=%f" % alpha)
#         ax.plot(nums, testing_scores, label="Testing Score:alpha=%f" % alpha)
#     ax.set_xlabel("estimator num")
#     ax.set_ylabel("score")
#     ax.legend(loc="lower right", framealpha=0.4)
#     ax.set_ylim(0, 1.05)
#     ax.set_title("loss=%huber")
#     plt.suptitle("GradientBoostingRegressor")
#     #### 绘制 ls  和 lad
#     ax = fig.add_subplot(2, 1, 2)
#     for loss in ['ls', 'lad']:
#         testing_scores = []
#         training_scores = []
#         for num in nums:
#             regr = ensemble.GradientBoostingRegressor(n_estimators=num, loss=loss)
#             regr.fit(X_train, y_train)
#             training_scores.append(regr.score(X_train, y_train))
#             testing_scores.append(regr.score(X_test, y_test))
#         ax.plot(nums, training_scores, label="Training Score:loss=%s" % loss)
#         ax.plot(nums, testing_scores, label="Testing Score:loss=%s" % loss)
#     ax.set_xlabel("estimator num")
#     ax.set_ylabel("score")
#     ax.legend(loc="lower right", framealpha=0.4)
#     ax.set_ylim(0, 1.05)
#     ax.set_title("loss=ls,lad")
#     plt.suptitle("GradientBoostingRegressor")
#     plt.show()
#     print('finish')
#
# # 调用 test_GradientBoostingRegressor_loss
# test_GradientBoostingRegressor_loss(X_train, X_test, y_train, y_test)
