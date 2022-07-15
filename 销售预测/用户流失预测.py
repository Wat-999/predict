"""
用户流失预测
用户的获取和流失是一个相对概念，就好比一个水池，有进口有出口。我们不能只关心进口的进水率，却忽略了出水口的出水速率。
挽留一个老用户相比拉动一个新用户，在增加营业收入、产品周期维护方面都是有好处的。并且获得一个新用户的成本是留存一个老用户的5～6倍
算法原理及项目背景
1算法原理
流失用户是指那些赠镜使用过产品或服务，由于对产品失去兴趣等种种原因，不再使用产品或服务对用户
根据流失用户所处对用户关系生命周期阶段可以将流失用户分为4类，即考察阶段流失用户、形成阶段流失用户、稳定阶段流失用户和衰退阶段流失用户
根据用户流失对原因也可以将流失用户分为4类，以下进行介绍
(1)第1类流失用户是自然消亡类。
例如用户破产、身故、移民或迁徙等，使用户无法再享受企业等产品或服务，或者用户目前所处的地理位置位于企业产品和服务的覆盖范围之外
(2)第2类流失用户是需求变化类。
用户自身的需求发生类变化，需求变化类用户的大量出现，往往是伴随着科技进步和社会习俗的变化而产生
(3)第3类流失用户是趋利流失类
因为企业竞争对手的营销活动诱惑，用户终止与该企业的用户关系，而转变为企业竞争对手的用户
(4)第4类流失用户是失望流失类
因对该企业对产品或服务不满意，用户终止与该企业对用户关系
根据以上用户流失的原因，我们在原始的51个特征种重新提取分析出50个新的特征，再利用线性回归进行分析，求得回归系数，最后根据这个系数进行判断。
2项目背景
中国领先的综合性旅行服务公司，每天向超过2。5亿会员提供全方位的旅行服务，在这海量的网站访问量种，可分析用户的行为数据来挖掘潜在的信息资源
其中，客户流失率是考虑业务成绩的一个非常关键的指标。此次，分析的目的是为来深入了解使用者画像及行为偏好，找到最优算法，挖掘出影响用户流失的关键因素
从而更好地完善产品设计、提升用户体验
经由大数据分析，可以更加准确地了解用户需要什么，这样可以提升用户的入住意愿。随着时代的发展，用户对酒店对要求也越来越高，
因此要用数据分析用户不满的原因，比如用户是因为不满意服务或是价格从而选择来其他公司的产品。掌握到这些信息后就能更加有效地开发新用户。
由于历史数据种可以得知用户对房间价格、房间格局、入住时间段等偏好特征，可以给予每位用户最精准对信息和服务，通过数据分析可以紧紧地抓住每一位用户的心
3数据说明
tablel表结构说明
 0   label                             689945 non-null  int64   用户是否流失
 1   sampleid                          689945 non-null  int64   样本ID
 2   d                                 689945 non-null  object  访问时间
 3   arrival                           689945 non-null  object  入住时间
 4   iforderpv_24h                     689945 non-null  int64   24小时内是否询问订单填写
 5   decisionhabit_user                385450 non-null  float64 用户行为类型(决策习惯)
 6   historyvisit_7ordernum            82915 non-null   float64 近7天用户历史订单数
 7   historyvisit_totalordernum        386525 non-null  float64 近一年用户历史订单数
 8   hotelcr                           689148 non-null  float64 当前酒店历史流动率
 9   ordercanceledprecent              447831 non-null  float64 用户一年内取消订单率
 10  landhalfhours                     661312 non-null  float64 24小时登录时长
 11  ordercanncelednum                 447831 non-null  float64 用户一年内取消订单数
 12  commentnums                       622029 non-null  float64 当前酒店点评数
 13  starprefer                        464892 non-null  float64 星级偏好
 14  novoters                          672918 non-null  float64 当前酒店评分人数
 15  consuming_capacity                463837 non-null  float64 消费能力指数
 16  historyvisit_avghotelnum          387876 non-null  float64 酒店对平均历史访客数
 17  cancelrate                        678227 non-null  float64 当前酒店历史取消率
 18  historyvisit_visit_detailpagenum  307234 non-null  float64 酒店详情页对访客数
 19  delta_price1                      437146 non-null  float64 用户偏好价格
 20  price_sensitive                   463837 non-null  float64 价格敏感指数
 21  hoteluv                           689148 non-null  float64 当前酒店历史UV
 22  businessrate_pre                  483896 non-null  float64 24小时内历史浏览次数最多对酒店的商务属性指数
 23  ordernum_oneyear                  447831 non-null  float64 用户一年内订单数
 24  cr_pre                            660548 non-null  float64 24小时历史浏览次数最多的酒店的历史流动率
 25  avgprice                          457261 non-null  float64 平均价格
 26  lowestprice                       687931 non-null  float64 当前酒店可订最低价格
 27  firstorder_bu                     376993 non-null  float64 首个订单
 28  customereval_pre2                 661312 non-null  float64 24小时历史浏览酒店客户评分均值
 29  delta_price2                      437750 non-null  float64 用户偏好价格，算法：近24小时内浏览酒店的平均价格
 30  commentnums_pre                   598368 non-null  float64 24小时内历史浏览次数最多的酒店的点评数
 31  customer_value_profit             439123 non-null  float64 近一年的用户价值
 32  commentnums_pre2                  648457 non-null  float64 24小时内历史浏览酒店并点评的次数均值
 33  cancelrate_pre                    653015 non-null  float64 24小时内访问次数最多的酒店的历史取消率
 34  novoters_pre2                     657616 non-null  float64 24小时内历史浏览酒店评分数均值
 35  novoters_pre                      648956 non-null  float64 24小时内历史浏览次数最多的酒店的评分数
 36  ctrip_profits                     445187 non-null  float64 客户价值
 37  deltaprice_pre2_t1                543180 non-null  float64 24小时内访问酒店价格与对手价差均值T+1
 38  lowestprice_pre                   659689 non-null  float64 20小时内最低的价格
 39  uv_pre                            660548 non-null  float64 24小时内历史浏览次数最多的酒店历史UV
 40  uv_pre2                           661189 non-null  float64 24小时内历史浏览次数最多的酒店历史UV均值
 41  lowestprice_pre2                  660664 non-null  float64 24小时内访问次数最多的酒店的可订最低价
 42  lasthtlordergap                   447831 non-null  float64 一年内距离上次下单时长
 43  businessrate_pre2                 602960 non-null  float64 24小时内访问酒店的商务属性均值
 44  cityuvs                           682274 non-null  float64 城市的访客数量
 45  cityorders                        651263 non-null  float64 城市的订单量
 46  lastpvgap                         592818 non-null  float64 最终PV的差值
 47  cr                                457896 non-null  float64 流动率
 48  sid                               689945 non-null  int64   唯一身份编码
 49  visitnum_oneyear                  592910 non-null  float64 年访问次数
 50  h                                 689945 non-null  int64   访问时间点
4项目实现思路
分析流程分为两个阶段，分别如下。
(1)准备阶段
1导入与导出数据表
2客户基本数据分析处理、缺失值填补
3询问与入住日期的转换，产生新特征的分析与数据清理
4缺失值处理与归一化，新特征的分析与产生、缺失值调整、极值调整
(2)数据挖掘阶段
1将特征与目标数据表进行合并的动作，产生新的数据集用于数据挖掘
2以GBM、XGBoost为例对客户流失概率进行预测"""

#1导入与导出数据
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.cluster import KMeans
from pandas import MultiIndex
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')  #忽略警告设置


table = pd.read_csv('rawdata/table1.csv')
#获取列信息
F1_1 = table[['label', 'sampleid', 'historyvisit_7ordernum', 'historyvisit_totalordernum', 'ordercanceledprecent', 'ordercanncelednum',
              'historyvisit_avghotelnum', 'delta_price1', 'businessrate_pre', 'cr_pre', 'landhalfhours', 'starprefer', 'price_sensitive',
              'commentnums_pre2', 'cancelrate_pre', 'novoters_pre2', 'novoters_pre', 'commentnums_pre2', 'cancelrate_pre', 'lowestprice_pre',
              'uv_pre', 'uv_pre2', 'hoteluv', 'cancelrate', 'novoters', 'commentnums', 'hotelcr', 'visitnum_oneyear', 'ordernum_oneyear',
              'cityorders', 'iforderpv_24h', 'consuming_capacity', 'avgprice', 'ctrip_profits', 'customer_value_profit', 'commentnums_pre',
              'delta_price2', 'ordernum_oneyear', 'firstorder_bu', 'd', 'arrival']]
#重命名列名
F1_1.columns = ['label', 'ID', 'F1.1', 'F1.2', 'F1.3', 'F1.4', 'F1.5', 'F1.6', 'F1.7', 'F1.8', 'F1.9', 'F1.10', 'F1.11', 'F1.12', 'F1.13', 'F1.14',
                'F1.15', 'F1.16', 'F1.17', 'F1.18', 'F1.19', 'F1.20', 'F1.21', 'F1.22', 'F1.23', 'F1.24', 'F1.25', 'F1.26', 'F1.27', 'F1.28', 'F1.29',
                'F1.30', 'F1.31', 'F1.32', 'F1.33', 'F1.34', 'F1.35', 'F1.36', 'F1.37', 'F1.38', 'F1.39']
#空值替换为NA
F1_1 = F1_1.fillna('NA')
F1_1.to_csv('workeddata/F1_1.csv', index=False, encoding="utf_8_sig")
#table.info()
#print(F1_1)


#2客户基本数据分析处理、缺失值填补

F1_1 = pd.read_csv('workeddata/F1_1.csv')
#读取列
F1_2_1 = F1_1[['ID', 'F1.1', 'F1.2', 'F1.4', 'F1.5', 'F1.9', 'F1.15', 'F1.23', 'F1.24', 'F1.27', 'F1.35', 'F1.38', 'F1.39']]
#空值替换为0
F1_2_1 = F1_2_1.fillna(0)
#读取列
F1_2_2 = F1_1[['label', 'ID']]
#设置所需列的空值替换为均值
title = ['F1.3', 'F1.6', 'F1.7', 'F1.8', 'F1.10', 'F1.11', 'F1.12', 'F1.13', 'F1.14', 'F1.16', 'F1.17', 'F1.18', 'F1.19',
         'F1.20', 'F1.21', 'F1.22', 'F1.25', 'F1.26', 'F1.28', 'F1.29', 'F1.30', 'F1.31', 'F1.32', 'F1.33', 'F1.34', 'F1.36',
         'F1.37']
for t in title:
    #获取每一列的均值
    mean = F1_1[[t]].mean().values[0]
    #获取列
    null = F1_1[['ID', t]]
    #空值替换为均值
    null = null.fillna(mean)
    #根据用户ID合并特征
    F1_2_2 = F1_2_2.join(null.set_index('ID'), on='ID')
#根据用户ID合并表格
F1_2 = F1_2_1.join(F1_2_2.set_index('ID'), on='ID')
#计算所需列值除以该列的均值，结果替换该列的值
for t in list(F1_2):
    #跳过以下几列
    if t == 'ID' or t == 'label' or t == 'F1.38' or t == 'F1.39':
        continue #跳出循环，执行后面的
    mean = F1_2[[t]].mean().values[0]
    #列值/均值，然后赋值到原有列
    F1_2[t] = F1_2[t]/mean

F1_2 = F1_2[['label', 'ID', 'F1.1', 'F1.2', 'F1.3', 'F1.4', 'F1.5', 'F1.6', 'F1.7', 'F1.8', 'F1.9', 'F1.10',
             'F1.11', 'F1.12', 'F1.13', 'F1.14', 'F1.15', 'F1.16', 'F1.17', 'F1.18', 'F1.19', 'F1.20', 'F1.21',
             'F1.22', 'F1.23', 'F1.24', 'F1.25', 'F1.26', 'F1.27', 'F1.28', 'F1.29', 'F1.30', 'F1.31', 'F1.32',
             'F1.33', 'F1.34', 'F1.35', 'F1.36', 'F1.37', 'F1.38', 'F1.39']]
#为相对路径
F1_2.to_csv('workeddata/F1_2.csv', index=False, encoding="utf_8_sig")
#print(F1_2)


#3询问与入住日期的装换，产生新特征的分析与数据清理

F1_2 = pd.read_csv('workeddata/F1_2.csv')
# 将日期作转换，算出入住日期与询问日期相差几天
F1_2['F1.40'] = pd.to_datetime(F1_2['F1.39']) - pd.to_datetime(F1_2['F1.38'])
#F1.40的值：0 days, 将格式转为str
F1_2['F1.40'] = F1_2['F1.40'].astype('str')
#获取数值
F1_2['F1.40'] = F1_2['F1.40'].apply(lambda x: x.split(' ')[0])
# 将实际日期中的假日(周六、周日)转化为1，其余转化成0
#将原先的格式转为datetime
F1_2['F1.41'] = pd.to_datetime(F1_2['F1.38'])
F1_2['F1.42'] = pd.to_datetime(F1_2['F1.39'])
#将日期转为星期几
F1_2['F1.41'] = F1_2['F1.41'].dt.dayofweek
F1_2['F1.42'] = F1_2['F1.42'].dt.dayofweek
#将周一到周五的数值替换为0
F1_2.loc[F1_2['F1.41'] <= 5, 'F1.41'] = 0
F1_2.loc[F1_2['F1.42'] <= 5, 'F1.42'] = 0
#将周末的数值替换为1
F1_2.loc[F1_2['F1.41'] > 5, 'F1.41'] = 1
F1_2.loc[F1_2['F1.42'] > 5, 'F1.42'] = 1

#4缺失值处理
# 二次产生的变数
F1_2['F1.43'] = F1_2['F1.26']/F1_2['F1.27']
F1_2['F1.44'] = F1_2['F1.32']/F1_2['F1.33']
F1_2['F1.45'] = F1_2['F1.24']/F1_2['F1.23']

# 空值取代
F1_2 = F1_2.fillna(0)
# 二次产生的变数
F1_2['F1.46'] = F1_2['F1.34']/F1_2['F1.15']


mean = F1_2[['F1.46']].mean().values[0]
#空值替换为均值
F1_2 = F1_2.fillna(mean)
#print(F1_2)

#用聚类算法产生新特征，并把结果写入表中
# 设置要进行聚类的字段
loan1 = np.array(F1_2[['F1.1', 'F1.2', 'F1.3', 'F1.4', 'F1.5', 'F1.6', 'F1.7']])
# 将用户分成3类
clf1 = KMeans(n_clusters=3)
# 将数据代入到聚类模型中
clf1 = clf1.fit(loan1)
# 在原始数据表中增加聚类结果标签
F1_2['F1.47'] = clf1.labels_

# 设置要进行聚类的字段
loan2 = np.array(F1_2[['F1.21', 'F1.22', 'F1.23', 'F1.24', 'F1.25']])
# 将用户分成3类 设置类别为3
clf2 = KMeans(n_clusters=3)
# 将数据代入到聚类模型中
clf2 = clf2.fit(loan2)
# 在原始数据表中增加聚类结果标签
F1_2['F1.48'] = clf2.labels_

table_database = F1_2
table_database.to_csv('workeddata/table_database.csv', index=False, encoding="utf_8_sig")
#print(F1_2)


#数据挖掘

#读取文件
table_database = pd.read_csv('workeddata/table_database.csv')
#名义特征设定。大部分名义特征在读取时会被转变为数值特征，为此，要将这些特征转换为名义特征
table_database['F1.47'] = pd.factorize(table_database['F1.47'])[0].astype(np.uint16)
table_database['F1.48'] = pd.factorize(table_database['F1.48'])[0].astype(np.uint16)
#astype指定数据类型；pd.factorize()做的也是“因式分解”，把常见的字符型变量分解为数字
#factorize(train['F2.19']),将指定的列，按列值的多少种取值，进行编码(目的让模型识别)
#装换后是一个array元组；这个元组包含两个array,分别是我们想要的数字，以及原来的index(即本身未编码之前的值)
#因为我们只要前一列，即数字。所以为factorize(train['F2.19'])[0]

#删除列
table_database = table_database.drop(['F1.38', 'F1.39'], axis=1)

#替换数据集中的inf与nan，并替换为所在列的均值（平均时分母不计inf nan数量）
#注意：在数据集做运算，若其中一列为缺失值或0，就会出现inf、nan，会导致在划分数据集时报错
table_database_inf = table_database.mask(np.isinf, None) #只把元素是np.isinf，全部替换为指定值  mask：显示为假值，替换为真值，戴上面具看到的是假面
table_database_inf = table_database_inf.fillna(table_database_inf.apply('mean'))

#设为目标 所要划分的样本结果(测试集)
df_train = table_database_inf['label'].values
#删除列  所要划分的样本特征集(训练集)
train = table_database_inf.drop(['label'], axis=1)

# 随机抽取90%的数据作为训练数据，剩余10%作为测试资料
X_train, X_test, y_train, y_test = train_test_split(train, df_train, test_size=0.1, random_state=1)
#test_size：测试数据的比例
#random_state：是一个随机种子，是在任意带有随机性的类或函数里作为参数来控制随机模式。当random_state取某一个值时，也就确定了一种规则
# 使用XGBoost的原生版本需要对数据进行转化(封装训练和测试数据)

# 使用XGBoost的原生版本需要对数据进行转化
data_train = xgb.DMatrix(X_train, y_train) #构造训练集
data_test = xgb.DMatrix(X_test, y_test) #构造测试集
# 设置参数
# 以XGBoos训练。max.depth表示树的深度，eta表示权重参数，objective表示训练目标的学习函数
param = {'max_depth': 4, 'eta': 0.2, 'objective': 'reg:linear'}
watchlist = [(data_test, 'test'), (data_train, 'train')]
# 表示训练次数
n_round = 10
# 训练数据载入模型
data_train_booster = xgb.train(param, data_train, num_boost_round=n_round, evals=watchlist)
#params：字典类型，用于指定各种参数，例如：{‘booster’:‘gbtree’,‘eta’:0.1}
#data_train：用于训练的数据，通过给下面的方法传递数据和标签来构造
#num_boost_round：指定最大迭代次数，默认值为10
#evals：列表类型，用于指定训练过程中用于评估的数据及数据的名称。例如：[(dtrain,‘train’),(dval,‘val’)]


# 以XGBoost测试。分别对训练与测试数据进行测试，其中auc为分类器评价指标，其值越大，则分类器效果越好
# 计算错误率
y_predicted = data_train_booster.predict(data_train)
y = data_train.get_label()
accuracy = sum(y == (y_predicted > 0.5)) #用于设定阀值，概率大于0.5
accuracy_rate = float(accuracy) / len(y_predicted)
print('样本总数：{0}'.format(len(y_predicted)))
print('正确数目：{0}'.format(accuracy))
print('正确率：{0:.10f}'.format((accuracy_rate)))

# 使用F-measure评价测试
#将数组转为dataframe
y_train_f = pd.DataFrame(y_train)
y_predicted_f = pd.DataFrame(y_predicted)
#新建列，列值为索引值
y_train_f['index'] = y_train_f.index.values
y_predicted_f['index'] = y_predicted_f.index.values
#重命名为列名
y_train_f.columns = ['train', 'index']
y_predicted_f.columns = ['y_n', 'index']
#新建列，列值为0
y_predicted_f['test'] = 0
#当y_n列值大于0.5时，把test列当值替换为1
y_predicted_f.loc[y_predicted_f['y_n'] > 0.5, 'test'] = 1
#读取列
y_predicted_f = y_predicted_f[['test', 'index']]
#根据index合并表
F = y_train_f.join(y_predicted_f.set_index('index'), on='index')
#读取列
F = F[['train', 'test']]
#求train等于1和text等于1的数据量
tp = F[(F.train == 1) & (F.test == 1)].test.count()
#求train等于0和text等于1的数据量
fp = F[(F.train == 0) & (F.test == 1)].test.count()
#求train等于1和text等于0的数据量
fn = F[(F.train == 1) & (F.test == 0)].test.count()
#求train等于0和text等于0的数据量
tn = F[(F.train == 0) & (F.test == 0)].test.count()

# 对比两种方式的准确率，可以知道F-measure的方式较AUC效果来的差。
P = tp/(tp+fp)
R = tn/(tn+fn)
F1 = 2*P*R/(P+R)
print('F-measure值：{0:.10f}'.format(F1))
#F值介于[0, 1]说明模型比较有效








