"""
用户成单预测可以准确地提高用户的精准度，筛选出我们需要的用户数据，提高用户的转化率
1算法原理
成单预测是用户关系管理中一个部分，指在交易双方还未进行互动前，对双方是否能达成交易进行的预测。其主要的工作原理是，根据过去某个商品的数据资料的状况，
预测其现在状况。传统的成单预测主要着眼于一个或者几个相对较少的指标进行检定预测，由于指标相对较少，所得结果的效果好坏与指定的指标好坏有很大的关联，
因此相当考验预测者对指标的敏感度，且传统的预测方式所使用的数据资料数量有限，预测的精确程度也相对不稳定。
相传统的成单预测，基于大数据分析的成单预测更具有有时。首先，通过大数据分析的方法，我们可以选择大量的指标，将所有指标纳入一个体系中进行分析。
相对原本的逐个指标检定更有效率，也能在分析时自动调整指标的重要程度，节省时间。当然，人为地选择指标，依旧是一种提高分析效果与效率的好方法。
其次，我们同样也知道，预测分析结果的好坏与预测前我们分析过去资料的资料量有一定的关系，一般来说，分析的资料量越大，分析的效果越好，分析结果约准确。
而恰恰大数据分析可以分析大量的资料，只要我们的预测模型合理，那么其分析上限仅与分析设备的硬件有关。也就是说，硬件能力越好，速度就越快。
特别是随着大数据时代的到来，使这一优势更加明显。
本次项目是根据原始的用户数据和用户订单数据进行用户特征的清洗，由原先的16个原始数据清洗成154个特征数据，
再利用线性回归方法对特征进行分析，求得回归系数，最后根据这个系数进行判断
2项目背景
某旅游公司是一个为中国出境游用户提供全球中文包车游服务的平台
由于消费者消费能力逐渐增强、旅游信息不透明程度下降，游客的行为逐渐变得难以预测。传统旅行社的旅游模式已经不能满足游客需求，而对于企业来说，
传统旅游线路对其营业利润也越来越有限，公司希望提升服务品质，通过附加值更高的精品服务来改变目前遇到的困境。为此该该公司除了提供普通的旅行服务，
也提出了品质相对更好的精品旅行服务。但是，从公司角度触发的所谓精品服务，是否用户心中的精品服务，用户是否会为精品服务买单，这些问题就变得微妙了
回答这些问题的首要步骤就是找到"哪些用户会选择这些精品服务"也只有了解了这个问题的答案，才能对"精品服务"的推进进行更深入的了解与优化
于是，本项目的主题便是精品旅行服务成单预测，即希望通过分析用户的行为，了解不同用户的需求，对它们下一次是否购买精品服务进行预测
3数据说明
表零保存了待预测的订单数据，后续将主要围绕此表给出的数据，按照用户ID索引，进行用户是否会买单的预测分析工作
表一保存了用户个人信息，主要以个人生物属性为主，其中部分字段缺失比例较高，在分析过程中需要对其进行缺失值处理
表二保存了用户历史订单数据，详细记录了发生交易行为的用户订单相关信息，其信息为后续分析预测工作起到了至关重要的作用
表三保存了用户行为信息，详细记录了用户在使用APP过程中的相关信息，包括唤醒APP、浏览产品、填写表单等相关信息
4项目实现思路
整个分析的流程分为两个阶段，分别如下
(1)准备阶段，在用户基本资料、用户订单资料、用户APP行为资料中选取并变换出与目标相关的指标货特征，做出一系列的数值处理。具体包括以下6步。
一、导入与导出数据表
二、用户基本资料分析处理，主要是缺失值填补
三、用户订单资料分析处理，主要是新特征的分析与产生
四、用户APP行为的分析处理，主要是新特征的分析与产生、缺失值调整、极值调整
五、基于用户订单资料与用户APP行为的整合分析处理，强调基于已产生的特征进行再此特征发现
六、汇总所有特征，并处理缺失值
本项目有大量的特征需要从原始资料中提取，过程比较烦琐，可结合代码调试
(2)数据挖掘阶段，主要是技能型成单预测，要先将前整理汇总的特征与目标组合成能进行分析的格式，而后通过分析工具(分类器)对用户是否会购买服务进行预测
并将预测结果与实际结果进行比较，测试模型的准确程度。具体包括以下两步
一、将特含恶搞与目标数据表进行合并，产生新的数据集用于数据挖掘
二、以XGBoost为例对精品旅行服务成单进行预测
"""
#1数据准备
import pandas as pd
import math
import numpy as np
#读取原始数据
table_target = pd.read_csv('rawdata/table_0.csv')
#重命名列名，ID为用户id，target为预测结果
table_target.columns = ['ID', 'target']
#把数据写入table_target.csv文件
table_target.to_csv('workeddata/table_target.csv', index=False, encoding='utf_8_sig')
#print(table_target)

#2用户基本资料分析处理
F1 = pd.read_csv('rawdata/table_1.csv')
#把表格中对缺失值替换成'未知'，以此区别于其他特征
F1 = F1.fillna('未知')
# 重命名列名， F1.1:性别 , F1.2:省份 , F1.3:年龄段
F1.columns = ['ID', 'F1.1', 'F1.2', 'F1.3']
F1.to_csv('workeddata/F1.csv', index=False, encoding="utf_8_sig")
#print(F1)

#3用户订单资料分析处理
userOrder = pd.read_csv('rawdata/table_2.csv')
"""提取以下特征：F2.1:订单的数量  F2.2:是否为精品订单 F2.3精品订单的数量 F2.4精品订单的占比"""

#读取订单类型为1对数据，把订单类型为1对订单定义为精品订单 orderType为订单类型
JPorder = userOrder[userOrder.orderType == 1]
# F2.1:订单的数量， userid为用户ID，orderid为订单id
orderNum = userOrder[['userid', 'orderid']]
F2 = orderNum.groupby('userid', as_index=False).count()  #as_index=False时,表示为默认自然数字索引，不将主键设置为索引
#将F2表的字段重命名
F2.columns = ['ID', 'F2.1']

# F2.3 精品订单_个数 ,
orderType = JPorder[['userid']]  #读取精品订单用户ID
orderType['number'] = 1  #新建number列，列初始值为1
orderType = orderType.groupby('userid', as_index=False).sum() #根据用户ID分组求精品订单数
#根据用户id合并orderType和F2两张表
orderType = orderType.join(F2.set_index('ID'), on='userid')
# F2.4 精品订单_占比
orderType['F2.4'] = orderType['number']/orderType['F2.1']
# F2.2 精品订单_是否有,1表示精品订单，0表示没有精品订单
orderType['F2.2'] = 1
F2_2_3_4 = orderType[['userid', 'F2.2', 'number', 'F2.4']]
F2_2_3_4.columns = ['ID', 'F2.2', 'F2.3', 'F2.4'] #重命名列名
#把F2表和F2_2_3_4表合并
F2 = F2.join(F2_2_3_4.set_index('ID'), on='ID')
#print(F2)

#从userOder表清洗出以下新增字段
#F2.5:旅游最多城市次数
#F2.6:旅游城市数
#F2.7:旅游最多国家次数
#F2.8:旅游国家数
#F2.9:旅游最多大洲次数
#F2.10:旅游大洲数
#创建分组，分别为城市(city)、国家(country)、和大洲(continent）
site = ['city', 'country', 'continent']
a = 5
for i in range(0, 3):
    #设定列名
    title1 = 'F2.' + str(a)
    title2 = 'F2.' + str(a+1)
    #读取读取全部订单的userid,city,country,continent字段到siteinfo表
    siteinfo = userOrder[['userid', site[i]]]
    #新建number列，列值为1
    siteinfo['number'] = 1
    #根据userid和city,country,continent分组，求每个用户去过每个城市/国家/大洲的次数汇总求和
    siteinfo = siteinfo.groupby(['userid', site[i]], as_index=False).sum()
    #根据userid分组，求用户去过最多城市/国家/大洲的次数
    siteinfo1 = siteinfo.groupby('userid', as_index=False).max()
    siteinfo1 = siteinfo1[['userid', 'number']]
    #重命名列名
    siteinfo1.columns = ['ID', title1]
    #根据userid分组，求用户去过城市/国家/大洲的次数的计数
    siteinfo2 = siteinfo.groupby('userid', as_index=False).count()
    siteinfo2 = siteinfo2[['userid', site[i]]]
    #重命名列名
    siteinfo2.columns = ['ID', title2]
    #根据ID合并特征
    F2 = F2.join(siteinfo1.set_index('ID'), on='ID')
    F2 = F2.join(siteinfo2.set_index('ID'), on='ID')
    a = a + 2

#print(F2)

#从JPorder表清洗出以下新增字段
#F2.11:精品旅游最多城市次数
#F2.12:精品旅游城市数
#F2.13:精品旅游最多国家次数
#F2.14:精品旅游国家数
#F2.15:精品旅游最多大洲次数
#F2.16:精品旅游大洲数

a = 11
for i in range(0, 3):
    # 设定列名
    title1 = 'F2.' + str(a)
    title2 = 'F2.' + str(a + 1)
    # 读取userid,city,country,continent字段到siteinfo表
    JPsiteinfo = JPorder[['userid', site[i]]]
    # 新建number列，列值为1
    JPsiteinfo['number'] = 1
    # 根据userid和city,country,continent分组，求每个用户去过每个城市/国家/大洲的次数汇总求和
    JPsiteinfo = JPsiteinfo.groupby(['userid', site[i]], as_index=False).sum()
    # 根据userid分组，求用户去过最多城市/国家/大洲的次数
    JPsiteinfo1 = JPsiteinfo.groupby('userid', as_index=False).max()
    JPsiteinfo1 = JPsiteinfo1[['userid', 'number']]
    # 重命名列名
    JPsiteinfo1.columns = ['ID', title1]
    # 根据userid分组，求用户去过城市/国家/大洲的次数的计数
    JPsiteinfo2 = JPsiteinfo.groupby('userid', as_index=False).count()
    JPsiteinfo2 = JPsiteinfo2[['userid', site[i]]]
    # 重命名列名
    JPsiteinfo2.columns = ['ID', title2]
    # 根据ID合并特征
    F2 = F2.join(JPsiteinfo1.set_index('ID'), on='ID')
    F2 = F2.join(JPsiteinfo2.set_index('ID'), on='ID')
    a = a + 2

#print(F2)

#清洗出订单的时间间隔，并命名为F2.17   orderTime为订单时间
period = userOrder.orderTime.max() - userOrder.orderTime.min()
F2['F2.17'] = period/F2['F2.1']    #订单的平均时间间隔

#清洗出精品订单的时间间隔，并命名为F2.18
JPperiod = JPorder.orderTime.max() - JPorder.orderTime.min()
F2['F2.18'] = JPperiod/F2['F2.3']  #精品订单的平均时间间隔
#print(F2)

#从userOder表清洗出以下新增字段
# F2.19 订单_热门城市_是否访问
# F2.20 订单_热门城市_访问城市数
# F2.21 订单_热门城市_访问次数
# F2.22 订单_热门国家_是否访问
# F2.23 订单_热门国家_访问国家数
# F2.24 订单_热门国家_访问次数
# F2.25 订单_热门大洲_是否访问
# F2.26 订单_热门大洲_访问大洲数
# F2.27 订单_热门大洲_访问次数

a = 19
for i in range(0, 3):
    title1 = 'F2.' + str(a)
    title2 = 'F2.' + str(a + 1)
    title3 = 'F2.' + str(a + 2)
    # 读取全部订单的userid,city,country,continent字段到siteinfo表
    siteinfo = userOrder[['userid', site[i]]]
    #根据city/country/continent分组，求城市/国家/大洲的订单数(计数)
    topsite = siteinfo.groupby(site[i], as_index=False).count()
    #获取前20%的热门城市/国家/大洲的信息
    topsite = topsite.sort_values('userid', ascending=False).head(math.floor((len(topsite) * 0.2)))
    #math.floor()方法是数学模块的库方法，用于获取给定数字的下限值，用于获取数字的下限值，它接受数字/数值表达式并返回最大整数(整数)值，该值不大于数字。
    #获取热门城市/国家/大洲
    topsite = topsite[[site[i]]]
    #获取去过热门城市/国家/大洲全部订单信息
    topsiteOrder = topsite.join(siteinfo.set_index(site[i]), on=site[i])
    #新建number列，列值为1
    topsiteOrder['number'] = 1
    #根据userid和city,country,continent分组，求每个用户去过的每个城市/国家/大洲的次数(汇总求和)
    topsiteOrder1 = topsiteOrder.groupby(['userid', site[i]], as_index=False).sum()
    # 根据userid分组，求每个用户去过的热门城市/国家/大洲数(计数)
    topsiteOrder1 = topsiteOrder1.groupby('userid', as_index=False).count()
    topsiteOrder1 = topsiteOrder1[['userid', site[i]]]
    #重命名列名
    topsiteOrder1.columns = ['ID', title2]
    #新建列是否访问过热门城市/国家/大洲，1为访问过，0为没有访问过
    topsiteOrder1[title1] = 1
    #根据userid分组，求每个用户去过的热门城市/国家/大洲的次数(汇总求和)
    topsiteOrder2 = topsiteOrder.groupby('userid', as_index=False).sum()
    #重命名列名
    topsiteOrder2.columns = ['ID', title3]
    #根据ID合并特征
    F2 = F2.join(topsiteOrder1.set_index('ID'), on='ID')
    F2 = F2.join(topsiteOrder2.set_index('ID'), on='ID')
    a = a + 3

#print(F2)


#从JPoder表清洗出以下新增字段
# F2.28 精品订单_热门城市_是否访问
# F2.29 精品订单_热门城市_访问城市数
# F2.30 精品订单_热门城市_访问次数
# F2.31 精品订单_热门国家_是否访问
# F2.32 精品订单_热门国家_访问国家数
# F2.33 精品订单_热门国家_访问次数
# F2.34 精品订单_热门大洲_是否访问
# F2.35 精品订单_热门大洲_访问大洲数
# F2.36 精品订单_热门大洲_访问次数

a = 28
for i in range(0, 3):
    title1 = 'F2.' + str(a)
    title2 = 'F2.' + str(a + 1)
    title3 = 'F2.' + str(a + 2)
    # 读取全部订单的userid,city,country,continent字段到siteinfo表
    JPsiteinfo = JPorder[['userid', site[i]]]
    #根据city/country/continent分组，求城市/国家/大洲的订单数(计数)
    JPtopsite = JPsiteinfo.groupby(site[i], as_index=False).count()
    #获取前20%的热门城市/国家/大洲的信息
    JPtopsite = JPtopsite.sort_values('userid', ascending=False).head(math.floor((len(JPtopsite) * 0.2)))
    #math.floor()方法是数学模块的库方法，用于获取给定数字的下限值，用于获取数字的下限值，它接受数字/数值表达式并返回最大整数(整数)值，该值不大于数字。
    #获取热门城市/国家/大洲
    JPtopsite = JPtopsite[[site[i]]]
    #获取去过热门城市/国家/大洲全部订单信息
    JPtopsiteOrder = JPtopsite.join(JPsiteinfo.set_index(site[i]), on=site[i])
    #新建number列，列值为1
    JPtopsiteOrder['number'] = 1
    #根据userid和city,country,continent分组，求每个用户去过的每个城市/国家/大洲的次数(汇总求和)
    JPtopsiteOrder1 = JPtopsiteOrder.groupby(['userid', site[i]], as_index=False).sum()
    # 根据userid分组，求每个用户去过的热门城市/国家/大洲数(计数)
    JPtopsiteOrder1 = JPtopsiteOrder1.groupby('userid', as_index=False).count()
    JPtopsiteOrder1 = JPtopsiteOrder1[['userid', site[i]]]
    #重命名列名
    JPtopsiteOrder1.columns = ['ID', title2]
    #新建列是否访问过热门城市/国家/大洲，1为访问过，0为没有访问过
    JPtopsiteOrder1[title1] = 1
    #根据userid分组，求每个用户去过的热门城市/国家/大洲的次数(汇总求和)
    JPtopsiteOrder2 = JPtopsiteOrder.groupby('userid', as_index=False).sum()
    #重命名列名
    JPtopsiteOrder2.columns = ['ID', title3]
    #根据ID合并特征
    F2 = F2.join(JPtopsiteOrder1.set_index('ID'), on='ID')
    F2 = F2.join(JPtopsiteOrder2.set_index('ID'), on='ID')
    a = a + 3

#print(F2)
#将全部空值替换为0
F2 = F2.fillna(0)
F2 = F2[['ID', 'F2.1', 'F2.2', 'F2.3', 'F2.4', 'F2.5', 'F2.6', 'F2.7', 'F2.8', 'F2.9', 'F2.10',
    'F2.11', 'F2.12', 'F2.13', 'F2.14', 'F2.15', 'F2.16', 'F2.17', 'F2.18', 'F2.19', 'F2.20',
    'F2.21', 'F2.22', 'F2.23', 'F2.24', 'F2.25', 'F2.26', 'F2.27', 'F2.28', 'F2.29', 'F2.30',
    'F2.31', 'F2.32', 'F2.33', 'F2.34', 'F2.35', 'F2.36']]
#取出F2表中所需的特征字段，并写入文件中
F2.to_csv('workeddata/F2.csv', index=False, encoding="utf_8_sig")
#print(F2)

#4用户APP行为的分析处理
"""这里的数据必须要进行一个摊平的动作，摊平的指标使用行为类型，因为从table_3表actionType字段(行为类型)了解到行为类型一共有9个，其中1是唤醒APP；
2～4是浏览产品，无先后关系；5～9则是有先后关系的，从填写表单到提交订单再到支付。因此，可以先摊平，然后根据摊平后的部分，做出特征变换的处理"""

userAction = pd.read_csv('rawdata/table_3.csv')
# F3.1 所有动作_总次数
#根据userid分组求每个用户的动作次数(计数)
F3_1 = userAction.groupby('userid', as_index=False).count()
F3_1 = F3_1[['userid', 'actionType']]
F3_1.columns = ['ID', 'F3.1']  #重命名列名

# F3.2 非支付动作_次数
#筛选动作编号小于5的，再根据userid分组求每个用户的动作次数(计数)
F3_2 = userAction[userAction.actionType < 5].groupby('userid', as_index=False).count()
F3_2 = F3_2[['userid', 'actionType']]
F3_2.columns = ['ID', 'F3.2']

# F3.3 支付动作_次数
#筛选动作编号大于或等于5的，再根据userid分组求每个用户的动作次数(计数)
F3_3 = userAction[userAction.actionType >= 5].groupby('userid', as_index=False).count()
F3_3 = F3_3[['userid', 'actionType']]
F3_3.columns = ['ID', 'F3.3']

# 合并
F3 = F3_1.join(F3_2.set_index('ID'), on='ID')
F3 = F3.join(F3_3.set_index('ID'), on='ID')
#print(F3)

#从userAction表清洗出以下新增字段
# F3.4 动作1_次数
# F3.5 动作2_次数
# F3.6 动作3_次数
# F3.7 动作4_次数
# F3.8 动作5_次数
# F3.9 动作6_次数
# F3.10 动作7_次数
# F3.11 动作8_次数
# F3.12 动作9_次数
a1 = 4
for i in range(1, 10):
    #列名
    title1 = 'F3.' + str(a1)
    #获取每一个动作的信息，再根据userid分组，求每个用户每个动作的次数
    action1 = userAction[userAction.actionType == i].groupby('userid', as_index=False).count()
    action1 = action1[['userid', 'actionType']]
    #重命名列名
    action1.columns = ['ID', title1]
    F3 = F3.join(action1.set_index('ID'), on='ID') #合并特征
    a1 = a1 + 1
#0替换空值
F3 = F3.fillna(0)
#print(F3)

#从userAction表清洗出以下新增字段
# F3.13 非支付动作_占比
# F3.14 支付动作_占比
# F3.15 动作1_占比
# F3.16 动作2_占比
# F3.17 动作3_占比
# F3.18 动作4_占比
# F3.19 动作5_占比
# F3.20 动作6_占比
# F3.21 动作7_占比
# F3.22 动作8_占比
# F3.23 动作9_占比
a2 = 13
for i in range(2, 13):
    #设置列名
    title2 = 'F3.' + str(a2)
    actiontitle = 'F3.' + str(i)
    #求每种动作的占比
    F3[title2] = F3[actiontitle] / F3['F3.1']
    a2 = a2 + 1

#print(F3)

#使用diff(actionTime)函数计算时间间隔，然后计算出均值、方差、最小值、最大值
#读取userid和actionTime两列
timeinterval = userAction[['userid', 'actionTime']]

#跟俊userid分组，用diff函数计算出每一行actionTime与上一行的差值，结果赋值到新列interval
timeinterval['interval'] = timeinterval.groupby('userid').actionTime.diff()

#读取userid和interval两列
timeinterval1 = timeinterval[['userid', 'interval']]

# F3.24 时间间隔_均值
#根据userid分组，求均值
F3_24 = timeinterval1.groupby('userid', as_index=False).mean()
F3_24.columns = ['ID', 'F3.24']  #重命名列名
F3 = F3.join(F3_24.set_index('ID'), on='ID')

# F3.25 时间间隔_方差
#根据userid分组，求方差
F3_25 = timeinterval1.groupby('userid', as_index=False).var()
F3_25.columns = ['ID', 'F3.25']  #重命名列名
F3 = F3.join(F3_25.set_index('ID'), on='ID')

# F3.26 时间间隔_最小值
#根据userid分组，求最小值
F3_26 = timeinterval1.groupby('userid', as_index=False).min()
F3_26.columns = ['ID', 'F3.26']  #重命名列名
F3 = F3.join(F3_26.set_index('ID'), on='ID')

# F3.27 时间间隔_最大值
#根据userid分组，求最大值
F3_27 = timeinterval1.groupby('userid', as_index=False).max()
F3_27.columns = ['ID', 'F3.27']  #重命名列名
F3 = F3.join(F3_27.set_index('ID'), on='ID') #合并特征

#print(F3)
#获得最后3个时间的时间间隔与动作，可能有点客户没有3个动作(从填写表单到提交订单再到支付)，因此要对空值进行填补，填补值为该特征最大值
#根据actionTime 降序，再根据userid分组，获取前3条数据
top3time = timeinterval.sort_values('actionTime', ascending=False).groupby('userid', as_index=False).head(3)
#根据userid分组，求最大值
top3timemax = top3time.groupby('userid').max()
# F3.28 时间间隔_倒数第1个
#根据userid分组，获取第一条数据
F3_28 = top3time.groupby('userid', as_index=False).head(1)
F3_28 = F3_28[['userid', 'interval']]  #重命名列名
# 对空值进行填补，填补值为该特征最大值
F3_28null = F3_28.set_index('userid').isnull()  #isnull()判断缺失值，若该处为缺失值，返回True，该处不为缺失值，则返回False
F3_28null = F3_28null[F3_28null.interval == True]  #Interval类描述了一个连续的范围区间，这个区间可以是闭、开、半闭半开、无穷的，他的区间值不一定是数字，可以包含任何的数据类型，比如字符串，时间等等，同时他和python的各种操作(=, >等)也是兼容的
for i in F3_28null.index.values:
    max = top3timemax.at[i, "interval"]  #单元格选取(点选取)：df.at[]，df.iat[]。准确定位一个单元格。
    F3_28.loc[F3_28['userid'] == i, 'interval'] = max
F3_28.columns = ['ID', 'F3.28']
F3 = F3.join(F3_28.set_index('ID'), on='ID')   #合并特征

# F3.29 时间间隔_倒数第2个
F3_29 = top3time.groupby('userid', as_index=False).head(2)  #获取前2条
F3_29 = top3time.groupby('userid', as_index=False).tail(1)  #获取最后一条数据
F3_29 = F3_29[['userid', 'interval']]
# 对空值进行填补，填补值为该特征最大值
F3_29null = F3_29.set_index('userid').isnull()
F3_29null = F3_29null[F3_29null.interval == True]
for i in F3_29null.index.values:
    max = top3timemax.at[i, "interval"]
    F3_29.loc[F3_29['userid'] == i, 'interval'] = max
F3_29.columns = ['ID', 'F3.29']
F3 = F3.join(F3_29.set_index('ID'), on='ID')

# F3.30 时间间隔_倒数第3个
F3_30 = top3time.groupby('userid', as_index=False).tail(1)
F3_30 = F3_30[['userid', 'interval']]
# 对空值进行填补，填补值为该特征最大值
F3_30null = F3_30.set_index('userid').isnull()
F3_30null = F3_30null[F3_30null.interval == True]
for i in F3_30null.index.values:
    max = top3timemax.at[i, "interval"]
    F3_30.loc[F3_30['userid'] == i, 'interval'] = max
F3_30.columns = ['ID', 'F3.30']
F3 = F3.join(F3_30.set_index('ID'), on='ID')
#print(F3)

# 继续处理剩余特征
#根据actionTime 降序，再根据userid分组，获取前3条数据
top3action = userAction.sort_values('actionTime', ascending=False).groupby('userid', as_index=False).head(3)
#读取userid和actionType两列
top3actionmax = top3action[['userid', 'actionType']]  #actionType行为类型
top3actionmax = top3actionmax.groupby('userid').max()  #根据userid分组，求最大值
# F3.31 动作_倒数第1个
F3_31 = top3action.groupby('userid', as_index=False).head(1)
F3_31 = F3_31[['userid', 'actionType']]  #重新命名列名
# 对空值进行填补，填补值为该特征最大值
F3_31null = F3_31.set_index('userid').isnull()
F3_31null = F3_31null[F3_31null.actionType == True]
for i in F3_31null.index.values:
    max = top3actionmax.at[i, "actionType"]
    F3_31.loc[F3_31['userid'] == i, 'actionType'] = max
F3_31.columns = ['ID', 'F3.31']
F3 = F3.join(F3_31.set_index('ID'), on='ID')  #合并特征

# F3.32 动作_倒数第2个
F3_32 = top3action.groupby('userid', as_index=False).head(2)
F3_32 = top3action.groupby('userid', as_index=False).tail(1)
F3_32 = F3_32[['userid', 'actionType']]
# 填充空值
F3_32null = F3_32.set_index('userid').isnull()
F3_32null = F3_32null[F3_32null.actionType == True]
for i in F3_32null.index.values:
    max = top3actionmax.at[i, "actionType"]
    F3_32.loc[F3_32['userid'] == i, 'actionType'] = max
F3_32.columns = ['ID', 'F3.32']
F3 = F3.join(F3_32.set_index('ID'), on='ID')

# F3.33 动作_倒数第3个
F3_33 = top3action.groupby('userid', as_index=False).tail(1)
F3_33 = F3_33[['userid', 'actionType']]
# 对空值进行填补，填补值为该特征最大值
F3_33null = F3_33.set_index('userid').isnull()
F3_33null = F3_33null[F3_33null.actionType == True]
for i in F3_33null.index.values:
    max = top3actionmax.at[i, "actionType"]
    F3_33.loc[F3_33['userid'] == i, 'actionType'] = max
F3_33.columns = ['ID', 'F3.33']
F3 = F3.join(F3_33.set_index('ID'), on='ID')
#print(F3)

#继续处理剩余特征
# F3.34 时间间隔_倒数3个_均值
F3_34 = top3time[['userid', 'interval']].groupby('userid',as_index=False).mean()
F3_34.columns = ['ID', 'F3.34']
F3 = F3.join(F3_34.set_index('ID'), on='ID')

# F3.35 时间间隔_倒数3个_方差
F3_35 = top3time[['userid', 'interval']].groupby('userid', as_index=False).var()
F3_35.columns = ['ID', 'F3.35']
F3 = F3.join(F3_35.set_index('ID'), on='ID')
#print(F3)


"""
下面分析1～9每个动作对最后一次动作时间距离最后一个动作的时间间隔
首先计算出最后一个动作的时间，然后分别计算出每个动作的的最后一次的动作时间，再将两者相减，就可以得到想要的特征。
同样也要对空值进行填补，填补值为空值所在特征对最大值
"""
#根据actionTime降序，再根据userid分组，获取第一条数据（计算出最后一个动作的时间）
lastTime = userAction.sort_values('actionTime', ascending=False).groupby('userid', as_index=False).head(1)
#读取userid和actionTime两列
lastTime = lastTime[['userid', 'actionTime']]
lastTime.columns = ['userid', 'lastTime']#重命名列名
#根据actionTime降序，再根据userid和actionType分组，获取第一条数据（计算出每个动作的的最后一次的动作时间）
lastActionTime = userAction.sort_values('actionTime', ascending=False).groupby(['userid', 'actionType'], as_index=False).head(1)
lastActionTime.columns = ['userid', 'actionType', 'lastActionTime']  #重命名列名
actionType = lastActionTime
lastActionTime = lastActionTime.join(lastTime.set_index('userid'), on='userid') #合并两张表，指定userid
#计算每一个动作的最后一次动作时间与最后一次动作时间的差值
lastActionTime['diff'] = lastActionTime['lastTime'] - lastActionTime['lastActionTime']
#读取actionType和diff两列，再根据actionType分组，求最大值
lastActionTimemax = lastActionTime[['actionType', 'diff']].groupby('actionType').max()

# F3.36 时间间隔_最近动作1
# F3.37 时间间隔_最近动作2
# F3.38 时间间隔_最近动作3
# F3.39 时间间隔_最近动作4
# F3.40 时间间隔_最近动作5
# F3.41 时间间隔_最近动作6
# F3.42 时间间隔_最近动作7
# F3.43 时间间隔_最近动作8
# F3.44 时间间隔_最近动作9
a3 = 36
for i in range(1, 10):
    #列名
    title3 = 'F3.' + str(a3)
    #读取每一个动作的数据
    action3 = lastActionTime[lastActionTime.actionType == i]
    #读取userid和diff两列
    action3 = action3[['userid', 'diff']]
    #重命名列名
    action3.columns = ['ID', title3]
    #合并特征
    F3 = F3.join(action3.set_index('ID'), on='ID')
    a3 = a3 + 1
    # 对空值进行填补，填补值为该特征最大值
    action3null = F3[['ID', title3]]
    action3null = action3null.set_index('ID').isnull()
    action3null = action3null[action3null[title3] == True]
    for id in action3null.index.values:
        max = lastActionTimemax.at[i, "diff"]
        F3.loc[F3['ID'] == id, title3] = max
#print(F3)

"""通过上面代码知道了1～9每个动作对最后一次动作的时间，因此，只需要知道客户操作时间大于每个动作对最后一次动作时间的资料笔数，
就是动作距离，空值填补为每个特征最大值"""

# F3.45 动作距离_最近动作1
# F3.46 动作距离_最近动作2
# F3.47 动作距离_最近动作3
# F3.48 动作距离_最近动作4
# F3.49 动作距离_最近动作5
# F3.50 动作距离_最近动作6
# F3.51 动作距离_最近动作7
# F3.52 动作距离_最近动作8
# F3.53 动作距离_最近动作9
a4 = 45
for i in range(1, 10):
    title4 = 'F3.' + str(a4)
    #获取每个动作的数据
    Type = actionType[actionType.actionType == i]
    #读取userid和lastActionTime两列
    Type = Type[['userid', 'lastActionTime']]
    #根据userid合并userAction和Type两张表
    action4 = userAction.join(Type.set_index('userid'), on='userid')
    #获取actionTime大于等于lastActionTime的数据
    action4 = action4[action4.actionTime >= action4.lastActionTime]
    #根据userid分组，求每个用户actionTime大于等于lastActionTime的数据计数
    action4 = action4.groupby('userid', as_index=False).count()
    #读取userid和'actionType两列数据
    action4 = action4[['userid', 'actionType']]
    #根据actionType降序，获取第一条数据
    action4max = action4.sort_values('actionType', ascending=False).head(1)
    #重命名列名
    action4.columns = ['ID', title4]
    #合并特征
    F3 = F3.join(action4.set_index('ID'), on='ID')
    a4 = a4 + 1
    #获取该特征最大值
    max = action4max.get('actionType').values[0] #values() 方法，这个方法把dict转换成一个包含所有value的list，因前面按actionType降序，再values[0]即取第一个也就是最大值
    # 对空值进行填补，填补值为该特征最大值
    action4null = F3[['ID', title4]]
    action4null = action4null.set_index('ID').isnull()
    action4null = action4null[action4null[title4] == True]
    for id in action4null.index.values:
        F3.loc[F3['ID'] == id, title4] = max

#print(F3)
"""下面计算动作1～9时间间隔对均值、方差、最小值、最大值。首先筛选出相同动作的操作，然后按照userid进行分组，分别计算时间间隔，
之后筛选出大于0的时间间隔。最后分别以userid分组计算不同动作是按间隔的均值、方差、最小值、最大值"""

# 3-54 时间间隔_动作1_均值
# 3-55 时间间隔_动作1_方差
# 3-56 时间间隔_动作1_最小值
# 3-57 时间间隔_动作1_最大值
# 3-58 时间间隔_动作2_均值
# 3-59 时间间隔_动作2_方差
# 3-60 时间间隔_动作2_最小值
# 3-61 时间间隔_动作2_最大值
# 3-62 时间间隔_动作3_均值
# 3-63 时间间隔_动作3_方差
# 3-64 时间间隔_动作3_最小值
# 3-65 时间间隔_动作3_最大值
# 3-66 时间间隔_动作4_均值
# 3-67 时间间隔_动作4_方差
# 3-68 时间间隔_动作4_最小值
# 3-69 时间间隔_动作4_最大值
# 3-70 时间间隔_动作5_均值
# 3-71 时间间隔_动作5_方差
# 3-72 时间间隔_动作5_最小值
# 3-73 时间间隔_动作5_最大值
# 3-74 时间间隔_动作6_均值
# 3-75 时间间隔_动作6_方差
# 3-76 时间间隔_动作6_最小值
# 3-77 时间间隔_动作6_最大值
# 3-78 时间间隔_动作7_均值
# 3-79 时间间隔_动作7_方差
# 3-80 时间间隔_动作7_最小值
# 3-81 时间间隔_动作7_最大值
# 3-82 时间间隔_动作8_均值
# 3-83 时间间隔_动作8_方差
# 3-84 时间间隔_动作8_最小值
# 3-85 时间间隔_动作8_最大值
# 3-86 时间间隔_动作9_均值
# 3-87 时间间隔_动作9_方差
# 3-88 时间间隔_动作9_最小值
# 3-89 时间间隔_动作9_最大值

#读取userid、'actionType、actionTime三列
timeinterval2 = userAction[['userid', 'actionType', 'actionTime']]
#根据'userid', 'actionType'分组，获取actionTime列每一行值与上一行的差值，并赋值到新列interval
timeinterval2['interval'] = timeinterval2.groupby(['userid', 'actionType']).actionTime.diff()
a5 = 54
for i in range(1, 10):
    #列名
    actionMeanTitle = 'F3.' + str(a5)
    actionVarTitle = 'F3.' + str(a5 + 1)
    actionMinTitle = 'F3.' + str(a5 + 2)
    actionMaxTitle = 'F3.' + str(a5 + 3)
    #读取每个动作的数据
    actionType = timeinterval2[timeinterval2.actionType == i]
    #读取'userid', 'interval'两列
    actionType = actionType[['userid', 'interval']]
    #根据userid分组，求均值
    actionMean = actionType.groupby('userid', as_index=False).mean()
    #重命名列
    actionMean.columns = ['ID', actionMeanTitle]
    # 根据userid分组，求方差
    actionVar = actionType.groupby('userid', as_index=False).var()
    actionVar.columns = ['ID', actionVarTitle]
    #根据userid分组，求最小值
    actionMin = actionType.groupby('userid', as_index=False).min()
    actionMin.columns = ['ID', actionMinTitle]
    # 根据userid分组，求最大值
    actionMax = actionType.groupby('userid', as_index=False).max()
    actionMax.columns = ['ID', actionMaxTitle]
    #合并特征
    F3 = F3.join(actionMean.set_index('ID'), on='ID')
    F3 = F3.join(actionVar.set_index('ID'), on='ID')
    F3 = F3.join(actionMin.set_index('ID'), on='ID')
    F3 = F3.join(actionMax.set_index('ID'), on='ID')
    a5 = a5 + 4

#将NA替换空值
F3 = F3.fillna('NA')
#把数据写入到F3.csv,路径为相对路径
F3.to_csv('workeddata/F3.csv', index=False, encoding="utf_8_sig")
#print(F3)

#5基于用户订单资料与用户APP行为的整合分析处理

F2 = pd.read_csv('workeddata/F2.csv')
F3 = pd.read_csv('workeddata/F3.csv')
F23 = F2.join(F3.set_index('ID'), on='ID')

# F2.3.1 所有动作_订单_占比
# F2.3.2 非支付动作_订单_占比
# F2.3.3 支付动作_订单_占比
# F2.3.4 动作1_订单_占比
# F2.3.5 动作2_订单_占比
# F2.3.6 动作3_订单_占比
# F2.3.7 动作4_订单_占比
# F2.3.8 动作5_订单_占比
# F2.3.9 动作6_订单_占比
# F2.3.10 动作7_订单_占比
# F2.3.11 动作8_订单_占比
# F2.3.12 动作9_订单_占比
# F2.3.13 所有动作_精品订单_占比
# F2.3.14 非支付动作_精品订单_占比
# F2.3.15 支付动作_精品订单_占比
# F2.3.16 动作1_精品订单_占比
# F2.3.17 动作2_精品订单_占比
# F2.3.18 动作3_精品订单_占比
# F2.3.19 动作4_精品订单_占比
# F2.3.20 动作5_精品订单_占比
# F2.3.21 动作6_精品订单_占比
# F2.3.22 动作7_精品订单_占比
# F2.3.23 动作8_精品订单_占比
# F2.3.24 动作9_精品订单_占比
F23['F2.3.1'] = F23['F3.1'] / F23['F2.1']
F23['F2.3.2'] = F23['F3.2'] / F23['F2.1']
F23['F2.3.3'] = F23['F3.3'] / F23['F2.1']
F23['F2.3.4'] = F23['F3.4'] / F23['F2.1']
F23['F2.3.5'] = F23['F3.5'] / F23['F2.1']
F23['F2.3.6'] = F23['F3.6'] / F23['F2.1']
F23['F2.3.7'] = F23['F3.7'] / F23['F2.1']
F23['F2.3.8'] = F23['F3.8'] / F23['F2.1']
F23['F2.3.9'] = F23['F3.9'] / F23['F2.1']
F23['F2.3.10'] = F23['F3.10'] / F23['F2.1']
F23['F2.3.11'] = F23['F3.11'] / F23['F2.1']
F23['F2.3.12'] = F23['F3.12'] / F23['F2.1']
F23['F2.3.13'] = F23['F3.1'] / F23['F2.3']
F23['F2.3.14'] = F23['F3.2'] / F23['F2.3']
F23['F2.3.15'] = F23['F3.3'] / F23['F2.3']
F23['F2.3.16'] = F23['F3.4'] / F23['F2.3']
F23['F2.3.17'] = F23['F3.5'] / F23['F2.3']
F23['F2.3.18'] = F23['F3.6'] / F23['F2.3']
F23['F2.3.19'] = F23['F3.7'] / F23['F2.3']
F23['F2.3.20'] = F23['F3.8'] / F23['F2.3']
F23['F2.3.21'] = F23['F3.9'] / F23['F2.3']
F23['F2.3.22'] = F23['F3.10'] / F23['F2.3']
F23['F2.3.23'] = F23['F3.11'] / F23['F2.3']
F23['F2.3.24'] = F23['F3.12'] / F23['F2.3']
F23 = F23[['ID', 'F2.3.1', 'F2.3.2', 'F2.3.3', 'F2.3.4', 'F2.3.5', 'F2.3.6', 'F2.3.7', 'F2.3.8', 'F2.3.9', 'F2.3.10',
           'F2.3.11', 'F2.3.12', 'F2.3.13', 'F2.3.14', 'F2.3.15', 'F2.3.16', 'F2.3.17', 'F2.3.18', 'F2.3.19', 'F2.3.20',
           'F2.3.21', 'F2.3.22', 'F2.3.23', 'F2.3.24']]
#把空值替换为0
F23 = F23.fillna(0)
#把无穷大和无穷小替换为空
F23 = F23.replace([np.inf, -np.inf], np.nan)
F23 = F23.fillna('NA')
F23.to_csv('workeddata/F2.3.csv', index=False, encoding="utf_8_sig")
#print(F23)


#6特征汇总

F1 = pd.read_csv('workeddata/F1.csv')
F2 = pd.read_csv('workeddata/F2.csv')
F3 = pd.read_csv('workeddata/F3.csv')
F23 = pd.read_csv('workeddata/F2.3.csv')

#读取列
F2no = F2[['ID', 'F2.2', 'F2.19', 'F2.22', 'F2.25', 'F2.28', 'F2.31', 'F2.34']]
F3no = F3[['ID', 'F3.13', 'F3.14', 'F3.15', 'F3.16', 'F3.17', 'F3.18', 'F3.19', 'F3.20',
           'F3.21', 'F3.22', 'F3.23', 'F3.31', 'F3.32', 'F3.33']]

#删除列
F2 = F2.drop(['F2.2', 'F2.19', 'F2.22', 'F2.25', 'F2.28', 'F2.31', 'F2.34'], axis=1)
F3 = F3.drop(['F3.13', 'F3.14', 'F3.15', 'F3.16', 'F3.17', 'F3.18', 'F3.19', 'F3.20',
              'F3.21', 'F3.22', 'F3.23', 'F3.31', 'F3.32', 'F3.33'], axis=1)

#根据用户ID合并表
feature = F1.join(F2.set_index('ID'), on='ID')
feature = feature.join(F3.set_index('ID'), on='ID')
feature = feature.join(F23.set_index('ID'), on='ID')
#print(feature)


"""特征归一化"""
#读取所有列名遍历
c = 0
for t in list(feature):
    #跳过前4列
    if c < 4:
        c = c + 1
        continue   #跳出循环
    #读取列
    demo = feature[[t]]
    #获取当前列最大值
    max = demo.sort_values(t, ascending=False).head(1)
    maxvalue = max.get(t).values[0]
    #归一化：列值/最大值
    feature[t] = feature[t] / maxvalue
#print(feature)


#把数据写入表格，以待建模时读取

#空值替换为1
feature = feature.fillna(1)
#根据用户ID合并表
feature = feature.join(F2no.set_index('ID'), on='ID')
feature = feature.join(F3no.set_index('ID'), on='ID')
feature = feature[['ID', 'F1.1', 'F1.2', 'F1.3', 'F2.1', 'F2.2', 'F2.3', 'F2.4', 'F2.5', 'F2.6', 'F2.7', 'F2.8', 'F2.9', 'F2.10',
                   'F2.11', 'F2.12', 'F2.13', 'F2.14', 'F2.15', 'F2.16', 'F2.17', 'F2.18', 'F2.19', 'F2.20',
                   'F2.21', 'F2.22', 'F2.23', 'F2.24', 'F2.25', 'F2.26', 'F2.27', 'F2.28', 'F2.29', 'F2.30',
                   'F2.31', 'F2.32', 'F2.33', 'F2.34', 'F2.35', 'F2.36',
                   'F3.1', 'F3.2', 'F3.3', 'F3.4', 'F3.5', 'F3.6', 'F3.7', 'F3.8', 'F3.9','F3.10',
                   'F3.11', 'F3.12', 'F3.13', 'F3.14', 'F3.15', 'F3.16', 'F3.17', 'F3.18', 'F3.19', 'F3.20',
                   'F3.21', 'F3.22', 'F3.23', 'F3.24', 'F3.25', 'F3.26', 'F3.27', 'F3.28', 'F3.29', 'F3.30',
                   'F3.31', 'F3.32', 'F3.33', 'F3.34', 'F3.35', 'F3.36', 'F3.37', 'F3.38', 'F3.39', 'F3.40',
                   'F3.41', 'F3.42', 'F3.43', 'F3.44', 'F3.45', 'F3.46', 'F3.47', 'F3.48', 'F3.49', 'F3.50',
                   'F3.51', 'F3.52', 'F3.53', 'F3.54', 'F3.55', 'F3.56', 'F3.57', 'F3.58', 'F3.59', 'F3.60',
                   'F3.61', 'F3.62', 'F3.63', 'F3.64', 'F3.65', 'F3.66', 'F3.67', 'F3.68', 'F3.69', 'F3.70',
                   'F3.71', 'F3.72', 'F3.73', 'F3.74', 'F3.75', 'F3.76', 'F3.77', 'F3.78', 'F3.79', 'F3.80',
                   'F3.81', 'F3.82', 'F3.83', 'F3.84', 'F3.85', 'F3.86', 'F3.87', 'F3.88', 'F3.89',
                   'F2.3.1', 'F2.3.2', 'F2.3.3', 'F2.3.4', 'F2.3.5', 'F2.3.6', 'F2.3.7', 'F2.3.8', 'F2.3.9', 'F2.3.10',
                   'F2.3.11', 'F2.3.12', 'F2.3.13', 'F2.3.14', 'F2.3.15', 'F2.3.16', 'F2.3.17', 'F2.3.18', 'F2.3.19', 'F2.3.20',
                   'F2.3.21', 'F2.3.22', 'F2.3.23', 'F2.3.24']]
#空值替换为0
feature = feature.fillna(0)
#把数据写入table_feature.csv，为相对路径
feature.to_csv('workeddata/table_feature.csv', index=False, encoding="utf_8_sig")
#print(feature)

#2建立模型
"""最后是进行数据分析的阶段。本阶段会用到上一步产生的数据集，然后将数据随机抽样90%作为训练数据集，剩下10%作为测试数据集，
并且按照XGBoost函数的格式进行数据挖掘的计算。而后针对训练出来的模型，将测试数据导入其中，得到预测数据。将预测数据与是假数据对比，
通过计算模型评估指标(ACU)进行计算后，对训练模型做出评价"""
import xgboost as xgb
from sklearn.model_selection import train_test_split

#读取文件
train = pd.read_csv('workeddata/table_database.csv')
#名义特征设定。大部分名义特征在读取是会被装变为数值特真，为次，要将这些特征装换为名义特征
train['F2.19'] = pd.factorize(train['F2.19'])[0].astype(np.uint16)
#astype指定数据类型；pd.factorize()做的也是“因式分解”，把常见的字符型变量分解为数字
#factorize(train['F2.19']),将指定的列，按列值的多少种取值，进行编码(目的让模型识别)
#装换后是一个array元组；这个元组包含两个array,分别是我们想要的数字，以及原来的index(即本身未编码之前的值)
#因为我们只要前一列，即数字。所以为factorize(train['F2.19'])[0]
train['F2.22'] = pd.factorize(train['F2.19'])[0].astype(np.uint16)
train['F2.25'] = pd.factorize(train['F2.19'])[0].astype(np.uint16)
train['F2.28'] = pd.factorize(train['F2.19'])[0].astype(np.uint16)
train['F2.31'] = pd.factorize(train['F2.19'])[0].astype(np.uint16)
train['F2.34'] = pd.factorize(train['F2.19'])[0].astype(np.uint16)
train['F3.31'] = pd.factorize(train['F2.19'])[0].astype(np.uint16)
train['F3.32'] = pd.factorize(train['F2.19'])[0].astype(np.uint16)
train['F3.33'] = pd.factorize(train['F2.19'])[0].astype(np.uint16)
#删除列
train = train.drop(['F1.1', 'F1.2', 'F1.3'], axis=1)

#设为目标 所要划分的样本结果(测试集)
df_train = train['target'].values

#删除列  所要划分的样本特征集(训练集)
train = train.drop(['target'], axis=1)


#随机抽取90%的资料作为训练数据，剩余10%作为测试数据
X_train, X_test, y_train, y_test = train_test_split(train, df_train, test_size=0.1, random_state=1)
#test_size：测试数据的比例
#random_state：是一个随机种子，是在任意带有随机性的类或函数里作为参数来控制随机模式。当random_state取某一个值时，也就确定了一种规则
# 使用XGBoost的原生版本需要对数据进行转化(封装训练和测试数据)
data_train = xgb.DMatrix(X_train, y_train) #构造训练集
data_test = xgb.DMatrix(X_test, y_test) #构造测试集
# 设置参数 'max_depth':表示树的深度   'eta':表示权重参数  'objective':表示训练目标的学习函数
param = {'max_depth': 4, 'eta': 0.2, 'objective': 'reg:linear'}
watchlist = [(data_test, 'test'), (data_train, 'train')]
#表示训练次数
n_round = 1000
# 训练数据载入模型
data_train_booster = xgb.train(param, data_train, num_boost_round=n_round, evals=watchlist)
#params：字典类型，用于指定各种参数，例如：{‘booster’:‘gbtree’,‘eta’:0.1}
#data_train：用于训练的数据，通过给下面的方法传递数据和标签来构造
#num_boost_round：指定最大迭代次数，默认值为10
#evals：列表类型，用于指定训练过程中用于评估的数据及数据的名称。例如：[(dtrain,‘train’),(dval,‘val’)]
# 计算错误率
y_predicted = data_train_booster.predict(data_test)  #测试集的预测
y = data_test.get_label()
accuracy = sum(y == (y_predicted > 0.5))  #用于设定阀值，概率大于0.5
accuracy_rate = float(accuracy) / len(y_predicted)
print('预测样本总数：{0}'.format(len(y_predicted)))
print('正确数目：{0}'.format(accuracy))
print('正确率：{0:.10f}'.format((accuracy_rate)))
#对测试集进行预测(回归问题用MSE:均方误差即预测误差
from sklearn.metrics import mean_squared_error
dtest = xgb.DMatrix(X_test)
ans = data_train_booster.predict(dtest)
print('mse:', mean_squared_error(y_test, ans))


