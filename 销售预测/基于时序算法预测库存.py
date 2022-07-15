"""
时间序列可以用于连续数据的预测，是一种基本的预测方法，预测时可以和其他方法一起使用，对预测结果尽心补充
算法原理
时间序列是一种统计分析方法，在营销工作中根据一定时间的数据序列预测未来的发展趋势，也称为时间序列趋势外推法。这种方法适合预测处于连续过程中的事物。
它需要有若干年的数据资料，按时间排列成数据序列，其变化趋势和相互关系要明确和稳定。公预测用的历史数据资料的变化可以表现出比较强的规律性，
由于它过去的变动趋势将会连续到未来，这样就可以直接利用过去的变动趋势预测未来。但多数的历史数据由于受偶然性因素的影响，其变化不太规则。利用这些资料时
要消除偶然性因素的影响，把时间序列作为随机变量序列，采用算术平均、加权平均和指数平均等来减少偶然因素，提高预测的准确性。常用的时间序列法有移动平均法
加权移动平均法和指数平均法。
分析背景
每年的双十一，都是对电商行业对一次考验。往往考验对是电商企业接单、打包的能力，物流公司人员配置是否充分等。当以上两点对于客户购物体验的提升达到阀值时，考验的
就是品牌和平台能否对供应链进行合理对调控，通过对库存和线上线下对协调减轻物流压力。
这也是商业智能分析中的"终极问题"————销售预测
在销售、市场和运营工作中，销售预测无处不再。往大来说，销售预测影响着企业的整体规划；往小来说，销售预测影响这企业每一次营销活动的成本投入。
在零售行业中，销售预测的重要性更加凸显。我们知道，零售行业的收益如何，取决于供应链能否良好的运转：没有库存的压力也没有缺货的现象、不同的商品都被储存在自己销售
情况最好区域的仓库中、新商品的生产和旧商品的售卖能形成衔接。
分析目的
预测该公司在未来的一年中每个店铺、每个品类、每个月的销售情况。从而让公司对每个店铺、每个品类对配货提供强有力对指导。
数据说明
train文件中的数据是该公司所有零售商近四年所有的销售数据，该公司有10个店铺，50个品类
实现思路
1对数据进行检查，确保可以使用时序算法
2使用arima()函数对该公司汇总的数据进行预测
3使用for循环对该企业所有对商品进行预测
"""

#1导入源数据
"""
parse_dates解析为时间索引。low_memory是一个布尔值，默认值为true，表示分块加载到内存，在低内存消耗中进行解析。
但是这种方式可能出现类型混淆，确保类型不被混淆需要设置为false，或者使用dtype参数指定类型
"""

import pandas as pd
import numpy as np
import sys
import warnings
import statsmodels.api as sm
import itertools  # itertools是用于高效循环的迭代函数集合
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
warnings.filterwarnings('ignore')  # 忽略ARIMA模型无法估计出结果时的报警信息

df_raw = pd.read_csv('/Users/macbookair/Desktop/数据分析/书本配套资料及电子书/python商业数据分析(零售和电子商务)/《Python商业数据分析：零售和电子商务案例详解》/第六章Python与销售预测案例/案例：基于时序算法预测库存/train.csv',
                     low_memory=False, parse_dates=['date'], index_col=['date'])
#index_col 默认值（index_col = None）——重新设置一列成为index值
#2数据检查及时序检查(先简单地观察数据，了解数据对结果，以及数据是否规整
#观察数据
print(df_raw.head())
print(df_raw.shape)
#观察数据对字段信息
df_raw.info()
#检查时间序列对季节性，输入值必须是float type(小数形式)
df_raw['sales'] = df_raw['sales'] * 1.0
#选取几个示例店铺对销售数据进行预测(可以使用for循环，把所有店铺都添加进来)
sales_a = df_raw[df_raw.store == 2]['sales'].sort_index(ascending=True)
sales_b = df_raw[df_raw.store == 3]['sales'].sort_index(ascending=True)
sales_c = df_raw[df_raw.store == 1]['sales'].sort_index(ascending=True)
sales_d = df_raw[df_raw.store == 4]['sales'].sort_index(ascending=True)

#画出时序图
import matplotlib.pyplot as plt
# 平稳序列的时序图在一个常数附近波动，而且波动范围有界。
sales_a.resample('W').sum().plot()  #按周对数据重采样
plt.show()
"""通过输出对时序图可以看出数据有明显对趋势性、周期性，不是非平稳定序列(平稳序列的时序图在一个常数附近波动，而且波动范围有界。）"""

# 平稳序列具有短期相关性，随着延迟期数对增加，平稳序列对自相关系数会较快地衰减并趋向于零，并在零附近随机波动
from statsmodels.graphics.tsaplots import plot_acf
#acf又称自相关函数，用来度量同一事件在不同时期之间的相关程度，或者说是一个信号经过类似于反射、折射等其它情况的延时后的副本信号与原信号的相似程度，简单讲就是比较不同时间延迟两个序列的相似程度
#利用图中发现其规律，看序列在经过了多少个t滞后和序列的相关又达到了峰值，从而判断序列的周期
plot_acf(sales_a.resample('W').sum()) #MS月
plt.show()

#把abcd的时序图绘制在一张画布中
f, (ax1, ax2, ax3, ax4) = plt.subplots(4, figsize=(12, 13))
c = '#386B7F'#线条的颜色编号

#按周('W')对数据重采样，对加总值绘制折线图，resample()函数用来对原始数据采样，如果调整成每3分钟重采样一次，那么输入对不是W而是3T
sales_a.resample('W').sum().plot(color=c, ax=ax1)
sales_b.resample('W').sum().plot(color=c, ax=ax2)
sales_c.resample('W').sum().plot(color=c, ax=ax3)
sales_d.resample('W').sum().plot(color=c, ax=ax4)
plt.show()

#利用统计模型进行线性回归
import statsmodels.api as sm

#按年加总绘制，所有店铺对可视化图形都是一样的（时序列数据分解成趋势（trend）季节性（seasonality）和误差（residual）- sm.tsa.seasonal_decompose())
f, (ax1, ax2, ax3, ax4) = plt.subplots(4, figsize=(12, 13))
decomposition_a = sm.tsa.seasonal_decompose(sales_a, model='additive', period=365)  #model参数表示时间序列分解的类型其中additive表示为线性增长， multiplicative表示为指数增长
decomposition_a.trend.plot(color=c, ax=ax1)
decomposition_b = sm.tsa.seasonal_decompose(sales_b, model='additive', period=365)  #period参数表示时间序列的周期，这里表示按年
decomposition_b.trend.plot(color=c, ax=ax2)
decomposition_c = sm.tsa.seasonal_decompose(sales_c, model='additive', period=365)
decomposition_c.trend.plot(color=c, ax=ax3)
decomposition_d = sm.tsa.seasonal_decompose(sales_d, model='additive', period=365)
decomposition_d.trend.plot(color=c, ax=ax4)
# 可以从中发现非常明显的季节性。
plt.show()

# 把店铺和品类消除，只观察时间跟销售额的关系。
date_sales = df_raw.drop(['store', 'item'], axis=1).copy()
# 这是一个临时的数组，原来的数组dr_raw没有受到影响。
# 只有一个维度
print(date_sales.head())
date_sales.info()

# 画出时序图
y = date_sales['sales'].resample('MS').sum() #每月销售的总和  对销售额按照按月('MS')对数据重采样
y.plot(figsize=(15, 6))#绘制月均销售数据图
plt.show()  #可以发现季节性波动，在每年的年末销售量最低，在年中的销售量最高

#为了进一步探究数据情况，可以对数据进行时间序列对分解，分解为趋势、季节性和误差，
#使用加法模型(additive)分解时间序列
decomposition = sm.tsa.seasonal_decompose(y, model='additive')
decomposition.plot()
plt.show()

#使用乘法法模型(multiplicative)分解时间序列
decomposition = sm.tsa.seasonal_decompose(y, model='multiplicative')
decomposition.plot()
plt.show()

# 时间序列建模(建立ARIMA模型，使用常见的时间序列模型ARIMA(p,d,q)来进行预测；ARIMA是指将非平稳时间序列转化为平稳时间序列，然后将因变量对它对滞后值，
#以及随机误差项对现值和滞后值进行回归建立对模型，其中p,d,q分别代表数据中对季节性、趋势、噪声。AR代指Auto-Regressive(p):p是指lags滞后对阶数，
#例如p=3，那么我们会用x(t-1)、x(t-2)和x(t-3)来预测x(t)。I代指Integrated(d)：代表非季节性差异，例如在这个案例中，我们使用来一阶差分，所以我们让d=0
#MA代指Moving Averages(q):代表预测中滞后代预测误差
# 预测中滞后的预测误差
# itertools是用于高效循环的迭代函数集合
import itertools  # itertools是用于高效循环的迭代函数集合
p = d = q = range(0, 3)
# 对p,d,q的所有可能取值，进行配对组合
pdq = list(itertools.product(p, d, q)) #对p、d、q的所有可能取之，进行配对组合
# itertools.product()求笛卡尔积。itertools这个模块中有相当多的牛逼闪闪的数学算法，比如全排列函数permutations，组合函数combinations等等
#print(pdq)
#生成指定周期为12的参数组合
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
#print(seasonal_pdq)

#导入acf和pacf图的绘制工具，绘制多组子图subplot
#注意：其中各个参数也可以用逗号分隔开。第一个参数代表子图的行数；第2个参数代表该行图像的列数；第三个参数代表每行的第几个图像
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
plt.figure(figsize=(12, 16), dpi=300)  #plt.figure的作用是定义一个大的图纸，可以设置图纸的大小、分辨率（dpi)
# acf and pacf for A
plt.subplot(421)  #即表示一次性在figure上创建成4*2的网格并是第一个位置
plot_acf(sales_a, lags=50, ax=plt.gca(), color=c)  #lags参数表示所计算的lag范围，即图表的横坐标轴的数字 表示滞后阶数
#acf表示绘制自相关图，pacf表示绘制偏自相关图
plt.subplot(422)
plot_pacf(sales_a, lags=50, ax=plt.gca(), color=c) #ax=plt.gca( )进行坐标轴的移动

# acf and pacf for B
plt.subplot(423)
plot_acf(sales_b, lags=50, ax=plt.gca(), color=c)

plt.subplot(424)
plot_pacf(sales_b, lags=50, ax=plt.gca(), color=c)

# acf and pacf for C
plt.subplot(425)
plot_acf(sales_c, lags=50, ax=plt.gca(), color=c)

plt.subplot(426)
plot_pacf(sales_c, lags=50, ax=plt.gca(), color=c)


# acf and pacf for D
plt.subplot(427)
plot_acf(sales_d, lags=50, ax=plt.gca(), color=c)

plt.subplot(428)
plot_pacf(sales_d, lags=50, ax=plt.gca(), color=c)

plt.show()  #输出的图形展示来时间序列是有自相关性的
"""图形解读
Autocorrelation画的各店铺序列的各阶自相关系数的大小，该图的高度值对应的是各阶自相关系数的值，蓝色区域是95%置信区间(为以横轴为参照线上下0.05为蓝色区域)，
这两条界线是检测自相关系数是否为0时所使用的判别标准：当代表自相关系数的柱条超过这两条界线时，可以认定自相关系数显著不为0。
观察图，1阶(横坐标轴的数字则表示阶数)的自相关系数都在蓝色范围外，也就是落在了95%置信区间外，所以初步判断该序列可能存在短期的自相关性。"""

#由于有些组合不能收敛，所以使用try-except来寻找最佳的参数组合，需要数分钟的时间运行，可以使用网格搜索来迭代地探索参数的不同组合
#对于参数的每个组合，可以使用statsmodels模块的SARIMAX函数拟合一个新的季节性ARIMA模型，并评估其整体质量。一旦探索来参数的整个范围，产生最佳性能的参数将是我们感兴趣的

#生成希望评估的各种参数组合
import sys
warnings.filterwarnings("ignore") # 忽略ARIMA模型无法估计出结果时的报警信息
best_aic = np.inf    #表示+∞，是没有确切的数值的,类型为浮点型
best_pdq = None
best_seasonal_pdq = None
temp_model = None

for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            temp_model = sm.tsa.statespace.SARIMAX(y,
                                             order=param,
                                             seasonal_order=param_seasonal,
                                             enforce_stationarity=False,
                                             enforce_invertibility=False)
            results = temp_model.fit()
            if results.aic < best_aic:
                best_aic = results.aic
                best_pdq = param
                best_seasonal_pdq = param_seasonal
        except:
            continue
print("Best SARIMAX{}x{}12 model - AIC:{}".format(best_pdq, best_seasonal_pdq, best_aic))
#模型结果解读
"""
# 通过执行该脚本，获得的最优模型为：SARIMAX(2, 0, 1)x(0, 1, 1, 12)，各参数的含义是：
# 第一个(2, 0, 1)代表该模型是ARIMA(2, 0, 1),即0阶差分I（0）模型；
# 第二个(0, 1, 1, 12)代表季节效应为ARIMA(0, 1, 1)，也是1阶差分I（1）模型；
# 最后的12代表季节效应为12期，这是我们之前自己设置的。"""
#保存最佳模型、AIC、参数。发现最小AIC的参数是SARIMAX(2,0,1)x(2,2,0,12)

# 使用最优参数进行模型拟合
mod = sm.tsa.statespace.SARIMAX(y,
                                order=(2, 0, 1),
                                seasonal_order=(2, 2, 0, 12),
                                enforce_stationarity=False,
                                enforce_invertibility=False)
results = mod.fit()
print(results.summary().tables[1])#拟合结果展示

# 使用上述模型参数进行预测。
pred = results.get_prediction(start=pd.to_datetime('2017-01-01'), dynamic=False)
pred_ci = pred.conf_int()#置信区间的公式
# 绘制结果图
ax = y['2014':].plot(label='observed')
pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7, figsize=(10, 6))
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.2)
ax.set_xlabel('Date')
ax.set_ylabel('Sales')
plt.legend()
plt.show() #可见拟合效果还不错

"""上面仅仅拟合的是这家公司全部的销售数据，但是对于一家拥有10个店铺、50个品类的零售企业来说，只有详细地预测每一类商品在每个店铺才能指导这家公司进行库存管理"""
# 建一个for循环的ARIMA模型
# 创建一个空列表，用于循环预测时，最后导入预测结果使用
subs_add = pd.DataFrame({'month': [], 'sales_forecast': [], 'item': [], 'store': []})
import itertools  # itertools是用于高效循环的迭代函数集合

# 设置参数范围。
p = d = q = range(0, 3)
pdq = list(itertools.product(p, d, q))  # 对p,d,q的所有可能取值，进行配对组合
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
# 自定义得到最佳参数的函数
def param_func(y):
    cnt = 0
    pdq_test = []
    seasonal_pdq_test = []
    AIC = []

    for param in pdq:
        for param_seasonal in seasonal_pdq:
            try:
                mod = sm.tsa.statespace.SARIMAX(y,
                                                order=param,
                                                seasonal_order=param_seasonal,
                                                enforce_stationarity=False,
                                                enforce_invertibility=False)
                results = mod.fit()
                cnt += 1
                if cnt % 50:
                    pdq_test.append(param)
                    seasonal_pdq_test.append(param_seasonal)
                    AIC.append(results.aic)
                    # print('Current Iter - {}, ARIMA{}x{} 12 - AIC:{}'.format(cnt, param, param_seasonal, results.aic))
            except:
                continue
    v = AIC.index(min(AIC))
    pdq_opt = pdq_test[v]
    seasonal_pdq_opt = seasonal_pdq_test[v]
    param_opt = [pdq_opt, seasonal_pdq_opt]
    return param_opt
#print(param_opt)
# 自定义得到预测结果，并添加至subs_add的函数。
def forecast_func(y):
    param_opt = param_func(y)
    pdq_opt = param_opt[0]
    seasonal_pdq_opt = param_opt[1]
    #print('开始拟合模型第{}，{}次'.format(m,n))
    #使用最优参数进行模型拟合
    mod = sm.tsa.statespace.SARIMAX(y,
                                order=pdq_opt,
                                seasonal_order=seasonal_pdq_opt,
                                enforce_stationarity=False,
                                enforce_invertibility=False)
    results = mod.fit()

    # 利用下面代码预测未来12期(即steps=12)的情况；这里的get_forecast是外推N步预测，get_prediction是按照时间点要求，对数据进行预测
    pred_forecast = results.get_forecast(steps=12)
    forecast = pred_forecast.predicted_mean
    # forecast数据类型为pandas.core.series.Series，为了后续操作方便，我们将其转为pandas dataframe。
    dict_f = {'month': forecast.index, 'sales_forecast': forecast.values}  # 先将数据转变为字典
    df_forecast = pd.DataFrame(dict_f)  # 转换数据类型为dataframe

    # 在dataframe中添加“item”和“store”列，赋值与循环到的m, n值一致。
    df_forecast['item'] = m
    df_forecast['store'] = n

    # 向最终结果列表中，按行添加最终预测结果。
    global subs_add
    subs_add.append(df_forecast, ignore_index=True)

    # 需要赋值给subs_add，不然数据没有写入的对象。
    subs_add = subs_add.append(df_forecast, ignore_index=True)
    return subs_add

# 使用for循环，依次得到所有item(m)和store(n)组合下的销量预测。  m表示品类数  n表示店铺数
# 运行单次一般需要数分钟时间，运行50X10次，意味着需要数百分钟（至少5个小时以上）
# 为了观察，把下面的 m in range(1,51) 改为（1,3)，n in range(1,11)改为(4,6)
# 这样调整后的运行时间会缩短到10分钟左右。
pair = []
for m in range(1, 3):
    for n in range(4, 6): #封装多个变量的循环
        df = df_raw.query('item == {} & store == {}'.format(m, n))
        y = df['sales'].resample('MS').sum()
        print('开始参数测算第{}，{}次'.format(m, n))
        df_forecast = forecast_func(y)
        pair.append([m, n])
# 观察pair检查是否所有配对的（m,n）都完成。
#print(pair)

#print(subs_add.head())

# 保证店铺和产品编号为整数，转换成int类型。
subs_add['item'] = subs_add['item'].astype('int')
subs_add['store'] = subs_add['store'].astype('int')
# 打印所有商店各类产品的销售情况。
#print(subs_add)

# 将预测结果保存到本地cvs文件。
subs_add.to_csv('Result.csv')
df_fe = df_raw.reset_index(drop=False)
df_fe.head()

"""这种循环方法，效率其实较低，所以需要使用更有效率的机器学习算法来替代
机器学习本质上是学习特征和结果变量之间的关系，但是我们所拥有的训练数据中，只有date、item、sales几个特征变量
并不够充分，所以在处理前，我们往往会从日期中提取更多信息，比如这一天是一周的第几天、第一周是这年的第几周
为来让日期成为一个变量列，可以使用reset_index()函数，重新进行行索引，drop为false时，则索引列会被还原成普通列，否则会丢失
df_fe = df_raw.reset_index(drop=false)
df_fe.head()"""

#提取日期特征
df_fe['dayofmonth'] = df_fe.date.dt.day
df_fe['dayofyear'] = df_fe.date.dt.dayofyear##数据DF.列.dt.dayoftyear转换为是一年中的哪一天
df_fe['dayofweek'] = df_fe.date.dt.dayofweek
df_fe['month'] = df_fe.date.dt.month
df_fe['year'] = df_fe.date.dt.year
df_fe['weekofyear'] = df_fe.date.dt.weekofyear
df_fe['is_month_start'] = (df_fe.date.dt.is_month_start).astype(int)
df_fe['is_month_end'] = (df_fe.date.dt.is_month_end).astype(int)

print(df_fe.head())

"""通过提取日期特征，这样可以更好地体现销量时间序列的季节性、趋势性等特征。这个过程称为特征工程(FE)，在完成FE过程后，再考虑使用XGBoost等流行的算法进行处理"""