import pandas as pd
import warnings
warnings.filterwarnings('ignore')

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
import itertools
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

#plt.show()  #输出的图形展示来时间序列是有自相关性的
"""图形解读
Autocorrelation画的各店铺序列的各阶自相关系数的大小，该图的高度值对应的是各阶自相关系数的值，蓝色区域是95%置信区间(为以横轴为参照线上下0.5为蓝色区域)，
这两条界线是检测自相关系数是否为0时所使用的判别标准：当代表自相关系数的柱条超过这两条界线时，可以认定自相关系数显著不为0。
观察图，1阶(横坐标轴的数字则表示阶数)的自相关系数都在蓝色范围外，也就是落在了95%置信区间外，所以初步判断该序列可能存在短期的自相关性。"""

#由于有些组合不能收敛，所以使用try-except来寻找最佳的参数组合，需要数分钟的时间运行，可以使用网格搜索来迭代地探索参数的不同组合
#对于参数的每个组合，可以使用statsmodels模块的SARIMAX函数拟合一个新的季节性ARIMA模型，并评估其整体质量。一旦探索来参数的整个范围，产生最佳性能的参数将是我们感兴趣的

#生成希望评估的各种参数组合
import os
import sys
file_handle = open('/Users/macbookair/PycharmProjects/pythonProject1/venv/零售和电子商务/销售预测/a.txt', mode='w')
cnt = 0
for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(y,
                                            order=param,
                                            seasonal_order=param_seasonal,
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)
            results = mod.fit()
            print(results.aic <= 20)
            cnt += 1
            if cnt % 50:
                cu = 'Current Iter - {}, ARIMA{}x{} 12 - AIC:{}'.format(cnt, param, param_seasonal, results.aic) #fotmat作为Python的的格式字符串函数
                print(cu)
                file_handle.write(cu)
                file_handle.write('\n')


        except:
            continue  #结束循环





file_handle.close()
print("写入完成")