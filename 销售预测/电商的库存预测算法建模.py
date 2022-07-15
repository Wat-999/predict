"""
1算法原理
要预测商品的补货周期，我们要先算出上一个周期的销量，再将当前库存减去上一个周期的销量。当库存充足时，我们把当前库存除以上一个周期的销量，
再乘补货周期的天数可以算出几个周期以后需要补货。当库存不足时，我们要进行补货，补货量的多少是上一个周期的销量减去当前库存量再加上一个补货周期的销量。

2数据说明
库存预测算法模型需要准备好三大类数据：
订单报表数据、商品报表数据、库存表数据

3案例实现思路
1计算每个商家编码近n天的销量，N为补货周期的日期
2计算多少天后需要补货，计算公式为：库存量/近N天销量*补货周期
"""

#准备数据
import pymysql
import pandas as pd
import math

#创建自定义函数df()，把pymysql库读取的数据装换成dataframe(数据框)格式
def df(result, col_result):
    #获取字段信息
    columns = []
    for i in range(len(col_result)):
        columns.append((col_result[i][0]))
    #创建dataframe
    df = pd.DataFrame(columns=columns)
    #插入数据
    for i in range(len(result)):
        df.loc[i] = list(result[i])
    return df


#打开数据库
mmm = pymysql.connect(host='127.0.0.1', user='root', passwd='123456', port=3306, db='kucun', charset='utf8')
# 创建游标
cursor = mmm.cursor()


#创建sql指令
sql = '''select
a.`商家编码`,
MAX(a.`库存`) `库存`,
MAX(a.`补货周期`) `补货周期`,
SUM(b.`购买数量`) `近N天销量`
from `库存表` a
LEFT JOIN `宝贝报表` b on a.`商家编码` = b.`商家编码`
LEFT JOIN `订单报表` c on b.`订单编号` = c.`订单编号`
where DATEDIFF('2019-01-02', date(c.`付款时间`)) <= a.`补货周期` 
GROUP BY a.`商家编码`'''

#执行sql指令
cursor.execute(sql)

# 将读取的数据库数据转成DataFrame格式
df = df(cursor.fetchall(), cursor.description)

# 关闭连接
cursor.close()
mmm.close()
print(df.head())
# 数据读取分析(创建一个dataframe，并指定好字段名称)
supply_model = pd.DataFrame(columns=['商家编码', '库存', '补货周期', '近N天销量', '多少天后需要补货', '备一周期货量'])

#把数据写入dataframe中
for index, row in df.iterrows():
    # 商家编码
    id = row['商家编码']
    # 库存
    stock = int(row['库存'])
    # 补货周期
    cycle = int(row['补货周期'])
    # 近N天销量
    count = int(row['近N天销量'])
    # 多少天后需要补货
    buy_date = math.floor(stock / count) * cycle
    # 备一周期货量
    number = 0
    if (stock-count) < 0:
        number = count - stock + count

    data = {'商家编码': id, '库存': stock, '补货周期': cycle, '近N天销量': count, '多少天后需要补货': buy_date, '备一周期货量': number}
    supply_model = supply_model.append(data, ignore_index=True)

# 输出预测补货数量
print(supply_model[supply_model['备一周期货量'] > 0])

"""
"备一周期货量"是最基本单位， 如果按"备一周期货量备货量"则会偏少，备货时可以是这个数量的2～4倍
"""




