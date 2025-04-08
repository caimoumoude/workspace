from codecs import utf_8_decode
from uu import decode
import pandas as pd
from config.cicconfig import mapping
# 1. 读取原始 CSV 文件
# 假设你的原始文件名为 'captured_data.csv'，请替换为实际文件名
df = pd.read_csv('2025-04-07_Flow.csv',encoding='gbk')
df = df.drop_duplicates()
# 2. 特征名称映射
# 根据抓包工具的典型输出和目标特征值，我推测了一个映射字典
# 请根据你的实际 CSV 文件列名调整此映射


# 重命名列
df = df.rename(columns=mapping)

# # 删除重复行
# df = df.drop_duplicates()

# 将所有Label特征的值替换为指定的文字
# 您可以修改下面的值来设置所需的标签文字
specified_label = "bad"  # 在这里修改为您想要的标签文字
df.info()
df['Label'] = specified_label
print(df.columns)
# 5. 保存转换后的 CSV 文件
df.to_csv('transformed_data.csv', index=False)
file=open('hebing2.txt','w')
for i in df.columns:
    file.write(i)
    file.write('\n')
print("特征转换完成，已保存为 'transformed_data.csv'")