# 从原数据文件中提取并且处理信息
mode = "utf-8"
import json 
import os
import sys
import cpca
import pandas as pd
sys.path.append('..')

from utensil.get_text import Case
# please implement your local file path here, if you want run this script
case_reader = Case("./path_to_file")
for i in range(225160):
    court1 = case_reader.get_dict()[1]['court_name']
    df = cpca.transform([court1])
    df.to_csv("location_distribution.txt",mode='a',header=None)

#注意，我没有保存无关信息，读取的时候 pandas 自动补全 
df1 = pd.read_csv("./location_distribution.txt")
df1 = df1.drop(['0'],axis=1)
df1.to_csv("location_distribution.txt",index=None)
df1 = pd.read_csv("./location_distribution.txt",index_col=None)

#cpca 的处理还是不能完全正确，这个文件是，不规则的名称 
file = open("clean_court_name.txt",'w',encoding='utf-8')
for i in df1['法院']:
    
    if type(i)!=str or (i !='人民法院' and i!="中级人民法院"):
        print(i)
        file.write(str(i)+'\n')
file.close()