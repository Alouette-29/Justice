import json
import os 
import numpy as np 
class Case():
    def __init__(self,filename,encoding) -> None:
        if os.path.exists(filename) and os.path.isfile(filename):
            self.file = open(filename,"r",encoding=encoding)
        else:
            self.file = None
        self.line = 0
    # 每次从file里面读取1行，返回 行号和字典 
    # 注意 不可重入。 避免重入的方法是： 重新初始化这个类
    def get_dict(self):
        file = self.file
        suit = file.readline()
        dictionary = json.loads(suit)
        self.line+=1
        return self.line-1,dictionary
    # esay to understand 
    def close(self):
        self.file.close()
    # 加载 一个非0 的数据列字典  
    def load_log(self,colpath):
        columns = np.load(colpath,allow_pickle=True).item()
        self.columns =  columns
    # 增加一个成员函数
# 统计一下false 的个数， 也就是content 不存在的情况下，其他text存在的情况
# 再结合一下 总的统计数据 可以计算出 content 存在的情况下 text 不存在的情况，由此有一个2*2*6的矩阵 
import pandas as pd
import numpy as np    
columns = ['basics_text','judge_record_text','head_text','tail_text','judge_result_text','judge_reson_text']
def get_boolean(dictionary):
    content = dictionary['content']
    if content ==None:
        boolarray = [dictionary['basics_text']==None,
        dictionary['judge_record_text']==None,
        dictionary['head_text']==None,
        dictionary['tail_text']==None,
        dictionary['judge_result_text']==None,
        dictionary['judge_reson_text']==None]
    df = pd.DataFrame(boolarray).T
    df.to_csv("None_relation.txt",mode = 'a',header=None,index=None)
    return dictionary
def regulize(case_string):
    #处理逻辑
    # part2 wash content
    # 将连续的 \n转换成 一个\n  
    import re
    pattern1 = re.compile("[ \t\r\f]+")# 匹配除了\n的连续空白符号 
    pattern2 = re.compile('\n+') # 匹配连续的加号
    pattern3 = re.compile("\u3000") #清除奇怪的符号
    result = re.sub(pattern1,' ',case_string)
    result = re.sub(pattern2,'\n',result)
    result = re.sub(pattern3,'',result)
    return case_string
def analysis_boolarray(filename):
    # 输入文件是一个表格，每一行是一个样本的数据
    # 取出当 content 字段为空时，其余字段为空的情况
    # 如果为空，则为True ,否则为false 
    # 通过分析这个表格，可以看出 content 为空 和其他text 为空 是否有相关性
    # 分析的结果是， content == None  236297 条记录  ， 其中其他六个字段全部为None 的有  235352条
    #reports 存的是 六个数据列里面 True和False 的数量 
    #value_combination 存的是 一排不同的取值的组合的数量 
    df = pd.read_csv(filename,encoding='utf-8',header=None,index_col=None)
    reports = pd.DataFrame(columns=['True','False'],index=[i for i in range(6)])
    for col in range(6):
        Tnum,Fnum = df[col].value_counts()
        reports['True'][col] = Tnum
        reports['False'][col] = Fnum
    value_combination = df.value_counts()
    total = np.load("E:/xiaochuang/Justice/code/utensil/keys_without_0.npy",allow_pickle=True).item()

    for i in range(6):
        mat = cal_matrix(i,reports,total)
        print(mat[0])
        print(mat[1])
    return reports,value_combination
def cal_matrix(col:int,reports,total:dict):
    # matrix form #
    #                col==None  | col!= None|
    #               ————————————————————————————————
    # content!=None             |  cal      | tt_case-tt_content
    #               ——————————————————————————————
    # content==None     given   |           | tt_content 
    #                   given                 tt_case
    tt_case = 1125799
    tt_content = 889502
    mat = np.zeros(9).reshape((3,3))
    mat[1][0] = reports['True'][col]
    mat[2][0] = tt_case - total[columns[col]]
    mat[0][0] = mat[2][0]-mat[1][0]

    mat[0][2] = tt_content
    mat[1][2] = tt_case-tt_content
    mat[2][2] = tt_case

    mat[1][1] = reports['False'][col]
    mat[2][1] = total[columns[col]]
    mat[0][1] = mat[0][2]-mat[0][0]
    #print(mat)
    chart_name = columns[col]+"matrix"
    return chart_name,mat.astype("int")
from time import time 
import json
pre_timer = time()
clean_set = 'E:/xiaochuang/Justice/code/cleanset.txt'
filename = "E:/xiaochuang/p2-1-2021/p2-1-2021.txt"
with open(filename,"r",encoding="utf-8") as f:
    line = 0
    writelines = 0 
    while(True):
        content = f.readline()
        if content==None or len(content)==1:
            print("%d line has been visited"% line)
            break
        if line>1125898:
            print("%d line has been visited"% line)
            break

        dicts  = json.loads(content)
        
        if dicts['content']==None:
            get_boolean(dicts)
        else:
            writelines+=1
            # 修改content的值 
            regulize(content)
            with open(clean_set,'a',encoding='utf-8') as file:
                file.write(content)
        line+=1
        if(writelines%10000==0):
            crt_timer = time()
            print("%d lines has been writed %.2f seconds has been used " % (writelines,crt_timer-pre_timer))
            pre_timer = crt_timer
