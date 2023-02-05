import json
import os 
import numpy as np 
import pickle

class Case():
    def __init__(self, filename, encoding) -> None:
        if os.path.exists(filename) and os.path.isfile(filename):
            self.file = open(filename, "r", encoding=encoding)
        else:
            self.file = None
        self.line = 0

        # xby修改=====
        self.dictionary = {}

    # 每次从file里面读取1行，返回 行号和字典 
    # 注意 不可重入。 避免重入的方法是： 重新初始化这个类
    def get_dict(self):
        file = self.file
        suit = file.readline()
        dictionary = json.loads(suit)
        self.line += 1

        # xby修改======
        self.dictionary = dictionary

        return self.line - 1, dictionary
    # esay to understand 
    def close(self):
        self.file.close()
    # 加载 一个非0 的数据列字典  
    def load_log(self,colpath):
        columns = np.load(colpath, allow_pickle=True).item()
        self.columns = columns

    # ============= xby =======================
    # 把self.dict中的appellor字段去重，并存为list类
    def appellor_dedup(self):
        item = self.dictionary["appellor"]  # str类型
        item = item.split(',')  # lisr类型
        self.dictionary["appellor"] = list(set(item))  # 修改self.dictionary

    # 设置court_id
    def set_court_id(self):
        courtname = self.dictionary["court_name"]
        self.dictionary["court_id"] = courts.index(courtname)



# 下面一段代码不是测试，是必要的根据fulltext得到courts(list)的过程

filename = "D:/LabTestCases/p2-1-2021.txt"  # 自己修改
encoding = "utf-8"
fulltext = open(filename, "r", encoding=encoding)
courts = []  # list中的索引代表court_id，记录法院名(sorted list)

# 防止每次都要生成一遍courts（它是只要p21那个txt不变就不变的一个大list，长为6227）
try:  # 如果可以读文件加载courts（我已经把它推仓库了，courts.pkl）
    with open ("courts.pkl", 'rb') as f:
        courts = pickle.load(f)

except:  # 如果没有现成的文件，需要自己跑一遍原始的生成courts的代码（如下）
    while 1:
        line = fulltext.readline()
        if line != '':  # 还没结束
            dictionary = json.loads(line)
            courtname = dictionary["court_name"]  # 得到当前行的法院名
            if courtname in courts:
                continue  # 无需操作，去看下一行
            else:
                courts.append(courtname)

        else:  # 文件读完了
            break

    fulltext.close()
    courts = sorted(courts)

    # 存储为pkl文件(二进制对象文件)
    with open ("courts.pkl", 'wb') as f:
        pickle.dump(courts, f) 


# 下面是xby对自己写的函数的测试
case = Case(filename=filename, encoding=encoding)
case.get_dict()
case.appellor_dedup()
case.set_court_id()
print(case.dictionary)
print(courts)