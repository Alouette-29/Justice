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
        #print(self.line)
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
    # 以下为ybk的函数的实验版,功能是去除某些键值对。注意:要实现多人代码的流水线协作，应该传入一个olddict
    def delete_none_value(self,olddict,colpath):
        #经测试，顺序删除原字典中的键值对会报错，原因是每次执行后字典长度发生改变。所以考虑用新字典来存放。
        #olddict=self.get_dict()[1] #取得原字典
        self.load_log(colpath)#加载对照字典
        newdict = {}
        for k in olddict.items(): #遍历原字典的键值对
            if k[0] in self.columns:  #在对照字典里，添加进去
                newdict[k[0]] = k[1]
        return newdict  #返回值为新的字典
    #ybk的实验版:处理court_name,olddict为传入的字典
    def modify_courtname(self,olddict):
        str=olddict['court_name']
        dt = {}
        slen = len(str)
        id1 = str.find('省')
        if (id1!=-1):
            dt['省'] = str[0:id1 + 1]

        id2 = str.find('市')
        #print(id2)
        if (id2 != -1):
            dt['市'] = str[id1 + 1:id2 + 1]
            dt['法院'] = str[id2 + 1:slen]
        else:
            id5=str.find('自治州')+2
            if(id5!=-1):
                dt['市']=str[id1+1:id5+1]
                dt['法院']=str[id5+1:slen]
            else:
                id3 = str.find('人民法院')
                if (id3 != -1):
                    dt['县'] = str[id1 + 1:id3]
                    dt['法院'] = str[id3:slen]

                else:
                    id4 = str.find('法院')
                    dt['县'] = str[id1 + 1:id4]
                    dt['法院'] = str[id4:slen]

        newdict=dict(olddict)
        newdict['court_name'] = dt
        return newdict


'''
A=Case(filename="E:/p2-1-2021/p2-1-2021.txt",encoding='UTF-8')
for i in range(0,10):
    dic=A.get_dict()[1]
    olddict=A.delete_none_value(dic,"E:/xiaochuang/Justice/code/utensil/keys_without_0.npy")
    newdict=A.modify_courtname(olddict)
    print(olddict['court_name'],newdict['court_name'])
'''