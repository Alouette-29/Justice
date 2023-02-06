from time import time 
from tqdm import tqdm 
import json
import re
import json
import os 
import numpy as np 
import pandas as pd 
# 注意  所有的文件名都传参了  ，为了方便修改 
class Case():
    def __init__(self,filename='../p2-1-2021.txt',encoding='utf-8',colpath = '../keys_without_0.npy') -> None:
        # 这个主要是数据清洗的时候要用
        # 因为我们可以从文件名打开文件，而不调用self.get_dict()
        self.filename = filename 
         # xby修改=====
        self.dictionary = {}
        if os.path.exists(filename) and os.path.isfile(filename):
            self.file = open(filename,"r",encoding=encoding)
        else:
            self.file = None
        self.line = 0
        self.dictionary = {}
        #self.ttline = 1125799
        self.ttline = 0 
        self.columns = self.load_log(colpath)
        self.get_ttline()
        #print(self.line)
    # 每次从file里面读取1行，返回 行号和字典 
    # 注意 不可重入。 避免重入的方法是： 重新初始化这个类
    def get_ttline(self):
        if self.ttline != 0 :
            # 说明这个函数已经执行过了 
            return 

        storage = os.stat(self.filename).st_size
        read = 0
        buffer_size = 1024*8192
        thefile=open(self.filename,encoding='utf-8')
        start_time = time()
       
        while True:
            buffer=thefile.read(buffer_size)
            if not buffer:
                break
            read += buffer_size
            self.ttline+=buffer.count('\n')
            crt_timer = time()
            print("\r %.2f has been counted,%.2f seconds has been taken "%((read/storage),(crt_timer-start_time)) ,end="")
        thefile.close()
        print("Total lines of the file is ",self.ttline)
        return 

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
        #print(self.columns)
        return self.columns
    @staticmethod
    def dict2string(dictionary,filename=None):
        string = re.sub('\'',"\"",str(dictionary))
        string = re.sub("None","null",str(string))
        if filename!=None:
            with open(filename,'a',encoding='utf-8') as f:
                f.write(string+'\n')
        return string
    def appellor_dedup(self,dictionary):
        item = dictionary['appellor']
        item = item.split(",")
        dictionary['appellor']= list(set(item))
        return dictionary
    @staticmethod
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
    # 以下为ybk的函数的实验版,功能是去除某些键值对。注意:要实现多人代码的流水线协作，应该传入一个olddict
    #@staticmethod
    def delete_none_value(self,olddict):
        #经测试，顺序删除原字典中的键值对会报错，原因是每次执行后字典长度发生改变。所以考虑用新字典来存放。
        #olddict=self.get_dict()[1] #取得原字典
        #self.load_log(colpath)#加载对照字典
        # 这个字典只用加载一次，放preprocess 里 
        newdict = {}
        for k in olddict.items(): #遍历原字典的键值对
            if k[0] in self.columns \
                and k[0]!='is_private' and k[0]!='crawl_time' and k[0]!='d':  #在对照字典里，添加进去
                newdict[k[0]] = k[1]
        return newdict  #返回值为新的字典
    def modify_courtname(self,olddict,oldframe):
        str=olddict['court_name']
        ls=[]
        dt = {}
        slen = len(str)
        id1 = str.find('省')
        if (id1!=-1):
            dt['省'] = str[0:id1 + 1]
        id2 = str.find('市')
        #print(id2)
        if (id2 != -1):
            dt['市'] = str[id1 + 1:id2 + 1]
        else:
            id2=str.find('自治州')+2
            if(id2!=-1):
                dt['市']=str[id1+1:id2+1]
        idm1=max(id1,id2)
        id3=str.find('区')
        if(id3!=-1):
            dt['区']=str[idm1+1:id3+1]
            dt['地址']=str[id3+1:slen]
        else:
            id3=id2
            dt['地址']=str[id3+1:slen]

        #id=olddict['court_id'] #adcode
        province=dt.get('省') if dt.get('省')!=None else '' #省
        city=dt.get('市') if dt.get('市')!=None else '' #市
        district=dt.get('区')if dt.get('区')!=None else '' #区
        loc=dt.get('地址')if dt.get('地址')!=None else '' #地址
        ls.append(province)
        ls.append(city)
        ls.append(district)
        ls.append(loc)
        #ls.append(id)
        df2= pd.DataFrame(np.insert(oldframe.values, len(oldframe.index), values=ls, axis=0)) #更新
        return df2
    
    def preprocess(self,clean_set = "../cleanset.txt"):
        # 这是总的预处理函数，负责提供遍历数据集的骨架
        # 下面是子函数 ，在不同的分支调用不同的处理函数
        # 函数逻辑是 按行处理  行间互不干扰
        # 需要做的工作有 删掉content==None 的字段 
        # 否则 删掉 全None 的字段 
        # 然后 对appellor 去重 
        pre_timer = time()
        with open(self.filename,"r",encoding="utf-8") as f:
            # 一些统计量 
            writelines = 0 
            for line in tqdm(range(self.ttline)):
                content = f.readline()
                if content==None or content[0]==None :
                    print("%d line has been visited"% line)
                    break
                try:
                    # 先编辑字符串 
                    content = self.regulize(content)
                    dicts  = json.loads(content)
                except:
                    # 我并不希望程序停止运行
                    # 相反我希望程序继续运行 
                    # 留下一行报错信息即可 
                    continue
                if dicts['content']==None:
                    # 此处应该统计 与其他六个字段为None 的关系 
                    continue
                else:
                    writelines+=1
                    # 如果这一行不为空
                    # 那么我们需要 删掉空字段 
                    # 并且对其去重 
                    #print(dicts)
                    dicts = self.delete_none_value(dicts)
                    dicts = self.appellor_dedup(dicts)
                    result_content = self.dict2string(dicts)
                    with open(clean_set,'a',encoding='utf-8') as file:
                        file.write(result_content)
                #途中的输出  预计加一个进度条  提高一下代码质量 
                if(writelines%10000==0):
                    crt_timer = time()
                    # if(crt_timer-pre_timer<0.05):
                    #     continue
                    # 上面这段改成直接 直接修改这一行的输出就行了 
                    print("\r%d lines has been writed %.2f seconds has been used， average speech : %.2f" % (writelines,crt_timer-pre_timer,(crt_timer-pre_timer)/(writelines/10000)),end="")

if __name__=='__main__':
    reader = Case()
    reader.preprocess()