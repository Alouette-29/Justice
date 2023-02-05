import json
import os 
import numpy as np 
class Case():
    def __init__(self,filename,encoding) -> None:
        self.filename = filename 
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
    def dict2string(self,dictionary,filename=None):
        import re
        string = re.sub('\'',"\"",str(dictionary))
        string = re.sub("None","null",str(string))
        if filename!=None:
            with open(filename,'a',encoding='utf-8') as f:
                f.write(string+'\n')
        return string
    def delete_null_sample(self,clean_set = "../cleanset.txt"):
        from time import time 
        import json
        pre_timer = time()
        with open(self.filename,"r",encoding="utf-8") as f:
            line = 0
            writelines = 0 
            while(True):
                content = f.readline()
                if content==None:
                    print("%d line has been visited"% line)
                    break
                dicts  = json.loads(content)
                if dicts['content']==None:
                    continue
                else:
                    writelines+=1
                    with open(clean_set,'a',encoding='utf-8') as file:
                        file.write(content)
                line+=1
                if(writelines%10000==0):
                    crt_timer = time()
                    print("%d lines has been writed %.2f seconds has been used " % (writelines,crt_timer-pre_timer))
                    pre_timer = crt_timer
                    break