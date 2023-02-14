from time import time
from tqdm import tqdm
import json,re,os
import numpy as np
import pandas as pd
#import sys
#sys.path.append("../")
filename="E:/xiaochuang/Justice/cleanset.txt"
colpath='E:/xiaochuang/Justice/code/clean_count.json'
from case_reader_release2 import Case


from tqdm import trange  # 进度条




def get_yuangao(reader):
    num = reader.ttline  # 总个数
    nameset = {}  # 原告人名字典，初始构想为键为人名，值为出现的id
    print(num)
    for i in trange(num):  # 遍历
        case = reader.get_dict()[1]  # 得到一条案子的信息
        s1 = case['appellor']
        s = s1.split(',')  # 这条案子的人并转化成列表
        con = case['content']  # 案件正文
        cid = case['id']  # 案子的编号
        yg = ""  # 原告名字
        if len(s) == 2:  # 两个人打官司
            id1 = con.find(s[0])  # 第一个人第一次出现的位置
            id2 = con.find(s[1])  # 第二个人第一次出现的位置
            if id1 < id2:  # 谁先出现谁是原告
                yg = s[0]
            else:
                yg = s[1]
        elif len(s) == 3 and ("法院" in s1 or "检察院" in s1):
            s2 = [i for i in s if ("法院" not in i and "检察院" not in i)]  # 去除法院和检察院，只保留人名。
            id1 = con.find(s2[0])  # 第一个人第一次出现的位置
            id2 = con.find(s2[1])  # 第二个人第一次出现的位置
            if id1 < id2:  # 谁先出现谁是原告
                yg = s2[0]
            else:
                yg = s2[1]
        # 注：此处可插入更多的情况
        dc = nameset.get(yg)  # 查找此人
        if dc != None:  # 查到此人，添加
            dc.append(cid)
        else:  # 查无此人，创建
            ls = [cid]
            nameset[yg] = ls

    return nameset

def get_all_yuangao(reader):
    num = reader.ttline  # 总个数
    nameset={}
    print(num)
    for i in trange(num):  # 遍历
        case = reader.get_dict()[1]  # 得到一条案子的信息
        s1 = case["appellor"]
        s = s1.split(',')  # 这条案子的人并转化成列表
        con = case['content']  # 案件正文
        if s==None:
            print(con)
        cid = case['id']  # 案子的编号
        # 原告名字
        for yg in s:
            # 注：此处可插入更多的情况
            dc = nameset.get(yg)  # 查找此人
            if dc != None:  # 查到此人，添加
                dc.append(cid)
            else:  # 查无此人，创建
                ls = [cid]
                nameset[yg] = ls

    return nameset
def get_no_appellor(reader,kid):
    num = reader.ttline  # 总个数
    ls=[]
    for i in trange(num):  # 遍历
        case = reader.get_dict()[1]  # 得到一条案子的信息
        cid=case['id']
        if cid in kid:
            con=case['content']
            ls.append((cid,con))
    return ls

def get_special(reader):
    num=reader.ttline
    ls=[]
    flag=0
    for i in trange(num):
        flag=0
        case=reader.get_dict()[1]
        cid=case['id']
        ap=case['appellor']
        if ap=='':
            flag=1
        else:
            for a in ap:
                for j in range(0,10):
                    if str(j) in a or '某' in a:
                        flag=1

        if flag==1:
            ls.append(cid)
    return ls

def get_courtname(reader,n):
    num = reader.ttline
    ls = []
    nameset={}
    for i in trange(num):
        case=reader.get_dict()[1]
        id=case['id']
        if id in n:
            name=case['court_name']
            ap=case['appellor']
            ls.append((ap,name))
            dc=nameset.get(name)
            if(dc!=None):
                nameset[name]+=1
            else:
                nameset[name]=1



    return ls,nameset

def read_text(file):
    ls=[]
    with open(file,'r',encoding='utf-8') as f:
      for line in f:
        ls.append(line.replace('\n',''))
    return ls

def special_company(reader,company):
    num = reader.ttline
    ls = []
    for i in trange(num):
        case = reader.get_dict()[1]
        ap=case['appellor']
        if company in ap:
            ls.append(case['content']+'\n')
    return ls

def get_companies(reader):
    num=reader.ttline
    ls=[]
    for i in trange(num):
        case=reader.get_dict()[1]
        ap=case['appellor'].split(',')
        tmp=[j for j in ap if j[-2:]=='公司']
        if tmp!=[]:
            ls.append((','.join(tmp),case['content']))


    return ls

def select_company(file,wfile):
    ls = []
    with open(file, 'r', encoding='utf-8') as f:
        for line in f:
            if '公司' in line:
                ls.append(line)
    with open (wfile,'w',encoding='utf-8') as f2:
        for l in ls:
            f2.write(l)
    return ls
if __name__=="__main__":
    # 测试
    reader=Case(filename=filename,colpath=colpath)
    '''
    nst = get_all_yuangao(reader)
    ls=sorted(nst.items(),key=lambda x:len(x[1]),reverse=True)
    #print("测试结果如下:")

    kid=[] #l[0]为空
    for l in ls:
        if(l[0]==''):
            kid=l[1]
    print(kid)
    reader2 = Case(filename=filename, colpath=colpath)
    kls = get_no_appellor(reader2,kid)
    with open ('No_appellor.txt','w',encoding='utf-8') as t2:
        t2.write('把所有appellor为空的结果提取出来，得到的结果如下:\n')
        for k in kls:
            s="id为:"+str(k[0])+"\n"+"content为:"+str(k[1])+"\n"
            t2.write(s)
    
    ls=get_special(reader)
    with open('Special.txt','w',encoding='utf-8') as f:
        for l in ls:
            s=str(l)+'\n'
            f.write(s)
    
    n=read_text('E:/xiaochuang/Justice/code/Special.txt')
    print(n)
    tup=get_courtname(reader,n)
    courtname=tup[0]
    countset=tup[1]
    with open ('court_name.txt','w',encoding='utf-8') as f1:
        for c in courtname:
            s="appellor: "+str(c[0])+" court_name: "+str(c[1])+"\n"
            f1.write(s)
    ls = sorted(countset.items(), key=lambda x: x[1], reverse=True)
    with open('count_courtnames.txt','w',encoding='utf-8') as f2:
        for l in ls:
            s="court_name: "+str(l[0])+" count:"+str(l[1])+"\n"
            f2.write(s)
    
    company=str(input('请输入公司名:'))
    ls=special_company(reader,company)
    with open(company+'.txt', 'w', encoding='utf-8') as f1:
        for l in ls:
            f1.write(l)
    

    ls=get_companies(reader)
    print(len(ls))
    with open('companies.txt','w',encoding='utf-8') as f2:
        for l in ls:
            s="名称:"+l[0]+" 文本:"+l[1]+'\n'
            f2.write(s)
    '''
    wfile='company_names.txt'
    select_company('All_yuangao.txt',wfile)