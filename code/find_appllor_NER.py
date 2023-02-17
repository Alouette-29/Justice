with open("output_215.txt",'r',encoding='utf-8') as f:
    names = f.readlines()
for i in range(len(names)):
    if len(names[i])!=1:
        # 如果不是只有一个'\n'
        namelist  =  names[i][:-1].split(",")
        for j in range(len(namelist)):
            if len(namelist[j])==1:
                namelist[j]+='某'
        names[i] = ','.join(namelist)+'\n'
with open("output_215.txt",'w',encoding='utf-8') as f:
     f.write(''.join(names))

with open("name_to_find.txt",'r',encoding='utf-8') as f:
    ids = f.readlines()

with open("output_215.txt",'w',encoding='utf-8') as f:
    for i in range(len(names)):
        f.write(ids[i][:-1]+' '+names[i])