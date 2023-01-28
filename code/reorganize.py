import os
database = "p2-1-2021.txt"
def reorganize(database:str):
    casepfile = 1000#case per file
    filepbatch =100#file per batch
    fileflag = 0
    batchflag = 0
    basepath  = "./casebase"
    with open(database,encoding='utf-8') as f1:
        while(1):

            folderpath = basepath+"/"+str(batchflag)
            os.makedirs(folderpath,exist_ok=True)
            path = folderpath+"/"+str(fileflag)+".txt"
            with open(path,"a",encoding='utf-8') as f2:
                for i in range(casepfile):     
                    content = f1.readline()
                    if content == "":
                        return 0
                    f2.write(content)  
            fileflag+=1

            if fileflag == filepbatch:
                fileflag=0
                batchflag+=1

def check_base():
    basepath = "./casebase/"
    database = "p2-1-2021.txt"
    if os.path.exists(basepath):
        batchnum = len(os.listdir(basepath))
        filenum = 0
        for (root, dirs, files) in os.walk(basepath):
            filenum+=len(files)

        dir1 = os.listdir(basepath)
        dirnum= sorted([int(i) for i in dir1])
        lastdir = str(dirnum[-1])
        dir1 = os.listdir(os.path.join(basepath,lastdir))
        dirnum = sorted([int(i[:-4]) for i in dir1])
        lastfile = os.path.join(basepath,lastdir,str(dirnum[-1])+".txt")
        print(lastfile)
        with open(lastfile,encoding='utf-8') as f:
            casenum = len(f.readlines())
    
        return (filenum-1)*1000+casenum
    else:
        print("database don't exists, now calling reorganize")
        reorganize(database)


check_base()