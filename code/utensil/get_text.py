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
        columns = np.load("./no0log.npy",allow_pickle=True).item()
        columns =  list(columns.keys())
    def get_dict(self):
        file = self.file
        suit = file.readline()
        dictionary = json.loads(suit)
        self.line+=1
        #print(self.line)
        #print("here")
        return self.line-1,dictionary
    def close(self):
        self.file.close()