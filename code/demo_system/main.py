from flask import Flask, request
import os
# import torch
# from transformers import AutoTokenizer
# from transformers import AutoModel
# #定义下游模型
# pretrained = AutoModel.from_pretrained('hfl/rbt6')
# class Model(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.tuneing = False
#         self.pretrained = None

#         self.rnn = torch.nn.GRU(768, 768,batch_first=True)
#         self.fc = torch.nn.Linear(768, 8)

#     def forward(self, inputs):
#         #print(inputs.keys())
#         if self.tuneing:
#             out = self.pretrained(**inputs).last_hidden_state
#         else:
#             with torch.no_grad():
#                 out = pretrained(**inputs).last_hidden_state

#         out, _ = self.rnn(out)

#         out = self.fc(out).softmax(dim=2)

#         return out

#     def fine_tuneing(self, tuneing):
#         self.tuneing = tuneing
#         if tuneing:
#             for i in pretrained.parameters():
#                 i.requires_grad = True

#             pretrained.train()
#             self.pretrained = pretrained
#         else:
#             for i in pretrained.parameters():
#                 i.requires_grad_(False)

#             pretrained.eval()
#             self.pretrained = None

# def deduction(sentences,output_file=None):
#     model = Model()
#     tokenizer = AutoTokenizer.from_pretrained('hfl/rbt6')
#     model = torch.load('D:/GitHub/NER_in_Chinese/model/命名实体识别_中文.model')
#     model.eval()
#     for i in range(len(sentences)):
#         sentences[i] = [word for word in sentences[i]]
#     token = tokenizer.batch_encode_plus(
#         sentences,
#         add_special_tokens=True,
#         padding=True,
#         truncation=True,
#         max_length=137,
#         is_split_into_words=True,
#         return_tensors='pt',
#         pad_to_multiple_of=137
#         )

#     with torch.no_grad():
#         #[b, lens] -> [b, lens, 8] -> [b, lens]
#         outs = model(token).argmax(dim=2)

#     for i in range(1):
#         #移除pad
#         select = token['attention_mask'][i] == 1
#         input_id = token['input_ids'][i, select]
#         out = outs[i, select]
        
#         #输出原句子
#         #print(tokenizer.decode(input_id).replace(' ', ''))
#                 #输出tag
#         for tag in [out]:
#             s = ''
#             for j in range(len(tag)):
#                 if tag[j] == 0:
#                     s += '·'
#                     continue
#                 s += tokenizer.decode(input_id[j])
#                 s += str(tag[j].item())

#             print(s)
#         #print('==========================')
#     if output_file!=None:
#         with open(output_file,'r') as f:
#             f.write(s)
#     return s 

#给网页对象一个名字 
app = Flask(__name__)
#指定处理函数路径 
@app.route("/process/entity_recongnition",methods = ['POST'])
def process_entity_recongnition():
    text = request.get_data(as_text=True)
    #ret = deduction(text)
    ret = "somthing"
    return ret,200
#打开一个文件 
@app.route("/resource/<file_name>",methods = ['GET'])
def get_static(file_name):
    path = os.path.join("./resource",file_name)
    d = "<html><body><p>write somthing to read <p/><body/><html/>"
    if os.path.exists(path):
        with open(path,'rb') as f:
            d = f.read()
        if d is not None:
            return d,200
    return d,404


#打开主页面 
@app.route("/",methods= ['GET'])
@app.route("/index.html",methods = ['GET'])
def main_index():
    return get_static('index.html')

#主函数  
if __name__ =="__main__":
    print("here")
    app.run(host='0.0.0.0',port=8800)
    from waitress import serve
    #serve(app,host='0.0.0.0',port = 8800)