import torch
class EssenceModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = torch.nn.Linear(384,384)
        self.fc2 = torch.nn.Linear(384,2)
    def forward(self,embeddings):
        out = self.fc1(embeddings)
        out = self.fc2(out)
        return out 

model = EssenceModel()
model = torch.load("./check_dup/35th_relation_predict_model.model")
from tqdm import trange
model.eval()
length = 280375
tensors = torch.load("./tensor_fake_facts.pth")
with torch.no_grad():
    prediction = torch.argmax(model(tensors),axis=1)
torch.save(prediction,'prediction_fake_set.pth')

# find essence tensors and indexs 

essence_tensors = tensors[prediction]
torch.save(essence_tensors,"ensence_tensor.pth")

# essence dataset 

with open("./文本分段/fake_facts.json",encoding='utf-8') as file:
    for i in trange(length):
        line = file.readline()
        if prediction[i] ==1:
            with open("./essence_sentence.json",'a',encoding='utf-8') as output:
                output.write(line)


