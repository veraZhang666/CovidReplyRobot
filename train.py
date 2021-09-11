
import spacy # 3.x version
from spacy import displacy
import re # regular expressions
import numpy as np
from collections import defaultdict
import pandas as pd
import torch
from sentence_transformers  import SentenceTransformer

#load question and intent dataset
data_df = pd.read_csv('covid_19.csv',names=["questions","intent"])

response_df = pd.read_csv('response.csv',names=["intent","response"])
responses_dict = response_df.set_index('intent').T.to_dict('list')
print(responses_dict)


trainining_sentences =data_df.questions

#conver the intent to one hot code
training_intents =  pd.get_dummies(data_df.intent)


def getWordVectors(trainining_sentences):
    model_name = 'bert-base-nli-mean-tokens'
    model = SentenceTransformer(model_name)
    X_train = model.encode(trainining_sentences)

    return X_train

X_train = getWordVectors(trainining_sentences)

print("----====",X_train.shape) # (90, 768)
y_labels = []

intent2label = {'greetings':0,'information':1,'prevention':2,
                'symptoms':3, 'travel':4,'vaccine' :5}

label_tensors = torch.zeros(len(trainining_sentences),1)

for intent in intent2label.keys():
    l = training_intents.loc[training_intents[intent] == 1].index.tolist()
    for idx in l :
        label_tensors[idx][0]=intent2label[intent]


X_train = torch.from_numpy(X_train)
y_train = label_tensors
# ===============================dataloader =====================


from torch.utils.data import DataLoader,Dataset
import torch.nn as nn

dataset = Dataset()


class MyDataset(Dataset):
    def __init__(self,X,y):
        X = X_train
        y = y_train
        self.X = X
        self.y = y
    def __getitem__(self, idx):
        return self.X[idx],self.y[idx]
    def __len__(self):
        return self.X.shape[0]
dataset  = MyDataset(X_train,y_train)

dataloader = DataLoader(dataset = dataset, shuffle = True, batch_size = 1 )


#===========================Modeling ========================



torch.set_default_tensor_type(torch.DoubleTensor)

NUM_EPOCHS = 300

fc=torch.nn.Linear(768,6) #只使用一层线性分类器

fc = fc.double()

criterion=torch.nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(fc.parameters())

for epoch in range(NUM_EPOCHS):
    for idx, (images,labels) in enumerate(dataloader):
        x =images.reshape(-1,768)
        x = x.to(torch.double)
        labels = labels.squeeze(1).long()
        optimizer.zero_grad() #梯度清零
        preds=fc(x) #计算预测

        loss=criterion(preds,labels) #计算损失
        loss.backward() # 计算参数梯度
        optimizer.step() # 更新迭代梯度
        if epoch % 50 ==0:
            if idx % 20 ==0:
                print('epoch={}:idx={},loss={:g}'.format(epoch,idx,loss))

correct=0
total=0

for idx,(images,labels) in enumerate(dataloader):
    x =images.reshape(-1,768)
    x = x.to(torch.double)
    preds=fc(x)
    predicted=torch.argmax(preds,dim=1) #在dim=1中选取max值的索引
    if idx ==0:
        print('x size:{}'.format(x.size()))
        print('preds size:{}'.format(preds.size()))
        print('predicted size:{}'.format(predicted.size()))

    total+=labels.size(0)
    correct+=(predicted == labels).sum().item()
    #print('##########################\nidx:{}\npreds:{}\nactual:{}\n##########################\n'.format(idx,predicted,labels))

accuracy=correct/total
print('{:1%}'.format(accuracy))

torch.save(fc.state_dict(),'classify_model1.pth')
